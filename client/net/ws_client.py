"""WebSocket uplink: stream captured audio to the GPU server.

The capture callback runs on a PortAudio thread and the socket runs on an
asyncio loop, so the two are joined by a bounded queue.  The rule that shapes
this whole module: **the audio thread must never block and never wait.**  A
slow or dead network has to cost audio, not a stuttering capture.  So
:meth:`StreamClient.send` is non-blocking and drops the oldest chunk when the
queue is full, exactly like the capture queue does.

Reconnection is automatic.  A meeting outlives a flaky Wi-Fi hiccup, so
dropping the connection must not end the session: the client backs off,
reconnects, sends a fresh ``hello`` and carries on.  Audio captured while
disconnected is discarded rather than replayed - a burst of stale audio would
put the server's VAD and ASR further behind real time, which is worse than
the gap it is trying to fill.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from common.protocol import (
    CHUNK_BYTES,
    Hello,
    ProtocolError,
    ServerMessage,
    make_bye,
    parse_message,
)

log = logging.getLogger(__name__)

#: Chunks buffered while the socket is busy. 50 * 200 ms = 10 s of audio.
DEFAULT_QUEUE_CHUNKS = 50
#: Reconnect backoff, seconds.
BACKOFF_START = 0.5
BACKOFF_MAX = 10.0
#: Keepalive, so a silently dead TCP connection is noticed.
PING_INTERVAL = 20.0
PING_TIMEOUT = 20.0

MessageHandler = Callable[[dict], Optional[Awaitable[None]]]


@dataclass
class ClientStats:
    """Counters for one client run, across every reconnect."""

    chunks_sent: int = 0
    chunks_dropped: int = 0
    bytes_sent: int = 0
    messages_received: int = 0
    connects: int = 0
    disconnects: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def audio_seconds_sent(self) -> float:
        return self.bytes_sent / 2 / 16_000


class StreamClient:
    """Send audio chunks to the server and surface whatever comes back.

    Usage::

        client = StreamClient("ws://host:8000/ws/stream", on_message=print)
        task = asyncio.create_task(client.run())
        client.send(chunk)            # from any thread
        ...
        await client.stop()
    """

    def __init__(
        self,
        url: str,
        on_message: Optional[MessageHandler] = None,
        session_id: Optional[str] = None,
        queue_chunks: int = DEFAULT_QUEUE_CHUNKS,
        client_name: str = "windows-client",
        reconnect: bool = True,
        max_attempts: Optional[int] = None,
    ) -> None:
        self.url = url
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.on_message = on_message
        self.client_name = client_name
        self.reconnect = reconnect
        self.max_attempts = max_attempts
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_chunks)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = asyncio.Event()
        self._stopping = asyncio.Event()
        self.stats = ClientStats()

    # -- producer side (may be called from the capture thread) --------------
    def send(self, chunk: bytes) -> bool:
        """Queue one chunk. Returns False if it had to displace an older one.

        Never blocks and never raises, because the caller is usually the
        PortAudio callback.
        """
        if len(chunk) != CHUNK_BYTES:
            self.stats.errors.append(
                f"refused a {len(chunk)} byte chunk, expected {CHUNK_BYTES}"
            )
            return False
        try:
            self._queue.put_nowait(chunk)
            return True
        except asyncio.QueueFull:
            # Drop the oldest: during a stall, recent audio is worth more than
            # stale audio, and the server is already behind.
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self.stats.chunks_dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(chunk)
            return False

    def send_threadsafe(self, chunk: bytes) -> None:
        """Queue a chunk from a non-asyncio thread."""
        loop = self._loop
        if loop is None:
            self.stats.chunks_dropped += 1
            return
        loop.call_soon_threadsafe(self.send, chunk)

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def queued_chunks(self) -> int:
        return self._queue.qsize()

    async def wait_connected(self, timeout: float = 10.0) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # -- lifecycle ----------------------------------------------------------
    async def run(self) -> ClientStats:
        """Connect, stream, and keep reconnecting until :meth:`stop`."""
        self._loop = asyncio.get_running_loop()
        backoff = BACKOFF_START
        attempts = 0

        while not self._stopping.is_set():
            attempts += 1
            try:
                await self._session()
                backoff = BACKOFF_START          # a clean session resets it
            except (OSError, ConnectionClosed, ProtocolError) as exc:
                self.stats.errors.append(f"{type(exc).__name__}: {exc}")
                log.warning("Connection to %s failed: %s", self.url, exc)
            finally:
                self._connected.clear()

            if self._stopping.is_set() or not self.reconnect:
                break
            if self.max_attempts is not None and attempts >= self.max_attempts:
                break

            log.info("Reconnecting to %s in %.1fs", self.url, backoff)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)

        return self.stats

    async def stop(self, drain_timeout: float = 2.0) -> ClientStats:
        """Ask the run loop to finish, giving queued audio a chance to go out."""
        deadline = asyncio.get_running_loop().time() + drain_timeout
        while (
            self.is_connected
            and not self._queue.empty()
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)
        self._stopping.set()
        return self.stats

    # -- one connection -----------------------------------------------------
    async def _session(self) -> None:
        async with websockets.connect(
            self.url,
            ping_interval=PING_INTERVAL,
            ping_timeout=PING_TIMEOUT,
            max_queue=None,
        ) as socket:
            self.stats.connects += 1
            await socket.send(
                Hello(session_id=self.session_id, client=self.client_name).to_json()
            )
            await self._await_ready(socket)
            self._connected.set()
            log.info("Streaming to %s as session %s", self.url, self.session_id)

            # Both directions have to be watched together. If only the send
            # side were awaited, a server that hangs up while the queue is
            # empty would go unnoticed: the send loop never touches the
            # socket, so it would spin on an empty queue forever instead of
            # reconnecting.
            receiver = asyncio.create_task(self._receive_loop(socket))
            sender = asyncio.create_task(self._send_loop(socket))
            try:
                done, pending = await asyncio.wait(
                    {receiver, sender}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                for task in done:
                    task.result()               # re-raise whatever ended it

                if self._stopping.is_set():
                    with contextlib.suppress(ConnectionClosed):
                        await socket.send(make_bye("client stopped"))
            finally:
                self._connected.clear()
                self.stats.disconnects += 1

    async def _await_ready(self, socket) -> None:
        """Block until the server accepts the handshake, or fail loudly."""
        raw = await socket.recv()
        if isinstance(raw, bytes):
            raise ProtocolError("server answered the handshake with binary data")
        payload = parse_message(raw)
        kind = payload.get("type")
        if kind == ServerMessage.ERROR.value:
            raise ProtocolError(f"server rejected the session: {payload.get('message')}")
        if kind != ServerMessage.READY.value:
            raise ProtocolError(f"expected {ServerMessage.READY.value!r}, got {kind!r}")
        await self._dispatch(payload)

    async def _send_loop(self, socket) -> None:
        """Drain the queue onto the socket until asked to stop."""
        while not self._stopping.is_set():
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue                     # lets _stopping be noticed promptly
            await socket.send(chunk)
            self.stats.chunks_sent += 1
            self.stats.bytes_sent += len(chunk)

    async def _receive_loop(self, socket) -> None:
        async for raw in socket:
            if isinstance(raw, bytes):
                self.stats.errors.append("server sent unexpected binary data")
                continue
            self.stats.messages_received += 1
            try:
                payload = parse_message(raw)
            except ProtocolError as exc:
                self.stats.errors.append(str(exc))
                continue
            await self._dispatch(payload)

    async def _dispatch(self, payload: dict) -> None:
        if self.on_message is None:
            return
        result = self.on_message(payload)
        if asyncio.iscoroutine(result):
            await result
