"""Tests for the client WebSocket uplink.

The queue tests are pure logic.  The rest run against a real WebSocket server
on localhost, so the handshake, the binary framing and the reconnect loop are
exercised end to end without needing the GPU pod.

Run with::

    .venv\\Scripts\\python.exe -m pytest client/tests -v
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.net.ws_client import StreamClient
from common.protocol import (
    CHUNK_BYTES,
    ClientMessage,
    make_error,
    make_ready,
    make_vad,
    parse_message,
)


def chunk(value: int = 0) -> bytes:
    return bytes([value % 256]) * CHUNK_BYTES


# ---------------------------------------------------------------------------
# A stub server, standing in for the pod
# ---------------------------------------------------------------------------
@dataclass
class StubServer:
    """Records what arrived and replays a scripted reaction."""

    reject: str = ""                     # non-empty -> refuse the handshake
    push_on_chunk: int = -1              # send a vad event after this chunk
    close_after_chunks: int = -1         # hang up mid-stream, to test reconnect
    end_on_bye: bool = False             # answer a bye with a closing event
    hellos: list[dict] = field(default_factory=list)
    chunks: list[bytes] = field(default_factory=list)
    byes: list[dict] = field(default_factory=list)
    connections: int = 0

    async def handler(self, socket) -> None:
        self.connections += 1
        raw = await socket.recv()
        payload = parse_message(raw)
        self.hellos.append(payload)

        if self.reject:
            await socket.send(make_error(self.reject))
            await socket.close()
            return
        await socket.send(make_ready(payload["session_id"]))

        async for message in socket:
            if isinstance(message, bytes):
                self.chunks.append(message)
                if len(self.chunks) == self.push_on_chunk:
                    await socket.send(make_vad("speech_start", 123.4))
                if len(self.chunks) == self.close_after_chunks:
                    await socket.close()
                    return
            else:
                control = parse_message(message)
                if control["type"] == ClientMessage.BYE.value:
                    self.byes.append(control)
                    if self.end_on_bye:
                        await socket.send(make_vad("speech_end", 4200.0))
                    return


@contextlib.asynccontextmanager
async def running(stub: StubServer):
    server = await websockets.serve(stub.handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"ws://127.0.0.1:{port}/ws/stream"
    finally:
        server.close()
        await server.wait_closed()


async def drain(client: StreamClient, task: asyncio.Task) -> None:
    await client.stop(drain_timeout=1.0)
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, timeout=5.0)
    if not task.done():                                  # pragma: no cover
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll until ``predicate`` holds, so tests never race the event loop."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# Queue behaviour (no network)
# ---------------------------------------------------------------------------
async def test_a_wrong_sized_chunk_is_refused_before_it_reaches_the_wire():
    client = StreamClient("ws://unused")
    assert client.send(bytes(100)) is False
    assert client.queued_chunks == 0
    assert "refused a 100 byte chunk" in client.stats.errors[0]


async def test_chunks_queue_up_while_disconnected():
    client = StreamClient("ws://unused", queue_chunks=3)
    assert all(client.send(chunk(i)) for i in range(3))
    assert client.queued_chunks == 3


async def test_a_full_queue_drops_the_oldest_chunk():
    """During a stall, recent audio is worth more than stale audio."""
    client = StreamClient("ws://unused", queue_chunks=2)
    client.send(chunk(1))
    client.send(chunk(2))
    assert client.send(chunk(3)) is False
    assert client.queued_chunks == 2
    assert client.stats.chunks_dropped == 1
    assert client._queue.get_nowait() == chunk(2)        # chunk(1) was evicted


async def test_send_threadsafe_before_the_loop_is_running_is_counted_not_crashed():
    client = StreamClient("ws://unused")
    client.send_threadsafe(chunk())
    assert client.stats.chunks_dropped == 1


# ---------------------------------------------------------------------------
# Handshake and streaming over localhost
# ---------------------------------------------------------------------------
async def test_the_client_introduces_itself_with_the_audio_format():
    stub = StubServer()
    async with running(stub) as url:
        client = StreamClient(url, session_id="abc", client_name="test-rig")
        task = asyncio.create_task(client.run())
        assert await client.wait_connected(5.0)
        await drain(client, task)

    assert len(stub.hellos) == 1
    hello = stub.hellos[0]
    assert hello["session_id"] == "abc"
    assert hello["client"] == "test-rig"
    assert (hello["sample_rate"], hello["channels"], hello["chunk_ms"]) == (
        16_000, 1, 200,
    )


async def test_queued_audio_reaches_the_server_intact():
    stub = StubServer()
    async with running(stub) as url:
        client = StreamClient(url)
        task = asyncio.create_task(client.run())
        assert await client.wait_connected(5.0)
        for i in range(5):
            client.send(chunk(i))
        assert await wait_for(lambda: len(stub.chunks) == 5)
        await drain(client, task)

    assert stub.chunks == [chunk(i) for i in range(5)]
    assert client.stats.chunks_sent == 5
    assert client.stats.bytes_sent == 5 * CHUNK_BYTES
    assert client.stats.audio_seconds_sent == pytest.approx(1.0)


async def test_audio_queued_before_the_socket_opens_is_not_lost():
    stub = StubServer()
    async with running(stub) as url:
        client = StreamClient(url)
        client.send(chunk(7))                    # queued while disconnected
        task = asyncio.create_task(client.run())
        assert await wait_for(lambda: stub.chunks == [chunk(7)])
        await drain(client, task)


async def test_server_messages_are_handed_to_the_callback():
    received: list[dict] = []
    stub = StubServer(push_on_chunk=1)
    async with running(stub) as url:
        client = StreamClient(url, on_message=received.append)
        task = asyncio.create_task(client.run())
        assert await client.wait_connected(5.0)
        client.send(chunk())
        assert await wait_for(lambda: any(m["type"] == "vad" for m in received))
        await drain(client, task)

    kinds = [m["type"] for m in received]
    assert kinds == ["ready", "vad"]             # ready is dispatched too
    assert received[1]["at_ms"] == 123.4


async def test_an_async_callback_is_awaited():
    received: list[dict] = []

    async def on_message(payload: dict) -> None:
        await asyncio.sleep(0)
        received.append(payload)

    stub = StubServer()
    async with running(stub) as url:
        client = StreamClient(url, on_message=on_message)
        task = asyncio.create_task(client.run())
        assert await client.wait_connected(5.0)
        await drain(client, task)

    assert [m["type"] for m in received] == ["ready"]


async def test_a_clean_stop_says_bye():
    stub = StubServer()
    async with running(stub) as url:
        client = StreamClient(url)
        task = asyncio.create_task(client.run())
        assert await client.wait_connected(5.0)
        await drain(client, task)
        assert await wait_for(lambda: len(stub.byes) == 1)

    assert stub.byes[0]["reason"] == "client stopped"


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
async def test_a_rejected_handshake_is_recorded_and_not_retried_forever():
    stub = StubServer(reject="unsupported audio format: sample_rate=48000")
    async with running(stub) as url:
        client = StreamClient(url, reconnect=False)
        stats = await asyncio.wait_for(client.run(), timeout=5.0)

    assert client.is_connected is False
    assert any("rejected the session" in e for e in stats.errors)
    assert "sample_rate" in stats.errors[0]


async def test_an_unreachable_server_is_reported_not_raised():
    client = StreamClient("ws://127.0.0.1:9/ws/stream", max_attempts=2)
    stats = await asyncio.wait_for(client.run(), timeout=10.0)
    assert stats.connects == 0
    assert len(stats.errors) == 2                 # one per attempt


async def test_the_client_reconnects_after_the_server_hangs_up():
    """A Wi-Fi hiccup must not end the meeting."""
    stub = StubServer(close_after_chunks=1)
    async with running(stub) as url:
        client = StreamClient(url)
        task = asyncio.create_task(client.run())
        assert await client.wait_connected(5.0)
        client.send(chunk())
        assert await wait_for(lambda: stub.connections >= 2, timeout=8.0)
        await drain(client, task)

    assert client.stats.connects >= 2
    assert len(stub.hellos) >= 2                  # a fresh hello each time


async def test_the_servers_closing_event_survives_a_clean_stop():
    """Found by the real test: a clean stop lost the final speech_end.

    The server closes an open speech segment when the client says bye, but
    the client used to cancel its receive loop before sending the bye, so
    that closing event was never read and the last sentence of the meeting
    would never be finalised.
    """
    received: list[dict] = []
    stub = StubServer(end_on_bye=True)
    async with running(stub) as url:
        client = StreamClient(url, on_message=received.append)
        task = asyncio.create_task(client.run())
        assert await client.wait_connected(5.0)
        client.send(chunk())
        assert await wait_for(lambda: len(stub.chunks) == 1)
        await drain(client, task)

    kinds = [m["type"] for m in received]
    assert "vad" in kinds, f"closing event was dropped; got {kinds}"
    closing = [m for m in received if m["type"] == "vad"][-1]
    assert closing["event"] == "speech_end"
    assert closing["at_ms"] == 4200.0


async def test_a_clean_stop_does_not_hang_when_the_server_stays_silent():
    """The grace period must be a timeout, not a wait forever."""
    stub = StubServer()                       # never answers the bye
    async with running(stub) as url:
        client = StreamClient(url)
        task = asyncio.create_task(client.run())
        assert await client.wait_connected(5.0)
        started = asyncio.get_running_loop().time()
        await drain(client, task)
        assert asyncio.get_running_loop().time() - started < 8.0
