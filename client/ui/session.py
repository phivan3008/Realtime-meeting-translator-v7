"""Runs the capture-and-stream loop off the Qt thread.

Qt owns one event loop and asyncio owns another, and neither can host the
other. So the whole client session - socket, capture, resampling - runs in a
worker thread with its own asyncio loop, and reaches the window only through
Qt signals, which are the one thing Qt lets another thread touch safely.

Nothing here decides what appears on screen. Messages go to
:class:`~client.ui.transcript.TranscriptModel` and the window redraws from
that.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from PySide6.QtCore import QObject, Signal

from client.audio.capture import AudioCaptureError, LoopbackCapture
from client.config import CHUNK_DURATION_MS
from client.net.ws_client import StreamClient

log = logging.getLogger(__name__)

#: How long to wait for a chunk before looking at the stop flag again.
READ_TIMEOUT_S = 0.5


class MeetingSession(QObject):
    """Capture audio, stream it, and hand every server message to Qt."""

    message = Signal(dict)          # one server message, already parsed
    status = Signal(str)            # something worth showing in the status bar
    failed = Signal(str)            # the session cannot continue
    stopped = Signal()

    def __init__(self, url: str, device_hint: Optional[str] = None,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.url = url.rstrip("/")
        self.device_hint = device_hint
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stopping = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._run, name="meeting",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the session to finish and wait for it, without blocking Qt long.

        The server needs the ``bye`` to close the last sentence and translate
        it, so this waits rather than killing the thread.
        """
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # -- worker thread ------------------------------------------------------
    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._session())
        except AudioCaptureError as exc:
            self.failed.emit(f"Không mở được thiết bị âm thanh: {exc}")
        except Exception as exc:                # noqa: BLE001 - surfaced to UI
            log.exception("Meeting session ended badly")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self._loop.close()
            self._loop = None
            self.stopped.emit()

    async def _session(self) -> None:
        client = StreamClient(f"{self.url}/ws/stream",
                              on_message=self.message.emit)
        task = asyncio.create_task(client.run())

        if not await client.wait_connected(timeout=10.0):
            reason = "; ".join(client.stats.errors) or "hết thời gian chờ"
            self.failed.emit(f"Không kết nối được máy chủ: {reason}")
            await client.stop()
            await task
            return

        self.status.emit(f"Đã kết nối · phiên {client.session_id}")
        capture = LoopbackCapture(device_name_hint=self.device_hint)
        device = capture.start()
        self.status.emit(f"Đang nghe · {device.name}")

        try:
            await self._pump(capture, client)
        finally:
            capture.stop()
            # Wait for the goodbye: the last sentence of the meeting is
            # committed and translated after it.
            await client.stop(drain_timeout=3.0)
            await asyncio.wait_for(task, timeout=10.0)

    async def _pump(self, capture, client: StreamClient) -> None:
        """Read audio and send it until asked to stop."""
        while not self._stopping.is_set():
            chunk = await asyncio.to_thread(capture.read, READ_TIMEOUT_S)
            if chunk is None:
                continue
            client.send(chunk)

    # -- for the window's status bar ----------------------------------------
    @staticmethod
    def audio_seconds(chunks: int) -> float:
        return chunks * CHUNK_DURATION_MS / 1000.0
