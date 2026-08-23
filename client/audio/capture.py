"""WASAPI loopback audio capture for the Windows client.

The module grabs everything Windows is playing back (Zoom, Teams, a browser
tab, ...) through the WASAPI loopback interface exposed by ``pyaudiowpatch``,
converts it to the 16 kHz mono 16-bit PCM contract from ``DESIGN.md`` and hands
it out as fixed size chunks through a thread safe queue.

Typical use::

    with LoopbackCapture() as capture:
        chunk = capture.read(timeout=1.0)      # 6400 bytes = 200 ms @ 16 kHz

Only :class:`LoopbackCapture` touches the sound card.  The buffering logic
lives in :class:`ChunkAssembler`, which is pure Python and therefore fully
unit testable on the Dev PC.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Iterator, Optional

from client.config import (
    CAPTURE_FRAMES_PER_BUFFER,
    CHUNK_BYTES,
    CHUNK_DURATION_MS,
    MAX_QUEUED_CHUNKS,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
)
from client.audio.resampler import StreamResampler

try:                                        # pragma: no cover - Windows only
    import pyaudiowpatch as pyaudio

    _HAS_PYAUDIOWPATCH = True
except ImportError:                         # pragma: no cover - Dev PC / Linux
    pyaudio = None
    _HAS_PYAUDIOWPATCH = False

log = logging.getLogger(__name__)


class AudioCaptureError(RuntimeError):
    """Raised when no usable loopback device or format can be found."""


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceInfo:
    """The subset of the pyaudiowpatch device dict that we actually need."""

    index: int
    name: str
    channels: int
    default_sample_rate: int
    is_loopback: bool

    @classmethod
    def from_dict(cls, raw: dict) -> "DeviceInfo":
        return cls(
            index=int(raw["index"]),
            name=str(raw["name"]),
            channels=int(raw["maxInputChannels"]),
            default_sample_rate=int(round(float(raw["defaultSampleRate"]))),
            is_loopback=bool(raw.get("isLoopbackDevice", False)),
        )


def _require_backend() -> None:
    if not _HAS_PYAUDIOWPATCH:
        raise AudioCaptureError(
            "pyaudiowpatch is not installed. Run "
            "`py -3.11 -m pip install -r client/requirements.txt` on the "
            "Windows Client PC."
        )


def list_loopback_devices() -> list[DeviceInfo]:
    """Return every WASAPI loopback device currently visible to Windows."""
    _require_backend()
    with pyaudio.PyAudio() as pa:
        return [
            DeviceInfo.from_dict(dev)
            for dev in pa.get_loopback_device_info_generator()
        ]


def find_loopback_device(name_hint: Optional[str] = None) -> DeviceInfo:
    """Pick the loopback device to record from.

    With no hint the loopback that mirrors the *default* Windows output device
    is returned, which is what a meeting participant actually hears.  With a
    hint, the first loopback whose name contains that (case insensitive)
    substring wins.
    """
    _require_backend()
    with pyaudio.PyAudio() as pa:
        loopbacks = [
            DeviceInfo.from_dict(dev)
            for dev in pa.get_loopback_device_info_generator()
        ]
        if not loopbacks:
            raise AudioCaptureError(
                "No WASAPI loopback device found. Check that a playback device "
                "is enabled in the Windows sound settings."
            )

        if name_hint:
            hint = name_hint.lower()
            for dev in loopbacks:
                if hint in dev.name.lower():
                    return dev
            raise AudioCaptureError(
                f"No loopback device matching {name_hint!r}. Available: "
                + ", ".join(d.name for d in loopbacks)
            )

        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_output = pa.get_device_info_by_index(
            int(wasapi["defaultOutputDevice"])
        )
        default_name = str(default_output["name"]).lower()
        for dev in loopbacks:
            # The loopback endpoint is named "<speaker name> [Loopback]".
            if default_name in dev.name.lower():
                return dev

        log.warning(
            "Default output %r has no matching loopback; falling back to %r",
            default_output["name"],
            loopbacks[0].name,
        )
        return loopbacks[0]


# ---------------------------------------------------------------------------
# Chunk buffering
# ---------------------------------------------------------------------------
class ChunkAssembler:
    """Slice a byte stream into fixed size chunks.

    The audio callback delivers buffers whose size depends on the sound card,
    while the network layer wants exactly ``chunk_bytes`` per message.  This
    class keeps the remainder between calls.
    """

    def __init__(self, chunk_bytes: int = CHUNK_BYTES) -> None:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        self.chunk_bytes = chunk_bytes
        self._buffer = bytearray()

    def push(self, data: bytes) -> list[bytes]:
        """Add ``data`` and return every complete chunk it made available."""
        self._buffer.extend(data)
        chunks: list[bytes] = []
        while len(self._buffer) >= self.chunk_bytes:
            chunks.append(bytes(self._buffer[: self.chunk_bytes]))
            del self._buffer[: self.chunk_bytes]
        return chunks

    def flush(self, pad: bool = True) -> Optional[bytes]:
        """Return the trailing partial chunk, zero padded when ``pad``."""
        if not self._buffer:
            return None
        tail = bytes(self._buffer)
        self._buffer.clear()
        if pad and len(tail) < self.chunk_bytes:
            tail += b"\x00" * (self.chunk_bytes - len(tail))
        return tail

    @property
    def pending_bytes(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
@dataclass
class CaptureStats:
    """Counters filled in while a capture session runs."""

    callbacks: int = 0
    device_frames: int = 0
    chunks_emitted: int = 0
    chunks_dropped: int = 0
    input_overflows: int = 0
    device: Optional[DeviceInfo] = field(default=None)
    device_format: str = ""

    @property
    def captured_seconds(self) -> float:
        if not self.device:
            return 0.0
        return self.device_frames / float(self.device.default_sample_rate)

    @property
    def emitted_seconds(self) -> float:
        return self.chunks_emitted * CHUNK_DURATION_MS / 1000.0


class LoopbackCapture:
    """Record the Windows playback mix as 16 kHz mono 16-bit PCM chunks."""

    #: Device formats tried in order until one is accepted by WASAPI.
    _FORMAT_CANDIDATES = ("int16", "float32", "int32")

    def __init__(
        self,
        device_name_hint: Optional[str] = None,
        device: Optional[DeviceInfo] = None,
        chunk_bytes: int = CHUNK_BYTES,
        frames_per_buffer: int = CAPTURE_FRAMES_PER_BUFFER,
        max_queued_chunks: int = MAX_QUEUED_CHUNKS,
    ) -> None:
        _require_backend()
        self._device_name_hint = device_name_hint
        self._device = device
        self._chunk_bytes = chunk_bytes
        self._frames_per_buffer = frames_per_buffer
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=max_queued_chunks)
        self._assembler = ChunkAssembler(chunk_bytes)
        self._resampler: Optional[StreamResampler] = None
        self._pa = None
        self._stream = None
        self._lock = threading.Lock()
        self._running = False
        self.stats = CaptureStats()

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> DeviceInfo:
        """Open the loopback stream and start filling the chunk queue."""
        if self._running:
            raise AudioCaptureError("Capture already running")

        self._pa = pyaudio.PyAudio()
        try:
            device = self._device or find_loopback_device(self._device_name_hint)
            pa_format, fmt_name = self._negotiate_format(device)
            self._resampler = StreamResampler(
                input_rate=device.default_sample_rate,
                input_channels=device.channels,
                sample_format=fmt_name,
                output_rate=TARGET_SAMPLE_RATE,
            )
            self.stats = CaptureStats(device=device, device_format=fmt_name)
            self._stream = self._pa.open(
                format=pa_format,
                channels=device.channels,
                rate=device.default_sample_rate,
                input=True,
                input_device_index=device.index,
                frames_per_buffer=self._frames_per_buffer,
                stream_callback=self._on_audio,
            )
            self._running = True
            self._stream.start_stream()
        except Exception:
            self._teardown()
            raise

        log.info(
            "Capturing %r (%d ch @ %d Hz, %s) -> %d Hz mono int16, %s backend",
            device.name,
            device.channels,
            device.default_sample_rate,
            fmt_name,
            TARGET_SAMPLE_RATE,
            self._resampler.backend,
        )
        return device

    def stop(self) -> CaptureStats:
        """Stop the stream and push any trailing partial chunk to the queue."""
        if not self._running:
            return self.stats
        self._running = False
        try:
            if self._stream is not None:
                self._stream.stop_stream()
            if self._resampler is not None:
                self._offer(self._resampler.flush())
            tail = self._assembler.flush(pad=True)
            if tail:
                self._enqueue(tail)
        finally:
            self._teardown()
        return self.stats

    def _teardown(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:                # pragma: no cover - driver noise
                log.debug("stream.close() failed", exc_info=True)
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:                # pragma: no cover - driver noise
                log.debug("PyAudio.terminate() failed", exc_info=True)
            self._pa = None

    def __enter__(self) -> "LoopbackCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # -- consumer API -------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queued_chunks(self) -> int:
        return self._queue.qsize()

    def read(self, timeout: Optional[float] = 1.0) -> Optional[bytes]:
        """Pop one chunk, or ``None`` when nothing arrived within ``timeout``."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stream(self, timeout: Optional[float] = 1.0) -> Iterator[bytes]:
        """Yield chunks until the capture is stopped and the queue drains."""
        while self._running or not self._queue.empty():
            chunk = self.read(timeout=timeout)
            if chunk is not None:
                yield chunk

    # -- internals ----------------------------------------------------------
    def _negotiate_format(self, device: DeviceInfo):
        """Find a sample format WASAPI accepts for this loopback endpoint."""
        formats = {
            "int16": pyaudio.paInt16,
            "float32": pyaudio.paFloat32,
            "int32": pyaudio.paInt32,
        }
        for name in self._FORMAT_CANDIDATES:
            pa_format = formats[name]
            try:
                supported = self._pa.is_format_supported(
                    device.default_sample_rate,
                    input_device=device.index,
                    input_channels=device.channels,
                    input_format=pa_format,
                )
            except Exception:                # ValueError from PortAudio
                continue
            if supported:
                return pa_format, name
        raise AudioCaptureError(
            f"Device {device.name!r} accepts none of {self._FORMAT_CANDIDATES}"
        )

    def _on_audio(self, in_data, frame_count, time_info, status):
        """PortAudio callback: runs on the PortAudio thread, must stay fast."""
        if status:
            self.stats.input_overflows += 1
        self.stats.callbacks += 1
        self.stats.device_frames += frame_count
        try:
            self._offer(self._resampler.process(in_data))
        except Exception:                    # pragma: no cover - defensive
            log.exception("Audio callback failed")
        return (None, pyaudio.paContinue if self._running else pyaudio.paComplete)

    def _offer(self, pcm16k: bytes) -> None:
        if not pcm16k:
            return
        for chunk in self._assembler.push(pcm16k):
            self._enqueue(chunk)

    def _enqueue(self, chunk: bytes) -> None:
        """Never block the audio thread: drop the oldest chunk when full."""
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            with self._lock:
                try:
                    self._queue.get_nowait()
                    self.stats.chunks_dropped += 1
                except queue.Empty:          # pragma: no cover - race
                    pass
                try:
                    self._queue.put_nowait(chunk)
                except queue.Full:           # pragma: no cover - race
                    self.stats.chunks_dropped += 1
                    return
        self.stats.chunks_emitted += 1


def pcm16_to_wav_bytes(
    pcm: bytes,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
) -> bytes:
    """Wrap raw 16-bit PCM in a WAV container so it can be played back."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()
