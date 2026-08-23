"""Silero VAD - step 0 of the server pipeline.

The client streams the raw 16 kHz mono capture without any filtering, so this
is the first thing the audio meets on the server.  It has two jobs:

1. **Segment.**  Report where speech starts and stops, with stream
   timestamps.  The Stream Buffer Manager (``DESIGN.md`` section 3.1) needs
   exactly this to fire its Finalize Event on a pause longer than 400 ms.
2. **Drop.**  Keep the silence out of the expensive stages downstream
   (YAMNet, PyAnnote, Whisper, vLLM).  Running Whisper over an empty meeting
   room is pure GPU waste.

VAD used to live on the client, but Silero drags ``torch`` and ``torchaudio``
onto a Windows machine we do not control, and it broke there.  Moving it next
to the buffer manager also removed a protocol: the client no longer has to
announce the pauses it deleted, because it no longer deletes any.

Layering
--------
``FrameSplitter``
    Cuts the 200 ms client chunks into the exactly 512 sample frames that
    Silero v5 requires at 16 kHz.  Pure Python.

``SpeechStateMachine``
    Turns a stream of per-frame speech probabilities into speech segments,
    with hysteresis, a minimum speech duration and a silence hangover.  Pure
    Python, so it is unit tested without loading any model.

``SileroVAD``
    Thin wrapper around the Silero torch/ONNX model.  The only part that
    needs the real model file.

``VADSegmenter``
    Wires the three together, produces the audio the rest of the pipeline
    should see, and timestamps every boundary.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from server.config import (
    SAMPLE_RATE,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_SPEECH_PAD_MS,
    VAD_THRESHOLD,
)

log = logging.getLogger(__name__)

#: Silero v5 only accepts this exact frame size at 16 kHz.
VAD_FRAME_SAMPLES = 512
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * 2
VAD_FRAME_MS = VAD_FRAME_SAMPLES * 1000 // SAMPLE_RATE      # 32 ms


class VADError(RuntimeError):
    """Raised when the Silero model cannot be loaded or used."""


# ---------------------------------------------------------------------------
# Frame splitting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    """One VAD frame in both representations.

    ``samples`` feeds the model, ``pcm`` is the untouched slice of the input
    stream.  Keeping the original bytes matters: rebuilding them from the
    float view would shift every sample by one LSB, and that degraded audio
    would then be what Whisper receives.
    """

    samples: np.ndarray          # float32 in [-1, 1), length == frame_samples
    pcm: bytes                   # the same audio as little endian 16-bit PCM


class FrameSplitter:
    """Cut a 16-bit PCM byte stream into fixed size frames.

    Client chunks are 3200 samples (200 ms) while Silero wants 512, which is
    not a divisor, so the remainder has to survive between calls.
    """

    def __init__(self, frame_samples: int = VAD_FRAME_SAMPLES) -> None:
        if frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        self.frame_samples = frame_samples
        self._buffer = bytearray()

    def push(self, pcm: bytes) -> list[Frame]:
        """Add PCM and return every complete frame it made available."""
        self._buffer.extend(pcm)
        frame_bytes = self.frame_samples * 2
        frames: list[Frame] = []
        while len(self._buffer) >= frame_bytes:
            raw = bytes(self._buffer[:frame_bytes])
            del self._buffer[:frame_bytes]
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            frames.append(Frame(samples=samples, pcm=raw))
        return frames

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def pending_samples(self) -> int:
        return len(self._buffer) // 2


# ---------------------------------------------------------------------------
# Speech state machine
# ---------------------------------------------------------------------------
class VADEvent(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(frozen=True)
class SegmentEvent:
    """A speech boundary, positioned in the stream.

    ``at_ms`` is measured from the first sample the segmenter ever saw.  For
    ``SPEECH_START`` it points at the first forwarded sample (the pre-roll, not
    the frame that happened to trigger the model); for ``SPEECH_END`` it points
    just past the last forwarded sample.  So a start/end pair delimits exactly
    the audio that was handed downstream.
    """

    kind: VADEvent
    at_ms: float


@dataclass
class FrameDecision:
    """What the state machine concluded about one 32 ms frame."""

    probability: float
    is_speech: bool                 # inside a speech segment (hangover included)
    event: Optional[VADEvent] = None


class SpeechStateMachine:
    """Hysteresis + duration rules on top of raw Silero probabilities.

    A single probability crossing the threshold is not enough to open a
    segment: Silero fires briefly on keyboard clicks and door slams.  A
    segment opens only after ``min_speech_ms`` of speech-like frames and
    closes only after ``min_silence_ms`` of quiet, which keeps the pauses
    inside a sentence intact.
    """

    def __init__(
        self,
        threshold: float = VAD_THRESHOLD,
        neg_threshold: Optional[float] = None,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        min_silence_ms: int = VAD_MIN_SILENCE_MS,
        frame_ms: int = VAD_FRAME_MS,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        # Closing on a lower threshold than opening avoids flapping when the
        # probability hovers right at the decision point.
        self.neg_threshold = (
            neg_threshold if neg_threshold is not None else max(threshold - 0.15, 0.01)
        )
        self.frame_ms = frame_ms
        self.min_speech_frames = max(1, round(min_speech_ms / frame_ms))
        self.min_silence_frames = max(1, round(min_silence_ms / frame_ms))
        self.reset()

    def reset(self) -> None:
        self._triggered = False
        self._speech_run = 0            # candidate frames before a segment opens
        self._silence_run = 0           # quiet frames since the last speech frame
        self.frames_seen = 0

    @property
    def is_speech(self) -> bool:
        return self._triggered

    @property
    def trailing_silence_ms(self) -> int:
        return self._silence_run * self.frame_ms

    def push(self, probability: float) -> FrameDecision:
        """Feed one frame probability and get the resulting decision."""
        self.frames_seen += 1
        event: Optional[VADEvent] = None

        if not self._triggered:
            if probability >= self.threshold:
                self._speech_run += 1
                if self._speech_run >= self.min_speech_frames:
                    self._triggered = True
                    self._silence_run = 0
                    event = VADEvent.SPEECH_START
            else:
                self._speech_run = 0
        else:
            if probability >= self.neg_threshold:
                self._silence_run = 0
            else:
                self._silence_run += 1
                if self._silence_run >= self.min_silence_frames:
                    self._triggered = False
                    self._speech_run = 0
                    event = VADEvent.SPEECH_END

        return FrameDecision(
            probability=probability, is_speech=self._triggered, event=event
        )

    def close(self) -> Optional[VADEvent]:
        """End of stream: close an open segment so nothing is left dangling."""
        if self._triggered:
            self._triggered = False
            self._speech_run = 0
            self._silence_run = 0
            return VADEvent.SPEECH_END
        return None


# ---------------------------------------------------------------------------
# Silero model wrapper
# ---------------------------------------------------------------------------
class SileroVAD:
    """Per-frame speech probability from the Silero VAD model."""

    def __init__(self, onnx: bool = False, num_threads: int = 1) -> None:
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:                      # pragma: no cover
            raise VADError(
                "silero-vad / torch are not installed. Run "
                "`python3.11 -m pip install -r server/requirements.txt`."
            ) from exc

        self._torch = torch
        # The model is tiny and runs on one CPU core in far less than real
        # time; letting torch grab every core would only steal them from the
        # GPU feeder threads.
        torch.set_num_threads(num_threads)
        try:
            self._model = load_silero_vad(onnx=onnx)
        except Exception as exc:                        # pragma: no cover
            raise VADError(f"Could not load the Silero VAD model: {exc}") from exc
        self.onnx = onnx

    def probability(self, frame: np.ndarray) -> float:
        """Speech probability in [0, 1] for one 512 sample float32 frame."""
        if frame.shape[-1] != VAD_FRAME_SAMPLES:
            raise ValueError(
                f"Silero needs exactly {VAD_FRAME_SAMPLES} samples per frame, "
                f"got {frame.shape[-1]}"
            )
        tensor = self._torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        with self._torch.no_grad():
            return float(self._model(tensor, SAMPLE_RATE).item())

    def reset(self) -> None:
        """Clear the model's recurrent state between sessions."""
        self._model.reset_states()


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------
@dataclass
class VADStats:
    frames_total: int = 0
    frames_speech: int = 0
    segments: int = 0
    bytes_in: int = 0
    bytes_out: int = 0

    @property
    def speech_ratio(self) -> float:
        return self.frames_speech / self.frames_total if self.frames_total else 0.0

    @property
    def dropped_ratio(self) -> float:
        """Fraction of the incoming audio kept away from the heavy stages."""
        if not self.bytes_in:
            return 0.0
        return 1.0 - self.bytes_out / self.bytes_in


@dataclass
class SegmenterOutput:
    """Result of pushing one client chunk through the segmenter."""

    pcm: bytes = b""                                   # speech audio, may be b""
    events: list[SegmentEvent] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    is_speech: bool = False
    position_ms: float = 0.0                           # stream time after this chunk

    @property
    def has_audio(self) -> bool:
        return bool(self.pcm)

    @property
    def max_probability(self) -> float:
        return max(self.probabilities) if self.probabilities else 0.0

    @property
    def event_kinds(self) -> list[VADEvent]:
        return [e.kind for e in self.events]


class VADSegmenter:
    """Cut the incoming stream into timestamped speech segments.

    ``speech_pad_ms`` of audio is kept in a ring buffer at all times and
    prepended when a segment opens, because Silero needs a few frames of
    evidence before it triggers and those frames contain the start of the
    word.  Whisper transcribes a clipped onset as a different word, so the
    pad is not cosmetic.
    """

    def __init__(
        self,
        vad: Optional[SileroVAD] = None,
        threshold: float = VAD_THRESHOLD,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        min_silence_ms: int = VAD_MIN_SILENCE_MS,
        speech_pad_ms: int = VAD_SPEECH_PAD_MS,
        frame_samples: int = VAD_FRAME_SAMPLES,
    ) -> None:
        self.vad = vad if vad is not None else SileroVAD()
        self.splitter = FrameSplitter(frame_samples)
        self.frame_ms = frame_samples * 1000 // SAMPLE_RATE
        self.state = SpeechStateMachine(
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            frame_ms=self.frame_ms,
        )
        pad_frames = max(0, round(speech_pad_ms / self.frame_ms))
        self._preroll: deque[bytes] = deque(maxlen=pad_frames)
        self._frames_consumed = 0
        self.stats = VADStats()

    @property
    def position_ms(self) -> float:
        """Stream time of the next frame to be processed."""
        return self._frames_consumed * self.frame_ms

    def push(self, pcm: bytes) -> SegmenterOutput:
        """Feed one client chunk; get back the speech worth processing."""
        out = SegmenterOutput()
        self.stats.bytes_in += len(pcm)
        keep = bytearray()

        for frame in self.splitter.push(pcm):
            frame_start_ms = self._frames_consumed * self.frame_ms
            decision = self.state.push(self.vad.probability(frame.samples))
            out.probabilities.append(decision.probability)
            self.stats.frames_total += 1
            self._frames_consumed += 1

            if decision.event is VADEvent.SPEECH_START:
                self.stats.segments += 1
                # The forwarded audio begins at the oldest pre-roll frame, not
                # at the frame that convinced the model.
                out.events.append(
                    SegmentEvent(
                        kind=VADEvent.SPEECH_START,
                        at_ms=frame_start_ms - len(self._preroll) * self.frame_ms,
                    )
                )
                keep.extend(b"".join(self._preroll))
                self._preroll.clear()

            if decision.is_speech:
                self.stats.frames_speech += 1
                keep.extend(frame.pcm)
            else:
                # Frames after a SPEECH_END belong to the hangover: they were
                # already forwarded while the segment was still open.
                self._preroll.append(frame.pcm)

            if decision.event is VADEvent.SPEECH_END:
                # The hangover frames up to the previous frame were forwarded,
                # so the segment ends where this frame begins.
                out.events.append(
                    SegmentEvent(kind=VADEvent.SPEECH_END, at_ms=frame_start_ms)
                )

        out.pcm = bytes(keep)
        out.is_speech = self.state.is_speech
        out.position_ms = self.position_ms
        self.stats.bytes_out += len(out.pcm)
        return out

    def close(self) -> SegmenterOutput:
        """Flush at the end of a session so no segment stays open."""
        out = SegmenterOutput(is_speech=False, position_ms=self.position_ms)
        if self.state.close() is not None:
            out.events.append(
                SegmentEvent(kind=VADEvent.SPEECH_END, at_ms=self.position_ms)
            )
        return out

    def reset(self) -> None:
        """Start a fresh session without reloading the model."""
        self.splitter.reset()
        self.state.reset()
        self._preroll.clear()
        self.vad.reset()
        self._frames_consumed = 0
        self.stats = VADStats()
