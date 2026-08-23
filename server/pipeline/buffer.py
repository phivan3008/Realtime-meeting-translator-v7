"""Stream Buffer Manager - step 2 of the server pipeline.

``DESIGN.md`` section 3.2: gather the speech coming out of the VAD into a
sliding window, and fire a Finalize Event when the sentence is over.  Three
things end a sentence:

``pause``
    The VAD closed the segment, which it only does after a silence longer
    than the finalize threshold.  This is the normal case.

``max_duration``
    Somebody has been talking for more than seven seconds without a real
    pause.  Waiting longer would leave the viewer staring at grey partial
    text, so the utterance is cut even though the sentence is not over.

``speaker_change``
    Reserved for the diarization stage.  The hook exists
    (:meth:`BufferManager.notify_speaker_change`) but nothing calls it yet.

While an utterance is open it is also handed out periodically as a *partial*
window, which is what feeds the greyed-out running transcript.

Where the max-duration cut lands
--------------------------------
Cutting at exactly 7000 ms would usually land in the middle of a word, and
Whisper turns a half word into a different word.  So the cut looks back over
the last ``split_search_ms`` and lands on the quietest 32 ms frame it finds
there - the gap between words, if there is one.  The audio is never
duplicated across the two halves: whatever follows the cut opens the next
utterance, which is marked ``continues_previous`` so the translation stage
knows it is reading the middle of a sentence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from server.config import (
    FINALIZE_MAX_DURATION_MS,
    PARTIAL_INTERVAL_MS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    SPLIT_SEARCH_MS,
)
from server.pipeline.vad import VAD_FRAME_SAMPLES, SegmenterOutput

log = logging.getLogger(__name__)

FRAME_BYTES = VAD_FRAME_SAMPLES * SAMPLE_WIDTH
BYTES_PER_MS = SAMPLE_RATE * SAMPLE_WIDTH / 1000.0


def ms_to_bytes(ms: float) -> int:
    """Whole samples worth of ``ms``, so a slice never splits a sample."""
    return int(ms * BYTES_PER_MS) // SAMPLE_WIDTH * SAMPLE_WIDTH


def bytes_to_ms(size: int) -> float:
    return size / BYTES_PER_MS


class FinalizeReason(str, Enum):
    PAUSE = "pause"
    MAX_DURATION = "max_duration"
    SPEAKER_CHANGE = "speaker_change"
    END_OF_STREAM = "end_of_stream"


@dataclass(frozen=True)
class Utterance:
    """A committed span of speech, ready for the final ASR pass."""

    index: int
    pcm: bytes
    start_ms: float
    reason: FinalizeReason
    continues_previous: bool = False

    @property
    def duration_ms(self) -> float:
        return bytes_to_ms(len(self.pcm))

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms


@dataclass(frozen=True)
class PartialWindow:
    """The open utterance so far, for the running prediction."""

    index: int
    pcm: bytes
    start_ms: float

    @property
    def duration_ms(self) -> float:
        return bytes_to_ms(len(self.pcm))

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms


@dataclass
class BufferOutput:
    finals: list[Utterance] = field(default_factory=list)
    partial: Optional[PartialWindow] = None

    @property
    def has_work(self) -> bool:
        return bool(self.finals) or self.partial is not None


@dataclass
class BufferStats:
    utterances: int = 0
    partials: int = 0
    bytes_in: int = 0
    finalized_by: dict[str, int] = field(default_factory=dict)

    def record(self, reason: FinalizeReason) -> None:
        self.utterances += 1
        self.finalized_by[reason.value] = self.finalized_by.get(reason.value, 0) + 1


class BufferManager:
    """Turn a stream of VAD spans into finalised utterances and partials."""

    def __init__(
        self,
        max_duration_ms: int = FINALIZE_MAX_DURATION_MS,
        partial_interval_ms: int = PARTIAL_INTERVAL_MS,
        split_search_ms: int = SPLIT_SEARCH_MS,
    ) -> None:
        if max_duration_ms <= 0:
            raise ValueError("max_duration_ms must be positive")
        if split_search_ms >= max_duration_ms:
            raise ValueError("split_search_ms must be shorter than max_duration_ms")
        self.max_duration_ms = max_duration_ms
        self.partial_interval_ms = partial_interval_ms
        self.split_search_ms = split_search_ms
        self.stats = BufferStats()
        self._reset_state()

    def _reset_state(self) -> None:
        self._pcm = bytearray()
        self._start_ms: Optional[float] = None
        self._continues = False
        self._next_index = 0
        self._partial_at_ms = 0.0

    # -- introspection ------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._start_ms is not None

    @property
    def open_duration_ms(self) -> float:
        return bytes_to_ms(len(self._pcm))

    @property
    def open_index(self) -> int:
        return self._next_index

    # -- main entry point ---------------------------------------------------
    def push(self, out: SegmenterOutput) -> BufferOutput:
        """Absorb one segmenter output; return whatever it completed."""
        result = BufferOutput()
        self.stats.bytes_in += sum(len(span.pcm) for span in out.spans)

        for span in out.spans:
            if span.opens_segment and self.is_open:
                # The VAD always closes a segment before opening the next, so
                # this means a lost SPEECH_END. Commit what we have rather
                # than gluing two sentences together.
                log.warning("Segment opened while an utterance was still open")
                result.finals.append(self._finalize(FinalizeReason.PAUSE))

            if not self.is_open:
                if not span.opens_segment and not span.pcm:
                    continue            # a bare marker with nothing to buffer
                self._open(span.start_ms)

            self._pcm.extend(span.pcm)
            while self.open_duration_ms >= self.max_duration_ms:
                result.finals.append(self._cut_for_length())

            if span.closes_segment and self.is_open:
                if self._pcm:
                    result.finals.append(self._finalize(FinalizeReason.PAUSE))
                else:
                    # A length cut consumed everything just before the close.
                    self._start_ms = None
                    self._continues = False

        result.partial = self._maybe_partial()
        if result.partial is not None:
            self.stats.partials += 1
        return result

    def notify_speaker_change(self) -> BufferOutput:
        """Diarization hook: a new voice means the previous sentence is done."""
        if not self.is_open:
            return BufferOutput()
        return BufferOutput(finals=[self._finalize(FinalizeReason.SPEAKER_CHANGE)])

    def flush(self, reason: FinalizeReason = FinalizeReason.END_OF_STREAM
              ) -> BufferOutput:
        """Commit whatever is still open, at the end of a session."""
        if not self.is_open or not self._pcm:
            self._start_ms = None
            return BufferOutput()
        return BufferOutput(finals=[self._finalize(reason)])

    def reset(self) -> None:
        self._reset_state()
        self.stats = BufferStats()

    # -- internals ----------------------------------------------------------
    def _open(self, start_ms: float) -> None:
        self._start_ms = start_ms
        self._partial_at_ms = start_ms

    def _finalize(self, reason: FinalizeReason) -> Utterance:
        assert self._start_ms is not None
        utterance = Utterance(
            index=self._next_index,
            pcm=bytes(self._pcm),
            start_ms=self._start_ms,
            reason=reason,
            continues_previous=self._continues,
        )
        self.stats.record(reason)
        self._next_index += 1
        self._pcm = bytearray()
        self._start_ms = None
        self._continues = False
        return utterance

    def _cut_for_length(self) -> Utterance:
        """Commit the first part of an over-long utterance and keep the rest."""
        assert self._start_ms is not None
        cut = self._quietest_split_point()
        head, tail = bytes(self._pcm[:cut]), bytes(self._pcm[cut:])
        start_ms = self._start_ms

        self._pcm = bytearray(head)
        utterance = self._finalize(FinalizeReason.MAX_DURATION)

        # The speaker never stopped, so the next utterance carries straight on
        # from the cut - no gap, no repeated audio.
        self._open(start_ms + bytes_to_ms(len(head)))
        self._pcm = bytearray(tail)
        self._continues = True
        return utterance

    def _quietest_split_point(self) -> int:
        """Byte offset of the quietest frame boundary near the length limit."""
        limit = ms_to_bytes(self.max_duration_ms)
        earliest = max(FRAME_BYTES, ms_to_bytes(self.max_duration_ms
                                                - self.split_search_ms))
        if limit <= earliest:                       # pragma: no cover - guarded
            return limit

        window = np.frombuffer(self._pcm[earliest:limit], dtype="<i2")
        frames = window.size // VAD_FRAME_SAMPLES
        if frames == 0:
            return limit

        usable = frames * VAD_FRAME_SAMPLES
        energy = (
            window[:usable]
            .astype(np.float32)
            .reshape(frames, VAD_FRAME_SAMPLES)
        )
        quietest = int(np.argmin(np.mean(energy * energy, axis=1)))
        # Cut after the quiet frame, so the silence stays with the first half
        # rather than opening the next utterance with it.
        return earliest + (quietest + 1) * FRAME_BYTES

    def _maybe_partial(self) -> Optional[PartialWindow]:
        if not self.is_open or not self._pcm:
            return None
        assert self._start_ms is not None
        end_ms = self._start_ms + self.open_duration_ms
        if end_ms - self._partial_at_ms < self.partial_interval_ms:
            return None
        self._partial_at_ms = end_ms
        return PartialWindow(
            index=self._next_index,
            pcm=bytes(self._pcm),
            start_ms=self._start_ms,
        )
