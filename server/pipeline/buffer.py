"""Stream Buffer Manager - step 2 of the server pipeline.

``DESIGN.md`` 3.2: gather the speech coming out of the VAD into sentences and
fire a Finalize Event when one is over. Three things end a sentence: a pause
(the VAD closed the segment), a max-duration cut (somebody has talked past
the limit without stopping), or a speaker change, which
:mod:`server.pipeline.speaker_change` detects and turns into a call to
:meth:`BufferManager.cut_at`.

A max-duration cut lands on the quietest 32 ms frame within
``SPLIT_SEARCH_MS`` rather than exactly on the limit, because Whisper turns
half a word into a different word. Audio is never duplicated across the two
halves; the second is marked ``continues_previous``.

While an utterance is open it is also handed out periodically as a partial
window, which feeds the greyed-out running transcript.
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


def quietest_split_point(pcm: bytes, limit: int, search: int,
                         prefer_late: bool = False) -> int:
    """Byte offset of the quietest frame boundary in ``search`` before ``limit``.

    Cutting speech mid-word turns half a word into a different word, so a cut
    goes looking for the nearest dip. ``prefer_late`` settles ties at the
    latest frame rather than the earliest: speech with no dip in it has no
    boundary to find, and then the honest answer is ``limit`` itself rather
    than a search-window earlier.
    """
    limit = min(limit, len(pcm))
    earliest = max(FRAME_BYTES, limit - search)
    if limit <= earliest:
        return limit

    window = np.frombuffer(pcm[earliest:limit], dtype="<i2")
    frames = window.size // VAD_FRAME_SAMPLES
    if frames == 0:
        return limit

    usable = frames * VAD_FRAME_SAMPLES
    energy = (
        window[:usable].astype(np.float32).reshape(frames, VAD_FRAME_SAMPLES)
    )
    power = np.mean(energy * energy, axis=1)
    quietest = (frames - 1 - int(np.argmin(power[::-1])) if prefer_late
                else int(np.argmin(power)))
    # Cut after the quiet frame, so the silence stays with the first half
    # rather than opening the next utterance with it.
    return earliest + (quietest + 1) * FRAME_BYTES


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

    def tail(self, seconds: float) -> "PartialWindow":
        """The last ``seconds`` of it, for the ASR to decode.

        The running prediction re-decodes the whole open utterance every
        600 ms, so a seven-second sentence is decoded eleven times at growing
        lengths - about 45 seconds of audio for 7 seconds of speech, and the
        last pass is always the most expensive. Measured over ten minutes,
        partial decoding took 97.8 s against 21.6 s for every committed
        sentence put together, and one pass reached 4.7 s while the slowest
        sentence was 0.4 s. Every second of that is a second the socket was
        not being read.

        Capping the window bounds the worst pass and cuts the total. What it
        costs is the start of a long sentence: the grey text shows what is
        being said now rather than the whole sentence so far. The committed
        sentence is unaffected - it is decoded once, in full.

        The cut is a plain one, mid-word if that is where it falls. Hunting
        for a quiet frame is what the max-duration split does, and that
        matters because the committed sentence keeps the result; here the text
        is replaced 600 ms later.
        """
        wanted = int(seconds * SAMPLE_RATE) * SAMPLE_WIDTH
        if wanted <= 0 or len(self.pcm) <= wanted:
            return self
        cut = len(self.pcm) - wanted
        return PartialWindow(
            index=self.index,
            pcm=self.pcm[cut:],
            start_ms=self.start_ms + bytes_to_ms(cut),
        )


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

    def _discard_open(self) -> None:
        """Forget the open utterance, keeping the session's numbering."""
        self._pcm = bytearray()
        self._start_ms = None
        self._continues = False

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
                    self._discard_open()

        result.partial = self._maybe_partial()
        if result.partial is not None:
            self.stats.partials += 1
        return result

    def cut_at(self, offset_ms: float) -> BufferOutput:
        """End the open utterance ``offset_ms`` into it and keep the rest.

        Used when a second voice has taken over: the head belongs to whoever
        started, the tail to whoever is talking now. The split lands on the
        quietest frame in the ``split_search_ms`` before the offset, so it
        falls between words and keeps the newcomer's audio out of the
        committed half.

        Ties go to the latest frame. Two people talking over each other leave
        no dip to find, and cutting half a second early there leaves the old
        voice at the head of the new utterance, which is detected as another
        change and cut again - one handover, two sentences.
        """
        if not self.is_open:
            return BufferOutput()
        wanted = ms_to_bytes(offset_ms)
        if wanted <= 0 or wanted >= len(self._pcm):
            return BufferOutput()
        cut = self._quietest_split_point(wanted, prefer_late=True)
        if cut <= 0 or cut >= len(self._pcm):
            return BufferOutput()
        return BufferOutput(finals=[
            self._split(cut, FinalizeReason.SPEAKER_CHANGE, continues=False)])

    def flush(self, reason: FinalizeReason = FinalizeReason.END_OF_STREAM
              ) -> BufferOutput:
        """Commit whatever is still open, at the end of a session."""
        if not self.is_open or not self._pcm:
            self._discard_open()
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
        """Commit the first part of an over-long utterance and keep the rest.

        The speaker never stopped, so the second half carries straight on from
        the cut - no gap, no repeated audio.
        """
        return self._split(self._quietest_split_point(),
                           FinalizeReason.MAX_DURATION, continues=True)

    def _split(self, cut: int, reason: FinalizeReason,
               continues: bool) -> Utterance:
        """Commit ``self._pcm[:cut]`` and reopen on the rest."""
        assert self._start_ms is not None
        head, tail = bytes(self._pcm[:cut]), bytes(self._pcm[cut:])
        start_ms = self._start_ms

        self._pcm = bytearray(head)
        utterance = self._finalize(reason)

        self._open(start_ms + bytes_to_ms(len(head)))
        self._pcm = bytearray(tail)
        self._continues = continues
        return utterance

    def _quietest_split_point(self, limit: Optional[int] = None,
                              prefer_late: bool = False) -> int:
        """Where to cut the open utterance. ``limit`` defaults to the
        max-duration cut point."""
        if limit is None:
            limit = ms_to_bytes(self.max_duration_ms)
        return quietest_split_point(bytes(self._pcm), limit,
                                    ms_to_bytes(self.split_search_ms),
                                    prefer_late)

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
