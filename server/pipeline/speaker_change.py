"""Sentence boundaries at a change of voice - step 2b of the pipeline.

``DESIGN.md`` 3.2 lists three things that end a sentence. Two of them were
wired: a pause, and a max-duration cut. The third, a speaker change, existed
as a hook that nothing called, and a one-hour meeting showed what that costs.

The VAD closes a segment after ``VAD_MIN_SILENCE_MS`` of silence, so when the
next person starts sooner than that, both voices land in one utterance. That
utterance then gets one voiceprint, one language and one ASR pass:

* the two speakers come out as one, and the registry's centroid drifts toward
  the mixture, so the merged speaker goes on to absorb others
* the language identifier sees both languages and picks one, and Whisper is
  forced into it for the whole thing, which turns the other half into noise
* the running text only decodes the last ``PARTIAL_WINDOW_SECONDS``, so the
  first speaker's words scroll out of the prediction and never arrive as a
  committed sentence either

This watches the open utterance for the moment the voice changes. The first
second of it is the anchor - whoever started talking - and every partial the
last second is compared against that anchor. When they stop matching, the
utterance is cut just before the window that disagreed.

The anchor is embedded once per utterance, so the running cost is one voice
embedding per partial interval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from server.config import (
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    SPEAKER_CHANGE_THRESHOLD,
    SPEAKER_CHANGE_WINDOW_MS,
)
from server.pipeline.buffer import PartialWindow, bytes_to_ms
from server.pipeline.diarization import Embedder, cosine_similarity

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Change:
    """Where the voice changed, measured from the utterance's own start."""

    at_ms: float
    similarity: float


@dataclass
class ChangeStats:
    checks: int = 0
    changes: int = 0
    #: Every score seen, so a real meeting can be read back for where the
    #: same-voice and different-voice ranges actually sit.
    scores: list[float] = None          # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.scores is None:
            self.scores = []

    def record(self, similarity: float, changed: bool) -> None:
        self.checks += 1
        self.scores.append(similarity)
        if changed:
            self.changes += 1


class SpeakerChangeDetector:
    """Watch one open utterance for a second voice taking over."""

    def __init__(
        self,
        embedder: Embedder,
        window_ms: int = SPEAKER_CHANGE_WINDOW_MS,
        threshold: float = SPEAKER_CHANGE_THRESHOLD,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if window_ms <= 0:
            raise ValueError("window_ms must be positive")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a cosine, between -1 and 1")
        self.embedder = embedder
        self.window_ms = window_ms
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.stats = ChangeStats()
        self._anchor: Optional[np.ndarray] = None
        self._anchor_index = -1
        #: Byte offset the anchor was taken from. Non-zero after a change the
        #: buffer declined to act on, where the utterance still opens with the
        #: previous voice.
        self._anchor_from = 0

    @property
    def window_bytes(self) -> int:
        return int(self.window_ms * self.sample_rate / 1000.0) * SAMPLE_WIDTH

    def observe(self, window: PartialWindow) -> Optional[Change]:
        """Compare the newest second against whoever opened the utterance."""
        size = self.window_bytes
        if window.index != self._anchor_index:
            self._anchor = None
            self._anchor_index = window.index
            self._anchor_from = 0

        if len(window.pcm) < self._anchor_from + 2 * size:
            # Not enough for an anchor and a non-overlapping comparison yet.
            return None

        if self._anchor is None:
            anchor = self._embed(
                window.pcm[self._anchor_from:self._anchor_from + size])
            if anchor is None:
                return None
            self._anchor = anchor

        latest = self._embed(window.pcm[-size:])
        if latest is None:
            return None

        similarity = cosine_similarity(self._anchor, latest)
        changed = similarity < self.threshold
        self.stats.record(similarity, changed)
        log.debug("utterance %d voice similarity %.3f%s", window.index,
                  similarity, " - changed" if changed else "")
        if not changed:
            return None

        # Re-anchor on the newcomer: from here on the voice to follow is the
        # new one, not the one that opened the utterance. Normally the buffer
        # cuts and the next partial is a fresh utterance anyway, but when it
        # declines the cut this is what stops the same change being reported
        # on every partial for the rest of the sentence.
        self._anchor = None
        self._anchor_from = len(window.pcm) - size
        log.info("utterance %d: a second voice took over at %.0f ms "
                 "(similarity %.3f)", window.index,
                 bytes_to_ms(len(window.pcm) - size), similarity)
        return Change(at_ms=bytes_to_ms(len(window.pcm) - size),
                      similarity=similarity)

    def reset(self) -> None:
        """A new meeting starts with nobody anchored."""
        self._anchor = None
        self._anchor_index = -1
        self._anchor_from = 0
        self.stats = ChangeStats()

    def _embed(self, pcm: bytes) -> Optional[np.ndarray]:
        embedding = self.embedder.embed(pcm)
        if embedding.size == 0 or not np.all(np.isfinite(embedding)):
            return None
        return embedding
