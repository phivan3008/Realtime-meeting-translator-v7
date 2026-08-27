"""Split an utterance that holds two languages - step 6b of the pipeline.

The VAD closes a segment after ``VAD_MIN_SILENCE_MS`` of silence, and people
answer each other faster than that. When the reply is in the other language,
both land in one utterance, the language identifier has to pick one, and
Whisper is forced into it for all of it. The half in the other language does
not come out mistranslated - it does not come out at all.

Measured over a real ten-minute meeting: the running text and the committed
sentence disagreed about the language on 8 of 119 utterances, and reading
those eight back, **4 had genuinely lost a turn**. Half the flags were real.

Fifty percent is not good enough to cut on, so the flag is not what decides.
It was only ever a proxy for the question worth asking, which is whether the
audio actually contains two languages - and the language identifier can be
asked that directly.

So: probe the start, probe the end, and if they disagree, binary-search the
boundary between them. Nothing is cut unless the two ends are confidently
different languages, which is a far stronger test than a partial and a final
disagreeing. Then the cut is snapped to the quietest frame nearby, because
Whisper turns half a word into a different word.

About five probes per utterance, at roughly 6 ms each. Compare with the
speaker boundary that had to be abandoned: that one fired on 53% of its
comparisons because one-second voiceprints do not separate speakers. This one
asks a model a question it is good at, over windows it was measured on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

from server.config import (
    LANGUAGE_SPLIT_PROBE_MS,
    LANGUAGE_SPLIT_STEPS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)

log = logging.getLogger(__name__)


class Identifier(Protocol):
    """What this needs of the LID; a stub satisfies it in the tests."""

    def identify(self, pcm: bytes):
        ...                                         # pragma: no cover


@dataclass(frozen=True)
class Boundary:
    """Where one language gives way to the other, from the utterance's start."""

    at_ms: float
    before: str
    after: str


@dataclass
class SplitStats:
    checked: int = 0
    split: int = 0
    probes: int = 0


class LanguageSplitter:
    """Find the point in an utterance where the language changes."""

    def __init__(
        self,
        identifier: Identifier,
        probe_ms: int = LANGUAGE_SPLIT_PROBE_MS,
        steps: int = LANGUAGE_SPLIT_STEPS,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if probe_ms <= 0:
            raise ValueError("probe_ms must be positive")
        if steps < 0:
            raise ValueError("steps cannot be negative")
        self.identifier = identifier
        self.probe_ms = probe_ms
        self.steps = steps
        self.sample_rate = sample_rate
        self.stats = SplitStats()

    @property
    def probe_bytes(self) -> int:
        return int(self.probe_ms * self.sample_rate / 1000.0) * SAMPLE_WIDTH

    def find(self, pcm: bytes) -> Optional[Boundary]:
        """The boundary between two languages, or None if there is only one."""
        probe = self.probe_bytes
        if len(pcm) < 2 * probe:
            # One probe at each end, not overlapping. Anything shorter cannot
            # answer the question, and the LID does not trust windows this
            # short anyway.
            return None

        self.stats.checked += 1
        first = self._language(pcm[:probe])
        last = self._language(pcm[-probe:])
        if not first or not last or first == last:
            return None

        # Everything up to `low` sounds like the first language, everything
        # from `high` like the second. Narrow the gap between them.
        low, high = 0, len(pcm) - probe
        for _ in range(self.steps):
            middle = (low + high) // 2
            if middle <= low or middle >= high:
                break
            if self._language(pcm[middle:middle + probe]) == first:
                low = middle
            else:
                high = middle

        at_ms = high / (self.sample_rate * SAMPLE_WIDTH) * 1000.0
        self.stats.split += 1
        log.info("two languages in one utterance: %r until %.0f ms, then %r",
                 first, at_ms, last)
        return Boundary(at_ms=at_ms, before=first, after=last)

    def reset(self) -> None:
        self.stats = SplitStats()

    def _language(self, pcm: bytes) -> str:
        self.stats.probes += 1
        return self.identifier.identify(pcm).lang_code
