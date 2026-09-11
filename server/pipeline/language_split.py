"""Cut an utterance that holds two languages.

The VAD closes a segment on 500 ms of silence, and people answer each other
faster than that.  So a reply in the other language lands inside the same
utterance, the LID is asked which language it is and has to name one, and
Whisper is forced into that one for all of it.  What comes back is fluent,
confident, and missing a turn:

    partial  [vi] không có dung lượng để mình thực hiện đi kiểm chứng tiếp
    partial  [ja] ステップ011の方は
    final    [vi] Còn cái dấu hiệu đấy thì mình đi kiếm tiếp tục.

The Vietnamese that was really said is gone, and so is the Japanese.  Nothing
downstream can recover either, because the audio only gets decoded once.

So the question is asked of the audio directly, and it is a question the LID
is good at: probe the start, probe the end, and only a *confident*
disagreement is worth acting on.  Same language, or either end undecided, and
the utterance is left alone - two probes, about 6 ms each, on the ordinary
case.

Where the cut lands matters as much as whether to cut.  Whisper turns half a
word into a different word, so the boundary the search returns is snapped to
the quietest 32 ms frame near it.

And a fragment too short to transcribe is not left empty - Whisper fills it
in.  That is not a hypothetical: an earlier version of this search reported a
boundary a few hundred milliseconds from the start, the quiet-frame snap took
another 500 ms off it, and the second half came back as a YouTube subscribe
line 31 ms after the first half was committed.  Two floors guard it now: the
search never starts closer than one probe to either end, and a boundary the
snap pulls under the floor is refused rather than used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

from server.config import (
    LANGUAGE_SPLIT_MAX_STEPS,
    LANGUAGE_SPLIT_MIN_PART_MS,
    LANGUAGE_SPLIT_PROBE_MS,
    LANGUAGE_SPLIT_SNAP_MS,
)
from server.pipeline.buffer import bytes_to_ms, ms_to_bytes, quietest_split_point

log = logging.getLogger(__name__)


class Prober(Protocol):
    """What this needs of the LID; a stub satisfies it in the tests."""

    def identify(self, pcm: bytes):
        ...                                         # pragma: no cover


@dataclass(frozen=True)
class Split:
    """Where an utterance changes language, and what it changes between."""

    #: Byte offset of the cut, always on a sample boundary.
    at: int
    first: str
    second: str
    #: LID calls it took, so the cost is visible in the session summary.
    probes: int

    @property
    def at_ms(self) -> float:
        return bytes_to_ms(self.at)


def find_split(
    pcm: bytes,
    prober: Prober,
    probe_ms: float = LANGUAGE_SPLIT_PROBE_MS,
    min_part_ms: float = LANGUAGE_SPLIT_MIN_PART_MS,
    snap_ms: float = LANGUAGE_SPLIT_SNAP_MS,
    max_steps: int = LANGUAGE_SPLIT_MAX_STEPS,
) -> Optional[Split]:
    """Where this utterance changes language, or None to leave it whole.

    Returns None far more often than not, and cheaply: two probes decide it
    unless the two ends confidently disagree.
    """
    probe = ms_to_bytes(probe_ms)
    floor = ms_to_bytes(min_part_ms)
    if len(pcm) < 2 * max(probe, floor):
        # Too short to hold two turns, and too short to probe both ends
        # without the windows overlapping - which would compare a span with
        # itself.
        return None

    probes = 2
    first = prober.identify(pcm[:probe])
    second = prober.identify(pcm[-probe:])
    if not first.known or not second.known:
        # The LID says it cannot tell. That is not evidence of one language
        # and it is certainly not evidence of two.
        return None
    if first.lang_code == second.lang_code:
        return None

    # The change is somewhere between the two probes. Each step asks what
    # language the audio just before the midpoint is in, which is the
    # question a boundary search can actually answer: a window straddling the
    # change reads as the language it ends in.
    low, high = probe, len(pcm) - probe
    for _step in range(max_steps):
        if high - low <= probe:
            break
        middle = _on_sample(low + (high - low) // 2)
        probes += 1
        decision = prober.identify(pcm[max(middle - probe, 0):middle])
        if not decision.known:
            break
        if decision.lang_code == first.lang_code:
            low = middle
        elif decision.lang_code == second.lang_code:
            high = middle
        else:
            break                       # a third language; stop guessing

    boundary = _on_sample(low + (high - low) // 2)
    at = quietest_split_point(pcm, boundary, ms_to_bytes(snap_ms))

    if at < floor or len(pcm) - at < floor:
        # The snap pulled the cut under the floor, or the search landed there
        # to begin with. A sliver is worse than no cut: Whisper does not
        # return nothing for it, it returns a sign-off.
        log.debug("Language split refused: %.0f ms / %.0f ms around a %s->%s "
                  "change", bytes_to_ms(at), bytes_to_ms(len(pcm) - at),
                  first.lang_code, second.lang_code)
        return None

    return Split(at=at, first=first.lang_code, second=second.lang_code,
                 probes=probes)


def _on_sample(offset: int) -> int:
    """Round an offset down to a whole sample, so a slice never splits one."""
    return offset - (offset % 2)
