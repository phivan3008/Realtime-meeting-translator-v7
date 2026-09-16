"""Cut an utterance that holds two languages.

The VAD closes a segment on 500 ms of silence, and people answer each other
faster than that.  So a reply in the other language lands inside the same
utterance, the LID is asked which language it is and has to name one, and
Whisper is forced into that one for all of it.  What comes back is fluent,
confident, and missing a turn::

    partial  [vi] không có dung lượng để mình thực hiện đi kiểm chứng tiếp
    partial  [ja] ステップ011の方は
    final    [vi] Còn cái dấu hiệu đấy thì mình đi kiếm tiếp tục.

So the question is asked of the audio directly, and it is one the LID is good
at: probe near the start, probe near the end, and only a *confident*
disagreement is worth acting on.  Two probes on the ordinary case.

**Near** the ends, not at them.  The start of an utterance is VAD pre-roll and
the end is its silence hangover, the least representative audio there is.
Probing exactly there gave 26 of 41 visible splits the same language on both
halves afterwards.

Which is also why a cut is reviewed before it is taken: the two halves that
would actually be emitted are probed, from their middles, and unless *they*
confidently disagree the utterance is left whole.

Where the cut lands matters as much as whether to cut.  Whisper turns half a
word into a different word, so the boundary is placed on the quietest 32 ms
frame inside the range the search narrowed it to.  And a fragment too short
to transcribe is not left empty - Whisper fills it in ("All right, they
will." over audio that said "bending the material"), so no half under
``LANGUAGE_SPLIT_MIN_PART_MS`` of speech is ever returned.

Every refusal is counted by reason.  A splitter that declines in silence is
how the case it was built for once went by with no log line at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

from server.config import (
    LANGUAGE_SPLIT_EDGE_MS,
    LANGUAGE_SPLIT_MAX_STEPS,
    LANGUAGE_SPLIT_MIN_MARGIN,
    LANGUAGE_SPLIT_MIN_PART_MS,
    LANGUAGE_SPLIT_PROBE_MS,
    LANGUAGE_SPLIT_REVIEW_MS,
    LANGUAGE_SPLIT_SNAP_MS,
    VAD_MIN_SILENCE_MS,
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


@dataclass(frozen=True)
class Verdict:
    """What the search concluded, cut or not, and why."""

    split: Optional[Split]
    #: One of the :class:`SplitStats` counter names.
    reason: str
    probes: int
    detail: str = ""


@dataclass
class SplitStats:
    checked: int = 0
    split: int = 0
    #: Both ends confidently the same language.
    one_language: int = 0
    #: An end the LID could not decide, or not firmly enough to cut on.
    undecided: int = 0
    #: Too short to hold two turns.
    too_short: int = 0
    #: A cut would have left a fragment Whisper fills in.
    sliver: int = 0
    #: The screening probes disagreed and the halves did not.
    refused_on_review: int = 0
    probes: int = 0
    #: What the last utterance looked like, for the lost-turn log line.
    last: str = ""


def middle_of(pcm: bytes, most: int) -> bytes:
    """The middle ``most`` bytes of a span, so a probe skips both edges."""
    if len(pcm) <= most:
        return pcm
    start = _on_sample((len(pcm) - most) // 2)
    return pcm[start:start + most]


def _confident(decision, margin: float) -> bool:
    """Whether a probe is sure enough to be worth cutting on.

    ``known`` alone is the bar for forcing a language on the ASR, which is a
    visible sort of wrong. Cutting manufactures a second utterance, and a
    fragment does not come back empty from Whisper.
    """
    return decision.known and decision.margin >= margin


def find_split(
    pcm: bytes,
    prober: Prober,
    probe_ms: float = LANGUAGE_SPLIT_PROBE_MS,
    min_part_ms: float = LANGUAGE_SPLIT_MIN_PART_MS,
    snap_ms: float = LANGUAGE_SPLIT_SNAP_MS,
    max_steps: int = LANGUAGE_SPLIT_MAX_STEPS,
    edge_ms: float = LANGUAGE_SPLIT_EDGE_MS,
    min_margin: float = LANGUAGE_SPLIT_MIN_MARGIN,
    review_ms: float = LANGUAGE_SPLIT_REVIEW_MS,
    hangover_ms: float = VAD_MIN_SILENCE_MS,
) -> Verdict:
    """Where this utterance changes language, if it does.

    ``hangover_ms`` is how much of the end of ``pcm`` is silence the VAD
    forwarded: a tail has to clear the floor in speech, not in bytes.
    """
    probe = ms_to_bytes(probe_ms)
    floor = ms_to_bytes(min_part_ms)
    tail_floor = ms_to_bytes(min_part_ms + hangover_ms)
    edge = ms_to_bytes(edge_ms)
    if len(pcm) < 2 * (probe + edge) or len(pcm) < floor + tail_floor:
        # Too short to hold two turns, and too short to probe both ends
        # without the windows overlapping.
        return Verdict(None, "too_short", 0,
                       f"{bytes_to_ms(len(pcm)):.0f} ms")

    probes = 2
    first = prober.identify(pcm[edge:edge + probe])
    second = prober.identify(pcm[len(pcm) - edge - probe:len(pcm) - edge])
    ends = (f"{first.lang_code or '?'} ({first.margin:.2f}) ... "
            f"{second.lang_code or '?'} ({second.margin:.2f})")
    if not _confident(first, min_margin) or not _confident(second, min_margin):
        return Verdict(None, "undecided", probes, ends)
    if first.lang_code == second.lang_code:
        return Verdict(None, "one_language", probes, ends)

    # The change is somewhere between the two probes. Each step asks what
    # language the audio just before the midpoint is in: a window straddling
    # the change reads as the language it ends in.
    low, high = edge + probe, len(pcm) - edge - probe
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

    # Cut on the quietest frame inside what the search narrowed it to. Three
    # halvings leave hundreds of milliseconds of uncertainty, and a fixed
    # short look-back could not reach the pause between the turns.
    search = max(high - low, ms_to_bytes(snap_ms))
    at = quietest_split_point(pcm, high, search)

    if at < floor or len(pcm) - at < tail_floor:
        return Verdict(None, "sliver", probes,
                       f"{bytes_to_ms(at):.0f} ms / "
                       f"{bytes_to_ms(len(pcm) - at):.0f} ms around a "
                       f"{first.lang_code}->{second.lang_code} change")

    # Verify on the halves that would actually be emitted.
    probes += 2
    review = ms_to_bytes(review_ms)
    head = prober.identify(middle_of(pcm[:at], review))
    tail = prober.identify(middle_of(pcm[at:], review))
    if (not _confident(head, min_margin) or not _confident(tail, min_margin)
            or head.lang_code == tail.lang_code):
        return Verdict(None, "refused_on_review", probes,
                       f"{first.lang_code}/{second.lang_code} across the "
                       f"probes but {head.lang_code or '?'}/"
                       f"{tail.lang_code or '?'} across the halves")

    split = Split(at=at, first=head.lang_code, second=tail.lang_code,
                  probes=probes)
    return Verdict(split, "split", probes,
                   f"{split.first} until {split.at_ms:.0f} ms, "
                   f"then {split.second}")


class LanguageSplitter:
    """:func:`find_split` with the counting the session summary reports."""

    def __init__(self, prober: Prober, **settings) -> None:
        self.prober = prober
        self.settings = settings
        self.stats = SplitStats()

    def find(self, pcm: bytes,
             hangover_ms: float = VAD_MIN_SILENCE_MS) -> Optional[Split]:
        verdict = find_split(pcm, self.prober, hangover_ms=hangover_ms,
                             **self.settings)
        stats = self.stats
        stats.probes += verdict.probes
        if verdict.reason != "too_short":
            stats.checked += 1
        setattr(stats, verdict.reason, getattr(stats, verdict.reason) + 1)
        stats.last = f"{verdict.reason.replace('_', ' ')}: {verdict.detail}"
        if verdict.split is None:
            log.debug("Language split declined - %s", stats.last)
        return verdict.split

    def reset(self) -> None:
        self.stats = SplitStats()


def _on_sample(offset: int) -> int:
    """Round an offset down to a whole sample, so a slice never splits one."""
    return offset - (offset % 2)
