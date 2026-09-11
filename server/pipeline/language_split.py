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
is good at: probe near the start, probe near the end, and only a *confident*
disagreement is worth acting on.  Same language, either end undecided, or
either end decided without much margin, and the utterance is left alone - two
probes, about 6 ms each, on the ordinary case.

**Near** the start and the end, not at them.  The first ``VAD_SPEECH_PAD_MS``
of an utterance is pre-roll kept for the word onset and the last
``VAD_MIN_SILENCE_MS`` is hangover, so the outermost second is the least
representative audio there is.  A first version probed exactly that, and over
one real meeting 26 of the 41 splits a viewer could see gave both halves the
same language afterwards - the probes had been wrong about audio the whole
halves agree on.

Which is why a cut is reviewed before it is taken.  The two halves that would
actually be emitted are probed in full, and unless *they* confidently
disagree the utterance is left whole.  Two more probes, and they are the ones
that decide.

Where the cut lands matters as much as whether to cut.  Whisper turns half a
word into a different word, so the boundary is placed on the quietest 32 ms
frame inside the range the search narrowed it to.

And a fragment too short to transcribe is not left empty - Whisper fills it
in.  That is not a hypothetical.  An earlier version reported a boundary a few
hundred milliseconds from the start and the half that was left came back as a
YouTube subscribe line 31 ms after the first was committed; a later one, with
its floor at 800 ms, produced "All right, they will." where the audio said
"bending the material".  So: the search never starts closer than one probe
plus the edge inset to either end, no half under
``LANGUAGE_SPLIT_MIN_PART_MS`` is returned, and the review has the last word.
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
)
from server.config import VAD_MIN_SILENCE_MS
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


def middle_of(pcm: bytes, most: int) -> bytes:
    """The middle ``most`` bytes of a span, for a probe that must not read
    the edges.

    The first VAD_SPEECH_PAD_MS of an utterance is pre-roll and the last
    VAD_MIN_SILENCE_MS is hangover. A probe over a whole half reads both, and
    a probe over a *tail* half reads the hangover as a large share of what it
    is given - which is how it came back confident about silence.
    """
    if len(pcm) <= most:
        return pcm
    start = _on_sample((len(pcm) - most) // 2)
    return pcm[start:start + most]


def _confident(decision, margin: float) -> bool:
    """Whether a probe is sure enough to be worth cutting on.

    ``known`` alone is the bar for forcing a language on the ASR, which is a
    reversible sort of wrong - the words come out in the wrong script and a
    person can see it. Cutting is not reversible: it manufactures a second
    utterance, and a fragment does not come back empty from Whisper.
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
) -> Optional[Split]:
    """Where this utterance changes language, or None to leave it whole.

    Returns None far more often than not, and cheaply: two probes decide it
    unless the two ends confidently disagree.
    """
    probe = ms_to_bytes(probe_ms)
    floor = ms_to_bytes(min_part_ms)
    # The last half second of every utterance is the VAD's hangover, so a
    # tail of `floor` bytes holds less speech than a head of the same size.
    # The floor on the right pays for that difference.
    tail_floor = ms_to_bytes(min_part_ms + hangover_ms)
    edge = ms_to_bytes(edge_ms)
    if len(pcm) < 2 * (probe + edge) or len(pcm) < floor + tail_floor:
        # Too short to hold two turns, and too short to probe both ends
        # without the windows overlapping - which would compare a span with
        # itself.
        return None

    # Inset from both ends. The first VAD_SPEECH_PAD_MS is pre-roll and the
    # last VAD_MIN_SILENCE_MS is hangover, so the outermost second is the
    # least representative audio in the utterance - and it was what the first
    # version of this probed.
    probes = 2
    first = prober.identify(pcm[edge:edge + probe])
    second = prober.identify(pcm[len(pcm) - edge - probe:len(pcm) - edge])
    if not _confident(first, min_margin) or not _confident(second, min_margin):
        # The LID says it cannot tell, or cannot tell firmly enough. Neither
        # is evidence of two languages.
        return None
    if first.lang_code == second.lang_code:
        return None

    # The change is somewhere between the two probes. Each step asks what
    # language the audio just before the midpoint is in, which is the
    # question a boundary search can actually answer: a window straddling the
    # change reads as the language it ends in.
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

    # Cut on the quietest frame inside what the search narrowed it to, not
    # within a fixed window of the midpoint. Three halvings of a five-second
    # utterance leave about 600 ms of uncertainty, and looking back only
    # 200 ms from the middle of that cannot reach the pause between the two
    # turns - it lands inside the second one, through a word.
    search = max(high - low, ms_to_bytes(snap_ms))
    at = quietest_split_point(pcm, high, search)

    if at < floor or len(pcm) - at < tail_floor:
        # The snap pulled the cut under the floor, or the search landed there
        # to begin with. A sliver is worse than no cut: Whisper does not
        # return nothing for it, it returns a sign-off.
        log.debug("Language split refused: %.0f ms / %.0f ms around a %s->%s "
                  "change", bytes_to_ms(at), bytes_to_ms(len(pcm) - at),
                  first.lang_code, second.lang_code)
        return None

    # Verify on the halves that would actually be emitted, rather than on two
    # one-second windows that may have been the only parts in disagreement.
    # This is the check that was missing: over a real meeting, 26 of 41
    # visible splits gave both halves the same language afterwards, which is
    # the screening probes having been wrong about audio the whole halves
    # agree on.
    probes += 2
    review = ms_to_bytes(review_ms)
    head = prober.identify(middle_of(pcm[:at], review))
    tail = prober.identify(middle_of(pcm[at:], review))
    if (not _confident(head, min_margin) or not _confident(tail, min_margin)
            or head.lang_code == tail.lang_code):
        log.debug("Language split refused on review: %s/%s across the probes "
                  "but %s/%s across the halves",
                  first.lang_code, second.lang_code,
                  head.lang_code or "?", tail.lang_code or "?")
        return None

    return Split(at=at, first=head.lang_code, second=tail.lang_code,
                 probes=probes)


def _on_sample(offset: int) -> int:
    """Round an offset down to a whole sample, so a slice never splits one."""
    return offset - (offset % 2)
