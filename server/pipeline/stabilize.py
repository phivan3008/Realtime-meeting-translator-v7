"""Agreement across running texts - what the open utterance has settled on.

The running text is decoded independently every 800 ms on the last four
seconds of the open utterance.  Two things follow from that, and this module
exists because of both.

**The window slides.**  Past four seconds the head of the sentence leaves the
window, so the newest running text is not an extension of the one before it -
it is a later, shorter view of the same sentence.  Measured over three real
meetings, two thirds of the updates wiped more than half the text that was on
screen.  So the running texts are stitched back together by their overlap
before anything is compared.

**Each decode is a fresh draw.**  Two decodes of the same audio agree on what
was clearly said and differ on what was not, which makes agreement between
consecutive decodes a usable confidence signal - the idea behind
LocalAgreement, and the part of it worth having.

What this module does *not* do is build the text out of tokens.  A stable
prefix is always a slice of a string Whisper itself produced, joined at the
few points where two running texts meet, and even there a space is inserted
only between characters that take one.  Rendering a Japanese sentence from a
word list is how ``アプリケーション`` becomes ``ア プ リ ケ ー ショ ン``: correct
on the server, unreadable on screen, and worse in a translation prompt, where
the model is handed characters where it expects words.

Nothing here decides anything on its own.  It produces a reference; the
session decides what to do with it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from server.config import (
    STABLE_MIN_AGREEMENT,
    STABLE_MIN_OVERLAP_TOKENS,
    STABLE_OVERLAP_MATCH,
    STABLE_HISTORY,
)
from server.pipeline.asr import normalise_for_pattern
from server.pipeline.textdiff import drift

#: One token: a single CJK character, or a run of word characters in any
#: other script. The CJK branch comes first so a Japanese run yields one
#: token per character - Japanese is written without spaces, and matching a
#: whole run would make agreement all-or-nothing for a whole clause.
_TOKEN = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿]"
    r"|[^\W_]+",
    re.UNICODE,
)

#: Characters that do not take a space beside them.
_TIGHT = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")


@dataclass(frozen=True)
class Token:
    """One word of one running text, and where it came from.

    ``source`` and the offsets are kept so the stable text can be rendered by
    slicing the original string rather than by joining the tokens back
    together.
    """

    normalized: str
    source: str
    start: int
    end: int


def tokenize(text: str) -> list:
    """Split into comparable tokens without losing the original spelling."""
    return [
        Token(normalized=match.group(0).casefold(), source=text,
              start=match.start(), end=match.end())
        for match in _TOKEN.finditer(text)
    ]


def join(left: str, right: str) -> str:
    """Put two fragments together, with a space only where one belongs."""
    left, right = left.rstrip(), right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if _TIGHT.search(left[-1]) or _TIGHT.search(right[0]):
        return left + right
    return f"{left} {right}"


def render(tokens: list) -> str:
    """The original text behind a run of tokens, spliced where sources change.

    Tokens from one running text are rendered as one slice of it, so the
    spacing, casing and punctuation are whatever Whisper wrote. Only where
    two running texts meet is there a join to make, and :func:`join` makes it
    the way the scripts on either side want it.
    """
    if not tokens:
        return ""

    def slice_of(source: str, start: int, end: int) -> str:
        """The run, plus a trailing full stop when nothing else follows it.

        Punctuation is not a token, so a run that reaches the end of its
        source would otherwise drop the sentence's own full stop. A run that
        stops mid-source keeps its cut at the token boundary, which is what a
        prefix wants: a settled prefix must not claim the comma belonging to
        a word that is still moving.
        """
        if not _TOKEN.search(source, end):
            return source[start:]
        return source[start:end]

    text = ""
    run_source, run_start, run_end = tokens[0].source, tokens[0].start, tokens[0].end
    for token in tokens[1:]:
        if token.source is run_source and token.start >= run_end:
            run_end = token.end
            continue
        text = join(text, slice_of(run_source, run_start, run_end))
        run_source, run_start, run_end = token.source, token.start, token.end
    return join(text, slice_of(run_source, run_start, run_end)).strip()


def overlap(earlier: list, later: list, minimum: int,
            match: float = STABLE_OVERLAP_MATCH) -> int:
    """How many tokens at the end of ``earlier`` open ``later``.

    Consecutive running texts cover overlapping audio - four seconds of
    window advancing 800 ms at a time - so the newest one usually starts in
    the middle of the previous one. The longest such overlap is where to
    splice them.

    The match is approximate on purpose. Two decodes of the same audio agree
    on most of it and reword the rest, so demanding an exact token match
    finds no overlap at all on about a third of real pairs, and a splice that
    is not found is a sentence that gets said twice: measured on one meeting,
    exact matching put "ステップ011の方は" into one reference three times over.

    Returns 0 when there is no overlap worth trusting. Two tokens is the
    floor: a single shared word matches by accident constantly.
    """
    limit = min(len(earlier), len(later))
    for size in range(limit, minimum - 1, -1):
        tail = [token.normalized for token in earlier[-size:]]
        head = [token.normalized for token in later[:size]]
        same = sum(1 for a, b in zip(tail, head) if a == b)
        if same >= match * size:
            return size
    return 0


def stitch(earlier: list, later: list, minimum: int) -> list:
    """Extend ``earlier`` with whatever ``later`` adds beyond the overlap."""
    if not earlier:
        return list(later)
    if not later:
        return list(earlier)
    size = overlap(earlier, later, minimum)
    if size:
        return list(earlier) + list(later[size:])
    # No overlap at all. Concatenating here is what makes a reference say the
    # same clause twice, and a reference that repeats itself is worse than a
    # short one: both of the things it is used for - voting on the language,
    # and asking whether the sentence has wandered off entirely - survive
    # being short, and neither survives being wrong. Keep whichever view
    # holds more.
    return list(earlier) if len(earlier) >= len(later) else list(later)


def agreed_prefix(hypotheses: list, needed: int) -> int:
    """How many leading tokens the last ``needed`` hypotheses all share."""
    if len(hypotheses) < needed:
        return 0
    recent = hypotheses[-needed:]
    shortest = min(len(hypothesis) for hypothesis in recent)
    count = 0
    while count < shortest:
        word = recent[0][count].normalized
        if any(hypothesis[count].normalized != word for hypothesis in recent):
            break
        count += 1
    return count


@dataclass(frozen=True)
class Stable:
    """What an open utterance has settled on, and what it has not."""

    #: The agreed prefix, as Whisper wrote it.
    text: str
    #: The rest of the newest view, still liable to change.
    tail: str
    #: Everything seen so far, agreed or not. This is the reference a
    #: committed sentence is compared against: it covers the whole utterance,
    #: where the newest running text covers only the last four seconds.
    whole: str
    #: What the running texts agreed this utterance was in.
    language: str
    views: int = 0

    @property
    def has_text(self) -> bool:
        return bool(self.whole.strip())


@dataclass
class UtteranceView:
    hypotheses: list = field(default_factory=list)
    languages: list = field(default_factory=list)


class Stabilizer:
    """Accumulate the running texts of each open utterance."""

    def __init__(self, min_agreement: int = STABLE_MIN_AGREEMENT,
                 min_overlap: int = STABLE_MIN_OVERLAP_TOKENS,
                 history: int = STABLE_HISTORY) -> None:
        if min_agreement < 2:
            raise ValueError("min_agreement must be at least 2 - agreement "
                             "between one decode and itself is not evidence")
        if min_overlap < 1:
            raise ValueError("min_overlap must be at least 1")
        self.min_agreement = min_agreement
        self.min_overlap = min_overlap
        self.history = history
        self._views: dict = {}

    def observe(self, index: int, text: str, lang_code: str = "") -> Stable:
        """Take one running text for the utterance numbered ``index``."""
        view = self._views.setdefault(index, UtteranceView())
        if lang_code:
            view.languages.append(lang_code)
        if text.strip():
            previous = view.hypotheses[-1] if view.hypotheses else []
            view.hypotheses.append(
                stitch(previous, tokenize(text), self.min_overlap))
            if len(view.hypotheses) > self.history:
                del view.hypotheses[:-self.history]
        return self._stable(view)

    def stable(self, index: int) -> Stable:
        view = self._views.get(index)
        return self._stable(view) if view is not None else Stable("", "", "", "")

    def release(self, index: int) -> Stable:
        """Read an utterance's view and forget it - the sentence is committed."""
        view = self._views.pop(index, None)
        return self._stable(view) if view is not None else Stable("", "", "", "")

    def reset(self) -> None:
        self._views = {}

    def _stable(self, view: UtteranceView) -> Stable:
        if not view.hypotheses:
            return Stable("", "", "", self._language(view))
        newest = view.hypotheses[-1]
        count = agreed_prefix(view.hypotheses, self.min_agreement)
        return Stable(
            text=render(newest[:count]),
            tail=render(newest[count:]),
            whole=render(newest),
            language=self._language(view),
            views=len(view.hypotheses),
        )

    @staticmethod
    def _language(view: UtteranceView) -> str:
        if not view.languages:
            return ""
        return Counter(view.languages).most_common(1)[0][0]


def drift_from(text: str, reference: str) -> float:
    """Share of ``reference`` the sentence did not keep, 0.0 to 1.0.

    Both sides are reduced to spoken words first: a sentence that differs
    only in punctuation and casing is the same sentence, and Whisper
    punctuates the same audio differently on every pass.
    """
    return drift(normalise_for_pattern(text), normalise_for_pattern(reference))
