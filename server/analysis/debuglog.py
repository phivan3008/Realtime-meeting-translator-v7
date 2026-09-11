"""Read the client's debug log and measure what a meeting actually showed.

The client writes one line per event - the running text as it was replaced,
the sentence when it was committed, the translation when it arrived. That
file is the only record of what a person sitting in the meeting saw, which
makes it the only place some faults are visible at all: a sentence that reads
correctly on the server and arrives as spaced-out characters on screen is a
fault of the same size as a wrong word, and nothing server-side reports it.

Two runs of the same recording can be compared line for line here. Two runs
of *different* recordings cannot, and the first thing this module does is say
which of the two it is holding, because the mistake is easy to make and
expensive: it turns noise into a finding.

Pure text. No audio, no model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from collections import Counter

from server.analysis.drift import compare, immediate_repeats
from server.pipeline.asr import normalise_for_pattern

#: ``10:44:01.360     13.1s  final       #1 Speaker_01 [vi] Các thông tin ...``
_LINE = re.compile(
    r"^(?P<clock>\d\d:\d\d:\d\d\.\d+)\s+"
    r"(?P<at>-?[\d.]+)s\s+"
    r"(?P<kind>\w+)\s+"
    r"(?P<payload>.*)$"
)

_FINAL = re.compile(
    r"^#(?P<sentence>\d+)\s+(?P<speaker>\S+)\s+\[(?P<lang>\w*)\]\s*(?P<text>.*)$"
)
_PARTIAL = re.compile(r"^\[(?P<lang>\w*)\]\s*(?P<text>.*)$")
_TRANSLATION = re.compile(r"^#(?P<sentence>\d+)\s*(?P<text>.*)$")

#: A translation the server gave up on. The client writes the reason in
#: place of the text, so a refused one is visible without the server log.
_REFUSED = re.compile(r"^\(từ chối:")

#: Han, Hiragana, Katakana. Japanese is written without spaces, so a space
#: between two of these is the renderer's, not the speaker's.
_CJK = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
# A lookahead, not a second character class: the gaps overlap, and consuming
# the character after the space would count half of them.
_CJK_SPACED = re.compile(f"[{_CJK}]\\s+(?=[{_CJK}])")
_CJK_ANY = re.compile(f"[{_CJK}]")

#: Past this share of the running text rewritten, the sentence is not a
#: revision of it any more - it is a different sentence. Read against the
#: 2026-09-11 run: below it the two read as the same thing said twice; above
#: it they are unrelated, and about a third of those are inventions.
FAR = 0.6


@dataclass
class Final:
    """One committed sentence, as the viewer saw it."""

    sentence: int
    at: float
    speaker: str
    lang: str
    text: str
    #: Every running text shown since the previous sentence was committed.
    partials: list = field(default_factory=list)
    translation: str = ""
    refused: bool = False

    @property
    def last_partial(self) -> str:
        return self.partials[-1][1] if self.partials else ""

    @property
    def partial_language(self) -> str:
        """What the running texts agreed this utterance was in.

        A second opinion on the same audio, taken several times while the
        sentence was still open, against the one the LID formed once. When
        the two disagree the sentence is four times as likely to have nothing
        to do with what was said - the language is forced on the decode, and
        forcing the wrong one does not fail loudly, it produces fluent text
        in a language nobody spoke.
        """
        languages = [lang for lang, text, _at in self.partials if lang and text]
        if not languages:
            return ""
        return Counter(languages).most_common(1)[0][0]

    @property
    def language_disagrees(self) -> bool:
        spoken = self.partial_language
        return bool(spoken) and bool(self.lang) and spoken != self.lang


@dataclass
class Run:
    """One meeting, as one version of the app rendered it."""

    path: Path
    finals: list = field(default_factory=list)
    partial_count: int = 0
    summary: str = ""

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def seconds(self) -> float:
        return self.finals[-1].at if self.finals else 0.0


def parse(path: Path) -> Run:
    """Read one debug log into sentences, each carrying its running texts."""
    run = Run(path=path)
    pending: list = []
    by_sentence: dict = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _LINE.match(line.rstrip())
        if match is None:
            continue
        kind = match["kind"]
        at = float(match["at"])
        payload = match["payload"].strip()

        if kind == "partial":
            found = _PARTIAL.match(payload)
            if found:
                run.partial_count += 1
                pending.append((found["lang"], found["text"].strip(), at))
        elif kind == "final":
            found = _FINAL.match(payload)
            if found:
                final = Final(sentence=int(found["sentence"]), at=at,
                              speaker=found["speaker"], lang=found["lang"],
                              text=found["text"].strip(), partials=pending)
                run.finals.append(final)
                by_sentence[final.sentence] = final
                pending = []
        elif kind == "translation":
            found = _TRANSLATION.match(payload)
            if found and int(found["sentence"]) in by_sentence:
                final = by_sentence[int(found["sentence"])]
                final.translation = found["text"].strip()
                final.refused = bool(_REFUSED.match(final.translation))
        elif kind == "end":
            run.summary = payload
    return run


# ---------------------------------------------------------------------------
# What a viewer would notice
# ---------------------------------------------------------------------------
def cjk_spacing(text: str) -> int:
    """Spaces inserted between Japanese characters.

    A renderer that joins word timestamps with a space is correct for
    Vietnamese and wrong for Japanese, and the result is unreadable on
    screen and worse in a translation prompt - the model is handed
    characters where it expects words.
    """
    return len(_CJK_SPACED.findall(text))


def is_cjk(text: str) -> bool:
    return bool(_CJK_ANY.search(text))


def measure(run: Run) -> dict:
    """The counts worth comparing between two runs of the same recording."""
    finals = run.finals
    cjk = [final for final in finals if is_cjk(final.text)]
    spaced = [final for final in cjk if cjk_spacing(final.text)]
    repeated = [final for final in finals if immediate_repeats(final.text)]
    drifts = [compare(final.text, final.last_partial, 0.0,
                      max(final.at * 1000.0, 1.0))
              for final in finals if final.last_partial]
    rewrites = sorted(drift.rewrite for drift in drifts)

    disagreed = [final for final in finals if final.language_disagrees]
    far_and_disagreed = [
        final for final in disagreed
        if final.last_partial
        and compare(final.text, final.last_partial, 0.0, 1.0).rewrite > FAR
    ]

    return {
        "run": run.name,
        "sentences": len(finals),
        "partials": run.partial_count,
        "characters": sum(len(final.text) for final in finals),
        "japanese_sentences": len(cjk),
        "japanese_spaced": len(spaced),
        "with_repeats": len(repeated),
        "repeat_total": sum(immediate_repeats(final.text) for final in finals),
        "refused_translations": sum(1 for final in finals if final.refused),
        "compared": len(drifts),
        "mean_rewrite": (sum(rewrites) / len(rewrites)) if rewrites else 0.0,
        "identical_to_partial": sum(1 for value in rewrites if value == 0.0),
        "far_from_partial": sum(1 for value in rewrites if value > FAR),
        "language_disagrees": len(disagreed),
        "far_and_language_disagrees": len(far_and_disagreed),
        "summary": run.summary,
    }


def same_meeting(left: Run, right: Run, window: float = 8.0,
                 sample: int = 12) -> dict:
    """Whether two runs are even comparable, and how well they line up.

    Comparing two different recordings sentence by sentence produces a page
    of differences that mean nothing at all, so this is checked before
    anything else is measured. The test is content, not timing: sentences
    near the same moment should share words if they are the same meeting.
    """
    hits, checked = 0, 0
    for final in left.finals[:sample]:
        words = {word for word in normalise_for_pattern(final.text).split()
                 if len(word) > 2}
        if not words:
            continue
        checked += 1
        near = [other for other in right.finals
                if abs(other.at - final.at) <= window]
        if any(words & set(normalise_for_pattern(other.text).split())
               for other in near):
            hits += 1
    share = hits / checked if checked else 0.0
    return {"checked": checked, "matched": hits, "share": share,
            "same": share >= 0.4}
