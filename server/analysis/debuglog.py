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
import time
from dataclasses import dataclass, field
from pathlib import Path

from collections import Counter

from server.analysis.drift import compare, immediate_repeats
from server.analysis.guards import near_miss
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

#: Running texts in another language that make a dropped turn.
LOST_TURN_PARTIALS = 3


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
        languages = [written_in(text, lang)
                     for lang, text, _at in self.partials if text]
        languages = [lang for lang in languages if lang]
        if not languages:
            return ""
        return Counter(languages).most_common(1)[0][0]

    @property
    def lost_turn(self) -> str:
        """A language the running text spoke at length and the sentence lacks.

        Three running texts of some length in a language the sentence is not
        in is a second turn the sentence dropped: "Đi kiểm chứng tiếp" shown
        for three seconds, then a sentence holding only the Japanese after
        it. Short backchannels ("ừ ừ" for うん) do not count.
        """
        counts: dict = {}
        for tag, text, _at in self.partials:
            lang = written_in(text, tag)
            if lang and lang != self.lang and len(text.replace(" ", "")) >= 6:
                counts[lang] = counts.get(lang, 0) + 1
        for lang, count in counts.items():
            if count >= LOST_TURN_PARTIALS:
                return lang
        return ""

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


class LogWriter:
    """Write server messages as the client's debug log would record them.

    For replaying a meeting on the pod, which does not carry the client
    package. Only the lines :func:`parse` reads are written, in the client's
    exact format, so a replay can be compared with a real client log.
    """

    def __init__(self, path: Path, clock) -> None:
        self.path = Path(path)
        self.clock = clock
        self.started = clock()
        self.sentences = 0
        self.refused = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="\n")
        self.note("start", "session=replay")

    def apply(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "partial":
            text = message.get("transcript", "")
            if text:
                self.note("partial", f"[{message.get('lang_code', '')}] {text}")
        elif kind == "final":
            self.sentences += 1
            self.note("final", f"#{message.get('sentence_id', 0)} "
                               f"{message.get('speaker_id') or '?'} "
                               f"[{message.get('lang_code', '')}] "
                               f"{message.get('transcript', '')}")
        elif kind == "translation":
            translation = message.get("translation", "")
            if not translation.strip():
                self.refused += 1
            self.note("translation", f"#{message.get('sentence_id')} " + (
                translation if translation.strip()
                else f"(từ chối: {message.get('reason') or 'không rõ'})"))
        elif kind == "utterance" and not message.get("kept", True):
            self.note("dropped", f"utterance {message.get('index')} — "
                                 f"{message.get('label') or 'không rõ'}")
        elif kind == "error":
            self.note("error", message.get("message", ""))

    def note(self, kind: str, text: str) -> None:
        now = self.clock()
        clock = time.strftime("%H:%M:%S", time.localtime(now))
        self._file.write(f"{clock}.{int(now % 1 * 1000):03d}  "
                         f"{now - self.started:7.1f}s  {kind:<11} {text}\n")
        self._file.flush()

    def close(self) -> None:
        if self._file.closed:
            return
        self.note("end", f"{self.sentences} câu, "
                         f"{self.refused} không dịch được")
        self._file.close()


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


#: A letter only Vietnamese writes: Latin with a diacritic, including đ.
_VIETNAMESE = re.compile("[À-ɏḀ-ỿ]")


def written_in(text: str, tag: str = "") -> str:
    """The language a line is written in, judged from its script.

    The tag a running text carries is the language it was decoded in, and
    Whisper forced into Vietnamese still writes Japanese audio as Japanese
    now and then: "マップの制作として" arrived tagged [vi]. The script says
    what was on screen. Lines in neither script keep their tag.
    """
    if is_cjk(text):
        return "ja"
    if _VIETNAMESE.search(text):
        return "vi"
    return tag


def is_mixed(text: str) -> bool:
    """A sentence holding both Japanese and Vietnamese.

    Not a thing people say in these meetings - English terms inside Japanese
    are plain ASCII and do not count. On a real run it was two decodes in two
    languages merged into one sentence: "これからこのタスは có thểので ...".
    """
    return is_cjk(text) and bool(_VIETNAMESE.search(text))


def survival(before: str, after: str) -> float:
    """Share of the running text on screen that the next update kept.

    What a reader notices is not accuracy but stability: an update that
    rewrites the line somebody is half way through reading. Read as the
    common prefix, because that is the part the eye has already passed.
    """
    if not before:
        return 1.0
    kept = 0
    for left, right in zip(before, after):
        if left != right:
            break
        kept += 1
    return kept / len(before)


def stability(run: Run) -> tuple:
    """Mean survival over every update, and the share that wiped over half.

    Measured on the debug logs of one meeting: the sliding-window running
    text kept 33% of what was on screen per update and wiped more than half
    of it 67% of the time; the local-agreement rewrite kept 67% and wiped
    31%.
    """
    shares = []
    for final in run.finals:
        texts = [text for _lang, text, _at in final.partials]
        shares += [survival(before, after)
                   for before, after in zip(texts, texts[1:])]
    if not shares:
        return 0.0, 0.0
    wiped = sum(1 for share in shares if share < 0.5)
    return sum(shares) / len(shares), wiped / len(shares)


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

    # The cost side of the vocabulary prompt, and of anything else that
    # makes Whisper readier to write. A committed sentence that reads like a
    # line already on the block list, with words changed, is what a prompt
    # produces over near-silence.
    sign_offs = [final for final in finals if near_miss(final.text)]

    disagreed = [final for final in finals if final.language_disagrees]
    far_and_disagreed = [
        final for final in disagreed
        if final.last_partial
        and compare(final.text, final.last_partial, 0.0, 1.0).rewrite > FAR
    ]

    kept_share, wiped_share = stability(run)
    return {
        "run": run.name,
        "update_survival": kept_share,
        "updates_wiping_half": wiped_share,
        "sentences": len(finals),
        "partials": run.partial_count,
        "characters": sum(len(final.text) for final in finals),
        "japanese_sentences": len(cjk),
        "japanese_spaced": len(spaced),
        "with_repeats": len(repeated),
        "repeat_total": sum(immediate_repeats(final.text) for final in finals),
        "refused_translations": sum(1 for final in finals if final.refused),
        "near_block_list": len(sign_offs),
        "compared": len(drifts),
        "mean_rewrite": (sum(rewrites) / len(rewrites)) if rewrites else 0.0,
        "identical_to_partial": sum(1 for value in rewrites if value == 0.0),
        "far_from_partial": sum(1 for value in rewrites if value > FAR),
        "language_disagrees": len(disagreed),
        "mixed_language": sum(1 for final in finals if is_mixed(final.text)),
        "lost_turns": sum(1 for final in finals if final.lost_turn),
        "far_and_language_disagrees": len(far_and_disagreed),
        "summary": run.summary,
    }


def rare_words(run: Run, shortest: int = 4, most: int = 1) -> dict:
    """Words a run said exactly once, and when it said them.

    Common words are useless for lining two runs up - "this" happens
    everywhere - and so is a word that appears thirty times. What locates a
    moment is a term used once, and *once* is the point: a word said twice
    offers two timestamps and votes for two offsets, which lets one word
    support several answers at the same time. Restricted to one occurrence, a
    shared word casts exactly one vote and the histogram means what it looks
    like it means.

    Japanese has no spaces, so there the unit is a four-character run, which
    survives being resegmented.
    """
    when: dict = {}
    for final in run.finals:
        spoken = normalise_for_pattern(final.text)
        tokens = [word for word in spoken.split() if len(word) >= shortest]
        dense = spoken.replace(" ", "")
        if is_cjk(dense):
            tokens += [dense[index:index + shortest]
                       for index in range(max(len(dense) - shortest + 1, 0))]
        for token in set(tokens):
            when.setdefault(token, []).append(final.at)
    return {word: times for word, times in when.items() if len(times) <= most}


def align(left: Run, right: Run, tolerance: float = 4.0,
          share: float = 0.30, votes: int = 15) -> dict:
    """Line two runs up in time, and say how well they line up.

    Two runs of the same recording say the same things in the same order,
    offset by however long after the start the capture began. Two runs of
    different meetings by the same team share most of their vocabulary - the
    jargon, the names, the function words - and line up at no offset at all.
    So the test is agreement on *when*, not on *what*: each rare word both
    runs said votes for the offset that would put its two timestamps
    together, and a real pairing puts most of the votes in one bin.

    An earlier version matched sentences inside a fixed time window, which
    fell apart the first time a change moved the sentence boundaries - the
    language split dropped it to three matches of eleven and it called two
    runs of one meeting different recordings. The version after that compared
    vocabularies as sets, and called two genuinely different meetings the
    same because both were this team talking about this project.
    """
    mine, theirs = rare_words(left), rare_words(right)
    shared = set(mine) & set(theirs)
    if not shared:
        return {"offset": 0.0, "votes": 0, "shared": 0, "share": 0.0,
                "same": False}

    ballot: dict = {}
    for word in shared:
        offset = theirs[word][0] - mine[word][0]
        ballot.setdefault(round(offset / tolerance), set()).add(word)

    best = max(ballot, key=lambda box: len(ballot[box]))
    agreed = len(ballot[best])
    found = agreed / len(shared)
    return {
        "offset": best * tolerance,
        "votes": agreed,
        "shared": len(shared),
        "share": found,
        "same": found >= share and agreed >= votes,
    }


def same_meeting(left: Run, right: Run, **options) -> dict:
    """Whether two runs are even comparable.

    Comparing two different recordings sentence by sentence produces a page
    of differences that mean nothing at all, so this is checked before
    anything else is measured.
    """
    return align(left, right, **options)
