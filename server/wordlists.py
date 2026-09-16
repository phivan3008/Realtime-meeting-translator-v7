"""Word lists loaded from ``server/data``, so they can be edited without code.

Three files, all plain text, one entry per line, ``#`` starts a comment:

``hallucinations.txt``
    Whole sentences Whisper invents. Matched in full.

``hallucination_patterns.txt``
    The same, as regular expressions, for inventions with a hole in them -
    usually a channel name.

``keep.txt``
    Sentences that must never be treated as invented. Beats both lists above.

``vocabulary.txt``
    Terms Whisper keeps mishearing, turned into its ``initial_prompt``.

``server/data/README.md`` explains what belongs in each and how to test an
entry before adding it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from server.config import ASR_PROMPT_MAX_CHARS, MEETING_DATA_DIR

log = logging.getLogger(__name__)

DATA_DIR = Path(MEETING_DATA_DIR)

#: Punctuation and spacing. Diacritics are letters and stay: folding them
#: would make "tắt" and "tab" the same word.
_UNSPOKEN = re.compile(
    r"[\s.,!?;:\-‐-―、。・！？，．"
    r"\"'‘’“”()\[\]]+"
)
#: Punctuation only, spacing kept, for the pattern rules to read as sentences.
_PUNCTUATION = re.compile(
    r"[.,!?;:\-‐-―、。・！？，．"
    r"\"'‘’“”()\[\]]+"
)


def normalise_exact(text: str) -> str:
    """Reduce a line to what was said, for whole-sentence matching."""
    return _UNSPOKEN.sub("", text).casefold()


def normalise_spaced(text: str) -> str:
    """The same, keeping word boundaries, for pattern matching."""
    return " ".join(_PUNCTUATION.sub(" ", text).casefold().split())


def read_lines(name: str) -> list[str]:
    """Entries from one data file. Missing file means an empty list."""
    path = DATA_DIR / name
    if not path.exists():
        log.warning("No word list at %s; treating it as empty", path)
        return []
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def read_patterns(name: str) -> tuple[re.Pattern, ...]:
    """Compiled patterns from one data file, skipping any that will not."""
    compiled = []
    for line in read_lines(name):
        try:
            compiled.append(re.compile(line, re.IGNORECASE))
        except re.error as exc:
            log.error("Ignoring bad pattern in %s: %s (%s)", name, line, exc)
    return tuple(compiled)


class Hallucinations:
    """The three lists, and the question they answer together."""

    def __init__(self, exact: list[str] | None = None,
                 patterns: tuple[re.Pattern, ...] | None = None,
                 keep: list[str] | None = None) -> None:
        source = read_lines("hallucinations.txt") if exact is None else exact
        kept = read_lines("keep.txt") if keep is None else keep
        self.exact = frozenset(normalise_exact(line) for line in source)
        self.keep = frozenset(normalise_exact(line) for line in kept)
        self.patterns = (read_patterns("hallucination_patterns.txt")
                         if patterns is None else patterns)

    def is_invented(self, text: str) -> bool:
        """Would this line be shown to somebody who never heard it said?"""
        exact = normalise_exact(text)
        if exact in self.keep:
            return False
        if exact in self.exact:
            return True
        spaced = normalise_spaced(text)
        return any(pattern.fullmatch(spaced) for pattern in self.patterns)

    def __len__(self) -> int:
        return len(self.exact) + len(self.patterns)


#: A trailing comment, but only where the ``#`` opens one: after a space. A
#: term may contain one - "C#" is a name a meeting can say - and a term that
#: reached Whisper with its own annotation attached is how this was found.
_TRAILING_COMMENT = re.compile(r"\s+#.*$")


def vocabulary_terms() -> list[str]:
    """The terms in ``vocabulary.txt``, in order, without repeats."""
    terms: list[str] = []
    for line in read_lines("vocabulary.txt"):
        term = _TRAILING_COMMENT.sub("", line).strip()
        if term and term not in terms:
            terms.append(term)
    return terms


def vocabulary_prompt(limit: int = ASR_PROMPT_MAX_CHARS) -> str:
    """Whisper's ``initial_prompt``, built from ``vocabulary.txt``.

    Words a meeting keeps using that Whisper keeps mishearing: on a real run
    the running text heard ``Slack`` and the committed sentence turned it into
    ``quạt nắp``. The prompt does not force anything - it tilts the model when
    it is undecided.

    Capped on a term boundary, because Whisper reads only the start of it and
    a stuffed prompt makes the model produce those very words over silence.
    An empty file means no prompt at all rather than an empty string, which
    Whisper treats as a prompt of its own - the caller turns "" into None.
    """
    terms = vocabulary_terms()
    kept: list[str] = []
    length = 0
    for term in terms:
        addition = len(term) + (2 if kept else 0)
        if length + addition > limit:
            break
        kept.append(term)
        length += addition
    if len(kept) < len(terms):
        log.warning("Vocabulary trimmed to %d of %d terms by "
                    "ASR_PROMPT_MAX_CHARS=%d; the rest never reach Whisper",
                    len(kept), len(terms), limit)
    return ", ".join(kept)
