"""Word lists loaded from ``server/data``, so they can be edited without code.

Three files, all plain text, one entry per line, ``#`` starts a comment:

``hallucinations.txt``
    Whole sentences Whisper invents. Matched in full.

``hallucination_patterns.txt``
    The same, as regular expressions, for inventions with a hole in them -
    usually a channel name.

``keep.txt``
    Sentences that must never be treated as invented. Beats both lists above.

``server/data/README.md`` explains what belongs in each and how to test an
entry before adding it.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get(
    "MEETING_DATA_DIR", Path(__file__).resolve().parent / "data"))

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
