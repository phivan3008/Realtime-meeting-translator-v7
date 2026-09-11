"""The meeting's own words, as a prompt for the ASR.

``server/data/vocabulary.txt`` holds terms this meeting uses that Whisper
gets wrong, and this turns that file into an ``initial_prompt``.  The prompt
does not force anything; it tilts the model where it is already undecided
between two readings of one sound - which is the case being fixed, because
the running text hears "solution" on the same audio the committed sentence
turns into "sau lưu sinh".

Two things here are not detail.

**An empty list means no prompt**, not an empty one.  ``""`` is itself a
prompt as far as Whisper is concerned, and passing it is not the same as
passing nothing.

**The list is capped.**  A non-empty prompt makes Whisper fill near-silence
rather than leave it, which is the failure this project has spent the most
time on, so the cost of the list grows with its length while its benefit does
not - Whisper only reads the start of it.  The cap cuts on a term boundary,
because half a term is a string nobody said and the prompt is meant to
contain words the meeting contains.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from server.config import ASR_PROMPT_MAX_CHARS, VOCABULARY_PATH

log = logging.getLogger(__name__)

#: A comment runs to the end of the line, but only where the ``#`` opens one:
#: at the start, or after a space. A term may contain one - "C#" is a name a
#: meeting can say - and an earlier version of this elsewhere sent a term to
#: Whisper with its own annotation still attached.
_COMMENT = re.compile(r"(?:^|\s)#.*$")


def read_terms(path: Optional[Path] = None) -> list:
    """The terms in the vocabulary file, in order, without repeats.

    A missing file is not an error. The meeting still runs; it just runs
    without the tilt.
    """
    path = Path(path) if path is not None else Path(VOCABULARY_PATH)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.info("No vocabulary file at %s (%s); Whisper gets no prompt",
                 path, exc.__class__.__name__)
        return []

    terms: list = []
    for line in raw.splitlines():
        term = _COMMENT.sub("", line).strip()
        if term and term not in terms:
            terms.append(term)
    return terms


def build_prompt(terms: list, limit: int = ASR_PROMPT_MAX_CHARS) -> str:
    """The terms as one comma-separated line, cut on a term boundary."""
    kept: list = []
    length = 0
    for term in terms:
        addition = len(term) + (2 if kept else 0)
        if length + addition > limit:
            break
        kept.append(term)
        length += addition
    return ", ".join(kept)


def load_prompt(path: Optional[Path] = None,
                limit: int = ASR_PROMPT_MAX_CHARS) -> Optional[str]:
    """The prompt to hand Whisper, or None to hand it nothing at all."""
    terms = read_terms(path)
    if not terms:
        return None
    prompt = build_prompt(terms, limit)
    if not prompt:
        return None
    if len(terms) > prompt.count(",") + 1:
        log.warning(
            "Vocabulary trimmed to %d of %d terms by ASR_PROMPT_MAX_CHARS=%d; "
            "the rest never reach Whisper, which reads only the start of a "
            "prompt anyway", prompt.count(",") + 1, len(terms), limit)
    log.info("Whisper prompt: %d terms, %d characters",
             prompt.count(",") + 1, len(prompt))
    return prompt
