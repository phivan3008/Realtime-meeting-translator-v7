"""Compare a committed sentence against the last running text that preceded it.

The running text is decoded greedily on the last four seconds of an open
utterance; the sentence is decoded again from scratch, with a beam search, on
the whole utterance.  They are two different draws, not a draft and its
revision, and on real meeting audio the sentence is sometimes the worse of
the two - a code-switched word replaced by a fluent Vietnamese phrase, a
clause invented in the trailing silence, a word repeated.

Nothing here can say which text is *correct*; only a person reading them can.
What it can do is count the four ways the sentence is known to drift away from
the running text, so a ten-minute recording produces four numbers instead of
four hundred lines to read.

Pure text: no audio, no models, no torch.  It runs on the Dev PC.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from server.pipeline.asr import normalise_for_pattern
from server.pipeline.textdiff import edit_distance, substring_distance

#: A sentence that rewrote more than this share of the running text it was
#: supposed to be confirming. A quarter is well past punctuation and casing -
#: those survive `normalise_for_pattern` anyway.
REWRITE_FLAG = 0.25

#: The running text stops while somebody is still speaking; the sentence also
#: carries the VAD's trailing hangover. Audio added beyond this is real speech
#: the running text never saw, and text explained by it is not an invention.
TAIL_AUDIO_MS = 700.0

#: Below this, an added tail is a word or two of ordinary lag, not a clause.
TAIL_MIN_CHARS = 8

#: Text added to the tail at more than this multiple of the speaking rate of
#: the sentence itself was not spoken at that speed, because nobody was.
TAIL_RATE_FACTOR = 2.0

#: Unaccented ASCII runs shorter than this are initials, and Vietnamese has
#: real words of one and two letters ("do", "la"). Three is where the odds
#: tip towards a borrowed term. The report prints the words it found, so a
#: false positive is visible rather than silently counted.
LATIN_MIN_CHARS = 3

#: One run of letters or digits, in any script. Vietnamese is Latin script
#: too, so a word is only a candidate if the *whole* run is unaccented ASCII -
#: matching ASCII inside a word would turn "thì" into the "word" "th".
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def latin_words(text: str) -> tuple[str, ...]:
    """The Latin-script words in a Vietnamese or Japanese line.

    These are where the beam search does its damage: an English term in a
    forced-Vietnamese decode is a low-prior token sequence, and a beam that
    scores whole hypotheses will happily trade it for a fluent Vietnamese
    phrase that sounds nearly the same.
    """
    return tuple(
        word.casefold() for word in _WORD.findall(text)
        if len(word) >= LATIN_MIN_CHARS
        and word.isascii()
        and any(character.isalpha() for character in word)
    )


def immediate_repeats(text: str) -> int:
    """Count adjacent duplicates - the small end of Whisper's looping.

    Vietnamese is spaced, so words are the unit. Japanese is not, so there
    the unit is a two-character bigram; single characters repeat legitimately
    far too often to count.
    """
    spaced = normalise_for_pattern(text)
    tokens = spaced.split()
    if len(tokens) >= 2:
        return sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    dense = spaced.replace(" ", "")
    return sum(1 for i in range(len(dense) - 3)
               if dense[i:i + 2] == dense[i + 2:i + 4])


@dataclass(frozen=True)
class Drift:
    """How one committed sentence differs from the running text before it."""

    partial: str
    final: str
    #: Share of the running text the sentence did not keep. 0.0 means the
    #: sentence contains it verbatim somewhere.
    rewrite: float
    #: Where the running text matched inside the sentence.
    matched_start: int
    matched_end: int
    #: What the sentence added after that match, and how much audio there was
    #: to justify it.
    tail_text: str
    extra_audio_ms: float
    tail_rate: float
    body_rate: float
    lost_latin: tuple[str, ...]
    gained_latin: tuple[str, ...]
    new_repeats: int
    flags: tuple[str, ...]

    @property
    def tail_chars(self) -> int:
        return len(self.tail_text.strip())


def compare(final: str, partial: str, extra_audio_ms: float,
            final_audio_ms: float) -> Drift:
    """Measure one sentence against the running text it replaced.

    ``extra_audio_ms`` is how much more audio the sentence covers than the
    last running text did - the speech that arrived after it, plus the VAD's
    trailing hangover. It is what decides whether an added tail is late news
    or invention.
    """
    normal_final = normalise_for_pattern(final)
    normal_partial = normalise_for_pattern(partial)
    extra_audio_ms = max(extra_audio_ms, 0.0)

    distance, start, end = substring_distance(normal_partial, normal_final)
    rewrite = distance / len(normal_partial) if normal_partial else 0.0
    tail_text = normal_final[end:]

    seconds = max(final_audio_ms, 1.0) / 1000.0
    body_rate = len(normal_final) / seconds
    tail_seconds = extra_audio_ms / 1000.0
    tail_rate = len(tail_text.strip()) / tail_seconds if tail_seconds else 0.0

    in_final = set(latin_words(final))
    in_partial = set(latin_words(partial))
    lost = tuple(sorted(in_partial - in_final))
    gained = tuple(sorted(in_final - in_partial))
    new_repeats = max(immediate_repeats(final) - immediate_repeats(partial), 0)

    flags = []
    if not normal_final and normal_partial:
        flags.append("final_empty")
    if rewrite > REWRITE_FLAG:
        flags.append("rewrite")
    if (extra_audio_ms <= TAIL_AUDIO_MS
            and len(tail_text.strip()) >= TAIL_MIN_CHARS
            and tail_rate > TAIL_RATE_FACTOR * body_rate):
        flags.append("tail_invention")
    if lost:
        flags.append("latin_lost")
    if new_repeats:
        flags.append("repetition")

    return Drift(
        partial=partial,
        final=final,
        rewrite=rewrite,
        matched_start=start,
        matched_end=end,
        tail_text=tail_text.strip(),
        extra_audio_ms=extra_audio_ms,
        tail_rate=tail_rate,
        body_rate=body_rate,
        lost_latin=lost,
        gained_latin=gained,
        new_repeats=new_repeats,
        flags=tuple(flags),
    )
