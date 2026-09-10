"""Replay the ASR guards over segments that were already decoded.

Whether a segment reaches the screen is a decision about three numbers -
``no_speech_prob``, ``avg_logprob``, ``compression_ratio`` - plus two word
lists. None of it needs a GPU. So once a run has recorded those numbers, the
question "what would a different rule have kept?" is arithmetic, and the
answer is exact rather than an estimate: the decoder is not consulted again
because its output has not changed.

Two rules:

``current``
    What ``Transcriber._refuse`` does today. ``no_speech_prob`` above the
    threshold refuses the segment on its own.

``whisper``
    What faster-whisper itself does::

        should_skip = result.no_speech_prob > no_speech_threshold
        if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
            # don't skip if the logprob is high enough, despite the no_speech_prob
            should_skip = False

    A confident decode survives an uncertain ``no_speech_prob``. Note what
    this implies: a segment whose ``avg_logprob`` is *not* high enough is
    refused by the low-confidence guard anyway, so under this rule
    ``no_speech_prob`` never refuses anything on its own.

Everything else - the order of the checks, the thresholds, the word lists -
comes from the live ``Transcriber``, so a change to the policy cannot drift
away from what is simulated here.
"""

from __future__ import annotations

from typing import Optional

from server.analysis.drift import edit_distance
from server.config import ASR_HALLUCINATIONS
from server.pipeline.asr import (
    Piece,
    Transcriber,
    normalise_for_match,
    normalise_for_pattern,
)

RULES = ("current", "whisper")


class _NoDecoder:
    """Stands in for the model. Nothing here decodes anything."""

    source = "no decoder - guards only"

    def decode(self, samples, lang_code, beam_size):    # pragma: no cover
        raise RuntimeError("the guard simulation never decodes")


def make_transcriber() -> Transcriber:
    """A Transcriber carrying the live thresholds and word lists, no model."""
    return Transcriber(decoder=_NoDecoder())


def piece_from(record: dict) -> Piece:
    return Piece(
        text=record["text"],
        avg_logprob=float(record["avg_logprob"]),
        no_speech_prob=float(record["no_speech_prob"]),
        compression_ratio=float(record["compression_ratio"]),
    )


def refuse(transcriber: Transcriber, piece: Piece,
           rule: str = "current") -> Optional[str]:
    """Why this segment should not be shown, under the named rule.

    ``current`` defers to the live ``Transcriber`` so the two can never
    disagree. ``whisper`` repeats the same checks in the same order with one
    clause changed, and ``test_guards_unit`` pins the two together on every
    case but the one that is meant to differ.
    """
    if rule == "current":
        return transcriber._refuse(piece)
    if rule != "whisper":
        raise ValueError(f"unknown guard rule {rule!r}")

    if not piece.text.strip():
        return "empty"
    if (piece.no_speech_prob > transcriber.no_speech_threshold
            and piece.avg_logprob <= transcriber.log_prob_threshold):
        return "no speech"
    if piece.avg_logprob < transcriber.log_prob_threshold:
        return "low confidence"
    if piece.compression_ratio > transcriber.max_compression_ratio:
        return "repetition"
    if normalise_for_match(piece.text) in transcriber.hallucinations:
        return "known hallucination"
    spaced = normalise_for_pattern(piece.text)
    for pattern in transcriber.hallucination_patterns:
        if pattern.fullmatch(spaced):
            return "known hallucination"
    return None


def decide(transcriber: Transcriber, pieces: list, rule: str) -> dict:
    """Run one rule over one sentence's segments, in the order they arrived."""
    kept, dropped = [], []
    for piece in pieces:
        reason = refuse(transcriber, piece, rule)
        if reason is None:
            kept.append(piece)
        else:
            dropped.append((piece, reason))
    return {
        "text": " ".join(piece.text.strip() for piece in kept).strip(),
        "kept": kept,
        "dropped": dropped,
    }


def simulate(run: dict, variant: str, rule: str,
             transcriber: Optional[Transcriber] = None) -> dict:
    """Every sentence of a run under one guard rule, keyed by index.

    Sentences whose segments were not recorded are skipped rather than
    guessed at - an older run has no scores to replay.
    """
    transcriber = transcriber or make_transcriber()
    out: dict = {}
    for case in run["cases"]:
        recorded = case["finals"].get(variant, {}).get("pieces")
        if not recorded:
            # Either the run predates the recording of scores, or the decoder
            # returned nothing at all. Neither can be replayed, and a sentence
            # the model never spoke a word for cannot be recovered by any
            # rule, so leaving it out keeps both sides of a comparison over
            # the same sentences.
            continue
        verdict = decide(transcriber,
                         [piece_from(record) for record in recorded], rule)
        out[case["index"]] = {
            "text": verdict["text"],
            "reasons": [reason for _piece, reason in verdict["dropped"]],
            "kept": len(verdict["kept"]),
        }
    return out


#: A line this close to one already on the block list is the same invention
#: with words changed. Measured against the run that motivated it: "Cảm ơn
#: các bạn." sits at 0.35 from "Cảm ơn các bạn đã theo dõi.", and the nearest
#: real sentence in that meeting was past 0.7.
NEAR_MISS_DISTANCE = 0.5

#: Shared opening words that make two lines the same shape. Two is enough for
#: "Hẹn gặp lại ..." and short enough to stay cheap; it is a candidate list
#: for a person to read, not a verdict.
NEAR_MISS_PREFIX_WORDS = 2


def near_miss(text: str, phrases=ASR_HALLUCINATIONS) -> Optional[dict]:
    """The blocked line this one most resembles, if it resembles one.

    The block list matches whole segments, so an invention with a couple of
    words changed walks straight through it - the list itself says so, in the
    comment above ``ASR_HALLUCINATION_PATTERNS``. This does not block
    anything. It says which lines are worth a person's attention, and against
    what, so the list can be grown from evidence rather than from guesses.
    """
    spoken = normalise_for_pattern(text)
    if not spoken:
        return None
    words = spoken.split()
    # The raw phrases, not ``Transcriber.hallucinations`` - that set has had
    # its spacing stripped for exact matching, and the shape of a line is in
    # its words.
    best = None
    for phrase in phrases:
        listed = normalise_for_pattern(phrase)
        if not listed:
            continue
        longest = max(len(spoken), len(listed))
        score = edit_distance(spoken, listed) / longest
        listed_words = listed.split()
        shared = min(len(words), len(listed_words), NEAR_MISS_PREFIX_WORDS)
        prefix = (shared >= NEAR_MISS_PREFIX_WORDS
                  and words[:shared] == listed_words[:shared])
        if score > NEAR_MISS_DISTANCE and not prefix:
            continue
        if best is None or score < best["distance"]:
            best = {"listed": phrase, "distance": score, "shared_prefix": prefix}
    return best


def has_scores(run: dict, variant: str) -> bool:
    """Whether this run recorded enough to replay the guards at all."""
    return any(case["finals"].get(variant, {}).get("pieces")
               for case in run["cases"])
