"""ASR, step 7 of the server pipeline.

Whisper is not a streaming-native recognizer. This module turns repeated
Whisper decoding into a stabilized stream with three result states:

partial
    An unstable hypothesis. The client replaces the previous partial text.

committed
    Newly stabilized text. The client appends it exactly once. Committed text
    is immutable and is never replaced by a later decode.

utterance_final
    Silero VAD has detected the end of an utterance. Only the remaining
    unstable audio tail is decoded again. Already committed text is preserved.

The final transcript is therefore:

    append-only committed text + reconciled final unstable tail

It is not a new full-utterance decode that overwrites all previous results.

Important caller contract
-------------------------
Partial calls must provide:

    utterance_id
    pcm containing the current rolling ASR window
    window_start_seconds relative to the utterance start

The final call must provide:

    utterance_id
    full utterance PCM
    utterance_start_seconds, normally 0.0
    speech_end_seconds from Silero VAD, excluding endpoint silence

All PCM must be signed PCM16 little-endian, mono, at SAMPLE_RATE.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable, Literal, Optional, Protocol

import numpy as np

from server.config import (
    ASR_BEAM_SIZE_PARTIAL,
    ASR_CACHE_DIR,
    ASR_COMPUTE_TYPE,
    ASR_CONDITION_ON_PREVIOUS,
    ASR_DEVICE,
    ASR_HALLUCINATIONS,
    ASR_HALLUCINATION_PATTERNS,
    ASR_LOG_PROB_THRESHOLD,
    ASR_MAX_COMPRESSION_RATIO,
    ASR_MODEL,
    ASR_NO_SPEECH_THRESHOLD,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming policy defaults
# ---------------------------------------------------------------------------

# A word must agree across this many consecutive hypotheses before it can be
# treated as stable.
DEFAULT_MIN_AGREEMENT = 2

# Do not commit words too close to the newest received audio. Whisper is most
# likely to revise words in this region.
DEFAULT_COMMIT_MARGIN_SECONDS = 1.0

# When finalizing, decode a little committed audio before the unstable tail.
# The overlap gives Whisper enough left context without decoding the complete
# utterance again.
DEFAULT_FINAL_OVERLAP_SECONDS = 1.2

# Keep only a short amount of audio after Silero's last speech frame. Silence
# used to detect an endpoint must not all be sent to Whisper.
DEFAULT_FINAL_POST_ROLL_SECONDS = 0.20

# Tolerance when matching the same word between overlapping rolling windows.
DEFAULT_WORD_TIME_TOLERANCE_SECONDS = 0.45

# Number of recent hypotheses retained for consensus and diagnostics.
DEFAULT_HISTORY_SIZE = 5


ResultKind = Literal["partial", "committed", "utterance_final"]


class AsrError(RuntimeError):
    """Raised when the ASR model cannot be loaded or used."""


# ---------------------------------------------------------------------------
# Model output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Word:
    """One timestamped word returned by faster-whisper."""

    text: str
    start: float
    end: float
    probability: float

    @property
    def normalized(self) -> str:
        return normalise_word(self.text)


@dataclass(frozen=True)
class Piece:
    """One segment returned by the decoder before policy filtering."""

    text: str
    start: float
    end: float
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class Hypothesis:
    """A decoded rolling window expressed on the utterance timeline."""

    words: tuple[Word, ...]
    text: str
    window_start: float
    window_end: float
    lang_code: str
    pieces: tuple[Piece, ...] = ()
    dropped: tuple[tuple[Piece, str], ...] = ()


@dataclass(frozen=True)
class Transcript:
    """One event emitted by the stabilized streaming recognizer.

    ``text`` has different semantics by event type:

    partial
        Complete current unstable text. Replace the previous partial.

    committed
        Newly committed delta. Append exactly once.

    utterance_final
        Complete final utterance text.
    """

    text: str
    lang_code: str
    kind: ResultKind

    # Complete immutable transcript accumulated so far.
    committed_text: str = ""

    # Complete current unstable suffix.
    partial_text: str = ""

    kept: tuple[Piece, ...] = ()
    dropped: tuple[tuple[Piece, str], ...] = ()

    @property
    def is_final(self) -> bool:
        return self.kind == "utterance_final"

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    def summary(self) -> str:
        return (
            f"{self.kind} [{self.lang_code or 'auto'}] "
            f"{len(self.kept)} kept, {len(self.dropped)} dropped: "
            f"{self.text[:60]!r}"
        )


@dataclass
class AsrStats:
    partials: int = 0
    committed_events: int = 0
    finals: int = 0
    empty: int = 0
    dropped_pieces: int = 0
    dropped_reasons: dict[str, int] = field(default_factory=dict)
    decode_seconds: float = 0.0
    audio_seconds: float = 0.0

    @property
    def realtime_factor(self) -> float:
        if not self.audio_seconds:
            return 0.0
        return self.decode_seconds / self.audio_seconds

    def record(self, transcript: Transcript) -> None:
        if transcript.kind == "partial":
            self.partials += 1
        elif transcript.kind == "committed":
            self.committed_events += 1
        else:
            self.finals += 1

        if not transcript.has_text:
            self.empty += 1

        for _piece, reason in transcript.dropped:
            self.dropped_pieces += 1
            self.dropped_reasons[reason] = (
                self.dropped_reasons.get(reason, 0) + 1
            )


@dataclass
class UtteranceState:
    """Mutable stabilization state for one active utterance."""

    utterance_id: str
    lang_code: str = ""

    committed_words: list[Word] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)

    # Absolute end time, relative to utterance start, of immutable text.
    committed_end: float = 0.0

    revision: int = 0

    @property
    def committed_text(self) -> str:
        return render_words(self.committed_words)

    @property
    def last_hypothesis(self) -> Optional[Hypothesis]:
        if not self.hypotheses:
            return None
        return self.hypotheses[-1]


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_UNSPOKEN = re.compile(
    r"[\s.,!?;:\-\u2010-\u2015\u3001\u3002\u30fb\uff01\uff1f\uff0c\uff0e"
    r"\"'\u2018\u2019\u201c\u201d()\[\]]+"
)

_PUNCTUATION = re.compile(
    r"[.,!?;:\-\u2010-\u2015\u3001\u3002\u30fb\uff01\uff1f\uff0c\uff0e"
    r"\"'\u2018\u2019\u201c\u201d()\[\]]+"
)


def normalise_for_pattern(text: str) -> str:
    """Strip punctuation and collapse spaces for hallucination patterns."""
    return " ".join(_PUNCTUATION.sub(" ", text).casefold().split())


def normalise_for_match(text: str) -> str:
    """Strip punctuation and spacing for exact known-hallucination matching."""
    return _UNSPOKEN.sub("", text).casefold()


def normalise_word(text: str) -> str:
    """Normalize one word while preserving Vietnamese diacritics."""
    return normalise_for_pattern(text)


def render_words(words: Iterable[Word]) -> str:
    """Render timestamped words without depending on tokenizer spacing."""

    parts = [word.text.strip() for word in words if word.text.strip()]
    text = " ".join(parts)

    # Remove spaces before common punctuation.
    text = re.sub(r"\s+([.,!?;:%)\]\}])", r"\1", text)

    # Remove spaces after opening punctuation.
    text = re.sub(r"([(\[\{])\s+", r"\1", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Decoder protocol
# ---------------------------------------------------------------------------

class Decoder(Protocol):
    """What :class:`StreamingTranscriber` requires from a decoder."""

    def decode(
        self,
        samples: np.ndarray,
        lang_code: str,
        beam_size: int,
    ) -> tuple[list[Piece], str]:
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Streaming policy
# ---------------------------------------------------------------------------

class StreamingTranscriber:
    """Stabilize repeated Whisper hypotheses into an append-only transcript."""

    def __init__(
        self,
        decoder: Optional[Decoder] = None,
        no_speech_threshold: float = ASR_NO_SPEECH_THRESHOLD,
        log_prob_threshold: float = ASR_LOG_PROB_THRESHOLD,
        max_compression_ratio: float = ASR_MAX_COMPRESSION_RATIO,
        hallucinations: Iterable[str] = ASR_HALLUCINATIONS,
        hallucination_patterns: Iterable[str] = ASR_HALLUCINATION_PATTERNS,
        min_agreement: int = DEFAULT_MIN_AGREEMENT,
        commit_margin_seconds: float = DEFAULT_COMMIT_MARGIN_SECONDS,
        final_overlap_seconds: float = DEFAULT_FINAL_OVERLAP_SECONDS,
        final_post_roll_seconds: float = DEFAULT_FINAL_POST_ROLL_SECONDS,
        word_time_tolerance_seconds: float = (
            DEFAULT_WORD_TIME_TOLERANCE_SECONDS
        ),
        history_size: int = DEFAULT_HISTORY_SIZE,
    ) -> None:
        if min_agreement < 2:
            raise ValueError("min_agreement must be at least 2")

        self.decoder = decoder if decoder is not None else WhisperDecoder()

        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self.max_compression_ratio = max_compression_ratio

        self.min_agreement = min_agreement
        self.commit_margin_seconds = commit_margin_seconds
        self.final_overlap_seconds = final_overlap_seconds
        self.final_post_roll_seconds = final_post_roll_seconds
        self.word_time_tolerance_seconds = word_time_tolerance_seconds
        self.history_size = history_size

        self.hallucinations = frozenset(
            normalise_for_match(phrase)
            for phrase in hallucinations
        )

        self.hallucination_patterns = tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in hallucination_patterns
        )

        self._states: dict[str, UtteranceState] = {}
        self.stats = AsrStats()

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------

    def process_partial(
        self,
        pcm: bytes,
        *,
        utterance_id: str,
        window_start_seconds: float,
        lang_code: str = "",
    ) -> tuple[Transcript, ...]:
        """Process one rolling-window partial.

        The returned tuple can contain:

        1. A ``committed`` event if new stable words were found.
        2. A ``partial`` event containing the current unstable suffix.

        ``window_start_seconds`` is relative to the start of the utterance.
        """

        self._validate_pcm(pcm)

        state = self._states.setdefault(
            utterance_id,
            UtteranceState(
                utterance_id=utterance_id,
                lang_code=lang_code,
            ),
        )

        if lang_code:
            state.lang_code = lang_code

        state.revision += 1

        if not pcm:
            partial = self._make_partial_event(state, None)
            self.stats.record(partial)
            return (partial,)

        samples = self._pcm_to_samples(pcm)

        hypothesis = self._decode_hypothesis(
            samples=samples,
            requested_language=state.lang_code,
            beam_size=ASR_BEAM_SIZE_PARTIAL,
            window_start_seconds=window_start_seconds,
        )

        if not state.lang_code:
            state.lang_code = hypothesis.lang_code

        state.hypotheses.append(hypothesis)
        state.hypotheses = state.hypotheses[-self.history_size:]

        events: list[Transcript] = []

        newly_committed = self._commit_stable_prefix(state)
        if newly_committed:
            committed_event = Transcript(
                text=render_words(newly_committed),
                lang_code=state.lang_code,
                kind="committed",
                committed_text=state.committed_text,
                partial_text=self._unstable_text(state),
                kept=hypothesis.pieces,
                dropped=hypothesis.dropped,
            )
            self.stats.record(committed_event)
            events.append(committed_event)

        partial_event = self._make_partial_event(state, hypothesis)
        self.stats.record(partial_event)
        events.append(partial_event)

        return tuple(events)

    def finish_utterance(
        self,
        full_pcm: bytes,
        *,
        utterance_id: str,
        utterance_start_seconds: float = 0.0,
        speech_end_seconds: Optional[float] = None,
        lang_code: str = "",
    ) -> Transcript:
        """Finalize one utterance without re-decoding committed audio.

        ``full_pcm`` must start at ``utterance_start_seconds`` and contain the
        complete utterance audio.

        ``speech_end_seconds`` is relative to the utterance timeline and should
        come from the upstream Silero VAD. Endpoint silence beyond the
        configured post-roll is excluded from Whisper input.
        """

        self._validate_pcm(full_pcm)

        state = self._states.setdefault(
            utterance_id,
            UtteranceState(
                utterance_id=utterance_id,
                lang_code=lang_code,
            ),
        )

        if lang_code:
            state.lang_code = lang_code

        if not full_pcm:
            final = Transcript(
                text=state.committed_text,
                lang_code=state.lang_code,
                kind="utterance_final",
                committed_text=state.committed_text,
            )
            self.stats.record(final)
            self._states.pop(utterance_id, None)
            return final

        full_samples = self._pcm_to_samples(full_pcm)
        audio_duration = full_samples.size / SAMPLE_RATE
        audio_end_seconds = utterance_start_seconds + audio_duration

        effective_speech_end = (
            min(speech_end_seconds, audio_end_seconds)
            if speech_end_seconds is not None
            else audio_end_seconds
        )

        # Start before the immutable boundary to give Whisper left context.
        tail_start = max(
            utterance_start_seconds,
            state.committed_end - self.final_overlap_seconds,
        )

        # Keep only a short post-roll after the final VAD speech frame.
        tail_end = min(
            audio_end_seconds,
            effective_speech_end + self.final_post_roll_seconds,
        )

        start_index = max(
            0,
            int(round(
                (tail_start - utterance_start_seconds) * SAMPLE_RATE
            )),
        )
        end_index = min(
            full_samples.size,
            int(round(
                (tail_end - utterance_start_seconds) * SAMPLE_RATE
            )),
        )

        tail_samples = full_samples[start_index:end_index]

        final_hypothesis: Optional[Hypothesis] = None

        if tail_samples.size:
            final_hypothesis = self._decode_hypothesis(
                samples=tail_samples,
                requested_language=state.lang_code,
                beam_size=ASR_BEAM_SIZE_PARTIAL,
                window_start_seconds=tail_start,
            )

            if not state.lang_code:
                state.lang_code = final_hypothesis.lang_code

        final_tail = self._reconcile_final_tail(
            state=state,
            final_hypothesis=final_hypothesis,
            speech_end_seconds=effective_speech_end,
        )

        final_words = [
            *state.committed_words,
            *final_tail,
        ]

        final_text = render_words(final_words)

        final = Transcript(
            text=final_text,
            lang_code=state.lang_code,
            kind="utterance_final",
            committed_text=final_text,
            partial_text="",
            kept=(
                final_hypothesis.pieces
                if final_hypothesis is not None
                else ()
            ),
            dropped=(
                final_hypothesis.dropped
                if final_hypothesis is not None
                else ()
            ),
        )

        self.stats.record(final)
        self._states.pop(utterance_id, None)

        return final

    def cancel_utterance(self, utterance_id: str) -> None:
        """Discard one active utterance without producing a final result."""
        self._states.pop(utterance_id, None)

    def reset(self) -> None:
        self._states.clear()
        self.stats = AsrStats()

    # ------------------------------------------------------------------
    # Decode and filtering
    # ------------------------------------------------------------------

    def _decode_hypothesis(
        self,
        *,
        samples: np.ndarray,
        requested_language: str,
        beam_size: int,
        window_start_seconds: float,
    ) -> Hypothesis:
        started = perf_counter()

        pieces, detected = self.decoder.decode(
            samples,
            requested_language,
            beam_size,
        )

        self.stats.decode_seconds += perf_counter() - started
        self.stats.audio_seconds += samples.size / SAMPLE_RATE

        kept: list[Piece] = []
        dropped: list[tuple[Piece, str]] = []

        for piece in pieces:
            reason = self._refuse(piece)
            if reason is None:
                kept.append(piece)
            else:
                dropped.append((piece, reason))

        if dropped:
            log.info(
                "ASR dropped %d segment(s): %s",
                len(dropped),
                [
                    (reason, piece.text.strip()[:60])
                    for piece, reason in dropped
                ],
            )

        absolute_words = tuple(
            Word(
                text=word.text,
                start=word.start + window_start_seconds,
                end=word.end + window_start_seconds,
                probability=word.probability,
            )
            for piece in kept
            for word in piece.words
            if word.text.strip()
        )

        window_end = (
            window_start_seconds + samples.size / SAMPLE_RATE
        )

        return Hypothesis(
            words=absolute_words,
            text=render_words(absolute_words),
            window_start=window_start_seconds,
            window_end=window_end,
            lang_code=requested_language or detected,
            pieces=tuple(kept),
            dropped=tuple(dropped),
        )

    def _refuse(self, piece: Piece) -> Optional[str]:
        """Return why a segment is rejectedf it is accepted."""

        if not piece.text.strip():
            return "empty"

        if (
            piece.no_speech_prob > self.no_speech_threshold
            and piece.avg_logprob < self.log_prob_threshold
        ):
            return "no speech"

        if piece.no_speech_prob > 0.95:
            return "very high no speech"

        if piece.avg_logprob < self.log_prob_threshold:
            return "low confidence"

        if piece.compression_ratio > self.max_compression_ratio:
            return "repetition"

        if normalise_for_match(piece.text) in self.hallucinations:
            return "known hallucination"

        spaced = normalise_for_pattern(piece.text)

        for pattern in self.hallucination_patterns:
            if pattern.fullmatch(spaced):
                return "known hallucination"

        return None

    # ------------------------------------------------------------------
    # Stabilization
    # ------------------------------------------------------------------

    def _commit_stable_prefix(
        self,
        state: UtteranceState,
    ) -> list[Word]:
        """Commit the contiguous stable prefix of the newest hypothesis."""

        if len(state.hypotheses) < self.min_agreement:
            return []

        current = state.hypotheses[-1]
        cutoff = current.window_end - self.commit_margin_seconds

        candidates = [
            word
            for word in current.words
            if (
                word.end > state.committed_end + 0.01
                and word.end <= cutoff
            )
        ]

        newly_committed: list[Word] = []

        for word in candidates:
            if not word.normalized:
                continue

            agreement = self._agreement_count(
                state=state,
                candidate=word,
            )

            if agreement < self.min_agreement:
                # Only commit a contiguous stable prefix. Once one word is
                # unstable, every later word remains partial.
                break

            if (
                newly_committed
                and word.start
                < (
                    newly_committed[-1].end
                    - self.word_time_tolerance_seconds
                )
            ):
                continue

            newly_committed.append(word)

        if newly_committed:
            state.committed_words.extend(newly_committed)
            state.committed_end = max(
                state.committed_end,
                newly_committed[-1].end,
            )

            log.debug(
                "ASR utterance=%s committed=%r committed_end=%.3f",
                state.utterance_id,
                render_words(newly_committed),
                state.committed_end,
            )

        return newly_committed

    def _agreement_count(
        self,
        *,
        state: UtteranceState,
        candidate: Word,
    ) -> int:
        """Count recent hypotheses containing the same timestamped word."""

        count = 0

        for hypothesis in reversed(state.hypotheses):
            if self._find_matching_word(
                hypothesis.words,
                candidate,
            ) is not None:
                count += 1
            else:
                # Require consecutive agreement rather than arbitrary votes
                # spread throughout the history.
                break

        return count

    def _find_matching_word(
        self,
        words: Iterable[Word],
        target: Word,
    ) -> Optional[Word]:
        """e normalized word at approximately the same time."""

        target_text = target.normalized
        if not target_text:
            return None

        best = None
        best_distance = float("inf")

        for word in words:
            if word.normalized != target_text:
                continue

            start_distance = abs(word.start - target.start)
            end_distance = abs(word.end - target.end)
            distance = max(start_distance, end_distance)

            if (
                distance <= self.word_time_tolerance_seconds
                and distance < best_distance
            ):
                best = word
                best_distance = distance

        return best

    def _unstable_words(
        self,
        state: UtteranceState,
    ) -> list[Word]:
        """Return words in the latest hypothesis that are not committed yet."""

        hypothesis = state.last_hypothesis

        if hypothesis is None:
            return []

        unstable_words: list[Word] = []

        for word in hypothesis.words:
            if word.end > state.committed_end + 0.01:
                unstable_words.append(word)

        return unstable_words

    def _unstable_text(self, state: UtteranceState) -> str:
        return render_words(self._unstable_words(state))

    def _make_partial_event(
        self,
        state: UtteranceState,
        hypothesis: Optional[Hypothesis],
    ) -> Transcript:
        partial_text = self._unstable_text(state)

        return Transcript(
            text=partial_text,
            lang_code=state.lang_code,
            kind="partial",
            committed_text=state.committed_text,
            partial_text=partial_text,
            kept=hypothesis.pieces if hypothesis else (),
            dropped=hypothesis.dropped if hypothesis else (),
        )

    # ------------------------------------------------------------------
    # Endpoint finalization
    # ------------------------------------------------------------------

    def _reconcile_final_tail(
        self,
        *,
        state: UtteranceState,
        final_hypothesis: Optional[Hypothesis],
        speech_end_seconds: float,
    ) -> list[Word]:
        """Merge partial consensus with the endpoint tail decode.

        Stable words observed repeatedly in partial hypotheses take precedence
        over conflicting words from the one-off endpoint decode.

        Final-only words after the VAD speech boundary are rejected.
        """

        historical = self._consensus_unstable_words(state)

        if final_hypothesis is None:
            final_words: list[Word] = []
        else:
            final_words = []

            for word in final_hypothesis.words:
                is_after_committed = (
                    word.end > state.committed_end + 0.01
                )
                is_inside_speech_boundary = (
                    word.start
                    <= speech_end_seconds + self.final_post_roll_seconds
                )

                if is_after_committed and is_inside_speech_boundary:
                    final_words.append(word)

        if not historical:
            return self._deduplicate_words(final_words)

        if not final_words:
            historical_words = [
                word
                for word, agreement in historical
                if agreement >= self.min_agreement
            ]
            return self._deduplicate_words(historical_words)

        merged: list[Word] = []
        used_final_indexes: set[int] = set()

        for historical_word, agreement in historical:
            final_index = self._nearest_time_index(
                final_words,
                historical_word,
                used_final_indexes,
            )

            if final_index is None:
                if agreement >= self.min_agreement:
                    merged.append(historical_word)

                continue

            final_word = final_words[final_index]
            used_final_indexes.add(final_index)

            if historical_word.normalized == final_word.normalized:
                # Both hypotheses agree on the text. Prefer the endpoint
                # candidate because its timestamps include final right context.
                merged.append(final_word)
                continue

            if agreement >= self.min_agreement:
                # A word observed consistently in multiple partial hypotheses
                # takes precedence over one conflicting endpoint hypothesis.
                log.info(
                    "ASR protected stable partial word %r from final "
                    "regression to %r at %.2f-%.2f",
                    historical_word.text,
                    final_word.text,
                    historical_word.start,
                    historical_word.end,
                )
                merged.append(historical_word)
            else:
                merged.append(final_word)

        # Add words that exist only in the final endpoint hypothesis.
        for index, word in enumerate(final_words):
            if index in used_final_indexes:
                continue

            if word.start > speech_end_seconds + 0.05:
                log.info(
                    "ASR dropped final-only word after speech end: "
                    "%r at %.2f-%.2f, speech_end=%.2f",
                    word.text,
                    word.start,
                    word.end,
                    speech_end_seconds,
                )
                continue

            if not self._overlaps_any(word, merged):
                merged.append(word)

        merged.sort(
            key=lambda item: (
                item.start,
                item.end,
            )
        )

        return self._deduplicate_words(merged)

    def _consensus_unstable_words(
        self,
        state: UtteranceState,
    ) -> list[tuple[Word, int]]:
        """Return newest unstable words with consecutive agreement counts."""

        newest = state.last_hypothesis
        if newest is None:
            return []

        result: list[tuple[Word, int]] = []

        for word in newest.words:
            if word.end <= state.committed_end + 0.01:
                continue

            result.append(
                (
                    word,
                    self._agreement_count(
                        state=state,
                        candidate=word,
                    ),
                )
            )

        return result
    
    def _nearest_time_index(
        self,
        words: list[Word],
        target: Word,
        excluded: set[int],
    ) -> Optional[int]:
        """Find the closest word occupying approximately the same time slot."""

        best_index: Optional[int] = None
        best_score = float("inf")
        
        for index, word in enumerate(words):
            if index in excluded:
                continue

            overlap = min(word.end, target.end) - max(
                word.start,
                target.start,
            )

            center_word = (word.start + word.end) / 2.0
            center_target = (target.start + target.end) / 2.0
            center_distance = abs(center_word - center_target)

            is_near_same_time = (
                overlap >= -self.word_time_tolerance_seconds
                and center_distance <= self.word_time_tolerance_seconds
            )

            if is_near_same_time and center_distance < best_score:
                best_index = index
                best_score = center_distance

        return best_index

    def _overlaps_any(
        self,
        candidate: Word,
        words: Iterable[Word],
    ) -> bool:
        for word in words:
            overlap = min(candidate.end, word.end) - max(
                candidate.start,
                word.start,
            )
            if overlap > 0:
                return True
        return False

    def _deduplicate_words(
        self,
        words: Iterable[Word],
    ) -> list[Word]:
        result: list[Word] = []

        for word in words:
            if not word.text.strip():
                continue

            if result:
                previous = result[-1]

                same_text = (
                    previous.normalized
                    and previous.normalized == word.normalized
                )

                close_in_time = (
                    word.start
                    <= previous.end + self.word_time_tolerance_seconds
                )

                if same_text and close_in_time:
                    # Keep the candidate with higher word probability.
                    if word.probability > previous.probability:
                        result[-1] = word
                    continue

            result.append(word)

        return result

    # ------------------------------------------------------------------
    # Audio validation
    # ------------------------------------------------------------------

    def _validate_pcm(self, pcm: bytes) -> None:
        if not isinstance(pcm, bytes):
            raise AsrError(
                "ASR input must be immutable bytes containing raw PCM16LE"
            )

        if len(pcm) % SAMPLE_WIDTH != 0:
            raise AsrError(
                "PCM byte length is not aligned to SAMPLE_WIDTH"
            )

    def _pcm_to_samples(self, pcm: bytes) -> np.ndarray:
        samples = np.frombuffer(
            pcm,
            dtype="<i2",
        ).astype(np.float32)

        samples /= 32768.0

        if samples.ndim != 1:
            raise AsrError("ASR audio must be mono")

        if not np.all(np.isfinite(samples)):
            raise AsrError("ASR audio contains non-finite samples")

        return samples


# Backward-friendly name for code importing Transcriber.
Transcriber = StreamingTranscriber


# ---------------------------------------------------------------------------
# faster-whisper model
# ---------------------------------------------------------------------------

class WhisperDecoder:
    """Timestamped deterministic faster-whisper decoder."""

    def __init__(
        self,
        model_id: str = "",
        device: str = "",
        compute_type: str = "",
    ) -> None:
        try:
            import torch
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise AsrError(
                "faster-whisper is not installed. Run "
                "`python3.11 -m pip install -r "
                "server/requirements.txt`."
            ) from exc

        self.model_id = model_id or ASR_MODEL

        chosen = device or ASR_DEVICE
        if not chosen:
            chosen = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = chosen

        self.compute_type = (
            compute_type
            or ASR_COMPUTE_TYPE
            or (
                "float16"
                if self.device.startswith("cuda")
                else "int8"
            )
        )

        try:
            self._model = WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=self.compute_type,
                download_root=ASR_CACHE_DIR,
            )
        except Exception as exc:  # pragma: no cover
            raise AsrError(
                f"Could not load Whisper {self.model_id!r} "
                f"on {self.device} as {self.compute_type}: {exc}"
            ) from exc

        self.warmup()

        log.info(
            "Whisper ready on %s: %s (%s)",
            self.device,
            self.model_id,
            self.compute_type,
        )

    def warmup(self, seconds: float = 1.0) -> None:
        """Pay the first-decode cost before processing real speech."""

        self.decode(
            np.zeros(
                int(seconds * SAMPLE_RATE),
                dtype=np.float32,
            ),
            "",
            ASR_BEAM_SIZE_PARTIAL,
        )

    def decode(
        self,
        samples: np.ndarray,
        lang_code: str,
        beam_size: int,
    ) -> tuple[list[Piece], str]:
        segments, info = self._model.transcribe(
            samples,
            language=lang_code or None,
            task="transcribe",

            # Deterministic decoding. Final quality is produced by
            # stabilization and reconciliation rather than a separate
            # full-utterance beam-search pass.
            beam_size=beam_size,
            temperature=0.0,

            condition_on_previous_text=ASR_CONDITION_ON_PREVIOUS,

            no_speech_threshold=ASR_NO_SPEECH_THRESHOLD,
            log_prob_threshold=ASR_LOG_PROB_THRESHOLD,
            compression_ratio_threshold=ASR_MAX_COMPRESSION_RATIO,

            # Timestamps are essential for local agreement and for protecting
            # stable words from a final regression.
            without_timestamps=False,
            word_timestamps=True,

            # Silero already runs upstream.
            vad_filter=False,
        )

        pieces: list[Piece] = []

        for segment in segments:
            words = tuple(
                Word(
                    text=word.word,
                    start=float(word.start),
                    end=float(word.end),
                    probability=float(word.probability),
                )
                for word in (segment.words or ())
                if word.word.strip()
            )

            pieces.append(
                Piece(
                    text=segment.text,
                    start=float(segment.start),
                    end=float(segment.end),
                    avg_logprob=float(segment.avg_logprob),
                    no_speech_prob=float(segment.no_speech_prob),
                    compression_ratio=float(
                        segment.compression_ratio
                    ),
                    words=words,
                )
            )

        detected = str(
            getattr(info, "language", "") or ""
        )

        return pieces, detected

    @property
    def source(self) -> str:
        return (
            f"whisper {self.model_id} on "
            f"{self.device} ({self.compute_type})"
        )


def pcm_seconds(pcm: bytes) -> float:
    return len(pcm) / SAMPLE_WIDTH / SAMPLE_RATE