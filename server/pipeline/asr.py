"""ASR, step 7 of the server pipeline.

Whisper is not a streaming-native recognizer. This module turns repeated
Whisper decoding into a stabilized stream with three result states:

partial
    An unstable hypothesis. The client replaces the previous partial text.

committed
    Newly stabilized text. Committed text is immutable and is never replaced
    by a later decode.

utterance_final
    The VAD has ended the utterance. Only the audio after the committed text
    is decoded again, with a beam search and the vocabulary prompt, and
    committed text is preserved.

The final transcript is therefore::

    append-only committed text + reconciled final tail

Whisper invents text, and it does so confidently
------------------------------------------------
Given near-silence it returns a fluent sentence nobody said. Three statistical
guards catch the unconfident kind - ``no_speech_prob`` (only together with a
poor ``avg_logprob``, except on very short audio), ``avg_logprob`` and
``compression_ratio``. The confident kind needs the word lists in
``server/data/``. Every refusal is counted and logged with its scores.

Rendering
---------
Text is rebuilt from Whisper's own word strings, which already carry the
space in front of a word in a spaced language and none in Japanese. Joining
words with a space of our own put a space between every Japanese character
on screen - 88% of the Japanese sentences of one real run.

Caller contract
---------------
Partial calls provide ``utterance_id``, the rolling window PCM and
``window_start_seconds`` relative to the utterance start. The final call
provides the same ``utterance_id``, the full utterance PCM and
``speech_end_seconds`` - where the VAD says the speech ended, before its
silence hangover. :meth:`StreamingTranscriber.transcribe` decodes one clip
with no streaming state, for audio that has no running text behind it.

All PCM is signed PCM16 little-endian, mono, at SAMPLE_RATE.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable, Literal, Optional, Protocol, Union

import numpy as np

from server.config import (
    ASR_BEAM_SIZE_FINAL,
    ASR_BEAM_SIZE_PARTIAL,
    ASR_CACHE_DIR,
    ASR_COMPUTE_TYPE,
    ASR_CONDITION_ON_PREVIOUS,
    ASR_DEVICE,
    ASR_LOG_PROB_THRESHOLD,
    ASR_MAX_COMPRESSION_RATIO,
    ASR_MODEL,
    ASR_NO_SPEECH_CERTAIN,
    ASR_NO_SPEECH_THRESHOLD,
    ASR_PROMPT_ON_PARTIALS,
    ASR_SHORT_UTTERANCE_MS,
    ASR_STREAM_COMMIT_MARGIN_SECONDS,
    ASR_STREAM_FINAL_OVERLAP_SECONDS,
    ASR_STREAM_FINAL_POST_ROLL_SECONDS,
    ASR_STREAM_HISTORY,
    ASR_STREAM_MIN_AGREEMENT,
    ASR_STREAM_WORD_TOLERANCE_SECONDS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from server.wordlists import (
    Hallucinations,
    normalise_exact,
    normalise_spaced,
    vocabulary_prompt,
)

log = logging.getLogger(__name__)


ResultKind = Literal["partial", "committed", "utterance_final"]


class AsrError(RuntimeError):
    """Raised when the ASR model cannot be loaded or used."""


# ---------------------------------------------------------------------------
# Model output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Word:
    """One timestamped word, as faster-whisper wrote it.

    ``text`` keeps Whisper's own leading space, or the lack of one.
    """

    text: str
    start: float
    end: float
    probability: float = 1.0

    @property
    def normalized(self) -> str:
        return normalise_word(self.text)

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass(frozen=True)
class Piece:
    """One segment returned by the decoder before policy filtering.

    Times are relative to the start of the audio the decoder was given.
    A decoder without word timestamps may leave ``words`` empty; the words
    are then spread evenly over the segment.
    """

    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    start: float = 0.0
    end: float = 0.0
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class Hypothesis:
    """A decoded window expressed on the utterance timeline."""

    words: tuple[Word, ...]
    text: str
    window_start: float
    window_end: float
    lang_code: str
    pieces: tuple[Piece, ...] = ()
    dropped: tuple[tuple[Piece, str], ...] = ()


@dataclass(frozen=True)
class Transcript:
    """One event emitted by the recognizer.

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
    kind: ResultKind = "utterance_final"

    # Complete immutable transcript accumulated so far.
    committed_text: str = ""

    # Complete current unstable suffix.
    partial_text: str = ""

    kept: tuple[Piece, ...] = ()
    dropped: tuple[tuple[Piece, str], ...] = ()

    #: A final only: the language the caller asked for and did not get,
    #: because the running text had already committed words in another one.
    overruled_language: str = ""

    @property
    def is_final(self) -> bool:
        return self.kind == "utterance_final"

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    @property
    def running_text(self) -> str:
        """Committed and unstable text together, for a replace-only screen."""
        return join_texts(self.committed_text, self.partial_text)

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
    #: Words a final decode repeated from the committed text and that were
    #: removed at the join.
    duplicates_removed: int = 0

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

    # End time, relative to utterance start, of immutable text.
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
# Text
# ---------------------------------------------------------------------------

def normalise_for_pattern(text: str) -> str:
    """Strip punctuation and collapse spaces for hallucination patterns."""
    return normalise_spaced(text)


def normalise_for_match(text: str) -> str:
    """Strip punctuation and spacing for exact known-hallucination matching."""
    return normalise_exact(text)


def normalise_word(text: str) -> str:
    """Normalize one word while preserving Vietnamese diacritics."""
    return normalise_spaced(text)


def is_unspaced(char: str) -> bool:
    """Whether a character belongs to a script written without spaces."""
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF          # hiragana, katakana
        or 0x3400 <= code <= 0x4DBF       # CJK extension A
        or 0x4E00 <= code <= 0x9FFF       # CJK unified ideographs
        or 0xF900 <= code <= 0xFAFF       # CJK compatibility ideographs
        or 0xFF66 <= code <= 0xFF9F       # half-width katakana
        or 0x3000 <= code <= 0x303F       # CJK punctuation
        or 0xFF00 <= code <= 0xFF65       # full-width forms
    )


def _needs_space(left: str, right: str) -> bool:
    """Whether two pieces of text written apart need a space between them."""
    if not left or not right:
        return False
    if left[-1].isspace() or right[0].isspace():
        return False
    a, b = left[-1], right[0]
    return (a.isalnum() and not is_unspaced(a)
            and b.isalnum() and not is_unspaced(b))


def join_texts(left: str, right: str) -> str:
    """Two texts one after the other, spaced only where the script wants it.

    The committed text and the unstable text come from different decodes, so
    neither carries the space that belongs between them - and Japanese wants
    none at all.
    """
    left, right = left.strip(), right.strip()
    if not left or not right:
        return left or right
    return f"{left} {right}" if _needs_space(left, right) else left + right


def render_words(words: Iterable[Word]) -> str:
    """Rebuild text from Whisper's own word strings.

    Never adds a space between words of its own: Whisper's word strings
    already carry the space a spaced language needs, and for Japanese it
    splits on token boundaries - "GLM5.2" can arrive as four words that must
    be put back together as written.
    """
    text = "".join(word.text for word in words if word.text.strip())

    text = re.sub(r"\s+", " ", text)
    # Remove spaces before common punctuation.
    text = re.sub(r"\s+([.,!?;:%)\]\}、。！？])", r"\1", text)
    # Remove spaces after opening punctuation.
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    return text.strip()


_TOKEN = re.compile(r"\s*\S+")


def spread_words(piece: Piece, fallback_end: float) -> tuple[Word, ...]:
    """Words for a piece whose decoder gave no word timestamps.

    Split on whitespace, each token keeping a leading space unless it opens
    with a character from an unspaced script, and spread evenly over the
    segment - or over the whole window when the segment has no times.
    """
    tokens = [token.strip() for token in _TOKEN.findall(piece.text)]
    tokens = [token for token in tokens if token]
    if not tokens:
        return ()
    start = piece.start
    end = piece.end if piece.end > piece.start else max(fallback_end, start)
    step = (end - start) / len(tokens)
    return tuple(
        Word(
            text=token if is_unspaced(token[0]) else f" {token}",
            start=start + index * step,
            end=start + (index + 1) * step,
        )
        for index, token in enumerate(tokens)
    )


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
        prompt: Optional[str] = None,
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
        short_utterance_ms: float = ASR_SHORT_UTTERANCE_MS,
        no_speech_certain: float = ASR_NO_SPEECH_CERTAIN,
        hallucinations: Union[Hallucinations, Iterable[str], None] = None,
        hallucination_patterns: Optional[Iterable[str]] = None,
        prompt: Optional[str] = None,
        min_agreement: int = ASR_STREAM_MIN_AGREEMENT,
        commit_margin_seconds: float = ASR_STREAM_COMMIT_MARGIN_SECONDS,
        final_overlap_seconds: float = ASR_STREAM_FINAL_OVERLAP_SECONDS,
        final_post_roll_seconds: float = ASR_STREAM_FINAL_POST_ROLL_SECONDS,
        word_time_tolerance_seconds: float = (
            ASR_STREAM_WORD_TOLERANCE_SECONDS
        ),
        history_size: int = ASR_STREAM_HISTORY,
    ) -> None:
        if min_agreement < 2:
            raise ValueError("min_agreement must be at least 2")

        self.decoder = decoder if decoder is not None else WhisperDecoder()

        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self.max_compression_ratio = max_compression_ratio
        self.short_utterance_ms = short_utterance_ms
        self.no_speech_certain = no_speech_certain

        self.min_agreement = min_agreement
        self.commit_margin_seconds = commit_margin_seconds
        self.final_overlap_seconds = final_overlap_seconds
        self.final_post_roll_seconds = final_post_roll_seconds
        self.word_time_tolerance_seconds = word_time_tolerance_seconds
        self.history_size = history_size

        if isinstance(hallucinations, Hallucinations):
            self.hallucinations = hallucinations
        else:
            self.hallucinations = Hallucinations(
                exact=None if hallucinations is None else list(hallucinations),
                patterns=None if hallucination_patterns is None else tuple(
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in hallucination_patterns
                ),
            )

        # The meeting's own words, from server/data/vocabulary.txt. `or None`
        # is the invariant, not tidiness: Whisper treats "" as a prompt it
        # was given, which is not the same as no prompt.
        self.prompt = (prompt if prompt is not None
                       else vocabulary_prompt()) or None
        if self.prompt:
            log.info("Whisper vocabulary prompt (%d chars): %s",
                     len(self.prompt), self.prompt)

        self._states: dict[str, UtteranceState] = {}
        self.stats = AsrStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_committed(self, utterance_id: str) -> bool:
        """Whether the running text has already committed words here."""
        state = self._states.get(utterance_id)
        return bool(state and state.committed_words)

    def transcribe(
        self,
        pcm: bytes,
        lang_code: str = "",
        is_final: bool = True,
    ) -> Transcript:
        """Decode one clip on its own, with no streaming state.

        For audio no running text was kept for - the halves of an utterance
        cut on a language change - and for the real tests, which put the
        model and the guards to a recording one sentence at a time.
        """
        self._validate_pcm(pcm)
        kind: ResultKind = "utterance_final" if is_final else "partial"
        if not pcm:
            transcript = Transcript(text="", lang_code=lang_code, kind=kind)
            self.stats.record(transcript)
            return transcript

        samples = self._pcm_to_samples(pcm)
        hypothesis = self._decode_hypothesis(
            samples=samples,
            requested_language=lang_code,
            beam_size=ASR_BEAM_SIZE_FINAL if is_final else ASR_BEAM_SIZE_PARTIAL,
            window_start_seconds=0.0,
            utterance_ms=samples.size / SAMPLE_RATE * 1000.0,
            prompt=self._prompt_for(is_final),
        )
        text = hypothesis.text
        transcript = Transcript(
            text=text,
            lang_code=hypothesis.lang_code,
            kind=kind,
            committed_text=text if is_final else "",
            partial_text="" if is_final else text,
            kept=hypothesis.pieces,
            dropped=hypothesis.dropped,
        )
        self.stats.record(transcript)
        return transcript

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
            UtteranceState(utterance_id=utterance_id, lang_code=lang_code),
        )

        if lang_code:
            state.lang_code = lang_code

        state.revision += 1

        if not pcm:
            partial = self._make_partial_event(state, None)
            self.stats.record(partial)
            return (partial,)

        samples = self._pcm_to_samples(pcm)
        window_seconds = samples.size / SAMPLE_RATE

        hypothesis = self._decode_hypothesis(
            samples=samples,
            requested_language=state.lang_code,
            beam_size=ASR_BEAM_SIZE_PARTIAL,
            window_start_seconds=window_start_seconds,
            utterance_ms=(window_start_seconds + window_seconds) * 1000.0,
            prompt=self._prompt_for(False),
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
        come from the VAD, excluding its silence hangover. Audio beyond the
        configured post-roll is not given to Whisper.

        ``lang_code`` is used unless the running text has already committed
        words in another language: those words were decoded in that language,
        and a sentence whose language disagrees with its running text was
        measured three to four times as likely to be unrelated to what was
        said. The language that lost is reported as ``overruled_language``.
        """

        self._validate_pcm(full_pcm)

        state = self._states.setdefault(
            utterance_id,
            UtteranceState(utterance_id=utterance_id, lang_code=lang_code),
        )

        overruled = ""
        if lang_code:
            if not state.committed_words or not state.lang_code:
                state.lang_code = lang_code
            elif lang_code != state.lang_code:
                overruled = lang_code

        if not full_pcm:
            final = Transcript(
                text=state.committed_text,
                lang_code=state.lang_code,
                kind="utterance_final",
                committed_text=state.committed_text,
                overruled_language=overruled,
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
            int(round((tail_start - utterance_start_seconds) * SAMPLE_RATE)),
        )
        end_index = min(
            full_samples.size,
            int(round((tail_end - utterance_start_seconds) * SAMPLE_RATE)),
        )

        tail_samples = full_samples[start_index:end_index]

        final_hypothesis: Optional[Hypothesis] = None

        if tail_samples.size:
            final_hypothesis = self._decode_hypothesis(
                samples=tail_samples,
                requested_language=state.lang_code,
                # The sentence the viewer keeps is worth a beam search. Greedy
                # decoding here doubled the repeated words on a real meeting.
                beam_size=ASR_BEAM_SIZE_FINAL,
                window_start_seconds=tail_start,
                # The whole utterance, hangover included: that is the length
                # the short-audio rule was measured on.
                utterance_ms=audio_duration * 1000.0,
                prompt=self._prompt_for(True),
            )

            if not state.lang_code:
                state.lang_code = final_hypothesis.lang_code

        final_tail = self._reconcile_final_tail(
            state=state,
            final_hypothesis=final_hypothesis,
            speech_end_seconds=effective_speech_end,
        )
        final_tail = self._drop_repeat_at_join(state.committed_words,
                                               final_tail)

        final_text = join_texts(state.committed_text, render_words(final_tail))

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
            overruled_language=overruled,
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

    def _prompt_for(self, is_final: bool) -> Optional[str]:
        """The vocabulary prompt, for the decodes that should carry it.

        The running text is decoded six times more often than a sentence, and
        a prompt makes Whisper fill near-silence rather than leave it.
        """
        return self.prompt if (is_final or ASR_PROMPT_ON_PARTIALS) else None

    def _decode_hypothesis(
        self,
        *,
        samples: np.ndarray,
        requested_language: str,
        beam_size: int,
        window_start_seconds: float,
        utterance_ms: float,
        prompt: Optional[str] = None,
    ) -> Hypothesis:
        started = perf_counter()

        pieces, detected = self.decoder.decode(
            samples,
            requested_language,
            beam_size,
            prompt,
        )

        self.stats.decode_seconds += perf_counter() - started
        self.stats.audio_seconds += samples.size / SAMPLE_RATE

        kept: list[Piece] = []
        dropped: list[tuple[Piece, str]] = []

        for piece in pieces:
            reason = self._refuse(piece, utterance_ms)
            if reason is None:
                kept.append(piece)
            else:
                dropped.append((piece, reason))

        if dropped:
            # Scores included: they are the evidence for where the thresholds
            # belong, and a guard that hides what it refused turns one bug
            # into two.
            log.info(
                "ASR dropped %d segment(s): %s",
                len(dropped),
                [
                    (reason, round(piece.no_speech_prob, 2),
                     round(piece.avg_logprob, 2), piece.text.strip()[:60])
                    for piece, reason in dropped
                ],
            )
        for piece in kept:
            if piece.no_speech_prob > self.no_speech_threshold:
                # The segments the no_speech_prob-alone rule would have thrown
                # away. Logged so the threshold can be placed from a meeting.
                log.info("ASR kept a segment scored as silence: "
                         "no_speech %.2f, logprob %.2f, %r",
                         piece.no_speech_prob, piece.avg_logprob,
                         piece.text.strip()[:60])

        window_seconds = samples.size / SAMPLE_RATE
        absolute_words = tuple(
            Word(
                text=word.text,
                start=word.start + window_start_seconds,
                end=word.end + window_start_seconds,
                probability=word.probability,
            )
            for piece in kept
            for word in (piece.words or spread_words(piece, window_seconds))
            if word.text.strip()
        )

        return Hypothesis(
            words=absolute_words,
            text=render_words(absolute_words),
            window_start=window_start_seconds,
            window_end=window_start_seconds + window_seconds,
            lang_code=requested_language or detected,
            pieces=tuple(kept),
            dropped=tuple(dropped),
        )

    def _refuse(self, piece: Piece,
                utterance_ms: float = float("inf")) -> Optional[str]:
        """Why a segment should not be shown, or None to keep it."""

        if not piece.text.strip():
            return "empty"

        if piece.no_speech_prob > self.no_speech_threshold:
            # faster-whisper's own rule needs both signals: "don't skip if the
            # logprob is high enough, despite the no_speech_prob". Read alone
            # it refused 68 confidently decoded segments of one meeting.
            #
            # But that benefit of the doubt belongs to audio long enough to
            # hold a sentence. Every invention confirmed on a real meeting
            # came from a scrap under ASR_SHORT_UTTERANCE_MS.
            if (utterance_ms < self.short_utterance_ms
                    or piece.avg_logprob <= self.log_prob_threshold
                    or piece.no_speech_prob > self.no_speech_certain):
                return "no speech"

        if piece.avg_logprob < self.log_prob_threshold:
            return "low confidence"

        if piece.compression_ratio > self.max_compression_ratio:
            return "repetition"

        if self.hallucinations.is_invented(piece.text):
            # Nothing above catches these: Whisper writes its sign-offs with
            # more confidence than it writes real speech. See server/data/.
            return "known hallucination"

        return None

    # ------------------------------------------------------------------
    # Stabilization
    # ------------------------------------------------------------------

    def _after_committed(self, word: Word, state: UtteranceState) -> bool:
        """Whether a word lies past the committed text.

        Judged on the word's middle, not its end. Two decodes place the same
        word a little differently, and a word that merely *ends* past the
        boundary is usually the last committed word again - which is how a
        word came to be shown twice in 45% of one run's sentences.
        """
        return word.center > state.committed_end + 0.01

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
            if self._after_committed(word, state) and word.end <= cutoff
        ]

        newly_committed: list[Word] = []

        for word in candidates:
            if not word.normalized:
                continue

            agreement = self._agreement_count(state=state, candidate=word)

            if agreement < self.min_agreement:
                # Only commit a contiguous stable prefix. Once one word is
                # unstable, every later word remains partial.
                break

            if (
                newly_committed
                and word.start
                < newly_committed[-1].end - self.word_time_tolerance_seconds
            ):
                continue

            newly_committed.append(word)

        newly_committed = self._drop_repeat_at_join(state.committed_words,
                                                    newly_committed)
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

    def _drop_repeat_at_join(self, committed: list[Word],
                             following: list[Word]) -> list[Word]:
        """Remove the committed text's last word where the next text repeats it.

        A second decode of the same audio places a word a little earlier or
        later, and its copy of the last committed word can still land past the
        boundary. It is the same word said once.
        """
        following = list(following)
        while (committed and following
               and following[0].normalized
               and following[0].normalized == committed[-1].normalized
               and following[0].start
               <= committed[-1].end + self.word_time_tolerance_seconds):
            following.pop(0)
            self.stats.duplicates_removed += 1
        return following

    def _agreement_count(
        self,
        *,
        state: UtteranceState,
        candidate: Word,
    ) -> int:
        """Count recent hypotheses containing the same timestamped word."""

        count = 0

        for hypothesis in reversed(state.hypotheses):
            if self._find_matching_word(hypothesis.words, candidate) is not None:
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
        """The same normalized word at approximately the same time."""

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
        """Words in the latest hypothesis that are not committed yet."""

        hypothesis = state.last_hypothesis

        if hypothesis is None:
            return []

        unstable = [word for word in hypothesis.words
                    if self._after_committed(word, state)]
        return self._drop_repeat_at_join(state.committed_words, unstable)

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
            final_words = [
                word
                for word in final_hypothesis.words
                if self._after_committed(word, state)
                and word.start
                <= speech_end_seconds + self.final_post_roll_seconds
            ]

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

        merged.sort(key=lambda item: (item.start, item.end))

        return self._deduplicate_words(merged)

    def _consensus_unstable_words(
        self,
        state: UtteranceState,
    ) -> list[tuple[Word, int]]:
        """Newest unstable words with their consecutive agreement counts."""

        newest = state.last_hypothesis
        if newest is None:
            return []

        return [
            (word, self._agreement_count(state=state, candidate=word))
            for word in newest.words
            if self._after_committed(word, state)
        ]

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

            overlap = min(word.end, target.end) - max(word.start, target.start)
            center_distance = abs(word.center - target.center)

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
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        samples /= 32768.0

        if not np.all(np.isfinite(samples)):
            raise AsrError("ASR audio contains non-finite samples")

        return samples


# The name the rest of the pipeline imports.
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
                "`python3.11 -m pip install -r server/requirements.txt`."
            ) from exc

        self.model_id = model_id or ASR_MODEL

        chosen = device or ASR_DEVICE
        if not chosen:
            chosen = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = chosen

        self.compute_type = (
            compute_type
            or ASR_COMPUTE_TYPE
            or ("float16" if self.device.startswith("cuda") else "int8")
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
            np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32),
            "",
            ASR_BEAM_SIZE_PARTIAL,
        )

    def decode(
        self,
        samples: np.ndarray,
        lang_code: str,
        beam_size: int,
        prompt: Optional[str] = None,
    ) -> tuple[list[Piece], str]:
        segments, info = self._model.transcribe(
            samples,
            language=lang_code or None,
            task="transcribe",
            beam_size=beam_size,
            temperature=0.0,
            # None, not "". An empty string is itself a prompt as far as
            # Whisper is concerned.
            initial_prompt=prompt or None,
            condition_on_previous_text=ASR_CONDITION_ON_PREVIOUS,
            no_speech_threshold=ASR_NO_SPEECH_THRESHOLD,
            log_prob_threshold=ASR_LOG_PROB_THRESHOLD,
            compression_ratio_threshold=ASR_MAX_COMPRESSION_RATIO,
            # Timestamps are essential for local agreement and for protecting
            # stable words from a final regression.
            without_timestamps=False,
            word_timestamps=True,
            # Silero already runs upstream, and a second VAD eats the pre-roll
            # that holds the word onsets.
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
                    avg_logprob=float(segment.avg_logprob),
                    no_speech_prob=float(segment.no_speech_prob),
                    compression_ratio=float(segment.compression_ratio),
                    start=float(segment.start),
                    end=float(segment.end),
                    words=words,
                )
            )

        detected = str(getattr(info, "language", "") or "")

        return pieces, detected

    @property
    def source(self) -> str:
        return (
            f"whisper {self.model_id} on "
            f"{self.device} ({self.compute_type})"
        )


def pcm_seconds(pcm: bytes) -> float:
    return len(pcm) / SAMPLE_WIDTH / SAMPLE_RATE
