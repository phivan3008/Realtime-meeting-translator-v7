"""ASR - step 7 of the server pipeline.

``DESIGN.md`` section 3.7: Whisper large-v3 through faster-whisper, in two
modes.  A *partial* runs while somebody is still talking and is thrown away
as soon as the next one arrives, so it is decoded greedily.  A *final* runs
once the buffer manager has committed the sentence and is worth a beam
search, because it is what the viewer keeps and what gets translated.

Whisper invents text, and it does so confidently
------------------------------------------------
Given near-silence or noise it does not return nothing; it returns a fluent
sentence that was never said - "Thank you for watching", subtitle credits,
whatever its training data had in the quiet parts.  Worse, it sometimes locks
into a loop and repeats one phrase to fill the time.

Three guards, and all three matter:

``no_speech_prob``
    Whisper's own estimate that a segment holds no speech.

``avg_logprob``
    How confident the decoder was. Invented text scores badly.

``compression_ratio``
    A repetition detector. Natural speech gzips to around 1.5-2.0; a segment
    that compresses far better than that is a loop.

Segments failing a guard are dropped and counted rather than quietly deleted,
so a run where the guards start eating real speech is visible in the stats.

Two settings deliberately differ from faster-whisper's defaults
---------------------------------------------------------------
``vad_filter`` is off. Silero already ran, at the front of this pipeline, and
running a second VAD over audio the first one already trimmed removes the
leading consonants that the pre-roll exists to preserve.

``condition_on_previous_text`` is off. It feeds the previous sentence in as
the next one's prompt, which is exactly how a single invented sentence turns
into a paragraph of them. Every utterance here is already a complete thought.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol

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
    ASR_HALLUCINATIONS,
    ASR_HALLUCINATION_PATTERNS,
    ASR_NO_SPEECH_THRESHOLD,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)

log = logging.getLogger(__name__)


class AsrError(RuntimeError):
    """Raised when the ASR model cannot be loaded or used."""


@dataclass(frozen=True)
class Piece:
    """One segment as the decoder returned it, before any judgement."""

    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


@dataclass(frozen=True)
class Transcript:
    """What was said, and what had to be thrown away to say it."""

    text: str
    lang_code: str
    is_final: bool
    kept: tuple[Piece, ...] = ()
    dropped: tuple[tuple[Piece, str], ...] = ()

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    def summary(self) -> str:
        state = "final" if self.is_final else "partial"
        return (f"{state} [{self.lang_code or 'auto'}] "
                f"{len(self.kept)} kept, {len(self.dropped)} dropped: "
                f"{self.text[:60]!r}")


@dataclass
class AsrStats:
    partials: int = 0
    finals: int = 0
    empty: int = 0
    dropped_pieces: int = 0
    dropped_reasons: dict[str, int] = field(default_factory=dict)
    decode_seconds: float = 0.0
    audio_seconds: float = 0.0

    @property
    def realtime_factor(self) -> float:
        return self.decode_seconds / self.audio_seconds if self.audio_seconds else 0.0

    def record(self, transcript: Transcript) -> None:
        if transcript.is_final:
            self.finals += 1
        else:
            self.partials += 1
        if not transcript.has_text:
            self.empty += 1
        for _piece, reason in transcript.dropped:
            self.dropped_pieces += 1
            self.dropped_reasons[reason] = self.dropped_reasons.get(reason, 0) + 1


# Punctuation and spacing only. Vietnamese diacritics are letters and stay:
# stripping them would fold real words together.
_UNSPOKEN = re.compile(
    r"[\s.,!?;:\-\u2010-\u2015\u3001\u3002\u30fb\uff01\uff1f\uff0c\uff0e"
    r"\"'\u2018\u2019\u201c\u201d()\[\]]+"
)


#: Punctuation only, spacing kept. The pattern rules read as sentences, so
#: they need the word boundaries the exact rules throw away.
_PUNCTUATION = re.compile(
    r"[.,!?;:\-\u2010-\u2015\u3001\u3002\u30fb\uff01\uff1f\uff0c\uff0e"
    r"\"'\u2018\u2019\u201c\u201d()\[\]]+"
)


def normalise_for_pattern(text: str) -> str:
    """Strip punctuation and collapse spacing, for the pattern rules."""
    return " ".join(_PUNCTUATION.sub(" ", text).casefold().split())


def normalise_for_match(text: str) -> str:
    """Reduce a segment to what was said, for comparing against a known line.

    Whisper punctuates its own inventions inconsistently - the same sign-off
    arrives with and without the full stop - so punctuation cannot be part of
    the comparison.
    """
    return _UNSPOKEN.sub("", text).casefold()


class Decoder(Protocol):
    """What :class:`Transcriber` needs; a stub satisfies it in the tests."""

    def decode(self, samples: np.ndarray, lang_code: str,
               beam_size: int) -> tuple[list[Piece], str]:
        ...                                         # pragma: no cover


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
class Transcriber:
    """Turn one utterance into text, refusing the parts Whisper invented."""

    def __init__(
        self,
        decoder: Optional[Decoder] = None,
        no_speech_threshold: float = ASR_NO_SPEECH_THRESHOLD,
        log_prob_threshold: float = ASR_LOG_PROB_THRESHOLD,
        max_compression_ratio: float = ASR_MAX_COMPRESSION_RATIO,
        hallucinations: Iterable[str] = ASR_HALLUCINATIONS,
        hallucination_patterns: Iterable[str] = ASR_HALLUCINATION_PATTERNS,
    ) -> None:
        self.decoder = decoder if decoder is not None else WhisperDecoder()
        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self.max_compression_ratio = max_compression_ratio
        self.hallucinations = frozenset(
            normalise_for_match(phrase) for phrase in hallucinations
        )
        # Anchored: a pattern describes a whole invented line, not a phrase
        # that might appear inside a real sentence.
        self.hallucination_patterns = tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in hallucination_patterns
        )
        self.stats = AsrStats()

    def transcribe(self, pcm: bytes, lang_code: str = "",
                   is_final: bool = True) -> Transcript:
        if not pcm:
            transcript = Transcript("", lang_code, is_final)
            self.stats.record(transcript)
            return transcript

        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        beam = ASR_BEAM_SIZE_FINAL if is_final else ASR_BEAM_SIZE_PARTIAL
        pieces, detected = self.decoder.decode(samples, lang_code, beam)

        kept, dropped = [], []
        for piece in pieces:
            reason = self._refuse(piece)
            if reason is None:
                kept.append(piece)
            else:
                dropped.append((piece, reason))

        transcript = Transcript(
            text=" ".join(piece.text.strip() for piece in kept).strip(),
            lang_code=lang_code or detected,
            is_final=is_final,
            kept=tuple(kept),
            dropped=tuple(dropped),
        )
        self.stats.record(transcript)
        self.stats.audio_seconds += samples.size / SAMPLE_RATE
        if dropped:
            # The text, not just the count and the reason. A refusal that
            # hides what it refused turns one bug into two, and this project
            # has paid for that lesson twice.
            log.info("ASR dropped %d segment(s): %s", len(dropped),
                     [(reason, piece.text.strip()[:60])
                      for piece, reason in dropped])
        return transcript

    def _refuse(self, piece: Piece) -> Optional[str]:
        """Why this segment should not be shown, or None to keep it."""
        if not piece.text.strip():
            return "empty"
        if piece.no_speech_prob > self.no_speech_threshold:
            return "no speech"
        if piece.avg_logprob < self.log_prob_threshold:
            return "low confidence"
        if piece.compression_ratio > self.max_compression_ratio:
            return "repetition"
        if normalise_for_match(piece.text) in self.hallucinations:
            # Every check above is statistical, and this line passes all of
            # them: Whisper writes its sign-offs with more confidence than it
            # writes real speech.
            return "known hallucination"
        spaced = normalise_for_pattern(piece.text)
        for pattern in self.hallucination_patterns:
            if pattern.fullmatch(spaced):
                # The same invention with the channel name swapped. Matching
                # whole lines let one through two runs after the other was
                # listed; the name is a hole and there is no end of names.
                return "known hallucination"
        return None

    def reset(self) -> None:
        self.stats = AsrStats()


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class WhisperDecoder:
    """faster-whisper, configured for one committed utterance at a time."""

    def __init__(self, model_id: str = "", device: str = "",
                 compute_type: str = "") -> None:
        try:
            import torch
            from faster_whisper import WhisperModel
        except ImportError as exc:                      # pragma: no cover
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
            compute_type or ASR_COMPUTE_TYPE
            or ("float16" if self.device.startswith("cuda") else "int8")
        )
        try:
            self._model = WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=self.compute_type,
                download_root=ASR_CACHE_DIR,
            )
        except Exception as exc:                        # pragma: no cover
            raise AsrError(
                f"Could not load Whisper {self.model_id!r} on {self.device} "
                f"as {self.compute_type}: {exc}"
            ) from exc
        self.warmup()
        log.info("Whisper ready on %s: %s (%s)", self.device, self.model_id,
                 self.compute_type)

    def warmup(self, seconds: float = 1.0) -> None:
        """Pay the first-decode cost now, not on the meeting's first sentence."""
        self.decode(np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32),
                    "", ASR_BEAM_SIZE_PARTIAL)

    def decode(self, samples: np.ndarray, lang_code: str,
               beam_size: int) -> tuple[list[Piece], str]:
        segments, info = self._model.transcribe(
            samples,
            language=lang_code or None,
            beam_size=beam_size,
            condition_on_previous_text=ASR_CONDITION_ON_PREVIOUS,
            no_speech_threshold=ASR_NO_SPEECH_THRESHOLD,
            log_prob_threshold=ASR_LOG_PROB_THRESHOLD,
            without_timestamps=True,
            # Silero already trimmed this audio at the front of the pipeline.
            # A second VAD pass would cut into the pre-roll that exists to
            # keep the first consonant of the sentence.
            vad_filter=False,
        )
        pieces = [
            Piece(text=segment.text,
                  avg_logprob=float(segment.avg_logprob),
                  no_speech_prob=float(segment.no_speech_prob),
                  compression_ratio=float(segment.compression_ratio))
            for segment in segments
        ]
        return pieces, str(getattr(info, "language", "") or "")

    @property
    def source(self) -> str:
        return f"whisper {self.model_id} on {self.device} ({self.compute_type})"


def pcm_seconds(pcm: bytes) -> float:
    return len(pcm) / SAMPLE_WIDTH / SAMPLE_RATE
