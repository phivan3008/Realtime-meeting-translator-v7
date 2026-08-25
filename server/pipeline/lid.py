"""Language ID - step 6 of the server pipeline.

``DESIGN.md`` 3.6: SpeechBrain VoxLingua107. Returns ``'vi'``, ``'ja'`` or
empty.

The model knows 107 languages but only the two the meeting can contain are
read, then renormalised between them. Left free it answers Korean or Chinese
for Japanese - reasonable for the model, useless here, since the only thing
downstream does with the answer is force Whisper's ``language``.

**Forcing the wrong language does not raise.** Whisper returns fluent,
confident, wrong text and the translator faithfully translates the nonsense.
So when the two scores are too close the answer is empty, and the session
falls back to the meeting's last known language rather than letting Whisper
choose from 99.

Layering:

``LanguageIdentifier``
    The policy. Pure Python, tested with a stub scorer.

``VoxLinguaClassifier``
    The model.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from server.config import (
    LID_CACHE_DIR,
    LID_DEVICE,
    LID_LANGUAGES,
    LID_MIN_DURATION_MS,
    LID_MIN_MARGIN,
    LID_MODEL,
    LID_UNKNOWN,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)

log = logging.getLogger(__name__)


class LanguageIdError(RuntimeError):
    """Raised when the language model cannot be loaded or used."""


def language_code(label: str) -> str:
    """Reduce a VoxLingua107 label to its two-letter code.

    The checkpoint labels look like ``"vi: Vietnamese"`` in some revisions and
    plain ``"vi"`` in others, so take whichever is in front.
    """
    return label.split(":", 1)[0].strip().lower()


def two_way_probabilities(scores: dict[str, float]) -> dict[str, float]:
    """Softmax over just the languages we care about.

    The model's outputs are log-probabilities over all 107 languages;
    re-normalising over two of them gives the conditional probability of each,
    *given* that the sentence is one of the two - which is exactly the question
    being asked.
    """
    if not scores:
        return {}
    largest = max(scores.values())
    weights = {name: math.exp(value - largest) for name, value in scores.items()}
    total = sum(weights.values())
    if total <= 0.0:                                    # pragma: no cover
        return {name: 1.0 / len(scores) for name in scores}
    return {name: weight / total for name, weight in weights.items()}


@dataclass(frozen=True)
class LanguageDecision:
    """What language a sentence is in, and how sure the model was."""

    lang_code: str
    confidence: float
    margin: float
    reason: str
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return self.lang_code != LID_UNKNOWN


@dataclass
class LanguageStats:
    seen: int = 0
    unknown: int = 0
    per_language: dict[str, int] = field(default_factory=dict)

    @property
    def decided(self) -> int:
        return self.seen - self.unknown

    def record(self, decision: LanguageDecision) -> None:
        self.seen += 1
        if not decision.known:
            self.unknown += 1
        key = decision.lang_code or "unknown"
        self.per_language[key] = self.per_language.get(key, 0) + 1


class Scorer(Protocol):
    """What :class:`LanguageIdentifier` needs; a stub satisfies it in tests."""

    def scores(self, pcm: bytes) -> dict[str, float]:
        ...                                         # pragma: no cover


class LanguageIdentifier:
    """Decide between the meeting's two languages, or admit to not knowing."""

    def __init__(
        self,
        scorer: Optional[Scorer] = None,
        languages: tuple[str, ...] = LID_LANGUAGES,
        min_margin: float = LID_MIN_MARGIN,
        min_duration_ms: int = LID_MIN_DURATION_MS,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if len(languages) < 2:
            raise ValueError("at least two languages are needed to choose between")
        if not 0.0 <= min_margin <= 1.0:
            raise ValueError("min_margin must be between 0 and 1")
        self.scorer = scorer if scorer is not None else VoxLinguaClassifier()
        self.languages = tuple(languages)
        self.min_margin = min_margin
        self.min_duration_ms = min_duration_ms
        self.sample_rate = sample_rate
        self.stats = LanguageStats()

    def identify(self, pcm: bytes) -> LanguageDecision:
        duration_ms = len(pcm) / SAMPLE_WIDTH / self.sample_rate * 1000.0
        if duration_ms < self.min_duration_ms:
            return self._record(LanguageDecision(
                LID_UNKNOWN, 0.0, 0.0,
                f"only {duration_ms:.0f} ms, too short to tell the languages apart",
            ))

        raw = self.scorer.scores(pcm)
        wanted = {name: raw[name] for name in self.languages if name in raw}
        if len(wanted) < len(self.languages):
            missing = sorted(set(self.languages) - set(wanted))
            return self._record(LanguageDecision(
                LID_UNKNOWN, 0.0, 0.0,
                f"the model reported nothing for {missing}",
            ))

        if len(wanted) < 2:
            return self._record(LanguageDecision(
                LID_UNKNOWN, 0.0, 0.0,
                f"only one language configured: {sorted(wanted)}",
            ))

        probabilities = two_way_probabilities(wanted)
        ranked = sorted(probabilities.items(), key=lambda item: item[1],
                        reverse=True)
        best, runner_up = ranked[0], ranked[1]
        margin = best[1] - runner_up[1]

        if margin < self.min_margin:
            return self._record(LanguageDecision(
                LID_UNKNOWN, best[1], margin,
                f"{best[0]} and {runner_up[0]} are only {margin:.2f} apart; "
                "letting the ASR detect it",
                probabilities,
            ))
        return self._record(LanguageDecision(
            best[0], best[1], margin, f"{best[0]} by {margin:.2f}", probabilities,
        ))

    def _record(self, decision: LanguageDecision) -> LanguageDecision:
        self.stats.record(decision)
        if not decision.known:
            log.info("Language undecided: %s", decision.reason)
        return decision

    def reset(self) -> None:
        self.stats = LanguageStats()


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class VoxLinguaClassifier:
    """SpeechBrain's VoxLingua107 ECAPA language classifier."""

    def __init__(self, model_id: str = "", device: str = "",
                 languages: tuple[str, ...] = LID_LANGUAGES) -> None:
        try:
            import torch
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError as exc:                      # pragma: no cover
            raise LanguageIdError(
                "speechbrain / torch are not installed. Run "
                "`python3.11 -m pip install -r server/requirements.txt`."
            ) from exc

        self._torch = torch
        self.model_id = model_id or LID_MODEL
        chosen = device or LID_DEVICE
        if not chosen:
            chosen = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cuda:0" if chosen == "cuda" else chosen
        try:
            self._model = EncoderClassifier.from_hparams(
                source=self.model_id,
                savedir=str(Path(LID_CACHE_DIR) / self.model_id.replace("/", "_")),
                run_opts={"device": self.device},
            )
        except Exception as exc:                        # pragma: no cover
            raise LanguageIdError(
                f"Could not load {self.model_id!r}: {exc}"
            ) from exc

        self.index_of = self._locate(languages)
        self.warmup()
        log.info("Language ID ready on %s: %s, tracking %s",
                 self.device, self.model_id, sorted(self.index_of))

    def _locate(self, languages: tuple[str, ...]) -> dict[str, int]:
        """Map each wanted language to its column in the model's output.

        Done once, and loudly: a checkpoint that does not know Vietnamese
        would otherwise return nothing for it on every sentence, and the
        policy would read that as "undecided" forever - a filter that looks
        like it is working while deciding nothing.
        """
        encoder = self._model.hparams.label_encoder
        # SpeechBrain warns on every load unless told the length is known to
        # be right. We only read the mapping, never resize it.
        if hasattr(encoder, "ignore_len"):
            encoder.ignore_len()
        labels = encoder.ind2lab
        found: dict[str, int] = {}
        for index in sorted(labels):
            code = language_code(str(labels[index]))
            if code in languages and code not in found:
                found[code] = int(index)
        missing = sorted(set(languages) - set(found))
        if missing:
            raise LanguageIdError(
                f"{self.model_id!r} has no class for {missing}; "
                f"it knows {len(labels)} languages"
            )
        return found

    def warmup(self, seconds: float = 1.0) -> None:
        """Pay the first-inference cost now, not on the meeting's first word."""
        self.scores(np.zeros(int(seconds * SAMPLE_RATE), dtype="<i2").tobytes())

    def scores(self, pcm: bytes) -> dict[str, float]:
        """The model's raw score for each language we are choosing between."""
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        waveform = self._torch.from_numpy(samples).unsqueeze(0)
        with self._torch.no_grad():
            out_prob, _score, _index, _label = self._model.classify_batch(
                waveform.to(self.device)
            )
        row = out_prob.squeeze().float().cpu().numpy()
        return {code: float(row[index]) for code, index in self.index_of.items()}

    @property
    def source(self) -> str:
        return f"{self.model_id} ({self.device})"
