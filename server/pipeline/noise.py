"""Deep Noise Filter - step 3 of the server pipeline.

``DESIGN.md`` section 3.3: classify the audio and drop what is not speech -
keyboard clatter, a cough, a chair scraping - before it reaches the stages
that cost real GPU time.

Silero already removes silence, but it is a *voice activity* detector, not a
sound classifier: it fires happily on a cough, a laugh, a door slam.  An
AudioSet classifier knows the difference, so it gets the last word on whether
an utterance is worth transcribing.

Why AST and not YAMNet
----------------------
``DESIGN.md`` allows either.  YAMNet means TensorFlow, and TF pins
``numpy < 2.1`` and ``protobuf 4.x``.  Inside our own venv that happens to be
satisfiable, but it makes the noise filter the one stage that dictates the
numpy version for everything downstream - and the pod's system interpreter,
which carries a newer torch and vllm, cannot host TF at all.  AST needs only
the torch already in use and reads the same AudioSet labels, so it works in
either interpreter and constrains nothing.  The policy below is unchanged
either way; only the backend differs.

The filter is deliberately timid
--------------------------------
The two mistakes are not symmetric.  Letting a cough through costs one wasted
Whisper call.  Dropping real speech loses a sentence from the meeting
permanently, and the participant never learns why.  So an utterance survives
unless the classifier is confident it contains no speech *and* something
non-speech scored higher.  A borderline utterance is always kept.

Layering
--------
``NoiseFilter``
    The policy: given scores, decide keep or drop.  Pure Python, unit tested
    without torch.

``AstClassifier``
    The model wrapper.  The only part that needs transformers, and the only
    part that cannot run on the Dev PC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol

import numpy as np

from server.config import (
    AST_MODEL_ID,
    NOISE_DEVICE,
    NOISE_MIN_SPEECH_SCORE,
    NOISE_REQUIRE_LOUDER_NOISE,
    NOISE_WINDOW_SECONDS,
    SAMPLE_RATE,
)

log = logging.getLogger(__name__)


class NoiseFilterError(RuntimeError):
    """Raised when the audio classifier cannot be loaded or used."""


#: AudioSet labels that mean "a person is talking". Anything here counts as
#: evidence to keep the utterance.
SPEECH_LABELS = frozenset({
    "Speech",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Whispering",
    "Shout",
    "Yell",
})

#: Labels that mean "this is the meeting-room noise we are here to remove".
#: Ambient classes such as "Inside, small room" are deliberately absent: they
#: score high under everything, speech included, so they prove nothing.
NOISE_LABELS = frozenset({
    "Typing",
    "Computer keyboard",
    "Typewriter",
    "Keyboard (musical)",
    "Mouse",
    "Click",
    "Keys jangling",
    "Cough",
    "Sneeze",
    "Throat clearing",
    "Sniff",
    "Breathing",
    "Laughter",
    "Chuckle, chortle",
    "Applause",
    "Clapping",
    "Door",
    "Cupboard open or close",
    "Drawer open or close",
    "Writing",
    "Rustle",
    "Tap",
    "Knock",
    "Silence",
    "Music",
    "Musical instrument",
    "Air conditioning",
    "Mechanical fan",
    "Hum",
})


@dataclass(frozen=True)
class Classification:
    """What the classifier made of one piece of audio."""

    speech_score: float
    noise_score: float
    noise_label: str = ""
    top: tuple[tuple[str, float], ...] = ()

    @property
    def top_label(self) -> str:
        return self.top[0][0] if self.top else ""

    def summary(self) -> str:
        parts = ", ".join(f"{label} {score:.2f}" for label, score in self.top[:3])
        return f"speech {self.speech_score:.2f} | {parts}"


@dataclass(frozen=True)
class Verdict:
    """The filter's decision about one utterance."""

    keep: bool
    reason: str
    classification: Classification


@dataclass
class NoiseStats:
    seen: int = 0
    dropped: int = 0
    dropped_labels: dict[str, int] = field(default_factory=dict)

    @property
    def kept(self) -> int:
        return self.seen - self.dropped

    def record(self, verdict: Verdict) -> None:
        self.seen += 1
        if not verdict.keep:
            self.dropped += 1
            label = verdict.classification.noise_label or "unknown"
            self.dropped_labels[label] = self.dropped_labels.get(label, 0) + 1


class Classifier(Protocol):
    """What :class:`NoiseFilter` needs; a stub satisfies it in the tests."""

    def classify(self, pcm: bytes) -> Classification:
        ...                                         # pragma: no cover


class NoiseFilter:
    """Decide whether an utterance is worth transcribing."""

    def __init__(
        self,
        classifier: Optional[Classifier] = None,
        min_speech_score: float = NOISE_MIN_SPEECH_SCORE,
        require_louder_noise: bool = NOISE_REQUIRE_LOUDER_NOISE,
    ) -> None:
        if not 0.0 <= min_speech_score <= 1.0:
            raise ValueError("min_speech_score must be between 0 and 1")
        self.classifier = classifier if classifier is not None else AstClassifier()
        self.min_speech_score = min_speech_score
        self.require_louder_noise = require_louder_noise
        self.stats = NoiseStats()

    def judge(self, pcm: bytes) -> Verdict:
        """Classify one utterance and apply the keep/drop policy."""
        if not pcm:
            verdict = Verdict(False, "empty audio", Classification(0.0, 0.0))
            self.stats.record(verdict)
            return verdict

        result = self.classifier.classify(pcm)
        verdict = self._decide(result)
        self.stats.record(verdict)
        if not verdict.keep:
            log.info("Dropped an utterance: %s (%s)", verdict.reason,
                     result.summary())
        return verdict

    def _decide(self, result: Classification) -> Verdict:
        if result.speech_score >= self.min_speech_score:
            return Verdict(True, "speech detected", result)
        if self.require_louder_noise and result.noise_score <= result.speech_score:
            # Nothing recognisable at all. Whisper gets to decide, because
            # silence costs one call while a lost sentence costs the meeting.
            return Verdict(True, "nothing conclusive, keeping it", result)
        label = result.noise_label or result.top_label or "non-speech"
        return Verdict(False, f"no speech, sounds like {label}", result)

    def reset(self) -> None:
        self.stats = NoiseStats()


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def aggregate(scores: np.ndarray, labels: list[str],
              wanted: Iterable[str]) -> tuple[float, str]:
    """Best score across windows for any of ``wanted``, and which label won.

    Max, not mean: a seven second utterance that is mostly keyboard but holds
    one clear sentence must still count as speech.
    """
    best_score = 0.0
    best_label = ""
    wanted = set(wanted)
    for index, label in enumerate(labels):
        if label not in wanted:
            continue
        score = float(scores[:, index].max())
        if score > best_score:
            best_score, best_label = score, label
    return best_score, best_label


def top_labels(scores: np.ndarray, labels: list[str],
               count: int = 5) -> tuple[tuple[str, float], ...]:
    peaks = scores.max(axis=0)
    order = np.argsort(peaks)[::-1][:count]
    return tuple((labels[i], float(peaks[i])) for i in order)


def split_windows(waveform: np.ndarray, window_samples: int) -> list[np.ndarray]:
    """Cut a waveform into classifier-sized windows, keeping the tail."""
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if waveform.size <= window_samples:
        return [waveform]
    return [
        waveform[offset : offset + window_samples]
        for offset in range(0, waveform.size, window_samples)
    ]


def fill_window(waveform: np.ndarray, window_samples: int) -> np.ndarray:
    """Repeat a short clip until it fills the classifier window.

    AST reads a fixed 10.24 s window and zero-pads anything shorter.  A 1.2 s
    utterance therefore arrives as 88% silence, and the evidence for speech is
    diluted by everything that is not there - which is how a perfectly good
    one-second sentence gets scored as noise and thrown away.

    Repeating the clip instead keeps the window full of the sound actually
    being judged.  The classifier is asked "what is this?", not "how long was
    it?", so tiling changes nothing it should care about.
    """
    if waveform.size == 0 or waveform.size >= window_samples:
        return waveform
    repeats = int(np.ceil(window_samples / waveform.size))
    return np.tile(waveform, repeats)[:window_samples]


class AstClassifier:
    """Audio Spectrogram Transformer, fine-tuned on AudioSet."""

    def __init__(self, model_id: str = "", device: str = "") -> None:
        try:
            import torch
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
        except ImportError as exc:                      # pragma: no cover
            raise NoiseFilterError(
                "transformers / torch are not installed. Run "
                "`python3.11 -m pip install -r server/requirements.txt`."
            ) from exc

        self._torch = torch
        self.model_id = model_id or AST_MODEL_ID
        chosen = device or NOISE_DEVICE
        if not chosen:
            chosen = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = chosen

        try:
            self._extractor = ASTFeatureExtractor.from_pretrained(self.model_id)
            model = ASTForAudioClassification.from_pretrained(self.model_id)
        except Exception as exc:                        # pragma: no cover
            raise NoiseFilterError(
                f"Could not load {self.model_id!r}. On a pod without internet "
                "access, download it once and point AST_MODEL_ID at the local "
                "directory."
            ) from exc

        # Half precision on the GPU: the verdict is a threshold comparison, so
        # the lost precision cannot change a decision that was not already a
        # coin flip.
        if self.device.startswith("cuda"):
            model = model.half()
        self._model = model.to(self.device).eval()

        self.labels = [
            model.config.id2label[i] for i in range(model.config.num_labels)
        ]
        self._check_labels()
        self.window_samples = int(NOISE_WINDOW_SECONDS * SAMPLE_RATE)
        log.info("AST ready on %s: %s, %d labels", self.device, self.model_id,
                 len(self.labels))

    def _check_labels(self) -> None:
        """Fail loudly if this checkpoint does not use AudioSet display names.

        A silently empty intersection would make every score zero, which the
        timid policy reads as "nothing conclusive" - so the filter would keep
        everything and look like it was working.
        """
        known = set(self.labels)
        missing_speech = SPEECH_LABELS - known
        if len(missing_speech) == len(SPEECH_LABELS):
            raise NoiseFilterError(
                f"{self.model_id!r} shares no speech label with AudioSet; "
                f"its first labels are {self.labels[:5]}"
            )
        if not NOISE_LABELS & known:
            raise NoiseFilterError(
                f"{self.model_id!r} shares no noise label with AudioSet"
            )
        if missing_speech:
            log.warning("Speech labels absent from this checkpoint: %s",
                        sorted(missing_speech))

    def classify(self, pcm: bytes) -> Classification:
        """Score one utterance. ``pcm`` is 16 kHz mono 16-bit, as everywhere."""
        waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        windows = split_windows(waveform, self.window_samples)
        scores = np.stack([
            self._score_window(fill_window(w, self.window_samples))
            for w in windows
        ])

        speech_score, _ = aggregate(scores, self.labels, SPEECH_LABELS)
        noise_score, noise_label = aggregate(scores, self.labels, NOISE_LABELS)
        return Classification(
            speech_score=speech_score,
            noise_score=noise_score,
            noise_label=noise_label,
            top=top_labels(scores, self.labels),
        )

    def _score_window(self, waveform: np.ndarray) -> np.ndarray:
        torch = self._torch
        features = self._extractor(
            waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        values = features["input_values"].to(self.device)
        if self.device.startswith("cuda"):
            values = values.half()
        with torch.no_grad():
            logits = self._model(input_values=values).logits
        # AudioSet is multi-label: several classes can be true at once, so each
        # gets its own sigmoid rather than competing in a softmax.
        return torch.sigmoid(logits.float())[0].cpu().numpy()

    @property
    def source(self) -> str:
        return f"{self.model_id} ({self.device})"

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE
