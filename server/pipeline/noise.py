"""Deep Noise Filter - step 3 of the server pipeline.

``DESIGN.md`` section 3.3: classify the audio and drop what is not speech -
keyboard clatter, a cough, a chair scraping - before it reaches the stages
that cost real GPU time.

Silero already removes silence, but it is a *voice activity* detector, not a
sound classifier: it fires happily on a cough, a laugh, a door slam.  YAMNet
knows the difference, so it gets the last word on whether an utterance is
worth transcribing.

The filter is deliberately timid
--------------------------------
The two mistakes are not symmetric.  Letting a cough through costs one
wasted Whisper call.  Dropping real speech loses a sentence from the meeting
permanently, and the participant never learns why.  So an utterance survives
unless YAMNet is confident it contains no speech *and* something non-speech
scored higher.  A borderline utterance is always kept.

Layering
--------
``NoiseFilter``
    The policy: given scores, decide keep or drop.  Pure Python, unit tested
    without TensorFlow.

``YamnetClassifier``
    The model wrapper.  The only part that needs TensorFlow, and the only
    part that cannot run on the Dev PC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol

import numpy as np

from server.config import (
    NOISE_MIN_SPEECH_SCORE,
    NOISE_REQUIRE_LOUDER_NOISE,
    SAMPLE_RATE,
    YAMNET_HUB_URL,
    YAMNET_MIN_SAMPLES,
    YAMNET_MODEL_DIR,
)

log = logging.getLogger(__name__)


class NoiseFilterError(RuntimeError):
    """Raised when YAMNet cannot be loaded or used."""


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
    """What YAMNet made of one piece of audio."""

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
        self.classifier = classifier if classifier is not None else YamnetClassifier()
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
    """Best score across frames for any of ``wanted``, and which label won.

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


class YamnetClassifier:
    """YAMNet from TF Hub, pinned to the CPU."""

    def __init__(self, model_dir: str = "", url: str = YAMNET_HUB_URL) -> None:
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as exc:                      # pragma: no cover
            raise NoiseFilterError(
                "tensorflow / tensorflow-hub are not installed. Run "
                "`python3.11 -m pip install -r server/requirements.txt`."
            ) from exc

        # Whisper and vLLM need the whole H100. YAMNet is small enough that
        # the CPU runs it faster than the argument is worth having, and TF
        # would otherwise reserve VRAM the moment it sees the device.
        tf.config.set_visible_devices([], "GPU")
        self._tf = tf

        source = model_dir or YAMNET_MODEL_DIR or url
        try:
            self._model = hub.load(source)
        except Exception as exc:                        # pragma: no cover
            raise NoiseFilterError(
                f"Could not load YAMNet from {source!r}. On a pod without "
                "internet access, download the SavedModel once and point "
                "YAMNET_MODEL_DIR at the directory."
            ) from exc
        self.source = source
        self.labels = self._read_labels()

    def _read_labels(self) -> list[str]:
        import csv

        path = self._model.class_map_path().numpy().decode("utf-8")
        with open(path, newline="", encoding="utf-8") as handle:
            return [row["display_name"] for row in csv.DictReader(handle)]

    def classify(self, pcm: bytes) -> Classification:
        """Score one utterance. ``pcm`` is 16 kHz mono 16-bit, as everywhere."""
        waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if waveform.size < YAMNET_MIN_SAMPLES:
            # Shorter than one YAMNet frame; pad rather than get zero frames
            # back and have to guess.
            waveform = np.pad(waveform, (0, YAMNET_MIN_SAMPLES - waveform.size))

        scores, _embeddings, _spectrogram = self._model(waveform)
        scores = scores.numpy()
        speech_score, _ = aggregate(scores, self.labels, SPEECH_LABELS)
        noise_score, noise_label = aggregate(scores, self.labels, NOISE_LABELS)
        return Classification(
            speech_score=speech_score,
            noise_score=noise_score,
            noise_label=noise_label,
            top=top_labels(scores, self.labels),
        )

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE
