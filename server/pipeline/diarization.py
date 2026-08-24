"""Speaker Diarization - step 5 of the server pipeline.

``DESIGN.md`` section 3.5: take a voiceprint of each sentence, match it
against the voices heard so far by cosine similarity, and label it
``Speaker_01``, ``Speaker_02`` and so on.

Online identification, not offline diarization
----------------------------------------------
The usual pyannote pipeline reads a whole recording and clusters it once,
which is the right tool for a file and the wrong one for a meeting that is
still happening.  Here every sentence has to be labelled the moment the
buffer manager commits it, before the next one arrives, and the set of
speakers is discovered as the meeting goes.

So this keeps a running centroid per speaker and compares each new
voiceprint to them.  Above the threshold it is that person and their
centroid moves a little towards the new sample; below it, a new speaker is
born.  That is a greedy, order-dependent algorithm - a late-arriving voice
cannot retroactively split an earlier mistake - and it is the price of
labelling in real time.

An honest unknown beats a confident guess
-----------------------------------------
A sentence too short to embed reliably is labelled ``Speaker_unknown``
rather than assigned to whoever spoke last.  Continuity sounds like a
sensible heuristic until you notice that short interjections - "ah, I see",
"right" - usually come from whoever is *listening*, so guessing the previous
speaker would be wrong precisely where it is most tempting.

Layering
--------
``SpeakerRegistry``
    The centroids and the matching rule.  Pure numpy, unit tested without a
    model.

``EcapaEmbedder``
    The voiceprint model.  Needs torch and the checkpoint.

``SpeakerIdentifier``
    Wires them together and owns the too-short policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from server.config import (
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    SPEAKER_CACHE_DIR,
    SPEAKER_CENTROID_MOMENTUM,
    SPEAKER_DEVICE,
    SPEAKER_EMBEDDING_MODEL,
    SPEAKER_MATCH_THRESHOLD,
    SPEAKER_MAX_SPEAKERS,
    SPEAKER_MIN_DURATION_MS,
    SPEAKER_UNKNOWN,
)

log = logging.getLogger(__name__)


class DiarizationError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two voiceprints, 0 when either has no length."""
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 0.0:
        return 0.0
    return float(np.dot(a, b) / norm)


def label_for(index: int) -> str:
    """``Speaker_01``, ``Speaker_02``, ... as named in DESIGN.md."""
    return f"Speaker_{index:02d}"


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
@dataclass
class Speaker:
    label: str
    centroid: np.ndarray
    utterances: int = 1


@dataclass(frozen=True)
class Assignment:
    """Who the registry thinks said this, and how sure it is."""

    speaker_id: str
    similarity: float
    is_new: bool
    reason: str


class SpeakerRegistry:
    """Running voiceprints for the speakers heard so far."""

    def __init__(
        self,
        match_threshold: float = SPEAKER_MATCH_THRESHOLD,
        max_speakers: int = SPEAKER_MAX_SPEAKERS,
        momentum: float = SPEAKER_CENTROID_MOMENTUM,
    ) -> None:
        if not -1.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold must be a cosine, between -1 and 1")
        if max_speakers < 1:
            raise ValueError("max_speakers must be at least 1")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be between 0 and 1")
        self.match_threshold = match_threshold
        self.max_speakers = max_speakers
        self.momentum = momentum
        self.speakers: list[Speaker] = []

    def assign(self, embedding: np.ndarray) -> Assignment:
        """Label a voiceprint, creating a new speaker if nobody matches."""
        best, best_score = None, -1.0
        for speaker in self.speakers:
            score = cosine_similarity(embedding, speaker.centroid)
            if score > best_score:
                best, best_score = speaker, score

        if best is not None and best_score >= self.match_threshold:
            self._update(best, embedding)
            return Assignment(best.label, best_score, False, "matched a known voice")

        if len(self.speakers) >= self.max_speakers:
            # Refusing to invent speaker 13 is not politeness; a meeting that
            # has produced that many means the threshold is wrong, and adding
            # more would only bury the evidence.
            if best is not None:
                self._update(best, embedding)
                return Assignment(
                    best.label, best_score, False,
                    f"speaker limit {self.max_speakers} reached, "
                    "assigned to the closest voice",
                )

        speaker = Speaker(label=label_for(len(self.speakers) + 1),
                          centroid=np.array(embedding, dtype=np.float64, copy=True))
        self.speakers.append(speaker)
        return Assignment(speaker.label, best_score if best else 0.0, True,
                          "a voice not heard before")

    def _update(self, speaker: Speaker, embedding: np.ndarray) -> None:
        speaker.centroid = (
            self.momentum * speaker.centroid
            + (1.0 - self.momentum) * np.asarray(embedding, dtype=np.float64)
        )
        speaker.utterances += 1

    @property
    def count(self) -> int:
        return len(self.speakers)

    def reset(self) -> None:
        self.speakers = []


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class Embedder(Protocol):
    """What :class:`SpeakerIdentifier` needs; a stub satisfies it in tests."""

    def embed(self, pcm: bytes) -> np.ndarray:
        ...                                         # pragma: no cover


class EcapaEmbedder:
    """ECAPA-TDNN voiceprints, straight from SpeechBrain.

    ``DESIGN.md`` names pyannote.audio for this stage, and pyannote does ship
    a wrapper - ``PretrainedSpeakerEmbedding`` - around this very checkpoint.
    It cannot be used here: pyannote 4.0.7 calls SpeechBrain with ``token``,
    ``huggingface_cache_dir`` and ``revision``, and SpeechBrain 1.1.0 accepts
    none of the three, so the wrapper raises before it ever loads a model.
    pyannote declares no version bound on SpeechBrain either, so no resolver
    and no ``pip check`` can see the mismatch - only running it does.

    Calling SpeechBrain directly loads the same weights through one less
    layer, and the layer removed is the broken one.
    """

    def __init__(self, model_id: str = "", device: str = "") -> None:
        try:
            import torch
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError as exc:                      # pragma: no cover
            raise DiarizationError(
                "speechbrain / torch are not installed. Run "
                "`python3.11 -m pip install -r server/requirements.txt`."
            ) from exc

        self._torch = torch
        self.model_id = model_id or SPEAKER_EMBEDDING_MODEL
        chosen = device or SPEAKER_DEVICE
        if not chosen:
            chosen = "cuda" if torch.cuda.is_available() else "cpu"
        # SpeechBrain parses this itself and wants an index: a bare "cuda"
        # makes it warn and fall back to device 0.
        self.device = "cuda:0" if chosen == "cuda" else chosen
        try:
            self._model = EncoderClassifier.from_hparams(
                source=self.model_id,
                savedir=str(Path(SPEAKER_CACHE_DIR) / self.model_id.replace("/", "_")),
                run_opts={"device": self.device},
            )
        except Exception as exc:                        # pragma: no cover
            raise DiarizationError(
                f"Could not load {self.model_id!r}: {exc}"
            ) from exc
        self.warmup()
        log.info("Speaker embedding ready on %s: %s", self.device, self.model_id)

    def warmup(self, seconds: float = 1.0) -> None:
        """Pay the first-inference cost now, not on the meeting's first word."""
        silence = np.zeros(int(seconds * SAMPLE_RATE), dtype="<i2")
        self.embed(silence.tobytes())

    def embed(self, pcm: bytes) -> np.ndarray:
        """One voiceprint for one utterance of 16 kHz mono 16-bit PCM."""
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        waveform = self._torch.from_numpy(samples).unsqueeze(0)   # (1, samples)
        with self._torch.no_grad():
            embedding = self._model.encode_batch(waveform.to(self.device))
        return np.asarray(embedding.squeeze().cpu(), dtype=np.float64).reshape(-1)

    @property
    def source(self) -> str:
        return f"{self.model_id} ({self.device})"


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
@dataclass
class DiarizationStats:
    seen: int = 0
    identified: int = 0
    unknown: int = 0
    new_speakers: int = 0
    per_speaker: dict[str, int] = field(default_factory=dict)

    def record(self, assignment: Assignment) -> None:
        self.seen += 1
        if assignment.speaker_id == SPEAKER_UNKNOWN:
            self.unknown += 1
        else:
            self.identified += 1
            if assignment.is_new:
                self.new_speakers += 1
        self.per_speaker[assignment.speaker_id] = (
            self.per_speaker.get(assignment.speaker_id, 0) + 1
        )


class SpeakerIdentifier:
    """Label each committed sentence with who said it."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        registry: Optional[SpeakerRegistry] = None,
        min_duration_ms: int = SPEAKER_MIN_DURATION_MS,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.embedder = embedder if embedder is not None else EcapaEmbedder()
        self.registry = registry if registry is not None else SpeakerRegistry()
        self.min_duration_ms = min_duration_ms
        self.sample_rate = sample_rate
        self.stats = DiarizationStats()

    def identify(self, pcm: bytes) -> Assignment:
        duration_ms = len(pcm) / SAMPLE_WIDTH / self.sample_rate * 1000.0
        if duration_ms < self.min_duration_ms:
            assignment = Assignment(
                SPEAKER_UNKNOWN, 0.0, False,
                f"only {duration_ms:.0f} ms, too short for a voiceprint",
            )
            self.stats.record(assignment)
            return assignment

        embedding = self.embedder.embed(pcm)
        if embedding.size == 0 or not np.all(np.isfinite(embedding)):
            assignment = Assignment(SPEAKER_UNKNOWN, 0.0, False,
                                    "the model returned no usable voiceprint")
            self.stats.record(assignment)
            return assignment

        assignment = self.registry.assign(embedding)
        self.stats.record(assignment)
        return assignment

    def reset(self) -> None:
        """A new meeting starts with nobody known."""
        self.registry.reset()
        self.stats = DiarizationStats()
