"""Overlap Resolver - step 4 of the server pipeline.

``DESIGN.md`` section 3.4: when two people talk over each other, favour the
louder voice and squash the quieter one, so the ASR has one voice to work
with instead of two.

What this stage is, and is not
------------------------------
A noise gate and a compressor cannot separate two overlapping speakers.
Nothing in this file is source separation, and pretending otherwise would
set up the ASR stage to be blamed for a problem that was never solved here.

What they *can* do is decide, moment by moment, that anything well below the
dominant voice is not worth passing on: the far-end participant bleeding
through someone's speaker, a second person murmuring under the main speaker,
room noise between words.  Whisper transcribes a clean dominant voice far
better than a muddy mix of two, so attenuating the quieter layer is worth
doing even though the quieter layer is not removed.

Everything is measured relative to the utterance's own loudness
---------------------------------------------------------------
An absolute threshold is useless here.  Meeting audio arrives at whatever
level the client's mixer happened to produce, and the same -30 dBFS is a
shouting match in one recording and a whisper in another.  So the gate opens
at a fixed distance *below this utterance's own speaking level*: "quiet" is
defined by whoever is dominating this sentence, not by a number chosen in
advance.

Two things had to be right for that to work, and the first version of this
file got both wrong.

*The level must ignore the pauses.*  An utterance carries the VAD's hangover
silence and every pause between words on purpose; on a real recording the
median 20 ms frame sat 28 dB below the voice.  A global RMS measures those
pauses as much as the speaker, so a threshold derived from it lands far too
low.

*The level must be measured in the same currency the gate uses.*  Pedalboard's
noise gate compares its threshold against the signal *peak*, not its RMS.  A
threshold set from an RMS-based level is a threshold in the wrong units: with
a second voice 20 dB down, the RMS-based gate attenuated it by 0.1 dB, while
the peak-based one attenuated it by 24 dB and left the dominant voice at
0.0 dB.  So the gate threshold comes from a high percentile of the *peak*
envelope, and the RMS-based level is kept only for reporting and for deciding
that an utterance is too quiet to bother with.

Layering
--------
``OverlapResolver``
    The measurement and the policy - what to gate, at what level, and when to
    leave the audio alone.  Pure Python, testable with a stub processor.

``PedalboardProcessor``
    The DSP itself.  Unlike the other model stages this one needs no GPU and
    no downloaded weights, so it is exercised for real in the unit tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from server.config import (
    OVERLAP_COMPRESSOR_ABOVE_DB,
    OVERLAP_COMPRESSOR_ATTACK_MS,
    OVERLAP_COMPRESSOR_RATIO,
    OVERLAP_COMPRESSOR_RELEASE_MS,
    OVERLAP_ENVELOPE_MS,
    OVERLAP_GATE_ATTACK_MS,
    OVERLAP_GATE_BELOW_DB,
    OVERLAP_GATE_RATIO,
    OVERLAP_GATE_RELEASE_MS,
    OVERLAP_LEVEL_PERCENTILE,
    OVERLAP_MIN_LEVEL_DBFS,
    SAMPLE_RATE,
)

log = logging.getLogger(__name__)

#: Returned instead of -inf so arithmetic and formatting stay well behaved.
SILENT_DBFS = -120.0


class OverlapError(RuntimeError):
    """Raised when the DSP backend cannot be loaded."""


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def dbfs(amplitude: float) -> float:
    """Convert a linear 0..1 amplitude to dBFS, floored rather than -inf."""
    if amplitude <= 0.0:
        return SILENT_DBFS
    return max(SILENT_DBFS, 20.0 * float(np.log10(amplitude)))


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return SILENT_DBFS
    return dbfs(float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))))


def peak_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return SILENT_DBFS
    return dbfs(float(np.max(np.abs(samples))))


def envelope_dbfs(samples: np.ndarray, sample_rate: int = SAMPLE_RATE,
                  window_ms: int = OVERLAP_ENVELOPE_MS) -> np.ndarray:
    """Short-term RMS of the signal, one value per window, in dBFS."""
    window = max(1, int(sample_rate * window_ms / 1000))
    count = samples.size // window
    if count == 0:
        return np.array([rms_dbfs(samples)], dtype=np.float64)
    frames = samples[: count * window].reshape(count, window).astype(np.float64)
    power = np.sqrt(np.mean(np.square(frames), axis=1))
    return np.maximum(20.0 * np.log10(np.maximum(power, 1e-12)), SILENT_DBFS)


def peak_envelope_dbfs(samples: np.ndarray, sample_rate: int = SAMPLE_RATE,
                       window_ms: int = OVERLAP_ENVELOPE_MS) -> np.ndarray:
    """Largest sample in each window, in dBFS - what a gate detector follows."""
    window = max(1, int(sample_rate * window_ms / 1000))
    count = samples.size // window
    if count == 0:
        return np.array([peak_dbfs(samples)], dtype=np.float64)
    frames = np.abs(samples[: count * window].reshape(count, window))
    peaks = np.max(frames, axis=1).astype(np.float64)
    return np.maximum(20.0 * np.log10(np.maximum(peaks, 1e-12)), SILENT_DBFS)


def speaking_level_dbfs(samples: np.ndarray, sample_rate: int = SAMPLE_RATE,
                        percentile: float = OVERLAP_LEVEL_PERCENTILE) -> float:
    """How loud this utterance is *while someone is speaking*, as RMS.

    The pauses are deliberately part of the utterance, so they must not be
    part of the measurement. Used for reporting and for the "too quiet to
    bother" decision, never for the gate threshold.
    """
    if samples.size == 0:
        return SILENT_DBFS
    return float(np.percentile(envelope_dbfs(samples, sample_rate), percentile))


def speaking_peak_dbfs(samples: np.ndarray, sample_rate: int = SAMPLE_RATE,
                       percentile: float = OVERLAP_LEVEL_PERCENTILE) -> float:
    """The same measurement in peak terms, which is what the gate compares to."""
    if samples.size == 0:
        return SILENT_DBFS
    return float(
        np.percentile(peak_envelope_dbfs(samples, sample_rate), percentile)
    )


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Resolved:
    """One utterance after the resolver has had a look at it."""

    pcm: bytes
    level_before_dbfs: float
    level_after_dbfs: float
    gate_threshold_dbfs: float
    shaped: bool
    reason: str

    @property
    def gain_db(self) -> float:
        """How much the speaking level moved.

        Measured on the speaking level rather than the overall RMS, so that
        gating the pauses - the stage working correctly - does not read as
        damage to the voice.
        """
        return self.level_after_dbfs - self.level_before_dbfs


@dataclass
class OverlapStats:
    seen: int = 0
    shaped: int = 0
    skipped: int = 0
    total_reduction_db: float = 0.0
    skipped_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def mean_reduction_db(self) -> float:
        return self.total_reduction_db / self.shaped if self.shaped else 0.0

    def record(self, result: Resolved) -> None:
        self.seen += 1
        if result.shaped:
            self.shaped += 1
            self.total_reduction_db += result.gain_db
        else:
            self.skipped += 1
            self.skipped_reasons[result.reason] = (
                self.skipped_reasons.get(result.reason, 0) + 1
            )


class Processor(Protocol):
    """The DSP backend; a stub satisfies this in the tests."""

    def process(self, samples: np.ndarray, sample_rate: int,
                gate_threshold_db: float,
                compressor_threshold_db: float) -> np.ndarray:
        ...                                         # pragma: no cover


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
class OverlapResolver:
    """Gate and compress an utterance around its own dominant voice."""

    def __init__(
        self,
        processor: Optional[Processor] = None,
        gate_below_db: float = OVERLAP_GATE_BELOW_DB,
        compressor_above_db: float = OVERLAP_COMPRESSOR_ABOVE_DB,
        min_level_dbfs: float = OVERLAP_MIN_LEVEL_DBFS,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if gate_below_db <= 0:
            raise ValueError("gate_below_db must be positive")
        if compressor_above_db <= 0:
            raise ValueError("compressor_above_db must be positive")
        self.processor = (
            processor if processor is not None else PedalboardProcessor()
        )
        self.gate_below_db = gate_below_db
        self.compressor_above_db = compressor_above_db
        self.min_level_dbfs = min_level_dbfs
        self.sample_rate = sample_rate
        self.stats = OverlapStats()

    def resolve(self, pcm: bytes) -> Resolved:
        """Shape one utterance, or explain why it was left alone."""
        samples = pcm_to_float(pcm)
        level = speaking_level_dbfs(samples, self.sample_rate)

        if samples.size == 0:
            return self._skip(pcm, level, "empty audio")
        if level <= self.min_level_dbfs:
            # Nothing to favour over anything else. Gating this would only
            # remove the little that is there.
            return self._skip(pcm, level, "too quiet to shape")

        # Both thresholds are peak-referenced, because that is what the gate
        # and the compressor detectors actually measure.
        peak_level = speaking_peak_dbfs(samples, self.sample_rate)
        gate_threshold = peak_level - self.gate_below_db
        compressor_threshold = peak_level + self.compressor_above_db
        shaped = self.processor.process(
            samples,
            sample_rate=self.sample_rate,
            gate_threshold_db=gate_threshold,
            compressor_threshold_db=compressor_threshold,
        )
        result = Resolved(
            pcm=float_to_pcm(shaped),
            level_before_dbfs=level,
            level_after_dbfs=speaking_level_dbfs(shaped, self.sample_rate),
            gate_threshold_dbfs=gate_threshold,
            shaped=True,
            reason="gated and compressed",
        )
        self.stats.record(result)
        return result

    def _skip(self, pcm: bytes, level: float, reason: str) -> Resolved:
        result = Resolved(
            pcm=pcm,
            level_before_dbfs=level,
            level_after_dbfs=level,
            gate_threshold_dbfs=SILENT_DBFS,
            shaped=False,
            reason=reason,
        )
        self.stats.record(result)
        return result

    def reset(self) -> None:
        self.stats = OverlapStats()


# ---------------------------------------------------------------------------
# The DSP
# ---------------------------------------------------------------------------
class PedalboardProcessor:
    """Noise gate then compressor, rebuilt per utterance around its level."""

    def __init__(self) -> None:
        try:
            from pedalboard import Compressor, NoiseGate, Pedalboard
        except ImportError as exc:                      # pragma: no cover
            raise OverlapError(
                "pedalboard is not installed. Run "
                "`python3.11 -m pip install -r server/requirements.txt`."
            ) from exc
        self._Compressor = Compressor
        self._NoiseGate = NoiseGate
        self._Pedalboard = Pedalboard

    def process(self, samples: np.ndarray, sample_rate: int,
                gate_threshold_db: float,
                compressor_threshold_db: float) -> np.ndarray:
        # The board is cheap to build and the thresholds change with every
        # utterance, so there is nothing to gain by keeping one around.
        board = self._Pedalboard([
            self._NoiseGate(
                threshold_db=gate_threshold_db,
                ratio=OVERLAP_GATE_RATIO,
                attack_ms=OVERLAP_GATE_ATTACK_MS,
                release_ms=OVERLAP_GATE_RELEASE_MS,
            ),
            self._Compressor(
                threshold_db=compressor_threshold_db,
                ratio=OVERLAP_COMPRESSOR_RATIO,
                attack_ms=OVERLAP_COMPRESSOR_ATTACK_MS,
                release_ms=OVERLAP_COMPRESSOR_RELEASE_MS,
            ),
        ])
        return board(np.ascontiguousarray(samples, dtype=np.float32),
                     sample_rate=float(sample_rate))
