"""Unit tests for the Overlap Resolver.

Unlike the other model stages, the DSP here needs no GPU and no downloaded
weights, so the real pedalboard backend is exercised directly. The policy is
tested separately with a stub, because "did it decide to shape this?" is a
different question from "did the filter do what a filter does?".

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import SAMPLE_RATE
from server.pipeline.overlap import (
    SILENT_DBFS,
    OverlapResolver,
    PedalboardProcessor,
    dbfs,
    envelope_dbfs,
    float_to_pcm,
    pcm_to_float,
    peak_dbfs,
    peak_envelope_dbfs,
    rms_dbfs,
    speaking_level_dbfs,
    speaking_peak_dbfs,
)


class StubProcessor:
    """Records the thresholds it was handed and returns the audio unchanged."""

    def __init__(self, gain: float = 1.0):
        self.gain = gain
        self.calls: list[dict] = []

    def process(self, samples, sample_rate, gate_threshold_db,
                compressor_threshold_db):
        self.calls.append({
            "samples": samples,
            "sample_rate": sample_rate,
            "gate_threshold_db": gate_threshold_db,
            "compressor_threshold_db": compressor_threshold_db,
        })
        return samples * self.gain


def tone(seconds: float, amplitude: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def pcm(samples: np.ndarray) -> bytes:
    return float_to_pcm(samples)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def test_dbfs_of_full_scale_is_zero():
    assert dbfs(1.0) == pytest.approx(0.0)


def test_dbfs_of_half_amplitude_is_about_minus_six():
    assert dbfs(0.5) == pytest.approx(-6.02, abs=0.01)


def test_silence_reports_a_floor_rather_than_negative_infinity():
    """-inf would poison every average and every format string downstream."""
    assert dbfs(0.0) == SILENT_DBFS
    assert rms_dbfs(np.zeros(100, dtype=np.float32)) == SILENT_DBFS
    assert peak_dbfs(np.array([], dtype=np.float32)) == SILENT_DBFS


def test_rms_of_a_sine_is_three_db_under_its_peak():
    samples = tone(0.5, 0.5)
    assert peak_dbfs(samples) == pytest.approx(-6.02, abs=0.1)
    assert rms_dbfs(samples) == pytest.approx(-9.03, abs=0.1)


def test_pcm_round_trips_through_float():
    original = np.array([0.0, 0.5, -0.5, 0.999], dtype=np.float32)
    restored = pcm_to_float(float_to_pcm(original))
    assert np.max(np.abs(restored - original)) < 1e-4


def test_float_to_pcm_clips_rather_than_wrapping():
    """Wrapping would turn a loud voice into white noise."""
    loud = np.array([2.0, -2.0], dtype=np.float32)
    assert np.frombuffer(float_to_pcm(loud), dtype="<i2").tolist() == [32767, -32767]


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
def test_thresholds_are_set_relative_to_this_utterance():
    """An absolute threshold would be a whisper in one meeting and a shout in another."""
    stub = StubProcessor()
    resolver = OverlapResolver(processor=stub, gate_below_db=12.0,
                               compressor_above_db=6.0)
    samples = tone(0.5, 0.5)
    # Peak-referenced, because that is what the gate detector compares against.
    peak = speaking_peak_dbfs(samples)
    resolver.resolve(pcm(samples))
    call = stub.calls[0]
    assert call["gate_threshold_db"] == pytest.approx(peak - 12.0, abs=0.1)
    assert call["compressor_threshold_db"] == pytest.approx(peak + 6.0, abs=0.1)
    assert call["sample_rate"] == SAMPLE_RATE


def test_a_loud_utterance_gets_a_higher_gate_than_a_quiet_one():
    stub = StubProcessor()
    resolver = OverlapResolver(processor=stub)
    resolver.resolve(pcm(tone(0.5, 0.6)))
    resolver.resolve(pcm(tone(0.5, 0.06)))
    assert stub.calls[0]["gate_threshold_db"] > stub.calls[1]["gate_threshold_db"]


def test_a_shaped_utterance_reports_what_changed():
    resolver = OverlapResolver(processor=StubProcessor(gain=0.5))
    result = resolver.resolve(pcm(tone(0.5, 0.5)))
    assert result.shaped is True
    assert result.reason == "gated and compressed"
    assert result.gain_db == pytest.approx(-6.02, abs=0.2)


def test_audio_too_quiet_to_shape_is_passed_through_untouched():
    """Gating a near-silent utterance would only eat what little is there."""
    stub = StubProcessor()
    resolver = OverlapResolver(processor=stub, min_level_dbfs=-40.0)
    original = pcm(tone(0.5, 0.001))
    result = resolver.resolve(original)
    assert result.shaped is False
    assert result.reason == "too quiet to shape"
    assert result.pcm == original
    assert stub.calls == []


def test_empty_audio_is_passed_through_untouched():
    stub = StubProcessor()
    result = OverlapResolver(processor=stub).resolve(b"")
    assert result.shaped is False
    assert result.reason == "empty audio"
    assert result.pcm == b""
    assert stub.calls == []


def test_the_resolver_rejects_nonsense_settings():
    with pytest.raises(ValueError, match="gate_below_db"):
        OverlapResolver(processor=StubProcessor(), gate_below_db=0)
    with pytest.raises(ValueError, match="compressor_above_db"):
        OverlapResolver(processor=StubProcessor(), compressor_above_db=-1)


def test_stats_separate_what_was_shaped_from_what_was_skipped():
    resolver = OverlapResolver(processor=StubProcessor(gain=0.5),
                               min_level_dbfs=-40.0)
    resolver.resolve(pcm(tone(0.5, 0.5)))
    resolver.resolve(pcm(tone(0.5, 0.001)))
    resolver.resolve(b"")
    assert resolver.stats.seen == 3
    assert resolver.stats.shaped == 1
    assert resolver.stats.skipped == 2
    assert resolver.stats.skipped_reasons == {
        "too quiet to shape": 1, "empty audio": 1
    }
    assert resolver.stats.mean_reduction_db == pytest.approx(-6.02, abs=0.2)


def test_mean_reduction_is_zero_when_nothing_was_shaped():
    resolver = OverlapResolver(processor=StubProcessor())
    assert resolver.stats.mean_reduction_db == 0.0


def test_reset_clears_the_counters():
    resolver = OverlapResolver(processor=StubProcessor())
    resolver.resolve(pcm(tone(0.5, 0.5)))
    resolver.reset()
    assert resolver.stats.seen == 0


# ---------------------------------------------------------------------------
# The real DSP
# ---------------------------------------------------------------------------
def two_voices(loud: float, quiet: float, seconds: float = 2.0) -> np.ndarray:
    """A dominant voice with a quieter one underneath, as in a crosstalk moment."""
    return tone(seconds, loud, freq=180.0) + tone(seconds, quiet, freq=400.0)


def test_the_real_board_leaves_the_dominant_voice_alone():
    resolver = OverlapResolver(processor=PedalboardProcessor())
    samples = tone(2.0, 0.5)
    result = resolver.resolve(pcm(samples))
    assert result.shaped is True
    # A steady voice sits above its own gate, so only the compressor touches it.
    assert result.gain_db == pytest.approx(0.0, abs=3.0)


def test_the_real_board_attenuates_audio_under_the_dominant_voice():
    """The point of the stage: what is well below the speaker gets squashed."""
    resolver = OverlapResolver(processor=PedalboardProcessor())
    loud = tone(1.0, 0.5)
    quiet = tone(1.0, 0.01)                 # 34 dB down: bleed, not a speaker
    mixed = np.concatenate([loud, quiet])

    result = resolver.resolve(pcm(mixed))
    shaped = pcm_to_float(result.pcm)
    half = loud.size

    before = rms_dbfs(mixed[half:])
    after = rms_dbfs(shaped[half:])
    assert after < before - 6.0, "the quiet passage should have been gated"
    # ... while the loud passage survives.
    assert rms_dbfs(shaped[:half]) > rms_dbfs(mixed[:half]) - 3.0


def test_the_real_board_keeps_the_audio_the_same_length():
    """Anything else would break every timestamp downstream."""
    resolver = OverlapResolver(processor=PedalboardProcessor())
    original = pcm(tone(1.5, 0.4))
    assert len(resolver.resolve(original).pcm) == len(original)


def test_the_real_board_returns_finite_samples():
    resolver = OverlapResolver(processor=PedalboardProcessor())
    result = resolver.resolve(pcm(two_voices(0.5, 0.05)))
    assert np.all(np.isfinite(pcm_to_float(result.pcm)))


def test_the_real_board_does_not_invent_signal_in_silence():
    resolver = OverlapResolver(processor=PedalboardProcessor(),
                               min_level_dbfs=-120.0)
    samples = np.concatenate([tone(1.0, 0.5), np.zeros(SAMPLE_RATE, np.float32)])
    shaped = pcm_to_float(resolver.resolve(pcm(samples)).pcm)
    assert peak_dbfs(shaped[SAMPLE_RATE:]) < -60.0


# ---------------------------------------------------------------------------
# Measuring the speaker rather than the pauses
# ---------------------------------------------------------------------------
def speech_like(seconds: float, amplitude: float, duty: float = 0.4) -> np.ndarray:
    """Bursts separated by silence, the way an utterance actually looks."""
    samples = tone(seconds, amplitude)
    window = SAMPLE_RATE // 10
    for start in range(0, samples.size, window):
        if (start // window) % 10 >= duty * 10:
            samples[start : start + window] = 0.0
    return samples


def test_the_speaking_level_ignores_the_pauses():
    """Global RMS measures the silence the VAD deliberately kept."""
    samples = speech_like(4.0, 0.5)
    assert speaking_level_dbfs(samples) > rms_dbfs(samples) + 3.0


def test_the_speaking_level_of_a_steady_tone_is_its_rms():
    samples = tone(2.0, 0.5)
    assert speaking_level_dbfs(samples) == pytest.approx(rms_dbfs(samples), abs=0.5)


def test_the_peak_level_sits_above_the_rms_level_for_a_sine():
    samples = tone(2.0, 0.5)
    assert speaking_peak_dbfs(samples) == pytest.approx(
        speaking_level_dbfs(samples) + 3.0, abs=0.5
    )


def test_both_levels_track_a_change_in_loudness():
    loud, quiet = tone(2.0, 0.5), tone(2.0, 0.05)
    assert speaking_level_dbfs(loud) - speaking_level_dbfs(quiet) == pytest.approx(
        20.0, abs=0.5
    )
    assert speaking_peak_dbfs(loud) - speaking_peak_dbfs(quiet) == pytest.approx(
        20.0, abs=0.5
    )


def test_the_envelopes_have_one_value_per_window():
    samples = tone(1.0, 0.5)
    assert envelope_dbfs(samples, SAMPLE_RATE, window_ms=20).size == 50
    assert peak_envelope_dbfs(samples, SAMPLE_RATE, window_ms=20).size == 50


def test_audio_shorter_than_one_window_still_gets_a_level():
    tiny = tone(0.005, 0.5)
    assert envelope_dbfs(tiny, SAMPLE_RATE, window_ms=20).size == 1
    assert speaking_level_dbfs(tiny) > SILENT_DBFS
    assert speaking_peak_dbfs(tiny) > SILENT_DBFS


def test_empty_audio_has_no_level():
    empty = np.array([], dtype=np.float32)
    assert speaking_level_dbfs(empty) == SILENT_DBFS
    assert speaking_peak_dbfs(empty) == SILENT_DBFS


def test_the_real_board_squashes_a_second_voice_well_under_the_first():
    """The measurement that caught the units bug: RMS-based gave -0.1 dB here."""
    resolver = OverlapResolver(processor=PedalboardProcessor())
    loud = speech_like(2.0, 0.5)
    quiet = speech_like(2.0, 0.05)          # 20 dB down, a plausible bleed
    mixed = np.concatenate([loud, quiet])

    shaped = pcm_to_float(resolver.resolve(pcm(mixed)).pcm)
    edge = loud.size
    loud_change = speaking_level_dbfs(shaped[:edge]) - speaking_level_dbfs(loud)
    quiet_change = speaking_level_dbfs(shaped[edge:]) - speaking_level_dbfs(quiet)
    assert quiet_change < -6.0, f"second voice only moved {quiet_change:.1f} dB"
    assert loud_change > -3.0, f"dominant voice lost {loud_change:.1f} dB"
