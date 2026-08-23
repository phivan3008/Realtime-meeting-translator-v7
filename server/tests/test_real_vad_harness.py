"""Smoke tests for the ``server/tests_real/test_real_vad.py`` harness itself.

The real test only ever runs on the GPU pod, where a crash costs a full
round trip through the user.  These tests drive the same harness on the Dev
PC with a scripted model and synthetic WAV files, so a typo in the reporting
or timestamp code is caught here instead of there.

They do not validate VAD quality - that is what the real test is for.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server.config import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH  # noqa: E402
from server.pipeline.vad import VAD_FRAME_SAMPLES, VADEvent  # noqa: E402


def load_harness():
    """Import the real-test script by path; it is not an importable package."""
    path = ROOT / "server" / "tests_real" / "test_real_vad.py"
    spec = importlib.util.spec_from_file_location("real_vad_harness", path)
    module = importlib.util.module_from_spec(spec)
    # @dataclass looks the defining module up in sys.modules, so register it
    # before executing the body.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = load_harness()


class ScriptedVAD:
    """Same stub as the unit tests, with the TimedVAD interface."""

    def __init__(self, probabilities):
        self.script = list(probabilities)
        self.calls = 0
        self.resets = 0
        self.latencies_ms: list[float] = []

    def probability(self, frame: np.ndarray) -> float:
        value = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        self.latencies_ms.append(0.1)
        return float(value)

    def reset(self) -> None:
        self.resets += 1


def write_wav(path: Path, seconds: float, rate: int = SAMPLE_RATE,
              channels: int = CHANNELS) -> Path:
    samples = np.zeros(int(rate * seconds) * channels, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return path


# ---------------------------------------------------------------------------
# WAV format gate
# ---------------------------------------------------------------------------
def test_read_pcm_accepts_the_client_format(tmp_path):
    path = write_wav(tmp_path / "ok.wav", 1.0)
    assert len(harness.read_pcm(path)) == SAMPLE_RATE * SAMPLE_WIDTH


def test_read_pcm_rejects_the_wrong_sample_rate(tmp_path):
    path = write_wav(tmp_path / "48k.wav", 0.5, rate=48_000)
    with pytest.raises(ValueError, match="48000 Hz"):
        harness.read_pcm(path)


def test_read_pcm_rejects_stereo(tmp_path):
    path = write_wav(tmp_path / "stereo.wav", 0.5, channels=2)
    with pytest.raises(ValueError, match="2 ch"):
        harness.read_pcm(path)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def test_replay_on_silence_produces_no_segments(tmp_path):
    path = write_wav(tmp_path / "quiet.wav", 3.0)
    result = harness.replay(path, ScriptedVAD([0.02]))
    assert result.segments == 0
    assert result.gated_pcm == b""
    assert result.dropped_ratio == 1.0
    assert result.audio_seconds == pytest.approx(3.0)


def test_replay_resets_the_model_between_files(tmp_path):
    vad = ScriptedVAD([0.02])
    path = write_wav(tmp_path / "a.wav", 1.0)
    harness.replay(path, vad)
    harness.replay(path, vad)
    assert vad.resets == 2


def test_replay_on_speech_produces_a_closed_segment(tmp_path):
    path = write_wav(tmp_path / "speech.wav", 3.0)
    # Speak for ~1 s (31 frames), then go quiet for the rest.
    result = harness.replay(path, ScriptedVAD([0.9] * 31 + [0.02]))
    assert result.segments == 1
    assert [e.kind for e in result.events] == [
        VADEvent.SPEECH_START,
        VADEvent.SPEECH_END,
    ]
    assert result.gated_seconds > 0.0
    assert 0.0 < result.dropped_ratio < 1.0


def test_replay_closes_a_segment_left_open_at_the_end_of_the_file(tmp_path):
    path = write_wav(tmp_path / "talking.wav", 2.0)
    result = harness.replay(path, ScriptedVAD([0.9]))
    assert [e.kind for e in result.events][-1] is VADEvent.SPEECH_END


def test_replay_covers_the_whole_file(tmp_path):
    path = write_wav(tmp_path / "full.wav", 2.0)
    result = harness.replay(path, ScriptedVAD([0.02]))
    expected_frames = int(SAMPLE_RATE * 2.0) // VAD_FRAME_SAMPLES
    assert result.frames == expected_frames


# ---------------------------------------------------------------------------
# Checks and reporting
# ---------------------------------------------------------------------------
def test_pairs_zips_starts_with_ends(tmp_path):
    path = write_wav(tmp_path / "two.wav", 4.0)
    result = harness.replay(
        path, ScriptedVAD([0.9] * 10 + [0.02] * 20 + [0.9] * 10 + [0.02])
    )
    segments = harness.pairs(result.events)
    assert len(segments) == 2
    assert all(end.at_ms > start.at_ms for start, end in segments)


def test_all_checks_pass_on_a_well_formed_speech_replay(tmp_path):
    path = write_wav(tmp_path / "speech.wav", 4.0)
    vad = ScriptedVAD([0.9] * 60 + [0.02])
    result = harness.replay(path, vad)
    report = harness.Report()
    harness.check_latency(vad, report)
    harness.check_speech(result, report)
    harness.check_timestamps(result, report)
    assert report.failed == []


def test_checks_fail_loudly_when_silence_triggers_a_segment(tmp_path):
    path = write_wav(tmp_path / "quiet.wav", 3.0)
    result = harness.replay(path, ScriptedVAD([0.9]))     # a false trigger
    report = harness.Report()
    harness.check_silence(result, report)
    assert [c.name for c in report.failed] == [
        "Quiet recording triggers no speech segment",
        "Quiet recording is dropped before the heavy stages",
    ]


# ---------------------------------------------------------------------------
# Latency reporting
# ---------------------------------------------------------------------------
class FixedLatencyVAD(ScriptedVAD):
    """A stub whose per-frame timings are dictated, not measured."""

    def __init__(self, latencies_ms):
        super().__init__([0.02])
        self.latencies_ms = list(latencies_ms)

    def probability(self, frame: np.ndarray) -> float:
        self.calls += 1
        return 0.02


def latency_report(latencies_ms) -> harness.Report:
    report = harness.Report()
    harness.check_latency(FixedLatencyVAD(latencies_ms), report)
    return report


def test_percentile_uses_nearest_rank():
    values = [float(i) for i in range(1, 101)]
    assert harness.percentile(values, 0.95) == 95.0
    assert harness.percentile(values, 0.99) == 99.0
    assert harness.percentile([], 0.5) == 0.0
    assert harness.percentile([7.0], 0.99) == 7.0


def test_latency_check_passes_on_a_healthy_run():
    assert latency_report([0.3] * 1000).failed == []


def test_a_single_warmup_spike_does_not_fail_the_run():
    """The first inference allocates buffers; 140 ms once is not a stall."""
    report = latency_report([138.0] + [0.3] * 999)
    assert report.failed == []


def test_a_stall_longer_than_a_client_chunk_fails():
    report = latency_report([0.3] * 500 + [250.0] + [0.3] * 499)
    assert [c.name for c in report.failed] == ["No stall longer than one client chunk"]


def test_persistently_slow_frames_fail_even_when_none_is_a_stall():
    """40 ms per frame never stalls, but it cannot keep up with 32 ms of audio."""
    report = latency_report([40.0] * 1000)
    names = [c.name for c in report.failed]
    assert "Steady-state frames stay inside the frame budget" in names
    assert "VAD is faster than real time (mean)" in names


def test_latency_check_reports_the_slowest_frame_position(capsys):
    latency_report([0.3] * 400 + [50.0] + [0.3] * 599)
    out = capsys.readouterr().out
    assert "slowest frame is #400" in out
    assert "mid-stream" in out


def test_latency_check_calls_a_first_frame_spike_warm_up(capsys):
    latency_report([138.0] + [0.3] * 999)
    assert "model warm-up" in capsys.readouterr().out


def test_latency_check_handles_a_run_with_no_frames():
    report = harness.Report()
    harness.check_latency(FixedLatencyVAD([]), report)
    assert [c.name for c in report.failed] == ["VAD inference latency measured"]


def test_describe_prints_without_crashing(tmp_path, capsys):
    path = write_wav(tmp_path / "speech.wav", 3.0)
    result = harness.replay(path, ScriptedVAD([0.9] * 31 + [0.02]))
    harness.describe(result)
    out = capsys.readouterr().out
    assert "speech.wav" in out
    assert "segments:" in out


def test_save_gated_writes_a_playable_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "OUTPUT_DIR", tmp_path / "output")
    path = write_wav(tmp_path / "speech.wav", 3.0)
    result = harness.replay(path, ScriptedVAD([0.9] * 31 + [0.02]))
    written = harness.save_gated(result)
    with wave.open(str(written), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnchannels() == CHANNELS
        assert wav.getnframes() == len(result.gated_pcm) // SAMPLE_WIDTH
