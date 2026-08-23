"""Smoke tests for the ``server/tests_real/test_real_buffer.py`` harness.

Same reasoning as the other harness tests: the real script only ever runs on
the pod, so a crash in its reporting code costs a round trip through the
user. Here it is driven with a scripted model and synthetic audio.

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
from server.pipeline.buffer import (  # noqa: E402
    BufferStats,
    FinalizeReason,
    PartialWindow,
    Utterance,
)
from server.pipeline.vad import VAD_FRAME_SAMPLES  # noqa: E402


def load_harness():
    path = ROOT / "server" / "tests_real" / "test_real_buffer.py"
    spec = importlib.util.spec_from_file_location("real_buffer_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = load_harness()


class ScriptedVAD:
    def __init__(self, probabilities):
        self.script = list(probabilities)
        self.calls = 0
        self.resets = 0

    def probability(self, frame: np.ndarray) -> float:
        assert frame.shape[-1] == VAD_FRAME_SAMPLES
        value = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return float(value)

    def reset(self) -> None:
        self.resets += 1


def write_wav(path: Path, seconds: float) -> Path:
    samples = np.full(int(SAMPLE_RATE * seconds), 6000, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())
    return path


def utterance(index: int, start_ms: float, duration_ms: float,
              reason: FinalizeReason = FinalizeReason.PAUSE,
              continues: bool = False) -> Utterance:
    samples = int(duration_ms * SAMPLE_RATE / 1000)
    return Utterance(
        index=index,
        pcm=np.full(samples, 1000, dtype="<i2").tobytes(),
        start_ms=start_ms,
        reason=reason,
        continues_previous=continues,
    )


def replayed(utterances, forwarded=b"", partials=None):
    result = harness.Replay(path=Path("fake.wav"))
    result.utterances = list(utterances)
    result.forwarded = bytearray(
        forwarded or b"".join(u.pcm for u in utterances)
    )
    result.partials = list(partials or [])
    result.buffer_stats = BufferStats()
    return result


# ---------------------------------------------------------------------------
# Replay over synthetic audio
# ---------------------------------------------------------------------------
def test_a_talking_recording_yields_utterances(tmp_path):
    path = write_wav(tmp_path / "speech.wav", 20.0)
    result = harness.replay(path, ScriptedVAD([0.9]))
    assert result.utterances
    assert result.audio_seconds == pytest.approx(20.0)
    assert result.buffer_stats.utterances == len(result.utterances)


def test_a_long_monologue_is_cut_by_the_max_duration_rule(tmp_path):
    path = write_wav(tmp_path / "monologue.wav", 20.0)
    result = harness.replay(path, ScriptedVAD([0.9]))
    reasons = {u.reason for u in result.utterances}
    assert FinalizeReason.MAX_DURATION in reasons
    assert all(u.duration_ms <= 7_000 + 40 for u in result.utterances)


def test_a_silent_recording_yields_nothing(tmp_path):
    path = write_wav(tmp_path / "quiet.wav", 5.0)
    result = harness.replay(path, ScriptedVAD([0.02]))
    assert result.utterances == []
    assert result.forwarded == b""


def test_the_replay_resets_the_model_first(tmp_path):
    vad = ScriptedVAD([0.02])
    harness.replay(write_wav(tmp_path / "a.wav", 1.0), vad)
    assert vad.resets == 1


def test_read_pcm_rejects_the_wrong_format(tmp_path):
    path = tmp_path / "wrong.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(1000))
    with pytest.raises(ValueError, match="2 ch"):
        harness.read_pcm(path)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def test_a_clean_partition_passes():
    result = replayed([utterance(0, 0, 500), utterance(1, 2_000, 500)])
    report = harness.Report()
    harness.check_partition(result, report)
    assert report.failed == []


def test_lost_audio_between_sentences_is_caught():
    pieces = [utterance(0, 0, 500), utterance(1, 2_000, 500)]
    result = replayed(pieces,
                      forwarded=b"".join(u.pcm for u in pieces) + bytes(64))
    report = harness.Report()
    harness.check_partition(result, report)
    assert [c.name for c in report.failed] == [
        "Utterances are exactly the speech the VAD forwarded",
        "No audio is transcribed twice",
    ]


def test_an_over_long_utterance_is_caught():
    result = replayed([utterance(0, 0, 9_000)])
    report = harness.Report()
    harness.check_durations(result, report)
    assert "No utterance outstays the max duration" in [
        c.name for c in report.failed
    ]


def test_a_cut_that_lands_far_from_the_limit_is_reported():
    """Cutting way short every time means the search window is misbehaving."""
    result = replayed([utterance(0, 0, 6_000, FinalizeReason.MAX_DURATION)])
    report = harness.Report()
    harness.check_durations(result, report)
    assert "Length cuts land inside the search window, not on the raw limit" in [
        c.name for c in report.failed
    ]


def test_overlapping_utterances_are_caught():
    result = replayed([utterance(0, 0, 1_000), utterance(1, 500, 1_000)])
    report = harness.Report()
    harness.check_timestamps(result, report)
    assert "Utterances never overlap" in [c.name for c in report.failed]


def test_a_broken_continuation_join_is_caught():
    result = replayed([
        utterance(0, 0, 1_000),
        utterance(1, 5_000, 1_000, FinalizeReason.PAUSE, continues=True),
    ])
    report = harness.Report()
    harness.check_timestamps(result, report)
    assert "A continued sentence joins end to start with no gap" in [
        c.name for c in report.failed
    ]


def test_healthy_timestamps_pass():
    result = replayed([
        utterance(0, 0, 1_000),
        utterance(1, 1_000, 1_000, FinalizeReason.MAX_DURATION, continues=True),
    ])
    report = harness.Report()
    harness.check_timestamps(result, report)
    assert report.failed == []


def test_missing_partials_are_caught():
    report = harness.Report()
    harness.check_partials(replayed([utterance(0, 0, 1_000)]), report)
    assert [c.name for c in report.failed] == [
        "Partial windows are produced while somebody talks"
    ]


def test_a_stalled_partial_cadence_is_caught():
    partials = [
        PartialWindow(index=0, pcm=bytes(6_400), start_ms=0),
        PartialWindow(index=0, pcm=bytes(6_400 * 40), start_ms=0),
    ]
    result = replayed([utterance(0, 0, 2_000)], partials=partials)
    report = harness.Report()
    harness.check_partials(result, report)
    assert "Partials keep to the configured cadence" in [
        c.name for c in report.failed
    ]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def test_each_utterance_is_written_as_a_playable_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "OUTPUT_DIR", tmp_path / "output")
    result = replayed([
        utterance(0, 0, 500),
        utterance(1, 500, 500, FinalizeReason.MAX_DURATION, continues=True),
    ])
    directory = harness.save_utterances(result)
    names = sorted(p.name for p in directory.glob("*.wav"))
    assert names == ["000_pause.wav", "001_max_duration_cont.wav"]
    with wave.open(str(directory / names[0]), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnchannels() == CHANNELS


def test_describe_prints_without_crashing(capsys):
    result = replayed([utterance(0, 0, 500)])
    harness.describe(result)
    out = capsys.readouterr().out
    assert "Utterances:" in out
    assert "pause" in out
