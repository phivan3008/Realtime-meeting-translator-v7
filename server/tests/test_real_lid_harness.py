"""Smoke tests for ``server/tests_real/test_real_lid.py``.

Drives the script, main() included, with a stubbed model, so a crash in it is
caught here rather than after a round trip through the pod.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import importlib.util
import math
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server.config import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH  # noqa: E402
from server.pipeline.lid import LanguageIdentifier  # noqa: E402
from server.pipeline.vad import VAD_FRAME_SAMPLES  # noqa: E402


def load_harness():
    path = ROOT / "server" / "tests_real" / "test_real_lid.py"
    spec = importlib.util.spec_from_file_location("real_lid_harness", path)
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
        self.calls = 0


class RoutingScorer:
    """One scorer for the whole run, telling the files apart by loudness."""

    source = "routing stub (cpu)"
    index_of = {"vi": 0, "ja": 1}

    def __init__(self, loud: str = "vi", quiet: str = "ja",
                 boundary: int = 4500, confidence: float = 0.9):
        self.loud = loud
        self.quiet = quiet
        self.boundary = boundary
        self.confidence = confidence
        self.calls = 0

    def scores(self, pcm: bytes) -> dict:
        self.calls += 1
        amplitude = float(np.mean(np.abs(np.frombuffer(pcm, dtype="<i2"))))
        winner = self.loud if amplitude > self.boundary else self.quiet
        loser = "ja" if winner == "vi" else "vi"
        return {winner: math.log(self.confidence),
                loser: math.log(1.0 - self.confidence)}


def write_wav(path: Path, seconds: float, amplitude: int = 6000) -> Path:
    samples = np.full(int(SAMPLE_RATE * seconds), amplitude, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())
    return path


def recording_of(expected: str, verdicts: list[tuple[str, float, dict]]):
    """Build a Recording from decided (lang, margin, probabilities) triples."""
    from server.pipeline.buffer import FinalizeReason, Utterance
    from server.pipeline.lid import LanguageDecision

    recording = harness.Recording(path=Path(f"{expected}.wav"),
                                  expected=expected, seconds=10.0,
                                  lid_seconds=0.01)
    for index, (lang, margin, probabilities) in enumerate(verdicts):
        recording.samples.append(harness.Sample(
            utterance=Utterance(index=index, pcm=bytes(32_000),
                                start_ms=index * 1000.0,
                                reason=FinalizeReason.PAUSE),
            decision=LanguageDecision(lang, 0.9, margin, "stub", probabilities),
        ))
    return recording


VI_RIGHT = ("vi", 0.8, {"vi": 0.9, "ja": 0.1})
VI_WRONG = ("ja", 0.6, {"vi": 0.2, "ja": 0.8})
VI_UNSURE = ("", 0.1, {"vi": 0.55, "ja": 0.45})


# ---------------------------------------------------------------------------
# Reading and measuring
# ---------------------------------------------------------------------------
def test_read_pcm_rejects_the_wrong_format(tmp_path):
    path = tmp_path / "wrong.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(22_050)
        wav.writeframes(bytes(4000))
    with pytest.raises(ValueError, match="22050 Hz"):
        harness.read_pcm(path)


def test_a_recording_is_cut_into_sentences_and_judged(tmp_path):
    path = write_wav(tmp_path / "vi.wav", 12.0)
    identifier = LanguageIdentifier(scorer=RoutingScorer(), min_duration_ms=0)
    recording = harness.measure(path, "vi", identifier, ScriptedVAD([0.9]))
    assert recording.samples
    assert recording.seconds > 0
    assert all(s.decision.lang_code == "vi" for s in recording.samples)


def test_a_silent_recording_gives_nothing_to_judge(tmp_path):
    path = write_wav(tmp_path / "quiet.wav", 4.0)
    identifier = LanguageIdentifier(scorer=RoutingScorer(), min_duration_ms=0)
    recording = harness.measure(path, "vi", identifier, ScriptedVAD([0.02]))
    assert recording.samples == []
    assert recording.lid_ratio == 0.0


# ---------------------------------------------------------------------------
# Sorting right from wrong
# ---------------------------------------------------------------------------
def test_a_recording_counts_right_wrong_and_undecided():
    recording = recording_of("vi", [VI_RIGHT, VI_WRONG, VI_UNSURE])
    assert len(recording.correct) == 1
    assert len(recording.wrong) == 1
    assert len(recording.unknown) == 1


def test_an_all_correct_recording_passes():
    report = harness.Report()
    harness.check_recording(recording_of("vi", [VI_RIGHT, VI_RIGHT]), report)
    assert report.failed == []


def test_the_other_language_being_chosen_is_caught():
    """The failure that matters: Whisper would be forced into the wrong one."""
    report = harness.Report()
    harness.check_recording(recording_of("vi", [VI_RIGHT, VI_WRONG]), report)
    assert "vi.wav is never called the other language" in [
        c.name for c in report.failed
    ]


def test_a_margin_that_skips_everything_is_caught():
    report = harness.Report()
    harness.check_recording(
        recording_of("vi", [VI_UNSURE, VI_UNSURE, VI_UNSURE]), report)
    assert "vi.wav is mostly decided rather than skipped" in [
        c.name for c in report.failed
    ]


def test_an_empty_recording_is_reported_and_not_judged_further():
    report = harness.Report()
    harness.check_recording(recording_of("vi", []), report)
    assert [c.name for c in report.checks] == ["vi.wav produced sentences to judge"]


# ---------------------------------------------------------------------------
# Margins
# ---------------------------------------------------------------------------
def test_truth_margins_are_signed_towards_the_right_answer():
    recording = recording_of("vi", [VI_RIGHT, VI_WRONG])
    margins = harness.truth_margins(recording)
    assert margins[0] == pytest.approx(0.8)
    assert margins[1] == pytest.approx(-0.6)


def test_a_clean_sweep_reports_the_smallest_correct_margin(capsys):
    report = harness.Report()
    harness.check_margins([recording_of("vi", [VI_RIGHT, VI_RIGHT])], report)
    assert report.failed == []
    assert "every sentence went the right way" in capsys.readouterr().out


def test_a_separable_mistake_prints_the_usable_margin_range(capsys):
    report = harness.Report()
    harness.check_margins(
        [recording_of("vi", [VI_RIGHT, VI_RIGHT, VI_RIGHT,
                             ("ja", 0.04, {"vi": 0.48, "ja": 0.52})])],
        report,
    )
    assert report.failed == []
    assert "separates them" in capsys.readouterr().out


def test_a_confident_mistake_is_caught():
    """A wrong answer more confident than the weakest right one is unfixable."""
    report = harness.Report()
    harness.check_margins(
        [recording_of("vi", [("vi", 0.1, {"vi": 0.55, "ja": 0.45}), VI_WRONG])],
        report,
    )
    assert "A margin exists that keeps the right answers and drops the wrong" in [
        c.name for c in report.failed
    ]


def test_mostly_wrong_is_caught():
    report = harness.Report()
    harness.check_margins([recording_of("vi", [VI_WRONG, VI_WRONG, VI_RIGHT])],
                          report)
    assert "The model prefers the right language more often than not" in [
        c.name for c in report.failed
    ]


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def run_main(monkeypatch, tmp_path, both: bool = True) -> int:
    vietnamese = write_wav(tmp_path / "vi.wav", 12.0, amplitude=6000)
    japanese = write_wav(tmp_path / "ja.wav", 12.0, amplitude=2000)
    monkeypatch.setattr(harness, "VoxLinguaClassifier",
                        lambda **k: RoutingScorer())
    monkeypatch.setattr(harness, "SileroVAD", lambda **k: ScriptedVAD([0.9]))
    argv = ["x", "--vi", str(vietnamese)]
    if both:
        argv += ["--ja", str(japanese)]
    monkeypatch.setattr(sys, "argv", argv)
    return harness.main()


def test_main_runs_and_passes_on_both_languages(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path) == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out
    assert "routing stub" in out


def test_main_fails_when_only_one_language_was_tested(monkeypatch, tmp_path,
                                                      capsys):
    assert run_main(monkeypatch, tmp_path, both=False) == 1
    assert "Both languages were tested" in capsys.readouterr().out


def test_main_needs_at_least_one_recording(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x"])
    assert harness.main() == 2
    assert "at least one" in capsys.readouterr().out


def test_main_reports_a_model_that_will_not_load(monkeypatch, tmp_path, capsys):
    def explode(**_kwargs):
        raise harness.LanguageIdError("Could not load 'speechbrain/lang-id-...'")

    monkeypatch.setattr(harness, "VoxLinguaClassifier", explode)
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--vi", str(write_wav(tmp_path / "vi.wav", 2.0))],
    )
    assert harness.main() == 2
    assert "Could not load" in capsys.readouterr().out
