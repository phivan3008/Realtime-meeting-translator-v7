"""Smoke tests for the ``server/tests_real/test_real_noise.py`` harness.

The real script needs the pod's torch stack; this drives everything around
the model with stubs, including main() itself, so a crash in the script is
caught here rather than after a round trip through the user.

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
from server.pipeline.noise import Classification, NoiseFilter  # noqa: E402
from server.pipeline.vad import VAD_FRAME_SAMPLES  # noqa: E402


def load_harness():
    path = ROOT / "server" / "tests_real" / "test_real_noise.py"
    spec = importlib.util.spec_from_file_location("real_noise_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = load_harness()


class StubClassifier:
    def __init__(self, result: Classification):
        self.result = result
        self.calls = 0

    def classify(self, pcm: bytes) -> Classification:
        self.calls += 1
        return self.result


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


SPEECHY = Classification(speech_score=0.9, noise_score=0.05,
                         top=(("Speech", 0.9),))
KEYBOARD = Classification(speech_score=0.01, noise_score=0.8,
                          noise_label="Computer keyboard",
                          top=(("Computer keyboard", 0.8),))


def write_wav(path: Path, seconds: float, rate: int = SAMPLE_RATE) -> Path:
    samples = np.full(int(rate * seconds), 5000, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return path


# ---------------------------------------------------------------------------
# Reading and classifying a file
# ---------------------------------------------------------------------------
def test_read_pcm_rejects_the_wrong_sample_rate(tmp_path):
    path = write_wav(tmp_path / "wrong.wav", 1.0, rate=8_000)
    with pytest.raises(ValueError, match="8000 Hz"):
        harness.read_pcm(path)


def test_a_speech_file_is_reported_as_kept(tmp_path):
    path = write_wav(tmp_path / "speech.wav", 2.0)
    result = harness.classify_file(path, NoiseFilter(classifier=StubClassifier(SPEECHY)))
    assert result.kept is True
    assert result.audio_seconds == pytest.approx(2.0)
    assert result.classification.speech_score == pytest.approx(0.9)
    assert result.classify_seconds >= 0.0


def test_a_noise_file_is_reported_as_dropped(tmp_path):
    path = write_wav(tmp_path / "keyboard.wav", 2.0)
    result = harness.classify_file(path, NoiseFilter(classifier=StubClassifier(KEYBOARD)))
    assert result.kept is False
    assert "Computer keyboard" in result.reason


def test_the_classify_ratio_is_relative_to_the_audio_length(tmp_path):
    path = write_wav(tmp_path / "speech.wav", 4.0)
    result = harness.classify_file(path, NoiseFilter(classifier=StubClassifier(SPEECHY)))
    result.classify_seconds = 0.4
    assert result.classify_ratio == pytest.approx(0.1)


def test_an_empty_result_has_a_harmless_ratio():
    assert harness.FileResult(path=Path("x.wav")).classify_ratio == 0.0


# ---------------------------------------------------------------------------
# The whole chain
# ---------------------------------------------------------------------------
def test_speech_survives_the_full_pipeline(tmp_path):
    path = write_wav(tmp_path / "speech.wav", 6.0)
    total, kept = harness.run_pipeline(
        path, ScriptedVAD([0.9]), NoiseFilter(classifier=StubClassifier(SPEECHY))
    )
    assert total >= 1
    assert kept == total


def test_noise_leaves_nothing_for_the_asr(tmp_path):
    path = write_wav(tmp_path / "keyboard.wav", 6.0)
    total, kept = harness.run_pipeline(
        path, ScriptedVAD([0.9]), NoiseFilter(classifier=StubClassifier(KEYBOARD))
    )
    assert total >= 1
    assert kept == 0


def test_a_silent_file_produces_no_utterances_at_all(tmp_path):
    path = write_wav(tmp_path / "quiet.wav", 3.0)
    total, kept = harness.run_pipeline(
        path, ScriptedVAD([0.02]), NoiseFilter(classifier=StubClassifier(SPEECHY))
    )
    assert (total, kept) == (0, 0)


def test_the_pipeline_resets_the_vad_between_files(tmp_path):
    vad = ScriptedVAD([0.02])
    path = write_wav(tmp_path / "a.wav", 1.0)
    harness.run_pipeline(path, vad, NoiseFilter(classifier=StubClassifier(SPEECHY)))
    harness.run_pipeline(path, vad, NoiseFilter(classifier=StubClassifier(SPEECHY)))
    assert vad.resets == 2


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def test_describe_prints_both_the_verdict_and_the_cost(capsys):
    result = harness.FileResult(path=Path("keyboard.wav"), audio_seconds=2.0,
                                classify_seconds=0.05, classification=KEYBOARD,
                                kept=False, reason="no speech, sounds like x",
                                utterances=2, utterances_kept=0)
    harness.describe(result)
    out = capsys.readouterr().out
    assert "DROP" in out
    assert "keyboard.wav" in out
    assert "0 survived" in out


def test_the_report_separates_passes_from_failures():
    report = harness.Report()
    report.add("good", True)
    report.add("bad", False, "why")
    assert [c.name for c in report.failed] == ["bad"]


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
class StubAst:
    """Enough of AstClassifier for main() to run end to end."""

    labels = ["Speech", "Computer keyboard"] + [f"Class {i}" for i in range(500)]
    source = "stub (cpu)"

    def __init__(self, result: Classification, **_kwargs):
        self.result = result

    def classify(self, pcm: bytes) -> Classification:
        return self.result


def run_main(monkeypatch, tmp_path, result: Classification, noise_result=None,
             with_noise: bool = True) -> int:
    """Drive the real script exactly as the pod would, with stubs for models.

    Importing the module is not enough: a broken main() - a stale model class,
    a renamed flag - only shows up when it is actually called, and on the pod
    that costs a round trip.
    """
    speech = write_wav(tmp_path / "speech.wav", 6.0)
    noise = write_wav(tmp_path / "keyboard.wav", 6.0)

    results = {"speech.wav": result, "keyboard.wav": noise_result or KEYBOARD}

    class Router(StubAst):
        def __init__(self, **_kwargs):
            self.result = result

        def classify(self, pcm: bytes) -> Classification:
            return self.current

    router = Router()

    def make_classifier(**_kwargs):
        return router

    monkeypatch.setattr(harness, "AstClassifier", make_classifier)
    monkeypatch.setattr(harness, "SileroVAD", lambda *a, **k: ScriptedVAD([0.9]))

    original_classify_file = harness.classify_file
    original_pipeline = harness.run_pipeline

    def routed_classify_file(path, filt):
        router.current = results[path.name]
        return original_classify_file(path, filt)

    def routed_pipeline(path, vad, filt):
        router.current = results[path.name]
        return original_pipeline(path, vad, filt)

    monkeypatch.setattr(harness, "classify_file", routed_classify_file)
    monkeypatch.setattr(harness, "run_pipeline", routed_pipeline)

    argv = ["test_real_noise.py", "--speech", str(speech)]
    if with_noise:
        argv += ["--noise", str(noise)]
    monkeypatch.setattr(sys, "argv", argv)
    return harness.main()


def test_main_runs_and_passes_on_healthy_inputs(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, SPEECHY) == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out
    assert "stub (cpu)" in out


def test_main_fails_when_speech_is_thrown_away(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, KEYBOARD) == 1
    assert "RESULT: FAIL" in capsys.readouterr().out


def test_main_runs_without_any_noise_file(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, SPEECHY, with_noise=False) == 0
    assert "the drop side is unproven" in capsys.readouterr().out


def test_main_reports_a_model_that_will_not_load(monkeypatch, tmp_path, capsys):
    def explode(**_kwargs):
        raise harness.NoiseFilterError("Could not load 'MIT/ast-...'")

    monkeypatch.setattr(harness, "AstClassifier", explode)
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--speech", str(write_wav(tmp_path / "speech.wav", 2.0))],
    )
    assert harness.main() == 2
    assert "Could not load" in capsys.readouterr().out
