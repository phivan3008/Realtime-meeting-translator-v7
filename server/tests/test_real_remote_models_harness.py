r"""Unit tests for ``server/tests_real/test_real_remote_models.py``.

The pod is the real ``gpu_server/whisper_server.py`` app with a fake model,
so this proves the script runs and judges - not that the pod is fast.

Run with::

    .venv\Scripts\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gpu_server"))
sys.path.insert(0, str(Path(__file__).parent))

import whisper_server  # noqa: E402
from server.pipeline.asr_remote import RemoteWhisperDecoder  # noqa: E402
from test_asr_remote_unit import AppConnection, FakeModel  # noqa: E402


def load_harness():
    path = ROOT / "server" / "tests_real" / "test_real_remote_models.py"
    spec = importlib.util.spec_from_file_location("real_remote_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = load_harness()


def pod(model, device="cuda", compute_type="float16"):
    from fastapi.testclient import TestClient

    client = TestClient(whisper_server.create_app(
        whisper_server.Engine(model, "large-v3", device, compute_type)))

    def build(base_url="", model_id=""):
        return RemoteWhisperDecoder(
            base_url=base_url or "http://pod:8002", model_id=model_id,
            connection_factory=lambda: AppConnection(client, []))

    return build


def write_wav(path: Path, seconds: float) -> Path:
    samples = (np.sin(np.arange(int(16000 * seconds)) / 9) * 5000)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(samples.astype("<i2").tobytes())
    return path


def test_it_passes_against_a_healthy_pod(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(harness, "RemoteWhisperDecoder", pod(FakeModel()))
    wav = write_wav(tmp_path / "speech.wav", 8.0)
    assert harness.main(["--wav", str(wav), "--rounds", "3",
                         "--skip-translate"]) == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out
    assert "words with timestamps" in out


def test_it_fails_a_pod_decoding_on_cpu(monkeypatch, capsys):
    monkeypatch.setattr(harness, "RemoteWhisperDecoder",
                        pod(FakeModel(), device="cpu", compute_type="int8"))
    assert harness.main(["--rounds", "2", "--skip-translate"]) == 1
    assert "float16" in capsys.readouterr().out


def test_it_fails_an_answer_that_changes(monkeypatch, capsys):
    model = FakeModel()
    original = model.transcribe
    count = {"n": 0}

    def drifting(samples, **kwargs):
        count["n"] += 1
        segments, info = original(samples, **kwargs)
        segment = next(segments)
        segment = SimpleNamespace(**{**vars(segment),
                                     "avg_logprob": -0.1 * count["n"]})
        return iter([segment]), info

    model.transcribe = drifting
    monkeypatch.setattr(harness, "RemoteWhisperDecoder", pod(model))
    assert harness.main(["--rounds", "2", "--skip-translate"]) == 1
    assert "same answer twice" in capsys.readouterr().out


def test_it_reports_a_pod_that_is_not_there(capsys):
    assert harness.main(["--asr-url", "http://127.0.0.1:9", "--rounds", "1",
                         "--skip-translate"]) == 1
    assert "No Whisper server" in capsys.readouterr().out


def test_percentile():
    assert harness.percentile([], 0.95) == 0.0
    assert harness.percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert harness.percentile([1, 2, 3, 4, 5], 0.95) == 5
