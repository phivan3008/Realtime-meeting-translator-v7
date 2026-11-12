r"""Unit tests for ``gpu_server/whisper_server.py``.

faster-whisper is replaced by a fake that records how it was called, so these
run on the Dev PC. Whether the real model decodes is for
``server/tests_real/test_real_remote_models.py``.

Run with::

    .venv\Scripts\python.exe -m pytest gpu_server/tests -v
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

GPU_SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GPU_SERVER))

import whisper_server  # noqa: E402
from whisper_server import (  # noqa: E402
    DecodeOptions,
    Engine,
    RequestError,
    create_app,
    pcm_to_samples,
)


class FakeModel:
    """Answers like faster-whisper's WhisperModel.transcribe."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, dict]] = []

    def transcribe(self, samples, **kwargs):
        self.calls.append((samples, kwargs))
        words = [SimpleNamespace(word=" xin", start=0.1, end=0.3,
                                 probability=0.9),
                 SimpleNamespace(word=" chào", start=0.3, end=0.6,
                                 probability=0.8)]
        segment = SimpleNamespace(text=" xin chào", start=0.0, end=0.7,
                                  avg_logprob=-0.2, no_speech_prob=0.01,
                                  compression_ratio=1.1, words=words)
        info = SimpleNamespace(language="vi", language_probability=0.97)

        def generate():
            yield segment

        return generate(), info


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def client(model):
    from fastapi.testclient import TestClient

    engine = Engine(model, "large-v3", "cuda", "float16")
    return TestClient(create_app(engine))


def pcm(seconds: float = 0.5) -> bytes:
    samples = (np.sin(np.arange(int(16000 * seconds)) / 10) * 8000)
    return samples.astype("<i2").tobytes()


# ---------------------------------------------------------------------------
# The file must stand alone on the pod
# ---------------------------------------------------------------------------
def test_it_imports_nothing_from_the_pipeline():
    """The pod may not hold pipeline code; this file is copied there alone."""
    tree = ast.parse((GPU_SERVER / "whisper_server.py").read_text("utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert not modules & {"server", "common", "client"}, modules


def test_its_audio_contract_matches_the_pipeline():
    sys.path.insert(0, str(GPU_SERVER.parent))
    from common.protocol import SAMPLE_RATE, SAMPLE_WIDTH

    assert whisper_server.SAMPLE_RATE == SAMPLE_RATE
    assert whisper_server.SAMPLE_WIDTH == SAMPLE_WIDTH


def test_its_api_version_is_the_one_the_pipeline_speaks():
    sys.path.insert(0, str(GPU_SERVER.parent))
    from server.pipeline.asr_remote import API_VERSION

    assert whisper_server.API_VERSION == API_VERSION


# ---------------------------------------------------------------------------
# Options and audio
# ---------------------------------------------------------------------------
def test_options_default_to_a_greedy_decode_without_a_prompt():
    options = DecodeOptions.from_query({})
    assert options.beam_size == 1
    assert options.language is None
    assert options.prompt is None
    assert options.condition_on_previous_text is False


def test_an_empty_prompt_is_no_prompt():
    """Whisper treats "" as a prompt it was given."""
    assert DecodeOptions.from_query({"prompt": ""}).prompt is None


def test_options_are_parsed():
    options = DecodeOptions.from_query({
        "language": "ja", "beam_size": "5", "prompt": "議事録",
        "temperature": "0", "condition_on_previous_text": "0",
        "no_speech_threshold": "0.6", "log_prob_threshold": "-1.0",
        "compression_ratio_threshold": "2.4"})
    assert options == DecodeOptions(
        language="ja", beam_size=5, prompt="議事録", temperature=0.0,
        condition_on_previous_text=False, no_speech_threshold=0.6,
        log_prob_threshold=-1.0, compression_ratio_threshold=2.4)


@pytest.mark.parametrize("query", [
    {"beam": "5"},                       # misspelt
    {"beam_size": "0"},
    {"beam_size": "11"},
    {"beam_size": "five"},
    {"condition_on_previous_text": "maybe"},
    {"no_speech_threshold": "high"},
])
def test_a_bad_option_is_refused(query):
    with pytest.raises(RequestError):
        DecodeOptions.from_query(query)


def test_pcm_becomes_the_samples_the_pipeline_used_to_decode():
    raw = np.array([-32768, -1, 0, 1, 32767], dtype="<i2").tobytes()
    samples = pcm_to_samples(raw)
    assert samples.dtype == np.float32
    np.testing.assert_array_equal(
        samples, np.array([-32768, -1, 0, 1, 32767], np.float32) / 32768.0)


@pytest.mark.parametrize("body", [b"", b"\x00", bytes(2 * 16000 * 31)],
                         ids=["empty", "odd", "too-long"])
def test_bad_audio_is_refused(body):
    with pytest.raises(RequestError):
        pcm_to_samples(body)


# ---------------------------------------------------------------------------
# The decode call
# ---------------------------------------------------------------------------
def test_faster_whisper_gets_the_call_the_pipeline_used_to_make(model):
    engine = Engine(model, "large-v3", "cuda", "float16")
    engine.decode(np.zeros(8000, np.float32), DecodeOptions(
        language="vi", beam_size=5, prompt="FPT",
        no_speech_threshold=0.6, log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4))
    _samples, kwargs = model.calls[-1]
    assert kwargs == {
        "language": "vi", "task": "transcribe", "beam_size": 5,
        "temperature": 0.0, "initial_prompt": "FPT",
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6, "log_prob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "without_timestamps": False, "word_timestamps": True,
        "vad_filter": False,
    }


def test_every_score_and_word_comes_back(model):
    engine = Engine(model, "large-v3", "cuda", "float16")
    answer = engine.decode(np.zeros(8000, np.float32), DecodeOptions())
    assert answer["language"] == "vi"
    assert answer["language_probability"] == pytest.approx(0.97)
    (segment,) = answer["segments"]
    assert segment["text"] == " xin chào"
    assert segment["avg_logprob"] == pytest.approx(-0.2)
    assert segment["no_speech_prob"] == pytest.approx(0.01)
    assert segment["compression_ratio"] == pytest.approx(1.1)
    assert [word["text"] for word in segment["words"]] == [" xin", " chào"]
    assert segment["words"][0] == {"text": " xin", "start": 0.1,
                                   "end": 0.3, "probability": 0.9}
    assert engine.decodes == 1


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def test_health_names_the_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "large-v3"
    assert body["device"] == "cuda"
    assert body["compute_type"] == "float16"
    assert body["api_version"] == whisper_server.API_VERSION


def test_decode_over_http(client, model):
    response = client.post("/v1/decode?language=vi&beam_size=5",
                           content=pcm(),
                           headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 200, response.text
    assert response.json()["segments"][0]["text"] == " xin chào"
    samples, kwargs = model.calls[-1]
    assert len(samples) == 8000
    assert kwargs["language"] == "vi"
    assert kwargs["beam_size"] == 5


def test_a_bad_request_is_a_400_and_decodes_nothing(client, model):
    response = client.post("/v1/decode?beam=5", content=pcm())
    assert response.status_code == 400
    assert "unknown options" in response.json()["error"]
    response = client.post("/v1/decode", content=b"\x00")
    assert response.status_code == 400
    assert model.calls == []


def test_a_model_failure_is_a_500_with_the_reason(model):
    from fastapi.testclient import TestClient

    def broken(*_args, **_kwargs):
        raise RuntimeError("CUDA failed with error out of memory")

    model.transcribe = broken
    client = TestClient(create_app(Engine(model, "large-v3", "cuda",
                                          "float16")))
    response = client.post("/v1/decode", content=pcm())
    assert response.status_code == 500
    assert "out of memory" in response.json()["error"]
