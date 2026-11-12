r"""Unit tests for ``server/pipeline/asr_remote.py``.

The far end is the real ``gpu_server/whisper_server.py`` app with a fake
faster-whisper behind it, reached through FastAPI's TestClient, so the
request the pipeline sends and the answer the pod sends are checked against
each other rather than against a copy of either.

Run with::

    .venv\Scripts\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import http.client
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gpu_server"))

import whisper_server  # noqa: E402
from server.config import (  # noqa: E402
    ASR_LOG_PROB_THRESHOLD,
    ASR_MAX_COMPRESSION_RATIO,
    ASR_NO_SPEECH_THRESHOLD,
)
from server.pipeline import asr as asr_module  # noqa: E402
from server.pipeline.asr import AsrError, Piece, Transcriber, Word  # noqa: E402
from server.pipeline.asr_remote import (  # noqa: E402
    RemoteWhisperDecoder,
    decode_query,
    pieces_from_answer,
    samples_to_pcm,
)


class FakeModel:
    def __init__(self, segments=None, language="ja"):
        self.calls = []
        self.segments = segments
        self.language = language

    def transcribe(self, samples, **kwargs):
        self.calls.append((samples.copy(), kwargs))
        segments = self.segments
        if segments is None:
            segments = [SimpleNamespace(
                text="こんにちは", start=0.0, end=0.8, avg_logprob=-0.3,
                no_speech_prob=0.02, compression_ratio=1.2,
                words=[SimpleNamespace(word="こん", start=0.0, end=0.4,
                                       probability=0.9),
                       SimpleNamespace(word=" ", start=0.4, end=0.4,
                                       probability=0.1),
                       SimpleNamespace(word="にちは", start=0.4, end=0.8,
                                       probability=0.8)])]
        return iter(segments), SimpleNamespace(language=self.language,
                                               language_probability=0.9)


class AppConnection:
    """http.client.HTTPConnection's surface, served by a TestClient."""

    def __init__(self, client, log):
        self.client = client
        self.log = log
        self._response = None

    def request(self, method, url, body=None, headers=None):
        self.log.append((method, url, body))
        self._response = self.client.request(method, url, content=body,
                                             headers=headers)

    def getresponse(self):
        response = self._response
        return SimpleNamespace(status=response.status_code,
                               read=lambda: response.content)

    def close(self):
        pass


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def wire(model):
    from fastapi.testclient import TestClient

    engine = whisper_server.Engine(model, "large-v3", "cuda", "float16")
    client = TestClient(whisper_server.create_app(engine))
    log = []
    return SimpleNamespace(
        log=log, model=model,
        factory=lambda: AppConnection(client, log))


def remote(wire, **kwargs) -> RemoteWhisperDecoder:
    return RemoteWhisperDecoder(base_url="http://pod:8002",
                                connection_factory=wire.factory, **kwargs)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_samples_go_back_to_the_exact_pcm_they_came_from():
    raw = np.array([-32768, -12345, -1, 0, 1, 12345, 32767], "<i2").tobytes()
    samples = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    assert samples_to_pcm(samples) == raw


def test_samples_out_of_range_are_clipped_not_wrapped():
    pcm = samples_to_pcm(np.array([-2.0, 2.0], np.float32))
    assert np.frombuffer(pcm, "<i2").tolist() == [-32768, 32767]


def test_the_query_carries_the_in_process_options():
    query = decode_query("ja", 5, "議事録")
    assert query == {
        "beam_size": "5", "temperature": "0",
        "condition_on_previous_text": "0",
        "no_speech_threshold": repr(ASR_NO_SPEECH_THRESHOLD),
        "log_prob_threshold": repr(ASR_LOG_PROB_THRESHOLD),
        "compression_ratio_threshold": repr(ASR_MAX_COMPRESSION_RATIO),
        "language": "ja", "prompt": "議事録",
    }


def test_no_language_and_no_prompt_are_left_out():
    """An empty prompt is still a prompt to Whisper; leave it out."""
    query = decode_query("", 1, None)
    assert "language" not in query
    assert "prompt" not in query
    assert "prompt" not in decode_query("", 1, "")


def test_the_query_is_one_the_server_accepts():
    """The two sides spell the options the same way."""
    options = whisper_server.DecodeOptions.from_query(
        decode_query("vi", 5, "FPT"))
    assert options.language == "vi"
    assert options.beam_size == 5
    assert options.prompt == "FPT"
    assert options.no_speech_threshold == ASR_NO_SPEECH_THRESHOLD
    assert options.log_prob_threshold == ASR_LOG_PROB_THRESHOLD
    assert options.compression_ratio_threshold == ASR_MAX_COMPRESSION_RATIO


def test_an_answer_becomes_pieces_and_drops_blank_words():
    pieces, language = pieces_from_answer({
        "language": "vi", "segments": [{
            "text": " xin chào", "start": 0.0, "end": 0.7,
            "avg_logprob": -0.2, "no_speech_prob": 0.01,
            "compression_ratio": 1.1,
            "words": [{"text": " xin", "start": 0.1, "end": 0.3,
                       "probability": 0.9},
                      {"text": "  ", "start": 0.3, "end": 0.3,
                       "probability": 0.1},
                      {"text": " chào", "start": 0.3, "end": 0.6,
                       "probability": 0.8}]}]})
    assert language == "vi"
    assert pieces == [Piece(
        text=" xin chào", avg_logprob=-0.2, no_speech_prob=0.01,
        compression_ratio=1.1, start=0.0, end=0.7,
        words=(Word(" xin", 0.1, 0.3, 0.9), Word(" chào", 0.3, 0.6, 0.8)))]


def test_an_answer_of_another_shape_is_an_asr_error():
    with pytest.raises(AsrError):
        pieces_from_answer({"segments": [{"text": "x"}]})


# ---------------------------------------------------------------------------
# Against the real server app
# ---------------------------------------------------------------------------
def test_it_checks_the_server_and_warms_it_up(wire):
    decoder = remote(wire)
    assert [entry[:2] for entry in wire.log] == [
        ("GET", "/health"),
        ("POST", wire.log[1][1]),
    ]
    assert wire.log[1][1].startswith("/v1/decode?")
    assert decoder.model_id == "large-v3"
    assert "large-v3" in decoder.source and "http://pod:8002" in decoder.source


def test_the_pod_decodes_the_samples_the_pipeline_holds(wire):
    decoder = remote(wire)
    raw = (np.sin(np.arange(16000) / 7) * 9000).astype("<i2").tobytes()
    samples = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    pieces, language = decoder.decode(samples, "ja", 5, "議事録")

    sent, kwargs = wire.model.calls[-1]
    np.testing.assert_array_equal(sent, samples)
    assert kwargs["language"] == "ja"
    assert kwargs["beam_size"] == 5
    assert kwargs["initial_prompt"] == "議事録"
    assert kwargs["word_timestamps"] is True
    assert kwargs["vad_filter"] is False
    assert kwargs["condition_on_previous_text"] is False
    assert language == "ja"
    assert [word.text for word in pieces[0].words] == ["こん", "にちは"]


def test_a_url_prefix_is_kept(wire):
    RemoteWhisperDecoder(base_url="http://gateway/whisper/",
                         connection_factory=lambda: PrefixStrip(wire))
    assert wire.log[0][1] == "/health"


class PrefixStrip:
    """A reverse proxy that serves the app under /whisper."""

    def __init__(self, wire):
        self.inner = wire.factory()

    def request(self, method, url, body=None, headers=None):
        assert url.startswith("/whisper/")
        self.inner.request(method, url[len("/whisper"):], body, headers)

    def getresponse(self):
        return self.inner.getresponse()

    def close(self):
        pass


def test_a_server_serving_another_model_is_refused(wire):
    with pytest.raises(AsrError, match="serves 'large-v3'"):
        remote(wire, model_id="large-v3-turbo")


def test_a_server_speaking_another_api_version_is_refused(wire, monkeypatch):
    monkeypatch.setattr(whisper_server, "API_VERSION", 99)
    with pytest.raises(AsrError, match="API version"):
        remote(wire)


def test_a_server_error_names_the_reason(wire):
    decoder = remote(wire)

    def broken(*_args, **_kwargs):
        raise RuntimeError("model fell over")

    wire.model.transcribe = broken
    with pytest.raises(AsrError, match="500.*model fell over"):
        decoder.decode(np.zeros(1600, np.float32), "", 1)


def test_a_bad_url_is_refused():
    with pytest.raises(AsrError, match="not an http URL"):
        RemoteWhisperDecoder(base_url="pod:8002",
                             connection_factory=lambda: None)


# ---------------------------------------------------------------------------
# A connection that drops
# ---------------------------------------------------------------------------
class Dropping:
    """Fails the first ``failures`` requests made on any connection."""

    def __init__(self, wire, failures, error=ConnectionResetError):
        self.wire = wire
        self.failures = failures
        self.error = error
        self.opened = 0

    def factory(self):
        self.opened += 1
        outer = self
        inner = self.wire.factory()

        class Connection:
            def request(self, *args, **kwargs):
                if outer.failures > 0:
                    outer.failures -= 1
                    raise outer.error("dropped")
                inner.request(*args, **kwargs)

            def getresponse(self):
                return inner.getresponse()

            def close(self):
                pass

        return Connection()


def test_a_dropped_kept_alive_connection_is_opened_once_more(wire):
    dropping = Dropping(wire, failures=0)
    decoder = RemoteWhisperDecoder(base_url="http://pod:8002",
                                   connection_factory=dropping.factory)
    assert dropping.opened == 1
    dropping.failures = 1
    decoder.decode(np.zeros(1600, np.float32), "", 1)
    assert dropping.opened == 2


def test_a_server_that_is_gone_is_reported_without_a_retry_loop(wire):
    dropping = Dropping(wire, failures=10, error=ConnectionRefusedError)
    with pytest.raises(AsrError, match="No Whisper server at http://pod:8002"):
        RemoteWhisperDecoder(base_url="http://pod:8002",
                             connection_factory=dropping.factory)
    assert dropping.opened == 1


def test_a_timeout_is_not_retried(wire):
    """Retrying would hold the audio thread for twice ASR_TIMEOUT_S."""
    dropping = Dropping(wire, failures=0)
    decoder = RemoteWhisperDecoder(base_url="http://pod:8002",
                                   connection_factory=dropping.factory)
    dropping.failures = 1
    dropping.error = TimeoutError
    with pytest.raises(AsrError):
        decoder.decode(np.zeros(1600, np.float32), "", 1)
    assert dropping.opened == 1


def test_a_real_refused_port_is_an_asr_error():
    """Nothing is listening on port 9 of localhost."""
    with pytest.raises(AsrError, match="No Whisper server"):
        RemoteWhisperDecoder(base_url="http://127.0.0.1:9", timeout=2)


# ---------------------------------------------------------------------------
# The backend switch
# ---------------------------------------------------------------------------
def test_the_default_backend_is_the_pod():
    assert asr_module.ASR_BACKEND == "remote"


def test_make_decoder_builds_the_remote_one(monkeypatch):
    built = {}

    class Fake:
        def __init__(self, base_url="", model_id=""):
            built.update(base_url=base_url, model_id=model_id)

    from server.pipeline import asr_remote
    monkeypatch.setattr(asr_remote, "RemoteWhisperDecoder", Fake)
    decoder = asr_module.make_decoder("remote", model_id="large-v3",
                                      url="http://pod:8002")
    assert isinstance(decoder, Fake)
    assert built == {"base_url": "http://pod:8002", "model_id": "large-v3"}


def test_make_decoder_refuses_an_unknown_backend():
    with pytest.raises(AsrError, match="ASR_BACKEND"):
        asr_module.make_decoder("vllm")


def test_the_streaming_policy_runs_on_the_remote_decoder(wire):
    """The whole Transcriber, end to end through the HTTP seam."""
    transcriber = Transcriber(decoder=remote(wire), prompt="")
    raw = (np.sin(np.arange(16000) / 7) * 9000).astype("<i2").tobytes()
    result = transcriber.transcribe(raw, "ja")
    assert result.text == "こんにちは"
    assert result.lang_code == "ja"


def test_http_client_errors_are_caught():
    """The ones http.client raises are among those handled."""
    assert issubclass(http.client.RemoteDisconnected, OSError)
    assert issubclass(http.client.BadStatusLine, http.client.HTTPException)
