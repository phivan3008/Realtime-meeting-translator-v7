"""Unit tests for the streaming real test's own logic.

The models are stubbed, so this proves only that the script replays, writes
a log the client's reader understands, and judges it - not that Whisper does
anything right. That is the pod's job.

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
from server.pipeline.asr import Piece, Transcriber, Word  # noqa: E402
from server.pipeline.lid import LanguageDecision  # noqa: E402


def load_harness():
    path = ROOT / "server" / "tests_real" / "test_real_streaming.py"
    spec = importlib.util.spec_from_file_location("real_streaming_harness",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = load_harness()


class LoudIsSpeech:
    """A VAD that hears speech wherever the recording is loud."""

    def probability(self, frame: np.ndarray) -> float:
        return 0.9 if float(np.max(np.abs(frame))) > 0.05 else 0.02

    def reset(self) -> None:
        pass


class Decoder:
    """Answers with the given words, 0.4 s apart, exactly as spelled - so a
    test can hand the checks the faults they exist to catch."""

    def __init__(self, text: str):
        self.tokens = text.split("|")

    def decode(self, samples, lang_code="", beam_size=1, prompt=None):
        words = tuple(Word(token, index * 0.4, index * 0.4 + 0.2)
                      for index, token in enumerate(self.tokens))
        text = "".join(self.tokens)
        return ([Piece(text, -0.2, 0.01, 1.3, 0.0, words[-1].end, words)],
                lang_code or "ja")


class LID:
    def reset(self) -> None:
        pass

    def identify(self, pcm: bytes) -> LanguageDecision:
        return LanguageDecision("ja", 0.9, 0.8, "stub")


def write_meeting(path: Path, turns: int = 4) -> Path:
    """Two seconds of speech, then one of silence, a few times over."""
    speech = np.full(SAMPLE_RATE * 2, 6000, dtype="<i2")
    quiet = np.zeros(SAMPLE_RATE, dtype="<i2")
    audio = np.concatenate([np.concatenate([speech, quiet])] * turns)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio.tobytes())
    return path


def run_main(monkeypatch, tmp_path, text: str, *extra) -> int:
    wav = write_meeting(tmp_path / "meeting.wav")
    # No overlap resolver and no speaker model: the checks below are about
    # rendering, and their absence is reported as a failure of its own.
    loaded = {"vad": LoudIsSpeech(), "language": LID(),
              "asr": Transcriber(decoder=Decoder(text), prompt=""),
              "overlap": None, "speaker": None}
    monkeypatch.setattr(harness, "load_stages", lambda args: dict(loaded))
    monkeypatch.setattr(sys, "argv", [
        "x", "--wav", str(wav), "--out",
        str(tmp_path / "out" / "streaming.debug.txt"), *extra])
    return harness.main()


def test_a_clean_japanese_meeting_passes_the_rendering_checks(
        monkeypatch, tmp_path, capsys):
    run_main(monkeypatch, tmp_path, "今日は会議です")
    out = capsys.readouterr().out
    assert "[PASS] Japanese is written without spaces" in out
    assert "[PASS] A word is not shown twice at a join" in out
    assert (tmp_path / "out" / "streaming.debug.txt").exists()


def test_spaced_japanese_is_caught(monkeypatch, tmp_path, capsys):
    # The renderer closes these gaps now, so it is switched off here to prove
    # the check still catches them if that ever regresses.
    import server.pipeline.asr as asr
    monkeypatch.setattr(asr, "close_unspaced_gaps", lambda text: text)
    run_main(monkeypatch, tmp_path, "今| 日| は| 会| 議")
    out = capsys.readouterr().out
    assert "[FAIL] Japanese is written without spaces" in out


def test_a_doubled_word_is_caught(monkeypatch, tmp_path, capsys):
    run_main(monkeypatch, tmp_path, " xin| chào| chào")
    out = capsys.readouterr().out
    assert "[FAIL] A word is not shown twice at a join" in out


def test_missing_stages_are_reported_as_failures(monkeypatch, tmp_path,
                                                 capsys):
    code = run_main(monkeypatch, tmp_path, "今日は会議です")
    out = capsys.readouterr().out
    assert "[FAIL] speaker loads" in out
    assert code == 1


def test_a_baseline_of_a_different_meeting_is_refused(monkeypatch, tmp_path,
                                                      capsys):
    other = tmp_path / "other.debug.txt"
    other.write_text(
        "00:00:00.000      0.0s  start       session=?\n"
        "00:00:05.000      5.0s  final       #1 Speaker_01 [vi] "
        "Một cuộc họp hoàn toàn khác về ngân sách\n", encoding="utf-8")
    run_main(monkeypatch, tmp_path, "今日は会議です", "--baseline", str(other))
    out = capsys.readouterr().out
    assert "[FAIL] other.debug.txt is the same meeting" in out


def test_a_recording_in_the_wrong_format_is_refused(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44_100)
        wav.writeframes(bytes(400))
    with pytest.raises(ValueError, match="ffmpeg"):
        harness.read_pcm(path)


def test_nothing_the_pod_runs_needs_the_client_package():
    """The pod reported ModuleNotFoundError: No module named 'client' - the
    first version of the streaming replay wrote its log through
    client.record."""
    import re

    server = ROOT / "server"
    for path in server.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import) client\b", text,
                             re.MULTILINE), path


def test_a_sentence_mixing_the_languages_is_caught(monkeypatch, tmp_path,
                                                   capsys):
    run_main(monkeypatch, tmp_path, "これは| có| thểです")
    out = capsys.readouterr().out
    assert "[FAIL] No sentence mixes Japanese and Vietnamese" in out
