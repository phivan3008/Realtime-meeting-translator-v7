"""Smoke tests for ``server/tests_real/test_real_asr.py``.

Drives the script, main() included, with a stubbed decoder, so a crash in it
is caught here rather than after a round trip through the pod.

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
from server.pipeline.asr import Piece, Transcriber  # noqa: E402
from server.pipeline.vad import VAD_FRAME_SAMPLES  # noqa: E402


def load_harness():
    path = ROOT / "server" / "tests_real" / "test_real_asr.py"
    spec = importlib.util.spec_from_file_location("real_asr_harness", path)
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


class StubDecoder:
    """Speaks whatever language it is told, and invents over silence."""

    source = "stub decoder"

    def __init__(self, text: str = "xin chao", invent_on_silence: bool = True):
        self.text = text
        self.invent_on_silence = invent_on_silence
        self.calls: list[dict] = []

    def decode(self, samples, lang_code, beam_size):
        self.calls.append({"lang_code": lang_code, "beam_size": beam_size})
        loud = float(np.max(np.abs(samples))) if samples.size else 0.0
        if loud < 0.01:
            if not self.invent_on_silence:
                return [], "vi"
            # What Whisper really does over a quiet room.
            return [Piece(" Thank you for watching!", -0.3, 0.95, 1.3)], "en"
        return [Piece(f" {self.text}", -0.2, 0.05, 1.6)], lang_code or "vi"


class Passthrough:
    def process(self, samples, sample_rate, gate_threshold_db,
                compressor_threshold_db):
        return samples


def make_resolver():
    from server.pipeline.overlap import OverlapResolver

    return OverlapResolver(processor=Passthrough())


def write_wav(path: Path, seconds: float, amplitude: int = 6000) -> Path:
    samples = np.full(int(SAMPLE_RATE * seconds), amplitude, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())
    return path


def line_of(text: str, lang: str, expected: str, forced_s: float = 0.1,
            partial_s: float = 0.05, dropped=(), truncated: bool = False):
    from server.pipeline.asr import Transcript
    from server.pipeline.buffer import FinalizeReason, Utterance

    reason = (FinalizeReason.END_OF_STREAM if truncated
              else FinalizeReason.PAUSE)
    return harness.Line(
        utterance=Utterance(index=0, pcm=bytes(SAMPLE_RATE * SAMPLE_WIDTH),
                            start_ms=0.0, reason=reason),
        forced=Transcript(text, lang, True, dropped=tuple(dropped)),
        detected=Transcript(text, expected, True),
        forced_seconds=forced_s,
        partial_seconds=partial_s,
    )


def refusal(text: str = " Thank you for watching!"):
    return ((Piece(text, -0.3, 0.95, 1.3), "no speech"),)


def reading_of(language: str, lines) -> "harness.Reading":
    reading = harness.Reading(path=Path(f"{language}.wav"), language=language)
    reading.lines = list(lines)
    return reading


# ---------------------------------------------------------------------------
# Reading a file aloud
# ---------------------------------------------------------------------------
def test_read_pcm_rejects_the_wrong_format(tmp_path):
    path = tmp_path / "wrong.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(4000))
    with pytest.raises(ValueError, match="2 ch"):
        harness.read_pcm(path)


def test_a_recording_is_transcribed_sentence_by_sentence(tmp_path):
    path = write_wav(tmp_path / "vi.wav", 12.0)
    reading = harness.read_aloud(path, "vi", Transcriber(decoder=StubDecoder()),
                                 ScriptedVAD([0.9]), make_resolver())
    assert reading.lines
    assert all(line.forced.text == "xin chao" for line in reading.lines)
    assert reading.seconds > 0
    assert reading.final_rtf > 0


def test_each_sentence_is_decoded_forced_detected_and_as_a_partial(tmp_path):
    stub = StubDecoder()
    path = write_wav(tmp_path / "vi.wav", 12.0)
    reading = harness.read_aloud(path, "vi", Transcriber(decoder=stub),
                                 ScriptedVAD([0.9]), make_resolver())
    assert len(stub.calls) == 3 * len(reading.lines)
    languages = [call["lang_code"] for call in stub.calls[:3]]
    assert languages == ["vi", "", "vi"]


def test_a_silent_recording_gives_nothing_to_read(tmp_path):
    path = write_wav(tmp_path / "quiet.wav", 4.0)
    reading = harness.read_aloud(path, "vi", Transcriber(decoder=StubDecoder()),
                                 ScriptedVAD([0.02]), make_resolver())
    assert reading.lines == []
    assert reading.final_rtf == 0.0


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def test_a_healthy_reading_passes():
    report = harness.Report()
    harness.check_reading(reading_of("vi", [line_of("xin chao", "vi", "vi")]),
                          report)
    assert report.failed == []


def test_a_sentence_lost_for_no_stated_reason_is_caught():
    """Empty with nothing refused means Whisper simply said nothing."""
    report = harness.Report()
    harness.check_reading(reading_of("vi", [line_of("", "vi", "vi")]), report)
    assert "vi.wav never loses a sentence without saying why" in [
        c.name for c in report.failed
    ]


def test_a_refused_hallucination_is_not_a_failure():
    """Silence answered with "Thank you for watching" and thrown away is the
    stage doing its job, not losing a sentence."""
    report = harness.Report()
    harness.check_reading(
        reading_of("vi", [line_of("hello", "vi", "vi"),
                          line_of("hello", "vi", "vi"),
                          line_of("", "vi", "vi", dropped=refusal())]),
        report,
    )
    assert report.failed == []


def test_guards_eating_most_of_a_recording_are_caught():
    report = harness.Report()
    harness.check_reading(
        reading_of("vi", [line_of("", "vi", "vi", dropped=refusal()),
                          line_of("", "vi", "vi", dropped=refusal()),
                          line_of("hello", "vi", "vi")]),
        report,
    )
    assert "vi.wav guards refuse a minority of sentences" in [
        c.name for c in report.failed
    ]


def test_a_sentence_cut_off_by_the_end_of_the_file_is_not_judged(capsys):
    """A file stops mid-sentence; a meeting does not."""
    report = harness.Report()
    harness.check_reading(
        reading_of("vi", [line_of("hello", "vi", "vi"),
                          line_of("", "vi", "vi", truncated=True)]),
        report,
    )
    assert report.failed == []
    assert "cut off by the end of the recording" in capsys.readouterr().out


def test_decoding_slower_than_the_budget_is_caught():
    report = harness.Report()
    harness.check_reading(
        reading_of("vi", [line_of("hi", "vi", "vi", forced_s=0.9)]), report)
    assert "vi.wav decodes far faster than real time" in [
        c.name for c in report.failed
    ]


def test_partials_costing_as_much_as_finals_are_caught():
    report = harness.Report()
    harness.check_reading(
        reading_of("vi", [line_of("hi", "vi", "vi", partial_s=0.5)]), report)
    assert "vi.wav partials are cheaper than finals" in [
        c.name for c in report.failed
    ]


def test_a_forced_language_that_did_not_stick_is_caught():
    report = harness.Report()
    harness.check_reading(reading_of("vi", [line_of("hi", "ja", "ja")]), report)
    assert "vi.wav keeps the language it was told to use" in [
        c.name for c in report.failed
    ]


def test_an_empty_reading_stops_after_the_first_check():
    report = harness.Report()
    harness.check_reading(reading_of("vi", []), report)
    assert [c.name for c in report.checks] == ["vi.wav produced sentences"]


def test_agreement_reports_what_whisper_thought_by_itself(capsys):
    report = harness.Report()
    harness.check_agreement(
        reading_of("vi", [line_of("hi", "vi", "vi"), line_of("hi", "vi", "en")]),
        report,
    )
    out = capsys.readouterr().out
    assert "1/2" in out
    assert "same text for 2/2" in out


def test_invented_text_over_silence_is_caught():
    """The reason the guards exist, with the real symptom."""
    report = harness.Report()
    transcriber = Transcriber(decoder=StubDecoder(),
                              no_speech_threshold=0.99)
    harness.check_silence(bytes(SAMPLE_RATE * SAMPLE_WIDTH), transcriber, report)
    assert [c.name for c in report.failed] == ["Silence produces no transcript"]


def test_the_guards_stop_invented_text_reaching_the_transcript():
    report = harness.Report()
    harness.check_silence(bytes(SAMPLE_RATE * SAMPLE_WIDTH),
                          Transcriber(decoder=StubDecoder()), report)
    assert report.failed == []


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def run_main(monkeypatch, tmp_path, with_silence: bool = True) -> int:
    monkeypatch.setattr(harness, "WhisperDecoder", lambda **k: StubDecoder())
    monkeypatch.setattr(harness, "SileroVAD", lambda **k: ScriptedVAD([0.9]))
    monkeypatch.setattr(harness, "PedalboardProcessor", Passthrough)
    argv = ["x",
            "--vi", str(write_wav(tmp_path / "vi.wav", 12.0)),
            "--ja", str(write_wav(tmp_path / "ja.wav", 12.0))]
    if with_silence:
        argv += ["--silence", str(write_wav(tmp_path / "quiet.wav", 3.0,
                                            amplitude=0))]
    monkeypatch.setattr(sys, "argv", argv)
    return harness.main()


def test_main_runs_and_passes(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path) == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out
    assert "Nothing here can tell you whether they are right" in out


def test_main_says_when_invented_text_is_unproven(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, with_silence=False) == 0
    assert "invented text is unproven" in capsys.readouterr().out


def test_main_needs_at_least_one_recording(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x"])
    assert harness.main() == 2
    assert "at least one" in capsys.readouterr().out


def test_main_reports_a_model_that_will_not_load(monkeypatch, tmp_path, capsys):
    def explode(**_kwargs):
        raise harness.AsrError("Could not load Whisper 'large-v3'")

    monkeypatch.setattr(harness, "WhisperDecoder", explode)
    monkeypatch.setattr(
        sys, "argv", ["x", "--vi", str(write_wav(tmp_path / "vi.wav", 2.0))])
    assert harness.main() == 2
    assert "Could not load Whisper" in capsys.readouterr().out
