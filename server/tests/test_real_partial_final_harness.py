"""Smoke tests for ``server/tests_real/test_real_partial_final.py``.

Drives the script, main() included, with a stubbed VAD, decoder, language ID
and DSP, so a crash in it is caught here rather than after a round trip
through the pod and ten minutes of H100 time.

What is *not* tested here is whether the four counters mean anything on real
speech - a stub decoder says what it is told to say. That is the whole point
of running the script on the pod.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server.config import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH  # noqa: E402
from server.pipeline.asr import Piece, Transcriber  # noqa: E402
from server.pipeline.lid import LanguageDecision  # noqa: E402
from server.pipeline.buffer import PartialWindow  # noqa: E402
from server.pipeline.overlap import OverlapResolver  # noqa: E402
from server.pipeline.vad import VAD_FRAME_SAMPLES  # noqa: E402


def load_harness():
    path = ROOT / "server" / "tests_real" / "test_real_partial_final.py"
    spec = importlib.util.spec_from_file_location("real_drift_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = load_harness()


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class ScriptedVAD:
    """Speech probability from a script rather than from a model."""

    def __init__(self, probabilities):
        self.script = list(probabilities)
        self.calls = 0

    def probability(self, frame: np.ndarray) -> float:
        assert frame.shape[-1] == VAD_FRAME_SAMPLES
        value = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return float(value)

    def reset(self) -> None:
        self.calls = 0


class StubDecoder:
    """Answers differently by beam, which is how the variants are told apart.

    beam 1 says what the running text said in the real evidence; beam 5 says
    what the sentence said. That is not a claim about Whisper - it is a way
    to prove the script routes each variant to the decoder it should.
    """

    source = "stub decoder"

    def __init__(self):
        self.calls: list[dict] = []

    def decode(self, samples, lang_code, beam_size, prompt=None):
        self.calls.append({"lang_code": lang_code, "beam_size": beam_size,
                           "samples": int(samples.size)})
        if beam_size == 1:
            return [Piece(" về Solution và cái lý do", -0.2, 0.05, 1.6)], "vi"
        return ([Piece(" về sau lưu sinh và cái cái lý do thì cũng đã nắm à",
                       -0.2, 0.05, 1.6)], "vi")


class StubLanguageIdentifier:
    """Vietnamese, then undecided - so the fallback is exercised."""

    def __init__(self, decisions=None):
        self.decisions = list(decisions or [])
        self.calls = 0
        self.resets = 0

    def identify(self, pcm: bytes) -> LanguageDecision:
        if self.decisions:
            decision = self.decisions[min(self.calls, len(self.decisions) - 1)]
        else:
            decision = LanguageDecision("vi", 0.9, 0.5, "clear")
        self.calls += 1
        return decision

    def reset(self) -> None:
        self.resets += 1


class Passthrough:
    def process(self, samples, sample_rate, gate_threshold_db,
                compressor_threshold_db):
        return samples


def write_wav(path: Path, seconds: float, amplitude: int = 6000) -> Path:
    samples = np.full(int(SAMPLE_RATE * seconds), amplitude, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())
    return path


@pytest.fixture
def offline(monkeypatch):
    """Every model the script loads, replaced by something that cannot fail."""
    decoder = StubDecoder()
    identifier = StubLanguageIdentifier()
    monkeypatch.setattr(harness, "SileroVAD", lambda: ScriptedVAD([0.9]))
    monkeypatch.setattr(harness, "LanguageIdentifier", lambda: identifier)
    monkeypatch.setattr(harness, "OverlapResolver",
                        lambda: OverlapResolver(processor=Passthrough()))
    monkeypatch.setattr(harness.asr_module, "WhisperDecoder",
                        lambda model_id="", device="": decoder)
    return {"decoder": decoder, "identifier": identifier}


# ---------------------------------------------------------------------------
# Reading the recording
# ---------------------------------------------------------------------------
def test_read_pcm_rejects_the_wrong_format_and_says_how_to_convert(tmp_path):
    path = tmp_path / "meeting.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(48_000)
        wav.writeframes(bytes(4000))
    with pytest.raises(ValueError, match="ffmpeg"):
        harness.read_pcm(path, 0.0)


def test_limit_seconds_truncates(tmp_path):
    path = write_wav(tmp_path / "meeting.wav", 10.0)
    assert len(harness.read_pcm(path, 2.0)) == 2 * SAMPLE_RATE * SAMPLE_WIDTH
    assert len(harness.read_pcm(path, 0.0)) == 10 * SAMPLE_RATE * SAMPLE_WIDTH


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def test_replay_produces_sentences_and_running_texts(tmp_path):
    pcm = harness.read_pcm(write_wav(tmp_path / "m.wav", 20.0), 0.0)
    events = harness.replay(pcm, ScriptedVAD([0.9]))
    kinds = [kind for kind, _item in events]
    assert "final" in kinds and "partial" in kinds


def test_a_running_text_carries_the_index_of_the_sentence_it_becomes(tmp_path):
    """The join key between the two halves of the comparison.

    ``BufferManager`` numbers a partial with the index the *next* finalised
    utterance will get, which is what lets the last running text be matched
    to its sentence. If that ever stops being true this comparison silently
    compares unrelated pairs.
    """
    pcm = harness.read_pcm(write_wav(tmp_path / "m.wav", 20.0), 0.0)
    events = harness.replay(pcm, ScriptedVAD([0.9]))
    seen_partials = set()
    for kind, item in events:
        if kind == "partial":
            seen_partials.add(item.index)
        else:
            assert item.index in seen_partials


def test_silence_gives_nothing_to_compare(tmp_path):
    pcm = harness.read_pcm(write_wav(tmp_path / "quiet.wav", 5.0), 0.0)
    assert harness.replay(pcm, ScriptedVAD([0.02])) == []


# ---------------------------------------------------------------------------
# The trim probe
# ---------------------------------------------------------------------------
def test_trim_removes_exactly_the_asked_for_tail():
    pcm = bytes(SAMPLE_RATE * SAMPLE_WIDTH)             # one second
    trimmed = harness.trim_tail(pcm, 480.0)
    assert harness.bytes_to_ms(len(trimmed)) == pytest.approx(520.0, abs=1.0)


def test_trim_never_eats_a_short_sentence():
    pcm = bytes(harness.ms_to_bytes(500.0))
    trimmed = harness.trim_tail(pcm, 480.0)
    assert harness.bytes_to_ms(len(trimmed)) == pytest.approx(
        harness.MIN_KEPT_MS, abs=1.0)


def test_no_trim_is_a_pass_through():
    pcm = bytes(1000)
    assert harness.trim_tail(pcm, 0.0) is pcm


# ---------------------------------------------------------------------------
# Forcing the language
# ---------------------------------------------------------------------------
def scripted_events():
    """A running text, then its sentence, then the next running text."""
    from server.pipeline.buffer import FinalizeReason, Utterance

    second = bytes(SAMPLE_RATE * SAMPLE_WIDTH)
    return [
        ("partial", PartialWindow(index=0, pcm=second, start_ms=0.0)),
        ("final", Utterance(index=0, pcm=second, start_ms=0.0,
                            reason=FinalizeReason.PAUSE)),
        ("partial", PartialWindow(index=1, pcm=second, start_ms=1000.0)),
    ]


def test_an_undecided_event_falls_back_to_the_meeting_language():
    """``ServerSession._language_for``, reproduced.

    Whisper's own detector answered Swedish for a Vietnamese-Japanese meeting,
    which is why the fallback exists. If the script did not copy it, every
    variant would be measured against a decode the server never makes.

    Only a sentence writes the memory - the running text reads it. So the
    first running text of a meeting has nothing to fall back on, and the one
    after a decided sentence does.
    """
    events = scripted_events()
    identifier = StubLanguageIdentifier([
        LanguageDecision("", 0.4, 0.1, "too close to call"),
        LanguageDecision("vi", 0.9, 0.5, "clear"),
        LanguageDecision("", 0.4, 0.1, "too close to call"),
    ])
    forced = harness.language_pass(events, identifier)
    assert forced[("partial", 0, 0.0)] == ""
    assert forced[("final", 0, 0.0)] == "vi"
    assert forced[("partial", 1, 1000.0)] == "vi"
    assert identifier.resets == 1


def test_a_running_text_never_updates_the_meeting_language():
    """It is a fragment of a sentence, decided on less audio than the sentence
    gets. Letting it set the memory would let the weaker evidence win."""
    events = scripted_events()
    identifier = StubLanguageIdentifier([
        LanguageDecision("ja", 0.9, 0.5, "clear"),
        LanguageDecision("", 0.4, 0.1, "too close to call"),
        LanguageDecision("", 0.4, 0.1, "too close to call"),
    ])
    forced = harness.language_pass(events, identifier)
    assert forced[("final", 0, 0.0)] == ""
    assert forced[("partial", 1, 1000.0)] == ""


def test_no_language_model_forces_nothing(tmp_path):
    pcm = harness.read_pcm(write_wav(tmp_path / "m.wav", 12.0), 0.0)
    events = harness.replay(pcm, ScriptedVAD([0.9]))
    assert harness.language_pass(events, None) == {}


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def test_the_running_text_is_decoded_greedily_on_a_four_second_tail(tmp_path):
    pcm = harness.read_pcm(write_wav(tmp_path / "m.wav", 20.0), 0.0)
    events = harness.replay(pcm, ScriptedVAD([0.9]))
    decoder = StubDecoder()
    state, seconds, count = harness.decode_partials(
        events, Transcriber(decoder=decoder), {})
    assert count == sum(1 for kind, _ in events if kind == "partial")
    assert seconds >= 0.0
    assert {call["beam_size"] for call in decoder.calls} == {1}
    longest = max(call["samples"] for call in decoder.calls)
    assert longest <= harness.PARTIAL_WINDOW_SECONDS * SAMPLE_RATE
    assert state["last"]


def test_each_variant_decodes_with_its_own_beam_and_restores_the_default(
        tmp_path):
    pcm = harness.read_pcm(write_wav(tmp_path / "m.wav", 20.0), 0.0)
    events = harness.replay(pcm, ScriptedVAD([0.9]))
    decoder = StubDecoder()
    before = harness.asr_module.ASR_BEAM_SIZE_FINAL

    decoded, _seconds = harness.decode_finals(
        events, Transcriber(decoder=decoder), {},
        harness.Variant("beam1", 1, 0.0), None)

    assert decoded
    assert {call["beam_size"] for call in decoder.calls} == {1}
    assert harness.asr_module.ASR_BEAM_SIZE_FINAL == before


def test_the_trim_variant_hands_the_decoder_shorter_audio(tmp_path):
    pcm = harness.read_pcm(write_wav(tmp_path / "m.wav", 20.0), 0.0)
    events = harness.replay(pcm, ScriptedVAD([0.9]))

    plain = StubDecoder()
    harness.decode_finals(events, Transcriber(decoder=plain), {},
                          harness.Variant("baseline", 5, 0.0), None)
    trimmed = StubDecoder()
    harness.decode_finals(events, Transcriber(decoder=trimmed), {},
                          harness.Variant("trim", 5, 480.0), None)

    assert (sum(call["samples"] for call in trimmed.calls)
            < sum(call["samples"] for call in plain.calls))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def make_case(index: int, final: str, partial: str, extra_ms: float = 300.0):
    case = harness.Case(index=index, start_ms=0.0, end_ms=6000.0,
                        reason="pause", continues_previous=False,
                        lang_code="vi", partial_text=partial,
                        partial_end_ms=6000.0 - extra_ms, partial_count=7)
    case.finals["baseline"] = harness.Decoded(final, 0.2, 6000.0)
    case.drift["baseline"] = harness.compare(final, partial, extra_ms, 6000.0)
    return case


def test_a_summary_counts_flags_and_lists_the_words_that_went_missing():
    cases = [
        make_case(0, "về sau lưu sinh và cái cái lý do", "về Solution và cái lý do"),
        make_case(1, "nguyên nhân là biết rồi", "nhân là biết rồi"),
    ]
    row = harness.summarise(cases, harness.Variant("baseline", 5, 0.0))
    assert row["compared"] == 2
    assert row["clean"] == 1
    assert row["flagged"] == 1
    assert row["counts"]["latin_lost"] == 1
    assert row["lost_words"] == ["solution"]


def test_a_sentence_with_no_running_text_is_not_counted_as_clean():
    case = harness.Case(index=0, start_ms=0.0, end_ms=800.0, reason="pause",
                        continues_previous=False, lang_code="vi")
    case.finals["baseline"] = harness.Decoded("vâng", 0.1, 800.0)
    row = harness.summarise([case], harness.Variant("baseline", 5, 0.0))
    assert row["compared"] == 0


def test_printing_the_worst_cases_does_not_crash(capsys):
    cases = [make_case(0, "về sau lưu sinh và cái cái lý do",
                       "về Solution và cái lý do")]
    variants = [harness.Variant("baseline", 5, 0.0)]
    harness.print_summary([harness.summarise(cases, variants[0])], "baseline")
    harness.print_worst(cases, variants, "baseline", top=5)
    printed = capsys.readouterr().out
    assert "solution" in printed.casefold()
    assert "latin_lost" in printed


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_main_runs_every_variant_and_writes_the_json(tmp_path, offline,
                                                     monkeypatch, capsys):
    wav = write_wav(tmp_path / "meeting.wav", 30.0)
    out = tmp_path / "drift.json"
    monkeypatch.setattr(sys, "argv", [
        "test_real_partial_final.py", "--wav", str(wav),
        "--out", str(out), "--top", "3", "--variants", "all",
    ])

    assert harness.main() == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [variant["name"] for variant in payload["variants"]] == [
        "baseline", "beam1", "trim", "beam1+trim"]
    assert payload["cases"]
    assert all("baseline" in case["finals"] for case in payload["cases"])

    printed = capsys.readouterr().out
    assert "Against the baseline:" in printed
    # The stub decoder answers by beam, so beam1 must come out cleaner than a
    # baseline that drops "Solution" and doubles "cái".
    summary = {row["variant"]: row for row in payload["summary"]}
    assert summary["beam1"]["flagged"] < summary["baseline"]["flagged"]


def test_only_the_baseline_runs_unless_asked(tmp_path, offline, monkeypatch):
    """The other three were measured and none of them was the fault.

    Paying for them on every run is four times the GPU time for a question
    that already has an answer.
    """
    wav = write_wav(tmp_path / "meeting.wav", 20.0)
    out = tmp_path / "drift.json"
    monkeypatch.setattr(sys, "argv", [
        "test_real_partial_final.py", "--wav", str(wav), "--out", str(out)])
    assert harness.main() == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [variant["name"] for variant in payload["variants"]] == ["baseline"]


def test_every_segment_is_recorded_with_its_scores_and_verdict(
        tmp_path, offline, monkeypatch):
    """What makes a guard rule answerable later without a GPU."""
    wav = write_wav(tmp_path / "meeting.wav", 20.0)
    out = tmp_path / "drift.json"
    monkeypatch.setattr(sys, "argv", [
        "test_real_partial_final.py", "--wav", str(wav), "--out", str(out)])
    assert harness.main() == 0

    pieces = json.loads(out.read_text(encoding="utf-8")
                        )["cases"][0]["finals"]["baseline"]["pieces"]
    assert pieces
    assert set(pieces[0]) == {"text", "avg_logprob", "no_speech_prob",
                              "compression_ratio", "verdict"}
    assert pieces[0]["verdict"] == "kept"


def test_a_refused_segment_records_the_reason_it_was_refused(tmp_path,
                                                             monkeypatch):
    class Silent:
        """Whisper over near-silence: fluent, badly decoded, never said."""

        source = "stub"

        def decode(self, samples, lang_code, beam_size, prompt=None):
            return [Piece(" mumble over a quiet room", -1.4, 0.95, 1.3)], "vi"

    recorder = harness.RecordingDecoder(Silent())
    transcriber = Transcriber(decoder=recorder)
    transcriber.transcribe(bytes(SAMPLE_RATE * SAMPLE_WIDTH), "vi",
                           is_final=True)
    pieces = harness.record_pieces(recorder, transcriber)
    assert [piece["verdict"] for piece in pieces] == ["no speech"]
    assert pieces[0]["no_speech_prob"] == pytest.approx(0.95)
    assert pieces[0]["avg_logprob"] == pytest.approx(-1.4)


def test_the_recording_decoder_keeps_the_order_the_model_returned():
    """The sentence's text depends on it, and ``Transcript`` does not keep it."""
    class TwoPieces:
        source = "stub"

        def decode(self, samples, lang_code, beam_size, prompt=None):
            return ([Piece(" một", -0.2, 0.05, 1.6),
                     Piece(" hai", -0.2, 0.05, 1.6)], "vi")

    recorder = harness.RecordingDecoder(TwoPieces())
    transcriber = Transcriber(decoder=recorder)
    transcriber.transcribe(bytes(SAMPLE_RATE * SAMPLE_WIDTH), "vi")
    assert [piece["text"].strip()
            for piece in harness.record_pieces(recorder, transcriber)] == [
        "một", "hai"]


def test_running_texts_can_be_reused_from_an_earlier_run(tmp_path, offline,
                                                         monkeypatch, capsys):
    """Four fifths of the decoding, for a question about the sentence path."""
    wav = write_wav(tmp_path / "meeting.wav", 20.0)
    first = tmp_path / "first.json"
    monkeypatch.setattr(sys, "argv", [
        "test_real_partial_final.py", "--wav", str(wav), "--out", str(first)])
    assert harness.main() == 0
    original = json.loads(first.read_text(encoding="utf-8"))

    second = tmp_path / "second.json"
    monkeypatch.setattr(sys, "argv", [
        "test_real_partial_final.py", "--wav", str(wav), "--out", str(second),
        "--reuse-partials", str(first)])
    assert harness.main() == 0

    printed = capsys.readouterr().out
    assert "Reusing" in printed
    again = json.loads(second.read_text(encoding="utf-8"))
    assert again["partial_seconds"] == 0.0
    assert ([case["partial_text"] for case in again["cases"]]
            == [case["partial_text"] for case in original["cases"]])
    assert ([case["partial_end_ms"] for case in again["cases"]]
            == [case["partial_end_ms"] for case in original["cases"]])


def test_a_reuse_file_that_cannot_be_read_is_refused(tmp_path, offline,
                                                     monkeypatch):
    wav = write_wav(tmp_path / "meeting.wav", 12.0)
    monkeypatch.setattr(sys, "argv", [
        "test_real_partial_final.py", "--wav", str(wav),
        "--reuse-partials", str(tmp_path / "nope.json")])
    assert harness.main() == 2


def test_main_refuses_a_recording_it_cannot_read(tmp_path, offline, monkeypatch):
    bad = tmp_path / "stereo.wav"
    with wave.open(str(bad), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(48_000)
        wav.writeframes(bytes(4000))
    monkeypatch.setattr(sys, "argv",
                        ["test_real_partial_final.py", "--wav", str(bad)])
    assert harness.main() == 2


def test_main_can_be_narrowed_to_two_variants(tmp_path, offline, monkeypatch,
                                              capsys):
    wav = write_wav(tmp_path / "meeting.wav", 20.0)
    monkeypatch.setattr(sys, "argv", [
        "test_real_partial_final.py", "--wav", str(wav),
        "--variants", "baseline,beam1", "--no-lid", "--no-overlap",
    ])
    assert harness.main() == 0
    printed = capsys.readouterr().out
    assert "trim" not in printed.split("Against the baseline:")[1]


def test_main_says_so_when_there_is_no_speech(tmp_path, offline, monkeypatch):
    monkeypatch.setattr(harness, "SileroVAD", lambda: ScriptedVAD([0.02]))
    wav = write_wav(tmp_path / "quiet.wav", 5.0)
    monkeypatch.setattr(sys, "argv",
                        ["test_real_partial_final.py", "--wav", str(wav)])
    assert harness.main() == 1
