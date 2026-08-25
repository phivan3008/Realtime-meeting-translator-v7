"""Smoke tests for the ``client/tests_real/test_real_stream.py`` harness.

The audio capture part cannot run here - the Dev PC has no loopback device -
but everything around it can: the health probe, the lag arithmetic and the
reporting. Those are exactly the parts that would otherwise fail on the
Windows Client PC and cost a round trip.

Run with::

    .venv\\Scripts\\python.exe -m pytest client/tests -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.protocol import (  # noqa: E402
    CHUNK_BYTES,
    CHUNK_DURATION_MS,
    PROTOCOL_VERSION,
    SAMPLE_RATE,
)


def load_harness():
    path = ROOT / "client" / "tests_real" / "test_real_stream.py"
    spec = importlib.util.spec_from_file_location("real_stream_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = load_harness()


GOOD_HEALTH = {
    "status": "ok",
    "protocol_version": PROTOCOL_VERSION,
    "sample_rate": SAMPLE_RATE,
    "chunk_bytes": CHUNK_BYTES,
    "chunk_ms": CHUNK_DURATION_MS,
    "vad_loaded": True,
    "noise_filter_loaded": True,
    "noise_filter_error": "",
    "overlap_resolver_loaded": True,
    "speaker_model_loaded": True,
    "speaker_model_error": "",
    "language_model_loaded": True,
    "language_model_error": "",
    "asr_loaded": True,
    "asr_error": "",
    "translation_loaded": True,
    "translation_error": "",
    "session_active": False,
}


def serve_health(payload: dict | None):
    """Run a one-route HTTP server on a random port; returns its base URL."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                  # noqa: N802
            if self.path != "/health" or payload is None:
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):                     # keep pytest quiet
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"ws://127.0.0.1:{server.server_port}"


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------
def test_a_matching_server_passes_every_health_check():
    server, url = serve_health(GOOD_HEALTH)
    try:
        report = harness.Report()
        assert harness.check_health(url, report) is True
        assert report.failed == []
    finally:
        server.shutdown()


def test_a_mismatched_contract_is_caught_before_any_audio_is_sent():
    server, url = serve_health({**GOOD_HEALTH, "sample_rate": 48_000})
    try:
        report = harness.Report()
        assert harness.check_health(url, report) is False
        assert [c.name for c in report.failed] == [
            "Client and server agree on the audio contract"
        ]
    finally:
        server.shutdown()


def test_a_server_without_the_model_is_flagged():
    server, url = serve_health({**GOOD_HEALTH, "vad_loaded": False})
    try:
        report = harness.Report()
        harness.check_health(url, report)
        assert [c.name for c in report.failed] == [
            "Server has the VAD model loaded"
        ]
    finally:
        server.shutdown()


def test_an_unreachable_server_fails_without_raising():
    report = harness.Report()
    assert harness.check_health("ws://127.0.0.1:9", report) is False
    assert [c.name for c in report.failed] == ["Server is reachable"]


def test_the_https_scheme_is_translated_for_the_health_probe():
    server, url = serve_health(GOOD_HEALTH)
    try:
        report = harness.Report()
        assert harness.check_health(url.replace("ws://", "wss://"), report) is False
        # wss -> https against a plain HTTP server fails at the transport layer,
        # which is the point: the scheme is rewritten, not ignored.
        assert [c.name for c in report.failed] == ["Server is reachable"]
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Lag arithmetic
# ---------------------------------------------------------------------------
def make_collected(stream_start: float, events: list[tuple[float, float, str]]):
    """events = (arrival_perf_counter, at_ms, kind)."""
    collected = harness.Collected(stream_start=stream_start)
    for arrived, at_ms, kind in events:
        collected.messages.append(
            (arrived, {"type": "vad", "event": kind, "at_ms": at_ms})
        )
    return collected


def test_lag_is_the_delay_between_capture_and_arrival():
    # Audio at 1000 ms into the stream, event landed 1.4 s after start.
    collected = make_collected(100.0, [(101.4, 1000.0, "speech_start")])
    assert collected.lags_ms() == pytest.approx([400.0])


def test_only_vad_messages_count_towards_lag():
    collected = harness.Collected(stream_start=0.0)
    collected.messages.append((0.5, {"type": "ready", "session_id": "x"}))
    collected.messages.append((1.0, {"type": "vad", "event": "speech_start",
                                     "at_ms": 500.0}))
    assert len(collected.vad_events) == 1
    assert collected.lags_ms() == pytest.approx([500.0])


def test_record_stamps_each_message_on_arrival():
    collected = harness.Collected()
    collected.record({"type": "ready"})
    collected.record({"type": "vad", "event": "speech_end", "at_ms": 0})
    assert len(collected.messages) == 2
    assert collected.messages[0][0] <= collected.messages[1][0]


# ---------------------------------------------------------------------------
# Event checks
# ---------------------------------------------------------------------------
def test_a_healthy_event_stream_passes():
    collected = make_collected(
        0.0,
        [(0.6, 200.0, "speech_start"), (5.4, 5000.0, "speech_end")],
    )
    report = harness.Report()
    harness.check_events(collected, report)
    assert report.failed == []


def test_no_events_at_all_fails():
    report = harness.Report()
    harness.check_events(harness.Collected(stream_start=0.0), report)
    assert [c.name for c in report.failed] == ["The server heard speech"]


def test_an_event_that_arrives_too_late_fails():
    collected = make_collected(0.0, [(3.0, 200.0, "speech_start")])   # 2.8 s lag
    report = harness.Report()
    harness.check_events(collected, report)
    assert "Events come back fast enough to be useful" in [
        c.name for c in report.failed
    ]


def test_an_event_that_predates_its_own_audio_fails():
    """Impossible unless the timestamps are wrong somewhere."""
    collected = make_collected(0.0, [(0.1, 5000.0, "speech_start")])
    report = harness.Report()
    harness.check_events(collected, report)
    assert "No event arrives before its audio was captured" in [
        c.name for c in report.failed
    ]


def test_an_unclosed_final_segment_fails():
    """A start with no end means the last sentence never gets finalised."""
    collected = make_collected(
        0.0,
        [(0.3, 100.0, "speech_start"), (0.6, 200.0, "speech_end"),
         (0.9, 300.0, "speech_start")],
    )
    report = harness.Report()
    harness.check_events(collected, report)
    assert "Every speech segment is opened and closed" in [
        c.name for c in report.failed
    ]


# ---------------------------------------------------------------------------
# Stream checks
# ---------------------------------------------------------------------------
class FakeClient:
    def __init__(self, **overrides):
        from client.net.ws_client import ClientStats

        self.stats = ClientStats(
            chunks_sent=overrides.get("chunks_sent", 100),
            chunks_dropped=overrides.get("chunks_dropped", 0),
            bytes_sent=overrides.get("chunks_sent", 100) * CHUNK_BYTES,
            connects=overrides.get("connects", 1),
            disconnects=overrides.get("disconnects", 1),
            errors=overrides.get("errors", []),
        )


def test_a_clean_20_second_run_passes():
    report = harness.Report()
    harness.check_stream(FakeClient(chunks_sent=100), harness.Collected(),
                         20.0, report)
    assert report.failed == []


def test_a_short_run_is_caught():
    report = harness.Report()
    harness.check_stream(FakeClient(chunks_sent=40), harness.Collected(),
                         20.0, report)
    names = [c.name for c in report.failed]
    assert "Audio streamed for the whole run" in names
    assert "Audio duration matches the bytes sent" in names


def test_dropped_chunks_are_caught():
    report = harness.Report()
    harness.check_stream(FakeClient(chunks_sent=100, chunks_dropped=3),
                         harness.Collected(), 20.0, report)
    assert [c.name for c in report.failed] == ["Nothing dropped by back-pressure"]


def test_a_reconnect_during_the_run_is_reported():
    report = harness.Report()
    harness.check_stream(FakeClient(chunks_sent=100, connects=2),
                         harness.Collected(), 20.0, report)
    assert [c.name for c in report.failed] == ["The connection survived the run"]


# ---------------------------------------------------------------------------
# Utterance checks
# ---------------------------------------------------------------------------
def utterance(index: int, start_ms: float, end_ms: float,
              reason: str = "pause", continues: bool = False,
              kept: bool = True, label: str = "",
              speech_score: float = 0.8) -> dict:
    return {
        "type": "utterance",
        "index": index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "reason": reason,
        "continues_previous": continues,
        "kept": kept,
        "label": label,
        "speech_score": speech_score,
    }


def with_utterances(*payloads, vad_ends: int = 0):
    collected = harness.Collected(stream_start=0.0)
    for i in range(vad_ends):
        collected.messages.append(
            (0.1, {"type": "vad", "event": "speech_end", "at_ms": float(i)})
        )
    for payload in payloads:
        collected.messages.append((0.1, payload))
    return collected


def test_a_healthy_set_of_utterances_passes():
    collected = with_utterances(
        utterance(0, 0, 2_000),
        utterance(1, 5_000, 6_000),
        vad_ends=2,
    )
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert report.failed == []


def test_no_utterances_at_all_fails():
    report = harness.Report()
    harness.check_utterances(harness.Collected(stream_start=0.0), report)
    assert [c.name for c in report.failed] == [
        "The server committed at least one sentence"
    ]


def test_a_gap_in_the_indexes_is_caught():
    collected = with_utterances(utterance(0, 0, 100), utterance(2, 200, 300))
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "Utterance indexes are consecutive from zero" in [
        c.name for c in report.failed
    ]


def test_a_sentence_longer_than_the_limit_is_caught():
    collected = with_utterances(utterance(0, 0, 9_000))
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "No sentence outstays the max duration" in [
        c.name for c in report.failed
    ]


def test_overlapping_sentences_are_caught():
    collected = with_utterances(utterance(0, 0, 2_000), utterance(1, 1_000, 3_000))
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "Sentences never overlap" in [c.name for c in report.failed]


def test_a_continuation_with_a_gap_is_caught():
    collected = with_utterances(
        utterance(0, 0, 2_000),
        utterance(1, 4_000, 5_000, "max_duration", continues=True),
    )
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "A continued sentence joins the previous one with no gap" in [
        c.name for c in report.failed
    ]


def test_a_closed_segment_with_no_sentence_is_caught():
    collected = with_utterances(utterance(0, 0, 1_000), vad_ends=3)
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "Every closed speech segment produced at least one sentence" in [
        c.name for c in report.failed
    ]


def test_a_server_without_the_noise_filter_is_flagged():
    server, url = serve_health({
        **GOOD_HEALTH,
        "noise_filter_loaded": False,
        "noise_filter_error": "Could not load 'MIT/ast-...'",
    })
    try:
        report = harness.Report()
        harness.check_health(url, report)
        assert [c.name for c in report.failed] == [
            "Server has the noise filter loaded"
        ]
    finally:
        server.shutdown()


def test_a_dropped_sentence_without_a_label_is_caught():
    collected = with_utterances(
        utterance(0, 0, 1_000),
        utterance(1, 2_000, 3_000, kept=False, label="", speech_score=0.01),
    )
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "Every dropped sentence says what it sounded like" in [
        c.name for c in report.failed
    ]


def test_a_filter_that_drops_everything_is_caught():
    collected = with_utterances(
        utterance(0, 0, 1_000, kept=False, label="Typing", speech_score=0.01),
    )
    report = harness.Report()
    harness.check_utterances(collected, report)
    assert "The noise filter did not eat the whole meeting" in [
        c.name for c in report.failed
    ]


# ---------------------------------------------------------------------------
# A stage missing from the server
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flag, name", [
    ("overlap_resolver_loaded", "overlap resolver"),
    ("speaker_model_loaded", "speaker model"),
    ("language_model_loaded", "language model"),
    ("asr_loaded", "ASR"),
    ("translation_loaded", "translation server"),
])
def test_each_missing_stage_is_named(flag, name):
    """A pod serving with a stage missing still answers; the client says which."""
    server, url = serve_health({**GOOD_HEALTH, flag: False})
    try:
        report = harness.Report()
        harness.check_health(url, report)
        assert [c.name for c in report.failed] == [f"Server has the {name} loaded"]
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Transcripts and translations
# ---------------------------------------------------------------------------
def final(transcript: str = "xin chào", speaker: str = "Speaker_01",
          lang: str = "vi", sentence_id: int = 1) -> dict:
    """A committed sentence. No translation - that is its own message now."""
    return {"type": "final", "sentence_id": sentence_id,
            "speaker_id": speaker, "lang_code": lang,
            "transcript": transcript}


def translation(text: str = "こんにちは", sentence_id: int = 1,
                reason: str = "", raw: str = "") -> dict:
    return {"type": "translation", "sentence_id": sentence_id,
            "translation": text, "reason": reason, "raw": raw}


def sentence(transcript: str = "xin chào", text: str = "こんにちは",
             speaker: str = "Speaker_01", lang: str = "vi",
             sentence_id: int = 1, reason: str = "", raw: str = "",
             at: float = 2.0, answered: bool = True) -> list:
    """A sentence and its answer, as the pair of messages they now are.

    ``answered=False`` models the failure that only exists now they are
    separate: a translation that never arrives at all.
    """
    pair = [(at, final(transcript, speaker, lang, sentence_id))]
    if answered:
        pair.append((at + 0.3, translation(text, sentence_id, reason, raw)))
    return pair


def partial(transcript: str = "xin", lang: str = "vi") -> dict:
    return {"type": "partial", "speaker_id": "", "lang_code": lang,
            "transcript": transcript}


def collected_of(*stamped) -> "harness.Collected":
    collected = harness.Collected(stream_start=0.0)
    collected.messages = list(stamped)
    return collected


def test_a_healthy_run_of_transcripts_passes():
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()), *sentence()), report)
    assert report.failed == []


def test_a_meeting_with_no_committed_sentence_is_caught():
    report = harness.Report()
    harness.check_transcripts(collected_of((1.0, partial())), report)
    assert [c.name for c in report.failed] == [
        "The meeting produced committed sentences"
    ]


def test_an_untranslated_sentence_is_caught():
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(text="", reason="no translation server",
                               raw="boom")),
        report)
    assert "Committed sentences come back translated" in [
        c.name for c in report.failed
    ]


def test_an_untranslated_sentence_with_no_reason_is_caught_twice():
    """Blank plus silent is worse than blank: it hides why."""
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()), *sentence(text="")), report)
    failed = [c.name for c in report.failed]
    assert "Committed sentences come back translated" in failed
    assert "Every untranslated sentence says why" in failed


def test_the_refusal_reason_reaches_the_screen(capsys):
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(text="",
                               reason="the answer is not written in ja")),
        harness.Report())
    out = capsys.readouterr().out
    assert "NOT translated: the answer is not written in ja" in out


# ---------------------------------------------------------------------------
# The answer has to be in the other language's script
# ---------------------------------------------------------------------------
def test_a_japanese_sentence_answered_in_japanese_is_caught():
    """The real one: はい、今の画面の came back as はい、現在の画面の, and
    every other check passed it."""
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(lang="ja", transcript="はい、今の画面の",
                               text="はい、現在の画面の")),
        report)
    assert "No translation came back in the language it started in" in [
        c.name for c in report.failed
    ]


def test_a_vietnamese_sentence_answered_in_vietnamese_is_caught():
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(lang="vi", transcript="Cái đó thì mình chưa xem.",
                               text="Cái đó mình chưa xem.")),
        report)
    assert "No translation came back in the language it started in" in [
        c.name for c in report.failed
    ]


def test_a_japanese_answer_keeping_a_latin_initialism_is_accepted():
    """Measured at 0.86 on the real run; the threshold has to leave room."""
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(lang="vi",
                               transcript="các FCG có tặng một cái thêm",
                               text="FCG が贈呈する追加分が完了したら")),
        report)
    assert report.failed == []


def test_a_vietnamese_answer_keeping_a_latin_word_is_accepted():
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(lang="ja", transcript="あのタスクの",
                               text="Task đó")),
        report)
    assert report.failed == []


def test_a_numeric_answer_is_not_judged_on_script():
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(lang="ja", transcript="15", text="15")),
        report)
    assert "No translation came back in the language it started in" not in [
        c.name for c in report.failed
    ]


def test_a_translation_identical_to_the_sentence_is_caught():
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(transcript="xin chào", text="xin chào")),
        report)
    assert "A translation is not just the sentence again" in [
        c.name for c in report.failed
    ]


def test_a_sentence_with_no_speaker_is_caught():
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()), *sentence(speaker="")), report)
    assert "Every committed sentence names a speaker" in [
        c.name for c in report.failed
    ]


def test_running_text_that_never_appeared_is_caught():
    report = harness.Report()
    harness.check_transcripts(collected_of(*sentence()), report)
    assert "Running text appeared before the sentences were committed" in [
        c.name for c in report.failed
    ]


def test_a_final_arriving_before_any_partial_is_caught():
    report = harness.Report()
    harness.check_transcripts(
        collected_of(*sentence(at=1.0), (2.0, partial())), report)
    assert "The first partial beat the first final" in [
        c.name for c in report.failed
    ]


# ---------------------------------------------------------------------------
# Whisper sign-offs, checked independently of the server
# ---------------------------------------------------------------------------
SIGN_OFF_VI = ("C\u1ea3m \u01a1n c\u00e1c b\u1ea1n \u0111\u00e3 theo d\u00f5i "
               "v\u00e0 h\u1eb9n g\u1eb7p l\u1ea1i.")


def test_the_sign_off_fixture_survived_being_written_to_disk():
    assert SIGN_OFF_VI.startswith("C\u1ea3m \u01a1n")
    assert harness.is_invented(SIGN_OFF_VI)


def test_a_sign_off_reaching_the_running_text_is_caught():
    """It never became a sentence on the real run, and still got read."""
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial(transcript=SIGN_OFF_VI)), *sentence()),
        report)
    assert "No running text is a Whisper sign-off" in [
        c.name for c in report.failed
    ]


def test_a_sign_off_reaching_a_committed_sentence_is_caught():
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(transcript=SIGN_OFF_VI, text="\u3055\u3088\u3046\u306a\u3089")),
        report)
    assert "No committed sentence is a Whisper sign-off" in [
        c.name for c in report.failed
    ]


def test_a_real_sentence_containing_the_phrase_is_not_caught():
    real = ("C\u1ea3m \u01a1n c\u00e1c b\u1ea1n \u0111\u00e3 theo d\u00f5i "
            "b\u00e1o c\u00e1o n\u00e0y.")
    assert not harness.is_invented(real)


def test_the_confirmed_goodbye_is_caught():
    """Confirmed absent from the recording it was transcribed from."""
    assert harness.is_invented("Chào tạm biệt.")


def test_a_goodbye_with_anything_attached_survives():
    assert not harness.is_invented("Chào tạm biệt nhé.")
    assert not harness.is_invented("Tạm biệt.")


# ---------------------------------------------------------------------------
# What the model actually said
# ---------------------------------------------------------------------------
def test_a_refusal_that_hides_the_answer_is_caught():
    """"far longer than the sentence" reads the same whether the model
    rambled or the limit was too tight. Only the text tells them apart."""
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(text="",
                               reason="the answer is far longer than the sentence")),
        report)
    assert "Every untranslated sentence shows what the model said" in [
        c.name for c in report.failed
    ]


def test_a_refusal_that_shows_the_answer_passes():
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(text="",
                               reason="the answer is far longer than the sentence",
                               raw="a long rambling answer")),
        report)
    assert "Every untranslated sentence shows what the model said" not in [
        c.name for c in report.failed
    ]


def test_a_model_that_said_nothing_has_nothing_to_show():
    """Demanding text from a refusal for producing no text would be a check
    that cannot pass."""
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(text="",
                               reason="the model returned nothing")),
        report)
    assert "Every untranslated sentence shows what the model said" not in [
        c.name for c in report.failed
    ]


def test_the_raw_answer_reaches_the_screen(capsys):
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     *sentence(text="", reason="too long",
                               raw="Sure! Here is the translation: ...")),
        harness.Report())
    assert "the model said: Sure! Here is the translation" in capsys.readouterr().out


def test_a_successful_translation_shows_no_raw(capsys):
    """It would be noise: the translation is right there.

    Matched on the printed line, not the phrase - the check's own name
    contains the phrase, which made an earlier version of this fail on its
    own report.
    """
    harness.check_transcripts(
        collected_of((1.0, partial()), *sentence()), harness.Report())
    assert "the model said:" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The sentence and its translation are two messages
# ---------------------------------------------------------------------------
def test_a_translation_that_never_arrives_is_caught():
    """The failure that only exists now they are separate. An unbounded queue
    falling behind would look exactly like this."""
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()), *sentence(answered=False)), report)
    assert "Every sentence got an answer about its translation" in [
        c.name for c in report.failed
    ]


def test_a_missing_translation_is_named_on_screen(capsys):
    harness.check_transcripts(
        collected_of((1.0, partial()), *sentence(answered=False)),
        harness.Report())
    assert "NO TRANSLATION MESSAGE EVER ARRIVED" in capsys.readouterr().out


def test_translations_are_matched_by_sentence_id_not_by_order():
    """They arrive when the model finishes, which need not be in order."""
    messages = [(1.0, partial())]
    messages += [(2.0, final("first", sentence_id=1)),
                 (2.1, final("second", sentence_id=2)),
                 (2.5, translation("二番目", sentence_id=2)),
                 (3.0, translation("一番目", sentence_id=1))]
    collected = collected_of(*messages)
    assert collected.translation_for(1)["translation"] == "一番目"
    assert collected.translation_for(2)["translation"] == "二番目"


def test_a_translation_for_an_unknown_sentence_is_not_matched():
    collected = collected_of((2.0, final(sentence_id=1)),
                             (2.5, translation(sentence_id=99)))
    assert collected.translation_for(1) is None


def test_the_lag_is_measured_from_the_sentence_to_its_translation():
    collected = collected_of((2.0, final(sentence_id=1)),
                             (3.5, translation(sentence_id=1)))
    assert collected.translation_lags() == [(1, pytest.approx(1.5))]


def test_a_translation_drifting_far_behind_its_sentence_is_caught():
    """The number the split exists to keep small."""
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()),
                     (2.0, final(sentence_id=1)),
                     (2.0 + harness.MAX_TRANSLATION_LAG_S + 1,
                      translation(sentence_id=1))),
        report)
    assert "Translations keep up with their sentences" in [
        c.name for c in report.failed
    ]


def test_a_prompt_translation_passes_the_lag_check():
    report = harness.Report()
    harness.check_transcripts(
        collected_of((1.0, partial()), *sentence()), report)
    assert "Translations keep up with their sentences" not in [
        c.name for c in report.failed
    ]


def test_the_client_lag_budget_is_tighter_than_the_servers():
    """So drift shows up in the test before the server starts dropping."""
    from server.config import TRANSLATION_MAX_LAG_SECONDS
    assert harness.MAX_TRANSLATION_LAG_S < TRANSLATION_MAX_LAG_SECONDS


# ---------------------------------------------------------------------------
# Missing audio must not be reported as a slow server
# ---------------------------------------------------------------------------
def late_events(lag_ms: float = 13_000.0):
    """One event, arriving `lag_ms` after the audio it describes."""
    return make_collected(0.0, [(lag_ms / 1000.0, 0.0, "speech_start"),
                                (lag_ms / 1000.0 + 1, 1000.0, "speech_end")])


def test_a_slow_server_is_still_caught_when_all_the_audio_arrived():
    report = harness.Report()
    harness.check_events(late_events(), report,
                         audio_seconds=120.0, wall_seconds=120.0)
    assert "Events come back fast enough to be useful" in [
        c.name for c in report.failed
    ]


def test_missing_audio_is_not_blamed_on_the_server():
    """A run sent 107 s of audio over 120 s and reported a flat 13 s lag on
    every event. The server had done nothing wrong."""
    report = harness.Report()
    harness.check_events(late_events(), report,
                         audio_seconds=107.0, wall_seconds=120.0)
    assert "Events come back fast enough to be useful" not in [
        c.name for c in report.checks
    ]


def test_missing_audio_says_so_on_screen(capsys):
    """Refusing to judge without saying why is just a check that vanished."""
    harness.check_events(late_events(), harness.Report(),
                         audio_seconds=107.0, wall_seconds=120.0)
    out = capsys.readouterr().out
    assert "NOT JUDGED" in out
    assert "13.0 s of audio never left this machine" in out


def test_a_run_ending_mid_chunk_is_still_judged():
    """A fraction of a second short is how every run ends."""
    report = harness.Report()
    collected = make_collected(0.0, [(0.6, 200.0, "speech_start"),
                                     (5.4, 5000.0, "speech_end")])
    harness.check_events(collected, report,
                         audio_seconds=119.8, wall_seconds=120.0)
    assert report.failed == []
    assert "Events come back fast enough to be useful" in [
        c.name for c in report.checks
    ]


def test_the_lag_is_still_printed_when_it_is_not_judged(capsys):
    """The numbers are still worth reading; they just cannot be scored."""
    harness.check_events(late_events(), harness.Report(),
                         audio_seconds=107.0, wall_seconds=120.0)
    assert "End-to-end lag:" in capsys.readouterr().out


def test_without_a_wall_clock_nothing_is_assumed_missing():
    """The default for callers that have no figure to give."""
    report = harness.Report()
    harness.check_events(late_events(), report)
    assert "Events come back fast enough to be useful" in [
        c.name for c in report.checks
    ]


# ---------------------------------------------------------------------------
# The audio pump
#
# This loop is the one piece of the file that needs a sound card, so no test
# had ever run it - and a NameError in it reached the Windows machine and cost
# a whole 120 s run. It is split out now precisely so these can exist.
# ---------------------------------------------------------------------------
class StubCaptureStats:
    callbacks = 0
    chunks_emitted = 0
    chunks_dropped = 0
    input_overflows = 0


class StubCapture:
    """Hands back a scripted sequence of chunks, or stalls instead.

    A ``None`` in the script is a read that timed out: no audio, and time
    passing while none of it was recorded.
    """

    def __init__(self, script, stall_seconds: float = 0.6):
        self.script = list(script)
        self.stall_seconds = stall_seconds
        self.calls = 0
        self.stats = StubCaptureStats()

    def read(self, timeout=None):
        chunk = self.script[self.calls] if self.calls < len(self.script) else b""
        self.calls += 1
        if chunk is None:
            time.sleep(self.stall_seconds)
            return None
        return chunk

    def stop(self):
        pass


class StubStreamClient:
    """Records what the pump sent it. Not FakeClient, which models the stats
    of a finished run rather than a live one."""

    def __init__(self):
        self.sent: list[bytes] = []
        self.queued_chunks = 0
        self.stats = type("S", (), {"chunks_sent": 0, "chunks_dropped": 0})()

    def send(self, chunk: bytes) -> None:
        self.sent.append(chunk)
        self.stats.chunks_sent += 1


def run_pump(capture, seconds: float):
    collected = harness.Collected(stream_start=time.perf_counter())
    client = StubStreamClient()
    gaps = asyncio.run(harness.pump_audio(capture, client, collected, seconds))
    return client, gaps


def test_the_pump_sends_every_chunk_it_reads():
    client, gaps = run_pump(StubCapture([b"a" * 6400] * 3), seconds=0.25)
    assert client.sent
    assert gaps == []


def test_a_stalled_read_is_recorded_as_a_gap():
    """Audio that would have filled it was never recorded. That is not a slow
    server, however similar the lag figures look."""
    capture = StubCapture([None, b"a" * 6400], stall_seconds=0.5)
    _client, gaps = run_pump(capture, seconds=0.9)
    assert gaps, "the stall left no trace"
    at, waited = gaps[0]
    assert waited >= 0.5
    assert at >= 0


def test_a_prompt_read_is_not_recorded_as_a_gap():
    """One chunk is 200 ms; a read finishing inside that is the normal case."""
    _client, gaps = run_pump(StubCapture([b"a" * 6400] * 20), seconds=0.3)
    assert gaps == []


def test_a_timed_out_read_sends_nothing():
    client, _gaps = run_pump(StubCapture([None], stall_seconds=0.5), seconds=0.4)
    assert client.sent == []


def test_the_pump_stops_at_the_deadline():
    started = time.perf_counter()
    run_pump(StubCapture([b"a" * 6400] * 10_000), seconds=0.3)
    assert time.perf_counter() - started < 3.0


def test_capture_gaps_are_reported_with_their_timing(capsys):
    harness.report_capture_gaps(StubCapture([]), [(8.2, 10.4), (1.0, 0.5)])
    out = capsys.readouterr().out
    assert "stalled 2 time(s), 10.9 s in total" in out
    assert "t=   8.2s  waited 10.40s" in out


def test_a_clean_capture_reports_only_its_counters(capsys):
    harness.report_capture_gaps(StubCapture([]), [])
    out = capsys.readouterr().out
    assert "Capture:" in out
    assert "stalled" not in out


# ---------------------------------------------------------------------------
# stream_for, end to end, with no sound card
#
# The NameError that cost a run lived on the line joining the capture loop to
# the report. Neither piece was at fault; the join was, and nothing executed
# the join. These do.
# ---------------------------------------------------------------------------
class StubDevice:
    name = "Stub Loopback"


class StubLoopbackCapture:
    """Stands in for the WASAPI capture. Records that it was stopped."""

    last: "StubLoopbackCapture | None" = None

    def __init__(self, device_name_hint=None):
        self.script = [b"a" * 6400] * 4
        self.calls = 0
        self.stopped = False
        self.stats = StubCaptureStats()
        StubLoopbackCapture.last = self

    def start(self):
        return StubDevice()

    def read(self, timeout=None):
        # A real device paces the loop; without that the pump spins as fast
        # as the CPU allows and buries the test output in progress lines.
        time.sleep(0.02)
        chunk = self.script[self.calls] if self.calls < len(self.script) else b""
        self.calls += 1
        return chunk

    def stop(self):
        self.stopped = True


class StubSocketClient:
    """Stands in for StreamClient: connects, accepts chunks, stops."""

    def __init__(self, url, on_message=None, connects: bool = True):
        self.url = url
        self.on_message = on_message
        self._connects = connects
        self.session_id = "stub-session"
        self.sent: list[bytes] = []
        self.queued_chunks = 0
        self.stopped = False
        self.stats = type("S", (), {
            "chunks_sent": 0, "chunks_dropped": 0, "errors": [],
            "connects": 1, "disconnects": 0, "messages_received": 0,
            "audio_seconds_sent": 0.0,
        })()

    async def run(self):
        return None

    async def wait_connected(self, timeout=10.0):
        return self._connects

    def send(self, chunk):
        self.sent.append(chunk)
        self.stats.chunks_sent += 1

    async def stop(self, drain_timeout=0.0):
        self.stopped = True


def patch_stream(monkeypatch, connects: bool = True):
    monkeypatch.setattr(harness, "LoopbackCapture", StubLoopbackCapture)
    monkeypatch.setattr(
        harness, "StreamClient",
        lambda url, on_message=None: StubSocketClient(url, on_message, connects))


def test_stream_for_runs_end_to_end_without_a_sound_card(monkeypatch, capsys):
    patch_stream(monkeypatch)
    report = harness.Report()
    client, collected = asyncio.run(
        harness.stream_for("ws://stub", None, 0.3, report))
    assert client.sent, "no audio reached the client"
    assert collected.stream_start > 0
    assert report.failed == []
    out = capsys.readouterr().out
    assert "Loopback device opened" in out
    assert "Capture:" in out          # the join that used to raise


def test_stream_for_stops_the_capture_and_the_client(monkeypatch):
    patch_stream(monkeypatch)
    client, _collected = asyncio.run(
        harness.stream_for("ws://stub", None, 0.3, harness.Report()))
    assert StubLoopbackCapture.last.stopped
    assert client.stopped


def test_stream_for_stops_the_capture_even_when_the_loop_raises(monkeypatch):
    """A device left running holds the endpoint open for the next run."""
    patch_stream(monkeypatch)

    async def explode(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(harness, "pump_audio", explode)
    with pytest.raises(RuntimeError):
        asyncio.run(harness.stream_for("ws://stub", None, 0.3, harness.Report()))
    assert StubLoopbackCapture.last.stopped


def test_a_handshake_that_never_completes_opens_no_device(monkeypatch):
    patch_stream(monkeypatch, connects=False)
    StubLoopbackCapture.last = None
    report = harness.Report()
    asyncio.run(harness.stream_for("ws://stub", None, 0.3, report))
    assert [c.name for c in report.failed] == ["Handshake accepted"]
    assert StubLoopbackCapture.last is None, "the device was opened anyway"
