"""Unit tests for the meeting window itself.

Runs on Qt's offscreen platform, so no display is needed. What these check is
the wiring: that a server message reaches the screen, that the counts follow
the model, and that closing the window ends the session - the server needs
the goodbye to finish translating the last sentence.

Run with::

    .venv\\Scripts\\python.exe -m pytest client/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("PySide6", reason="the UI needs PySide6 installed")

from client.ui.window import MeetingWindow            # noqa: E402


@pytest.fixture
def window(qt_app, monkeypatch):
    """A window whose session never actually starts."""
    win = MeetingWindow("ws://127.0.0.1:8000")
    monkeypatch.setattr(win.session, "start", lambda: None)
    monkeypatch.setattr(win.session, "stop", lambda timeout=5.0: None)
    yield win
    win.close()


def final(sentence_id: int = 1, text: str = "Xin chào mọi người.",
          speaker: str = "Speaker_01", lang: str = "vi") -> dict:
    return {"type": "final", "sentence_id": sentence_id, "speaker_id": speaker,
            "lang_code": lang, "transcript": text, "speech_score": 0.85}


def translation(sentence_id: int = 1, text: str = "こんにちは皆様。",
                reason: str = "", raw: str = "") -> dict:
    return {"type": "translation", "sentence_id": sentence_id,
            "translation": text, "reason": reason, "raw": raw}


# ---------------------------------------------------------------------------
# What ends up on screen
# ---------------------------------------------------------------------------
def test_an_empty_window_explains_itself(window):
    assert "Bắt đầu" in window.transcript.toPlainText()


def test_a_committed_sentence_appears(window):
    window.on_message(final())
    assert "Xin chào mọi người." in window.transcript.toPlainText()


def test_its_translation_appears_under_it(window):
    window.on_message(final(sentence_id=3))
    window.on_message(translation(sentence_id=3))
    text = window.transcript.toPlainText()
    assert "Xin chào mọi người." in text
    assert "こんにちは皆様。" in text


def test_running_text_appears_and_is_replaced_by_the_sentence(window):
    window.on_message({"type": "partial", "transcript": "Xin chào mọi ng",
                       "lang_code": "vi"})
    assert "Xin chào mọi ng" in window.transcript.toPlainText()
    window.on_message(final())
    text = window.transcript.toPlainText()
    assert text.count("Xin chào mọi") == 1, "the prediction was left on screen"


def test_a_refusal_is_visible_rather_than_a_blank(window):
    window.on_message(final(sentence_id=2))
    window.on_message(translation(sentence_id=2, text="",
                                  reason="the answer is not written in ja",
                                  raw="OK. I will view it."))
    text = window.transcript.toPlainText()
    assert "không dịch được" in text
    assert "OK. I will view it." in text


def test_refusals_can_be_hidden(window):
    window.on_message(final(sentence_id=2))
    window.on_message(translation(sentence_id=2, text="", reason="too long"))
    window.show_refusals.setChecked(False)
    assert "không dịch được" not in window.transcript.toPlainText()
    assert "Xin chào mọi người." in window.transcript.toPlainText()


# ---------------------------------------------------------------------------
# The counters
# ---------------------------------------------------------------------------
def test_the_counts_follow_the_meeting(window):
    window.on_message(final(sentence_id=1))
    window.on_message(final(sentence_id=2, speaker="Speaker_02"))
    window.on_message(translation(sentence_id=1))
    counts = window.counts.text()
    assert "2 câu" in counts
    assert "1 đã dịch" in counts
    assert "1 đang chờ" in counts
    assert "2 người nói" in counts


def test_a_dropped_utterance_is_counted_on_screen(window):
    """So a noise filter that starts eating speech is visible."""
    window.on_message({"type": "utterance", "index": 0, "start_ms": 0.0,
                       "end_ms": 100.0, "reason": "pause", "kept": False,
                       "label": "Breathing"})
    assert "1 bị lọc" in window.counts.text()


# ---------------------------------------------------------------------------
# Session control
# ---------------------------------------------------------------------------
def test_starting_clears_the_previous_meeting(window, monkeypatch):
    window.on_message(final())
    monkeypatch.setattr(type(window.session), "running", property(lambda _: False))
    window.toggle()
    assert window.model.sentences == []
    assert "Xin chào" not in window.transcript.toPlainText()


def test_the_button_says_what_it_will_do(window, monkeypatch):
    monkeypatch.setattr(type(window.session), "running", property(lambda _: False))
    window.toggle()
    assert window.start_button.text() == "Dừng"
    window.on_stopped()
    assert window.start_button.text() == "Bắt đầu"


def test_closing_the_window_ends_the_session(qt_app, monkeypatch):
    """The server commits and translates the last sentence on the goodbye."""
    win = MeetingWindow("ws://127.0.0.1:8000")
    stopped = []
    monkeypatch.setattr(win.session, "stop",
                        lambda timeout=5.0: stopped.append(True))
    win.close()
    assert stopped, "the session was left running"


def test_a_failure_reaches_the_status_bar(window):
    window.on_failed("Không kết nối được máy chủ")
    assert "Không kết nối được" in window.statusBar().currentMessage()
