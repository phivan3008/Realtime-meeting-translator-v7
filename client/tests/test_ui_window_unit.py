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
    assert "Xin chào mọi ng" in window.running_text.text()
    window.on_message(final())
    assert "Xin chào mọi người." in window.transcript.toPlainText()
    assert window.running_text.text() == "", "the prediction was left on screen"


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


# ---------------------------------------------------------------------------
# Following the meeting
# ---------------------------------------------------------------------------
def fill(window, count: int = 40) -> None:
    """Enough sentences that the transcript scrolls."""
    window.resize(420, 200)
    window.show()
    for index in range(1, count + 1):
        window.on_message(final(sentence_id=index, text=f"Câu số {index}"))


def bar_of(window):
    return window.transcript.verticalScrollBar()


def at_bottom(window) -> bool:
    bar = bar_of(window)
    return bar.value() >= bar.maximum() - 4


def test_the_view_follows_new_sentences(window):
    fill(window)
    assert bar_of(window).maximum() > 0, "the transcript did not scroll at all"
    assert at_bottom(window)


def test_scrolling_back_survives_the_next_sentence(window):
    """setHtml rebuilds the document and resets the scrollbar to zero."""
    fill(window)
    bar = bar_of(window)
    bar.setValue(bar.maximum() // 2)
    keep = bar.value()
    window.on_message(final(sentence_id=99, text="Câu mới"))
    assert abs(bar_of(window).value() - keep) <= 4


def test_resizing_does_not_end_the_follow(window, qt_app):
    """The reported failure: after a resize it jumped to the top and stayed.

    The resize has to be laid out before it counts - it is the new scrollbar
    range, not the resize call, that used to end the follow. It also has to
    make the window *smaller*: growing it shrinks the scroll range, and Qt
    clamps the position down to the new maximum, which lands at the bottom
    by accident.
    """
    fill(window)
    was = bar_of(window).maximum()
    window.resize(300, 120)
    qt_app.processEvents()
    assert bar_of(window).maximum() > was, "the resize left the range alone"
    window.on_message(final(sentence_id=99, text="Câu mới"))
    assert at_bottom(window)


def test_the_running_text_does_not_disturb_the_document(window):
    """It arrives every 600 ms; redrawing 500 rows that often is the lag."""
    fill(window)
    drawn = []
    window.transcript.document().contentsChanged.connect(
        lambda: drawn.append(True))
    for index in range(10):
        window.on_message({"type": "partial", "lang_code": "vi",
                           "transcript": f"đang nói {index}"})
    assert drawn == [], "the running text rebuilt the transcript"
    assert "đang nói 9" in window.running_text.text()


# ---------------------------------------------------------------------------
# The button while the session is changing state
# ---------------------------------------------------------------------------
def test_the_button_is_disabled_while_connecting(window, monkeypatch):
    monkeypatch.setattr(type(window.session), "running", property(lambda _: False))
    window.toggle()
    assert not window.start_button.isEnabled()
    window.on_status("Đã kết nối · phiên abc")
    assert window.start_button.isEnabled()


def test_a_failed_connection_gives_the_button_back(window, monkeypatch):
    monkeypatch.setattr(type(window.session), "running", property(lambda _: False))
    window.toggle()
    window.on_failed("Không kết nối được máy chủ")
    assert window.start_button.isEnabled()


def test_the_button_is_disabled_while_stopping(window, monkeypatch):
    monkeypatch.setattr(type(window.session), "running", property(lambda _: True))
    window.toggle()
    assert not window.start_button.isEnabled()
    window.on_stopped()
    assert window.start_button.isEnabled()


def test_stopping_never_joins_the_worker_on_the_qt_thread(window, monkeypatch):
    """A join here freezes the window for as long as the goodbye takes."""
    joined = []
    asked = []
    monkeypatch.setattr(window.session, "stop",
                        lambda timeout=5.0: joined.append(timeout))
    monkeypatch.setattr(window.session, "request_stop", lambda: asked.append(True))
    monkeypatch.setattr(type(window.session), "running", property(lambda _: True))
    window.toggle()
    assert asked and not joined


def test_scrolling_back_near_the_bottom_resumes_the_follow(window):
    """Reported after four minutes: stuck on one sentence, and dragging the
    bar did not free it. A four-pixel target on a document tens of thousands
    of pixels tall cannot be hit by hand, and every new sentence pinned the
    reader back to the offset they were at."""
    fill(window, 60)
    bar = bar_of(window)
    bar.setValue(bar.maximum() // 3)
    window.on_message(final(sentence_id=98, text="Câu mới"))
    assert not at_bottom(window), "scrolling back was ignored"

    bar.setValue(bar.maximum() - 20)          # near the bottom, not exactly
    window.on_message(final(sentence_id=99, text="Câu mới nữa"))
    assert at_bottom(window), "the reader could not get back to the meeting"


def test_a_redraw_does_not_count_as_the_reader_scrolling(window):
    """Hiding refusals shrinks the document under a reader who is scrolled
    back. The shorter document can leave them at its new bottom, and that is
    the widget moving, not the reader - it must not restart the follow."""
    fill(window, 60)
    for index in range(61, 90):
        window.on_message(final(sentence_id=index, text=f"Câu {index}"))
        window.on_message(translation(sentence_id=index, text="",
                                      reason="quá dài", raw="x" * 200))
    bar = bar_of(window)
    bar.setValue(int(bar.maximum() * 0.55))
    window.on_message(final(sentence_id=95, text="Câu mới"))
    assert window._follow is False

    window.show_refusals.setChecked(False)
    window.on_message(final(sentence_id=96, text="Câu mới nữa"))
    # The position may land at the bottom - a shorter document has nowhere
    # else to put it - but that is the widget moving, not a decision.
    assert window._follow is False, "the follow restarted on its own"


# ---------------------------------------------------------------------------
# Keeping the meeting
# ---------------------------------------------------------------------------
def test_a_meeting_is_written_to_disk(qt_app, monkeypatch, tmp_path):
    win = MeetingWindow("ws://127.0.0.1:8000", out_dir=tmp_path)
    monkeypatch.setattr(win.session, "start", lambda: None)
    monkeypatch.setattr(win.session, "stop", lambda timeout=5.0: None)
    monkeypatch.setattr(type(win.session), "running", property(lambda _: False))
    win.toggle()
    win.on_message(final())
    win.on_message(translation())
    win.on_stopped()
    win.close()

    written = sorted(path.name for path in tmp_path.iterdir())
    assert len(written) == 2, written
    minutes = next(p for p in tmp_path.iterdir() if ".debug" not in p.name)
    text = minutes.read_text(encoding="utf-8")
    assert "Xin chào mọi người." in text
    assert "こんにちは皆様。" in text


def test_nothing_is_written_when_no_directory_was_asked_for(qt_app, monkeypatch):
    win = MeetingWindow("ws://127.0.0.1:8000")
    monkeypatch.setattr(win.session, "start", lambda: None)
    monkeypatch.setattr(win.session, "stop", lambda timeout=5.0: None)
    monkeypatch.setattr(type(win.session), "running", property(lambda _: False))
    win.toggle()
    win.on_message(final())
    assert win.recorder is None
    win.close()


def test_the_status_bar_says_where_the_meeting_went(window, monkeypatch, tmp_path):
    window.out_dir = tmp_path
    monkeypatch.setattr(type(window.session), "running", property(lambda _: False))
    window.toggle()
    window.on_stopped()
    assert "biên bản" in window.statusBar().currentMessage()
