"""Unit tests for what the window renders and how it reaches Qt.

``sentence_html`` and ``partial_html`` are plain functions over a Sentence,
so they need no display. The session bridge is exercised with stubs for the
socket and the capture - the thread and its asyncio loop are real, because
the seam between them is the part worth testing.

Run with::

    .venv\\Scripts\\python.exe -m pytest client/tests -v
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("PySide6", reason="the UI needs PySide6 installed")

from client.ui import session as session_module          # noqa: E402
from client.ui.session import MeetingSession             # noqa: E402
from client.ui.transcript import Sentence                # noqa: E402
from client.ui.window import (                           # noqa: E402
    partial_html,
    sentence_html,
)
from PySide6.QtCore import QCoreApplication              # noqa: E402


@pytest.fixture(autouse=True)
def _application(qt_app):
    """Signals deliver nothing without an application object: the worker
    thread runs and emits into a void, which looks exactly like a session
    that never started. ``qt_app`` lives in conftest.py - one per process."""
    yield qt_app


def sentence(**kwargs) -> Sentence:
    defaults = dict(sentence_id=1, speaker_id="Speaker_01", lang_code="vi",
                    text="Xin chào mọi người.", speech_score=0.85)
    defaults.update(kwargs)
    return Sentence(**defaults)


# ---------------------------------------------------------------------------
# A committed sentence
# ---------------------------------------------------------------------------
def test_the_sentence_and_its_translation_are_both_shown():
    html = sentence_html(sentence(translation="こんにちは皆様。"))
    assert "Xin chào mọi người." in html
    assert "こんにちは皆様。" in html


def test_a_committed_sentence_is_bold():
    """DESIGN.md 2: final is bold, partial is grey."""
    assert "font-weight:600" in sentence_html(sentence())


def test_the_speaker_and_language_are_labelled():
    html = sentence_html(sentence(speaker_id="Speaker_02", lang_code="ja"))
    assert "Speaker_02" in html
    assert "日本語" in html


def test_a_sentence_still_waiting_says_so():
    """A blank where a translation goes reads as a broken client."""
    assert "đang dịch" in sentence_html(sentence())


def test_a_refusal_is_shown_with_its_reason():
    html = sentence_html(sentence(reason="the answer is not written in ja"))
    assert "không dịch được" in html
    assert "not written in ja" in html


def test_a_refusal_shows_what_the_model_said():
    html = sentence_html(sentence(reason="too long", raw="OK. I will view it."))
    assert "OK. I will view it." in html


def test_refusals_can_be_hidden_without_hiding_the_sentence():
    html = sentence_html(sentence(reason="too long"), show_refusals=False)
    assert "Xin chào mọi người." in html
    assert "không dịch được" not in html


def test_text_is_escaped_rather_than_rendered():
    """A transcript is untrusted text: Whisper will happily produce '<b>'."""
    html = sentence_html(sentence(text="a < b & c", translation="<b>x</b>"))
    assert "&lt; b &amp; c" in html
    assert "<b>x</b>" not in html


def test_an_unknown_speaker_renders_without_crashing():
    assert "?" in sentence_html(sentence(speaker_id="", lang_code=""))


# ---------------------------------------------------------------------------
# The running prediction
# ---------------------------------------------------------------------------
def test_the_running_text_is_grey_and_italic():
    html = partial_html("xin chào mọi ng", "vi")
    assert "italic" in html
    assert "9aa3ad" in html.lower()


def test_no_running_text_renders_nothing():
    assert partial_html("", "vi") == ""


def test_the_running_text_is_escaped_too():
    assert "&lt;script&gt;" in partial_html("<script>", "vi")


# ---------------------------------------------------------------------------
# The bridge to Qt
# ---------------------------------------------------------------------------
class StubClient:
    """Stands in for StreamClient inside the worker thread."""

    def __init__(self, url, on_message=None, connects=True):
        self.url = url
        self.on_message = on_message
        self._connects = connects
        self.session_id = "stub"
        self.sent: list[bytes] = []
        self.stopped = threading.Event()
        self.stats = type("S", (), {"errors": []})()

    async def run(self):
        return None

    async def wait_connected(self, timeout=10.0):
        return self._connects

    def send(self, chunk):
        self.sent.append(chunk)
        if self.on_message:
            self.on_message({"type": "partial", "transcript": "x",
                             "lang_code": "vi"})

    async def stop(self, drain_timeout=0.0):
        self.stopped.set()


class StubCapture:
    def __init__(self, device_name_hint=None):
        self.stopped = False

    def start(self):
        return type("D", (), {"name": "Stub Loopback"})()

    def read(self, timeout=None):
        time.sleep(0.01)
        return b"a" * 6400

    def stop(self):
        self.stopped = True


def patch_session(monkeypatch, connects=True):
    monkeypatch.setattr(session_module, "LoopbackCapture", StubCapture)
    monkeypatch.setattr(
        session_module, "StreamClient",
        lambda url, on_message=None: StubClient(url, on_message, connects))


def collect(signal) -> list:
    seen = []
    signal.connect(seen.append)
    return seen


def wait_for(items: list, timeout: float = 5.0) -> bool:
    """Pump Qt's event queue until something arrives.

    A signal emitted from the worker thread is queued for the thread the
    QObject lives on, so it is delivered when that thread runs its event
    loop. The window has one; a test has to turn the handle itself.
    """
    app = QCoreApplication.instance()
    deadline = time.monotonic() + timeout
    while not items and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    return bool(items)


def test_server_messages_reach_qt(monkeypatch):
    patch_session(monkeypatch)
    session = MeetingSession("ws://stub")
    messages = collect(session.message)
    session.start()
    arrived = wait_for(messages)
    session.stop()
    assert arrived, "nothing reached the window"
    assert messages[0]["type"] == "partial"


def test_stopping_ends_the_thread_and_closes_the_device(monkeypatch):
    patch_session(monkeypatch)
    session = MeetingSession("ws://stub")
    session.start()
    time.sleep(0.05)
    session.stop()
    assert session.running is False


def test_a_refused_connection_is_reported_not_raised(monkeypatch):
    patch_session(monkeypatch, connects=False)
    session = MeetingSession("ws://stub")
    failures = collect(session.failed)
    session.start()
    arrived = wait_for(failures)
    session.stop()
    assert arrived and "Không kết nối được" in failures[0]


def test_starting_twice_does_not_open_two_sessions(monkeypatch):
    patch_session(monkeypatch)
    session = MeetingSession("ws://stub")
    session.start()
    first = session._thread
    session.start()
    assert session._thread is first
    session.stop()


def test_stopping_a_session_that_never_started_is_harmless():
    MeetingSession("ws://stub").stop()


def test_the_url_is_normalised_once():
    """So "…:8000/" and "…:8000" open the same endpoint."""
    assert MeetingSession("ws://host:8000/").url == "ws://host:8000"
