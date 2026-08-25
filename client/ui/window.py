"""The meeting window.

``DESIGN.md`` section 2: two states on screen. Running text is grey and
italic because it will be replaced within a second; a committed sentence is
bold, with its translation under it.

The window holds no state of its own - :class:`TranscriptModel` does, and
this redraws from it. A refusal is shown as such rather than as an empty
line, because a blank where a translation should be reads as a bug in the
client.
"""

from __future__ import annotations

import html
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from client.ui.session import MeetingSession
from client.ui.transcript import Sentence, TranscriptModel

LANGUAGE_NAMES = {"vi": "Tiếng Việt", "ja": "日本語", "": "?"}


def sentence_html(sentence: Sentence, show_refusals: bool = True) -> str:
    """One committed sentence: what was said, and what it became."""
    who = html.escape(sentence.speaker_id or "?")
    lang = LANGUAGE_NAMES.get(sentence.lang_code, sentence.lang_code)
    said = html.escape(sentence.text)

    rows = [
        f'<div style="margin-top:14px">'
        f'<span style="color:#7a8290;font-size:11px">{who} · {lang}</span></div>',
        f'<div style="font-weight:600;font-size:15px">{said}</div>',
    ]

    if sentence.translated:
        rows.append(f'<div style="color:#1f6feb;font-size:15px">'
                    f'{html.escape(sentence.translation)}</div>')
    elif sentence.answered and show_refusals:
        reason = html.escape(sentence.reason)
        rows.append(f'<div style="color:#a1471f;font-size:12px">'
                    f'không dịch được — {reason}</div>')
        if sentence.raw:
            rows.append(f'<div style="color:#8a6d3b;font-size:11px">'
                        f'máy trả lời: {html.escape(sentence.raw)}</div>')
    elif not sentence.answered:
        rows.append('<div style="color:#9aa3ad;font-size:13px">đang dịch…</div>')
    return "".join(rows)


def partial_html(text: str, lang_code: str) -> str:
    """The running prediction: grey, italic, and about to be replaced."""
    if not text:
        return ""
    lang = LANGUAGE_NAMES.get(lang_code, lang_code)
    return (f'<div style="margin-top:14px;color:#9aa3ad;font-style:italic">'
            f'[{lang}] {html.escape(text)}</div>')


class MeetingWindow(QMainWindow):
    """Live transcript and translation for one meeting."""

    def __init__(self, url: str, device_hint: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Phiên dịch cuộc họp VI ↔ JA")
        self.resize(900, 640)

        self.model = TranscriptModel()
        self.session = MeetingSession(url, device_hint, parent=self)
        self.session.message.connect(self.on_message)
        self.session.status.connect(self.on_status)
        self.session.failed.connect(self.on_failed)
        self.session.stopped.connect(self.on_stopped)

        self._build()
        self._redraw()

    # -- layout -------------------------------------------------------------
    def _build(self) -> None:
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(False)
        self.transcript.setFont(QFont("Segoe UI", 11))

        self.start_button = QPushButton("Bắt đầu")
        self.start_button.clicked.connect(self.toggle)
        self.show_refusals = QCheckBox("Hiện câu không dịch được")
        self.show_refusals.setChecked(True)
        self.show_refusals.stateChanged.connect(lambda _: self._redraw())
        self.counts = QLabel()
        self.counts.setStyleSheet("color:#7a8290")

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.show_refusals)
        controls.addStretch(1)
        controls.addWidget(self.counts)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.transcript, 1)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Chưa kết nối")

    # -- session ------------------------------------------------------------
    def toggle(self) -> None:
        if self.session.running:
            self.statusBar().showMessage("Đang kết thúc phiên…")
            self.session.stop()
            return
        self.model.clear()
        self._redraw()
        self.start_button.setText("Dừng")
        self.session.start()

    def on_message(self, message: dict) -> None:
        if self.model.apply(message) is not None or message.get("type") in {
            "partial", "utterance"
        }:
            self._redraw()

    def on_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def on_failed(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def on_stopped(self) -> None:
        self.start_button.setText("Bắt đầu")
        self.statusBar().showMessage("Đã dừng")

    def closeEvent(self, event) -> None:                    # noqa: N802 - Qt
        # The server needs the goodbye to finish the last sentence.
        self.session.stop()
        super().closeEvent(event)

    # -- drawing ------------------------------------------------------------
    def _redraw(self) -> None:
        show = self.show_refusals.isChecked()
        body = "".join(sentence_html(s, show) for s in self.model.sentences)
        body += partial_html(self.model.partial, self.model.partial_lang)
        at_bottom = self._scrolled_to_bottom()
        self.transcript.setHtml(body or self._empty_html())
        if at_bottom:
            self.transcript.moveCursor(QTextCursor.End)
        self._update_counts()

    def _scrolled_to_bottom(self) -> bool:
        """Only follow the meeting when the reader has not scrolled back."""
        bar = self.transcript.verticalScrollBar()
        return bar.value() >= bar.maximum() - 4

    def _update_counts(self) -> None:
        parts = [f"{len(self.model.sentences)} câu",
                 f"{self.model.translated} đã dịch"]
        if self.model.waiting:
            parts.append(f"{self.model.waiting} đang chờ")
        if self.model.dropped:
            parts.append(f"{self.model.dropped} bị lọc")
        speakers = self.model.speakers()
        if speakers:
            parts.append(f"{len(speakers)} người nói")
        self.counts.setText(" · ".join(parts))

    @staticmethod
    def _empty_html() -> str:
        return ('<div style="color:#9aa3ad">Bấm <b>Bắt đầu</b> rồi phát âm '
                'thanh cuộc họp. Chữ mờ là dự đoán đang chạy; chữ đậm là câu '
                'đã chốt, kèm bản dịch bên dưới.</div>')
