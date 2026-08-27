"""Write the meeting to disk, in two files with two readers in mind.

``<name>.txt``       what was said and what it means, for reading afterwards.
``<name>.debug.txt`` every message as it arrived, for working out why.

They are separate because they answer different questions. The minutes want
one entry per sentence, in the order the meeting happened, with the sentence
and its translation together - which is not the order the messages arrive in,
because a translation comes back after its sentence and sometimes after
several later ones. The debug log wants exactly the opposite: everything, in
arrival order, including the running text that was replaced a second later.

Two things arrive late and change what is already written: a translation, and
a corrected speaker label from the server's reclustering. The minutes file is
therefore rewritten whenever one lands. It is a few hundred kilobytes of text
and the alternative is a record that disagrees with the screen.

Both files are UTF-8 and flushed on every write. A meeting that ends by the
process being killed still leaves everything up to that point.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TextIO

log = logging.getLogger(__name__)

LANGUAGE_NAMES = {"vi": "Tiếng Việt", "ja": "日本語", "": "?"}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


@dataclass
class Line:
    """One committed sentence and everything learned about it since."""

    sentence_id: int
    at: float
    speaker_id: str
    lang_code: str
    text: str
    translation: str = ""
    reason: str = ""
    answered: bool = False


@dataclass
class RecorderStats:
    sentences: int = 0
    translated: int = 0
    refused: int = 0
    dropped: int = 0
    relabelled: int = 0
    partials: int = 0


class Recorder:
    """Keep the meeting's two files up to date as messages arrive."""

    def __init__(
        self,
        minutes_path: Path,
        debug_path: Path,
        session_id: str = "",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.minutes_path = Path(minutes_path)
        self.debug_path = Path(debug_path)
        self.session_id = session_id
        self.clock = clock
        self.started = clock()
        self.lines: list[Line] = []
        self.stats = RecorderStats()
        self._by_id: dict[int, Line] = {}

        self.minutes_path.parent.mkdir(parents=True, exist_ok=True)
        self.debug_path.parent.mkdir(parents=True, exist_ok=True)
        self._debug: Optional[TextIO] = self.debug_path.open(
            "w", encoding="utf-8", newline="\n")
        self.note("start", f"session={session_id or '?'}")
        self._write_minutes()

    # -- incoming -----------------------------------------------------------
    def apply(self, message: dict) -> None:
        """Absorb one server message. Unknown kinds are ignored, not an error."""
        if self._debug is None:
            return
        kind = message.get("type")
        handler = getattr(self, f"_on_{kind}", None)
        if handler is not None:
            handler(message)

    def _on_partial(self, message: dict) -> None:
        text = message.get("transcript", "")
        if not text:
            return
        self.stats.partials += 1
        self.note("partial", f"[{message.get('lang_code', '')}] {text}")

    def _on_final(self, message: dict) -> None:
        line = Line(
            sentence_id=message.get("sentence_id", 0),
            at=self.clock(),
            speaker_id=message.get("speaker_id", ""),
            lang_code=message.get("lang_code", ""),
            text=message.get("transcript", ""),
        )
        self.lines.append(line)
        self._by_id[line.sentence_id] = line
        self.stats.sentences += 1
        self.note("final", f"#{line.sentence_id} {line.speaker_id or '?'} "
                           f"[{line.lang_code}] {line.text}")
        self._write_minutes()

    def _on_translation(self, message: dict) -> None:
        line = self._by_id.get(message.get("sentence_id"))
        translation = message.get("translation", "")
        reason = message.get("reason", "")
        if translation.strip():
            self.stats.translated += 1
        else:
            self.stats.refused += 1
        self.note("translation", f"#{message.get('sentence_id')} "
                  + (translation if translation.strip()
                     else f"(từ chối: {reason or 'không rõ'})"))
        if line is None:
            return
        line.translation = translation
        line.reason = reason
        line.answered = True
        self._write_minutes()

    def _on_speakers(self, message: dict) -> None:
        labels = message.get("labels", {})
        changed = []
        for key, speaker_id in labels.items():
            line = self._by_id.get(int(key))
            if line is not None and line.speaker_id != speaker_id:
                line.speaker_id = speaker_id
                changed.append(f"#{key}->{speaker_id}")
        if not changed:
            return
        self.stats.relabelled += len(changed)
        self.note("speakers", ", ".join(changed))
        self._write_minutes()

    def _on_utterance(self, message: dict) -> None:
        if message.get("kept", True):
            return
        self.stats.dropped += 1
        self.note("dropped", f"utterance {message.get('index')} — "
                             f"{message.get('label') or 'không rõ'}")

    def _on_error(self, message: dict) -> None:
        self.note("error", message.get("message", ""))

    # -- outgoing -----------------------------------------------------------
    def note(self, kind: str, text: str) -> None:
        """One line in the debug file, in arrival order."""
        if self._debug is None:
            return
        now = self.clock()
        self._debug.write(
            f"{time.strftime('%H:%M:%S', time.localtime(now))}"
            f".{int(now % 1 * 1000):03d}  "
            f"{now - self.started:7.1f}s  {kind:<11} {text}\n")
        self._debug.flush()

    def _write_minutes(self) -> None:
        """Rewrite the minutes. Late translations and corrected speaker labels
        both change entries that are already in the file."""
        body = [
            f"# Biên bản cuộc họp — "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.started))}",
            f"# Phiên {self.session_id or '?'}",
            "",
        ]
        for line in self.lines:
            body.append(
                f"[{time.strftime('%H:%M:%S', time.localtime(line.at))}] "
                f"{line.speaker_id or '?'} · {language_name(line.lang_code)}")
            body.append(f"  {line.text}")
            if line.translation.strip():
                body.append(f"  {line.translation}")
            elif line.answered:
                body.append(f"  (không dịch được — {line.reason or 'không rõ'})")
            else:
                body.append("  (đang dịch…)")
            body.append("")
        self.minutes_path.write_text("\n".join(body), encoding="utf-8")

    def close(self) -> None:
        """Finish both files. Safe to call twice."""
        if self._debug is None:
            return
        self.note("end", f"{self.stats.sentences} câu, "
                         f"{self.stats.translated} đã dịch, "
                         f"{self.stats.refused} không dịch được, "
                         f"{self.stats.dropped} bị lọc, "
                         f"{self.stats.relabelled} nhãn được sửa")
        self._debug.close()
        self._debug = None
        self._write_minutes()
        log.info("Biên bản: %s", self.minutes_path)
        log.info("Nhật ký gỡ lỗi: %s", self.debug_path)


def paths_for(directory: Path, when: Optional[float] = None) -> tuple[Path, Path]:
    """The pair of files for a meeting starting now."""
    stamp = time.strftime("%Y%m%d-%H%M%S",
                          time.localtime(when if when is not None else time.time()))
    directory = Path(directory)
    return (directory / f"meeting-{stamp}.txt",
            directory / f"meeting-{stamp}.debug.txt")
