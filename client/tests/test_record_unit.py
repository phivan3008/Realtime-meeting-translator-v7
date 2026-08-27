"""Unit tests for writing the meeting to disk.

Real files, in a temporary directory, with a clock that does not tick unless
told to. No Qt, no socket, no server.

Run with::

    .venv\\Scripts\\python.exe -m pytest client/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.record import Recorder, paths_for


class Clock:
    """Wall-clock seconds, moved by hand."""

    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def recorder(tmp_path):
    clock = Clock()
    made = Recorder(tmp_path / "m.txt", tmp_path / "m.debug.txt",
                    session_id="abc123", clock=clock)
    made.clock_control = clock                  # for the tests to move time
    yield made
    made.close()


def final(sentence_id=1, text="Xin chào mọi người.", speaker="Speaker_01",
          lang="vi") -> dict:
    return {"type": "final", "sentence_id": sentence_id, "speaker_id": speaker,
            "lang_code": lang, "transcript": text, "speech_score": 0.85}


def translation(sentence_id=1, text="こんにちは皆様。", reason="") -> dict:
    return {"type": "translation", "sentence_id": sentence_id,
            "translation": text, "reason": reason, "raw": ""}


def minutes(recorder) -> str:
    return recorder.minutes_path.read_text(encoding="utf-8")


def debug(recorder) -> str:
    return recorder.debug_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The minutes
# ---------------------------------------------------------------------------
def test_a_sentence_and_its_translation_end_up_together(recorder):
    recorder.apply(final())
    recorder.apply(translation())
    text = minutes(recorder)
    assert "Xin chào mọi người." in text
    assert "こんにちは皆様。" in text
    assert text.index("Xin chào") < text.index("こんにちは")


def test_the_minutes_carry_who_said_it_and_when(recorder):
    recorder.apply(final(speaker="Speaker_02", lang="ja", text="はい。"))
    line = [row for row in minutes(recorder).splitlines()
            if "Speaker_02" in row][0]
    assert "日本語" in line
    assert line.startswith("[") and ":" in line


def test_the_minutes_do_not_leave_a_blank_where_a_translation_belongs(recorder):
    """A translation that was refused is a fact about the meeting."""
    recorder.apply(final(sentence_id=2))
    recorder.apply(translation(sentence_id=2, text="", reason="quá dài"))
    assert "không dịch được — quá dài" in minutes(recorder)


def test_a_sentence_still_waiting_says_so(recorder):
    recorder.apply(final())
    assert "đang dịch" in minutes(recorder)


def test_a_late_translation_reaches_a_sentence_written_earlier(recorder):
    """Translations come back out of order, and after later sentences."""
    recorder.apply(final(sentence_id=1, text="Câu một."))
    recorder.apply(final(sentence_id=2, text="Câu hai."))
    recorder.apply(translation(sentence_id=2, text="文二。"))
    recorder.apply(translation(sentence_id=1, text="文一。"))
    text = minutes(recorder)
    assert text.index("Câu một.") < text.index("Câu hai.")
    assert text.index("文一。") < text.index("Câu hai.")


def test_the_meeting_is_written_in_the_order_it_happened(recorder):
    for index in range(1, 6):
        recorder.apply(final(sentence_id=index, text=f"Câu {index}."))
    positions = [minutes(recorder).index(f"Câu {index}.")
                 for index in range(1, 6)]
    assert positions == sorted(positions)


def test_a_corrected_speaker_label_reaches_the_file(recorder):
    """The server clusters the meeting again and revises its own answers."""
    recorder.apply(final(sentence_id=3, speaker="Speaker_01"))
    recorder.apply({"type": "speakers", "labels": {"3": "Speaker_04"}})
    assert "Speaker_04" in minutes(recorder)
    assert "Speaker_01" not in minutes(recorder)


def test_the_running_text_never_reaches_the_minutes(recorder):
    """It is replaced within a second; nobody reads a meeting that way."""
    recorder.apply({"type": "partial", "transcript": "xin chào mọi ng",
                    "lang_code": "vi"})
    assert "xin chào mọi ng" not in minutes(recorder)


def test_the_minutes_exist_before_anybody_speaks(recorder):
    assert recorder.minutes_path.exists()
    assert "Biên bản cuộc họp" in minutes(recorder)


# ---------------------------------------------------------------------------
# The debug log
# ---------------------------------------------------------------------------
def test_the_debug_log_keeps_the_running_text(recorder):
    recorder.apply({"type": "partial", "transcript": "xin chào mọi ng",
                    "lang_code": "vi"})
    assert "xin chào mọi ng" in debug(recorder)


def test_the_debug_log_is_in_arrival_order(recorder):
    recorder.apply(final(sentence_id=1))
    recorder.apply(final(sentence_id=2, text="Câu hai."))
    recorder.apply(translation(sentence_id=1))
    kinds = [row.split()[2] for row in debug(recorder).splitlines()[1:]]
    assert kinds == ["final", "final", "translation"]


def test_every_line_carries_the_time_and_the_seconds_in(recorder):
    recorder.clock_control.tick(12.5)
    recorder.apply(final())
    row = [line for line in debug(recorder).splitlines() if "final" in line][0]
    assert "12.5s" in row
    assert row.count(":") >= 2, "no wall clock on the line"


def test_a_dropped_utterance_is_recorded_with_what_it_sounded_like(recorder):
    """So a noise filter that starts eating speech can be found afterwards."""
    recorder.apply({"type": "utterance", "index": 7, "kept": False,
                    "label": "Computer keyboard", "reason": "pause"})
    assert "Computer keyboard" in debug(recorder)


def test_a_kept_utterance_is_not_noise_in_the_log(recorder):
    recorder.apply({"type": "utterance", "index": 7, "kept": True, "label": ""})
    assert "utterance 7" not in debug(recorder)


def test_a_refusal_says_why_in_the_debug_log(recorder):
    recorder.apply(final(sentence_id=2))
    recorder.apply(translation(sentence_id=2, text="", reason="quá dài"))
    assert "từ chối: quá dài" in debug(recorder)


def test_a_server_error_is_recorded(recorder):
    recorder.apply({"type": "error", "message": "unsupported audio format",
                    "fatal": True})
    assert "unsupported audio format" in debug(recorder)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
def test_a_translation_for_a_sentence_never_seen_is_not_fatal(recorder):
    recorder.apply(translation(sentence_id=99))
    assert "#99" in debug(recorder)


def test_an_unknown_message_is_ignored(recorder):
    recorder.apply({"type": "something-new", "value": 1})
    recorder.apply({"type": "vad", "event": "speech_start", "at_ms": 1.0})


def test_both_files_survive_the_process_being_killed(tmp_path):
    """Everything is flushed as it is written, so there is no close to miss."""
    made = Recorder(tmp_path / "m.txt", tmp_path / "m.debug.txt", clock=Clock())
    made.apply(final())
    made.apply(translation())
    assert "こんにちは皆様。" in (tmp_path / "m.txt").read_text(encoding="utf-8")
    assert "final" in (tmp_path / "m.debug.txt").read_text(encoding="utf-8")


def test_closing_twice_is_harmless(tmp_path):
    made = Recorder(tmp_path / "m.txt", tmp_path / "m.debug.txt", clock=Clock())
    made.close()
    made.close()


def test_nothing_is_recorded_after_the_meeting_ends(tmp_path):
    made = Recorder(tmp_path / "m.txt", tmp_path / "m.debug.txt", clock=Clock())
    made.close()
    made.apply(final(text="Câu sau khi đóng."))
    assert "Câu sau khi đóng." not in (tmp_path / "m.txt").read_text(
        encoding="utf-8")


def test_the_summary_is_the_last_line_of_the_debug_log(tmp_path):
    made = Recorder(tmp_path / "m.txt", tmp_path / "m.debug.txt", clock=Clock())
    made.apply(final())
    made.apply(translation())
    made.close()
    last = (tmp_path / "m.debug.txt").read_text(encoding="utf-8").splitlines()[-1]
    assert "1 câu" in last and "1 đã dịch" in last


def test_the_directory_is_made_if_it_is_not_there(tmp_path):
    nested = tmp_path / "recordings" / "today"
    Recorder(nested / "m.txt", nested / "m.debug.txt", clock=Clock()).close()
    assert (nested / "m.txt").exists()


def test_vietnamese_and_japanese_survive_the_round_trip(tmp_path):
    """Windows writes cp1252 by default, which cannot hold either language."""
    made = Recorder(tmp_path / "m.txt", tmp_path / "m.debug.txt", clock=Clock())
    made.apply(final(text="Đường đến ngày vinh quang."))
    made.apply(translation(text="栄光への道。"))
    made.close()
    text = (tmp_path / "m.txt").read_text(encoding="utf-8")
    assert "Đường đến ngày vinh quang." in text
    assert "栄光への道。" in text


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def test_the_two_files_are_named_after_the_meeting(tmp_path):
    minutes_path, debug_path = paths_for(tmp_path, when=1_800_000_000.0)
    assert minutes_path.parent == tmp_path
    assert minutes_path.name.startswith("meeting-")
    assert debug_path.name == minutes_path.name.replace(".txt", ".debug.txt")


def test_two_meetings_do_not_share_a_file(tmp_path):
    first, _ = paths_for(tmp_path, when=1_800_000_000.0)
    second, _ = paths_for(tmp_path, when=1_800_000_060.0)
    assert first != second
