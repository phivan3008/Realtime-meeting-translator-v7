"""Unit tests for the editable word lists.

The lists live in ``server/data`` so they can be changed without touching
code. These tests cover the loader and the matching rules; what belongs in
the files is a question for a person with the recording.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server import wordlists
from server.wordlists import (
    Hallucinations,
    normalise_exact,
    normalise_spaced,
    read_lines,
    read_patterns,
)

SIGN_OFF = "Cảm ơn các bạn đã theo dõi và hẹn gặp lại."
PITCH = ("Hãy subscribe cho kênh La La School "
         "Để không bỏ lỡ những video hấp dẫn")


# ---------------------------------------------------------------------------
# Normalising
# ---------------------------------------------------------------------------
def test_punctuation_and_case_do_not_matter():
    assert normalise_exact("Chào tạm biệt.") == normalise_exact("chào tạm biệt")


def test_diacritics_do_matter():
    """Folding them would make "tắt" and "tab" the same word, and this
    meeting turns on that difference."""
    assert normalise_exact("tắt") != normalise_exact("tab")
    assert normalise_exact("đó") != normalise_exact("do")


def test_the_exact_form_drops_spacing_and_the_spaced_form_keeps_it():
    assert " " not in normalise_exact("hai từ")
    assert normalise_spaced("Hãy  subscribe   cho kênh!") == \
        "hãy subscribe cho kênh"


def test_japanese_punctuation_is_stripped_too():
    assert normalise_exact("ご視聴ありがとう。") == normalise_exact("ご視聴ありがとう")


# ---------------------------------------------------------------------------
# Reading the files
# ---------------------------------------------------------------------------
def test_comments_and_blank_lines_are_skipped(tmp_path, monkeypatch):
    (tmp_path / "x.txt").write_text(
        "# a comment\n\nfirst\n\n  second  \n# another\n", encoding="utf-8")
    monkeypatch.setattr(wordlists, "DATA_DIR", tmp_path)
    assert read_lines("x.txt") == ["first", "second"]


def test_a_missing_file_is_empty_and_says_so(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(wordlists, "DATA_DIR", tmp_path)
    with caplog.at_level(logging.WARNING, logger="server.wordlists"):
        assert read_lines("nothing.txt") == []
    assert "No word list" in caplog.text


def test_a_broken_pattern_is_skipped_rather_than_fatal(tmp_path, monkeypatch,
                                                       caplog):
    """A typo in a data file must not stop the server from starting."""
    (tmp_path / "p.txt").write_text("good.*\n[unclosed\n", encoding="utf-8")
    monkeypatch.setattr(wordlists, "DATA_DIR", tmp_path)
    with caplog.at_level(logging.ERROR, logger="server.wordlists"):
        patterns = read_patterns("p.txt")
    assert len(patterns) == 1
    assert "Ignoring bad pattern" in caplog.text


# ---------------------------------------------------------------------------
# The question the three lists answer together
# ---------------------------------------------------------------------------
def lists(exact=(), patterns=(), keep=()) -> Hallucinations:
    import re
    return Hallucinations(exact=list(exact),
                          patterns=tuple(re.compile(p, re.IGNORECASE)
                                         for p in patterns),
                          keep=list(keep))


def test_an_exact_entry_matches_the_whole_line_only():
    lst = lists(exact=["Chào tạm biệt."])
    assert lst.is_invented("chào tạm biệt")
    assert not lst.is_invented("Chào tạm biệt nhé.")


def test_a_pattern_fills_the_hole():
    lst = lists(patterns=[r"h[ãa]y subscribe cho k[êe]nh .{1,40} [đd][ểe] kh[ôo]ng"])
    assert lst.is_invented("Hãy subscribe cho kênh La La School Để không")


def test_a_pattern_can_cover_the_undotted_spelling():
    """Whisper drops Vietnamese diacritics when it is unsure, and dropping
    them takes the bar off the đ as well."""
    lst = lists(patterns=[r"h[ãa]y subscribe cho k[êe]nh .{1,40} [đd][ểe] kh[ôo]ng"])
    assert lst.is_invented("hay subscribe cho kenh X de khong")


def test_a_pattern_is_anchored_to_the_whole_line():
    """Or a real sentence quoting the pitch would be deleted with it."""
    lst = lists(patterns=[r"h[ãa]y subscribe cho k[êe]nh .{1,40}"])
    assert not lst.is_invented("Khách bảo hãy subscribe cho kênh đó, mình từ chối")


def test_keep_beats_the_exact_list():
    lst = lists(exact=[SIGN_OFF], keep=[SIGN_OFF])
    assert not lst.is_invented(SIGN_OFF)


def test_keep_beats_a_pattern():
    lst = lists(patterns=[r"h[ãa]y subscribe cho k[êe]nh .*"], keep=[PITCH])
    assert not lst.is_invented(PITCH)


def test_keep_ignores_punctuation_like_everything_else():
    lst = lists(exact=["Chào tạm biệt."], keep=["chào tạm biệt"])
    assert not lst.is_invented("Chào tạm biệt.")


def test_empty_lists_block_nothing():
    assert not lists().is_invented(SIGN_OFF)


# ---------------------------------------------------------------------------
# The files that ship
# ---------------------------------------------------------------------------
def test_the_shipped_lists_are_not_empty():
    """Empty files would make every check that uses them pass for nothing."""
    shipped = Hallucinations()
    assert shipped.exact
    assert shipped.patterns


@pytest.mark.parametrize("line", [
    "Cảm ơn các bạn đã theo dõi và hẹn gặp lại.",
    "Hẹn gặp lại các bạn trong những video tiếp theo.",
    "Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn",
    "Hãy subscribe cho kênh La La School Để không bỏ lỡ những video hấp dẫn",
    "Các bạn có thể nhớ like và share video này để ủng hộ kênh của mình",
    # Undotted, as Whisper writes it when it is unsure.
    "hay subscribe cho kenh Abc De khong bo lo nhung video hap dan",
    "ご視聴ありがとうございました",
    "you",
    "Chào tạm biệt.",
])
def test_every_confirmed_invention_is_blocked(line):
    """Each of these was read off a real run and confirmed unspoken."""
    assert Hallucinations().is_invented(line), line


@pytest.mark.parametrize("said", [
    "Cảm ơn các bạn đã theo dõi báo cáo này.",
    "Chào tạm biệt nhé.",
    "Thôi chào tạm biệt mọi người.",
    "Tạm biệt.",
    "Hãy đăng ký kênh Teams cho dự án này",
    "Hãy subscribe cho kênh nội bộ của team mình",
    "Các bạn có thể nhớ gửi tài liệu cho tôi",
    "Are you there?",
])
def test_a_real_meeting_sentence_survives_the_shipped_lists(said):
    """The cost of a wrong entry is a sentence nobody ever hears, so the
    near misses are worth more tests than the hits."""
    assert not Hallucinations().is_invented(said), said
