"""Unit tests for the meeting's own vocabulary as a Whisper prompt.

Two of these guard mistakes that have already cost a run of this project:
seeding the list with plausible guesses, and letting an empty list become an
empty prompt rather than no prompt at all.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import ASR_PROMPT_MAX_CHARS, VOCABULARY_PATH  # noqa: E402
from server.pipeline.vocabulary import (  # noqa: E402
    build_prompt,
    load_prompt,
    read_terms,
)


def write(tmp_path, text: str) -> Path:
    path = tmp_path / "vocabulary.txt"
    path.write_text(text, encoding="utf-8")
    return path


class TestReading:
    def test_terms_are_read_in_order(self, tmp_path):
        path = write(tmp_path, "solution\ntemplate\nSlack\n")
        assert read_terms(path) == ["solution", "template", "Slack"]

    def test_comments_and_blank_lines_are_skipped(self, tmp_path):
        path = write(tmp_path, "# what it was heard as\n\n  solution  \n\n")
        assert read_terms(path) == ["solution"]

    def test_a_comment_after_a_term_does_not_travel_with_it(self, tmp_path):
        """This is how a term reaches Whisper with its annotation attached,
        and the prompt starts containing text nobody said."""
        path = write(tmp_path, "solution  # heard as sau lưu sinh\n")
        assert read_terms(path) == ["solution"]

    def test_a_hash_inside_a_term_is_not_a_comment(self, tmp_path):
        """A meeting can say "C#". Only a `#` that opens a comment - at the
        start of a line, or after a space - ends one."""
        path = write(tmp_path, "C#\nF#\n")
        assert read_terms(path) == ["C#", "F#"]

    def test_repeats_are_dropped(self, tmp_path):
        path = write(tmp_path, "Slack\nplan\nSlack\n")
        assert read_terms(path) == ["Slack", "plan"]

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        """The meeting still runs. It just runs without the tilt."""
        assert read_terms(tmp_path / "nothing.txt") == []

    def test_multi_word_terms_survive(self, tmp_path):
        path = write(tmp_path, "dung lượng\nkiểm chứng\n")
        assert read_terms(path) == ["dung lượng", "kiểm chứng"]


class TestBuilding:
    def test_terms_become_one_comma_separated_line(self):
        assert build_prompt(["solution", "Slack"]) == "solution, Slack"

    def test_the_cut_lands_on_a_term_boundary(self):
        """Half a term is a string nobody said, and the prompt is meant to
        hold words the meeting holds."""
        prompt = build_prompt(["aaaa", "bbbb", "cccc"], limit=12)
        assert prompt == "aaaa, bbbb"

    def test_a_limit_shorter_than_the_first_term_gives_nothing(self):
        assert build_prompt(["solution"], limit=3) == ""

    def test_nothing_in_gives_nothing_out(self):
        assert build_prompt([]) == ""


class TestLoading:
    def test_an_empty_file_gives_no_prompt_at_all(self, tmp_path):
        """Not an empty one. "" is itself a prompt as far as Whisper is
        concerned, and passing it is not the same as passing nothing."""
        assert load_prompt(write(tmp_path, "\n# only a comment\n")) is None

    def test_a_missing_file_gives_no_prompt(self, tmp_path):
        assert load_prompt(tmp_path / "nothing.txt") is None

    def test_a_list_that_does_not_fit_at_all_gives_no_prompt(self, tmp_path):
        assert load_prompt(write(tmp_path, "solution\n"), limit=3) is None

    def test_a_real_list_becomes_a_prompt(self, tmp_path):
        path = write(tmp_path, "solution\ndung lượng\n")
        assert load_prompt(path) == "solution, dung lượng"

    def test_trimming_is_reported_rather_than_silent(self, tmp_path, caplog):
        path = write(tmp_path, "aaaa\nbbbb\ncccc\n")
        with caplog.at_level("WARNING"):
            load_prompt(path, limit=12)
        assert "never reach Whisper" in caplog.text


class TestTheShippedList:
    """The file as it stands, because both of its rules are breakable by
    editing a text file and nothing else would notice."""

    def test_it_fits_inside_the_cap(self):
        prompt = load_prompt()
        assert prompt is not None
        assert len(prompt) <= ASR_PROMPT_MAX_CHARS

    def test_every_term_reaches_whisper(self):
        """A term past the cap is a term someone added believing it would
        help. Better to notice here than to wonder later why it did not."""
        terms = read_terms()
        assert load_prompt() == build_prompt(terms)

    def test_it_holds_the_words_the_transcripts_showed_going_wrong(self):
        terms = {term.casefold() for term in read_terms()}
        assert {"solution", "dung lượng", "kiểm chứng"} <= terms

    def test_it_holds_the_names_only_the_meeting_can_supply(self):
        terms = {term.casefold() for term in read_terms()}
        assert {"miyake", "yamaguchi", "takahashi"} <= terms

    def test_no_annotation_leaked_into_a_term(self):
        assert all("#" not in term or len(term) <= 3
                   for term in read_terms())

    def test_the_file_is_where_the_config_says(self):
        assert Path(VOCABULARY_PATH).is_file()
