"""Unit tests for agreement across running texts.

The running texts below are copied from real meetings, including the two
shapes that make this hard: a window that has slid past the head of the
sentence, and Japanese, which has no spaces to join on.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.pipeline.stabilize import (  # noqa: E402
    Stabilizer,
    agreed_prefix,
    join,
    overlap,
    render,
    stitch,
    tokenize,
)


class TestTokenize:
    def test_vietnamese_words_stay_whole(self):
        assert [token.normalized for token in tokenize("Về Solution và cái")] == [
            "về", "solution", "và", "cái"]

    def test_japanese_is_one_token_per_character(self):
        """Japanese has no spaces, so a whole run would make agreement
        all-or-nothing for an entire clause."""
        assert [token.normalized for token in tokenize("画面共有")] == [
            "画", "面", "共", "有"]

    def test_punctuation_is_not_a_token_but_is_not_lost(self):
        tokens = tokenize("Cảm ơn, bác.")
        assert [token.normalized for token in tokens] == ["cảm", "ơn", "bác"]
        # It survives in the source, which is what gets rendered.
        assert render(tokens) == "Cảm ơn, bác."

    def test_a_token_remembers_where_it_came_from(self):
        text = "về Solution"
        token = tokenize(text)[1]
        assert text[token.start:token.end] == "Solution"


class TestJoin:
    def test_two_vietnamese_fragments_take_a_space(self):
        assert join("về Solution", "và cái lý do") == "về Solution và cái lý do"

    def test_two_japanese_fragments_take_none(self):
        """The fault this module exists to avoid. Joining word timestamps
        with a space turns アプリケーション into ア プ リ ケ ー ショ ン."""
        assert join("画面共有", "しましたが") == "画面共有しましたが"

    def test_a_mixed_boundary_takes_none(self):
        assert join("この件ですけど", "Tranium") == "この件ですけどTranium"

    def test_an_empty_side_is_not_padded(self):
        assert join("", "はい") == "はい"
        assert join("はい", "") == "はい"


class TestRender:
    def test_one_source_is_rendered_as_a_slice_of_itself(self):
        """Never rebuilt from tokens: the spacing and punctuation are
        whatever Whisper wrote."""
        text = "Về Solution và cái lý do."
        assert render(tokenize(text)) == text

    def test_japanese_from_one_source_keeps_its_spacing(self):
        text = "画面共有しましたが"
        assert render(tokenize(text)) == text

    def test_two_sources_are_spliced_by_script(self):
        left = tokenize("Về Solution")
        right = tokenize("và cái lý do")
        assert render(left + right) == "Về Solution và cái lý do"
        assert render(tokenize("画面") + tokenize("共有")) == "画面共有"

    def test_nothing_renders_as_nothing(self):
        assert render([]) == ""


class TestOverlap:
    def test_the_longest_overlap_is_found(self):
        earlier = tokenize("về Solution và cái lý do")
        later = tokenize("và cái lý do thì bác cũng đã nắm")
        assert overlap(earlier, later, 2) == 4

    def test_one_shared_word_is_not_an_overlap(self):
        """A single word matches by accident constantly, and a wrong splice
        puts words in an order nobody said."""
        earlier = tokenize("hôm nay mình họp về")
        later = tokenize("về sau thì khác hẳn")
        assert overlap(earlier, later, 2) == 0

    def test_no_shared_tail_is_no_overlap(self):
        assert overlap(tokenize("một hai"), tokenize("ba bốn"), 2) == 0


class TestStitch:
    def test_a_slid_window_is_put_back_together(self):
        """The running text loses the head of the sentence once the window
        moves past it. Prefix agreement is meaningless until this is undone."""
        first = tokenize("thì về Solution và cái lý do")
        second = tokenize("và cái lý do thì bác cũng đã nắm")
        stitched = stitch(first, second, 2)
        assert render(stitched) == "thì về Solution và cái lý do thì bác cũng đã nắm"

    def test_an_empty_side_is_returned_whole(self):
        tokens = tokenize("một hai")
        assert stitch([], tokens, 2) == tokens
        assert stitch(tokens, [], 2) == tokens

    def test_without_an_overlap_the_longer_view_wins(self):
        """Concatenating is what makes a reference say the same clause twice.

        Measured on one meeting: joining views that did not overlap put
        "ステップ011の方は" into a single reference three times over. A short
        reference is fine for both of its uses; a repeating one is not.
        """
        stitched = stitch(tokenize("một hai"), tokenize("ba bốn năm"), 2)
        assert render(stitched) == "ba bốn năm"

    def test_a_reworded_overlap_is_still_an_overlap(self):
        """Two decodes of the same audio agree on most of it and reword the
        rest. Demanding an exact match finds no overlap on about a third of
        real pairs."""
        first = tokenize("thì về Solution và cái lý do")
        second = tokenize("và cái lí do thì bác cũng đã nắm")
        assert render(stitch(first, second, 2)) == (
            "thì về Solution và cái lý do thì bác cũng đã nắm")


class TestAgreedPrefix:
    def test_words_two_views_share_are_settled(self):
        views = [tokenize("về Solution và cái"), tokenize("về Solution và lý")]
        assert agreed_prefix(views, 2) == 3

    def test_one_view_settles_nothing(self):
        assert agreed_prefix([tokenize("về Solution")], 2) == 0

    def test_a_disagreement_at_the_first_word_settles_nothing(self):
        views = [tokenize("về Solution"), tokenize("bờ Solution")]
        assert agreed_prefix(views, 2) == 0

    def test_only_the_most_recent_views_count(self):
        views = [tokenize("hoàn toàn khác"), tokenize("về Solution và"),
                 tokenize("về Solution và cái")]
        assert agreed_prefix(views, 2) == 3


class TestStabilizer:
    def test_the_reference_covers_the_whole_sentence_not_the_window(self):
        """The point of the module. The newest running text holds four
        seconds; a seven-second sentence is compared against all of it."""
        stabilizer = Stabilizer()
        stabilizer.observe(0, "thì về Solution và cái lý do", "vi")
        stable = stabilizer.observe(0, "và cái lý do thì bác cũng đã nắm", "vi")
        assert stable.whole == (
            "thì về Solution và cái lý do thì bác cũng đã nắm")

    def test_a_word_two_views_agree_on_is_settled(self):
        stabilizer = Stabilizer()
        stabilizer.observe(0, "Về Solution và cái", "vi")
        stable = stabilizer.observe(0, "Về Solution và cái lý do", "vi")
        assert stable.text.startswith("Về Solution và cái")
        assert "lý do" in stable.tail or "lý do" in stable.text

    def test_the_first_view_settles_nothing(self):
        stable = Stabilizer().observe(0, "Về Solution", "vi")
        assert stable.text == ""
        assert stable.tail == "Về Solution"
        assert stable.whole == "Về Solution"

    def test_japanese_never_grows_a_space(self):
        stabilizer = Stabilizer()
        stabilizer.observe(0, "画面共有しましたが", "ja")
        stable = stabilizer.observe(0, "しましたがはい見ます", "ja")
        assert " " not in stable.whole
        assert stable.whole == "画面共有しましたがはい見ます"

    def test_the_running_texts_vote_on_the_language(self):
        stabilizer = Stabilizer()
        stabilizer.observe(0, "はい", "ja")
        stabilizer.observe(0, "はいそうです", "ja")
        assert stabilizer.observe(0, "Vâng", "vi").language == "ja"

    def test_an_empty_running_text_is_not_a_view(self):
        stabilizer = Stabilizer()
        stabilizer.observe(0, "   ", "vi")
        assert stabilizer.stable(0).views == 0

    def test_utterances_do_not_leak_into_each_other(self):
        stabilizer = Stabilizer()
        stabilizer.observe(0, "câu thứ nhất", "vi")
        stabilizer.observe(1, "câu thứ hai", "vi")
        assert stabilizer.stable(0).whole == "câu thứ nhất"
        assert stabilizer.stable(1).whole == "câu thứ hai"

    def test_releasing_an_utterance_forgets_it(self):
        stabilizer = Stabilizer()
        stabilizer.observe(0, "câu thứ nhất", "vi")
        assert stabilizer.release(0).whole == "câu thứ nhất"
        assert stabilizer.stable(0).whole == ""

    def test_an_unknown_utterance_is_empty_rather_than_an_error(self):
        assert not Stabilizer().stable(7).has_text
        assert not Stabilizer().release(7).has_text

    def test_a_new_meeting_starts_with_nothing(self):
        stabilizer = Stabilizer()
        stabilizer.observe(0, "câu của phiên trước", "vi")
        stabilizer.reset()
        assert stabilizer.stable(0).whole == ""

    def test_only_the_recent_views_are_kept(self):
        stabilizer = Stabilizer(history=3)
        for number in range(6):
            stabilizer.observe(0, f"câu số {number}", "vi")
        assert stabilizer.stable(0).views == 3

    def test_agreement_of_one_is_refused(self):
        with pytest.raises(ValueError, match="at least 2"):
            Stabilizer(min_agreement=1)

    def test_an_overlap_of_zero_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            Stabilizer(min_overlap=0)
