"""Unit tests for the sentence-against-running-text comparison.

Pure text, no audio and no model, so these run on the Dev PC. Whether the
four counters actually separate a bad sentence from a good one on real
meeting audio is a question for
``server/tests_real/test_real_partial_final.py`` on the pod.

The two long cases are transcribed from a real run: the running text read
"Solution" and "fix", and the committed sentence replaced both.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.analysis.drift import (
    Drift,
    compare,
    immediate_repeats,
    latin_words,
    substring_distance,
)


class TestSubstringDistance:
    def test_a_verbatim_tail_costs_nothing(self):
        distance, start, end = substring_distance("world", "hello world")
        assert distance == 0
        assert (start, end) == (6, 11)

    def test_text_before_the_match_is_not_charged(self):
        # The running text is the last four seconds of the utterance, so the
        # sentence always has more in front of it. That head is not a
        # difference.
        distance, _start, _end = substring_distance("b c", "a b c")
        assert distance == 0

    def test_a_substitution_costs_one(self):
        distance, _start, _end = substring_distance("cat", "the bat sat")
        assert distance == 1

    def test_an_empty_needle_matches_anywhere(self):
        assert substring_distance("", "anything") == (0, 0, 0)

    def test_an_empty_haystack_costs_the_whole_needle(self):
        distance, _start, _end = substring_distance("abc", "")
        assert distance == 3


class TestLatinWords:
    def test_it_finds_an_english_term_in_a_vietnamese_line(self):
        assert latin_words("Về Solution và cái lý do") == ("solution",)

    def test_diacritics_are_not_latin_words(self):
        # Vietnamese is Latin script. Only runs of unaccented ASCII count, or
        # every word in the language would be a "code-switch".
        assert latin_words("thì bác cũng đã nắm") == ()

    def test_short_runs_are_ignored(self):
        # Vietnamese has unaccented words of one and two letters.
        assert latin_words("mục a và b do") == ()

    def test_it_keeps_case_folded_forms(self):
        assert latin_words("nhờ mình FIX à") == ("fix",)


class TestImmediateRepeats:
    def test_a_doubled_vietnamese_word_counts_once(self):
        assert immediate_repeats("và cái cái lý do") == 1

    def test_clean_text_counts_nothing(self):
        assert immediate_repeats("và cái lý do") == 0

    def test_punctuation_does_not_hide_a_repeat(self):
        assert immediate_repeats("vâng, vâng ạ") == 1

    def test_japanese_is_counted_by_bigram(self):
        # No spaces to tokenise on, and single characters repeat legitimately
        # far too often to count.
        assert immediate_repeats("そうですね") == 0
        assert immediate_repeats("はいはい") == 1
        assert immediate_repeats("そうですですね") == 1


class TestCompare:
    def test_a_sentence_that_keeps_the_running_text_flags_nothing(self):
        drift = compare(
            final="Nguyên nhân là biết rồi, bác cũng nắm rồi.",
            partial="là biết rồi, bác cũng nắm rồi",
            extra_audio_ms=400.0,
            final_audio_ms=6000.0,
        )
        assert drift.rewrite == 0.0
        assert drift.flags == ()

    def test_a_clause_added_over_the_hangover_is_an_invention(self):
        # Evidence 2: the running text ended at 39.3 s, the sentence at 39.7 s,
        # and the sentence carries a clause that no running text ever held.
        drift = compare(
            final="thì bác vẫn nhờ mình thích à? Không, biết thôi đẹp.",
            partial="thì bạn vẫn nhờ mình fix à?",
            extra_audio_ms=400.0,
            final_audio_ms=6700.0,
        )
        assert "tail_invention" in drift.flags
        assert "biết thôi đẹp" in drift.tail_text

    def test_a_tail_with_audio_behind_it_is_not_an_invention(self):
        # Same added words, but two seconds of speech arrived after the last
        # running text. That is lag, not invention.
        drift = compare(
            final="thì bác vẫn nhờ mình thích à? Không, biết thôi đẹp.",
            partial="thì bạn vẫn nhờ mình fix à?",
            extra_audio_ms=2000.0,
            final_audio_ms=6700.0,
        )
        assert "tail_invention" not in drift.flags

    def test_a_lost_english_term_is_counted(self):
        # Evidence 1: "Solution" in the running text, "sau lưu sinh" in the
        # sentence.
        drift = compare(
            final="Ừ thì về sau lưu sinh và cái cái lý do thì cũng đã nắm à",
            partial="về Solution và cái lý do",
            extra_audio_ms=300.0,
            final_audio_ms=6700.0,
        )
        assert drift.lost_latin == ("solution",)
        assert "latin_lost" in drift.flags

    def test_a_word_the_sentence_doubled_is_counted(self):
        drift = compare(
            final="Ừ thì về sau lưu sinh và cái cái lý do thì cũng đã nắm à",
            partial="về Solution và cái lý do",
            extra_audio_ms=300.0,
            final_audio_ms=6700.0,
        )
        assert drift.new_repeats == 1
        assert "repetition" in drift.flags

    def test_a_heavy_rewrite_is_flagged(self):
        drift = compare(
            final="Ừ thì về sau lưu sinh và cái cái lý do thì cũng đã nắm à",
            partial="về Solution và cái lý do",
            extra_audio_ms=300.0,
            final_audio_ms=6700.0,
        )
        assert drift.rewrite > 0.25
        assert "rewrite" in drift.flags

    def test_a_sentence_the_guards_emptied_is_flagged(self):
        drift = compare(final="", partial="nguyên nhân là biết rồi",
                        extra_audio_ms=300.0, final_audio_ms=4000.0)
        assert "final_empty" in drift.flags

    def test_no_running_text_flags_nothing(self):
        # Short utterances never produce one, and they are not evidence of
        # anything.
        drift = compare(final="Vâng.", partial="", extra_audio_ms=0.0,
                        final_audio_ms=500.0)
        assert drift.rewrite == 0.0
        assert drift.flags == ()

    def test_negative_extra_audio_is_clamped(self):
        drift = compare(final="a b c", partial="a b c", extra_audio_ms=-50.0,
                        final_audio_ms=1000.0)
        assert drift.extra_audio_ms == 0.0

    def test_it_returns_a_drift(self):
        assert isinstance(
            compare("x", "x", 0.0, 1000.0), Drift)
