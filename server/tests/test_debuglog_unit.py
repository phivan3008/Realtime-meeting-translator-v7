"""Unit tests for reading the client's debug log.

The lines below are copied from real logs, including the two faults the
module exists to count: Japanese rendered with a space between every
character, and a word committed twice at a hypothesis boundary.

Pure text, no audio and no model, so these run on the Dev PC.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.analysis import debuglog  # noqa: E402

LOG = """\
00:37:57.390      0.0s  start       session=?
00:37:59.027      1.6s  status      Đã kết nối · phiên 42206da278b7
00:38:24.622     27.2s  partial     [ja] 画面共有しましたが
00:38:28.599     31.2s  partial     [ja] はい
00:38:29.848     32.5s  final       #1 Speaker_01 [ja] 画面共有しましたが
00:38:29.944     32.6s  translation #1 Tôi đã chia sẻ màn hình
00:38:31.082     33.7s  partial     [vi] Tạp cấp, tạp cấp quay.
00:38:35.223     37.8s  partial     [vi] Về mục tiêu tác 02 thì vẫn đang bending
00:38:35.486     38.1s  final       #2 Speaker_01 [vi] Về mục tiêu tác 02 thì vẫn đang bending
00:38:35.825     38.4s  translation #2 (từ chối: the model returned the sentence untranslated)
00:44:54.353    417.0s  dropped     utterance 76 — Knock
08:22:15.656  27858.3s  end         279 câu, 261 đã dịch, 18 không dịch được
"""


@pytest.fixture
def run(tmp_path) -> debuglog.Run:
    path = tmp_path / "meeting.debug.txt"
    path.write_text(LOG, encoding="utf-8")
    return debuglog.parse(path)


class TestParse:
    def test_sentences_are_read_with_their_speaker_and_language(self, run):
        assert [final.sentence for final in run.finals] == [1, 2]
        assert run.finals[0].lang == "ja"
        assert run.finals[0].speaker == "Speaker_01"
        assert run.finals[0].text == "画面共有しましたが"

    def test_a_sentence_carries_the_running_texts_that_preceded_it(self, run):
        assert [text for _lang, text, _at in run.finals[0].partials] == [
            "画面共有しましたが", "はい"]
        assert run.finals[0].last_partial == "はい"

    def test_the_running_texts_are_not_shared_between_sentences(self, run):
        """Each sentence gets the ones shown since the last one was committed."""
        assert len(run.finals[1].partials) == 2
        assert run.finals[1].last_partial.endswith("bending")

    def test_a_translation_is_matched_to_its_sentence(self, run):
        assert run.finals[0].translation == "Tôi đã chia sẻ màn hình"
        assert not run.finals[0].refused

    def test_a_refused_translation_is_recognised(self, run):
        assert run.finals[1].refused

    def test_the_closing_summary_is_kept(self, run):
        assert run.summary.startswith("279 câu")

    def test_lines_it_does_not_understand_are_skipped(self, tmp_path):
        path = tmp_path / "noise.debug.txt"
        path.write_text("not a log line at all\n\n" + LOG, encoding="utf-8")
        assert len(debuglog.parse(path).finals) == 2

    def test_partials_are_counted_even_when_no_sentence_follows(self, run):
        assert run.partial_count == 4


class TestLanguageDisagreement:
    """The cheap signal for an utterance holding two languages.

    Measured over three real runs: when the running texts were in one
    language and the sentence came out in the other, the sentence was three
    to four times as likely to be unrelated to what was said.
    """

    def make(self, partial_langs, final_lang):
        return debuglog.Final(
            sentence=1, at=10.0, speaker="Speaker_01", lang=final_lang,
            text="whatever",
            partials=[(lang, "something", 1.0) for lang in partial_langs])

    def test_the_running_texts_vote(self):
        assert self.make(["ja", "ja", "vi"], "ja").partial_language == "ja"

    def test_a_disagreement_is_flagged(self):
        assert self.make(["ja", "ja"], "vi").language_disagrees

    def test_agreement_is_not(self):
        assert not self.make(["ja", "ja"], "ja").language_disagrees

    def test_no_running_text_is_not_a_disagreement(self):
        # Nothing to disagree with. Silence is not evidence.
        assert not self.make([], "vi").language_disagrees

    def test_an_undecided_sentence_is_not_a_disagreement(self):
        assert not self.make(["ja"], "").language_disagrees


class TestRendering:
    def test_a_space_between_japanese_characters_is_counted(self):
        """What joining word timestamps with " " does to a language that has
        no spaces. It is unreadable on screen and worse in a prompt."""
        assert debuglog.cjk_spacing("この 多 数 ク は 今 朝") == 6
        assert debuglog.cjk_spacing("この多数クは今朝") == 0

    def test_vietnamese_spacing_is_not_counted(self):
        assert debuglog.cjk_spacing("Về mục tiêu tác 02") == 0

    def test_japanese_is_recognised_in_a_mixed_line(self):
        assert debuglog.is_cjk("この件ですけど Tranium Access")
        assert not debuglog.is_cjk("Về mục tiêu")


class TestMeasure:
    def test_a_run_is_summarised(self, run):
        measured = debuglog.measure(run)
        assert measured["sentences"] == 2
        assert measured["partials"] == 4
        assert measured["japanese_sentences"] == 1
        assert measured["refused_translations"] == 1
        assert measured["compared"] == 2

    def test_spaced_japanese_shows_up_in_the_counts(self, tmp_path):
        path = tmp_path / "spaced.debug.txt"
        path.write_text(
            "14:41:38.168     17.3s  partial     [ja] この 多 数\n"
            "14:41:38.588     17.8s  final       #2 Speaker_01 [ja] "
            "この 多 数 ク は 今 朝\n",
            encoding="utf-8")
        measured = debuglog.measure(debuglog.parse(path))
        assert measured["japanese_spaced"] == 1
        assert measured["japanese_sentences"] == 1

    def test_a_word_committed_twice_shows_up_in_the_counts(self, tmp_path):
        path = tmp_path / "doubled.debug.txt"
        path.write_text(
            "14:41:31.888     11.1s  final       #1 Speaker_01 [vi] "
            "Hôm nay thì bác xe cây mới mới cung cấp\n",
            encoding="utf-8")
        measured = debuglog.measure(debuglog.parse(path))
        assert measured["with_repeats"] == 1
        assert measured["repeat_total"] == 1


class TestSameMeeting:
    """The check that has to come before every other number.

    Two runs of different recordings produce a page of differences that mean
    nothing, and the mistake is easy to make when the files are named by the
    day they were captured.
    """

    def write(self, tmp_path, name, sentences):
        path = tmp_path / name
        lines = []
        for index, (at, text) in enumerate(sentences, start=1):
            lines.append(f"00:00:00.000 {at:9.1f}s  final       "
                         f"#{index} Speaker_01 [vi] {text}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return debuglog.parse(path)

    def test_the_same_meeting_is_recognised_through_different_wording(
            self, tmp_path):
        left = self.write(tmp_path, "a.txt", [
            (10.0, "cung cấp thông tin về các training"),
            (20.0, "mình đang thực hiện test số 5"),
        ])
        right = self.write(tmp_path, "b.txt", [
            (11.0, "cung cấp thông tin về các trang trình"),
            (21.0, "hiện tại mình đang thực hiện test 05"),
        ])
        verdict = debuglog.same_meeting(left, right)
        assert verdict["same"]

    def test_two_different_meetings_are_refused(self, tmp_path):
        left = self.write(tmp_path, "a.txt", [
            (10.0, "cung cấp thông tin về các training"),
            (20.0, "mình đang thực hiện test số 5"),
        ])
        right = self.write(tmp_path, "b.txt", [
            (10.0, "vẫn đang bending cho Valkyrie"),
            (20.0, "nguyên nhân là biết rồi bác cũng nắm"),
        ])
        verdict = debuglog.same_meeting(left, right)
        assert not verdict["same"]

    def test_an_empty_run_is_not_claimed_to_match(self, tmp_path):
        left = self.write(tmp_path, "a.txt", [(10.0, "cung cấp thông tin")])
        right = self.write(tmp_path, "b.txt", [])
        assert not debuglog.same_meeting(left, right)["same"]
