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

NEWLINE = chr(10)

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


class TestAlign:
    """Lining two runs up, and the two ways the earlier versions got it wrong.

    The first matched sentences inside a time window, and fell apart the
    first time a change moved the sentence boundaries. The second compared
    vocabularies as sets, and called two meetings the same because both were
    one team talking about one project. This votes on the offset: each word
    both runs said exactly once puts its two timestamps together and names
    the offset that would do it, and a real pairing puts the votes in one
    bin.
    """

    def write(self, tmp_path, name, sentences):
        path = tmp_path / name
        lines = [f"00:00:00.000 {at:9.1f}s  final       "
                 f"#{index} Speaker_01 [vi] {text}"
                 for index, (at, text) in enumerate(sentences, start=1)]
        path.write_text(NEWLINE.join(lines) + NEWLINE,
                        encoding="utf-8")
        return debuglog.parse(path)

    def meeting(self, count: int = 60, start: float = 10.0):
        """A meeting of distinct sentences, one distinctive word each."""
        return [(start + index * 7.0, f"phần việc soantu{index:03d} đã xong")
                for index in range(count)]

    def test_the_same_meeting_captured_later_is_recognised(self, tmp_path):
        """What a second capture of one recording looks like: the same words
        in the same order, every timestamp shifted by the same amount."""
        left = self.write(tmp_path, "a.txt", self.meeting())
        right = self.write(tmp_path, "b.txt",
                           [(at + 44.0, text) for at, text in self.meeting()])
        verdict = debuglog.align(left, right)
        assert verdict["same"]
        assert verdict["offset"] == pytest.approx(44.0, abs=4.0)

    def test_wording_that_changed_does_not_break_the_alignment(self, tmp_path):
        """Two decodes of one recording disagree about plenty of words. What
        they agree on is when the rest of them were said."""
        left = self.write(tmp_path, "a.txt", self.meeting())
        scrambled = [(at, text if index % 3 else "hoàn toàn khác biệt hẳn")
                     for index, (at, text) in enumerate(self.meeting())]
        right = self.write(tmp_path, "b.txt", scrambled)
        assert debuglog.align(left, right)["same"]

    def test_two_meetings_sharing_a_vocabulary_are_not_the_same_meeting(
            self, tmp_path):
        """The failure of comparing vocabularies as sets. Same team, same
        project, same jargon - said at unrelated moments."""
        import random
        words = self.meeting()
        left = self.write(tmp_path, "a.txt", words)
        shuffled = [text for _at, text in words]
        random.Random(7).shuffle(shuffled)
        right = self.write(tmp_path, "b.txt",
                           [(at, text) for (at, _), text
                            in zip(words, shuffled)])
        assert not debuglog.align(left, right)["same"]

    def test_two_runs_with_nothing_in_common_are_not_the_same(self, tmp_path):
        left = self.write(tmp_path, "a.txt", self.meeting())
        right = self.write(tmp_path, "b.txt", [
            (10.0 + index * 7.0, f"chuyện khác nusotu{index:03d} rồi đấy")
            for index in range(60)])
        assert not debuglog.align(left, right)["same"]

    def test_an_empty_run_is_not_claimed_to_match(self, tmp_path):
        left = self.write(tmp_path, "a.txt", self.meeting())
        right = self.write(tmp_path, "b.txt", [])
        verdict = debuglog.align(left, right)
        assert not verdict["same"]
        assert verdict["shared"] == 0

    def test_a_word_said_twice_casts_no_vote(self, tmp_path):
        """It offers two timestamps, so it can support two answers at once -
        which is how an earlier version reported 0.9 agreement between two
        unrelated meetings."""
        run = self.write(tmp_path, "a.txt", [
            (10.0, "chỉ nói soantu001 một lần"),
            (20.0, "nhắc lại soantu002 lần nữa"),
            (30.0, "nhắc lại soantu002 lần nữa"),
        ])
        rare = debuglog.rare_words(run)
        assert "soantu001" in rare
        assert "soantu002" not in rare

    def test_same_meeting_is_the_same_check(self, tmp_path):
        left = self.write(tmp_path, "a.txt", self.meeting())
        right = self.write(tmp_path, "b.txt",
                           [(at + 44.0, text) for at, text in self.meeting()])
        assert debuglog.same_meeting(left, right) == debuglog.align(left, right)
