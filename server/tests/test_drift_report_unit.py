"""Unit tests for the second cut at a drift run.

The run these read is a hand-written miniature of a real one: a sentence the
guards emptied, a sentence that lost an English word, a sentence cut at
max_duration that a trim destroyed, and one too short to have a running text
at all. Pure JSON, no audio and no model, so they run on the Dev PC.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.analysis import report  # noqa: E402


def drift_of(final: str, partial: str, flags, rewrite: float = 0.0,
             lost_latin=(), tail: str = "") -> dict:
    return {
        "partial": partial,
        "final": final,
        "rewrite": rewrite,
        "matched_start": 0,
        "matched_end": len(final),
        "tail_text": tail,
        "extra_audio_ms": 300.0,
        "tail_rate": 0.0,
        "body_rate": 0.0,
        "lost_latin": list(lost_latin),
        "gained_latin": [],
        "new_repeats": 0,
        "flags": list(flags),
    }


def case_of(index: int, reason: str, start_ms: float, end_ms: float,
            partial: str, finals: dict, partial_count: int = 5) -> dict:
    """One sentence, with a final and a drift per variant it produced text for."""
    entry = {
        "index": index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "reason": reason,
        "continues_previous": False,
        "lang_code": "vi",
        "partial_text": partial,
        "partial_count": partial_count,
        "extra_audio_ms": 300.0,
        "finals": {},
        "drift": {},
    }
    for name, (text, dropped, flags, rewrite, lost, tail) in finals.items():
        entry["finals"][name] = {"text": text, "seconds": 0.2,
                                 "audio_ms": end_ms - start_ms,
                                 "dropped": list(dropped)}
        if partial:
            entry["drift"][name] = drift_of(text, partial, flags, rewrite,
                                            lost, tail)
    return entry


@pytest.fixture
def run() -> dict:
    empty = ("", ["no speech"], ["final_empty", "rewrite", "latin_lost"],
             1.0, ["step"], "")
    kept = ("cái này hơi bị detail quá", [], [], 0.1, [], "")
    lost = ("về sau lưu sinh và cái lý do", [],
            ["rewrite", "latin_lost"], 0.4, ["solution"], "")
    invented = ("thì bác vẫn nhờ mình thích à không biết thôi đẹp", [],
                ["tail_invention"], 0.1, [], "không biết thôi đẹp")
    return {
        "wav": "recordings/meeting_16k.wav",
        "audio_seconds": 1821.2,
        "variants": [{"name": "baseline", "beam": 5, "trim_ms": 0.0},
                     {"name": "trim", "beam": 5, "trim_ms": 468.0}],
        "summary": [],
        "partial_seconds": 229.7,
        "final_seconds": {"baseline": 56.3, "trim": 54.5},
        "cases": [
            # Emptied by the baseline, recovered by the trim.
            case_of(0, "pause", 0.0, 5000.0, "thì hiện tại mình đang step 1",
                    {"baseline": empty, "trim": kept}),
            # Fine under the baseline, destroyed by the trim - it was cut
            # mid-word, so there was no hangover to remove.
            case_of(1, "max_duration", 5000.0, 12000.0, "cái này hơi bị detail",
                    {"baseline": kept, "trim": empty}),
            # A code-switched word the sentence traded away, under both.
            case_of(2, "pause", 12000.0, 15000.0, "về Solution và cái lý do",
                    {"baseline": lost, "trim": lost}),
            # A tail with nothing behind it.
            case_of(3, "pause", 15000.0, 21000.0, "thì bác vẫn nhờ mình fix à",
                    {"baseline": invented, "trim": invented}),
            # Too short to have produced a running text at all.
            case_of(4, "pause", 21000.0, 21400.0, "",
                    {"baseline": kept, "trim": kept}, partial_count=0),
        ],
    }


class TestShape:
    def test_variant_names_keep_their_order(self, run):
        assert report.variant_names(run) == ["baseline", "trim"]

    def test_only_sentences_with_a_running_text_are_comparable(self, run):
        assert len(report.comparable(run, "baseline")) == 4

    def test_an_empty_sentence_is_recognised(self, run):
        assert report.is_empty(run["cases"][0], "baseline")
        assert not report.is_empty(run["cases"][0], "trim")


class TestWhyEmpty:
    def test_the_guard_that_did_it_is_named(self, run):
        assert report.dropped_histogram(run, "baseline", empty_only=True) == {
            "no speech": 1}

    def test_refusals_on_sentences_that_still_produced_text_are_separable(
            self, run):
        # Over every sentence rather than the empty ones, the trim's own
        # refusal shows up too.
        assert report.dropped_histogram(run, "trim") == {"no speech": 1}

    def test_empties_are_split_by_why_the_sentence_was_committed(self, run):
        assert report.empties_by_reason(run, "baseline") == {
            "pause": (1, 3), "max_duration": (0, 1)}

    def test_empties_are_bucketed_by_length(self, run):
        table = report.empties_by_duration(run, "baseline")
        assert table["4-6s"] == (1, 1)
        # Buckets are lower-inclusive: a sentence of exactly 6.0 s is in the
        # last one, not the one before it.
        assert table["6s+"] == (0, 2)


class TestReslice:
    def test_holding_the_empties_apart_changes_the_latin_count(self, run):
        """The point of the whole module.

        An empty sentence loses every word the running text had, so counting
        it as a lost code-switch reports a problem that is really the other
        problem wearing its clothes.
        """
        everything = report.reslice(run, "baseline", exclude_empty=False)
        text_only = report.reslice(run, "baseline", exclude_empty=True)
        assert everything["counts"]["latin_lost"] == 2
        assert text_only["counts"]["latin_lost"] == 1
        assert everything["lost_words"] == ["solution", "step"]
        assert text_only["lost_words"] == ["solution"]

    def test_the_mean_rewrite_drops_once_the_empties_are_out(self, run):
        everything = report.reslice(run, "baseline", exclude_empty=False)
        text_only = report.reslice(run, "baseline", exclude_empty=True)
        assert text_only["mean_rewrite"] < everything["mean_rewrite"]

    def test_a_clean_sentence_is_counted_clean(self, run):
        assert report.reslice(run, "baseline", exclude_empty=True)["clean"] == 1


class TestTradeOffs:
    def test_a_variant_is_scored_in_both_directions_at_once(self, run):
        """A trim that recovers one pause and empties one max_duration cut is
        two findings, not a wash."""
        effect = report.variant_effect(run, "baseline", "trim")
        assert effect["pause"] == (1, 0, 2)
        assert effect["max_duration"] == (0, 1, 0)

    def test_invented_tails_are_listed_with_their_audio(self, run):
        found = report.tail_inventions(run, "baseline")
        assert len(found) == 1
        assert found[0]["index"] == 3
        assert found[0]["tail"] == "không biết thôi đẹp"

    def test_lost_latin_skips_the_empty_sentences(self, run):
        found = report.lost_latin_cases(run, "baseline")
        assert [case["index"] for case in found] == [2]
        assert found[0]["words"] == ["solution"]


class TestUncompared:
    def test_a_sentence_too_short_for_a_running_text_is_explained(self, run):
        missing = report.without_running_text(run)
        assert missing["total"] == 1
        assert missing["no_partial_at_all"] == 1
        assert missing["partials_all_refused"] == 0

    def test_a_sentence_whose_running_texts_were_all_refused_is_separate(self):
        run = {
            "wav": "x.wav", "audio_seconds": 10.0, "cases": [
                {"index": 0, "start_ms": 0.0, "end_ms": 4000.0,
                 "reason": "pause", "continues_previous": False,
                 "lang_code": "vi", "partial_text": "", "partial_count": 6,
                 "extra_audio_ms": 0.0, "finals": {}, "drift": {}},
            ],
            "variants": [{"name": "baseline", "beam": 5, "trim_ms": 0.0}],
        }
        missing = report.without_running_text(run)
        assert missing["partials_all_refused"] == 1
        assert missing["no_partial_at_all"] == 0


def scored_run() -> dict:
    """A run that recorded segment scores, so the guards can be replayed."""
    def piece(text, no_speech_prob, avg_logprob, verdict):
        return {"text": text, "avg_logprob": avg_logprob,
                "no_speech_prob": no_speech_prob, "compression_ratio": 1.6,
                "verdict": verdict}

    def case(index, reason, seconds, partial, text, pieces):
        return {
            "index": index, "start_ms": index * 10_000.0,
            "end_ms": index * 10_000.0 + seconds * 1000.0, "reason": reason,
            "continues_previous": False, "lang_code": "vi",
            "partial_text": partial, "partial_count": 5,
            "extra_audio_ms": 300.0, "drift": {},
            "finals": {"baseline": {"text": text, "seconds": 0.2,
                                    "audio_ms": seconds * 1000.0,
                                    "dropped": [], "pieces": pieces}},
        }

    return {
        "wav": "x.wav", "audio_seconds": 60.0,
        "variants": [{"name": "baseline", "beam": 5, "trim_ms": 0.0}],
        "cases": [
            # Refused as silence, but decoded confidently: the whole question.
            case(0, "pause", 6.5, "bác cũng đã nắm", "",
                 [piece(" bác cũng đã nắm rồi", 0.95, -0.2, "no speech")]),
            # Refused as silence and decoded badly too - nothing to recover.
            case(1, "max_duration", 7.0, "ừm", "",
                 [piece(" ừm à", 0.95, -1.4, "no speech")]),
            # Ordinary, untouched by either rule.
            case(2, "pause", 3.0, "cái này hơi detail", "cái này hơi detail",
                 [piece(" cái này hơi detail", 0.05, -0.2, "kept")]),
        ],
    }


class TestGuardRules:
    def test_confident_refusals_are_separated_from_hopeless_ones(self):
        """If every refused segment was also decoded badly, there is nothing
        for the rule change to recover and it is not the fix."""
        scores = report.no_speech_scores(scored_run(), "baseline")
        assert scores["refused"] == 2
        assert scores["confident"] == 1
        assert scores["unsure"] == 1

    def test_todays_policy_is_scored_in_both_directions(self):
        """The run was recorded under the old rule, so this is the recovery."""
        effect = report.guard_effect(scored_run(), "baseline", "current")
        assert effect["empty_before"] == 2
        assert effect["empty_after"] == 1
        assert [entry["index"] for entry in effect["recovered"]] == [0]
        assert effect["lost"] == []
        assert effect["recovered"][0]["after"] == "bác cũng đã nắm rồi"

    def test_recovered_sentences_are_split_by_commit_reason(self):
        effect = report.guard_effect(scored_run(), "baseline", "current")
        assert report.recovered_by_reason(effect) == {"pause": 1}
        assert report.recovered_seconds(effect) == pytest.approx(6.5)

    def test_replaying_the_rule_the_run_used_changes_nothing(self):
        """The sanity check on the whole simulation.

        The recorded text came out of ``no_speech_alone``. Replaying that
        same rule over the same scores has to reproduce it exactly, or the
        arithmetic is not the arithmetic the meeting ran on.
        """
        effect = report.guard_effect(scored_run(), "baseline",
                                     "no_speech_alone")
        assert effect["recovered"] == []
        assert effect["lost"] == []
        assert effect["changed"] == []

    def test_a_run_without_scores_says_so_instead_of_guessing(self, run,
                                                              capsys):
        report.print_guard_rules(run, "baseline", examples=3)
        assert "recorded no segment scores" in capsys.readouterr().out

    def test_the_recovered_text_is_printed_for_a_person_to_read(self, capsys):
        report.print_guard_rules(scored_run(), "baseline", examples=3)
        printed = capsys.readouterr().out
        assert "would show 'bác cũng đã nắm rồi'" in printed
        assert "empty sentences 2 -> 1" in printed


class TestMain:
    def test_it_reports_and_returns_zero(self, run, tmp_path, capsys):
        path = tmp_path / "drift.json"
        path.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
        assert report.main([str(path)]) == 0
        printed = capsys.readouterr().out
        assert "the empties held apart" in printed
        assert "no speech" in printed
        assert "solution" in printed

    def test_a_missing_file_is_refused(self, tmp_path, capsys):
        assert report.main([str(tmp_path / "nope.json")]) == 2
        assert "cannot read" in capsys.readouterr().out

    def test_a_run_with_no_sentences_is_refused(self, tmp_path, capsys):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"wav": "x", "cases": []}), encoding="utf-8")
        assert report.main([str(path)]) == 1
        assert "no sentences" in capsys.readouterr().out
