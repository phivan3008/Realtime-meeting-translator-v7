"""Smoke tests for ``server/tests_real/test_real_translate.py``.

Drives the script, main() included, against a stub backend, so a crash in it
is caught here rather than after a round trip through the pod.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server.pipeline.translate import Translation, TranslationError, Translator  # noqa: E402


def load_harness():
    path = ROOT / "server" / "tests_real" / "test_real_translate.py"
    spec = importlib.util.spec_from_file_location("real_translate_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = load_harness()


class StubClient:
    """A model that translates, and that reads the history when given one.

    It answers in the script it was asked for. The earlier version replied
    with the same Japanese string whichever way it was pointed, which modelled
    a model that never translates out of Japanese - exactly the failure the
    wrong-script guard exists to catch, and the guard duly caught the stub.
    """

    model = "stub/qwen"
    source = "stub/qwen at http://stub/v1"

    #: Keyed by the language the prompt asks for.
    ANSWERS = {"Japanese": "こんにちは",
               "Vietnamese": "Xin chào"}
    #: What the history adds, in the same script as the answer.
    FROM_HISTORY = {"Japanese": "（その件）",
                    "Vietnamese": " (chuyện đó)"}

    def __init__(self, answer: str = "", chatty: bool = False,
                 ignores_history: bool = False):
        self.answer = answer
        self.chatty = chatty
        self.ignores_history = ignores_history
        self.calls: list[tuple[str, str]] = []

    def target_of(self, system: str) -> str:
        for name in self.ANSWERS:
            if f"Write it in {name}" in system:
                return name
        raise AssertionError(f"no target language in system prompt: {system!r}")

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        target = self.target_of(system)
        answer = self.answer if self.answer != "" else self.ANSWERS[target]
        if "do not translate" in user and not self.ignores_history:
            answer = f"{answer}{self.FROM_HISTORY[target]}"
        if self.chatty:
            return f"Sure! Here is the translation: {answer}"
        return answer


class BrokenClient:
    def complete(self, system: str, user: str) -> str:
        raise TranslationError("no translation server at http://stub/v1")


def attempt_of(source: str, text: str, seconds: float = 0.2,
               lang: str = "vi", reason: str = ""):
    target = "ja" if lang == "vi" else "vi"
    return harness.Attempt(
        source=source, lang_code=lang,
        result=Translation(text, source, lang, target, reason),
        seconds=seconds,
    )


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
def test_the_samples_cover_both_directions():
    languages = {lang for lang, _text in harness.SAMPLES}
    assert languages == {"vi", "ja"}


def test_the_context_case_needs_its_history_to_make_sense():
    """A bare Japanese "終わりました" has no subject to carry into Vietnamese
    unless the history supplies one."""
    assert harness.CONTEXT_HISTORY
    assert harness.CONTEXT_SENTENCE == ("ja", "終わりました。")


def test_expansion_is_measured_against_the_source():
    assert attempt_of("abcd", "abcdefgh").expansion == pytest.approx(2.0)
    assert attempt_of("", "x").expansion == 0.0


# ---------------------------------------------------------------------------
# Checks on the answers
# ---------------------------------------------------------------------------
def test_clean_translations_pass():
    report = harness.Report()
    harness.check_answers([attempt_of("xin chào", "こんにちは")], report)
    assert report.failed == []


def test_a_refused_sentence_is_caught():
    report = harness.Report()
    harness.check_answers(
        [attempt_of("xin chào", "", reason="the model returned nothing")],
        report)
    assert "Every sentence came back translated" in [c.name for c in report.failed]


def test_an_answer_that_is_just_the_input_is_caught():
    """A model that echoes has not translated anything."""
    report = harness.Report()
    harness.check_answers([attempt_of("xin chào", "xin chào")], report)
    assert "Nothing came back as its own input" in [c.name for c in report.failed]


def test_a_rambling_answer_is_caught():
    report = harness.Report()
    harness.check_answers([attempt_of("xin chào", "あ" * 100)], report)
    assert "No answer is far longer than its sentence" in [
        c.name for c in report.failed
    ]


def test_an_answer_that_talks_about_itself_is_caught():
    report = harness.Report()
    harness.check_answers(
        [attempt_of("xin chào", "Here is the translation of your sentence")],
        report)
    assert "No answer is a conversation about the translation" in [
        c.name for c in report.failed
    ]


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def test_a_fast_translation_passes(capsys):
    report = harness.Report()
    harness.check_latency([attempt_of("a", "b", seconds=0.3)], report)
    assert report.failed == []
    assert "stalls the socket" not in capsys.readouterr().out


def test_a_slow_translation_is_caught():
    report = harness.Report()
    harness.check_latency([attempt_of("a", "b", seconds=5.0)], report)
    assert "A translation is fast enough to stay on the audio path" in [
        c.name for c in report.failed
    ]


def test_a_translation_over_a_second_warns_about_the_audio_path(capsys):
    report = harness.Report()
    harness.check_latency([attempt_of("a", "b", seconds=1.5)], report)
    assert report.failed == []
    assert "stalls the socket" in capsys.readouterr().out


def test_latency_with_nothing_to_measure_is_not_judged():
    report = harness.Report()
    harness.check_latency([], report)
    assert report.checks == []


# ---------------------------------------------------------------------------
# Repeatability and context
# ---------------------------------------------------------------------------
def test_a_deterministic_backend_passes():
    client = StubClient()
    report = harness.Report()
    harness.check_repeatable(lambda: Translator(backend=client), report)
    assert report.failed == []


def test_a_backend_that_answers_differently_each_time_is_caught():
    answers = iter(["こんにちは", "こんばんは"])

    class Wandering:
        def complete(self, system, user):
            return next(answers)

    report = harness.Report()
    harness.check_repeatable(lambda: Translator(backend=Wandering()), report)
    assert "The same sentence twice gives the same translation" in [
        c.name for c in report.failed
    ]


def test_the_context_case_sends_the_history_to_the_model(capsys):
    client = StubClient()
    report = harness.Report()
    harness.check_context(client, report)
    assert report.failed == []
    with_history_prompt = client.calls[0][1]
    without_history_prompt = client.calls[1][1]
    assert "do not translate" in with_history_prompt
    assert "Bản dựng thứ ba" in with_history_prompt
    assert "do not translate" not in without_history_prompt
    assert "Identical output means" in capsys.readouterr().out


def test_a_model_that_ignores_the_history_is_caught():
    """The old check asserted only that both attempts answered, which no
    model could fail."""
    report = harness.Report()
    harness.check_context(StubClient(ignores_history=True), report)
    assert [c.name for c in report.failed] == [
        "The history changes the translation"
    ]


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def test_main_runs_and_passes(monkeypatch, capsys):
    monkeypatch.setattr(harness, "VllmClient", lambda **k: StubClient())
    monkeypatch.setattr(sys, "argv", ["x"])
    assert harness.main() == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out
    assert "Nothing here can tell you whether they are right" in out


def test_main_survives_a_chatty_model(monkeypatch, capsys):
    """The cleaning happens before any check sees the answer.

    Only the ``out:`` lines are examined. A ``raw:`` line is *supposed* to
    carry the model's preface verbatim - that is what it is for.
    """
    monkeypatch.setattr(harness, "VllmClient",
                        lambda **k: StubClient(chatty=True))
    monkeypatch.setattr(sys, "argv", ["x"])
    assert harness.main() == 0
    answers = [line for line in capsys.readouterr().out.splitlines()
               if line.strip().startswith("out:")]
    assert answers, "nothing was translated at all"
    assert not [line for line in answers if "Here is the translation" in line]


def test_main_reports_a_server_that_is_not_running(monkeypatch, capsys):
    def explode(**_kwargs):
        raise TranslationError("No translation server at http://127.0.0.1:8001/v1")

    monkeypatch.setattr(harness, "VllmClient", explode)
    monkeypatch.setattr(sys, "argv", ["x"])
    assert harness.main() == 2
    assert "No translation server" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Sentences a real meeting lost
# ---------------------------------------------------------------------------
def test_the_refusal_list_holds_both_kinds():
    """Mangled input and a complete sentence, or the check cannot separate
    the model's fault from the ASR's."""
    notes = [note for _lang, _source, note in harness.MEETING_REFUSALS]
    assert any(note.startswith("a complete") for note in notes)
    assert any(not note.startswith("a complete") for note in notes)


def test_a_working_model_passes_the_refusal_check():
    report = harness.Report()
    harness.check_meeting_refusals(StubClient(), report)
    assert report.failed == []


def test_a_model_that_refuses_the_complete_sentence_is_caught():
    class Echoing:
        """Hands every sentence straight back."""

        def complete(self, system: str, user: str) -> str:
            return user.strip().splitlines()[-1]

    report = harness.Report()
    harness.check_meeting_refusals(Echoing(), report)
    assert "A complete sentence is not refused" in [
        c.name for c in report.failed
    ]


def test_the_mangled_sentences_alone_do_not_fail_the_check():
    """A model cannot translate what the ASR never heard, and blaming it for
    that would make this check impossible to pass."""
    class OnlyTranslatesTheQuestion:
        def complete(self, system: str, user: str) -> str:
            line = user.strip().splitlines()[-1]
            if "\u3053\u3053\u306b\u4f5c\u3063\u3066" in line:
                return "\u0110ang t\u1ea1o \u1edf \u0111\u00e2y \u00e0?"
            return line

    report = harness.Report()
    harness.check_meeting_refusals(OnlyTranslatesTheQuestion(), report)
    assert report.failed == []


def test_the_raw_answer_is_printed_for_every_refusal(capsys):
    """The reason this list exists at all."""
    class Echoing:
        def complete(self, system: str, user: str) -> str:
            return user.strip().splitlines()[-1]

    harness.check_meeting_refusals(Echoing(), harness.Report())
    assert capsys.readouterr().out.count("raw    :") == len(
        harness.MEETING_REFUSALS)


# ---------------------------------------------------------------------------
# What is the history doing?
# ---------------------------------------------------------------------------
def test_the_cut_history_is_all_one_direction():
    """The shape under suspicion: every translation in the history lands in
    the language the next sentence is coming *from*."""
    assert {turn.lang_code for turn in harness.CUT_HISTORY} == {"vi"}
    assert [lang for lang, _s in harness.CUT_SENTENCES].count("ja") == 2


def test_the_cut_sentences_include_a_control_in_the_other_direction():
    """Cut by the same limit, going the other way, and the run translated it.
    Without it the cut and the direction cannot be told apart."""
    assert [lang for lang, _s in harness.CUT_SENTENCES].count("vi") == 1


def test_the_none_variant_carries_no_history():
    """It is the control; if it carried history it would prove nothing."""
    client = StubClient()
    harness.ask(client, "ja", "はい。", "none")
    assert "Earlier in the meeting" not in client.calls[0][1]


@pytest.mark.parametrize("variant", ["plain", "labelled", "sources"])
def test_every_other_variant_carries_the_history(variant):
    client = StubClient()
    harness.ask(client, "ja", "はい。", variant)
    assert "Earlier in the meeting" in client.calls[0][1]


def test_the_sources_variant_sends_no_translations():
    client = StubClient()
    harness.ask(client, "ja", "はい。", "sources")
    prompt = client.calls[0][1]
    assert "Cảm ơn." in prompt          # the source line is there
    assert "ありがとう" not in prompt   # its translation is not


def test_a_model_that_ignores_the_history_passes():
    report = harness.Report()
    harness.check_what_the_history_does(StubClient(), report)
    assert report.failed == []


def test_a_model_steered_only_by_the_plain_history_passes():
    """The style in use does its job, so nothing is reported against it."""
    class Steered:
        def complete(self, system: str, user: str) -> str:
            if "and into Vietnamese only" in user:
                return "Đang tạo ở đây à?"
            if "Write it in Japanese" in system:
                return "はい。"
            return "ここに作っているの？"

    report = harness.Report()
    harness.check_what_the_history_does(Steered(), report)
    assert report.failed == []


def test_a_style_that_does_not_fix_it_is_caught():
    """Steered by any history at all, however it is written.

    It must translate without one, or the sentence drops out of the reckoning
    as too broken - which is what an earlier version of this stub did, and it
    made the test pass while proving nothing.
    """
    class SteeredByAnyHistory:
        def complete(self, system: str, user: str) -> str:
            if "Write it in Japanese" in system:
                return "はい、承知しました。"
            if "Earlier in the meeting" not in user:
                return "Đang tạo ở đây à?"
            return "ここに作っているの？"

    report = harness.Report()
    harness.check_what_the_history_does(SteeredByAnyHistory(), report)
    assert [c.name for c in report.failed] == [
        f"The {harness.HISTORY_STYLE!r} history does not steer the language"
    ]


def test_a_sentence_that_fails_without_history_is_not_blamed_on_the_history(capsys):
    """Then the cut is the cause and no prompt change will translate it. It
    must also stop counting against the styles, or the check can never pass.
    """
    class CannotTranslateJapanese:
        def complete(self, system: str, user: str) -> str:
            if "Write it in Japanese" in system:
                return "はい。"
            return ""            # refuses every ja -> vi sentence

    report = harness.Report()
    harness.check_what_the_history_does(CannotTranslateJapanese(), report)
    assert report.failed == []
    assert "the cut is the cause, not the history" in capsys.readouterr().out


def test_a_style_that_is_not_earning_its_place_is_reported_as_such(capsys):
    """If every other style works too, HISTORY_STYLE is solving a problem
    this run does not show - and saying so matters more than keeping it."""
    class Working:
        def complete(self, system: str, user: str) -> str:
            if "Write it in Japanese" in system:
                return "はい。"
            return "Đang tạo ở đây à?"

    harness.check_what_the_history_does(Working(), harness.Report())
    assert "every other style translated them too" in capsys.readouterr().out


def test_every_variant_is_printed(capsys):
    harness.check_what_the_history_does(StubClient(), harness.Report())
    out = capsys.readouterr().out
    for variant in harness.HISTORY_VARIANTS:
        assert variant in out


# ---------------------------------------------------------------------------
# Short lines
# ---------------------------------------------------------------------------
def test_the_short_lines_include_the_one_that_failed():
    sources = [source for _lang, source, _note in harness.SHORT_LINES]
    assert "\u306f\u3044" in sources


def test_the_short_lines_include_ones_that_worked():
    """Otherwise there is nothing to say the trouble is single words rather
    than short lines in general."""
    notes = [note for _lang, _source, note in harness.SHORT_LINES]
    assert sum(1 for note in notes if note.startswith("translated")) >= 2


def test_the_short_lines_cover_both_directions():
    assert {lang for lang, _s, _n in harness.SHORT_LINES} == {"vi", "ja"}


def test_a_model_that_translates_short_lines_passes():
    report = harness.Report()
    harness.check_short_lines(StubClient(), report)
    assert report.failed == []


def test_a_model_that_hands_a_one_word_line_back_is_caught():
    """Only when the hint fails to help - the hint is what is in use."""
    class HandsBackShortLines:
        def complete(self, system: str, user: str) -> str:
            line = user.strip().splitlines()[-1]
            if len(line) <= 3:
                return line
            return ("\u3053\u3093\u306b\u3061\u306f"
                    if "Write it in Japanese" in system else "Xin ch\u00e0o")

    report = harness.Report()
    harness.check_short_lines(HandsBackShortLines(), report)
    assert "Every short line comes back translated" in [
        c.name for c in report.failed
    ]


def test_a_hint_that_rescues_the_line_passes():
    """The hint doing its job is the outcome this change is betting on."""
    class NeedsTheHint:
        def complete(self, system: str, user: str) -> str:
            line = user.strip().splitlines()[-1]
            if len(line) <= 3 and "still a line" not in system:
                return line
            return ("\u3053\u3093\u306b\u3061\u306f"
                    if "Write it in Japanese" in system else "V\u00e2ng")

    report = harness.Report()
    harness.check_short_lines(NeedsTheHint(), report)
    assert report.failed == []


def test_a_hint_that_changes_nothing_is_reported_as_such(capsys):
    """Then it should be removed rather than kept on faith."""
    harness.check_short_lines(StubClient(), harness.Report())
    assert "the plain prompt translated them all" in capsys.readouterr().out


def test_both_prompts_are_tried(capsys):
    harness.check_short_lines(StubClient(), harness.Report())
    out = capsys.readouterr().out
    assert "plain    :" in out
    assert "with hint:" in out


def test_the_short_lines_include_the_ones_still_failing():
    """A prompt change that fixes one must face the ones it has not."""
    sources = [source for _lang, source, _note in harness.SHORT_LINES]
    assert "\u30d0\u30c8\u30f3\u30bf\u30c3\u30c1" in sources
    assert "\u1edc" in sources        # U+1EDC is \u1edc; U+1EDE is \u1ede


def test_the_short_line_notes_say_what_happened_to_each():
    """The fixture is evidence, not a wish list: each line carries the
    outcome it actually had."""
    for _lang, source, note in harness.SHORT_LINES:
        assert note, source
        assert note.startswith(("translated", "refused", "failed")), note
