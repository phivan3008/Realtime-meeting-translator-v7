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
