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
    """Answers every request with the same short line."""

    model = "stub/qwen"
    source = "stub/qwen at http://stub/v1"

    def __init__(self, answer: str = "こんにちは", chatty: bool = False):
        self.answer = answer
        self.chatty = chatty
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.chatty:
            return f"Sure! Here is the translation: {self.answer}"
        return self.answer


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
    """"Vậy thì tôi duyệt nó" means nothing without the previous lines."""
    assert harness.CONTEXT_HISTORY
    assert harness.CONTEXT_SENTENCE[1]


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
    assert "ngân sách" in with_history_prompt
    assert "do not translate" not in without_history_prompt
    assert "if they are identical" in capsys.readouterr().out


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
    """The cleaning happens before any check sees the answer."""
    monkeypatch.setattr(harness, "VllmClient",
                        lambda **k: StubClient(chatty=True))
    monkeypatch.setattr(sys, "argv", ["x"])
    assert harness.main() == 0
    assert "Here is the translation" not in capsys.readouterr().out.split(
        "Checks:")[1]


def test_main_reports_a_server_that_is_not_running(monkeypatch, capsys):
    def explode(**_kwargs):
        raise TranslationError("No translation server at http://127.0.0.1:8001/v1")

    monkeypatch.setattr(harness, "VllmClient", explode)
    monkeypatch.setattr(sys, "argv", ["x"])
    assert harness.main() == 2
    assert "No translation server" in capsys.readouterr().out
