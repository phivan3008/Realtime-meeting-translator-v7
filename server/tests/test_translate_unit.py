"""Unit tests for the translation stage.

The model is stubbed - including stubs that behave like a chatty
instruction-tuned model - so these run without vLLM. Whether Qwen actually
translates well is a question for ``server/tests_real/test_real_translate.py``
with a server running.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import TRANSLATE_HISTORY, TRANSLATE_MODEL, TRANSLATE_PAIR
from server.pipeline.translate import (
    TranslationContext,
    TranslationError,
    Translator,
    Turn,
    clean,
    target_language,
)


class StubBackend:
    """Answers with a scripted reply and records what it was asked."""

    def __init__(self, *answers: str):
        self.answers = list(answers) or [""]
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]


class BrokenBackend:
    def complete(self, system: str, user: str) -> str:
        raise TranslationError("no translation server at http://127.0.0.1:8001/v1")


def make(*answers: str, **kwargs) -> Translator:
    return Translator(backend=StubBackend(*answers), **kwargs)


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------
def test_the_pair_is_the_two_meeting_languages():
    assert TRANSLATE_PAIR == {"vi": "ja", "ja": "vi"}


def test_each_language_becomes_the_other():
    assert target_language("vi") == "ja"
    assert target_language("ja") == "vi"


def test_an_unknown_language_has_no_direction():
    assert target_language("") == ""
    assert target_language("en") == ""


# ---------------------------------------------------------------------------
# Cleaning what a chatty model returns
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("answer, expected", [
    ("こんにちは", "こんにちは"),
    ("  こんにちは  ", "こんにちは"),
    ("Sure! Here is the translation: こんにちは", "こんにちは"),
    ("Here's the translation:\nこんにちは", "こんにちは"),
    ("Translation: こんにちは", "こんにちは"),
    ("翻訳：こんにちは", "こんにちは"),
    ('"こんにちは"', "こんにちは"),
    ("「こんにちは」", "こんにちは"),
    ("Certainly: 「こんにちは」", "こんにちは"),
])
def test_the_model_answering_the_request_is_stripped_back_to_the_answer(
    answer, expected
):
    """All of this would otherwise be shown as if somebody had said it."""
    assert clean(answer) == expected


def test_cleaning_leaves_a_sentence_that_merely_starts_with_a_colon_word():
    assert clean("Xin chào: hôm nay họp lúc mấy giờ?") == \
        "Xin chào: hôm nay họp lúc mấy giờ?"


def test_cleaning_does_not_strip_a_lone_quote_mark():
    assert clean('"') == '"'


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def turn(source: str, translation: str = "x", lang: str = "vi",
         speaker: str = "Speaker_01") -> Turn:
    return Turn(speaker_id=speaker, lang_code=lang, source=source,
                translation=translation)


def test_the_history_keeps_only_the_last_few_turns():
    """Enough to resolve a pronoun, not enough to summarise the meeting."""
    context = TranslationContext(size=3)
    for index in range(5):
        context.remember(turn(f"line {index}"))
    assert [t.source for t in context.turns] == ["line 2", "line 3", "line 4"]


def test_the_default_history_is_what_design_md_asks_for():
    assert TRANSLATE_HISTORY == 3


def test_the_history_prompt_names_the_speaker_and_the_language():
    context = TranslationContext()
    context.remember(turn("xin chào", "こんにちは"))
    text = context.as_prompt()
    assert "Speaker_01 (vi): xin chào" in text
    assert "-> こんにちは" in text


def test_an_empty_history_contributes_nothing():
    assert TranslationContext().as_prompt() == ""


def test_a_history_of_zero_remembers_nothing():
    context = TranslationContext(size=0)
    context.remember(turn("x"))
    assert context.turns == []


def test_a_negative_history_is_rejected():
    with pytest.raises(ValueError):
        TranslationContext(size=-1)


def test_clearing_the_history_forgets_the_meeting():
    context = TranslationContext()
    context.remember(turn("x"))
    context.clear()
    assert context.turns == []


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------
def test_the_prompt_names_both_languages():
    system, _user = make().build_prompt("xin chào", "vi")
    assert "Vietnamese" in system
    assert "Japanese" in system


def test_the_prompt_asks_for_the_translation_alone():
    system, _user = make().build_prompt("xin chào", "vi")
    assert "translation alone" in system
    assert "no explanation" in system


def test_the_sentence_to_translate_is_the_last_thing_in_the_prompt():
    translator = make()
    translator.context.remember(turn("câu trước"))
    _system, user = translator.build_prompt("câu này", "vi")
    assert user.strip().endswith("câu này")


def test_the_history_is_marked_as_context_not_as_work():
    translator = make()
    translator.context.remember(turn("câu trước"))
    _system, user = translator.build_prompt("câu này", "vi")
    assert "do not translate" in user
    assert "câu trước" in user


def test_without_history_the_prompt_is_just_the_sentence():
    _system, user = make().build_prompt("xin chào", "vi")
    assert "context" not in user
    assert user.endswith("xin chào")


# ---------------------------------------------------------------------------
# Translating
# ---------------------------------------------------------------------------
def test_a_sentence_is_translated_and_reported_both_ways():
    result = make("こんにちは").translate("xin chào", "vi")
    assert result.text == "こんにちは"
    assert result.source == "xin chào"
    assert result.lang_code == "vi"
    assert result.target == "ja"
    assert result.ok is True


def test_a_chatty_answer_is_cleaned_before_it_is_used():
    result = make("Sure! Here is the translation: こんにちは").translate(
        "xin chào", "vi")
    assert result.text == "こんにちは"


def test_a_translated_sentence_joins_the_history():
    translator = make("こんにちは")
    translator.translate("xin chào", "vi", speaker_id="Speaker_02")
    remembered = translator.context.turns[-1]
    assert remembered.source == "xin chào"
    assert remembered.translation == "こんにちは"
    assert remembered.speaker_id == "Speaker_02"


def test_an_empty_sentence_is_not_sent_to_the_model():
    backend = StubBackend("こんにちは")
    result = Translator(backend=backend).translate("   ", "vi")
    assert result.ok is False
    assert result.reason == "nothing to translate"
    assert backend.calls == []


def test_an_undecided_language_is_not_translated_in_a_guessed_direction():
    """Guessing is how a Vietnamese line comes back as Vietnamese again."""
    backend = StubBackend("こんにちは")
    result = Translator(backend=backend).translate("xin chào", "")
    assert result.ok is False
    assert "undecided" in result.reason
    assert backend.calls == []


def test_an_answer_far_longer_than_the_sentence_is_refused():
    """The model explaining itself, or looping, is not a translation."""
    rambling = "こんにちは。" + "この文は" * 200
    result = make(rambling).translate("xin chào", "vi")
    assert result.ok is False
    assert "far longer" in result.reason


def test_a_slightly_longer_answer_is_fine():
    """Japanese runs longer than Vietnamese; that is not a failure."""
    result = make("こんにちは、よろしくお願いします").translate("xin chào", "vi")
    assert result.ok is True


def test_an_empty_answer_is_refused():
    result = make("   ").translate("xin chào", "vi")
    assert result.ok is False
    assert "returned nothing" in result.reason


def test_a_refused_answer_does_not_poison_the_history():
    translator = make("")
    translator.translate("xin chào", "vi")
    assert translator.context.turns == []


def test_a_backend_that_is_down_is_reported_not_raised():
    """A meeting with no translations is worse than one that stops."""
    translator = Translator(backend=BrokenBackend())
    result = translator.translate("xin chào", "vi")
    assert result.ok is False
    assert "no translation server" in result.reason
    assert translator.stats.failed == 1


def test_the_expansion_limit_is_configurable():
    with pytest.raises(ValueError):
        make("x", max_expansion=1.0)
    assert make("あ" * 20, max_expansion=10.0).translate("xin chào", "vi").ok


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
def test_stats_separate_translated_refused_and_failed():
    translator = make("こんにちは", "", "こんばんは")
    translator.translate("xin chào", "vi")
    translator.translate("chào buổi tối", "vi")
    translator.translate("xin chào", "")
    assert translator.stats.seen == 3
    assert translator.stats.translated == 1
    assert translator.stats.refused == 2
    assert set(translator.stats.refused_reasons) == {
        "the model returned nothing", "the language was undecided"
    }


def test_reset_forgets_the_meeting_and_the_counters():
    translator = make("こんにちは")
    translator.translate("xin chào", "vi")
    translator.reset()
    assert translator.context.turns == []
    assert translator.stats.seen == 0


# ---------------------------------------------------------------------------
# Which checkpoint is actually serving
# ---------------------------------------------------------------------------
def test_the_configured_checkpoint_is_the_one_design_md_names():
    assert TRANSLATE_MODEL == "Qwen/Qwen3.5-9B"


def choose(wanted: str, served: list[str]) -> str:
    from server.pipeline.translate import choose_model

    return choose_model(wanted, served, "http://stub/v1")


def test_the_configured_model_is_taken_when_the_server_has_it():
    assert choose("Qwen/Qwen3.5-9B",
                  ["Qwen/Qwen3.5-9B"]) == "Qwen/Qwen3.5-9B"


def test_a_server_running_the_wrong_checkpoint_is_refused():
    """Started on the wrong model, vLLM answers happily and only the
    translations are worse - which no log would ever show."""
    with pytest.raises(TranslationError) as caught:
        choose("Qwen/Qwen3.5-9B", ["Qwen/Qwen2.5-7B-Instruct"])
    message = str(caught.value)
    assert "Qwen/Qwen3.5-9B" in message
    assert "Qwen/Qwen2.5-7B-Instruct" in message
    assert "--model" in message


def test_an_empty_setting_accepts_whatever_is_running():
    assert choose("", ["some/model", "other/model"]) == "some/model"


def test_a_server_with_no_model_is_refused():
    with pytest.raises(TranslationError, match="serving no model"):
        choose("Qwen/Qwen3.5-9B", [])
