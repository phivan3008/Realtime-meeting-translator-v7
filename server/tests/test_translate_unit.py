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
    japanese_ratio,
    looks_like_echo,
    target_language,
    wrong_script,
)
from server.config import TRANSLATE_MAX_WRONG_SCRIPT


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


# ---------------------------------------------------------------------------
# Reasoning models
# ---------------------------------------------------------------------------
def test_a_reasoning_block_is_removed_before_the_answer_is_read():
    """Qwen3 thinks out loud first unless the chat template says otherwise."""
    assert clean("<think>The speaker greets everyone...</think>こんにちは") ==         "こんにちは"


def test_reasoning_is_removed_even_across_lines():
    answer = chr(10).join(["<think>", "first I consider", "then I decide",
                           "</think>", "こんにちは"])
    assert clean(answer) == "こんにちは"


def test_thinking_that_never_finished_leaves_nothing_to_show():
    """512 tokens of working and no answer is not a translation."""
    assert clean("<think>I should consider what the speaker means by") == ""


def test_a_sentence_merely_mentioning_think_is_untouched():
    assert clean("I think so") == "I think so"


def test_a_refusal_keeps_what_the_model_actually_said():
    """A guard that hides its evidence turns one bug into two."""
    raw = "<think>" + "reasoning " * 200
    result = make(raw).translate("xin chào", "vi")
    assert result.ok is False
    assert result.raw == raw
    assert "reasoning" in result.raw


def test_a_successful_translation_also_keeps_the_raw_answer():
    result = make("Sure! こんにちは").translate("xin chào", "vi")
    assert result.text == "こんにちは"
    assert result.raw == "Sure! こんにちは"


def test_thinking_is_switched_off_in_the_request():
    from server.config import TRANSLATE_ENABLE_THINKING

    assert TRANSLATE_ENABLE_THINKING is False


# ---------------------------------------------------------------------------
# Handing the sentence back untranslated
# ---------------------------------------------------------------------------
def test_an_identical_answer_is_an_echo():
    assert looks_like_echo("xin chào", "xin chào") is True


def test_swapping_the_full_stop_does_not_make_it_a_translation():
    """Exactly what Qwen did: the Vietnamese sentence back, with a Japanese
    full stop, which slipped past a plain equality check."""
    source = "Bản dựng thứ ba sẽ xong trước ngày 15 tháng 4."
    assert looks_like_echo(source, source[:-1] + "。") is True


def test_punctuation_and_spacing_alone_never_count_as_translation():
    assert looks_like_echo("Xin chào, mọi người!", "xin chào mọi người") is True


def test_a_real_translation_is_not_an_echo():
    assert looks_like_echo("xin chào", "こんにちは") is False


def test_an_echoed_sentence_is_refused():
    result = make("Bản dựng thứ ba。").translate("Bản dựng thứ ba.", "vi")
    assert result.ok is False
    assert "untranslated" in result.reason
    assert result.raw == "Bản dựng thứ ba。"


def test_the_prompt_no_longer_offers_to_repeat_the_line():
    """That clause invited the echo it was meant to allow for."""
    system, _user = make().build_prompt("xin chào", "vi")
    assert "repeat it" not in system.lower()
    assert "never repeat the line back" in system.lower()


# ---------------------------------------------------------------------------
# The answer has to be in the target language's script
#
# Every pair below was produced by the 60 s end-to-end run or by the Module 10
# translation test. They are the measurements the threshold was chosen from,
# kept here so that changing it has to face them.
#
# Written as escapes on purpose: a heredoc has mangled non-ASCII in this repo
# more than once, and a test whose data quietly became mojibake still passes.
# ---------------------------------------------------------------------------
REAL_INTO_VIETNAMESE = [
    ("\u30d7\u30ec\u30fc", "Play"),
    ("\u3042\u306e\u30bf\u30b9\u30af\u306e", "Task \u0111\u00f3"),
    ("\u5ba2\u6307\u793a\u3055\u3093\u304c\u4f5c\u3063\u3066\u304f\u308c\u305f"
     "\u30bf\u30b9\u30af\u306e\u30b5\u30de\u30ea\u30da\u30fc\u30b8\u306e"
     "\u30d5\u30a9\u30fc\u30de\u30c3\u30c8\u304c",
     "Format c\u1ee7a trang t\u00f3m t\u1eaft cho c\u00e1c task \u0111\u01b0"
     "\u1ee3c kh\u00e1ch ch\u1ec9 \u0111\u1ecbnh t\u1ea1o ra l\u00e0"),
    ("\u3042\u308b\u3068\u601d\u3046\u306e\u3067",
     "T\u00f4i ngh\u0129 l\u00e0 v\u1eady"),
    ("\u3053\u306e\u8fba\u306e\u30bf\u30b9\u30af\u3082\u5168\u90e8\u3042\u306e"
     "\u30d5\u30a9\u30fc\u30de\u30c3\u30c8\u4f5c\u308a\u305f\u3044\u306a\u3068"
     "\u601d\u3063\u3066\u308b\u3093\u3067\u3059\u3088\u306d",
     "T\u00f4i c\u0169ng mu\u1ed1n t\u1ea1o ra format t\u01b0\u01a1ng t\u1ef1 "
     "cho t\u1ea5t c\u1ea3 c\u00e1c task \u1edf khu v\u1ef1c n\u00e0y."),
    ("\u306f\u3044\u3001\u627f\u77e5\u3057\u307e\u3057\u305f\u3002",
     "V\u00e2ng, t\u00f4i \u0111\u00e3 hi\u1ec3u."),
]

REAL_INTO_JAPANESE = [
    # Opens with a Latin initialism kept as-is: 0.86, the lowest correct
    # measurement there is, and the reason the threshold is not higher.
    ("c\u00e1c FCG c\u00f3 t\u1eb7ng m\u1ed9t c\u00e1i th\u00eam h\u1ebft cho "
     "c\u00e1i xong th\u00ec b\u00e1c mu\u1ed1n t\u1ea5t c\u1ea3 c\u00e1c",
     "FCG \u304c\u8d08\u5448\u3059\u308b\u8ffd\u52a0\u5206\u304c\u5b8c\u4e86"
     "\u3057\u305f\u3089\u3001\u3059\u3079\u3066\u3092"),
    ("C\u00e1i tab n\u00e0y \u0111\u1ec1u vi\u1ebft theo c\u00e1i template "
     "c\u00f3 \u0111\u01b0\u1ee3c.",
     "\u3053\u306e\u30bf\u30d6\u306f\u3059\u3079\u3066\u3001\u53d6\u5f97\u3057"
     "\u305f\u30c6\u30f3\u30d7\u30ec\u30fc\u30c8\u306b\u5f93\u3063\u3066"
     "\u8a18\u8ff0\u3055\u308c\u3066\u3044\u307e\u3059\u3002"),
    ("C\u00e1i \u0111\u00f3 th\u00ec m\u00ecnh ch\u01b0a xem. Kh\u00f4ng "
     "bi\u1ebft l\u00e0 ph\u1ea3i l\u00e0 m\u1edbi update.",
     "\u305d\u306e\u4ef6\u306f\u78ba\u8a8d\u3057\u3066\u3044\u307e\u305b\u3093"
     "\u3002\u66f4\u65b0\u304c\u5fc5\u8981\u304b\u3069\u3046\u304b\u306f"
     "\u308f\u304b\u308a\u307e\u305b\u3093\u3002"),
    ("B\u1ea3n d\u1ef1ng th\u1ee9 ba s\u1ebd xong tr\u01b0\u1edbc ng\u00e0y "
     "15 th\u00e1ng 4.",
     "\u7b2c 3 \u7248\u306f 4 \u6708 15 \u65e5\u307e\u3067\u306b\u5b8c\u4e86"
     "\u3057\u307e\u3059\u3002"),
]

#: The sentence that started this: Japanese in, Japanese out, and every guard
#: that existed let it through.
UNTRANSLATED = ("\u306f\u3044\u3001\u4eca\u306e\u753b\u9762\u306e",
                "\u306f\u3044\u3001\u73fe\u5728\u306e\u753b\u9762\u306e")


def test_the_fixtures_survived_being_written_to_disk():
    """A mangled fixture would make every test below pass for nothing."""
    assert REAL_INTO_VIETNAMESE[0][0] == "\u30d7\u30ec\u30fc"
    assert japanese_ratio(REAL_INTO_VIETNAMESE[0][0]) == 1.0
    assert japanese_ratio("Task \u0111\u00f3") == 0.0


@pytest.mark.parametrize("source,text", REAL_INTO_VIETNAMESE)
def test_a_real_vietnamese_translation_is_accepted(source, text):
    assert not wrong_script(text, "vi")


@pytest.mark.parametrize("source,text", REAL_INTO_JAPANESE)
def test_a_real_japanese_translation_is_accepted(source, text):
    assert not wrong_script(text, "ja")


def test_the_sentence_that_came_back_in_its_own_language_is_caught():
    _source, text = UNTRANSLATED
    assert wrong_script(text, "vi")


def test_the_threshold_sits_between_the_two_measured_groups():
    """Not chosen by taste: correct answers into Vietnamese measured 0.00, and
    correct answers into Japanese measured 0.86 upwards."""
    into_vi = [japanese_ratio(text) for _s, text in REAL_INTO_VIETNAMESE]
    into_ja = [japanese_ratio(text) for _s, text in REAL_INTO_JAPANESE]
    assert max(into_vi) < TRANSLATE_MAX_WRONG_SCRIPT < min(into_ja)


def test_a_bare_number_is_not_judged():
    """"15" translates to "15", and refusing that refuses a correct answer."""
    assert japanese_ratio("15") is None
    assert not wrong_script("15", "ja")
    assert not wrong_script("15", "vi")


def test_an_unknown_target_is_not_judged():
    assert not wrong_script("anything at all", "")


def test_the_guard_refuses_through_the_translator():
    """End to end through Translator, not just the helper."""
    source, echoed = UNTRANSLATED

    class Echoing:
        def complete(self, system: str, user: str) -> str:
            return echoed

    result = Translator(backend=Echoing()).translate(source, "ja")
    assert not result.ok
    assert result.reason == "the answer is not written in vi"
    # The refusal keeps what it refused, or the next bug costs a round trip.
    assert result.raw == echoed


def test_a_good_translation_still_passes_through_the_translator():
    answer = "V\u00e2ng, t\u00f4i \u0111\u00e3 hi\u1ec3u."

    class Working:
        def complete(self, system: str, user: str) -> str:
            return answer

    result = Translator(backend=Working()).translate(
        "\u306f\u3044\u3001\u627f\u77e5\u3057\u307e\u3057\u305f\u3002", "ja")
    assert result.ok
    assert result.text == answer
