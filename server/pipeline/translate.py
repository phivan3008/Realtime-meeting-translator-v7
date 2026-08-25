"""Translation - step 8 of the server pipeline.

``DESIGN.md`` 3.8: Qwen behind vLLM, given the sentence Whisper committed,
the language the LID decided, and the previous few turns of the meeting.

The history is there to resolve pronouns and dropped subjects, which both
languages leave out freely. Three turns, not thirty: enough for "that one",
short enough that the model cannot start summarising instead of translating.

Instruction-tuned models answer requests, so an answer arrives wrapped in
"Sure! Here is the translation:" or in quotes, and all of it would be shown
as if somebody had said it. The answer is cleaned, then checked.

Layering:

``TranslationContext``
    The rolling history. Pure Python.

``Translator``
    Prompt, cleaning and guards. Pure Python, tested against a stub.

``VllmClient``
    The HTTP call - the only part needing a server.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Protocol

from server.config import (
    LANGUAGE_NAMES,
    TRANSLATE_BASE_URL,
    TRANSLATE_ENABLE_THINKING,
    HISTORY_STYLE,
    SHORT_LINE_HINT_ENABLED,
    TRANSLATE_HISTORY,
    TRANSLATE_EXPANSION_SLACK,
    TRANSLATE_MAX_EXPANSION,
    TRANSLATE_MAX_WRONG_SCRIPT,
    TRANSLATE_MAX_TOKENS,
    TRANSLATE_MODEL,
    TRANSLATE_PAIR,
    TRANSLATE_TEMPERATURE,
    TRANSLATE_TIMEOUT_S,
)

log = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    """Raised when the translation backend cannot be reached or used."""


#: A polite opener, which the model may put in front of everything else.
_INTERJECTION = re.compile(
    r"^\s*(sure|certainly|of course|okay|ok|got it|alright)\b[!,.:：]*\s*",
    re.IGNORECASE,
)
#: The model announcing what it is about to give you.
_LEAD_IN = re.compile(
    r"^\s*(here(?:'s| is)(?: the)?[^:\r\n]{0,40}|the translation|translation|"
    r"translated(?: text)?|dịch|bản dịch|翻訳|訳)\s*[:：]\s*",
    re.IGNORECASE,
)
#: Reasoning models answer with their working first. Qwen3 emits this unless
#: the chat template is told otherwise, and a server that ignores the flag
#: would otherwise return 512 tokens of thinking and no translation.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)

#: Quote pairs a model reaches for when asked for exactly one line.
_QUOTES = (('"', '"'), ("'", "'"), ("「", "」"), ("“", "”"), ("『", "』"))


def clean(answer: str) -> str:
    """Strip what the model added around the translation.

    Applied repeatedly, because the additions stack: "Sure! Here is the
    translation: 「...」" is three of them in front of one sentence.
    """
    text = _THINK.sub("", answer).strip()
    if _UNCLOSED_THINK.match(text):
        # The whole budget went on thinking and the answer never arrived.
        return ""
    for _ in range(4):
        before = text
        text = _INTERJECTION.sub("", text, count=1).strip()
        text = _LEAD_IN.sub("", text, count=1).strip()
        for opening, closing in _QUOTES:
            if len(text) >= 2 and text.startswith(opening) and text.endswith(closing):
                text = text[1:-1].strip()
                break
        if text == before:
            break
    return text


def choose_model(wanted: str, served: list[str], where: str = "") -> str:
    """Insist on the configured checkpoint, or take what there is.

    Started against the wrong model, vLLM answers happily and the only symptom
    is translations that are worse than they should be - exactly the kind of
    difference that survives every log and every dashboard. So it is checked
    once, by name, at connect time.
    """
    if not served:
        raise TranslationError(f"{where or 'the server'} is serving no model")
    if not wanted:
        return served[0]
    if wanted in served:
        return wanted
    raise TranslationError(
        f"{where or 'the server'} is serving {served} but this pipeline is "
        f"configured for {wanted!r}. Start vLLM with --model {wanted}, or set "
        "TRANSLATE_MODEL to accept what is running."
    )


#: Anything that is punctuation, a symbol, or a space. Two sentences that
#: differ only in these are the same sentence.
_IGNORABLE = re.compile(r"[\s\W_]+", re.UNICODE)


def looks_like_echo(source: str, text: str) -> bool:
    """Did the model hand the sentence back instead of translating it?

    Compared without punctuation, because a model that echoes often swaps the
    full stop for the other language's - and one character was enough to slip
    a completely untranslated Vietnamese sentence past a plain equality check.
    """
    return (_IGNORABLE.sub("", source).casefold()
            == _IGNORABLE.sub("", text).casefold())


def japanese_ratio(text: str) -> Optional[float]:
    """Fraction of the letters that are kana or kanji, or None if no letters.

    Vietnamese and Japanese share no script, so this tells cheaply whether a
    translation came out in the language it was asked for. It cannot tell
    Vietnamese from English: ``プレー`` answered as ``Play`` still passes.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    japanese = sum(
        1 for c in letters
        if unicodedata.name(c, "").startswith(("HIRAGANA", "KATAKANA", "CJK"))
    )
    return japanese / len(letters)


def wrong_script(text: str, target: str,
                 limit: float = TRANSLATE_MAX_WRONG_SCRIPT) -> bool:
    """Is this answer written in the wrong language's script?

    Catches a sentence handed back in its own language when the wording
    changed enough that ``looks_like_echo`` cannot see it.
    """
    ratio = japanese_ratio(text)
    if ratio is None:
        return False
    if target == "ja":
        return ratio < limit
    if target == "vi":
        return ratio > limit
    return False


def target_language(lang_code: str) -> str:
    """Which language this sentence should become."""
    return TRANSLATE_PAIR.get(lang_code, "")


@dataclass(frozen=True)
class Turn:
    """One sentence of the meeting, as both languages."""

    speaker_id: str
    lang_code: str
    source: str
    translation: str


class TranslationContext:
    """The last few turns, and nothing older."""

    def __init__(self, size: int = TRANSLATE_HISTORY) -> None:
        if size < 0:
            raise ValueError("history size cannot be negative")
        self.size = size
        self._turns: deque[Turn] = deque(maxlen=size) if size else deque(maxlen=0)

    def remember(self, turn: Turn) -> None:
        self._turns.append(turn)

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    #: How the history is written into the prompt.
    #:
    #: ``plain``     source and translation, translation unmarked. The original.
    #: ``labelled``  the same, with each translation's language named.
    #: ``sources``   the source lines only. No translations, so no worked
    #:               examples, so nothing to imitate.
    STYLES = ("plain", "labelled", "sources")

    def as_prompt(self, style: str = HISTORY_STYLE) -> str:
        """The history, laid out for the model to read but not to translate.

        ``style`` decides how much of it the model sees:

        ``plain``     source and translation, translation unmarked
        ``labelled``  the same, with each translation's language named
        ``sources``   source lines only, so there are no worked examples

        With its translations in it the history reads as examples, and when
        several turns run the same way the model copies their language
        instead of following the instruction. See docs/TUNING.md.
        """
        if style not in self.STYLES:
            raise ValueError(f"unknown history style {style!r}, "
                             f"expected one of {self.STYLES}")
        if not self._turns:
            return ""
        lines = []
        for turn in self._turns:
            who = turn.speaker_id or "someone"
            lines.append(f"{who} ({turn.lang_code}): {turn.source}")
            if turn.translation and style != "sources":
                target = target_language(turn.lang_code)
                tag = f"({target}) " if style == "labelled" and target else ""
                lines.append(f"  -> {tag}{turn.translation}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()


@dataclass(frozen=True)
class Translation:
    """A translated sentence, or the reason there is not one."""

    text: str
    source: str
    lang_code: str
    target: str
    reason: str = ""
    #: What the model actually said, kept so a refusal can be looked at.
    #: A guard that hides its evidence turns one bug into two.
    raw: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


@dataclass
class TranslationStats:
    seen: int = 0
    translated: int = 0
    refused: int = 0
    failed: int = 0
    refused_reasons: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    def record(self, result: Translation, failed: bool = False) -> None:
        self.seen += 1
        if failed:
            self.failed += 1
        elif result.ok:
            self.translated += 1
        else:
            self.refused += 1
            self.refused_reasons[result.reason] = (
                self.refused_reasons.get(result.reason, 0) + 1
            )


class Backend(Protocol):
    """What :class:`Translator` needs; a stub satisfies it in the tests."""

    def complete(self, system: str, user: str) -> str:
        ...                                         # pragma: no cover


SYSTEM_PROMPT = (
    "You are a translator inside a live meeting transcript. "
    "The final line is in {source_name}. Write it in {target_name}. "
    "Reply with the {target_name} translation alone: no preface, no quotes, no "
    "notes, no explanation, no romanisation. Keep names, numbers and technical "
    "terms exactly as they are. Never repeat the line back in {source_name}."
)

#: Appended to the system prompt. A meeting is mostly short lines, and a
#: single word is what the model treats as nothing to do.
SHORT_LINE_HINT = (
    " A line of one word, an interjection or a filler is still a line: write "
    "it in {target_name} too, never leave it as it was."
)


class Translator:
    """Turn one committed sentence into the other language."""

    def __init__(
        self,
        backend: Optional[Backend] = None,
        context: Optional[TranslationContext] = None,
        max_expansion: dict[str, float] | float = TRANSLATE_MAX_EXPANSION,
        expansion_slack: int = TRANSLATE_EXPANSION_SLACK,
        history_style: str = HISTORY_STYLE,
        short_line_hint: bool = SHORT_LINE_HINT_ENABLED,
    ) -> None:
        if isinstance(max_expansion, (int, float)):
            max_expansion = {code: float(max_expansion)
                             for code in TRANSLATE_MAX_EXPANSION}
        if any(value <= 0.0 for value in max_expansion.values()):
            raise ValueError("every max_expansion must be greater than 0")
        self.backend = backend if backend is not None else VllmClient()
        self.context = context if context is not None else TranslationContext()
        self.max_expansion = dict(max_expansion)
        self.expansion_slack = expansion_slack
        if history_style not in TranslationContext.STYLES:
            raise ValueError(f"unknown history style {history_style!r}")
        self.history_style = history_style
        self.short_line_hint = short_line_hint
        self.stats = TranslationStats()

    def build_prompt(self, source: str, lang_code: str,
                     history_style: str = "",
                     short_line_hint: Optional[bool] = None) -> tuple[str, str]:
        """The system and user messages, so a test can read them.

        ``history_style`` overrides this translator's own, which is how
        server/tests_real/test_real_translate.py puts every version to a live
        model on the same history.
        """
        style = history_style or self.history_style
        hint = self.short_line_hint if short_line_hint is None else short_line_hint
        target = target_language(lang_code)
        target_name = LANGUAGE_NAMES.get(target, "the other language")
        system = (SYSTEM_PROMPT + (SHORT_LINE_HINT if hint else "")).format(
            source_name=LANGUAGE_NAMES.get(lang_code, "the source language"),
            target_name=target_name,
        )
        history = self.context.as_prompt(style=style)
        if not history:
            return system, f"Translate this line into {target_name}:\n{source}"
        if style == "plain":
            return system, (
                "Earlier in the meeting, for context only - do not translate "
                f"these:\n{history}\n\nTranslate this line:\n{source}"
            )
        # The instruction is repeated after the history as well as in the
        # system message. The history is the last text the model reads before
        # answering, and a run of lines all going the same way outweighed a
        # system prompt asking for the other one.
        user = (
            "Earlier in the meeting, for context only - do not translate "
            f"these:\n{history}\n\nNow translate the following line "
            f"into {target_name}, and into {target_name} only:\n{source}"
        )
        return system, user

    def translate(self, source: str, lang_code: str,
                  speaker_id: str = "") -> Translation:
        target = target_language(lang_code)
        if not source.strip():
            return self._refuse("", lang_code, target, "nothing to translate")
        if not target:
            # Without a language there is no direction to translate in, and
            # guessing one is how a Vietnamese sentence comes back as
            # Vietnamese-flavoured Japanese.
            return self._refuse(source, lang_code, target,
                                "the language was undecided")

        system, user = self.build_prompt(source, lang_code)
        try:
            answer = self.backend.complete(system, user)
        except TranslationError as exc:
            result = Translation("", source, lang_code, target, str(exc))
            self.stats.record(result, failed=True)
            log.warning("Translation failed: %s", exc)
            return result

        text = clean(answer)
        reason = self._refuse_reason(source, text, target)
        if reason:
            return self._refuse(source, lang_code, target, reason, answer)

        result = Translation(text, source, lang_code, target, raw=answer)
        self.stats.record(result)
        self.context.remember(
            Turn(speaker_id=speaker_id, lang_code=lang_code, source=source,
                 translation=text)
        )
        return result

    def _refuse_reason(self, source: str, text: str,
                       target: str = "") -> str:
        if not text:
            return "the model returned nothing"
        if looks_like_echo(source, text):
            # Handed back untranslated. Showing it would put the same sentence
            # in both columns and read as though the translation succeeded.
            return "the model returned the sentence untranslated"
        if wrong_script(text, target):
            # Answered in the language it was asked to translate *out* of.
            # Showing it would put Japanese in the Vietnamese column, which
            # reads as a translation until someone tries to read it.
            return f"the answer is not written in {target or 'the target'}"
        if len(text) > self.length_limit(source, target):
            # Not a translation any more: the model started explaining, or
            # looped, or answered a question nobody asked.
            return "the answer is far longer than the sentence"
        return ""

    def length_limit(self, source: str, target: str) -> float:
        """How long an answer may run before it stops being a translation.

        Per direction: Japanese carries the same meaning in far fewer
        characters. The slack keeps short sources out of it, where a ratio is
        mostly noise. See docs/TUNING.md.
        """
        ratio = self.max_expansion.get(target)
        if ratio is None:
            # An unknown target has no measurements behind it; judging it on
            # a number borrowed from another language pair would be a guess.
            return float("inf")
        return len(source) * ratio + self.expansion_slack

    def _refuse(self, source: str, lang_code: str, target: str,
                reason: str, raw: str = "") -> Translation:
        result = Translation("", source, lang_code, target, reason, raw)
        self.stats.record(result)
        if raw:
            log.info("Refused a translation (%s): %r", reason, raw[:200])
        return result

    def reset(self) -> None:
        """A new meeting remembers nothing of the last one."""
        self.context.clear()
        self.stats = TranslationStats()


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------
class VllmClient:
    """vLLM's OpenAI-compatible chat completions, over HTTP.

    Uses urllib rather than a client library on purpose: this is one POST to
    one endpoint on localhost, and the server process already carries enough
    dependencies.
    """

    def __init__(self, base_url: str = "", model: str = "",
                 timeout: float = TRANSLATE_TIMEOUT_S) -> None:
        self.base_url = (base_url or TRANSLATE_BASE_URL).rstrip("/")
        self.timeout = timeout
        wanted = model or TRANSLATE_MODEL
        served = self.served_models()
        self.model = choose_model(wanted, served, self.base_url)
        log.info("Translation backend ready: %s at %s", self.model,
                 self.base_url)

    def served_models(self) -> list[str]:
        """What this server is actually serving."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/models",
                                        timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise TranslationError(
                f"No translation server at {self.base_url}: {exc}. Start vLLM "
                f"with `python3.11 -m vllm.entrypoints.openai.api_server "
                f"--model {TRANSLATE_MODEL or '<checkpoint>'} --port 8001`."
            ) from exc
        return [str(entry.get("id", "")) for entry in (payload.get("data") or [])]


    def complete(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": TRANSLATE_TEMPERATURE,
            "max_tokens": TRANSLATE_MAX_TOKENS,
            "stream": False,
            # Qwen3 reasons before answering unless the chat template is told
            # not to. A server whose template ignores this is caught by the
            # <think> stripping instead.
            "chat_template_kwargs": {"enable_thinking": TRANSLATE_ENABLE_THINKING},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise TranslationError(f"translation request failed: {exc}") from exc

        choices = payload.get("choices") or []
        if not choices:
            raise TranslationError("the server returned no choices")
        message = choices[0].get("message", {})
        # vLLM splits reasoning out of `content` for some models. If it did,
        # `content` is already the answer alone; if it did not, the <think>
        # block is still in there and clean() takes it out.
        text = str(message.get("content") or "")
        if not text and message.get("reasoning_content"):
            raise TranslationError(
                "the model returned only reasoning and no answer; "
                "chat_template_kwargs.enable_thinking was not honoured"
            )
        if choices[0].get("finish_reason") == "length" and not text.strip():
            raise TranslationError(
                f"the model used its whole {TRANSLATE_MAX_TOKENS}-token budget "
                "without answering"
            )
        return text

    @property
    def source(self) -> str:
        return f"{self.model} at {self.base_url}"
