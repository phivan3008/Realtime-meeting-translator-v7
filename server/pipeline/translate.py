"""Translation - step 8 of the server pipeline, and the last one.

``DESIGN.md`` section 3.8: Gemma behind vLLM, text to text, given the previous
two or three sentences of the meeting plus the language the LID decided and
the sentence Whisper committed.

Gemma has no system role
------------------------
Gemma's chat template knows two roles, user and model, and vLLM rejects a
request carrying a ``system`` message. So there is no system prompt: the
whole instruction is the first line of the one user message, and the
sentence to translate is its last line.

What the history is for, and what it is not for
-----------------------------------------------
Meetings are full of sentences that mean nothing alone.  "That one." "Về
việc đó thì chưa." Pronouns, dropped subjects, agreement with something said
ten seconds ago - Japanese in particular drops subjects freely, and a
sentence-at-a-time translator has to guess at every one of them.

So the previous few turns go into the prompt.  Three, not thirty: enough to
resolve a pronoun, short enough that the model cannot start summarising the
meeting instead of translating the sentence in front of it.

The history a sentence is translated with is the history *as it stood when
the sentence was committed*. Translation runs on its own thread, behind a
queue, so by the time a sentence is translated the meeting has moved on - and
a history read at that moment held the sentence itself, sometimes the ones
after it, and every translated line twice. A model shown the sentence it is
asked to translate among "the previous lines" hands it back untranslated: on
a thirty-minute run, half of those echoes translated on a retry without the
history.

The model will try to talk to you
---------------------------------
Instruction-tuned models answer requests.  Asked to translate, they will
happily return "Sure! Here is the translation:" followed by the translation,
or add a note about an ambiguity, or wrap the answer in quotes.  All of that
would be shown to a meeting participant as if somebody had said it.

So the prompt asks for the translation and nothing else, and the answer is
cleaned and then checked: an answer several times longer than its source is
not a translation, it is the model explaining itself, and it is refused.

Layering
--------
``TranslationContext``
    The rolling history. Pure Python.

``Translator``
    The prompt, the cleaning and the guards. Pure Python, tested against a
    stub that answers like a chatty model.

``VllmClient``
    The HTTP call. The only part that needs a server running.
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
from typing import Optional, Protocol, Sequence

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
    TRANSLATE_SEED,
    TRANSLATE_STOP,
    TRANSLATE_TEMPERATURE,
    TRANSLATE_TIMEOUT_S,
    TRANSLATE_TOP_P,
    VLLM_MAX_MODEL_LEN,
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
#: Gemma's turn markers. The stop strings end generation on them and vLLM
#: normally drops special tokens from the text, so these should never arrive;
#: if a server is configured otherwise, they are not part of the translation.
_CONTROL_TOKENS = re.compile(r"<(?:start_of_turn|end_of_turn|eos|bos)>")

#: Quote pairs a model reaches for when asked for exactly one line.
_QUOTES = (('"', '"'), ("'", "'"), ("「", "」"), ("“", "”"), ("『", "』"))


def clean(answer: str) -> str:
    """Strip what the model added around the translation.

    Applied repeatedly, because the additions stack: "Sure! Here is the
    translation: 「...」" is three of them in front of one sentence.
    """
    text = _CONTROL_TOKENS.sub("", answer)
    text = _THINK.sub("", text).strip()
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
    """Fraction of the letters that are kana or kanji.

    ``None`` when there are no letters to judge - a bare number translates to
    itself, and refusing that would be refusing a correct answer.

    Vietnamese and Japanese do not share a script, which makes this a cheap and
    near-certain test of whether a translation came out in the language it was
    asked for. It is not a language detector: it cannot tell Vietnamese from
    English, so ``プレー`` coming back as ``Play`` still passes.
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

    The end-to-end run produced ``はい、今の画面の`` -> ``はい、現在の画面の``:
    a Japanese sentence answered in Japanese. It passed every guard, because
    the two strings differ and neither is longer than the other, and the check
    that should have caught it did not exist.
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

    def snapshot(self) -> tuple[Turn, ...]:
        """The turns as they stand now, for a translation that runs later."""
        return tuple(self._turns)

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

        The history sits between the system prompt and the sentence to
        translate, which makes it the last thing the model reads before
        answering - and written with its translations it reads as worked
        examples. When several turns in a row go the same way, every example
        ends in the same language and the model follows the examples over the
        instruction. Measured against the live model, same history, same
        moment, on ここに作っているの? going into Vietnamese::

            plain     ->  ここで作っているの？
            labelled  ->  Đang tạo ở đây à?

        Note what the plain answer is. Not the sentence handed back: に became
        で and ? became ？. The model translated it - into Japanese, because
        two Vietnamese turns of history had put two Japanese translations in
        front of it. That is also why ``looks_like_echo`` is blind here and
        only ``wrong_script`` caught it.

        Labelling was not enough on its own: it took the fourth end-to-end run
        from 6 of 10 sentences translated to 14 of 17, and the two that still
        came back in Japanese had three such turns behind them rather than
        two. ``sources`` removes the examples instead of annotating them. The
        history is there to say what "that one" refers to, and the source
        lines carry that on their own.
        """
        return self.render(self._turns, style)

    @classmethod
    def render(cls, turns: Sequence[Turn], style: str = HISTORY_STYLE) -> str:
        """Any sequence of turns, laid out as :meth:`as_prompt` does."""
        if style not in cls.STYLES:
            raise ValueError(f"unknown history style {style!r}, "
                             f"expected one of {cls.STYLES}")
        if not turns:
            return ""
        lines = []
        for turn in turns:
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
    #: Sentences the model handed straight back, and how many a second try
    #: without the history rescued. The ratio says whether the history is
    #: what makes it echo.
    retried: int = 0
    rescued: int = 0

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


#: The one refusal worth a second attempt: see :meth:`Translator.translate`.
ECHOED = "the model returned the sentence untranslated"


class Backend(Protocol):
    """What :class:`Translator` needs; a stub satisfies it in the tests."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        ...                                         # pragma: no cover


#: The instruction, as the first line of the user message. With no hint and
#: no history the whole message is exactly::
#:
#:     Bạn là một trợ lý phiên dịch trực tiếp. Hãy dịch câu tiếng Nhật sau
#:     sang tiếng Việt một cách tự nhiên và chính xác nhất. Chỉ trả về kết quả
#:     dịch, không giải thích thêm:
#:
#:     <sentence>
INSTRUCTION = (
    "Bạn là một trợ lý phiên dịch trực tiếp. Hãy dịch câu tiếng {source_name} "
    "sau sang tiếng {target_name} một cách tự nhiên và chính xác nhất.{hint} "
    "Chỉ trả về kết quả dịch, không giải thích thêm:"
)

#: Inserted into the instruction. A meeting is mostly short lines, and the
#: model handed はい straight back - it is a whole turn of a Japanese meeting
#: and one of the commonest lines there is. Other short lines translated fine
#: on the same run (えっ -> Eh?, いや違います -> Không, tôi nhầm rồi), so this
#: is about single words rather than length as such.
SHORT_LINE_HINT = (
    " Câu chỉ có một từ, một thán từ hay một từ đệm vẫn là một câu: cũng phải "
    "dịch sang tiếng {target_name}, không được giữ nguyên."
)

#: Introduces the history. Worded so the model reads it and leaves it alone.
HISTORY_HEADER = (
    "Các câu trước đó trong cuộc họp, chỉ để tham khảo ngữ cảnh, "
    "không dịch các câu này:"
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
                     short_line_hint: Optional[bool] = None,
                     history: Optional[Sequence[Turn]] = None) -> str:
        """The one user message, as text, so a test can read it.

        ``history_style`` overrides this translator's own, so
        ``server/tests_real/test_real_translate.py`` can put every version to
        a real model on the same history and read the answers side by side.
        ``"plain"`` leaves out the instruction repeated after the history,
        which is how the diagnosis was confirmed rather than assumed.

        ``history`` is the turns to show; None means this translator's own.
        """
        style = history_style or self.history_style
        hint = self.short_line_hint if short_line_hint is None else short_line_hint
        target = target_language(lang_code)
        target_name = LANGUAGE_NAMES.get(target, target)
        instruction = INSTRUCTION.format(
            source_name=LANGUAGE_NAMES.get(lang_code, lang_code),
            target_name=target_name,
            hint=SHORT_LINE_HINT.format(target_name=target_name) if hint else "",
        )
        history = (self.context.as_prompt(style=style) if history is None
                   else TranslationContext.render(history, style))
        if not history:
            return f"{instruction}\n\n{source}"
        if style == "plain":
            return (f"{instruction}\n\n{HISTORY_HEADER}\n{history}\n\n"
                    f"Câu cần dịch:\n{source}")
        # The direction is repeated after the history as well as in the
        # instruction. The history is the last text the model reads before
        # answering, and a run of lines all going the same way outweighed an
        # instruction asking for the other one.
        return (
            f"{instruction}\n\n{HISTORY_HEADER}\n{history}\n\n"
            f"Câu cần dịch sang tiếng {target_name}, và chỉ sang tiếng "
            f"{target_name}:\n{source}"
        )

    def build_messages(self, source: str, lang_code: str,
                       history_style: str = "",
                       short_line_hint: Optional[bool] = None,
                       history: Optional[Sequence[Turn]] = None,
                       ) -> list[dict[str, str]]:
        """The chat messages sent to vLLM: one user turn, never a system one."""
        return [{"role": "user",
                 "content": self.build_prompt(source, lang_code, history_style,
                                              short_line_hint, history)}]

    def translate(self, source: str, lang_code: str,
                  speaker_id: str = "",
                  history: Optional[Sequence[Turn]] = None) -> Translation:
        """Translate one sentence.

        ``history`` is the meeting as it stood when the sentence was
        committed, and whoever passes it owns the history: nothing is
        remembered here. Without it this translator keeps its own, which is
        what a caller translating sentences one after another wants.
        """
        target = target_language(lang_code)
        if not source.strip():
            return self._refuse("", lang_code, target, "nothing to translate")
        if not target:
            # Without a language there is no direction to translate in, and
            # guessing one is how a Vietnamese sentence comes back as
            # Vietnamese-flavoured Japanese.
            return self._refuse(source, lang_code, target,
                                "the language was undecided")

        owns_history = history is None
        turns = self.context.snapshot() if owns_history else tuple(history)

        try:
            answer = self._ask(source, lang_code, turns)
        except TranslationError as exc:
            result = Translation("", source, lang_code, target, str(exc))
            self.stats.record(result, failed=True)
            log.warning("Translation failed: %s", exc)
            return result

        text = clean(answer)
        reason = self._refuse_reason(source, text, target)
        if reason == ECHOED and TranslationContext.render(
                turns, self.history_style):
            # Handed back untranslated. Three source lines followed by a
            # request to translate a fourth can read as a list to continue,
            # so ask again without them - a fix and a measurement at once.
            self.stats.retried += 1
            try:
                retry = self._ask(source, lang_code, ())
            except TranslationError as exc:
                log.warning("Retry without the history failed: %s", exc)
            else:
                second = clean(retry)
                if not self._refuse_reason(source, second, target):
                    self.stats.rescued += 1
                    log.info("Retried without the history and it "
                             "translated: %r", source[:60])
                    answer, text, reason = retry, second, ""
        if reason:
            return self._refuse(source, lang_code, target, reason, answer)

        result = Translation(text, source, lang_code, target, raw=answer)
        self.stats.record(result)
        if owns_history:
            self.context.remember(
                Turn(speaker_id=speaker_id, lang_code=lang_code, source=source,
                     translation=text)
            )
        return result

    def _ask(self, source: str, lang_code: str,
             history: Sequence[Turn]) -> str:
        """One round trip. Raises :class:`TranslationError`."""
        return self.backend.complete(
            self.build_messages(source, lang_code, history=history))

    def _refuse_reason(self, source: str, text: str,
                       target: str = "") -> str:
        if not text:
            return "the model returned nothing"
        if looks_like_echo(source, text):
            # Handed back untranslated. Showing it would put the same sentence
            # in both columns and read as though the translation succeeded.
            return ECHOED
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

        Per direction, because Japanese carries the same meaning in far fewer
        characters than Vietnamese: measured over real pairs, ja -> vi ran up
        to 4.44x while vi -> ja never passed 0.70x. The slack keeps short
        sources out of it - at nine characters a ratio is mostly noise, and a
        shared limit refused a correct translation of あれこれ今下の方に.
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
        self._cards: dict[str, dict] = {}
        served = self.served_models()
        self.model = choose_model(wanted, served, self.base_url)
        #: The context length the server was started with, if it says.
        self.max_model_len: Optional[int] = self._cards.get(
            self.model, {}).get("max_model_len")
        if self.max_model_len is not None and self.max_model_len != VLLM_MAX_MODEL_LEN:
            log.warning("vLLM is serving %s with max_model_len=%s, not the "
                        "configured %d; it was not started by "
                        "server/launch_vllm.py", self.model,
                        self.max_model_len, VLLM_MAX_MODEL_LEN)
        log.info("Translation backend ready: %s at %s (max_model_len=%s)",
                 self.model, self.base_url, self.max_model_len)

    def served_models(self) -> list[str]:
        """What this server is actually serving."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/models",
                                        timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise TranslationError(
                f"No translation server at {self.base_url}: {exc}. Start vLLM "
                "with `python3.11 server/launch_vllm.py`."
            ) from exc
        cards = [entry for entry in (payload.get("data") or [])
                 if isinstance(entry, dict)]
        self._cards = {str(card.get("id", "")): card for card in cards}
        return [str(card.get("id", "")) for card in cards]

    @staticmethod
    def request_body(model: str, messages: list[dict[str, str]]) -> dict:
        """The chat completion request, separate so a test can read it."""
        roles = {message.get("role") for message in messages}
        if not messages or not roles <= {"user", "assistant"}:
            # Gemma's chat template raises on a system turn, and vLLM answers
            # that with a 400 for every sentence of the meeting.
            raise TranslationError(
                f"Gemma takes user and assistant turns only, got {sorted(roles)}")
        return {
            "model": model,
            "messages": messages,
            "temperature": TRANSLATE_TEMPERATURE,
            "top_p": TRANSLATE_TOP_P,
            "seed": TRANSLATE_SEED,
            "stop": list(TRANSLATE_STOP),
            "max_tokens": TRANSLATE_MAX_TOKENS,
            "stream": False,
            # Qwen3 reasons before answering unless the chat template is told
            # not to. A template without the switch ignores it, and a server
            # that ignores it is caught by the <think> stripping instead.
            "chat_template_kwargs": {"enable_thinking": TRANSLATE_ENABLE_THINKING},
        }

    def complete(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps(self.request_body(self.model, messages)).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # vLLM puts the reason in the body - a rejected role, a prompt past
            # max_model_len - and the status line alone does not say which.
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise TranslationError(
                f"translation request failed: {exc}: {detail}") from exc
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
