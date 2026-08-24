"""REAL TEST - translation through a running vLLM server.

MUST RUN ON: the GPU Server pod, with vLLM already serving.
DO NOT RUN ON: the Dev PC agent loop.

``DESIGN.md`` section 3.8. As with the ASR, a machine cannot tell you whether
a translation is *good* - it prints them for you to read. What it can measure
is everything around that:

* the server is reachable and says which model it is serving;
* the answer is a translation and not a conversation about one;
* the same sentence twice gives the same answer, because temperature is zero;
* the history reaches the model, by giving it a sentence that cannot be
  translated correctly without one;
* what a translation costs, which decides whether it can stay on the audio
  path or has to move off it.

Starting the server
-------------------
    python3.11 -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3.5-9B \\
        --port 8001 --gpu-memory-utilization 0.55

Leave room for Whisper: the audio pipeline needs a few GB of the same card.

The checkpoint has to match ``TRANSLATE_MODEL``. The client refuses a server
running anything else, because vLLM would answer a wrong-model request
perfectly happily and the only symptom would be worse translations.

Usage
-----
    python3.11 server/tests_real/test_real_translate.py
    python3.11 server/tests_real/test_real_translate.py --url http://127.0.0.1:8001/v1
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import TRANSLATE_HISTORY  # noqa: E402
from server.pipeline.translate import (  # noqa: E402
    TranslationContext,
    TranslationError,
    Translator,
    Turn,
    VllmClient,
    looks_like_echo,
)

# A translation is the last thing between a sentence being committed and it
# appearing on screen, and it currently runs on the same path that reads audio.
MAX_SECONDS = 3.0
# Japanese runs longer than Vietnamese and the reverse is shorter, but neither
# doubles. Well past that means the model added something of its own.
MAX_EXPANSION = 3.0


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, passed, detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
              + (f" - {detail}" if detail else ""))
        return passed

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


@dataclass
class Attempt:
    source: str
    lang_code: str
    result: object
    seconds: float = 0.0

    @property
    def expansion(self) -> float:
        if not self.source:
            return 0.0
        return len(self.result.text) / len(self.source)


# Sentences a meeting actually produces: a statement, a number, a question, a
# short reply, and one that needs the previous line to make sense.
SAMPLES = [
    ("vi", "Xin chào mọi người, hôm nay chúng ta họp về tiến độ dự án."),
    ("vi", "Bản dựng thứ ba sẽ xong trước ngày 15 tháng 4."),
    ("ja", "その件については、来週までに回答します。"),
    ("ja", "はい、承知しました。"),
    ("vi", "Anh có thể gửi lại tài liệu đó cho tôi không?"),
]

# The context test needs a sentence whose *translation* changes with the
# history, not merely one that reads oddly without it. Japanese drops subjects
# and objects as freely as Vietnamese does, so "Vậy thì tôi duyệt nó" came back
# as 「では、承認します。」 either way - a perfectly good translation that proved
# nothing.
#
# A bare Japanese "終わりました" going into Vietnamese is a sharper probe: with
# the history the subject is available to name, without it there is nothing to
# name.
CONTEXT_HISTORY = [
    Turn("Speaker_01", "vi", "Bản dựng thứ ba đang chạy trên máy chủ thử nghiệm.",
         "三番目のビルドがテストサーバーで実行中です。"),
    Turn("Speaker_02", "ja", "ビルドにはどのくらい時間がかかりますか。",
         "Bản dựng mất khoảng bao lâu?"),
]
CONTEXT_SENTENCE = ("ja", "終わりました。")


def attempt(translator: Translator, lang_code: str, source: str) -> Attempt:
    started = time.perf_counter()
    result = translator.translate(source, lang_code)
    return Attempt(source=source, lang_code=lang_code, result=result,
                   seconds=time.perf_counter() - started)


def describe(attempts: list[Attempt]) -> None:
    print("\n  Translations - read these:")
    for item in attempts:
        print(f"\n    [{item.lang_code} -> {item.result.target}] "
              f"{item.seconds:.2f}s")
        print(f"      in : {item.source}")
        if item.result.ok:
            print(f"      out: {item.result.text}")
        else:
            print(f"      refused: {item.result.reason}")
            if item.result.raw:
                # Whatever the model said is the only clue to why. Hiding it
                # once already cost a round trip: 512 tokens of <think> looked
                # exactly like "the answer is far longer than the sentence".
                print(f"      raw    : {item.result.raw[:300]!r}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_answers(attempts: list[Attempt], report: Report) -> None:
    refused = [item for item in attempts if not item.result.ok]
    report.add("Every sentence came back translated", not refused,
               f"{len(refused)} refused: "
               f"{[item.result.reason for item in refused][:3]}")

    translated = [item for item in attempts if item.result.ok]
    if not translated:
        return
    # Compared without punctuation: a model that echoes often swaps the full
    # stop for the other language's, and that one character was enough to slip
    # an untranslated Vietnamese sentence past a plain equality check.
    unchanged = [item for item in translated
                 if looks_like_echo(item.source, item.result.text)]
    report.add("Nothing came back as its own input", not unchanged,
               f"{len(unchanged)} unchanged: "
               f"{[item.source[:30] for item in unchanged][:2]}")

    worst = max(translated, key=lambda item: item.expansion)
    report.add("No answer is far longer than its sentence",
               worst.expansion <= MAX_EXPANSION,
               f"worst {worst.expansion:.1f}x on {worst.source[:30]!r}")

    chatty = [item for item in translated
              if any(marker in item.result.text.lower()
                     for marker in ("translation:", "here is", "sure,", "note:"))]
    report.add("No answer is a conversation about the translation",
               not chatty,
               f"{[item.result.text[:40] for item in chatty][:2]}")


def check_latency(attempts: list[Attempt], report: Report) -> None:
    times = [item.seconds for item in attempts]
    if not times:
        return
    print(f"\n  Translation cost: mean {statistics.fmean(times):.2f}s, "
          f"worst {max(times):.2f}s")
    report.add("A translation is fast enough to stay on the audio path",
               max(times) < MAX_SECONDS,
               f"worst {max(times):.2f}s < {MAX_SECONDS}s")
    if max(times) >= 1.0:
        print("    Note: audio arrives in 200 ms chunks and this currently "
              "runs inline, so a translation this slow stalls the socket for "
              "several chunks. Worth moving off the path.")


def check_repeatable(translator_factory, report: Report) -> None:
    """Temperature is zero, so the same sentence must give the same answer."""
    lang, source = SAMPLES[0]
    first = translator_factory().translate(source, lang)
    second = translator_factory().translate(source, lang)
    # Both empty is not agreement, it is the same failure twice.
    report.add("The same sentence twice gives the same translation",
               first.ok and second.ok and first.text == second.text,
               f"{first.text[:30]!r} vs {second.text[:30]!r}")


def check_context(client, report: Report) -> None:
    """A sentence that cannot be translated well without the previous ones."""
    lang, source = CONTEXT_SENTENCE
    context = TranslationContext(size=TRANSLATE_HISTORY)
    for turn in CONTEXT_HISTORY:
        context.remember(turn)
    with_history = Translator(backend=client, context=context)
    without = Translator(backend=client, context=TranslationContext())

    print("\n  With and without history:")
    print(f"    history: {[t.source for t in CONTEXT_HISTORY]}")
    print(f"    line   : {source}")
    first = with_history.translate(source, lang)
    second = without.translate(source, lang)
    print(f"      with   : {first.text}")
    print(f"      without: {second.text}")
    report.add("Both attempts answered", first.ok and second.ok,
               f"{first.reason or 'ok'} / {second.reason or 'ok'}")
    # Asserting only that both answered was a check that could not fail. The
    # question is whether the history changed anything.
    report.add(
        "The history changes the translation",
        first.ok and second.ok and first.text.strip() != second.text.strip(),
        f"{first.text!r} vs {second.text!r}",
    )
    print("    Read both. With the history the subject is available to name; "
          "without it there is nothing to name. Identical output means the "
          "history is not reaching the model, or is being ignored.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="", help="vLLM base URL")
    parser.add_argument("--model", default="",
                        help="defaults to whatever the server is serving")
    args = parser.parse_args()

    print("=" * 72)
    print("REAL TEST - TRANSLATION (server pipeline, step 8)")
    print("=" * 72)

    report = Report()
    try:
        print("\n  Connecting to the translation server ...")
        started = time.perf_counter()
        client = VllmClient(base_url=args.url, model=args.model)
        print(f"  Connected in {time.perf_counter() - started:.1f} s: "
              f"{client.source}")
        report.add("The translation server is reachable", True, client.source)
        report.add("It is serving a model", bool(client.model), client.model)

        translator = Translator(backend=client)
        attempts = [attempt(translator, lang, source) for lang, source in SAMPLES]
        describe(attempts)

        print("\nChecks:")
        check_answers(attempts, report)
        check_latency(attempts, report)
        check_repeatable(lambda: Translator(backend=client), report)
        check_context(client, report)

        print(f"\n  Translator stats: {translator.stats.seen} seen, "
              f"{translator.stats.translated} translated, "
              f"{translator.stats.refused} refused "
              f"{translator.stats.refused_reasons}")
        print("\n  Read the translations above. Nothing here can tell you "
              "whether they are right.")

        print("\n" + "=" * 72)
        if report.failed:
            print(f"RESULT: FAIL ({len(report.failed)} check(s) failed)")
            for check in report.failed:
                print(f"  - {check.name}: {check.detail}")
            print("=" * 72)
            return 1
        print(f"RESULT: PASS ({len(report.checks)} checks)")
        print("=" * 72)
        return 0

    except TranslationError as exc:
        print(f"\nRESULT: FAIL - {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
