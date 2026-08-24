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

from server.config import (  # noqa: E402
    HISTORY_STYLE,
    SHORT_LINE_HINT_ENABLED,
    TRANSLATE_HISTORY,
)
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
CONTEXT_SENTENCE = ("ja", "\u7d42\u308f\u308a\u307e\u3057\u305f\u3002")

# Sentences the pipeline actually committed and the translator then refused,
# on the third end-to-end run. Six of ten sentences came back translated; the
# other four are here, so the cause can be found without a microphone.
#
# Two kinds are mixed on purpose, because the point is to tell them apart:
#
#   * nothing to translate - the ASR misheard the sentence outright, or the
#     max-duration cut left half of one. A model handing those back is doing
#     the only sensible thing, and refusing them is correct.
#   * a complete sentence refused anyway - ここに作っているの? is an ordinary
#     Japanese question, and there is no honest reason for it to come back
#     in Japanese.
#
# Whatever the model actually said is printed under each one. That is the
# whole reason this list exists.
MEETING_REFUSALS = [
    ("ja", "\u3053\u3053\u306b\u4f5c\u3063\u3066\u3044\u308b\u306e?",
     "a complete question; refused as 'not written in vi'"),
    ("ja", "\u3042\u308c\u3053\u308c\u4eca\u4e0b\u306e\u65b9\u306b",
     "cut mid-sentence by the 7 s limit"),
    ("ja", "\u3042\u3089\u305f\u3063",
     "four characters, no meaning"),
    ("vi", "M\u00ecnh \u0111\u1ea9u g\u00f3i c\u1ee7a b\u00e1c t\u1edbi, "
           "m\u00ecnh \u0111\u00e1nh l\u1ea1i s\u1ed1 th\u00ec n\u00f3 "
           "nh\u1ea3y b\u1ecdn gi\u1ea3 trong n\u00e0y.",
     "the ASR misheard the whole sentence"),
]

# Labelling the history fixed that sentence and took the fourth end-to-end run
# from 6 of 10 sentences translated to 14 of 17. Two Japanese sentences were
# still refused as "not written in vi", and both had three Vietnamese turns
# behind them rather than two - every translated line in the history Japanese,
# three deep instead of two.
#
# But both were also cut mid-sentence by the 7 s limit, so there are two
# candidate causes and the run cannot separate them. This does:
#
#   no history        - if it translates here, the sentence is fine and the
#                       history is the problem
#   plain history     - the form before labelling
#   labelled history  - the first attempt at a fix
#   sources only      - the history with its translations removed, so there
#                       are no worked examples left to imitate
#
# If "no history" fails too, the sentence being cut is the cause, the history
# is not, and no amount of prompt work will help.
#
# It answered. Against the live model, on both Japanese sentences:
#
#     none      translated      translated
#     plain     REFUSED         REFUSED
#     labelled  REFUSED         REFUSED
#     sources   translated      translated
#
# So the cut was not the cause, labelling was not enough at three turns deep,
# and HISTORY_STYLE is now "sources". These stay as the regression test for
# that, and the check judges whichever style the pipeline is set to.
CUT_HISTORY = [
    Turn("Speaker_02", "vi", "Cái tab này đều viết theo cái template có được.",
         "このタブはすべて、取得したテンプレートに従って記述されています。"),
    Turn("Speaker_02", "vi", "Cái đó thì mình chưa xem, sẽ mình xem.",
         "その件は確認していませんが、確認します。"),
    Turn("Speaker_03", "vi", "Cảm ơn.", "ありがとうございます。"),
]
CUT_SENTENCES = [
    ("ja", "はい、画面を視聴しながら、そうなんですけど、今、薬師さんが作ったので、"),
    ("ja", "FT1のほうは意識していますが直近、FC3からまた全面更新する"),
    # Cut the same way by the same limit, but going the other way, and this
    # one the run did translate. If the cut were the cause, it would fail too.
    ("vi", "các FCG có tặng một cái thêm kết cho cái xong thì bác muốn tất cả các"),
]


# Short lines the pipeline committed, with what happened to each on the run
# they came from. A meeting is mostly these.
#
# はい is the one that failed - handed straight back, and it is a whole turn
# of a Japanese meeting and one of the commonest lines in one. The others
# translated on the same run, which is what says this is about single words
# rather than short lines in general.
SHORT_LINES = [
    ("ja", "はい", "handed back untranslated"),
    ("ja", "えっ", "translated: Eh?"),
    ("ja", "いや違います", "translated: Không, tôi nhầm rồi"),
    ("ja", "それじゃないの", "translated: Không phải vậy đâu"),
    ("vi", "Cảm ơn.", "translated: ありがとうございます。"),
    ("vi", "review kết quả hôm nay", "translated: 今日の結果をレビューする"),
]


def check_short_lines(client, report: Report) -> None:
    """Does a one-word turn survive?

    Both prompts, so the hint has to earn its place: if the plain prompt
    translates everything too, the hint is not doing anything and should go.
    """
    print("\n  Short lines, with and without the one-word hint:")
    refused = {True: [], False: []}
    for lang, source, note in SHORT_LINES:
        target = "vi" if lang == "ja" else "ja"
        print(f"\n    [{lang} -> {target}] {source}   ({note})")
        for hint in (False, True):
            translator = Translator(backend=client, short_line_hint=hint)
            result = translator.translate(source, lang)
            label = "with hint" if hint else "plain    "
            if result.ok:
                print(f"      {label}: {result.text}")
            else:
                refused[hint].append(source)
                print(f"      {label}: REFUSED ({result.reason}) "
                      f"raw={result.raw[:60]!r}")

    in_use = SHORT_LINE_HINT_ENABLED
    report.add("Every short line comes back translated",
               not refused[in_use],
               f"{len(refused[in_use])} refused: {refused[in_use][:3]}")

    print(f"\n    plain refused {len(refused[False])}, "
          f"with the hint {len(refused[True])}")
    if not refused[False]:
        print("    NOTE: the plain prompt translated them all. The hint is "
              "then not doing anything on this run and should be removed "
              "rather than kept on faith.")


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
#: The four ways the same sentence can be put to the model. "none" is the
#: control: it is the only one that carries no history at all.
HISTORY_VARIANTS = ("none", "plain", "labelled", "sources")


def ask(client, lang: str, source: str, variant: str):
    """One sentence, one way of writing the history."""
    context = TranslationContext(size=TRANSLATE_HISTORY)
    if variant != "none":
        for turn in CUT_HISTORY:
            context.remember(turn)
    style = "labelled" if variant == "none" else variant
    return Translator(backend=client, context=context,
                      history_style=style).translate(source, lang)


def check_what_the_history_does(client, report: Report) -> None:
    """Separate two explanations for the same failure.

    The fourth end-to-end run refused two Japanese sentences as "not written
    in vi". Both had three Vietnamese turns behind them - every translation in
    the history Japanese - and both had also been cut mid-sentence by the 7 s
    limit. Either could be the cause.

    Only the "none" column can tell them apart. If a sentence translates with
    no history and fails with it, the history is the cause. If it fails with
    no history either, the sentence is simply too broken to translate and no
    prompt will change that.

    The third sentence is the control in the other direction: cut by the same
    limit, but going vi -> ja, and the run translated it.
    """
    print("\n  What is the history doing?")
    print(f"    history: {len(CUT_HISTORY)} Vietnamese turns, so every "
          "translation in it is Japanese")

    answers: dict[str, dict[str, object]] = {}
    for lang, source in CUT_SENTENCES:
        target = "vi" if lang == "ja" else "ja"
        print(f"\n    [{lang} -> {target}] {source}")
        answers[source] = {}
        for variant in HISTORY_VARIANTS:
            result = ask(client, lang, source, variant)
            answers[source][variant] = result
            if result.ok:
                print(f"      {variant:<9}: {result.text}")
            else:
                print(f"      {variant:<9}: REFUSED ({result.reason}) "
                      f"raw={result.raw[:80]!r}")

    japanese = [source for lang, source in CUT_SENTENCES if lang == "ja"]
    broken = [source for source in japanese if not answers[source]["none"].ok]
    if broken:
        print("\n    NOTE: a sentence failed with no history at all. For that "
              "one the cut is the cause, not the history, and no prompt "
              "change will translate it.")

    # Only sentences that a bare model can translate are the history's fault.
    steerable = [source for source in japanese if source not in broken]

    def refused_by(name: str) -> list[str]:
        return [source for source in steerable if not answers[source][name].ok]

    # Only the style the pipeline actually uses is judged. The others are
    # measuring devices: "plain" is the original and "labelled" the first
    # attempt at fixing it, and both are printed above so a reader can see
    # whether the style in use is still earning its place.
    failed = refused_by(HISTORY_STYLE)
    report.add(f"The {HISTORY_STYLE!r} history does not steer the language",
               not failed, f"{len(failed)} of {len(steerable)} refused")

    others = {name: len(refused_by(name)) for name in TranslationContext.STYLES
              if name != HISTORY_STYLE}
    print(f"\n    For comparison, of {len(steerable)} sentence(s) the model "
          f"can translate: {others}")
    if steerable and not any(others.values()):
        print("    NOTE: every other style translated them too. The history "
              "is then not steering anything here, and HISTORY_STYLE is "
              "solving a problem this run does not show.")


def check_meeting_refusals(client, report: Report) -> None:
    """Replay the sentences a real meeting lost, and show what the model said.

    No check here can decide whether refusing a mangled sentence was right -
    that needs a person. What it can do is separate the mangled ones from the
    complete one, and put the model's own words next to each.
    """
    print("\n  Sentences a real meeting lost - read the raw answers:")
    translator = Translator(backend=client)
    complete_and_refused = []
    for lang, source, note in MEETING_REFUSALS:
        result = translator.translate(source, lang)
        print(f"\n    [{lang} -> {result.target}] {note}")
        print(f"      in : {source}")
        if result.ok:
            print(f"      out: {result.text}")
        else:
            print(f"      refused: {result.reason}")
            print(f"      raw    : {result.raw[:300]!r}")
            if note.startswith("a complete"):
                complete_and_refused.append((source, result))

    report.add(
        "A complete sentence is not refused",
        not complete_and_refused,
        f"{[(s[:20], r.reason) for s, r in complete_and_refused]}",
    )
    print("    The mangled ones are expected to fail; a model cannot "
          "translate a sentence the ASR did not hear. The complete one has "
          "no such excuse.")


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
        check_what_the_history_does(client, report)
        check_short_lines(client, report)
        check_meeting_refusals(client, report)

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
