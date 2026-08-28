"""Compare two translation models on the same sentences.

MUST RUN ON: anywhere. It reads two JSON files, not a GPU.

A 12B and a 9B do not share one card with Whisper, so the two models cannot be
served at once. The comparison is therefore two runs and a diff::

    # with vLLM serving Qwen
    python3.11 server/tests_real/test_real_translate.py \\
        --profile qwen --save server/tests_real/output/qwen.json

    # restart vLLM with Gemma, then
    python3.11 server/tests_real/test_real_translate.py \\
        --profile gemma --save server/tests_real/output/gemma.json

    python3.11 server/tests_real/compare_translate.py \\
        server/tests_real/output/qwen.json \\
        server/tests_real/output/gemma.json

What this can decide on its own: which model refused more, which was faster,
and whether either failed a check the other passed. Those are the three ways a
replacement can be worse without anybody noticing until a meeting.

What it cannot decide is whether the Japanese is *good*. It prints both
answers side by side for a person to read, which is the same bargain every
other real test in this project makes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seconds(run: dict) -> list[float]:
    return [answer["seconds"] for answer in run["answers"]]


def refused(run: dict) -> set[str]:
    return {answer["source"] for answer in run["answers"]
            if not answer["translation"].strip()}


def summarise(name: str, run: dict) -> None:
    times = seconds(run)
    stats = run["stats"]
    print(f"\n{name}: {run['model']}  (profile {run['profile']})")
    print(f"  {stats['translated']} translated, {stats['refused']} refused "
          f"{stats['refused_reasons'] or ''}")
    if times:
        print(f"  median {statistics.median(times):.2f} s, "
              f"slowest {max(times):.2f} s")
    if run["checks_failed"]:
        print(f"  checks failed: {', '.join(run['checks_failed'])}")


def shown(answer: Optional[dict]) -> str:
    """One answer as a line. A refusal says so rather than showing blank."""
    if answer is None:
        return "(not in this run)"
    if answer["translation"].strip():
        return answer["translation"]
    return f"(refused: {answer['reason'] or 'no reason given'})"


def compare_answers(first: dict, second: dict, names: tuple[str, str]) -> None:
    """Both answers for every sentence, for reading."""
    by_source = {answer["source"]: answer for answer in second["answers"]}
    print("\n" + "=" * 72)
    print("THE ANSWERS - nothing here can tell you which is better")
    print("=" * 72)
    for answer in first["answers"]:
        print(f"\n[{answer['lang_code']}] {answer['source']}")
        print(f"  {names[0]:<7} {shown(answer)}")
        print(f"  {names[1]:<7} {shown(by_source.get(answer['source']))}")


def verdict(first: dict, second: dict, names: tuple[str, str]) -> int:
    """The part a machine can decide. Returns a process exit code."""
    print("\n" + "=" * 72)
    print("WHAT CAN BE DECIDED WITHOUT READING")
    print("=" * 72)

    problems = []

    only_second = refused(second) - refused(first)
    only_first = refused(first) - refused(second)
    print(f"\n  Refused by {names[1]} alone: {len(only_second)}")
    for source in sorted(only_second):
        print(f"    {source}")
    print(f"  Refused by {names[0]} alone: {len(only_first)}")
    for source in sorted(only_first):
        print(f"    {source}")
    if only_second:
        problems.append(f"{names[1]} refuses {len(only_second)} sentence(s) "
                        f"that {names[0]} translates")

    first_times, second_times = seconds(first), seconds(second)
    if first_times and second_times:
        was, now = statistics.median(first_times), statistics.median(second_times)
        print(f"\n  Median latency: {names[0]} {was:.2f} s, "
              f"{names[1]} {now:.2f} s")
        # A translation runs off the audio thread, so slower is a delay before
        # the subtitle rather than a stall. Half a second again is a lot of
        # delay all the same.
        if now > was * 1.5 and now - was > 0.5:
            problems.append(f"{names[1]} is {now / was:.1f}x slower")

    new_failures = set(second["checks_failed"]) - set(first["checks_failed"])
    if new_failures:
        problems.append(f"{names[1]} fails checks {names[0]} passes: "
                        f"{', '.join(sorted(new_failures))}")

    print()
    if problems:
        print(f"RESULT: {names[1]} is WORSE on what can be measured")
        for problem in problems:
            print(f"  - {problem}")
        print("\nRead the answers above before deciding; a model that refuses "
              "more may still translate better.")
        return 1
    print(f"RESULT: {names[1]} is no worse on what can be measured")
    print("\nThat is not the same as being as good. Read the answers above.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="JSON from the model in use now")
    parser.add_argument("candidate", help="JSON from the model being tried")
    args = parser.parse_args(argv)

    first, second = load(args.baseline), load(args.candidate)
    names = (first["profile"], second["profile"])
    summarise("BASELINE ", first)
    summarise("CANDIDATE", second)
    compare_answers(first, second, names)
    return verdict(first, second, names)


if __name__ == "__main__":
    raise SystemExit(main())
