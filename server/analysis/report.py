"""Re-read a drift run without decoding it again.

``server/tests_real/test_real_partial_final.py`` writes every sentence, every
variant and every guard verdict to JSON. The summary it prints on the way
past is the first cut; this is the second, and it exists because the first
one was misleading in two specific ways.

**Empty sentences poison every other column.** A sentence the guards emptied
scores ``rewrite`` 1.00 and loses every Latin word the running text had, so a
run where a fifth of the sentences come back empty reports a Latin-word
problem and a rewriting problem that are both really the same problem. Every
comparison here is offered twice: over all sentences, and over the sentences
that actually produced text.

**A fixed trim is only a hangover for some sentences.** The VAD's silence
hangover is on the end of a sentence that finished on a *pause*. One cut
short at ``max_duration`` was interrupted mid-word, and trimming it removes
speech. Splitting the trim's effect by the reason the sentence was committed
is the difference between a result and an artefact.

Usage::

    python3.11 -m server.analysis.report /tmp/drift_full.json
    python3.11 -m server.analysis.report /tmp/drift_full.json --examples 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.analysis import guards  # noqa: E402

FLAGS = ("rewrite", "tail_invention", "latin_lost", "repetition", "final_empty")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def variant_names(run: dict) -> list:
    return [variant["name"] for variant in run["variants"]]


def comparable(run: dict, variant: str) -> list:
    """Sentences that had a running text to be compared against."""
    return [case for case in run["cases"] if variant in case.get("drift", {})]


def is_empty(case: dict, variant: str) -> bool:
    return not case["finals"].get(variant, {}).get("text", "").strip()


# ---------------------------------------------------------------------------
# Why a sentence came back empty
# ---------------------------------------------------------------------------
def dropped_histogram(run: dict, variant: str,
                      empty_only: bool = False) -> dict:
    """How often each guard refused a segment, by the reason it gave.

    ``_refuse`` returns one of: empty, no speech, low confidence, repetition,
    known hallucination. Which of them is doing the emptying decides what
    there is to fix - a threshold, a word list, or the rule itself.
    """
    counts: dict = {}
    for case in run["cases"]:
        if empty_only and not is_empty(case, variant):
            continue
        for reason in case["finals"].get(variant, {}).get("dropped", []):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def empties_by_reason(run: dict, variant: str) -> dict:
    """Empty sentences against total, split by why the sentence was committed.

    A sentence cut at ``max_duration`` is a fragment by construction, so it is
    the one most likely to be refused for honest reasons. If the empties are
    spread evenly instead, the refusal is not about fragments.
    """
    table: dict = {}
    for case in comparable(run, variant):
        empty, total = table.get(case["reason"], (0, 0))
        table[case["reason"]] = (empty + int(is_empty(case, variant)),
                                 total + 1)
    return dict(sorted(table.items(), key=lambda item: -item[1][1]))


def empties_by_duration(run: dict, variant: str,
                        edges=(1.0, 2.0, 4.0, 6.0)) -> dict:
    """The same count, bucketed by how long the sentence was."""
    labels = ([f"<{edges[0]:.0f}s"]
              + [f"{low:.0f}-{high:.0f}s"
                 for low, high in zip(edges, edges[1:])]
              + [f"{edges[-1]:.0f}s+"])
    table = {label: (0, 0) for label in labels}
    for case in comparable(run, variant):
        seconds = (case["end_ms"] - case["start_ms"]) / 1000.0
        index = sum(1 for edge in edges if seconds >= edge)
        empty, total = table[labels[index]]
        table[labels[index]] = (empty + int(is_empty(case, variant)),
                                total + 1)
    return table


# ---------------------------------------------------------------------------
# The comparison, with the empties held apart
# ---------------------------------------------------------------------------
def reslice(run: dict, variant: str, exclude_empty: bool) -> dict:
    """Flag counts and mean rewrite, optionally over non-empty sentences only."""
    cases = comparable(run, variant)
    if exclude_empty:
        cases = [case for case in cases if not is_empty(case, variant)]
    drifts = [case["drift"][variant] for case in cases]
    rewrites = [drift["rewrite"] for drift in drifts]
    return {
        "variant": variant,
        "compared": len(drifts),
        "clean": sum(1 for drift in drifts if not drift["flags"]),
        "flagged": sum(1 for drift in drifts if drift["flags"]),
        "mean_rewrite": statistics.fmean(rewrites) if rewrites else 0.0,
        "counts": {flag: sum(1 for drift in drifts if flag in drift["flags"])
                   for flag in FLAGS},
        "lost_words": sorted({word for drift in drifts
                              for word in drift["lost_latin"]}),
    }


def variant_effect(run: dict, baseline: str, variant: str) -> dict:
    """What a variant recovered and what it destroyed, by commit reason.

    One number for both directions hides the trade. A trim that recovers ten
    sentences finished on a pause and empties twelve cut at max_duration is
    not a wash - it is two findings.
    """
    table: dict = {}
    for case in comparable(run, baseline):
        if variant not in case["finals"]:
            continue
        was_empty = is_empty(case, baseline)
        now_empty = is_empty(case, variant)
        recovered, lost, same = table.get(case["reason"], (0, 0, 0))
        if was_empty and not now_empty:
            recovered += 1
        elif now_empty and not was_empty:
            lost += 1
        else:
            same += 1
        table[case["reason"]] = (recovered, lost, same)
    return dict(sorted(table.items(), key=lambda item: -sum(item[1])))


# ---------------------------------------------------------------------------
# Replaying the guards
# ---------------------------------------------------------------------------
def guard_effect(run: dict, variant: str, rule: str) -> dict:
    """What a different guard rule would have kept, and at what cost.

    Exact rather than estimated: the decoder's output is recorded, and the
    guards are a function of it. ``recovered`` is a sentence that was empty
    and is not any more - the text is carried along, because whether it is
    real speech or an invention the word lists do not know about is a
    question only a person can answer.
    """
    simulated = guards.simulate(run, variant, rule)
    recovered, lost, changed = [], [], []
    for case in run["cases"]:
        if case["index"] not in simulated:
            continue
        before = case["finals"][variant]["text"].strip()
        after = simulated[case["index"]]["text"].strip()
        entry = {
            "index": case["index"],
            "start_s": case["start_ms"] / 1000.0,
            "seconds": (case["end_ms"] - case["start_ms"]) / 1000.0,
            "reason": case["reason"],
            "partial": case["partial_text"],
            "before": before,
            "after": after,
            # Two independent readings of the same audio. The running text was
            # decoded on its own, four seconds at a time; if it said nothing
            # there, a sentence appearing out of the same audio deserves a
            # second look.
            "had_partial": bool(case["partial_text"].strip()),
            "near_miss": guards.near_miss(after) if after else None,
        }
        if not before and after:
            recovered.append(entry)
        elif before and not after:
            lost.append(entry)
        elif before != after:
            changed.append(entry)
    return {
        "rule": rule,
        "empty_before": sum(1 for case in run["cases"]
                            if case["index"] in simulated
                            and not case["finals"][variant]["text"].strip()),
        "empty_after": sum(1 for index, result in simulated.items()
                           if not result["text"].strip()),
        "recovered": recovered,
        "lost": lost,
        "changed": changed,
    }


def recovered_by_reason(effect: dict) -> dict:
    """Recovered sentences split by why they were committed."""
    table: dict = {}
    for entry in effect["recovered"]:
        table[entry["reason"]] = table.get(entry["reason"], 0) + 1
    return dict(sorted(table.items(), key=lambda item: -item[1]))


def recovered_seconds(effect: dict) -> float:
    """How much meeting the rule puts back on the screen."""
    return sum(entry["seconds"] for entry in effect["recovered"])


def no_speech_scores(run: dict, variant: str) -> dict:
    """How confident the decoder was about the segments it called silence.

    The whole argument for the Whisper rule is that these two numbers
    disagree: a segment can score badly on ``no_speech_prob`` and well on
    ``avg_logprob``, and today the first one wins alone. If the refused
    segments were also decoded badly, there is nothing here to recover and
    the rule change is not the fix.
    """
    confident, unsure = [], []
    threshold = guards.make_transcriber().log_prob_threshold
    for case in run["cases"]:
        for record in case["finals"].get(variant, {}).get("pieces", []):
            if record["verdict"] != "no speech":
                continue
            if record["avg_logprob"] > threshold:
                confident.append(record)
            else:
                unsure.append(record)
    return {
        "threshold": threshold,
        "refused": len(confident) + len(unsure),
        "confident": len(confident),
        "unsure": len(unsure),
        "median_confident_logprob": (
            statistics.median([r["avg_logprob"] for r in confident])
            if confident else 0.0),
        "median_unsure_logprob": (
            statistics.median([r["avg_logprob"] for r in unsure])
            if unsure else 0.0),
    }


def tail_inventions(run: dict, variant: str) -> list:
    """Every sentence that grew a tail with too little audio behind it."""
    found = []
    for case in comparable(run, variant):
        drift = case["drift"][variant]
        if "tail_invention" in drift["flags"]:
            found.append({
                "index": case["index"],
                "start_s": case["start_ms"] / 1000.0,
                "extra_audio_ms": case["extra_audio_ms"],
                "tail": drift["tail_text"],
                "partial": case["partial_text"],
                "final": drift["final"],
            })
    return found


def lost_latin_cases(run: dict, variant: str) -> list:
    """Latin words the running text had and the sentence dropped.

    Only over sentences that produced text: an empty sentence loses every
    word there was, and counting those says nothing about code-switching.
    """
    found = []
    for case in comparable(run, variant):
        if is_empty(case, variant):
            continue
        drift = case["drift"][variant]
        if drift["lost_latin"]:
            found.append({
                "index": case["index"],
                "start_s": case["start_ms"] / 1000.0,
                "words": list(drift["lost_latin"]),
                "partial": case["partial_text"],
                "final": drift["final"],
            })
    return found


def without_running_text(run: dict) -> dict:
    """Sentences never compared, and whether they were simply too short.

    A sentence shorter than a partial interval never gets a running text, and
    those are uninteresting. One that had several running texts and still
    could not be compared means every one of them was refused, which is the
    same fault at the other end of the pipeline.
    """
    compared = {case["index"] for case in run["cases"] if case.get("drift")}
    missing = [case for case in run["cases"] if case["index"] not in compared]
    return {
        "total": len(missing),
        "no_partial_at_all": sum(1 for case in missing
                                 if case["partial_count"] == 0),
        "partials_all_refused": sum(1 for case in missing
                                    if case["partial_count"] > 0),
        "median_seconds": (statistics.median(
            [(case["end_ms"] - case["start_ms"]) / 1000.0
             for case in missing]) if missing else 0.0),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_table(rows: list, title: str) -> None:
    print(f"\n{title}")
    header = (f"  {'variant':<12} {'cmp':>5} {'clean':>6} {'flag':>5} "
              f"{'rewrite':>8} " + " ".join(f"{flag[:9]:>10}" for flag in FLAGS))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        print(f"  {row['variant']:<12} {row['compared']:>5} {row['clean']:>6} "
              f"{row['flagged']:>5} {row['mean_rewrite']:>8.3f} "
              + " ".join(f"{row['counts'][flag]:>10}" for flag in FLAGS))


def print_guard_rules(run: dict, variant: str, examples: int) -> None:
    """The guard rules, replayed over the segments this run already recorded."""
    if not guards.has_scores(run, variant):
        print("\nGuard rules cannot be replayed: this run recorded no segment "
              "scores. Re-run test_real_partial_final.py - with "
              "--reuse-partials it costs about a minute.")
        return

    scores = no_speech_scores(run, variant)
    print(f"\nSegments refused as 'no speech' under {variant}: "
          f"{scores['refused']}")
    print(f"  {scores['confident']:>4} were decoded confidently anyway "
          f"(avg_logprob above {scores['threshold']}, median "
          f"{scores['median_confident_logprob']:.2f}) - these are the ones "
          f"Whisper's own rule would keep")
    print(f"  {scores['unsure']:>4} were also decoded badly (median "
          f"{scores['median_unsure_logprob']:.2f}) - the low-confidence guard "
          f"refuses these either way")

    effect = guard_effect(run, variant, "current")
    print(f"\nToday's policy against what this run recorded, on the same "
          f"decodes:")
    print(f"  empty sentences {effect['empty_before']} -> "
          f"{effect['empty_after']}")
    print(f"  {len(effect['recovered'])} recovered "
          f"({recovered_seconds(effect):.0f} s of meeting), "
          f"{len(effect['lost'])} lost, {len(effect['changed'])} reworded")
    by_reason = recovered_by_reason(effect)
    if by_reason:
        print("  recovered by commit reason: "
              + ", ".join(f"{reason} {count}"
                          for reason, count in by_reason.items()))

    backed = [entry for entry in effect["recovered"] if entry["had_partial"]]
    alone = [entry for entry in effect["recovered"] if not entry["had_partial"]]
    suspect = [entry for entry in effect["recovered"] if entry["near_miss"]]
    print(f"\n  {len(backed)} of them had a running text saying something over "
          f"the same audio; {len(alone)} appeared where the running text said "
          f"nothing")
    print(f"  {len(suspect)} resemble a line already on the block list, with "
          f"words changed")

    print(f"\nRecovered where the running text agreed something was said "
          f"({len(backed)}):")
    for entry in backed:
        print(f"  #{entry['index']} at {entry['start_s']:.1f}s "
              f"({entry['seconds']:.1f} s, {entry['reason']})")
        print(f"      partial    {entry['partial']!r}")
        print(f"      would show {entry['after']!r}")
        if entry["near_miss"]:
            print(f"      RESEMBLES  {entry['near_miss']['listed']!r} "
                  f"(distance {entry['near_miss']['distance']:.2f})")

    print(f"\nRecovered where the running text said nothing - read these "
          f"first ({len(alone)}):")
    for entry in alone:
        print(f"  #{entry['index']} at {entry['start_s']:.1f}s "
              f"({entry['seconds']:.1f} s, {entry['reason']})")
        print(f"      would show {entry['after']!r}")
        if entry["near_miss"]:
            print(f"      RESEMBLES  {entry['near_miss']['listed']!r} "
                  f"(distance {entry['near_miss']['distance']:.2f})")

    for entry in effect["lost"][:examples]:
        print(f"  LOST #{entry['index']} at {entry['start_s']:.1f}s "
              f"was {entry['before']!r}")


def report(run: dict, examples: int = 6) -> None:
    names = variant_names(run)
    baseline = names[0]

    print(f"{Path(run['wav']).name}: {run['audio_seconds'] / 60:.1f} minutes, "
          f"{len(run['cases'])} sentences")

    missing = without_running_text(run)
    print(f"\n{missing['total']} sentences had no running text to compare "
          f"against (median {missing['median_seconds']:.1f} s):")
    print(f"  {missing['no_partial_at_all']:>4} never produced one - shorter "
          f"than the partial interval")
    print(f"  {missing['partials_all_refused']:>4} produced one and every one "
          f"was refused")

    print_table([reslice(run, name, exclude_empty=False) for name in names],
                "All comparable sentences:")
    print_table([reslice(run, name, exclude_empty=True) for name in names],
                "Sentences that produced text - the empties held apart:")

    print("\nWhy the sentences that came back empty came back empty:")
    for name in names:
        counts = dropped_histogram(run, name, empty_only=True)
        empties = sum(1 for case in comparable(run, name)
                      if is_empty(case, name))
        total = ", ".join(f"{reason}={count}"
                          for reason, count in counts.items()) or "nothing"
        print(f"  {name:<12} {empties:>4} empty; guards refused: {total}")

    print("\nEmpty sentences by why the sentence was committed:")
    for name in names:
        table = empties_by_reason(run, name)
        parts = ", ".join(f"{reason} {empty}/{total}"
                          for reason, (empty, total) in table.items())
        print(f"  {name:<12} {parts}")

    print(f"\nEmpty sentences by length ({baseline}):")
    for label, (empty, total) in empties_by_duration(run, baseline).items():
        share = 100.0 * empty / total if total else 0.0
        print(f"  {label:<8} {empty:>4} / {total:<4} ({share:.0f}%)")

    for name in names[1:]:
        print(f"\nWhat {name} changed against {baseline}, by commit reason "
              f"(recovered / emptied / unchanged):")
        for reason, (recovered, lost, same) in variant_effect(
                run, baseline, name).items():
            print(f"  {reason:<15} +{recovered:<4} -{lost:<4} ={same}")

    print_guard_rules(run, baseline, examples)

    print(f"\nInvented tails under {baseline}: "
          f"{len(tail_inventions(run, baseline))}")
    for case in tail_inventions(run, baseline)[:examples]:
        print(f"  #{case['index']} at {case['start_s']:.1f}s, only "
              f"{case['extra_audio_ms']:.0f} ms of new audio")
        print(f"      partial {case['partial']!r}")
        print(f"      final   {case['final']!r}")
        print(f"      tail    {case['tail']!r}")

    for name in names:
        cases = lost_latin_cases(run, name)
        words = sorted({word for case in cases for word in case["words"]})
        print(f"\nLatin words lost by {name}, over sentences that produced "
              f"text: {len(cases)} sentences, {len(words)} distinct words")
        print(f"  {', '.join(words) if words else '(none)'}")

    print(f"\nExamples under {baseline}:")
    for case in lost_latin_cases(run, baseline)[:examples]:
        print(f"  #{case['index']} at {case['start_s']:.1f}s "
              f"lost {', '.join(case['words'])}")
        print(f"      partial {case['partial']!r}")
        print(f"      final   {case['final']!r}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-read a drift run without decoding it again.")
    parser.add_argument("json", type=Path,
                        help="the --out file from test_real_partial_final.py")
    parser.add_argument("--examples", type=int, default=6,
                        help="how many worked examples to print per section")
    args = parser.parse_args(argv)

    try:
        run = load(args.json)
    except (OSError, ValueError) as exc:
        print(f"cannot read {args.json}: {exc}")
        return 2
    if not run.get("cases"):
        print(f"{args.json} holds no sentences")
        return 1
    report(run, args.examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
