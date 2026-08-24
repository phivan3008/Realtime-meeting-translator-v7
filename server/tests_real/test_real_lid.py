"""REAL TEST - Vietnamese against Japanese on real recorded speech.

MUST RUN ON: the GPU Server pod.
DO NOT RUN ON: the Dev PC agent loop.

``DESIGN.md`` section 3.6 uses this to set Whisper's ``language`` per
sentence, so what matters is not the model's overall accuracy across 107
languages but two much narrower questions:

* does it get **these** two languages right on **your** recordings, and
* when it is wrong, is it wrong quietly or does the margin admit it?

The second question is the important one. Whisper told to transcribe
Vietnamese audio as Japanese does not fail - it returns fluent, confident,
wrong text, and the translation stage faithfully translates the nonsense. So
this reports every sentence's margin, not just the verdict, and measures
whether raising the margin would have caught the mistakes.

Usage
-----
    python3.11 server/tests_real/test_real_lid.py \\
        --vi recordings/vietnamese.wav \\
        --ja recordings/japanese.wav

Each file should be one language throughout. Both flags are repeatable.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (  # noqa: E402
    CHANNELS,
    CHUNK_BYTES,
    LID_LANGUAGES,
    LID_MIN_DURATION_MS,
    LID_MIN_MARGIN,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from server.pipeline.buffer import BufferManager, Utterance  # noqa: E402
from server.pipeline.lid import (  # noqa: E402
    LanguageIdentifier,
    LanguageIdError,
    VoxLinguaClassifier,
)
from server.pipeline.overlap import OverlapResolver, PedalboardProcessor  # noqa: E402
from server.pipeline.vad import SileroVAD, VADError, VADSegmenter  # noqa: E402

# Deciding the language must stay cheap next to transcribing the sentence.
MAX_LID_RATIO = 0.05
# How much of a recording may come back undecided before the margin is simply
# set too high to be useful.
MAX_UNKNOWN_SHARE = 0.4


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
class Sample:
    utterance: Utterance
    decision: object
    raw_margin_for_truth: float = 0.0


@dataclass
class Recording:
    path: Path
    expected: str
    samples: list[Sample] = field(default_factory=list)
    seconds: float = 0.0
    lid_seconds: float = 0.0

    @property
    def correct(self) -> list[Sample]:
        return [s for s in self.samples if s.decision.lang_code == self.expected]

    @property
    def wrong(self) -> list[Sample]:
        return [s for s in self.samples
                if s.decision.known and s.decision.lang_code != self.expected]

    @property
    def unknown(self) -> list[Sample]:
        return [s for s in self.samples if not s.decision.known]

    @property
    def lid_ratio(self) -> float:
        return self.lid_seconds / self.seconds if self.seconds else 0.0


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        actual = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
        if actual != (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH):
            raise ValueError(
                f"{path.name} is {actual[0]} Hz / {actual[1]} ch / "
                f"{actual[2] * 8}-bit; expected 16000 Hz / mono / 16-bit"
            )
        return wav.readframes(wav.getnframes())


def sentences_of(path: Path, vad: SileroVAD) -> list[Utterance]:
    pcm = read_pcm(path)
    vad.reset()
    segmenter = VADSegmenter(vad=vad)
    buffer = BufferManager()
    finals: list[Utterance] = []
    for offset in range(0, len(pcm), CHUNK_BYTES):
        finals += buffer.push(segmenter.push(pcm[offset : offset + CHUNK_BYTES])).finals
    segmenter.close()
    finals += buffer.flush().finals
    return [u for u in finals if u.duration_ms >= LID_MIN_DURATION_MS]


def measure(path: Path, expected: str, identifier: LanguageIdentifier,
            vad: SileroVAD) -> Recording:
    utterances = sentences_of(path, vad)
    recording = Recording(path=path, expected=expected,
                          seconds=sum(u.duration_ms for u in utterances) / 1000.0)
    started = time.perf_counter()
    for utterance in utterances:
        decision = identifier.identify(utterance.pcm)
        recording.samples.append(Sample(utterance=utterance, decision=decision))
    recording.lid_seconds = time.perf_counter() - started
    return recording


def describe(recording: Recording) -> None:
    print(f"\n  {recording.path.name} (expected {recording.expected}): "
          f"{len(recording.samples)} sentence(s), "
          f"{recording.seconds:.1f} s of speech, "
          f"decided in {recording.lid_seconds * 1000:.0f} ms "
          f"(ratio {recording.lid_ratio:.4f})")
    for sample in recording.samples:
        decision = sample.decision
        verdict = decision.lang_code or "unknown"
        mark = " " if verdict == recording.expected else "!"
        probabilities = "  ".join(
            f"{name} {value:.2f}"
            for name, value in sorted(decision.probabilities.items())
        )
        print(f"    {mark} #{sample.utterance.index} "
              f"{sample.utterance.duration_ms / 1000:4.1f}s -> {verdict:<7} "
              f"margin {decision.margin:.2f}   {probabilities}")


def truth_margins(recording: Recording) -> list[float]:
    """Signed margin towards the right answer, for every sentence.

    Positive means the correct language led, negative means the wrong one did.
    The distribution of this is what decides whether the margin threshold can
    separate right from wrong at all.
    """
    margins = []
    for sample in recording.samples:
        probabilities = sample.decision.probabilities
        if len(probabilities) < 2:
            continue
        others = [v for k, v in probabilities.items() if k != recording.expected]
        margins.append(probabilities.get(recording.expected, 0.0) - max(others))
    return margins


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_recording(recording: Recording, report: Report) -> None:
    name = recording.path.name
    total = len(recording.samples)
    report.add(f"{name} produced sentences to judge", total >= 1,
               f"{total} sentence(s)")
    if not total:
        return
    report.add(f"{name} is never called the other language",
               not recording.wrong,
               f"{len(recording.wrong)} wrong, "
               f"{len(recording.correct)} right, "
               f"{len(recording.unknown)} undecided")
    report.add(f"{name} is mostly decided rather than skipped",
               len(recording.unknown) / total <= MAX_UNKNOWN_SHARE,
               f"{len(recording.unknown)}/{total} undecided, "
               f"limit {MAX_UNKNOWN_SHARE:.0%}")
    report.add(f"{name} language ID is cheap",
               recording.lid_ratio < MAX_LID_RATIO,
               f"ratio {recording.lid_ratio:.4f} < {MAX_LID_RATIO}")


def check_margins(recordings: list[Recording], report: Report) -> None:
    """Would a different margin have done better?"""
    right = [m for r in recordings for m in truth_margins(r) if m > 0]
    wrong = [-m for r in recordings for m in truth_margins(r) if m <= 0]
    print("\n  Margin towards the correct language:")
    if right:
        print(f"    correct  : n={len(right)} min {min(right):.2f} "
              f"median {statistics.median(right):.2f} max {max(right):.2f}")
    if wrong:
        print(f"    incorrect: n={len(wrong)} min {min(wrong):.2f} "
              f"median {statistics.median(wrong):.2f} max {max(wrong):.2f}")
    else:
        print("    incorrect: none - the model never preferred the wrong "
              "language on these recordings")

    report.add("The model prefers the right language more often than not",
               len(right) > len(wrong),
               f"{len(right)} right, {len(wrong)} wrong")
    if right and wrong:
        report.add(
            "A margin exists that keeps the right answers and drops the wrong",
            min(right) > max(wrong),
            f"worst correct {min(right):.2f} vs worst mistake {max(wrong):.2f}",
        )
        if min(right) > max(wrong):
            print(f"    a margin in ({max(wrong):.2f}, {min(right):.2f}) "
                  f"separates them; LID_MIN_MARGIN is {LID_MIN_MARGIN}")
    elif right:
        print(f"    every sentence went the right way; the smallest correct "
              f"margin was {min(right):.2f} against LID_MIN_MARGIN "
              f"{LID_MIN_MARGIN}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vi", type=Path, action="append", default=[],
                        help="WAV of Vietnamese speech (repeatable)")
    parser.add_argument("--ja", type=Path, action="append", default=[],
                        help="WAV of Japanese speech (repeatable)")
    parser.add_argument("--device", default="", help='"cuda" or "cpu"')
    args = parser.parse_args()

    if not args.vi and not args.ja:
        print("Give at least one --vi or --ja recording.")
        return 2

    print("=" * 72)
    print("REAL TEST - LANGUAGE ID (server pipeline, step 6)")
    print(f"choosing between {LID_LANGUAGES}, undecided below a margin of "
          f"{LID_MIN_MARGIN}")
    print("=" * 72)

    report = Report()
    try:
        print("\n  Loading the language ID model ...")
        started = time.perf_counter()
        classifier = VoxLinguaClassifier(device=args.device)
        print(f"  Ready in {time.perf_counter() - started:.1f} s "
              f"from {classifier.source}")
        report.add("The language model loads on the pod", True, classifier.source)
        report.add("It knows both meeting languages",
                   set(classifier.index_of) == set(LID_LANGUAGES),
                   f"{sorted(classifier.index_of)}")

        identifier = LanguageIdentifier(scorer=classifier)
        vad = SileroVAD()

        recordings = [measure(path, "vi", identifier, vad) for path in args.vi]
        recordings += [measure(path, "ja", identifier, vad) for path in args.ja]
        for recording in recordings:
            describe(recording)

        print("\nChecks:")
        for recording in recordings:
            check_recording(recording, report)
        report.add("Both languages were tested",
                   bool(args.vi) and bool(args.ja),
                   f"{len(args.vi)} Vietnamese, {len(args.ja)} Japanese")
        check_margins(recordings, report)

        print(f"\n  Identifier stats: {identifier.stats.seen} judged, "
              f"{identifier.stats.per_language}")

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

    except (LanguageIdError, VADError, ValueError, FileNotFoundError) as exc:
        print(f"\nRESULT: FAIL - {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
