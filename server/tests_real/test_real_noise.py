"""REAL TEST - the Deep Noise Filter (AST) on real recorded audio.

MUST RUN ON: the GPU Server pod.
DO NOT RUN ON: the Dev PC agent loop.

Silero removes silence but not noise: it is a voice activity detector, and it
fires happily on a cough or a keyboard. AST gets the last word on whether an
utterance is worth transcribing, so this checks it on the two things
``DESIGN.md`` names - keyboard clatter and coughing - plus real speech.

What it proves
--------------
1. AST loads on the pod and names its source and device.
2. It is fast enough that the filter costs nothing next to the ASR call it
   saves.
3. Real speech is kept, with a high speech score.
4. Recorded keyboard and cough audio is dropped, and the label it is dropped
   under is a sensible one.
5. Run through the whole VAD -> buffer -> filter chain, a noise recording
   produces no surviving sentence, and a speech recording loses none.

Recording the inputs
--------------------
The client only ever hears the meeting audio, so a remote participant's
keyboard reaches us as playback, not as room sound. Recording the noise
files by *playing* typing and coughing through the speakers and capturing
the loopback is therefore the accurate simulation, not a shortcut.

Usage
-----
    python3.11 server/tests_real/test_real_noise.py \\
        --speech recordings/meeting_speech.wav \\
        --noise recordings/keyboard.wav \\
        --noise recordings/cough.wav
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (  # noqa: E402
    AST_MODEL_ID,
    CHANNELS,
    CHUNK_BYTES,
    NOISE_MIN_SPEECH_SCORE,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from server.pipeline.buffer import BufferManager, Utterance  # noqa: E402
from server.pipeline.noise import (  # noqa: E402
    AstClassifier,
    Classification,
    NoiseFilter,
    NoiseFilterError,
    Verdict,
)
from server.pipeline.vad import SileroVAD, VADError, VADSegmenter  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Classifying one utterance must stay far below the audio it covers, or the
# filter costs more than the Whisper call it is meant to save.
MAX_CLASSIFY_RATIO = 0.1


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
class FileResult:
    path: Path
    audio_seconds: float = 0.0
    classify_seconds: float = 0.0
    classification: Classification = field(
        default_factory=lambda: Classification(0.0, 0.0)
    )
    kept: bool = True
    reason: str = ""
    judged: list[tuple[Utterance, Verdict]] = field(default_factory=list)

    @property
    def utterances(self) -> int:
        return len(self.judged)

    @property
    def utterances_kept(self) -> int:
        return sum(1 for _, verdict in self.judged if verdict.keep)

    @property
    def dropped(self) -> list[tuple[Utterance, Verdict]]:
        return [(u, v) for u, v in self.judged if not v.keep]

    @property
    def classify_ratio(self) -> float:
        return (self.classify_seconds / self.audio_seconds
                if self.audio_seconds else 0.0)


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        actual = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
        if actual != (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH):
            raise ValueError(
                f"{path.name} is {actual[0]} Hz / {actual[1]} ch / "
                f"{actual[2] * 8}-bit; expected 16000 Hz / mono / 16-bit"
            )
        return wav.readframes(wav.getnframes())


def classify_file(path: Path, filt: NoiseFilter) -> FileResult:
    """Judge the whole file as if it were one utterance."""
    pcm = read_pcm(path)
    result = FileResult(path=path,
                        audio_seconds=len(pcm) / SAMPLE_WIDTH / SAMPLE_RATE)
    started = time.perf_counter()
    verdict = filt.judge(pcm)
    result.classify_seconds = time.perf_counter() - started
    result.classification = verdict.classification
    result.kept = verdict.keep
    result.reason = verdict.reason
    return result


def run_pipeline(path: Path, vad: SileroVAD,
                 filt: NoiseFilter) -> list[tuple[Utterance, Verdict]]:
    """VAD -> buffer -> filter, the way a live session runs it.

    Returns every sentence with the verdict it got, not just a count: when the
    filter eats a real sentence, the only useful next step is listening to
    that exact sentence.
    """
    pcm = read_pcm(path)
    vad.reset()
    segmenter = VADSegmenter(vad=vad)
    buffer = BufferManager()
    finals = []
    for offset in range(0, len(pcm), CHUNK_BYTES):
        finals += buffer.push(segmenter.push(pcm[offset : offset + CHUNK_BYTES])).finals
    segmenter.close()
    finals += buffer.flush().finals
    return [(utterance, filt.judge(utterance.pcm)) for utterance in finals]


def describe(result: FileResult) -> None:
    print(f"\n  {result.path.name}: {result.audio_seconds:.1f} s")
    print(f"    {result.classification.summary()}")
    print(f"    verdict: {'KEEP' if result.kept else 'DROP'} - {result.reason}")
    print(f"    classified in {result.classify_seconds * 1000:.0f} ms "
          f"(ratio {result.classify_ratio:.4f})")
    print(f"    through the full pipeline: {result.utterances} utterance(s), "
          f"{result.utterances_kept} survived the filter")
    for utterance, verdict in result.judged:
        mark = "keep" if verdict.keep else "DROP"
        print(f"      #{utterance.index}  "
              f"{utterance.start_ms / 1000:6.2f}s -> "
              f"{utterance.end_ms / 1000:6.2f}s "
              f"({utterance.duration_ms / 1000:4.2f}s)  {mark:<4}  "
              f"{verdict.classification.summary()}")


def save_dropped(result: FileResult) -> Optional[Path]:
    """Write out the sentences the filter refused, so they can be listened to.

    A count cannot tell you whether the filter was right. The audio can.
    """
    if not result.dropped:
        return None
    directory = OUTPUT_DIR / f"{result.path.stem}_dropped"
    directory.mkdir(parents=True, exist_ok=True)
    for utterance, verdict in result.dropped:
        label = (verdict.classification.noise_label or "unknown").replace(" ", "_")
        with wave.open(str(directory / f"{utterance.index:03d}_{label}.wav"),
                       "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(utterance.pcm)
    return directory


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech", type=Path, required=True,
                        help="WAV of real meeting speech")
    parser.add_argument("--noise", type=Path, action="append", default=[],
                        help="WAV of keyboard, coughing, ... (repeatable)")
    parser.add_argument("--model-id", default="",
                        help="override the AST checkpoint, or point at a "
                             "local directory on an offline pod")
    parser.add_argument("--device", default="",
                        help='"cuda" or "cpu"; auto by default')
    args = parser.parse_args()

    print("=" * 72)
    print("REAL TEST - DEEP NOISE FILTER (server pipeline, step 3)")
    print(f"keep unless speech score < {NOISE_MIN_SPEECH_SCORE} "
          "and something non-speech scored higher")
    print("=" * 72)

    report = Report()
    try:
        print()
        print(f"  Loading {args.model_id or AST_MODEL_ID} ...")
        started = time.perf_counter()
        classifier = AstClassifier(model_id=args.model_id, device=args.device)
        print(f"  Classifier ready in {time.perf_counter() - started:.1f} s "
              f"from {classifier.source}")
        print(f"  {len(classifier.labels)} classes, "
              f"e.g. {classifier.labels[:3]}")
        report.add("The classifier loads on the pod", True, classifier.source)
        report.add("The label set looks like AudioSet",
                   len(classifier.labels) >= 500,
                   f"{len(classifier.labels)} labels")

        filt = NoiseFilter(classifier=classifier)
        vad = SileroVAD()

        speech = classify_file(args.speech, filt)
        speech.judged = run_pipeline(args.speech, vad, filt)
        describe(speech)

        noises = []
        for path in args.noise:
            result = classify_file(path, filt)
            result.judged = run_pipeline(path, vad, filt)
            describe(result)
            noises.append(result)

        print("\nChecks:")
        report.add("Real speech is kept", speech.kept, speech.reason)
        report.add(
            "Speech scores well above the threshold",
            speech.classification.speech_score >= NOISE_MIN_SPEECH_SCORE * 2,
            f"speech {speech.classification.speech_score:.2f} vs threshold "
            f"{NOISE_MIN_SPEECH_SCORE}",
        )
        report.add(
            "No speech sentence is thrown away by the filter",
            speech.utterances_kept == speech.utterances,
            f"{speech.utterances_kept}/{speech.utterances} survived",
        )

        if not noises:
            print("  (no --noise file given; the drop side is unproven)")
        for result in noises:
            name = result.path.name
            report.add(f"{name} is dropped", not result.kept, result.reason)
            report.add(
                f"{name} is recognised as something non-speech",
                bool(result.classification.noise_label),
                f"label {result.classification.noise_label or 'none'!r}, "
                f"top {result.classification.top_label!r}",
            )
            report.add(
                f"{name} leaves no sentence for the ASR",
                result.utterances_kept == 0,
                f"{result.utterances_kept}/{result.utterances} survived",
            )

        ratios = [r.classify_ratio for r in [speech, *noises]]
        worst = max(ratios)
        print(f"\n  Classification cost: "
              f"mean ratio {statistics.fmean(ratios):.4f}, worst {worst:.4f}")
        report.add("The filter is cheap next to the audio it covers",
                   worst < MAX_CLASSIFY_RATIO,
                   f"worst ratio {worst:.4f} < {MAX_CLASSIFY_RATIO}")

        dropped_dir = save_dropped(speech)
        if dropped_dir is not None:
            print(f"\n  The filter refused {len(speech.dropped)} sentence(s) "
                  f"from {speech.path.name}: {dropped_dir}")
            print("  Listen to them. If they really are a cough or a throat "
                  "clear, the filter was right and this check is too strict. "
                  "If they are speech, the filter is broken.")

        print(f"\n  Filter stats: {filt.stats.seen} judged, "
              f"{filt.stats.dropped} dropped, labels {filt.stats.dropped_labels}")

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

    except (NoiseFilterError, VADError, ValueError, FileNotFoundError) as exc:
        print(f"\nRESULT: FAIL - {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
