"""REAL TEST - Whisper large-v3 on real Vietnamese and Japanese speech.

MUST RUN ON: the GPU Server pod.
DO NOT RUN ON: the Dev PC agent loop.

``DESIGN.md`` section 3.7. Machine checks cannot tell you whether a
transcript is *correct* - only you can read that - so this measures the
things a machine can, and prints the text for you to read:

* the model loads and decodes far faster than real time, in both modes;
* a partial is cheaper than a final, since it runs several times per sentence;
* forcing the language the LID chose beats letting Whisper guess, or at least
  does not hurt;
* silence produces nothing rather than "Thank you for watching";
* the guards drop invented text without eating real sentences.

The last two are the ones worth staring at. Whisper does not fail on
near-silence, it invents, and the invention is fluent enough to survive a
casual read of the logs.

Usage
-----
    python3.11 server/tests_real/test_real_asr.py \\
        --vi recordings/alice.wav \\
        --ja recordings/bob.wav \\
        --silence recordings/quiet_room.wav
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
    ASR_MODEL,
    CHANNELS,
    CHUNK_BYTES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from server.pipeline.asr import AsrError, Transcriber, WhisperDecoder, pcm_seconds  # noqa: E402
from server.pipeline.buffer import BufferManager, FinalizeReason, Utterance  # noqa: E402
from server.pipeline.overlap import OverlapResolver, PedalboardProcessor  # noqa: E402
from server.pipeline.vad import SileroVAD, VADError, VADSegmenter  # noqa: E402

# A final must decode far faster than the audio it covers, or the pipeline
# falls behind the meeting.
MAX_FINAL_RTF = 0.35
# A partial runs several times per sentence, so it has to be cheaper still.
MAX_PARTIAL_RTF = 0.20
# Sentences that come back empty because a guard refused invented text are a
# success, not a failure - but if most of a recording disappears that way, the
# guards have stopped telling invention from speech.
MAX_REFUSED_SHARE = 0.35


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
class Line:
    utterance: Utterance
    forced: object
    detected: object
    forced_seconds: float = 0.0
    partial_seconds: float = 0.0


@dataclass
class Reading:
    path: Path
    language: str
    lines: list[Line] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return sum(pcm_seconds(line.utterance.pcm) for line in self.lines)

    @property
    def final_rtf(self) -> float:
        total = sum(line.forced_seconds for line in self.lines)
        return total / self.seconds if self.seconds else 0.0

    @property
    def partial_rtf(self) -> float:
        total = sum(line.partial_seconds for line in self.lines)
        return total / self.seconds if self.seconds else 0.0

    @property
    def empty(self) -> list[Line]:
        return [line for line in self.lines if not line.forced.has_text]

    @property
    def refused(self) -> list[Line]:
        """Empty because a guard refused what Whisper produced.

        That is the stage working: near-silence gets a fluent invented
        sentence and the guards throw it away.
        """
        return [line for line in self.empty if line.forced.dropped]

    @property
    def unexplained(self) -> list[Line]:
        """Empty with nothing to explain it - Whisper simply said nothing."""
        return [line for line in self.empty if not line.forced.dropped]

    @property
    def truncated(self) -> list[Line]:
        """Cut off because the recording ended, not because of the pipeline.

        A file stops mid-sentence; a meeting does not. Whatever Whisper makes
        of half a sentence says nothing about how the stage behaves.
        """
        return [line for line in self.lines
                if line.utterance.reason is FinalizeReason.END_OF_STREAM]


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        actual = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
        if actual != (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH):
            raise ValueError(
                f"{path.name} is {actual[0]} Hz / {actual[1]} ch / "
                f"{actual[2] * 8}-bit; expected 16000 Hz / mono / 16-bit"
            )
        return wav.readframes(wav.getnframes())


def sentences_of(path: Path, vad: SileroVAD,
                 resolver: OverlapResolver) -> list[Utterance]:
    pcm = read_pcm(path)
    vad.reset()
    segmenter = VADSegmenter(vad=vad)
    buffer = BufferManager()
    finals: list[Utterance] = []
    for offset in range(0, len(pcm), CHUNK_BYTES):
        finals += buffer.push(segmenter.push(pcm[offset : offset + CHUNK_BYTES])).finals
    segmenter.close()
    finals += buffer.flush().finals
    return [
        Utterance(index=u.index, pcm=resolver.resolve(u.pcm).pcm,
                  start_ms=u.start_ms, reason=u.reason,
                  continues_previous=u.continues_previous)
        for u in finals
    ]


def read_aloud(path: Path, language: str, transcriber: Transcriber,
               vad: SileroVAD, resolver: OverlapResolver) -> Reading:
    reading = Reading(path=path, language=language)
    for utterance in sentences_of(path, vad, resolver):
        started = time.perf_counter()
        forced = transcriber.transcribe(utterance.pcm, language, is_final=True)
        forced_seconds = time.perf_counter() - started

        detected = transcriber.transcribe(utterance.pcm, "", is_final=True)

        started = time.perf_counter()
        transcriber.transcribe(utterance.pcm, language, is_final=False)
        partial_seconds = time.perf_counter() - started

        reading.lines.append(Line(utterance=utterance, forced=forced,
                                  detected=detected,
                                  forced_seconds=forced_seconds,
                                  partial_seconds=partial_seconds))
    return reading


def describe(reading: Reading) -> None:
    print(f"\n  {reading.path.name} (forced {reading.language}): "
          f"{len(reading.lines)} sentence(s), {reading.seconds:.1f} s")
    print(f"    final RTF {reading.final_rtf:.3f}, "
          f"partial RTF {reading.partial_rtf:.3f}")
    for line in reading.lines:
        forced, detected = line.forced, line.detected
        print(f"\n    #{line.utterance.index} "
              f"({pcm_seconds(line.utterance.pcm):.1f}s)")
        print(f"      forced   [{forced.lang_code}] {forced.text!r}")
        if detected.text != forced.text:
            print(f"      detected [{detected.lang_code}] {detected.text!r}")
        if forced.dropped:
            for piece, reason in forced.dropped:
                print(f"      dropped ({reason}): {piece.text.strip()[:50]!r}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_reading(reading: Reading, report: Report) -> None:
    name = reading.path.name
    report.add(f"{name} produced sentences", bool(reading.lines),
               f"{len(reading.lines)} sentence(s)")
    if not reading.lines:
        return
    truncated = {id(line) for line in reading.truncated}
    unexplained = [line for line in reading.unexplained
                   if id(line) not in truncated]
    report.add(
        f"{name} never loses a sentence without saying why",
        not unexplained,
        f"{len(unexplained)} empty with no refused segment to explain it",
    )
    judged = [line for line in reading.lines if id(line) not in truncated]
    refused = [line for line in reading.refused if id(line) not in truncated]
    share = len(refused) / len(judged) if judged else 0.0
    report.add(
        f"{name} guards refuse a minority of sentences",
        share <= MAX_REFUSED_SHARE,
        f"{len(refused)}/{len(judged)} refused as invented, "
        f"limit {MAX_REFUSED_SHARE:.0%}",
    )
    if reading.truncated:
        print(f"    ({len(reading.truncated)} sentence(s) cut off by the end "
              "of the recording, not judged)")
    report.add(f"{name} decodes far faster than real time",
               reading.final_rtf < MAX_FINAL_RTF,
               f"final RTF {reading.final_rtf:.3f} < {MAX_FINAL_RTF}")
    report.add(f"{name} partials are cheaper than finals",
               reading.partial_rtf < MAX_PARTIAL_RTF,
               f"partial RTF {reading.partial_rtf:.3f} < {MAX_PARTIAL_RTF}")
    forced_right = sum(1 for line in reading.lines
                       if line.forced.lang_code == reading.language)
    report.add(f"{name} keeps the language it was told to use",
               forced_right == len(reading.lines),
               f"{forced_right}/{len(reading.lines)}")


def check_agreement(reading: Reading, report: Report) -> None:
    """Forcing the LID's answer should not make the transcript worse."""
    detected_language = [line.detected.lang_code for line in reading.lines]
    agreed = sum(1 for code in detected_language if code == reading.language)
    print(f"\n  {reading.path.name}: Whisper detected "
          f"{reading.language} on its own for {agreed}/{len(reading.lines)} "
          f"sentence(s) - {sorted(set(detected_language))}")
    identical = sum(1 for line in reading.lines
                    if line.forced.text.strip() == line.detected.text.strip())
    print(f"    forced and detected produced the same text for "
          f"{identical}/{len(reading.lines)}")


def check_silence(pcm: bytes, transcriber: Transcriber, report: Report) -> None:
    """The invented-text case, which is the reason the guards exist."""
    transcript = transcriber.transcribe(pcm, "", is_final=True)
    print(f"\n  silence ({pcm_seconds(pcm):.1f}s) -> {transcript.text!r}")
    for piece, reason in transcript.dropped:
        print(f"    dropped ({reason}): {piece.text.strip()[:60]!r}")
    report.add("Silence produces no transcript", not transcript.has_text,
               f"{transcript.text[:60]!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vi", type=Path, default=None,
                        help="WAV of Vietnamese speech")
    parser.add_argument("--ja", type=Path, default=None,
                        help="WAV of Japanese speech")
    parser.add_argument("--silence", type=Path, default=None,
                        help="WAV of a quiet room, to catch invented text")
    parser.add_argument("--model", default="", help=f"default: {ASR_MODEL}")
    parser.add_argument("--device", default="", help='"cuda" or "cpu"')
    args = parser.parse_args()

    if not args.vi and not args.ja:
        print("Give at least one --vi or --ja recording.")
        return 2

    print("=" * 72)
    print("REAL TEST - ASR (server pipeline, step 7)")
    print(f"model {args.model or ASR_MODEL}")
    print("=" * 72)

    report = Report()
    try:
        print("\n  Loading Whisper ...")
        started = time.perf_counter()
        decoder = WhisperDecoder(model_id=args.model, device=args.device)
        print(f"  Ready in {time.perf_counter() - started:.1f} s "
              f"from {decoder.source}")
        report.add("Whisper loads on the pod", True, decoder.source)

        transcriber = Transcriber(decoder=decoder)
        vad = SileroVAD()
        resolver = OverlapResolver(processor=PedalboardProcessor())

        readings = []
        for path, language in ((args.vi, "vi"), (args.ja, "ja")):
            if path is not None:
                readings.append(read_aloud(path, language, transcriber, vad,
                                           resolver))
        for reading in readings:
            describe(reading)

        print("\nChecks:")
        for reading in readings:
            check_reading(reading, report)
        for reading in readings:
            check_agreement(reading, report)
        if args.silence is not None:
            check_silence(read_pcm(args.silence), transcriber, report)
        else:
            print("\n  (no --silence recording: invented text is unproven)")

        rtfs = [r.final_rtf for r in readings if r.lines]
        if rtfs:
            print(f"\n  Final RTF across recordings: "
                  f"mean {statistics.fmean(rtfs):.3f}, worst {max(rtfs):.3f}")
        print(f"  ASR stats: {transcriber.stats.finals} finals, "
              f"{transcriber.stats.partials} partials, "
              f"{transcriber.stats.dropped_pieces} segments dropped "
              f"{transcriber.stats.dropped_reasons}")

        print("\n  Read the transcripts above. Nothing here can tell you "
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

    except (AsrError, VADError, ValueError, FileNotFoundError) as exc:
        print(f"\nRESULT: FAIL - {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
