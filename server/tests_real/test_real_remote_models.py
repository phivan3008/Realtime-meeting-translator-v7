"""REAL TEST - the two models on the GPU pod, called from the Windows machine.

MUST RUN ON: the Windows machine that runs the pipeline, with
``gpu_server/whisper_server.py`` and vLLM running on the pod and both ports
reachable (port-forward or SSH tunnel - see gpu_server/README.md).
DO NOT RUN ON: the Dev PC agent loop.

The pipeline moved off the pod; the pod now only serves Whisper and Gemma 4.
Every 600 ms the running text needs one Whisper decode, and each of those now
crosses the network. This proves the seam before anything is built on it:

* both servers answer, and serve the checkpoints the pipeline expects;
* Whisper sends back what the streaming policy reads - word timestamps and
  the three scores per segment - and not placeholders;
* the same audio decodes to the same answer twice (determinism survives the
  wire, which is what makes the replay against the old baseline meaningful);
* a partial decode, round trip included, fits inside the 600 ms between two
  partials, and a final fits well inside the sentence it decodes;
* one translation comes back, and how long it took.

Usage
-----
    py -3.11 server/tests_real/test_real_remote_models.py
    py -3.11 server/tests_real/test_real_remote_models.py ^
        --wav recordings\\meeting_speech.wav ^
        --asr-url http://127.0.0.1:8002 ^
        --translate-url http://127.0.0.1:8001/v1

Without ``--wav`` a synthetic tone is decoded, which checks the plumbing and
the timing but not the words; give a recording of real speech to check that
words come back with timestamps.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (  # noqa: E402
    ASR_BEAM_SIZE_FINAL,
    ASR_BEAM_SIZE_PARTIAL,
    ASR_MODEL,
    ASR_SERVER_URL,
    PARTIAL_INTERVAL_MS,
    PARTIAL_WINDOW_SECONDS,
    SAMPLE_RATE,
    TRANSLATE_BASE_URL,
    VLLM_MAX_MODEL_LEN,
)
from server.pipeline.asr import AsrError  # noqa: E402
from server.pipeline.asr_remote import RemoteWhisperDecoder  # noqa: E402

#: A final decodes a whole sentence (up to 7 s) with a beam. In-process it had
#: to stay under 0.35 of the audio; the round trip is now inside that budget.
MAX_FINAL_RTF = 0.35
#: Longest sentence the buffer manager lets through.
FINAL_SECONDS = 7.0


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


def percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(share * (len(ordered) - 1))))
    return ordered[index]


def read_samples(path: Optional[Path], seconds: float) -> np.ndarray:
    """``seconds`` of audio as the pipeline holds it: PCM16 / 32768."""
    if path is None:
        t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
        pcm = (np.sin(2 * np.pi * 220 * t) * 3000).astype("<i2")
    else:
        with wave.open(str(path), "rb") as reader:
            if (reader.getframerate(), reader.getnchannels(),
                    reader.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise SystemExit(f"{path} is not 16 kHz mono 16-bit; convert "
                                 "it with ffmpeg -ac 1 -ar 16000 "
                                 "-sample_fmt s16")
            raw = reader.readframes(int(seconds * SAMPLE_RATE))
        pcm = np.frombuffer(raw, "<i2")
    return pcm.astype(np.float32) / 32768.0


def timed_decode(decoder, samples, beam_size, lang_code="", prompt=None):
    started = time.perf_counter()
    pieces, language = decoder.decode(samples, lang_code, beam_size, prompt)
    return pieces, language, time.perf_counter() - started


def answer_key(pieces) -> list:
    return [(piece.text, round(piece.avg_logprob, 4),
             [(word.text, round(word.start, 3)) for word in piece.words])
            for piece in pieces]


def check_whisper(decoder, wav: Optional[Path], rounds: int,
                  report: Report) -> None:
    report.add("Whisper serves the configured checkpoint",
               decoder.model_id == ASR_MODEL,
               f"{decoder.model_id!r}, ASR_MODEL={ASR_MODEL!r}")
    report.add("Whisper runs on the GPU in float16",
               decoder.device.startswith("cuda")
               and decoder.compute_type == "float16",
               f"{decoder.device} ({decoder.compute_type})")

    window = read_samples(wav, PARTIAL_WINDOW_SECONDS)
    times = []
    for _ in range(rounds):
        pieces, language, seconds = timed_decode(decoder, window,
                                                 ASR_BEAM_SIZE_PARTIAL)
        times.append(seconds)
    budget = PARTIAL_INTERVAL_MS / 1000
    print(f"\n  Partial decode of {PARTIAL_WINDOW_SECONDS:.0f} s, {rounds} "
          f"rounds: median {statistics.median(times) * 1000:.0f} ms, "
          f"p95 {percentile(times, 0.95) * 1000:.0f} ms, "
          f"max {max(times) * 1000:.0f} ms")
    report.add(f"A partial round trip fits between two partials "
               f"({PARTIAL_INTERVAL_MS} ms)",
               percentile(times, 0.95) < budget,
               f"p95 {percentile(times, 0.95) * 1000:.0f} ms")

    if wav is not None:
        words = [word for piece in pieces for word in piece.words]
        print(f"  [{language}] {''.join(piece.text for piece in pieces)!r}")
        report.add("Real speech comes back as words with timestamps",
                   bool(words) and all(w.end >= w.start for w in words),
                   f"{len(words)} words")
        report.add("Segments carry real scores, not placeholders",
                   bool(pieces) and any(piece.avg_logprob != 0.0
                                        for piece in pieces),
                   ", ".join(f"logprob {piece.avg_logprob:.2f} no_speech "
                             f"{piece.no_speech_prob:.2f} ratio "
                             f"{piece.compression_ratio:.2f}"
                             for piece in pieces[:3]))

    first, _language, _ = timed_decode(decoder, window, ASR_BEAM_SIZE_PARTIAL)
    second, _language, _ = timed_decode(decoder, window, ASR_BEAM_SIZE_PARTIAL)
    report.add("The same audio decodes to the same answer twice",
               answer_key(first) == answer_key(second))

    sentence = read_samples(wav, FINAL_SECONDS)
    seconds_of_audio = sentence.size / SAMPLE_RATE
    finals = [timed_decode(decoder, sentence, ASR_BEAM_SIZE_FINAL)[2]
              for _ in range(3)]
    slowest = max(finals)
    report.add(f"A final ({seconds_of_audio:.1f} s, beam "
               f"{ASR_BEAM_SIZE_FINAL}) fits its budget",
               slowest < MAX_FINAL_RTF * seconds_of_audio,
               f"slowest {slowest * 1000:.0f} ms, budget "
               f"{MAX_FINAL_RTF * seconds_of_audio * 1000:.0f} ms")


def check_translation(url: str, report: Report) -> None:
    from server.pipeline.translate import TranslationError, VllmClient

    try:
        client = VllmClient(base_url=url)
    except TranslationError as exc:
        report.add("vLLM answers", False, str(exc))
        return
    report.add("vLLM answers", True, client.source)
    report.add(f"vLLM was started with max_model_len {VLLM_MAX_MODEL_LEN}",
               client.max_model_len in (None, VLLM_MAX_MODEL_LEN),
               f"server reports {client.max_model_len}")
    started = time.perf_counter()
    try:
        text = client.complete([{"role": "user", "content":
                                 "Translate into Japanese. Reply with the "
                                 "translation only.\n\nXin chào mọi người."}])
    except TranslationError as exc:
        report.add("One translation comes back", False, str(exc))
        return
    seconds = time.perf_counter() - started
    print(f"  Gemma: {text.strip()!r}")
    report.add("One translation comes back", bool(text.strip()),
               f"{seconds * 1000:.0f} ms")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-url", default=ASR_SERVER_URL)
    parser.add_argument("--translate-url", default=TRANSLATE_BASE_URL)
    parser.add_argument("--wav", type=Path, default=None,
                        help="16 kHz mono 16-bit speech; without it a tone")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--skip-translate", action="store_true")
    args = parser.parse_args(argv)

    print("=" * 72)
    print("REAL TEST - models on the GPU pod, from the Windows machine")
    print(f"Whisper {args.asr_url}   vLLM {args.translate_url}")
    print("=" * 72)
    report = Report()

    print("\nWhisper:")
    try:
        started = time.perf_counter()
        decoder = RemoteWhisperDecoder(base_url=args.asr_url, model_id="")
        report.add("Whisper server answers", True,
                   f"{decoder.source}, ready in "
                   f"{(time.perf_counter() - started) * 1000:.0f} ms")
    except AsrError as exc:
        report.add("Whisper server answers", False, str(exc))
        decoder = None
    if decoder is not None:
        check_whisper(decoder, args.wav, args.rounds, report)

    if not args.skip_translate:
        print("\nTranslation:")
        check_translation(args.translate_url, report)

    print("\n" + "=" * 72)
    if report.failed:
        print(f"RESULT: FAIL ({len(report.failed)} of {len(report.checks)})")
        for check in report.failed:
            print(f"  - {check.name}: {check.detail}")
        return 1
    print(f"RESULT: PASS ({len(report.checks)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
