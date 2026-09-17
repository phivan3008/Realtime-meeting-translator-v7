"""REAL TEST - the whole server pipeline on a recorded meeting, as streamed.

MUST RUN ON: the GPU Server pod.
DO NOT RUN ON: the Dev PC agent loop.

``test_real_asr.py`` puts Whisper to one sentence at a time. This replays a
whole meeting through a real ``ServerSession`` - Silero VAD, the buffer
manager, the overlap resolver, ECAPA voiceprints, the language split, the
LID, streaming Whisper and, with ``--translate``, vLLM - in 200 ms chunks,
exactly as the client sends them. Every message the session sends is written
into a debug log in the client's own format, so the run can be compared with
the logs the client wrote for the same meeting.

What it checks is what a person watching the meeting would notice:

* Japanese is written without a space between every character (the 09-10 run
  of the streaming rewrite had 144 such sentences);
* a word is not shown twice at the join between two decodes (131 sentences
  on that run, 14-16 on the older sliding-window runs);
* the running text does not rewrite what the reader is half way through
  (the sliding window kept 34% of the line per update, the streaming rewrite
  64%);
* no stage broke, and the pipeline kept well inside real time.

Usage
-----
    python3.11 server/tests_real/test_real_streaming.py \\
        --wav recordings/meeting_30min.wav \\
        --limit-seconds 180 \\
        --out server/tests_real/output/streaming.debug.txt \\
        --baseline recordings/meeting-20260911-144023.debug.txt \\
        --baseline recordings/meeting-20260910-144120.debug.txt

The WAV must be 16 kHz mono 16-bit, the format the client streams. Convert a
meeting recording with::

    ffmpeg -i meeting.m4a -ac 1 -ar 16000 -sample_fmt s16 meeting_16k.wav

``--baseline`` logs are optional. They are compared only after the replay is
confirmed to be the same meeting, and only on the stretch the replay covers.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.protocol import Hello, make_bye, parse_message  # noqa: E402
from server.analysis.compare_logs import table  # noqa: E402
from server.analysis.debuglog import (  # noqa: E402
    LogWriter,
    measure,
    parse,
    same_meeting,
)
from server.config import CHANNELS, CHUNK_BYTES, SAMPLE_RATE, SAMPLE_WIDTH  # noqa: E402
from server.net.session import ServerSession, Stages  # noqa: E402

#: A replay faster than this share of real time leaves the live pipeline
#: room to breathe. The last trustworthy baseline spent 18.5%.
MAX_PIPELINE_SHARE = 0.35
#: The running text runs every 600 ms; one pass longer than this is a stall
#: the reader sees.
MAX_PARTIAL_SECONDS = 1.5
#: Sentences carrying a word twice in a row. The sliding-window runs of the
#: same meeting sat at 5%; the streaming rewrite before this fix at 45%.
MAX_REPEAT_SHARE = 0.08
#: Mean share of the running text an update keeps. The sliding window kept
#: 34%, the streaming rewrite 64%.
MIN_UPDATE_SURVIVAL = 0.50
#: Sentences whose language differs from their running text's. The
#: sliding-window run of the same meeting sat at 2.7%; the first streaming
#: run, with the language fixed on one window, at 12.8%.
MAX_DISAGREE_SHARE = 0.05
#: ... or this many sentences, whichever is more: on a three-minute replay of
#: 35 sentences, 5% is a single one.
MAX_DISAGREE_COUNT = 2
#: Translations refused, with --translate. The baseline refused 3%.
MAX_REFUSED_SHARE = 0.06


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

    def note(self, name: str, detail: str) -> None:
        print(f"  [INFO] {name} - {detail}")

    @property
    def failed(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]


def read_pcm(path: Path, limit_seconds: float = 0.0) -> bytes:
    with wave.open(str(path), "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(),
                reader.getsampwidth()) != (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH):
            raise ValueError(
                f"{path} is {reader.getframerate()} Hz, "
                f"{reader.getnchannels()} channel(s), "
                f"{reader.getsampwidth() * 8}-bit; convert it with "
                f"ffmpeg -ac 1 -ar 16000 -sample_fmt s16")
        frames = reader.getnframes()
        if limit_seconds > 0:
            frames = min(frames, int(limit_seconds * SAMPLE_RATE))
        return reader.readframes(frames)


def load_stages(args) -> tuple:
    """The real models, each reported as it loads."""
    from server.pipeline.asr import Transcriber, WhisperDecoder
    from server.pipeline.diarization import SpeakerIdentifier
    from server.pipeline.lid import LanguageIdentifier
    from server.pipeline.overlap import OverlapResolver
    from server.pipeline.vad import SileroVAD

    loaded = {}

    def attempt(name, build):
        started = time.perf_counter()
        try:
            loaded[name] = build()
            print(f"  {name}: ready in {time.perf_counter() - started:.1f} s")
        except Exception as exc:                        # noqa: BLE001
            loaded[name] = None
            print(f"  {name}: FAILED - {exc}")

    attempt("vad", SileroVAD)
    attempt("overlap", OverlapResolver)
    attempt("speaker", SpeakerIdentifier)
    attempt("language", LanguageIdentifier)
    attempt("asr", lambda: Transcriber(decoder=WhisperDecoder(
        model_id=args.model, device=args.device)))
    if args.translate:
        from server.pipeline.translate import Translator
        attempt("translate", Translator)
    return loaded


def replay(pcm: bytes, loaded: dict, recorder: LogWriter,
           clock: list) -> ServerSession:
    from server.pipeline.vad import VADSegmenter

    stages = Stages(
        overlap_resolver=loaded.get("overlap"),
        speaker_identifier=loaded.get("speaker"),
        language_identifier=loaded.get("language"),
        transcriber=loaded.get("asr"),
        translator=loaded.get("translate"),
    )
    session = ServerSession(
        segmenter_factory=lambda: VADSegmenter(vad=loaded["vad"]),
        stages=stages,
        # A replay runs faster than the meeting did, so a queue with a
        # real-time budget would give up on translations that live would
        # have made it. Inline keeps them all, in order.
        translation_inline=True,
    )

    def deliver(response) -> None:
        for raw in response.messages:
            recorder.apply(parse_message(raw))

    deliver(session.handle_text(Hello(session_id="replay",
                                      client="test_real_streaming").to_json()))
    total = len(pcm) // CHUNK_BYTES
    started = time.perf_counter()
    for index in range(total):
        clock[0] = (index + 1) * 0.2
        deliver(session.handle_binary(
            pcm[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES]))
        if index and index % 300 == 0:
            spent = time.perf_counter() - started
            print(f"    {clock[0]:7.1f} s of audio in {spent:6.1f} s")
    deliver(session.handle_text(make_bye("replay finished")))
    return session


def check(session: ServerSession, debug_path: Path, audio_seconds: float,
          args, report: Report) -> None:
    run = parse(debug_path)
    found = measure(run)
    stats = session.stats

    print("\nWhat the session did:")
    report.note("sentences", f"{found['sentences']} from "
                f"{stats.utterances} utterances, {stats.running_texts} "
                f"running texts")
    report.note("language", f"{stats.language_splits} utterances split "
                f"({stats.language_splits_refused} refused by the running "
                f"text), "
                f"{stats.language_flips} sentences decoded again in the "
                f"LID's language, {stats.running_language_changes} running "
                f"texts changed language, {stats.running_language_cuts} "
                f"cut there")
    if session.language_splitter is not None:
        split = session.language_splitter.stats
        report.note("language split", f"{split.split}/{split.checked} cut; "
                    f"declined: {split.one_language} one language, "
                    f"{split.undecided} undecided, {split.sliver} fragment, "
                    f"{split.refused_on_review} on review; "
                    f"{split.probes} probes")
    if session.transcriber is not None:
        asr = session.transcriber.stats
        report.note("asr", f"{asr.finals} finals ({asr.empty_finals} empty), "
                    f"{asr.committed_events} commits, "
                    f"{asr.duplicates_removed} repeats removed at a join, "
                    f"dropped {asr.dropped_reasons}")
    if session.speaker_history is not None:
        history = session.speaker_history.stats
        report.note("speakers", f"{history.speakers} after {history.runs} "
                    f"runs, sized {history.sizes}, {history.would_move} "
                    f"would move (not sent unless SPEAKER_RECLUSTER=1), "
                    f"{history.corrections} labels corrected, "
                    f"{history.forced_merges} merges forced by the cap, "
                    f"refused merges {sorted(history.stop_scores)[:5]}...")
    report.note("stages", {k: round(v, 1) for k, v in
                           sorted(stats.stage_seconds.items(),
                                  key=lambda kv: -kv[1])})

    print("\nChecks:")
    report.add("No stage raised",
               not stats.stage_failures and not stats.pipeline_errors,
               f"{stats.stage_failures or 'none'}, "
               f"{stats.pipeline_errors} pipeline errors")
    report.add("Sentences came out at all", found["sentences"] > 0,
               f"{found['sentences']}")
    report.add("Japanese is written without spaces between characters",
               found["japanese_spaced"] == 0,
               f"{found['japanese_spaced']} of "
               f"{found['japanese_sentences']} Japanese sentences")
    share = found["with_repeats"] / max(found["sentences"], 1)
    report.add("A word is not shown twice at a join",
               share <= MAX_REPEAT_SHARE,
               f"{found['with_repeats']} sentences ({share:.1%}), "
               f"limit {MAX_REPEAT_SHARE:.0%}")
    report.add("No sentence mixes Japanese and Vietnamese",
               found["mixed_language"] == 0,
               f"{found['mixed_language']} sentences")
    disagree = found["language_disagrees"] / max(found["sentences"], 1)
    report.add("Sentences mostly keep their running text's language",
               disagree <= MAX_DISAGREE_SHARE
               or found["language_disagrees"] <= MAX_DISAGREE_COUNT,
               f"{found['language_disagrees']} sentences ({disagree:.1%}), "
               f"limit {MAX_DISAGREE_SHARE:.0%} or {MAX_DISAGREE_COUNT}")
    lost = [final for final in run.finals if final.lost_turn]
    report.add("No turn the running text showed is missing from its sentence",
               not lost,
               f"{len(lost)} sentences"
               + "".join(f"\n        #{final.sentence} [{final.lang}] "
                         f"{final.text[:40]!r} lost [{final.lost_turn}] "
                         f"{final.partials[-1][1][:40]!r}"
                         for final in lost[:5]))
    report.add("The running text keeps what the reader is reading",
               found["update_survival"] >= MIN_UPDATE_SURVIVAL,
               f"{found['update_survival']:.1%} kept per update, "
               f"{found['updates_wiping_half']:.1%} wipe over half")
    spent = sum(stats.stage_seconds.values())
    report.add("The pipeline keeps well inside real time",
               spent <= MAX_PIPELINE_SHARE * audio_seconds,
               f"{spent:.1f} s for {audio_seconds:.1f} s "
               f"({spent / max(audio_seconds, 1e-9):.1%})")
    report.add("No running text stalls the socket",
               stats.slowest_partial_seconds <= MAX_PARTIAL_SECONDS,
               f"slowest {stats.slowest_partial_seconds:.2f} s, "
               f"slowest sentence {stats.slowest_utterance_seconds:.2f} s")
    report.note("near the block list",
                f"{found['near_block_list']} sentences read like a listed "
                f"invention with words changed (baseline runs: 12-16 over "
                f"the whole meeting) - read them in the log")
    if args.translate:
        refused = found["refused_translations"] / max(found["sentences"], 1)
        report.add("Translations are not refused more than the baseline",
                   refused <= MAX_REFUSED_SHARE,
                   f"{found['refused_translations']} refused ({refused:.1%})")
        if session.worker is not None:
            tr = session.worker.translator.stats
            report.note("echoes", f"{tr.retried} retried without the "
                        f"history, {tr.rescued} rescued")

    for path in args.baseline:
        baseline = parse(path)
        verdict = same_meeting(baseline, run)
        report.add(f"{path.name} is the same meeting as the replay",
                   verdict["same"],
                   f"offset {verdict['offset']:+.1f} s, "
                   f"{verdict['votes']}/{verdict['shared']} words agree")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--wav", type=Path, required=True,
                        help="16 kHz mono 16-bit recording of a meeting")
    parser.add_argument("--limit-seconds", type=float, default=0.0,
                        help="replay only the start of it")
    parser.add_argument("--out", type=Path,
                        default=Path("server/tests_real/output/"
                                     "streaming.debug.txt"),
                        help="the debug log to write")
    parser.add_argument("--baseline", type=Path, action="append", default=[],
                        help="a client debug log of the same meeting "
                             "(repeatable)")
    parser.add_argument("--translate", action="store_true",
                        help="translate as well; vLLM must be running")
    parser.add_argument("--model", default="", help="Whisper checkpoint")
    parser.add_argument("--device", default="", help='"cuda" or "cpu"')
    args = parser.parse_args()

    print("=" * 72)
    print("REAL TEST - streaming pipeline on a recorded meeting")
    print("=" * 72)
    report = Report()
    try:
        pcm = read_pcm(args.wav, args.limit_seconds)
        pcm += bytes(-len(pcm) % CHUNK_BYTES)
        audio_seconds = len(pcm) / SAMPLE_WIDTH / SAMPLE_RATE
        print(f"\n  {args.wav}: {audio_seconds:.1f} s")

        print("\nLoading the models:")
        loaded = load_stages(args)
        for name in ("vad", "asr", "language", "speaker", "overlap"):
            report.add(f"{name} loads", loaded.get(name) is not None)
        if args.translate:
            report.add("translate loads", loaded.get("translate") is not None)
        if loaded.get("vad") is None or loaded.get("asr") is None:
            raise RuntimeError("cannot replay without the VAD and Whisper")

        clock = [0.0]
        # What the session decided and why, next to the replayed log: the
        # splits, the language changes and the empty halves are only here.
        server_log = args.out.with_name(args.out.stem.split(".")[0]
                                        + ".server.log")
        server_log.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(server_log, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        server_logger = logging.getLogger("server")
        server_logger.addHandler(handler)
        server_logger.setLevel(logging.INFO)
        print(f"  server log at {server_log}")
        # Written by the server package itself: the pod does not need the
        # client package to replay a meeting.
        recorder = LogWriter(args.out, clock=lambda: clock[0])
        print("\nReplaying:")
        started = time.perf_counter()
        try:
            session = replay(pcm, loaded, recorder, clock)
        finally:
            recorder.close()
            server_logger.removeHandler(handler)
            handler.close()
        print(f"  done in {time.perf_counter() - started:.1f} s; "
              f"log at {args.out}")

        check(session, args.out, audio_seconds, args, report)

        if args.baseline:
            print("\nSide by side (the baselines cover the whole meeting; "
                  "compare rates, not counts, on a --limit-seconds run):")
            print(table([str(args.out)] + [str(p) for p in args.baseline]))

        print("\n  Read the sentences in the log above. Nothing here can tell "
              "you whether they are right.")
        print("\n" + "=" * 72)
        if report.failed:
            print(f"RESULT: FAIL ({len(report.failed)} check(s) failed)")
            for failed in report.failed:
                print(f"  - {failed.name}: {failed.detail}")
            print("=" * 72)
            return 1
        print(f"RESULT: PASS ({len(report.checks)} checks)")
        print("=" * 72)
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"\nRESULT: FAIL - {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
