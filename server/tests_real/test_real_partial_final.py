"""REAL TEST - why a committed sentence sometimes reads worse than the
running text that preceded it.

MUST RUN ON: the GPU Server pod.
DO NOT RUN ON: the Dev PC agent loop.

The running text and the sentence are not a draft and its revision. They are
two different decodes:

===============  ==============================  =========================
                 running text                    sentence
===============  ==============================  =========================
audio            last 4 s, raw PCM               whole utterance, shaped
beam             1 (greedy)                      5 (beam search)
trailing audio   stops while speech continues    carries the VAD hangover
===============  ==============================  =========================

So the sentence can lose what the running text had. This script replays one
recorded meeting through the real VAD and buffer manager, decodes the running
text once, then decodes every sentence under the variants asked for, counts
the four ways they drift apart, and records the score of every segment the
decoder returned.

    baseline    what the server does today
    beam1       beam search off for the sentence as well
    trim        the VAD's trailing hangover cut before the ASR sees it
    beam1+trim  both

Only ``baseline`` runs unless ``--variants`` says otherwise, because the
other three have been measured on thirty minutes of real meeting and none of
them was the fault. What was the fault is which segments the guards refuse,
and that is not a variant here: the scores are recorded, so
``server/analysis/report.py`` replays any guard rule over them exactly,
without a GPU and without decoding anything twice.

Decoding the running text once rather than per variant is what makes this
affordable, and ``--reuse-partials`` skips even that: the running text is
roughly four fifths of the decoding and it does not change when the sentence
path does.

What it cannot do is say which text is *correct*. It counts, ranks, and
prints the worst cases with every variant side by side for you to read.

Usage
-----
    python3.11 server/tests_real/test_real_partial_final.py \
        --wav recordings/meeting_30min.wav \
        --limit-seconds 120 \
        --out /tmp/drift_smoke.json

    python3.11 server/tests_real/test_real_partial_final.py \
        --wav recordings/meeting_30min.wav \
        --reuse-partials /tmp/drift_full.json \
        --out /tmp/drift_scored.json

Run it with ``--limit-seconds 120`` first. A full pass on thirty minutes is
roughly five minutes of H100 time for the baseline alone, or one minute with
``--reuse-partials``.

The WAV must be 16 kHz mono 16-bit - the format the client streams. Convert
a meeting recording with::

    ffmpeg -i meeting.m4a -ac 1 -ar 16000 -sample_fmt s16 meeting_16k.wav
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import server.pipeline.asr as asr_module                                # noqa: E402
from server.analysis.drift import compare                               # noqa: E402
from server.config import (                                             # noqa: E402
    ASR_BEAM_SIZE_FINAL,
    CHANNELS,
    CHUNK_BYTES,
    PARTIAL_WINDOW_SECONDS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    VAD_MIN_SILENCE_MS,
)
from server.pipeline.asr import AsrError, Transcriber                   # noqa: E402
from server.pipeline.buffer import (                                    # noqa: E402
    BufferManager,
    FinalizeReason,
    PartialWindow,
    bytes_to_ms,
    ms_to_bytes,
)
from server.pipeline.lid import LanguageIdError, LanguageIdentifier     # noqa: E402
from server.pipeline.overlap import OverlapError, OverlapResolver       # noqa: E402
from server.pipeline.vad import SileroVAD, VADError, VADSegmenter       # noqa: E402

#: How much trailing silence a finalised utterance carries. The VAD keeps
#: forwarding audio through its whole silence hangover and only stops on the
#: frame that fires SPEECH_END, so every sentence ends in roughly this much
#: quiet - and Whisper does not answer quiet with nothing.
HANGOVER_MS = VAD_MIN_SILENCE_MS - 32

#: Never trim a sentence below this. A short utterance is mostly hangover by
#: proportion, and cutting into it would measure the trim, not the tail.
MIN_KEPT_MS = 400.0


@dataclass(frozen=True)
class Variant:
    """One way of decoding the committed sentence."""

    name: str
    beam: int
    trim_ms: float

    @property
    def label(self) -> str:
        return f"{self.name} (beam={self.beam}, trim={self.trim_ms:.0f}ms)"


@dataclass
class Decoded:
    """One sentence under one variant.

    ``pieces`` is every segment the decoder returned, in order, with the three
    numbers the guards judge it by and the verdict it got. Recording them is
    what lets a different threshold or a different rule be answered later
    without a GPU - see ``server/analysis/guards.py``. It is a few hundred
    bytes a sentence and it has already saved one full re-run.
    """

    text: str
    seconds: float
    audio_ms: float
    dropped: tuple = ()
    pieces: list = field(default_factory=list)


@dataclass
class Case:
    """One utterance: its running text, and the sentence under each variant."""

    index: int
    start_ms: float
    end_ms: float
    reason: str
    continues_previous: bool
    lang_code: str
    partial_text: str = ""
    partial_end_ms: float = 0.0
    partial_count: int = 0
    finals: dict = field(default_factory=dict)          # variant -> Decoded
    drift: dict = field(default_factory=dict)           # variant -> Drift

    @property
    def audio_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def extra_audio_ms(self) -> float:
        """Audio the sentence covers that the last running text never saw."""
        return max(self.end_ms - self.partial_end_ms, 0.0)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def read_pcm(path: Path, limit_seconds: float) -> bytes:
    with wave.open(str(path), "rb") as wav:
        actual = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
        if actual != (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH):
            raise ValueError(
                f"{path.name} is {actual[0]} Hz / {actual[1]} ch / "
                f"{actual[2] * 8}-bit; this needs 16000 Hz / mono / 16-bit. "
                f"Convert it: ffmpeg -i {path.name} -ac 1 -ar 16000 "
                f"-sample_fmt s16 {path.stem}_16k.wav"
            )
        pcm = wav.readframes(wav.getnframes())
    if limit_seconds > 0:
        pcm = pcm[: ms_to_bytes(limit_seconds * 1000.0)]
    return pcm


def replay(pcm: bytes, vad: SileroVAD) -> list:
    """Drive the real VAD and buffer manager, in the order the server sees.

    Chunked at exactly the size the client streams, because the partial
    cadence depends on it: the buffer asks for 600 ms between running texts
    and audio arrives in 200 ms steps, so one actually appears every 800 ms.
    """
    vad.reset()
    segmenter = VADSegmenter(vad=vad)
    buffer = BufferManager()
    events: list = []

    for offset in range(0, len(pcm), CHUNK_BYTES):
        result = buffer.push(segmenter.push(pcm[offset : offset + CHUNK_BYTES]))
        for utterance in result.finals:
            events.append(("final", utterance))
        if result.partial is not None:
            events.append(("partial", result.partial))

    segmenter.close()
    for utterance in buffer.flush(FinalizeReason.END_OF_STREAM).finals:
        events.append(("final", utterance))
    return events


class RecordingDecoder:
    """Keeps the segments the model returned, in the order it returned them.

    ``Transcript`` reports what was kept and what was refused, but not the
    sequence they arrived in, and the sentence's text depends on that order.
    Wrapping the decoder is the way to have both without changing the ASR
    stage to serve a measurement.
    """

    def __init__(self, inner):
        self.inner = inner
        self.last: list = []

    @property
    def source(self) -> str:
        return getattr(self.inner, "source", "unknown decoder")

    def decode(self, samples, lang_code, beam_size, prompt=None):
        pieces, detected = self.inner.decode(samples, lang_code, beam_size,
                                             prompt)
        self.last = list(pieces)
        return pieces, detected


def record_pieces(recorder: RecordingDecoder,
                  transcriber: Transcriber) -> list:
    """The segments of the sentence just decoded, with scores and verdicts."""
    return [
        {
            "text": piece.text,
            "avg_logprob": piece.avg_logprob,
            "no_speech_prob": piece.no_speech_prob,
            "compression_ratio": piece.compression_ratio,
            "verdict": transcriber._refuse(piece) or "kept",
        }
        for piece in recorder.last
    ]


def trim_tail(pcm: bytes, trim_ms: float) -> bytes:
    """Cut a fixed amount off the end of an utterance.

    Fixed rather than energy-detected on purpose. This is a probe, not the
    fix: a fixed cut has one number behind it and cannot be confounded by a
    detector's own threshold. If the numbers say the hangover is what feeds
    the invented tails, the fix that ships can be smarter than this.
    """
    if trim_ms <= 0:
        return pcm
    keep = len(pcm) - ms_to_bytes(trim_ms)
    if keep < ms_to_bytes(MIN_KEPT_MS):
        keep = min(len(pcm), ms_to_bytes(MIN_KEPT_MS))
    return pcm[:keep]


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def language_pass(events: list,
                  identifier: Optional[LanguageIdentifier]) -> dict:
    """The language the server would force, per event, in stream order.

    Reproduces ``ServerSession._language_for``: a sentence the LID cannot
    decide falls back to the last language the meeting was confidently in,
    and only a *sentence* updates that memory - the running text reads it but
    never writes it.
    """
    forced: dict = {}
    last_language = ""
    if identifier is None:
        return forced
    identifier.reset()

    for kind, item in events:
        if kind == "final":
            decision = identifier.identify(item.pcm)
            if decision.known:
                last_language = decision.lang_code
            forced[(kind, item.index, item.start_ms)] = (
                decision.lang_code if decision.known else last_language)
        else:
            window = item.tail(PARTIAL_WINDOW_SECONDS)
            decision = identifier.identify(window.pcm)
            forced[(kind, item.index, item.start_ms)] = (
                decision.lang_code if decision.known else last_language)
    return forced


def decode_partials(events: list, transcriber: Transcriber,
                    forced: dict) -> tuple:
    """Every running text, once. Greedy, raw audio, four-second tail.

    Only the last one per utterance is kept: it is the text on screen at the
    moment the sentence replaces it, which is the comparison that matters.
    """
    last: dict = {}
    counts: dict = {}
    spent = 0.0
    total = 0

    for kind, item in events:
        if kind != "partial":
            continue
        window = item.tail(PARTIAL_WINDOW_SECONDS)
        lang = forced.get((kind, item.index, item.start_ms), "")
        started = time.perf_counter()
        transcript = transcriber.transcribe(window.pcm, lang, is_final=False)
        spent += time.perf_counter() - started
        total += 1
        counts[item.index] = counts.get(item.index, 0) + 1
        if transcript.has_text:
            last[item.index] = (item, transcript.text)
    return {"last": last, "counts": counts}, spent, total


def decode_finals(events: list, transcriber: Transcriber, forced: dict,
                  variant: Variant, resolver: Optional[OverlapResolver],
                  recorder: Optional[RecordingDecoder] = None) -> tuple:
    """Every sentence, under one variant.

    ``ASR_BEAM_SIZE_FINAL`` is read from the module at call time, so the beam
    is swapped there. That is a liberty a measurement script may take and the
    server may not; it keeps the decode path itself identical to production.
    """
    original = asr_module.ASR_BEAM_SIZE_FINAL
    asr_module.ASR_BEAM_SIZE_FINAL = variant.beam
    decoded: dict = {}
    spent = 0.0
    try:
        for kind, item in events:
            if kind != "final":
                continue
            audio = trim_tail(item.pcm, variant.trim_ms)
            if resolver is not None:
                audio = resolver.resolve(audio).pcm
            lang = forced.get((kind, item.index, item.start_ms), "")
            started = time.perf_counter()
            transcript = transcriber.transcribe(audio, lang, is_final=True)
            elapsed = time.perf_counter() - started
            spent += elapsed
            decoded[item.index] = Decoded(
                text=transcript.text,
                seconds=elapsed,
                audio_ms=bytes_to_ms(len(audio)),
                dropped=tuple(reason for _piece, reason in transcript.dropped),
                pieces=(record_pieces(recorder, transcriber)
                        if recorder is not None else []),
            )
    finally:
        asr_module.ASR_BEAM_SIZE_FINAL = original
    return decoded, spent


def reuse_partials(path: Path) -> dict:
    """The running texts from an earlier run over the same recording.

    They cost four fifths of the decoding and they do not change between
    variants of the sentence path, so re-deciding a threshold should not have
    to pay for them again. The utterance boundaries come from the VAD and the
    buffer manager, which are deterministic, so the indexes line up as long as
    it is the same recording - and if it is not, the mismatch shows up as
    sentences with no running text rather than as a silently wrong pairing.
    """
    earlier = json.loads(path.read_text(encoding="utf-8"))
    return {
        case["index"]: (case["partial_text"], case["partial_end_ms"]
                        if "partial_end_ms" in case
                        else case["end_ms"] - case["extra_audio_ms"],
                        case["partial_count"])
        for case in earlier["cases"] if case["partial_text"]
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
FLAGS = ("rewrite", "tail_invention", "latin_lost", "repetition", "final_empty")


def summarise(cases: list, variant: Variant) -> dict:
    drifts = [case.drift[variant.name] for case in cases
              if variant.name in case.drift]
    rewrites = [drift.rewrite for drift in drifts]
    counts = {flag: sum(1 for drift in drifts if flag in drift.flags)
              for flag in FLAGS}
    lost = sorted({word for drift in drifts for word in drift.lost_latin})
    return {
        "variant": variant.name,
        "beam": variant.beam,
        "trim_ms": variant.trim_ms,
        "compared": len(drifts),
        "mean_rewrite": statistics.fmean(rewrites) if rewrites else 0.0,
        "median_rewrite": statistics.median(rewrites) if rewrites else 0.0,
        "clean": sum(1 for drift in drifts if not drift.flags),
        "flagged": sum(1 for drift in drifts if drift.flags),
        "counts": counts,
        "lost_words": lost,
    }


def print_summary(rows: list, baseline: str) -> None:
    header = (f"{'variant':<12} {'cmp':>5} {'clean':>6} {'flag':>5} "
              f"{'rewrite':>8} " + " ".join(f"{flag[:9]:>10}" for flag in FLAGS))
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['variant']:<12} {row['compared']:>5} {row['clean']:>6} "
              f"{row['flagged']:>5} {row['mean_rewrite']:>8.3f} "
              + " ".join(f"{row['counts'][flag]:>10}" for flag in FLAGS))

    base = next(row for row in rows if row["variant"] == baseline)
    print("\nAgainst the baseline:")
    for row in rows:
        if row["variant"] == baseline:
            continue
        delta = row["flagged"] - base["flagged"]
        print(f"  {row['variant']:<12} {delta:+d} flagged sentences "
              f"({base['flagged']} -> {row['flagged']}), "
              f"mean rewrite {base['mean_rewrite']:.3f} -> "
              f"{row['mean_rewrite']:.3f}")


def print_worst(cases: list, variants: list, baseline: str, top: int) -> None:
    ranked = sorted(
        (case for case in cases if baseline in case.drift),
        key=lambda case: (-len(case.drift[baseline].flags),
                          -case.drift[baseline].rewrite),
    )[:top]
    if not ranked:
        return
    print(f"\nThe {len(ranked)} worst sentences under {baseline}, "
          f"every variant side by side:\n")
    for case in ranked:
        drift = case.drift[baseline]
        print(f"  #{case.index} "
              f"{case.start_ms / 1000:.1f}-{case.end_ms / 1000:.1f}s "
              f"[{case.lang_code or 'auto'}] {case.reason}"
              f"{' (continues)' if case.continues_previous else ''}"
              f"  flags={','.join(drift.flags) or '-'} "
              f"rewrite={drift.rewrite:.2f} "
              f"extra_audio={case.extra_audio_ms:.0f}ms")
        print(f"      partial   ({case.partial_count} of them) "
              f"{case.partial_text!r}")
        for variant in variants:
            decoded = case.finals.get(variant.name)
            if decoded is None:
                continue
            marker = "*" if variant.name == baseline else " "
            print(f"    {marker} {variant.name:<10} {decoded.text!r}")
        print()


# ---------------------------------------------------------------------------
def build_variants(trim_ms: float) -> list:
    return [
        Variant("baseline", ASR_BEAM_SIZE_FINAL, 0.0),
        Variant("beam1", 1, 0.0),
        Variant("trim", ASR_BEAM_SIZE_FINAL, trim_ms),
        Variant("beam1+trim", 1, trim_ms),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare committed sentences against the running text "
                    "that preceded them, under four decode variants.")
    parser.add_argument("--wav", type=Path, required=True,
                        help="16 kHz mono 16-bit recording of a real meeting")
    parser.add_argument("--limit-seconds", type=float, default=0.0,
                        help="only the first N seconds; 0 means the whole file")
    parser.add_argument("--trim-ms", type=float, default=float(HANGOVER_MS),
                        help=f"hangover to cut in the trim variants "
                             f"(default {HANGOVER_MS})")
    parser.add_argument("--top", type=int, default=15,
                        help="how many worst sentences to print")
    parser.add_argument("--out", type=Path, default=None,
                        help="write every sentence and its variants as JSON")
    parser.add_argument("--variants", default="baseline",
                        help="comma-separated subset of baseline, beam1, "
                             "trim, beam1+trim, or 'all'. Guard rules are not "
                             "variants - they are replayed offline from the "
                             "recorded scores by server.analysis.report")
    parser.add_argument("--reuse-partials", type=Path, default=None,
                        help="take the running texts from an earlier run's "
                             "JSON over the same recording instead of "
                             "decoding them again")
    parser.add_argument("--no-lid", action="store_true",
                        help="skip language ID and let Whisper detect")
    parser.add_argument("--no-overlap", action="store_true",
                        help="skip the overlap resolver on the sentence path")
    parser.add_argument("--model", default="", help="Whisper checkpoint")
    parser.add_argument("--device", default="", help='"cuda" or "cpu"')
    args = parser.parse_args()

    variants = build_variants(args.trim_ms)
    if args.variants and args.variants != "all":
        wanted = {name.strip() for name in args.variants.split(",")}
        variants = [variant for variant in variants if variant.name in wanted]
        if not variants:
            print(f"No variant matched {args.variants!r}")
            return 2
    baseline = variants[0].name

    print(f"Reading {args.wav} ...")
    try:
        pcm = read_pcm(args.wav, args.limit_seconds)
    except (OSError, ValueError, wave.Error) as exc:
        print(f"  cannot read it: {exc}")
        return 2
    print(f"  {bytes_to_ms(len(pcm)) / 1000:.1f} s of audio")

    try:
        vad = SileroVAD()
    except VADError as exc:
        print(f"  Silero VAD unavailable: {exc}")
        return 2

    started = time.perf_counter()
    events = replay(pcm, vad)
    finals = [item for kind, item in events if kind == "final"]
    partials = [item for kind, item in events if kind == "partial"]
    print(f"  {len(finals)} sentences, {len(partials)} running texts, "
          f"replayed in {time.perf_counter() - started:.1f} s")
    if not finals:
        print("  nothing to compare - the VAD found no speech")
        return 1

    identifier = None
    if not args.no_lid:
        try:
            identifier = LanguageIdentifier()
        except LanguageIdError as exc:
            print(f"  language ID unavailable, carrying on without it: {exc}")

    resolver = None
    if not args.no_overlap:
        try:
            resolver = OverlapResolver()
        except OverlapError as exc:
            print(f"  overlap resolver unavailable, carrying on without: {exc}")

    try:
        recorder = RecordingDecoder(
            asr_module.WhisperDecoder(model_id=args.model, device=args.device))
        transcriber = Transcriber(decoder=recorder)
    except AsrError as exc:
        print(f"  Whisper unavailable: {exc}")
        return 2

    print("\nForcing the language the server would force ...")
    started = time.perf_counter()
    forced = language_pass(events, identifier)
    print(f"  {len(forced)} decisions in {time.perf_counter() - started:.1f} s")

    earlier: dict = {}
    partial_seconds = 0.0
    partial_state = {"last": {}, "counts": {}}
    if args.reuse_partials:
        try:
            earlier = reuse_partials(args.reuse_partials)
        except (OSError, ValueError, KeyError) as exc:
            print(f"  cannot reuse {args.reuse_partials}: {exc}")
            return 2
        print(f"\nReusing {len(earlier)} running texts from "
              f"{args.reuse_partials}")
    else:
        print("\nDecoding the running text, once ...")
        partial_state, partial_seconds, partial_count = decode_partials(
            events, transcriber, forced)
        print(f"  {partial_count} decodes in {partial_seconds:.1f} s")

    cases: list = []
    by_index: dict = {}
    for utterance in finals:
        window_and_text = partial_state["last"].get(utterance.index)
        case = Case(
            index=utterance.index,
            start_ms=utterance.start_ms,
            end_ms=utterance.end_ms,
            reason=utterance.reason.value,
            continues_previous=utterance.continues_previous,
            lang_code=forced.get(("final", utterance.index,
                                  utterance.start_ms), ""),
            partial_count=partial_state["counts"].get(utterance.index, 0),
        )
        if window_and_text is not None:
            window, text = window_and_text
            case.partial_text = text
            case.partial_end_ms = window.end_ms
        elif utterance.index in earlier:
            text, end_ms, count = earlier[utterance.index]
            case.partial_text = text
            case.partial_end_ms = end_ms
            case.partial_count = count
        cases.append(case)
        by_index[utterance.index] = case

    final_seconds: dict = {}
    for variant in variants:
        print(f"\nDecoding the sentences: {variant.label} ...")
        decoded, spent = decode_finals(events, transcriber, forced, variant,
                                       resolver, recorder)
        final_seconds[variant.name] = spent
        print(f"  {len(decoded)} decodes in {spent:.1f} s")
        for index, result in decoded.items():
            case = by_index[index]
            case.finals[variant.name] = result
            if not case.partial_text:
                # No running text ever appeared for this utterance - too
                # short, or every one of them was refused. Nothing to compare
                # against, and counting it as clean would flatter the numbers.
                continue
            case.drift[variant.name] = compare(
                final=result.text,
                partial=case.partial_text,
                extra_audio_ms=case.extra_audio_ms,
                final_audio_ms=case.audio_ms,
            )

    rows = [summarise(cases, variant) for variant in variants]
    compared = rows[0]["compared"] if rows else 0
    print(f"\n{compared} of {len(cases)} sentences had a running text to "
          f"compare against.")
    print_summary(rows, baseline)
    print_worst(cases, variants, baseline, args.top)

    print("\nDecode cost:")
    print(f"  running text  {partial_seconds:>8.1f} s")
    for variant in variants:
        print(f"  {variant.name:<13} {final_seconds[variant.name]:>8.1f} s")

    if args.out:
        payload = {
            "wav": str(args.wav),
            "audio_seconds": bytes_to_ms(len(pcm)) / 1000.0,
            "variants": [asdict(variant) for variant in variants],
            "summary": rows,
            "partial_seconds": partial_seconds,
            "final_seconds": final_seconds,
            "cases": [
                {
                    "index": case.index,
                    "start_ms": case.start_ms,
                    "end_ms": case.end_ms,
                    "reason": case.reason,
                    "continues_previous": case.continues_previous,
                    "lang_code": case.lang_code,
                    "partial_text": case.partial_text,
                    "partial_end_ms": case.partial_end_ms,
                    "partial_count": case.partial_count,
                    "extra_audio_ms": case.extra_audio_ms,
                    "finals": {name: asdict(value)
                               for name, value in case.finals.items()},
                    "drift": {name: asdict(value)
                              for name, value in case.drift.items()},
                }
                for case in cases
            ],
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\nEvery sentence written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
