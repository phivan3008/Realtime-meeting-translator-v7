"""REAL TEST - speaker voiceprints on real recorded voices.

MUST RUN ON: the GPU Server pod.
DO NOT RUN ON: the Dev PC agent loop.

``DESIGN.md`` section 3.5 labels each sentence with who said it, by cosine
similarity between ECAPA-TDNN voiceprints. Everything rests on one number -
the threshold - so this measures the two distributions it has to separate:

* **same speaker**: every pair of sentences inside one recording
* **different speakers**: every pair across two recordings

Every ``--voice`` must be ONE person
------------------------------------
That is not a detail, it is the whole experiment.  A recording with three
people in it has no same-speaker pairs to measure, and the numbers that come
out of one mean nothing at all.

Worse, it also breaks the fallback diagnostic.  The VAD only closes a segment
after 500 ms of silence, while conversational turns are usually shorter than
that, so in a multi-speaker recording a single seven-second "sentence" can
easily hold two people - and then even the two halves of one sentence are not
the same speaker.  The first version of this test leaned on those halves as
ground truth and was measuring the very thing it was trying to rule out.

So: one voice per file, several files, at least two of them different people.
A minute of one person talking is enough.

Usage
-----
    python3.11 server/tests_real/test_real_diarization.py         --voice recordings/alice.wav         --voice recordings/bob.wav
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (  # noqa: E402
    CHANNELS,
    CHUNK_BYTES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    SPEAKER_MATCH_THRESHOLD,
    SPEAKER_MIN_DURATION_MS,
)
from server.pipeline.buffer import BufferManager, Utterance  # noqa: E402
from server.pipeline.diarization import (  # noqa: E402
    DiarizationError,
    EcapaEmbedder,
    SpeakerIdentifier,
    SpeakerRegistry,
    cosine_similarity,
)
from server.pipeline.overlap import OverlapResolver, PedalboardProcessor  # noqa: E402
from server.pipeline.vad import SileroVAD, VADError, VADSegmenter  # noqa: E402

# Embedding one sentence must stay cheap next to transcribing it.
MAX_EMBED_RATIO = 0.05


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
class Voice:
    """One recording, cut into sentences and embedded."""

    path: Path
    utterances: list[Utterance] = field(default_factory=list)
    raw_utterances: list[Utterance] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)
    raw_embeddings: list[np.ndarray] = field(default_factory=list)
    embed_seconds: float = 0.0

    @property
    def audio_seconds(self) -> float:
        return sum(u.duration_ms for u in self.utterances) / 1000.0

    @property
    def embed_ratio(self) -> float:
        return self.embed_seconds / self.audio_seconds if self.audio_seconds else 0.0


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
                 resolver: Optional[OverlapResolver]) -> list[Utterance]:
    """Run the real chain, so the voiceprints see what the server would."""
    pcm = read_pcm(path)
    vad.reset()
    segmenter = VADSegmenter(vad=vad)
    buffer = BufferManager()
    finals: list[Utterance] = []
    for offset in range(0, len(pcm), CHUNK_BYTES):
        finals += buffer.push(segmenter.push(pcm[offset : offset + CHUNK_BYTES])).finals
    segmenter.close()
    finals += buffer.flush().finals
    if resolver is None:
        return finals
    return [
        Utterance(index=u.index, pcm=resolver.resolve(u.pcm).pcm,
                  start_ms=u.start_ms, reason=u.reason,
                  continues_previous=u.continues_previous)
        for u in finals
    ]


def embed_all(path: Path, embedder: EcapaEmbedder, vad: SileroVAD,
              resolver: OverlapResolver) -> Voice:
    raw = sentences_of(path, vad, resolver=None)
    shaped = sentences_of(path, vad, resolver)
    keep = [i for i, u in enumerate(raw)
            if u.duration_ms >= SPEAKER_MIN_DURATION_MS]

    voice = Voice(path=path)
    voice.utterances = [shaped[i] for i in keep]
    voice.raw_utterances = [raw[i] for i in keep]
    started = time.perf_counter()
    voice.embeddings = [embedder.embed(u.pcm) for u in voice.utterances]
    voice.embed_seconds = time.perf_counter() - started
    voice.raw_embeddings = [embedder.embed(u.pcm) for u in voice.raw_utterances]
    return voice


def half_similarities(utterances: list[Utterance],
                      embedder: EcapaEmbedder) -> list[float]:
    """Cosine between the two halves of each sentence.

    The strongest same-speaker ground truth available without a labelled
    recording: whoever said the first half of a sentence said the second.
    If these score low, the voiceprints are wrong; if they score high, low
    scores between sentences mean the sentences really are different people.
    """
    scores = []
    for utterance in utterances:
        if utterance.duration_ms < 2 * SPEAKER_MIN_DURATION_MS:
            continue
        middle = len(utterance.pcm) // 2 // SAMPLE_WIDTH * SAMPLE_WIDTH
        scores.append(cosine_similarity(embedder.embed(utterance.pcm[:middle]),
                                        embedder.embed(utterance.pcm[middle:])))
    return scores


def describe(voice: Voice) -> None:
    print(f"\n  {voice.path.name}: {len(voice.utterances)} sentence(s) long "
          f"enough to identify, {voice.audio_seconds:.1f} s of speech")
    if voice.embeddings:
        print(f"    voiceprint dimension {voice.embeddings[0].size}, "
              f"embedded in {voice.embed_seconds * 1000:.0f} ms "
              f"(ratio {voice.embed_ratio:.4f})")


def pairwise(embeddings: list[np.ndarray]) -> list[float]:
    return [cosine_similarity(a, b)
            for a, b in itertools.combinations(embeddings, 2)]


def crosswise(left: list[np.ndarray], right: list[np.ndarray]) -> list[float]:
    return [cosine_similarity(a, b) for a in left for b in right]


def summarise(name: str, scores: list[float]) -> None:
    if not scores:
        print(f"    {name}: none")
        return
    print(f"    {name}: n={len(scores)} "
          f"min {min(scores):.3f}  median {statistics.median(scores):.3f}  "
          f"max {max(scores):.3f}")


# ---------------------------------------------------------------------------
# Diagnostics
#
# When sentence-to-sentence similarity comes out low there are three possible
# reasons and no way to tell them apart from that number alone: the embedder
# is being called wrongly, the overlap resolver is damaging the audio before
# it is embedded, or the recording simply holds more than one speaker. These
# measure all three at once rather than picking one to guess at.
# ---------------------------------------------------------------------------
def check_deterministic(voice: Voice, embedder: EcapaEmbedder,
                        report: Report) -> None:
    """The same audio twice must give the same voiceprint."""
    if not voice.utterances:
        return
    pcm = voice.utterances[0].pcm
    score = cosine_similarity(embedder.embed(pcm), embedder.embed(pcm))
    report.add("The same audio gives the same voiceprint", score > 0.999,
               f"cosine {score:.4f}")


def check_halves(same_utterance: list[float], report: Report) -> bool:
    """Both halves of one sentence are the same person, by definition."""
    if not same_utterance:
        print("    (no sentence long enough to split in half)")
        return True
    median = statistics.median(same_utterance)
    passed = report.add(
        "The two halves of one sentence match each other",
        median > SPEAKER_MATCH_THRESHOLD,
        f"median {median:.3f} > {SPEAKER_MATCH_THRESHOLD}, "
        f"worst {min(same_utterance):.3f}",
    )
    if not passed:
        print("    -> the voiceprints themselves are wrong. Nothing about the "
              "threshold or the recordings can be concluded until this passes.")
    return passed


def check_shaping(shaped: list[float], raw: list[float],
                  report: Report) -> None:
    """Confirm the pipeline embeds the audio that scores better.

    Measured once, gating before embedding cost 0.06 of same-speaker cosine,
    so the session embeds the raw utterance and lets the overlap resolver
    serve the ASR alone. This check keeps that decision honest: if shaped
    audio ever starts scoring better, the wiring should follow.
    """
    if not shaped or not raw:
        return
    shaped_median, raw_median = statistics.median(shaped), statistics.median(raw)
    print(f"    raw median {raw_median:.3f} vs shaped median "
          f"{shaped_median:.3f} ({raw_median - shaped_median:+.3f} for raw)")
    report.add(
        "Raw audio is still the better source for voiceprints",
        raw_median >= shaped_median,
        f"raw {raw_median:.3f}, shaped {shaped_median:.3f} - if this flips, "
        "embed the shaped audio in server/net/session.py instead",
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_embeddings(voice: Voice, report: Report) -> None:
    name = voice.path.name
    report.add(f"{name} produced sentences to identify",
               len(voice.embeddings) >= 2,
               f"{len(voice.embeddings)} voiceprint(s)")
    if not voice.embeddings:
        return
    sizes = {e.size for e in voice.embeddings}
    report.add(f"{name} voiceprints all have one dimension", len(sizes) == 1,
               f"{sorted(sizes)}")
    report.add(f"{name} voiceprints are finite",
               all(bool(np.all(np.isfinite(e))) for e in voice.embeddings), "")
    report.add(f"{name} embedding is cheap", voice.embed_ratio < MAX_EMBED_RATIO,
               f"ratio {voice.embed_ratio:.4f} < {MAX_EMBED_RATIO}")


def check_separation(same: list[float], different: list[float],
                     report: Report) -> None:
    if not same:
        report.add("There are same-speaker pairs to measure", False, "none")
        return
    report.add("A voice matches itself above the threshold",
               statistics.median(same) > SPEAKER_MATCH_THRESHOLD,
               f"median {statistics.median(same):.3f} > "
               f"{SPEAKER_MATCH_THRESHOLD}")

    if not different:
        print("    (no --other recording: the different-speaker side is "
              "unproven, and with it the threshold)")
        return

    report.add("Two different voices score below the threshold",
               max(different) < SPEAKER_MATCH_THRESHOLD,
               f"worst {max(different):.3f} < {SPEAKER_MATCH_THRESHOLD}")
    gap = min(same) - max(different)
    report.add("The two distributions do not overlap", gap > 0.0,
               f"worst same {min(same):.3f} vs best different "
               f"{max(different):.3f}, gap {gap:+.3f}")
    if gap > 0.0:
        print(f"\n  A threshold anywhere in ({max(different):.3f}, "
              f"{min(same):.3f}) separates them; the midpoint is "
              f"{(max(different) + min(same)) / 2:.3f}, "
              f"and SPEAKER_MATCH_THRESHOLD is {SPEAKER_MATCH_THRESHOLD}.")


def check_labelling(voices: list[Voice], report: Report) -> None:
    """What the registry actually does, not just what the numbers imply."""
    everything: list[np.ndarray] = []
    for voice in voices:
        everything += voice.embeddings
    identifier = SpeakerIdentifier(
        embedder=_Replay(everything),
        registry=SpeakerRegistry(),
        min_duration_ms=0,
    )

    print()
    labels_per_voice = []
    for voice in voices:
        labels = [identifier.identify(u.pcm).speaker_id for u in voice.utterances]
        labels_per_voice.append(labels)
        print(f"  labels for {voice.path.name}: {labels}")

    for voice, labels in zip(voices, labels_per_voice):
        report.add(f"{voice.path.name} is labelled as one speaker",
                   len(set(labels)) == 1, f"{sorted(set(labels))}")

    seen: set[str] = set()
    overlapping = False
    for labels in labels_per_voice:
        if seen & set(labels):
            overlapping = True
        seen |= set(labels)
    if len(voices) >= 2:
        report.add("Different recordings get different labels", not overlapping,
                   f"{[sorted(set(l)) for l in labels_per_voice]}")
        report.add(f"Exactly {len(voices)} speakers were found",
                   identifier.registry.count == len(voices),
                   f"{identifier.registry.count} speaker(s)")


class _Replay:
    """Hands back the voiceprints already computed, in order."""

    def __init__(self, embeddings: list[np.ndarray]):
        self.embeddings = embeddings
        self.calls = 0

    def embed(self, pcm: bytes) -> np.ndarray:
        value = self.embeddings[self.calls]
        self.calls += 1
        return value


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", type=Path, action="append", required=True,
                        help="WAV holding exactly ONE speaker; repeat for "
                             "each person. Two or more are needed before the "
                             "threshold means anything.")
    parser.add_argument("--device", default="", help='"cuda" or "cpu"')
    args = parser.parse_args()

    print("=" * 72)
    print("REAL TEST - SPEAKER DIARIZATION (server pipeline, step 5)")
    print(f"match above cosine {SPEAKER_MATCH_THRESHOLD}, "
          f"sentences under {SPEAKER_MIN_DURATION_MS} ms are not identified")
    print("=" * 72)

    report = Report()
    try:
        print("\n  Loading the speaker embedding model ...")
        started = time.perf_counter()
        embedder = EcapaEmbedder(device=args.device)
        print(f"  Ready in {time.perf_counter() - started:.1f} s "
              f"from {embedder.source}")
        report.add("The embedding model loads on the pod", True, embedder.source)

        vad = SileroVAD()
        resolver = OverlapResolver(processor=PedalboardProcessor())

        voices = [embed_all(path, embedder, vad, resolver) for path in args.voice]
        for voice in voices:
            describe(voice)

        same: list[float] = []
        halves: list[float] = []
        raw_same: list[float] = []
        for voice in voices:
            same += pairwise(voice.embeddings)
            raw_same += pairwise(voice.raw_embeddings)
            halves += half_similarities(voice.utterances, embedder)
        different: list[float] = []
        for left, right in itertools.combinations(voices, 2):
            different += crosswise(left.embeddings, right.embeddings)

        print()
        print("  Cosine similarity:")
        summarise("two halves of one sentence", halves)
        summarise("sentence to sentence, same voice", same)
        summarise("sentence to sentence, across voices", different)

        print()
        print("Diagnostics:")
        check_deterministic(voices[0], embedder, report)
        check_halves(halves, report)
        check_shaping(same, raw_same, report)

        print()
        print("Checks:")
        for voice in voices:
            check_embeddings(voice, report)
        report.add("At least two voices to compare", len(voices) >= 2,
                   f"{len(voices)} recording(s)")
        check_separation(same, different, report)
        check_labelling(voices, report)

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

    except (DiarizationError, VADError, ValueError, FileNotFoundError) as exc:
        print(f"\nRESULT: FAIL - {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
