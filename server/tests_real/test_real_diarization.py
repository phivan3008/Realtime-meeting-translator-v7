"""REAL TEST - speaker voiceprints on real recorded voices.

MUST RUN ON: the GPU Server pod.
DO NOT RUN ON: the Dev PC agent loop.

``DESIGN.md`` section 3.5 labels each sentence with who said it, by cosine
similarity between ECAPA-TDNN voiceprints. Everything about that rests on one
number - the threshold - and a threshold picked by taste is a guess. This
measures the two distributions it has to separate:

* **same speaker**: every pair of sentences from one recording
* **different speakers**: every pair across two recordings of different people

If those two clouds overlap, no threshold works and the design needs
rethinking rather than tuning. If they separate, the gap tells you where the
threshold belongs, and the test prints the midpoint.

Sentences come from the real chain - VAD, buffer manager, noise filter,
overlap resolver - so the voiceprints are taken from exactly the audio the
live server would embed, bleed already attenuated.

Before any of that means anything, three diagnostics run. A low
sentence-to-sentence score has three possible causes and the number alone
cannot tell them apart: the embedder is being called wrongly, the overlap
resolver is damaging the audio before it is embedded, or the recording holds
more than one speaker. So the test measures whether the same audio gives the
same voiceprint, whether the two halves of one sentence match each other
(they are the same person by definition), and whether shaped audio scores
worse than raw. Only once those are known does the threshold mean anything.

Usage
-----
    python3.11 server/tests_real/test_real_diarization.py \\
        --speech recordings/meeting_speech.wav \\
        --other recordings/other_voice.wav

``--other`` is optional but the test proves very little without it: one
recording can only show that a voice matches itself.
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
    """Does gating the audio before embedding help or hurt?"""
    if not shaped or not raw:
        return
    shaped_median, raw_median = statistics.median(shaped), statistics.median(raw)
    print(f"    shaped median {shaped_median:.3f} vs raw median "
          f"{raw_median:.3f} ({shaped_median - raw_median:+.3f})")
    report.add(
        "Shaping the audio does not damage the voiceprints",
        shaped_median >= raw_median - 0.05,
        f"shaped {shaped_median:.3f}, raw {raw_median:.3f}",
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


def check_labelling(first: Voice, second: Optional[Voice],
                    report: Report) -> None:
    """What the registry actually does, not just what the numbers imply."""
    identifier = SpeakerIdentifier(
        embedder=_Replay(first.embeddings + (second.embeddings if second else [])),
        registry=SpeakerRegistry(),
        min_duration_ms=0,
    )
    labels_first = [identifier.identify(u.pcm).speaker_id for u in first.utterances]
    print(f"\n  labels for {first.path.name}: {labels_first}")
    report.add("One speaker is labelled as one speaker",
               len(set(labels_first)) == 1,
               f"{sorted(set(labels_first))}")

    if second is None:
        return
    labels_second = [identifier.identify(u.pcm).speaker_id
                     for u in second.utterances]
    print(f"  labels for {second.path.name}: {labels_second}")
    report.add("The second speaker is labelled as one speaker",
               len(set(labels_second)) == 1,
               f"{sorted(set(labels_second))}")
    report.add("The two speakers get different labels",
               not (set(labels_first) & set(labels_second)),
               f"{sorted(set(labels_first))} vs {sorted(set(labels_second))}")
    report.add("Exactly two speakers were found",
               identifier.registry.count == 2,
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
    parser.add_argument("--speech", type=Path, required=True,
                        help="WAV of one speaker")
    parser.add_argument("--other", type=Path, default=None,
                        help="WAV of a different speaker; without it the "
                             "threshold cannot be judged")
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

        first = embed_all(args.speech, embedder, vad, resolver)
        describe(first)
        second = None
        if args.other is not None:
            second = embed_all(args.other, embedder, vad, resolver)
            describe(second)

        same = pairwise(first.embeddings)
        if second is not None:
            same += pairwise(second.embeddings)
        different = crosswise(first.embeddings,
                              second.embeddings) if second else []

        halves = half_similarities(first.utterances, embedder)
        raw_same = pairwise(first.raw_embeddings)
        if second is not None:
            halves += half_similarities(second.utterances, embedder)
            raw_same += pairwise(second.raw_embeddings)

        print("\n  Cosine similarity:")
        summarise("two halves of one sentence", halves)
        summarise("sentence to sentence, same recording", same)
        summarise("sentence to sentence, across recordings", different)

        print("\nDiagnostics:")
        check_deterministic(first, embedder, report)
        halves_ok = check_halves(halves, report)
        check_shaping(same, raw_same, report)

        print("\nChecks:")
        check_embeddings(first, report)
        if second is not None:
            check_embeddings(second, report)
        check_separation(same, different, report)
        check_labelling(first, second, report)

        if halves_ok and same and statistics.median(same) < SPEAKER_MATCH_THRESHOLD:
            print("\n  The voiceprints are sound (both halves of a sentence "
                  "match) but sentences within one recording do not.")
            print("  That points at the recordings rather than the code: a "
                  "file with several voices in it cannot be used as the "
                  "same-speaker ground truth.")

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
