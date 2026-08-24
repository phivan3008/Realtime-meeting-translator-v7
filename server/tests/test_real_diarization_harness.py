"""Smoke tests for ``server/tests_real/test_real_diarization.py``.

Drives the whole script, main() included, with stubbed models, so a crash in
it is caught here rather than after a round trip through the pod.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server.config import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH  # noqa: E402
from server.pipeline.vad import VAD_FRAME_SAMPLES  # noqa: E402


def load_harness():
    path = ROOT / "server" / "tests_real" / "test_real_diarization.py"
    spec = importlib.util.spec_from_file_location("real_diar_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = load_harness()


class ScriptedVAD:
    def __init__(self, probabilities):
        self.script = list(probabilities)
        self.calls = 0
        self.resets = 0

    def probability(self, frame: np.ndarray) -> float:
        assert frame.shape[-1] == VAD_FRAME_SAMPLES
        value = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return float(value)

    def reset(self) -> None:
        self.resets += 1
        self.calls = 0


class StubEmbedder:
    """Hands out voiceprints near a fixed direction, one per call."""

    source = "stub (cpu)"

    def __init__(self, direction: np.ndarray, jitter: float = 0.01):
        self.direction = np.asarray(direction, dtype=np.float64)
        self.jitter = jitter
        self.calls = 0

    def embed(self, pcm: bytes) -> np.ndarray:
        self.calls += 1
        wobble = np.zeros_like(self.direction)
        wobble[self.calls % self.direction.size] = self.jitter
        return self.direction + wobble


class RoutingEmbedder:
    """One embedder for the whole run, telling the voices apart by loudness.

    main() loads the model once and feeds it both recordings, so a stub that
    returns a fixed direction would make every speaker identical - which is a
    property of the stub, not of the code under test.
    """

    source = "routing stub (cpu)"

    def __init__(self, loud: np.ndarray, quiet: np.ndarray, boundary: int = 4500):
        self.loud = np.asarray(loud, dtype=np.float64)
        self.quiet = np.asarray(quiet, dtype=np.float64)
        self.boundary = boundary
        self.calls = 0

    def embed(self, pcm: bytes) -> np.ndarray:
        self.calls += 1
        amplitude = float(np.mean(np.abs(np.frombuffer(pcm, dtype="<i2"))))
        direction = self.loud if amplitude > self.boundary else self.quiet
        wobble = np.zeros_like(direction)
        wobble[self.calls % direction.size] = 0.01
        return direction + wobble


ALICE = np.array([1.0, 0.0, 0.0])
BOB = np.array([0.0, 1.0, 0.0])


def write_wav(path: Path, seconds: float, amplitude: int = 6000) -> Path:
    samples = np.full(int(SAMPLE_RATE * seconds), amplitude, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())
    return path


# ---------------------------------------------------------------------------
# Similarity bookkeeping
# ---------------------------------------------------------------------------
def test_pairwise_covers_every_pair_once():
    assert len(harness.pairwise([ALICE, BOB, ALICE])) == 3


def test_pairwise_of_one_voiceprint_is_empty():
    assert harness.pairwise([ALICE]) == []


def test_crosswise_covers_every_combination():
    assert len(harness.crosswise([ALICE, BOB], [ALICE, BOB, ALICE])) == 6


def test_summarise_survives_an_empty_list(capsys):
    harness.summarise("same", [])
    assert "none" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Cutting a recording into sentences
# ---------------------------------------------------------------------------
def make_resolver():
    from server.pipeline.overlap import OverlapResolver

    class Passthrough:
        def process(self, samples, sample_rate, gate_threshold_db,
                    compressor_threshold_db):
            return samples

    return OverlapResolver(processor=Passthrough())


def test_a_talking_recording_yields_embedded_sentences(tmp_path):
    path = write_wav(tmp_path / "alice.wav", 12.0)
    voice = harness.embed_all(path, StubEmbedder(ALICE), ScriptedVAD([0.9]),
                              make_resolver())
    assert len(voice.utterances) >= 1
    assert len(voice.embeddings) == len(voice.utterances)
    assert voice.audio_seconds > 0
    assert voice.embed_ratio >= 0.0


def test_sentences_too_short_to_identify_are_left_out(tmp_path):
    """A 0.3 s sentence has no voiceprint worth taking."""
    path = write_wav(tmp_path / "alice.wav", 12.0)
    voice = harness.embed_all(path, StubEmbedder(ALICE), ScriptedVAD([0.9]),
                              make_resolver())
    assert all(u.duration_ms >= 600 for u in voice.utterances)


def test_a_silent_recording_yields_nothing(tmp_path):
    path = write_wav(tmp_path / "quiet.wav", 4.0)
    voice = harness.embed_all(path, StubEmbedder(ALICE), ScriptedVAD([0.02]),
                              make_resolver())
    assert voice.utterances == []
    assert voice.embeddings == []


def test_read_pcm_rejects_the_wrong_format(tmp_path):
    path = tmp_path / "wrong.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(44_100)
        wav.writeframes(bytes(4000))
    with pytest.raises(ValueError, match="44100 Hz"):
        harness.read_pcm(path)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def voice_of(embeddings, name="a.wav"):
    from server.pipeline.buffer import FinalizeReason, Utterance

    voice = harness.Voice(path=Path(name))
    voice.embeddings = [np.asarray(e, dtype=np.float64) for e in embeddings]
    voice.utterances = [
        Utterance(index=i, pcm=bytes(32_000), start_ms=i * 1000.0,
                  reason=FinalizeReason.PAUSE)
        for i in range(len(voice.embeddings))
    ]
    voice.embed_seconds = 0.01
    return voice


def test_well_separated_distributions_pass():
    report = harness.Report()
    harness.check_separation([0.90, 0.88], [0.10, 0.15], report)
    assert report.failed == []


def test_distributions_that_overlap_are_caught():
    """No threshold works here, and the test has to say so."""
    report = harness.Report()
    harness.check_separation([0.60, 0.40], [0.35, 0.58], report)
    names = [c.name for c in report.failed]
    assert "The two distributions do not overlap" in names


def test_a_voice_that_does_not_match_itself_is_caught():
    report = harness.Report()
    harness.check_separation([0.20, 0.15], [0.05], report)
    assert "A voice matches itself above the threshold" in [
        c.name for c in report.failed
    ]


def test_one_recording_alone_leaves_the_threshold_unproven(capsys):
    report = harness.Report()
    harness.check_separation([0.9, 0.88], [], report)
    assert report.failed == []
    assert "unproven" in capsys.readouterr().out


def test_labelling_two_speakers_gives_two_labels():
    report = harness.Report()
    harness.check_labelling(voice_of([ALICE, ALICE + 0.02], "alice.wav"),
                            voice_of([BOB, BOB + 0.02], "bob.wav"), report)
    assert report.failed == []


def test_a_speaker_split_in_two_is_caught():
    report = harness.Report()
    harness.check_labelling(voice_of([ALICE, BOB], "alice.wav"), None, report)
    assert "One speaker is labelled as one speaker" in [
        c.name for c in report.failed
    ]


def test_two_speakers_merged_into_one_is_caught():
    report = harness.Report()
    harness.check_labelling(voice_of([ALICE, ALICE], "alice.wav"),
                            voice_of([ALICE, ALICE], "bob.wav"), report)
    names = [c.name for c in report.failed]
    assert "The two speakers get different labels" in names
    assert "Exactly two speakers were found" in names


def test_embedding_checks_catch_a_ragged_voiceprint():
    report = harness.Report()
    harness.check_embeddings(voice_of([ALICE, np.array([1.0, 2.0])]), report)
    assert "a.wav voiceprints all have one dimension" in [
        c.name for c in report.failed
    ]


def test_embedding_checks_catch_a_single_sentence():
    report = harness.Report()
    harness.check_embeddings(voice_of([ALICE]), report)
    assert "a.wav produced sentences to identify" in [
        c.name for c in report.failed
    ]


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def run_main(monkeypatch, tmp_path, with_other: bool) -> int:
    alice = write_wav(tmp_path / "alice.wav", 12.0, amplitude=6000)
    bob = write_wav(tmp_path / "bob.wav", 12.0, amplitude=2000)
    embedder = RoutingEmbedder(loud=ALICE, quiet=BOB)

    monkeypatch.setattr(harness, "EcapaEmbedder", lambda **k: embedder)
    monkeypatch.setattr(harness, "SileroVAD", lambda **k: ScriptedVAD([0.9]))
    monkeypatch.setattr(harness, "PedalboardProcessor", lambda: _Passthrough())

    argv = ["x", "--speech", str(alice)]
    if with_other:
        argv += ["--other", str(bob)]
    monkeypatch.setattr(sys, "argv", argv)
    return harness.main()


class _Passthrough:
    def process(self, samples, sample_rate, gate_threshold_db,
                compressor_threshold_db):
        return samples


def test_main_runs_and_passes_with_two_speakers(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, with_other=True) == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out
    assert "separates them" in out


def test_main_runs_with_one_recording(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, with_other=False) == 0
    assert "unproven" in capsys.readouterr().out


def test_main_reports_a_model_that_will_not_load(monkeypatch, tmp_path, capsys):
    def explode(**_kwargs):
        raise harness.DiarizationError("Could not load 'speechbrain/...'")

    monkeypatch.setattr(harness, "EcapaEmbedder", explode)
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--speech", str(write_wav(tmp_path / "a.wav", 2.0))],
    )
    assert harness.main() == 2
    assert "Could not load" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
class WanderingEmbedder:
    """Returns a different voiceprint every call - a broken model."""

    source = "wandering stub"

    def __init__(self):
        self.calls = 0

    def embed(self, pcm: bytes) -> np.ndarray:
        self.calls += 1
        return np.array([1.0, float(self.calls), 0.0])


def test_a_deterministic_embedder_passes():
    report = harness.Report()
    harness.check_deterministic(voice_of([ALICE]), StubEmbedder(ALICE, jitter=0.0),
                                report)
    assert report.failed == []


def test_an_embedder_that_wanders_is_caught():
    """Two identical inputs giving different voiceprints explains everything else."""
    report = harness.Report()
    harness.check_deterministic(voice_of([ALICE]), WanderingEmbedder(), report)
    assert [c.name for c in report.failed] == [
        "The same audio gives the same voiceprint"
    ]


def test_halves_that_match_pass():
    report = harness.Report()
    assert harness.check_halves([0.88, 0.91], report) is True
    assert report.failed == []


def test_halves_that_do_not_match_point_at_the_voiceprints(capsys):
    report = harness.Report()
    assert harness.check_halves([0.20, 0.15], report) is False
    out = capsys.readouterr().out
    assert "the voiceprints themselves are wrong" in out


def test_no_sentence_long_enough_to_split_is_not_a_failure(capsys):
    report = harness.Report()
    assert harness.check_halves([], report) is True
    assert report.failed == []
    assert "long enough to split" in capsys.readouterr().out


def test_shaping_that_helps_or_leaves_things_alone_passes():
    report = harness.Report()
    harness.check_shaping([0.80, 0.82], [0.78, 0.79], report)
    assert report.failed == []


def test_shaping_that_damages_the_voiceprints_is_caught():
    """If gating hurts, the answer is to embed the raw audio instead."""
    report = harness.Report()
    harness.check_shaping([0.30, 0.32], [0.80, 0.82], report)
    assert [c.name for c in report.failed] == [
        "Shaping the audio does not damage the voiceprints"
    ]


def test_shaping_is_not_judged_without_both_measurements():
    report = harness.Report()
    harness.check_shaping([], [0.8], report)
    assert report.checks == []


def test_half_similarities_skips_sentences_too_short_to_split():
    from server.pipeline.buffer import FinalizeReason, Utterance

    short = Utterance(index=0, pcm=bytes(16_000), start_ms=0.0,
                      reason=FinalizeReason.PAUSE)          # 0.5 s
    long = Utterance(index=1, pcm=bytes(96_000), start_ms=0.0,
                     reason=FinalizeReason.PAUSE)           # 3.0 s
    scores = harness.half_similarities([short, long], StubEmbedder(ALICE))
    assert len(scores) == 1


def test_embed_all_keeps_a_raw_copy_of_every_sentence(tmp_path):
    path = write_wav(tmp_path / "alice.wav", 12.0)
    voice = harness.embed_all(path, StubEmbedder(ALICE), ScriptedVAD([0.9]),
                              make_resolver())
    assert len(voice.raw_utterances) == len(voice.utterances)
    assert len(voice.raw_embeddings) == len(voice.embeddings)
