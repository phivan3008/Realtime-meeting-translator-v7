"""Unit tests for reclustering the meeting's speakers.

Pure numpy over hand-built voiceprints, so no model and no GPU. A "voice"
here is a direction in a small vector space; two voices of the same person
are the same direction with noise on it.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import SPEAKER_UNKNOWN
from server.pipeline.reclustering import (
    SpeakerHistory,
    cluster,
    name_clusters,
)

RNG = np.random.default_rng(7)


def voice(direction: int, jitter: float = 0.15) -> np.ndarray:
    """A voiceprint from speaker ``direction``, with a little noise."""
    base = np.zeros(8)
    base[direction] = 1.0
    return base + RNG.normal(0.0, jitter, size=8)


def history(**kwargs) -> SpeakerHistory:
    defaults = dict(threshold=0.30, every=2, max_voices=100)
    defaults.update(kwargs)
    return SpeakerHistory(**defaults)


# ---------------------------------------------------------------------------
# The clustering itself
# ---------------------------------------------------------------------------
def test_one_speaker_makes_one_cluster():
    embeddings = np.vstack([voice(0) for _ in range(6)])
    assert len(cluster(embeddings, 0.30)) == 1


def test_two_speakers_make_two_clusters():
    embeddings = np.vstack([voice(0) for _ in range(4)]
                           + [voice(3) for _ in range(4)])
    groups = cluster(embeddings, 0.30)
    assert len(groups) == 2
    assert sorted(len(group) for group in groups) == [4, 4]


def test_the_members_of_a_cluster_are_the_same_speaker():
    embeddings = np.vstack([voice(0), voice(3), voice(0), voice(3)])
    groups = {tuple(group) for group in cluster(embeddings, 0.30)}
    assert groups == {(0, 2), (1, 3)}


def test_nothing_to_cluster_is_not_an_error():
    assert cluster(np.zeros((0, 8)), 0.30) == []


def test_a_single_voice_is_its_own_cluster():
    assert cluster(np.vstack([voice(0)]), 0.30) == [[0]]


def test_average_linkage_does_not_let_one_sentence_chain_two_speakers():
    """Nearest-neighbour linkage would merge everything through the middle."""
    between = (voice(0, jitter=0.0) + voice(5, jitter=0.0)) / 2
    embeddings = np.vstack([voice(0), voice(0), between, voice(5), voice(5)])
    assert len(cluster(embeddings, 0.55)) > 1


def test_the_order_of_the_meeting_does_not_change_the_answer():
    """The live matcher's answer depends on it; this one must not."""
    voices = [voice(0), voice(3), voice(0), voice(3), voice(0)]
    forward = {tuple(sorted(g)) for g in cluster(np.vstack(voices), 0.30)}
    order = [4, 1, 0, 3, 2]
    shuffled = cluster(np.vstack([voices[i] for i in order]), 0.30)
    back = {tuple(sorted(order[i] for i in g)) for g in shuffled}
    assert forward == back


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def test_a_cluster_keeps_the_name_most_of_it_already_had():
    names = name_clusters([[0, 1, 2]], ["Speaker_02", "Speaker_02", "Speaker_07"])
    assert names == ["Speaker_02"]


def test_the_larger_cluster_keeps_a_contested_name():
    """Renaming the larger one moves more sentences on screen."""
    names = name_clusters([[0], [1, 2, 3]],
                          ["Speaker_01"] * 1 + ["Speaker_01"] * 3)
    assert names[1] == "Speaker_01"
    assert names[0] != "Speaker_01"


def test_a_cluster_with_no_name_of_its_own_gets_a_fresh_one():
    names = name_clusters([[0]], [SPEAKER_UNKNOWN])
    assert names[0].startswith("Speaker_")
    assert names[0] != SPEAKER_UNKNOWN


def test_no_two_clusters_share_a_name():
    names = name_clusters([[0, 1], [2, 3]], ["Speaker_01"] * 4)
    assert len(set(names)) == 2


# ---------------------------------------------------------------------------
# The history
# ---------------------------------------------------------------------------
def test_it_reports_only_the_labels_that_moved():
    watcher = history()
    watcher.add(1, voice(0), "Speaker_01")
    watcher.add(2, voice(0), "Speaker_01")
    watcher.add(3, voice(0), "Speaker_02")      # the live matcher split a voice
    assert watcher.recluster() == {3: "Speaker_01"}


def test_a_meeting_that_was_labelled_correctly_needs_no_corrections():
    watcher = history()
    watcher.add(1, voice(0), "Speaker_01")
    watcher.add(2, voice(3), "Speaker_02")
    watcher.add(3, voice(0), "Speaker_01")
    assert watcher.recluster() == {}


def test_two_speakers_merged_into_one_are_pulled_apart():
    """The reported failure: everybody came out as Speaker_01."""
    watcher = history()
    for index in range(1, 5):
        watcher.add(index, voice(0), "Speaker_01")
    for index in range(5, 9):
        watcher.add(index, voice(4), "Speaker_01")
    corrections = watcher.recluster()
    assert len(corrections) == 4, "one of the two voices was not separated"
    assert len(set(corrections.values())) == 1


def test_a_correction_is_not_reported_twice():
    watcher = history()
    watcher.add(1, voice(0), "Speaker_01")
    watcher.add(2, voice(0), "Speaker_02")
    assert watcher.recluster()
    assert watcher.recluster() == {}


def test_an_unidentifiable_voice_is_not_clustered():
    """Too short for a voiceprint means too short to cluster on."""
    watcher = history()
    watcher.add(1, voice(0), SPEAKER_UNKNOWN)
    watcher.add(2, voice(0), "Speaker_01")
    assert len(watcher.voices) == 1


def test_it_only_runs_when_enough_has_happened_since_the_last_time():
    watcher = history(every=3)
    watcher.add(1, voice(0), "Speaker_01")
    assert watcher.due is False
    watcher.add(2, voice(0), "Speaker_01")
    watcher.add(3, voice(0), "Speaker_01")
    assert watcher.due is True
    watcher.recluster()
    assert watcher.due is False


def test_the_oldest_sentences_fall_out():
    """Cost grows with the square of this, and they have scrolled away."""
    watcher = history(max_voices=5)
    for index in range(1, 12):
        watcher.add(index, voice(0), "Speaker_01")
    assert len(watcher.voices) == 5
    assert watcher.voices[0].sentence_id == 7


def test_reset_forgets_the_meeting():
    watcher = history()
    watcher.add(1, voice(0), "Speaker_01")
    watcher.reset()
    assert watcher.voices == []
    assert watcher.due is False


def test_clustering_a_full_history_is_quick_enough_for_the_audio_thread():
    """Everything here runs on the thread that reads the socket."""
    import time

    embeddings = np.vstack([voice(index % 6) for index in range(300)])
    started = time.perf_counter()
    cluster(embeddings, 0.30)
    spent = time.perf_counter() - started
    assert spent < 0.5, f"clustering 300 voiceprints took {spent:.2f} s"


def test_every_must_be_positive():
    with pytest.raises(ValueError):
        history(every=0)
