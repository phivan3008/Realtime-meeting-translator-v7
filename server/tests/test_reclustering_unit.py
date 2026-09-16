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
    # One confirmation, so each test below reads one run. The two-run rule
    # has tests of its own at the end.
    defaults = dict(threshold=0.30, every=2, max_voices=100, confirmations=1)
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


# ---------------------------------------------------------------------------
# The two faults a thirty-minute run found: 22 speakers, and 329 labels
# corrected across 313 sentences.
# ---------------------------------------------------------------------------
def test_the_speaker_cap_is_honoured():
    """The live matcher stops at SPEAKER_MAX_SPEAKERS; clustering did not."""
    embeddings = np.vstack([voice(index) for index in range(8)])
    assert len(cluster(embeddings, 0.30)) == 8
    assert len(cluster(embeddings, 0.30, max_clusters=3)) == 3


def test_a_forced_merge_is_counted():
    from server.pipeline.reclustering import cluster_scored
    embeddings = np.vstack([voice(index) for index in range(5)])
    result = cluster_scored(embeddings, 0.30, max_clusters=2)
    assert len(result.groups) == 2
    assert result.forced == 3


def test_the_refused_merge_is_reported_so_the_threshold_can_be_placed():
    from server.pipeline.reclustering import cluster_scored
    embeddings = np.vstack([voice(0), voice(0), voice(4), voice(4)])
    result = cluster_scored(embeddings, 0.30)
    assert result.stopped_at is not None
    assert result.stopped_at < 0.30
    assert all(score >= 0.30 for score in result.merges)


def test_the_history_never_reports_more_speakers_than_the_cap():
    watcher = history(max_speakers=2)
    for index in range(6):
        watcher.add(index + 1, voice(index), label_for_test(index + 1))
    watcher.recluster()
    assert watcher.stats.speakers == 2


def label_for_test(index: int) -> str:
    return f"Speaker_{index:02d}"


def test_a_label_moves_only_after_two_runs_agree():
    watcher = history(confirmations=2)
    watcher.add(1, voice(0), "Speaker_01")
    watcher.add(2, voice(0), "Speaker_01")
    watcher.add(3, voice(0), "Speaker_02")
    assert watcher.recluster() == {}, "one run is not enough to move a name"
    assert watcher.stats.pending == 1
    assert watcher.recluster() == {3: "Speaker_01"}
    assert watcher.stats.pending == 0


def test_a_proposal_that_is_not_repeated_is_forgotten():
    watcher = history(confirmations=2)
    watcher.add(1, voice(0), "Speaker_01")
    watcher.add(2, voice(0), "Speaker_02")
    watcher.recluster()
    assert watcher.stats.pending
    # The evidence changes: sentence 2 now sits with a voice of its own kind.
    watcher.voices[0].embedding = voice(5)
    assert watcher.recluster() == {}
    assert watcher.stats.pending == 0


def test_confirmations_must_be_positive():
    with pytest.raises(ValueError):
        history(confirmations=0)



def test_a_survey_counts_what_would_move_and_moves_nothing():
    watcher = history()
    watcher.add(1, voice(0), "Speaker_01")
    watcher.add(2, voice(0), "Speaker_01")
    watcher.add(3, voice(0), "Speaker_02")
    assert watcher.survey() == 1
    assert [v.label for v in watcher.voices] == [
        "Speaker_01", "Speaker_01", "Speaker_02"]
    assert watcher.stats.would_move == 1
    assert watcher.stats.sizes == [3]
    assert watcher.stats.corrections == 0
    assert watcher.due is False


def test_the_live_label_sizes_are_read_largest_first():
    from server.pipeline.reclustering import live_sizes
    assert live_sizes(["a", "b", "a", "c", "a", "b"]) == [3, 2, 1]
