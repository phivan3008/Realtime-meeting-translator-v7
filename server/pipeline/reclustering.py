"""Second thoughts about who said what - step 5b of the pipeline.

``SpeakerIdentifier`` has to answer immediately, from one voiceprint, against
whatever it has heard so far. That is a hard question, and it gets two things
wrong that no amount of threshold tuning fixes:

* the answer depends on the order the meeting happened in. The first
  utterance is compared against nothing and always founds a speaker, and
  every later one against centroids that have already moved.
* an answer, once given, stands. A mistake in the first minute survives the
  ten minutes of evidence that follow it.

**Measured, not sent, by default.** On a real meeting of four people the
live matcher gave four main labels (113/91/67/18 sentences); these
corrections moved 140 of 290 sentences, merged two people into one and left
seven clusters of one to three sentences. ``SPEAKER_RECLUSTER=1`` sends them;
otherwise :meth:`SpeakerHistory.survey` logs what they would have been.

Diarization is a clustering problem, not a streaming classification one. This
keeps every voiceprint of the meeting and periodically clusters the lot from
scratch, then reports the labels that came out different. The transcript is
keyed by ``sentence_id`` at both ends, so a corrected label is an update to a
row that is already on screen.

Clustering is average-linkage agglomerative over cosine similarity, cut at
``SPEAKER_RECLUSTER_THRESHOLD`` and never leaving more than
``SPEAKER_MAX_SPEAKERS`` clusters - the live matcher honours that cap, and a
thirty-minute run without it here found 22 speakers.

Labels are chosen to stay put. A cluster keeps whichever label most of its
members already carry, and a sentence's label only moves once
``SPEAKER_RECLUSTER_CONFIRMATIONS`` runs in a row agree on the new one. The
same run corrected 329 labels across 313 sentences: names that would not sit
still on screen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from server.config import (
    SPEAKER_MAX_SPEAKERS,
    SPEAKER_RECLUSTER_CONFIRMATIONS,
    SPEAKER_RECLUSTER_EVERY,
    SPEAKER_RECLUSTER_MAX,
    SPEAKER_RECLUSTER_THRESHOLD,
    SPEAKER_UNKNOWN,
)
from server.pipeline.diarization import label_for

log = logging.getLogger(__name__)


@dataclass
class Voice:
    """One committed sentence's voiceprint and the label it went out with."""

    sentence_id: int
    embedding: np.ndarray
    label: str


@dataclass
class Clustering:
    """The groups, and the scores the decision to stop was made on."""

    groups: list[list[int]]
    #: Average similarity of every merge taken, in order.
    merges: list[float] = field(default_factory=list)
    #: The best merge refused, or None when everything was merged.
    stopped_at: Optional[float] = None
    #: Merges taken below the threshold only because of the speaker cap.
    forced: int = 0


@dataclass
class ReclusterStats:
    runs: int = 0
    corrections: int = 0
    #: Speakers the clustering found, at the last run.
    speakers: int = 0
    #: Labels proposed and waiting for a second run to agree.
    pending: int = 0
    #: Merges the speaker cap forced below the threshold, over the meeting.
    forced_merges: int = 0
    #: The best refused merge of every run: where the threshold sits against
    #: the real distribution.
    stop_scores: list = field(default_factory=list)
    #: Cluster sizes at the last run, largest first.
    sizes: list = field(default_factory=list)
    #: Surveys only: sentences whose label the clustering disagreed with at
    #: the last run. Nothing was changed.
    would_move: int = 0

    def record(self, changed: int, result: Clustering, pending: int) -> None:
        self.runs += 1
        self.corrections += changed
        self.speakers = len(result.groups)
        self.sizes = sorted((len(group) for group in result.groups),
                            reverse=True)
        self.pending = pending
        self.forced_merges += result.forced
        if result.stopped_at is not None:
            self.stop_scores.append(result.stopped_at)


def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity of every voiceprint against every other."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms
    return unit @ unit.T


def cluster_scored(embeddings: np.ndarray, threshold: float,
                   max_clusters: int = 0) -> Clustering:
    """Average-linkage agglomerative clustering, cut at ``threshold``.

    Merges the closest pair of clusters until none are closer than the cut
    and no more than ``max_clusters`` remain (0 means no cap). Average linkage
    rather than nearest: one borderline sentence should not be able to chain
    two speakers together.
    """
    count = len(embeddings)
    if count == 0:
        return Clustering(groups=[])

    scores = similarity_matrix(embeddings)
    members: list[list[int]] = [[index] for index in range(count)]
    # Sum of similarities between each pair of clusters; the average is this
    # divided by the product of their sizes.
    totals = scores.astype(np.float64).copy()
    np.fill_diagonal(totals, -np.inf)
    alive = np.ones(count, dtype=bool)
    sizes = np.ones(count, dtype=np.float64)
    result = Clustering(groups=[])
    remaining = count

    while remaining > 1:
        averages = np.where(
            alive[:, None] & alive[None, :],
            totals / np.outer(sizes, sizes),
            -np.inf,
        )
        best = int(np.argmax(averages))
        left, right = divmod(best, count)
        score = float(averages[left, right])
        if score < threshold:
            if not max_clusters or remaining <= max_clusters:
                result.stopped_at = score
                break
            result.forced += 1

        result.merges.append(score)
        members[left] = members[left] + members[right]
        totals[left, :] += totals[right, :]
        totals[:, left] += totals[:, right]
        sizes[left] += sizes[right]
        alive[right] = False
        totals[left, left] = -np.inf
        remaining -= 1

    result.groups = [sorted(members[index])
                     for index in range(count) if alive[index]]
    return result


def cluster(embeddings: np.ndarray, threshold: float,
            max_clusters: int = 0) -> list[list[int]]:
    """The groups alone. See :func:`cluster_scored`."""
    return cluster_scored(embeddings, threshold, max_clusters).groups


def name_clusters(groups: list[list[int]], labels: list[str]) -> list[str]:
    """One label per cluster, chosen so most sentences keep the name they had.

    A cluster takes the label most of its members already carry. When two
    clusters want the same name the larger keeps it, because renaming the
    larger one moves more sentences on screen than renaming the smaller.
    """
    ranked = sorted(range(len(groups)), key=lambda index: -len(groups[index]))
    chosen: list[Optional[str]] = [None] * len(groups)
    taken: set[str] = set()

    for index in ranked:
        counts: dict[str, int] = {}
        for member in groups[index]:
            label = labels[member]
            if label and label != SPEAKER_UNKNOWN and label not in taken:
                counts[label] = counts.get(label, 0) + 1
        if counts:
            chosen[index] = max(counts.items(), key=lambda kv: kv[1])[0]
            taken.add(chosen[index])

    spare = 1
    for index in ranked:
        if chosen[index] is None:
            while label_for(spare) in taken:
                spare += 1
            chosen[index] = label_for(spare)
            taken.add(chosen[index])
    return [name for name in chosen]                    # type: ignore[misc]


def live_sizes(labels: list) -> list:
    """How many sentences each live label holds, largest first."""
    counts: dict = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.values(), reverse=True)


class SpeakerHistory:
    """Every voiceprint of the meeting, and second thoughts about the labels."""

    def __init__(
        self,
        threshold: float = SPEAKER_RECLUSTER_THRESHOLD,
        every: int = SPEAKER_RECLUSTER_EVERY,
        max_voices: int = SPEAKER_RECLUSTER_MAX,
        max_speakers: int = SPEAKER_MAX_SPEAKERS,
        confirmations: int = SPEAKER_RECLUSTER_CONFIRMATIONS,
    ) -> None:
        if every <= 0:
            raise ValueError("every must be positive")
        if confirmations <= 0:
            raise ValueError("confirmations must be positive")
        self.threshold = threshold
        self.every = every
        self.max_voices = max_voices
        self.max_speakers = max_speakers
        self.confirmations = confirmations
        self.voices: list[Voice] = []
        self.stats = ReclusterStats()
        self._since = 0
        #: sentence_id -> (proposed label, runs in a row that proposed it)
        self._pending: dict[int, tuple[str, int]] = {}

    def add(self, sentence_id: int, embedding: np.ndarray, label: str) -> None:
        """Remember one committed sentence. Unknown voices are not clustered."""
        if label == SPEAKER_UNKNOWN or embedding.size == 0:
            return
        self.voices.append(Voice(sentence_id, np.asarray(embedding), label))
        if len(self.voices) > self.max_voices:
            # The oldest sentences have scrolled out of the window anyway, and
            # the cost of clustering grows with the square of this.
            dropped = self.voices.pop(0)
            self._pending.pop(dropped.sentence_id, None)
        self._since += 1

    @property
    def due(self) -> bool:
        return self._since >= self.every and len(self.voices) >= 2

    def recluster(self) -> dict[int, str]:
        """Cluster the whole meeting again. Returns only the labels that moved."""
        self._since = 0
        if len(self.voices) < 2:
            return {}

        embeddings = np.vstack([voice.embedding for voice in self.voices])
        labels = [voice.label for voice in self.voices]
        result = cluster_scored(embeddings, self.threshold, self.max_speakers)
        names = name_clusters(result.groups, labels)

        corrections: dict[int, str] = {}
        proposed: dict[int, tuple[str, int]] = {}
        for group, name in zip(result.groups, names):
            for member in group:
                voice = self.voices[member]
                if voice.label == name:
                    continue
                previous, seen = self._pending.get(voice.sentence_id, ("", 0))
                seen = seen + 1 if previous == name else 1
                if seen >= self.confirmations:
                    corrections[voice.sentence_id] = name
                    voice.label = name
                else:
                    proposed[voice.sentence_id] = (name, seen)
        # A proposal not repeated this run is forgotten.
        self._pending = proposed

        self.stats.record(len(corrections), result, len(proposed))
        log.info(
            "Reclustered %d sentences into %d speakers; %d labels corrected, "
            "%d waiting for confirmation, %d merges forced by the %d-speaker "
            "cap; best refused merge %s (threshold %.2f)",
            len(self.voices), len(result.groups), len(corrections),
            len(proposed), result.forced, self.max_speakers,
            "none" if result.stopped_at is None
            else f"{result.stopped_at:.3f}", self.threshold)
        return corrections

    def survey(self) -> int:
        """Cluster the meeting and count what would move, changing nothing.

        On a real meeting of four people the live matcher gave four main
        labels and the corrections made them worse - two people merged into
        one and seven clusters of one to three sentences. Until clustering
        does better, it is measured here rather than sent.
        """
        self._since = 0
        if len(self.voices) < 2:
            return 0
        embeddings = np.vstack([voice.embedding for voice in self.voices])
        labels = [voice.label for voice in self.voices]
        result = cluster_scored(embeddings, self.threshold, self.max_speakers)
        names = name_clusters(result.groups, labels)
        would_move = sum(1 for group, name in zip(result.groups, names)
                         for member in group if labels[member] != name)
        self.stats.record(0, result, 0)
        self.stats.would_move = would_move
        log.info(
            "Speaker survey (not sent): %d sentences, live labels %s; "
            "clustering finds %d speakers sized %s and disagrees on %d; "
            "best refused merge %s",
            len(self.voices), live_sizes(labels), len(result.groups),
            self.stats.sizes, would_move,
            "none" if result.stopped_at is None
            else f"{result.stopped_at:.3f}")
        return would_move

    def reset(self) -> None:
        self.voices.clear()
        self.stats = ReclusterStats()
        self._since = 0
        self._pending.clear()
