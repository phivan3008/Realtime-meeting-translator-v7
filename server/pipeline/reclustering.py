"""Second thoughts about who said what - step 5b of the pipeline.

``SpeakerIdentifier`` has to answer immediately, from one voiceprint, against
whatever it has heard so far. That is a hard question, and it gets two things
wrong that no amount of threshold tuning fixes:

* the answer depends on the order the meeting happened in. The first
  utterance is compared against nothing and always founds a speaker, and
  every later one against centroids that have already moved.
* an answer, once given, stands. A mistake in the first minute survives the
  ten minutes of evidence that follow it.

Diarization is a clustering problem, not a streaming classification one. This
keeps every voiceprint of the meeting and periodically clusters the lot from
scratch, then reports the labels that came out different. The transcript is
keyed by ``sentence_id`` at both ends, so a corrected label is an update to a
row that is already on screen.

Clustering is average-linkage agglomerative over cosine similarity, cut at
``SPEAKER_MATCH_THRESHOLD`` - the same measured number the live matcher uses,
applied to the same whole-sentence voiceprints it was measured on.

Labels are chosen to stay put. A cluster keeps whichever label most of its
members already carry, so a correction moves the few sentences that were
wrong rather than renaming everybody.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from server.config import (
    SPEAKER_MATCH_THRESHOLD,
    SPEAKER_RECLUSTER_EVERY,
    SPEAKER_RECLUSTER_MAX,
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
class ReclusterStats:
    runs: int = 0
    corrections: int = 0
    #: Speakers the clustering found, at the last run.
    speakers: int = 0

    def record(self, changed: int, speakers: int) -> None:
        self.runs += 1
        self.corrections += changed
        self.speakers = speakers


def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity of every voiceprint against every other."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms
    return unit @ unit.T


def cluster(embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """Average-linkage agglomerative clustering, cut at ``threshold``.

    Merges the closest pair of clusters until none are closer than the cut.
    Average linkage rather than nearest: one borderline sentence should not
    be able to chain two speakers together.
    """
    count = len(embeddings)
    if count == 0:
        return []

    scores = similarity_matrix(embeddings)
    members: list[list[int]] = [[index] for index in range(count)]
    # Sum of similarities between each pair of clusters; the average is this
    # divided by the product of their sizes.
    totals = scores.astype(np.float64).copy()
    np.fill_diagonal(totals, -np.inf)
    alive = np.ones(count, dtype=bool)
    sizes = np.ones(count, dtype=np.float64)

    while True:
        averages = np.where(
            alive[:, None] & alive[None, :],
            totals / np.outer(sizes, sizes),
            -np.inf,
        )
        best = int(np.argmax(averages))
        left, right = divmod(best, count)
        if averages[left, right] < threshold:
            break

        members[left] = members[left] + members[right]
        totals[left, :] += totals[right, :]
        totals[:, left] += totals[:, right]
        sizes[left] += sizes[right]
        alive[right] = False
        totals[left, left] = -np.inf

    return [sorted(members[index]) for index in range(count) if alive[index]]


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


class SpeakerHistory:
    """Every voiceprint of the meeting, and second thoughts about the labels."""

    def __init__(
        self,
        threshold: float = SPEAKER_MATCH_THRESHOLD,
        every: int = SPEAKER_RECLUSTER_EVERY,
        max_voices: int = SPEAKER_RECLUSTER_MAX,
    ) -> None:
        if every <= 0:
            raise ValueError("every must be positive")
        self.threshold = threshold
        self.every = every
        self.max_voices = max_voices
        self.voices: list[Voice] = []
        self.stats = ReclusterStats()
        self._since = 0

    def add(self, sentence_id: int, embedding: np.ndarray, label: str) -> None:
        """Remember one committed sentence. Unknown voices are not clustered."""
        if label == SPEAKER_UNKNOWN or embedding.size == 0:
            return
        self.voices.append(Voice(sentence_id, np.asarray(embedding), label))
        if len(self.voices) > self.max_voices:
            # The oldest sentences have scrolled out of the window anyway, and
            # the cost of clustering grows with the square of this.
            self.voices.pop(0)
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
        groups = cluster(embeddings, self.threshold)
        names = name_clusters(groups, labels)

        corrections: dict[int, str] = {}
        for group, name in zip(groups, names):
            for member in group:
                voice = self.voices[member]
                if voice.label != name:
                    corrections[voice.sentence_id] = name
                    voice.label = name

        self.stats.record(len(corrections), len(groups))
        if corrections:
            log.info("Reclustered %d sentences into %d speakers; %d labels "
                     "corrected", len(self.voices), len(groups),
                     len(corrections))
        return corrections

    def reset(self) -> None:
        self.voices.clear()
        self.stats = ReclusterStats()
        self._since = 0
