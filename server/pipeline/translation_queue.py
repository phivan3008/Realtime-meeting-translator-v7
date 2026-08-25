"""Translation off the audio path.

Why this exists
---------------
Every stage of the pipeline used to run on the thread that reads the socket,
translation included. On the seventh end-to-end run one slow vLLM answer put
the whole connection twelve seconds behind: VAD events that cost a
millisecond to produce arrived 12.6 s late, and two sentences shared a
timestamp after twenty-one seconds of silence. Nothing was lost, but a
meeting cannot be read at that latency.

So a sentence now goes out the moment Whisper commits it, and its translation
follows as a separate message. The socket is never waiting on an LLM.

Why a queue needs limits
------------------------
Measured over three real runs, 66 gaps between committed sentences:

    median gap        3.58 s
    busiest 8 gaps    0.74 s mean, so 1.35 sentences/s
    one translation   0.15 s

At 3.7% utilisation the queue drains almost instantly - a ten second stall
leaves 13.5 sentences behind and clears them in two. Delays do not accumulate
across stalls, because the queue empties between them.

That is a fact about vLLM's speed today, not a property of the design. A
slower model, a busier GPU or a second session and the arithmetic changes,
and an unbounded queue would then fall further behind with no floor. So the
limits are here from the start:

* a sentence that has waited too long is dropped rather than translated. Not
  because the queue cannot cope, but because the answer has stopped being
  useful: at a 3.58 s median gap, ten seconds is three sentences ago, and a
  translation appearing under a sentence the reader has scrolled past reads
  as a translation of something else.
* the queue has a ceiling, so a pathological stall cannot grow it without
  bound.

Both drops are announced, with a reason. A translation that never arrives and
never says why is the silent failure this project has already paid for twice.

Layering
--------
``TranslationQueue``
    The policy: what to accept, what to drop, what to say about it. Pure
    Python, no threads, no clock of its own - the time source is injected, so
    a test can make a sentence three minutes old without waiting.

``TranslationWorker``
    Runs the queue against a translator. ``inline`` runs it on the calling
    thread, which is what the unit tests use; otherwise it owns a thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from server.config import (
    TRANSLATION_MAX_LAG_SECONDS,
    TRANSLATION_QUEUE_DEPTH,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    """One sentence waiting to be translated."""

    sentence_id: int
    text: str
    lang_code: str
    speaker_id: str
    #: When the sentence was committed, on the queue's clock.
    submitted_at: float

    def age(self, now: float) -> float:
        return now - self.submitted_at


@dataclass
class QueueStats:
    submitted: int = 0
    translated: int = 0
    #: Dropped because the answer would have arrived too late to read.
    dropped_late: int = 0
    #: Dropped because the queue was already full.
    dropped_full: int = 0
    deepest: int = 0
    worst_lag_seconds: float = 0.0

    @property
    def dropped(self) -> int:
        return self.dropped_late + self.dropped_full


@dataclass
class Done:
    """A finished job, ready to go on the wire."""

    sentence_id: int
    translation: str = ""
    reason: str = ""
    raw: str = ""
    lag_seconds: float = 0.0


class TranslationQueue:
    """What to translate, what to give up on, and what to say about it."""

    def __init__(
        self,
        max_lag_seconds: float = TRANSLATION_MAX_LAG_SECONDS,
        depth: int = TRANSLATION_QUEUE_DEPTH,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_lag_seconds <= 0:
            raise ValueError("max_lag_seconds must be positive")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.max_lag_seconds = max_lag_seconds
        self.depth = depth
        self.clock = clock
        self._waiting: list[Job] = []
        self.stats = QueueStats()

    def __len__(self) -> int:
        return len(self._waiting)

    def submit(self, sentence_id: int, text: str, lang_code: str,
               speaker_id: str = "") -> Optional[Done]:
        """Accept a sentence, or say why it was not accepted.

        Returns a :class:`Done` when the sentence is refused outright, so the
        client hears about it in the same breath as the sentence itself.
        """
        self.stats.submitted += 1
        if len(self._waiting) >= self.depth:
            # The oldest is the least useful, so it goes rather than the
            # newest: dropping the newest would keep a backlog nobody can
            # still use and discard the one sentence that is current.
            evicted = self._waiting.pop(0)
            self.stats.dropped_full += 1
            log.warning(
                "Translation queue full at %d; dropping sentence %d "
                "(waited %.1f s)",
                self.depth, evicted.sentence_id, evicted.age(self.clock()))
            self._waiting.append(
                Job(sentence_id, text, lang_code, speaker_id, self.clock()))
            self.stats.deepest = max(self.stats.deepest, len(self._waiting))
            return Done(
                sentence_id=evicted.sentence_id,
                reason=f"the translation queue was full ({self.depth} waiting)",
                lag_seconds=evicted.age(self.clock()),
            )

        self._waiting.append(
            Job(sentence_id, text, lang_code, speaker_id, self.clock()))
        self.stats.deepest = max(self.stats.deepest, len(self._waiting))
        return None

    def take(self) -> tuple[Optional[Job], list[Done]]:
        """The next sentence worth translating, and any given up on first.

        Staleness is judged when the job comes off the queue, not when it goes
        on: a job that waited two seconds behind a slow one is still worth
        translating, and only the ones that waited past the budget are not.
        """
        given_up: list[Done] = []
        now = self.clock()
        while self._waiting:
            job = self._waiting.pop(0)
            age = job.age(now)
            if age > self.max_lag_seconds:
                self.stats.dropped_late += 1
                log.warning(
                    "Sentence %d waited %.1f s for translation, past the "
                    "%.0f s budget; dropping it rather than showing a "
                    "translation of something the reader has scrolled past",
                    job.sentence_id, age, self.max_lag_seconds)
                given_up.append(Done(
                    sentence_id=job.sentence_id,
                    reason=f"not translated in time ({age:.1f} s, "
                           f"budget {self.max_lag_seconds:.0f} s)",
                    lag_seconds=age,
                ))
                continue
            return job, given_up
        return None, given_up

    def finished(self, job: Job, translation: str, reason: str = "",
                 raw: str = "") -> Done:
        """Record a completed translation and stamp it with its total lag."""
        lag = job.age(self.clock())
        self.stats.worst_lag_seconds = max(self.stats.worst_lag_seconds, lag)
        if translation:
            self.stats.translated += 1
        return Done(sentence_id=job.sentence_id, translation=translation,
                    reason=reason, raw=raw, lag_seconds=lag)

    def drain(self) -> list[Done]:
        """Give up on everything still waiting, for a session that is ending.

        A sentence left in here at the end has no later chance, and saying
        nothing about it is the silent failure this project keeps paying for.
        """
        now = self.clock()
        remaining = self._waiting
        self._waiting = []
        results = []
        for job in remaining:
            self.stats.dropped_late += 1
            results.append(Done(
                sentence_id=job.sentence_id,
                reason="the meeting ended before this was translated",
                lag_seconds=job.age(now),
            ))
        return results


class TranslationWorker:
    """Runs the queue against a translator, off the thread that reads audio.

    ``inline=True`` does the work on the calling thread instead. That is what
    the unit tests use: a thread would make them depend on timing, and this
    project has one timing-dependent test already and does not want two.
    """

    def __init__(self, translator, queue_: Optional[TranslationQueue] = None,
                 inline: bool = False) -> None:
        self.translator = translator
        self.queue = queue_ if queue_ is not None else TranslationQueue()
        self.inline = inline
        self._done: "queue.Queue[Done]" = queue.Queue()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self.inline or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="translation-worker", daemon=True)
        self._thread.start()

    def submit(self, sentence_id: int, text: str, lang_code: str,
               speaker_id: str = "") -> None:
        with self._lock:
            refused = self.queue.submit(sentence_id, text, lang_code, speaker_id)
        if refused is not None:
            self._done.put(refused)
        if self.inline:
            self._work_once()
        else:
            self._wake.set()

    def collect(self) -> list[Done]:
        """Everything finished since the last call. Never blocks."""
        results = []
        while True:
            try:
                results.append(self._done.get_nowait())
            except queue.Empty:
                return results

    def stop(self, timeout: float = 2.0) -> list[Done]:
        """Finish what is queued, then stop and account for the rest.

        The waiting comes first, and it has to. The last sentence of a meeting
        is committed and queued by the same call that stops the worker, so
        setting the stop flag straight away raced the thread to it: the thread
        woke, saw the flag, and left the sentence to be drained. On a
        ten-minute run that read as

            NOT translated: the meeting ended before this was translated

        for a sentence whose translation was 0.2 s away. Every sentence still
        gets an answer either way, which is why the "never answered" check
        passed and only a person reading the output caught it.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not len(self.queue):
                    break
            self._wake.set()
            time.sleep(0.01)

        self._stopping.set()
        self._wake.set()
        if self._thread is not None:
            # Whatever is mid-flight finishes here; the loop exits after it.
            self._thread.join(timeout=timeout)
            self._thread = None
        with self._lock:
            leftover = self.queue.drain()
        return self.collect() + leftover

    # -- internals ----------------------------------------------------------
    def _work_once(self) -> bool:
        """Translate one sentence. True if there was one."""
        with self._lock:
            job, given_up = self.queue.take()
        for done in given_up:
            self._done.put(done)
        if job is None:
            return False
        try:
            result = self.translator.translate(job.text, job.lang_code,
                                               job.speaker_id)
        except Exception:
            # A bug here must not take the worker down with it: the meeting
            # would then lose every translation after this one, silently.
            log.exception("Translating sentence %d raised", job.sentence_id)
            with self._lock:
                done = self.queue.finished(
                    job, "", reason="the translator raised")
            self._done.put(done)
            return True
        with self._lock:
            done = self.queue.finished(
                job, result.text, reason=result.reason,
                raw="" if result.ok else result.raw)
        self._done.put(done)
        return True

    def _run(self) -> None:
        while not self._stopping.is_set():
            if not self._work_once():
                self._wake.wait(timeout=0.2)
                self._wake.clear()
