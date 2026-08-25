"""Unit tests for translation off the audio path.

The clock is injected, so a sentence can be made three minutes old without
waiting three minutes, and the worker runs inline, so nothing here depends on
thread scheduling.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (
    TRANSLATION_MAX_LAG_SECONDS,
    TRANSLATION_QUEUE_DEPTH,
)
from server.pipeline.translate import Translation
from server.pipeline.translation_queue import (
    TranslationQueue,
    TranslationWorker,
)


class Clock:
    """A clock a test can move."""

    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


class StubTranslator:
    def __init__(self, answer: str = "こんにちは", ok: bool = True,
                 reason: str = "", raw: str = ""):
        self.answer = answer
        self.ok = ok
        self.reason = reason
        self.raw = raw
        self.seen: list[str] = []

    def translate(self, text: str, lang_code: str, speaker_id: str = ""):
        self.seen.append(text)
        return Translation(self.answer if self.ok else "", text, lang_code,
                           "ja", self.reason, self.raw)


def make(clock: Clock | None = None, **kwargs) -> TranslationQueue:
    return TranslationQueue(clock=clock or Clock(), **kwargs)


# ---------------------------------------------------------------------------
# Accepting work
# ---------------------------------------------------------------------------
def test_a_submitted_sentence_waits():
    q = make()
    assert q.submit(1, "xin chào", "vi") is None
    assert len(q) == 1


def test_sentences_come_out_in_the_order_they_went_in():
    """Order is not cosmetic: the history is built from it."""
    q = make()
    for index in range(3):
        q.submit(index, f"line {index}", "vi")
    taken = [q.take()[0].sentence_id for _ in range(3)]
    assert taken == [0, 1, 2]


def test_an_empty_queue_has_nothing_to_take():
    job, given_up = make().take()
    assert job is None
    assert given_up == []


# ---------------------------------------------------------------------------
# Too late to be useful
# ---------------------------------------------------------------------------
def test_a_sentence_past_the_budget_is_dropped():
    clock = Clock()
    q = make(clock, max_lag_seconds=10.0)
    q.submit(1, "xin chào", "vi")
    clock.tick(11.0)
    job, given_up = q.take()
    assert job is None
    assert [d.sentence_id for d in given_up] == [1]
    assert "not translated in time" in given_up[0].reason


def test_a_dropped_sentence_says_how_late_it_was():
    """A reason without a number cannot be argued with."""
    clock = Clock()
    q = make(clock, max_lag_seconds=10.0)
    q.submit(1, "xin chào", "vi")
    clock.tick(25.0)
    _job, given_up = q.take()
    assert "25.0 s" in given_up[0].reason
    assert given_up[0].lag_seconds == pytest.approx(25.0)


def test_a_sentence_inside_the_budget_survives():
    clock = Clock()
    q = make(clock, max_lag_seconds=10.0)
    q.submit(1, "xin chào", "vi")
    clock.tick(9.9)
    job, given_up = q.take()
    assert job.sentence_id == 1
    assert given_up == []


def test_staleness_is_judged_on_the_way_out_not_the_way_in():
    """A sentence that waited behind a slow one is still worth translating;
    only one that waited past the budget is not."""
    clock = Clock()
    q = make(clock, max_lag_seconds=10.0)
    q.submit(1, "first", "vi")
    clock.tick(3.0)
    q.submit(2, "second", "vi")
    clock.tick(3.0)                      # first is 6s old, second is 3s
    assert q.take()[0].sentence_id == 1
    assert q.take()[0].sentence_id == 2


def test_the_stale_ones_are_skipped_to_reach_a_fresh_one():
    clock = Clock()
    q = make(clock, max_lag_seconds=10.0)
    q.submit(1, "old", "vi")
    q.submit(2, "also old", "vi")
    clock.tick(11.0)
    q.submit(3, "fresh", "vi")
    job, given_up = q.take()
    assert job.sentence_id == 3
    assert [d.sentence_id for d in given_up] == [1, 2]


def test_dropping_late_work_is_counted_separately_from_dropping_full():
    clock = Clock()
    q = make(clock, max_lag_seconds=1.0)
    q.submit(1, "x", "vi")
    clock.tick(2.0)
    q.take()
    assert q.stats.dropped_late == 1
    assert q.stats.dropped_full == 0


# ---------------------------------------------------------------------------
# The ceiling
# ---------------------------------------------------------------------------
def test_a_full_queue_drops_its_oldest():
    """The oldest is the least useful. Dropping the newest would keep a
    backlog nobody can still read and throw away the current sentence."""
    q = make(depth=2)
    q.submit(1, "one", "vi")
    q.submit(2, "two", "vi")
    refused = q.submit(3, "three", "vi")
    assert refused.sentence_id == 1
    assert "queue was full" in refused.reason
    assert [q.take()[0].sentence_id for _ in range(2)] == [2, 3]


def test_the_queue_never_grows_past_its_ceiling():
    q = make(depth=4)
    for index in range(50):
        q.submit(index, "x", "vi")
    assert len(q) == 4
    assert q.stats.dropped_full == 46


def test_the_deepest_the_queue_ever_got_is_remembered():
    q = make(depth=8)
    for index in range(5):
        q.submit(index, "x", "vi")
    q.take()
    assert q.stats.deepest == 5


def test_a_ceiling_below_one_is_refused():
    with pytest.raises(ValueError):
        make(depth=0)


def test_a_budget_of_zero_is_refused():
    with pytest.raises(ValueError):
        make(max_lag_seconds=0)


# ---------------------------------------------------------------------------
# Finishing
# ---------------------------------------------------------------------------
def test_a_finished_translation_carries_its_total_lag():
    clock = Clock()
    q = make(clock)
    q.submit(1, "xin chào", "vi")
    clock.tick(0.4)
    job, _ = q.take()
    clock.tick(0.2)
    done = q.finished(job, "こんにちは")
    assert done.lag_seconds == pytest.approx(0.6)
    assert q.stats.worst_lag_seconds == pytest.approx(0.6)


def test_a_refused_translation_is_not_counted_as_translated():
    q = make()
    q.submit(1, "xin chào", "vi")
    job, _ = q.take()
    q.finished(job, "", reason="the model returned nothing")
    assert q.stats.translated == 0


# ---------------------------------------------------------------------------
# The end of a meeting
# ---------------------------------------------------------------------------
def test_whatever_is_still_waiting_is_accounted_for():
    """A sentence left in the queue has no later chance, and saying nothing
    about it is the silent failure this project keeps paying for."""
    q = make()
    q.submit(1, "one", "vi")
    q.submit(2, "two", "vi")
    leftover = q.drain()
    assert [d.sentence_id for d in leftover] == [1, 2]
    assert all("the meeting ended" in d.reason for d in leftover)
    assert len(q) == 0


def test_draining_an_empty_queue_says_nothing():
    assert make().drain() == []


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------
def worker(translator=None, clock: Clock | None = None, **kwargs):
    return TranslationWorker(
        translator or StubTranslator(),
        queue_=make(clock, **kwargs),
        inline=True,
    )


def test_the_worker_translates_what_it_is_given():
    translator = StubTranslator("こんにちは")
    w = worker(translator)
    w.submit(1, "xin chào", "vi")
    done = w.collect()
    assert [d.sentence_id for d in done] == [1]
    assert done[0].translation == "こんにちは"
    assert translator.seen == ["xin chào"]


def test_collecting_twice_does_not_repeat_anything():
    w = worker()
    w.submit(1, "xin chào", "vi")
    assert len(w.collect()) == 1
    assert w.collect() == []


def test_a_refusal_carries_its_reason_and_the_model_answer():
    w = worker(StubTranslator(ok=False, reason="too long", raw="rambling"))
    w.submit(1, "xin chào", "vi")
    done = w.collect()[0]
    assert done.translation == ""
    assert done.reason == "too long"
    assert done.raw == "rambling"


def test_a_successful_translation_carries_no_raw():
    """It would be noise: the translation is right there."""
    w = worker(StubTranslator("こんにちは", raw="こんにちは"))
    w.submit(1, "xin chào", "vi")
    assert w.collect()[0].raw == ""


def test_a_translator_that_raises_does_not_stop_the_worker():
    """Otherwise the meeting loses every translation after the first bug,
    and says nothing about any of them."""
    class Exploding:
        def __init__(self):
            self.calls = 0

        def translate(self, text, lang_code, speaker_id=""):
            self.calls += 1
            raise RuntimeError("boom")

    translator = Exploding()
    w = worker(translator)
    w.submit(1, "one", "vi")
    w.submit(2, "two", "vi")
    done = w.collect()
    assert [d.sentence_id for d in done] == [1, 2]
    assert all(d.reason == "the translator raised" for d in done)
    assert translator.calls == 2


def test_a_sentence_dropped_on_submit_is_reported_at_once():
    """The client hears about it in the same breath as the sentence.

    Not inline: an inline worker empties the queue on every submit, so it can
    never be full. This is the backlog case, which is the only way it fills.
    """
    w = TranslationWorker(StubTranslator(), queue_=make(depth=1), inline=False)
    w.submit(1, "one", "vi")
    w.submit(2, "two", "vi")
    dropped = w.collect()
    assert [d.sentence_id for d in dropped] == [1]
    assert "queue was full" in dropped[0].reason


def test_stopping_accounts_for_the_backlog():
    w = TranslationWorker(StubTranslator(), queue_=make(), inline=False)
    w.queue.submit(1, "never translated", "vi")
    leftover = w.stop(timeout=0.1)
    assert [d.sentence_id for d in leftover] == [1]


# ---------------------------------------------------------------------------
# The numbers behind the limits
# ---------------------------------------------------------------------------
def test_the_budget_is_several_sentences_of_meeting():
    """Measured median gap between committed sentences is 3.58 s, so ten
    seconds is about three sentences ago - past the point where a translation
    can be read as belonging to the sentence above it."""
    median_gap = 3.58
    assert TRANSLATION_MAX_LAG_SECONDS / median_gap == pytest.approx(2.8, abs=0.3)


def test_the_ceiling_is_well_past_anything_measured():
    """Busiest observed rate is 1.35 sentences/s; the ceiling is twelve
    seconds of solid backlog at that rate, which the budget empties first."""
    busiest_per_second = 1.35
    assert TRANSLATION_QUEUE_DEPTH / busiest_per_second > TRANSLATION_MAX_LAG_SECONDS


# ---------------------------------------------------------------------------
# Stopping must not race the sentence it was given
#
# The last sentence of a meeting is committed and queued by the same call that
# stops the worker. Setting the stop flag first raced the thread to it: the
# thread woke, saw the flag and left the sentence to be drained, so a run
# reported "the meeting ended before this was translated" for a sentence whose
# translation was 0.2 s away.
# ---------------------------------------------------------------------------
def test_stopping_finishes_what_was_just_queued():
    w = TranslationWorker(StubTranslator("こんにちは"), queue_=make(),
                          inline=False)
    w.start()
    try:
        w.submit(1, "xin chào", "vi")
        done = w.stop(timeout=2.0)
    finally:
        w.stop(timeout=0.1)
    assert [d.sentence_id for d in done] == [1]
    assert done[0].translation == "こんにちは", done[0].reason


def test_stopping_finishes_a_short_backlog():
    w = TranslationWorker(StubTranslator("こんにちは"), queue_=make(),
                          inline=False)
    w.start()
    try:
        for index in range(1, 6):
            w.submit(index, f"line {index}", "vi")
        done = w.stop(timeout=2.0)
    finally:
        w.stop(timeout=0.1)
    assert sorted(d.sentence_id for d in done) == [1, 2, 3, 4, 5]
    assert all(d.translation for d in done), [d.reason for d in done]


def test_stopping_does_not_wait_forever_on_a_stuck_translator():
    """A backlog that cannot drain must not hold the connection open; it is
    reported as abandoned instead."""
    class Hangs:
        def translate(self, text, lang_code, speaker_id=""):
            time.sleep(5.0)
            raise AssertionError("should not get here")

    w = TranslationWorker(Hangs(), queue_=make(), inline=False)
    w.start()
    try:
        for index in range(1, 4):
            w.submit(index, f"line {index}", "vi")
        started = time.monotonic()
        done = w.stop(timeout=0.3)
        assert time.monotonic() - started < 2.0
    finally:
        w._stopping.set()
    assert done, "the abandoned sentences said nothing at all"
    assert any("meeting ended" in d.reason for d in done)
