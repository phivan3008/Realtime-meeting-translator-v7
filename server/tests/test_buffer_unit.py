"""Unit tests for the Stream Buffer Manager.

Pure logic: the input is a hand-built ``SegmenterOutput``, so no model, no
socket and no GPU are involved.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import SAMPLE_RATE, SAMPLE_WIDTH
from server.pipeline.buffer import (
    FRAME_BYTES,
    BufferManager,
    FinalizeReason,
    PartialWindow,
    bytes_to_ms,
    ms_to_bytes,
)
from server.pipeline.vad import AudioSpan, SegmenterOutput

# Small numbers keep the tests readable: 1 s limit, 300 ms partial cadence.
MAX_MS = 1_000
PARTIAL_MS = 300
SEARCH_MS = 200


def manager(**kwargs) -> BufferManager:
    defaults = dict(max_duration_ms=MAX_MS, partial_interval_ms=PARTIAL_MS,
                    split_search_ms=SEARCH_MS)
    defaults.update(kwargs)
    return BufferManager(**defaults)


def tone(ms: float, amplitude: int = 8000) -> bytes:
    """Constant-amplitude PCM, so energy is uniform unless a test says otherwise."""
    samples = int(ms * SAMPLE_RATE / 1000)
    return np.full(samples, amplitude, dtype="<i2").tobytes()


def ramp(ms: float) -> bytes:
    """Every sample distinct, so byte-for-byte preservation is provable."""
    samples = int(ms * SAMPLE_RATE / 1000)
    return (np.arange(samples, dtype=np.int64) % 30_000).astype("<i2").tobytes()


def output(*spans: AudioSpan) -> SegmenterOutput:
    return SegmenterOutput(spans=list(spans))


def span(pcm: bytes, start_ms: float, opens: bool = False,
         closes: bool = False) -> AudioSpan:
    return AudioSpan(pcm=pcm, start_ms=start_ms, opens_segment=opens,
                     closes_segment=closes)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------
def test_ms_to_bytes_never_splits_a_sample():
    for ms in (0.1, 1, 33.3, 200, 7_000):
        assert ms_to_bytes(ms) % SAMPLE_WIDTH == 0


def test_the_conversions_round_trip():
    assert bytes_to_ms(ms_to_bytes(1_000)) == pytest.approx(1_000, abs=0.1)
    assert ms_to_bytes(200) == 6_400            # one client chunk


def test_construction_rejects_impossible_settings():
    with pytest.raises(ValueError, match="max_duration_ms"):
        manager(max_duration_ms=0)
    with pytest.raises(ValueError, match="split_search_ms"):
        manager(max_duration_ms=500, split_search_ms=500)


# ---------------------------------------------------------------------------
# The ordinary case: a pause ends the sentence
# ---------------------------------------------------------------------------
def test_nothing_is_emitted_for_an_empty_output():
    result = manager().push(output())
    assert result.finals == []
    assert result.partial is None
    assert result.has_work is False


def test_a_closed_segment_becomes_one_utterance():
    buffer = manager()
    result = buffer.push(output(span(tone(500), 1_000, opens=True, closes=True)))
    assert len(result.finals) == 1
    utterance = result.finals[0]
    assert utterance.index == 0
    assert utterance.reason is FinalizeReason.PAUSE
    assert utterance.start_ms == 1_000
    assert utterance.duration_ms == pytest.approx(500)
    assert utterance.end_ms == pytest.approx(1_500)
    assert utterance.continues_previous is False
    assert buffer.is_open is False


def test_a_segment_spread_over_several_chunks_is_joined():
    buffer = manager()
    assert buffer.push(output(span(ramp(200), 0, opens=True))).finals == []
    assert buffer.push(output(span(ramp(200), 200))).finals == []
    result = buffer.push(output(span(ramp(200), 400, closes=True)))
    assert len(result.finals) == 1
    assert result.finals[0].pcm == ramp(200) * 3
    assert result.finals[0].duration_ms == pytest.approx(600)


def test_two_sentences_produce_two_utterances_with_rising_indexes():
    buffer = manager()
    first = buffer.push(output(span(tone(300), 0, opens=True, closes=True)))
    second = buffer.push(output(span(tone(300), 2_000, opens=True, closes=True)))
    assert [u.index for u in first.finals + second.finals] == [0, 1]
    assert second.finals[0].start_ms == 2_000


def test_silence_between_sentences_is_not_buffered():
    """Only the spans the VAD forwarded ever reach here."""
    buffer = manager()
    buffer.push(output(span(tone(300), 0, opens=True, closes=True)))
    assert buffer.open_duration_ms == 0
    assert buffer.is_open is False


def test_a_missing_speech_end_does_not_glue_two_sentences_together():
    buffer = manager()
    buffer.push(output(span(tone(200), 0, opens=True)))          # never closed
    result = buffer.push(output(span(tone(200), 5_000, opens=True, closes=True)))
    assert [u.reason for u in result.finals] == [
        FinalizeReason.PAUSE, FinalizeReason.PAUSE
    ]
    assert result.finals[0].start_ms == 0
    assert result.finals[1].start_ms == 5_000


# ---------------------------------------------------------------------------
# Max duration
# ---------------------------------------------------------------------------
def test_a_long_monologue_is_cut_at_the_limit():
    buffer = manager()
    result = buffer.push(output(span(tone(2_500), 0, opens=True)))
    assert [u.reason for u in result.finals] == [
        FinalizeReason.MAX_DURATION, FinalizeReason.MAX_DURATION
    ]
    assert all(u.duration_ms <= MAX_MS for u in result.finals)
    assert buffer.is_open is True                 # the speaker is still going


def test_the_continuation_is_marked_so_the_translator_knows():
    buffer = manager()
    result = buffer.push(output(span(tone(2_500), 0, opens=True)))
    assert result.finals[0].continues_previous is False
    assert result.finals[1].continues_previous is True
    final = buffer.flush().finals[0]
    assert final.continues_previous is True


def test_a_length_cut_neither_loses_nor_repeats_audio():
    """The single most important property: the audio is a clean partition."""
    buffer = manager()
    original = ramp(2_500)
    result = buffer.push(output(span(original, 0, opens=True)))
    rest = buffer.flush()
    pieces = result.finals + rest.finals
    assert b"".join(u.pcm for u in pieces) == original
    assert sum(len(u.pcm) for u in pieces) == len(original)


def test_the_pieces_line_up_end_to_start_in_time():
    buffer = manager()
    result = buffer.push(output(span(tone(2_500), 4_000, opens=True)))
    pieces = result.finals + buffer.flush().finals
    for earlier, later in zip(pieces, pieces[1:]):
        assert later.start_ms == pytest.approx(earlier.end_ms)
    assert pieces[0].start_ms == 4_000
    assert pieces[-1].end_ms == pytest.approx(6_500)


def test_the_cut_lands_on_the_quietest_moment_not_on_the_hard_limit():
    """A gap between words inside the search window should attract the cut."""
    pcm = bytearray(tone(1_400, amplitude=12_000))
    # Frame aligned: the scan walks the search window one 32 ms frame at a
    # time, so a gap straddling two frames only makes both of them half quiet.
    gap_start = ms_to_bytes(MAX_MS - SEARCH_MS) + 2 * FRAME_BYTES
    pcm[gap_start:gap_start + FRAME_BYTES] = bytes(FRAME_BYTES)   # silence
    result = manager().push(output(span(bytes(pcm), 0, opens=True)))
    # The cut lands just after the quiet frame, so the silence stays with the
    # first half rather than opening the next utterance.
    assert result.finals[0].duration_ms == pytest.approx(
        bytes_to_ms(gap_start + FRAME_BYTES)
    )
    assert result.finals[0].duration_ms < MAX_MS


def test_the_cut_falls_back_to_the_limit_when_the_audio_is_uniformly_loud():
    result = manager().push(output(span(tone(1_400), 0, opens=True)))
    duration = result.finals[0].duration_ms
    assert MAX_MS - SEARCH_MS <= duration <= MAX_MS


def test_the_search_window_bounds_how_far_back_the_cut_can_go():
    """A quiet patch older than split_search_ms must not attract the cut."""
    pcm = bytearray(tone(1_400, amplitude=12_000))
    pcm[ms_to_bytes(100):ms_to_bytes(100) + FRAME_BYTES] = bytes(FRAME_BYTES)
    result = manager().push(output(span(bytes(pcm), 0, opens=True)))
    assert result.finals[0].duration_ms > MAX_MS - SEARCH_MS


def test_a_pause_right_after_a_length_cut_still_finalises_the_remainder():
    buffer = manager()
    result = buffer.push(output(span(tone(1_200), 0, opens=True, closes=True)))
    assert [u.reason for u in result.finals] == [
        FinalizeReason.MAX_DURATION, FinalizeReason.PAUSE
    ]
    assert buffer.is_open is False


# ---------------------------------------------------------------------------
# Partials
# ---------------------------------------------------------------------------
def test_no_partial_before_the_interval_has_passed():
    assert manager().push(output(span(tone(100), 0, opens=True))).partial is None


def test_a_partial_appears_once_the_interval_is_reached():
    buffer = manager()
    result = buffer.push(output(span(tone(PARTIAL_MS), 0, opens=True)))
    assert result.partial is not None
    assert result.partial.index == 0
    assert result.partial.start_ms == 0
    assert result.partial.duration_ms == pytest.approx(PARTIAL_MS)


def test_a_partial_carries_everything_said_so_far_not_just_the_new_audio():
    buffer = manager()
    buffer.push(output(span(ramp(PARTIAL_MS), 0, opens=True)))
    result = buffer.push(output(span(ramp(PARTIAL_MS), PARTIAL_MS)))
    assert result.partial.pcm == ramp(PARTIAL_MS) * 2


def test_partials_are_spaced_by_the_interval():
    buffer = manager()
    seen = []
    for i in range(10):
        result = buffer.push(output(span(tone(100), i * 100, opens=(i == 0))))
        if result.partial is not None:
            seen.append(result.partial.duration_ms)
    assert seen == pytest.approx([300, 600, 900])


def test_a_closed_utterance_produces_no_partial():
    buffer = manager()
    result = buffer.push(output(span(tone(500), 0, opens=True, closes=True)))
    assert result.finals and result.partial is None


def test_the_partial_index_matches_the_utterance_it_will_become():
    buffer = manager()
    buffer.push(output(span(tone(500), 0, opens=True, closes=True)))
    result = buffer.push(output(span(tone(PARTIAL_MS), 2_000, opens=True)))
    assert result.partial.index == 1
    assert buffer.flush().finals[0].index == 1


# ---------------------------------------------------------------------------
# Speaker change and end of stream
# ---------------------------------------------------------------------------
def test_a_speaker_change_finalises_the_open_utterance():
    buffer = manager()
    buffer.push(output(span(tone(400), 0, opens=True)))
    result = buffer.notify_speaker_change()
    assert [u.reason for u in result.finals] == [FinalizeReason.SPEAKER_CHANGE]
    assert buffer.is_open is False


def test_a_speaker_change_with_nothing_open_is_a_no_op():
    assert manager().notify_speaker_change().finals == []


def test_flush_commits_what_is_still_open():
    buffer = manager()
    buffer.push(output(span(tone(400), 1_000, opens=True)))
    result = buffer.flush()
    assert [u.reason for u in result.finals] == [FinalizeReason.END_OF_STREAM]
    assert result.finals[0].start_ms == 1_000
    assert buffer.is_open is False


def test_flush_on_an_idle_buffer_emits_nothing():
    assert manager().flush().finals == []


def test_flush_is_idempotent():
    buffer = manager()
    buffer.push(output(span(tone(400), 0, opens=True)))
    assert buffer.flush().finals
    assert buffer.flush().finals == []


# ---------------------------------------------------------------------------
# Stats and reset
# ---------------------------------------------------------------------------
def test_stats_count_each_finalize_reason():
    buffer = manager()
    buffer.push(output(span(tone(1_200), 0, opens=True, closes=True)))
    buffer.push(output(span(tone(300), 5_000, opens=True)))
    buffer.flush()
    assert buffer.stats.utterances == 3
    assert buffer.stats.finalized_by == {
        "max_duration": 1, "pause": 1, "end_of_stream": 1
    }


def test_stats_count_the_bytes_that_arrived():
    buffer = manager()
    buffer.push(output(span(tone(500), 0, opens=True, closes=True)))
    assert buffer.stats.bytes_in == ms_to_bytes(500)


def test_reset_clears_everything_for_the_next_meeting():
    buffer = manager()
    buffer.push(output(span(tone(400), 0, opens=True)))
    buffer.reset()
    assert buffer.is_open is False
    assert buffer.stats.utterances == 0
    assert buffer.open_index == 0


# ---------------------------------------------------------------------------
# The running prediction only decodes the tail of a long sentence
#
# Uncapped it re-decodes from the start every 600 ms, which over ten minutes
# cost 97.8 s against 21.6 s for every committed sentence put together, with
# one pass reaching 4.7 s while the slowest sentence took 0.4 s.
# ---------------------------------------------------------------------------
def window(seconds: float, start_ms: float = 0.0) -> PartialWindow:
    return PartialWindow(index=0, start_ms=start_ms,
                         pcm=bytes(int(seconds * SAMPLE_RATE) * SAMPLE_WIDTH))


def test_a_short_window_is_left_alone():
    """Most sentences never reach the cap, and copying them would be waste."""
    short = window(2.0)
    assert short.tail(4.0) is short


def test_a_window_exactly_at_the_cap_is_left_alone():
    assert window(4.0).tail(4.0).duration_ms == pytest.approx(4000.0)


def test_a_long_window_is_cut_to_the_cap():
    assert window(7.0).tail(4.0).duration_ms == pytest.approx(4000.0)


def test_the_tail_is_the_end_not_the_beginning():
    """It has to be the end: the point is to show what is being said now."""
    # A second of silence then a second of 0x11, so the tail must be the
    # second half. Sized from SAMPLE_RATE: at 8000 samples each half is only
    # half a second, the whole window fits inside the cap, and the test passes
    # by never cutting anything.
    full = PartialWindow(
        index=0, start_ms=0.0,
        pcm=b"\x00\x00" * SAMPLE_RATE + b"\x11\x11" * SAMPLE_RATE)
    assert full.duration_ms == pytest.approx(2000.0)
    assert set(full.tail(1.0).pcm) == {0x11}


def test_the_cut_window_says_where_it_now_starts():
    """Or the timestamps stop lining up with the audio the server received."""
    cut = window(7.0, start_ms=1000.0).tail(4.0)
    assert cut.start_ms == pytest.approx(4000.0)
    assert cut.end_ms == pytest.approx(8000.0)


def test_the_cut_window_keeps_its_index():
    assert window(7.0).tail(4.0).index == 0


def test_the_cut_lands_on_a_sample_boundary():
    """Half a sample would shift every value after it by one byte."""
    assert len(window(7.0).tail(3.3333).pcm) % SAMPLE_WIDTH == 0


def test_a_cap_of_zero_or_less_changes_nothing():
    """Rather than emptying the window and transcribing silence."""
    full = window(7.0)
    assert full.tail(0.0) is full
    assert full.tail(-1.0) is full


def test_the_configured_cap_bounds_the_worst_pass():
    """The 4.7 s pass that stalled a run could not happen under this."""
    from server.config import PARTIAL_WINDOW_SECONDS
    assert window(30.0).tail(PARTIAL_WINDOW_SECONDS).duration_ms / 1000.0 \
        == pytest.approx(PARTIAL_WINDOW_SECONDS)


def test_the_cap_is_below_the_max_utterance():
    """Above it the cap would never fire and nothing would change."""
    from server.config import FINALIZE_MAX_DURATION_MS, PARTIAL_WINDOW_SECONDS
    assert PARTIAL_WINDOW_SECONDS * 1000 < FINALIZE_MAX_DURATION_MS
