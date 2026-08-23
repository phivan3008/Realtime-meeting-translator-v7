"""Unit tests for the server-side Silero VAD segmenter.

The Silero model itself is replaced by a scripted stub, so these tests run on
the Dev PC without torch and without downloading a model.  The real model
behaviour is covered by ``server/tests_real/test_real_vad.py`` on the GPU pod.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (
    CHUNK_BYTES,
    FINALIZE_PAUSE_MS,
    VAD_MIN_SILENCE_MS,
)
from server.pipeline.vad import (
    VAD_FRAME_BYTES,
    VAD_FRAME_MS,
    VAD_FRAME_SAMPLES,
    AudioSpan,
    Frame,
    FrameSplitter,
    SegmenterOutput,
    SegmentEvent,
    SpeechStateMachine,
    VADEvent,
    VADSegmenter,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class ScriptedVAD:
    """Stand-in for :class:`SileroVAD` returning a fixed probability sequence.

    Once the script runs out it keeps returning the last value, which makes
    "speak, then stay quiet forever" scenarios easy to write.
    """

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


def pcm_frames(count: int, value: int = 1000) -> bytes:
    """``count`` VAD frames worth of constant-amplitude 16-bit PCM."""
    return np.full(count * VAD_FRAME_SAMPLES, value, dtype="<i2").tobytes()


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
def test_silero_frame_contract():
    assert VAD_FRAME_SAMPLES == 512          # mandated by Silero v5 at 16 kHz
    assert VAD_FRAME_BYTES == 1024
    assert VAD_FRAME_MS == 32


def test_a_client_chunk_is_not_a_whole_number_of_vad_frames():
    """The reason FrameSplitter has to keep a remainder between calls."""
    assert CHUNK_BYTES % VAD_FRAME_BYTES != 0


def test_the_hangover_outlasts_the_finalize_pause():
    """A closing segment must carry enough silence for the buffer manager."""
    assert VAD_MIN_SILENCE_MS > FINALIZE_PAUSE_MS


# ---------------------------------------------------------------------------
# FrameSplitter
# ---------------------------------------------------------------------------
def test_splitter_yields_whole_frames_and_keeps_the_remainder():
    splitter = FrameSplitter()
    frames = splitter.push(bytes(CHUNK_BYTES))           # 3200 samples
    assert len(frames) == 6                              # 6 * 512 = 3072
    assert all(isinstance(f, Frame) for f in frames)
    assert all(f.samples.shape == (VAD_FRAME_SAMPLES,) for f in frames)
    assert all(len(f.pcm) == VAD_FRAME_BYTES for f in frames)
    assert splitter.pending_samples == 128


def test_splitter_joins_the_remainder_with_the_next_chunk():
    splitter = FrameSplitter()
    splitter.push(bytes(CHUNK_BYTES))                    # 6 frames, 128 left
    frames = splitter.push(bytes(CHUNK_BYTES))           # 128 + 3200 = 3328
    assert len(frames) == 6
    assert splitter.pending_samples == 256


def test_splitter_normalises_pcm_into_the_unit_range():
    splitter = FrameSplitter()
    frame = splitter.push(pcm_frames(1, value=16384))[0]
    assert frame.samples.dtype == np.float32
    assert np.allclose(frame.samples, 0.5)


def test_splitter_reset_drops_the_remainder():
    splitter = FrameSplitter()
    splitter.push(bytes(VAD_FRAME_BYTES + 100))
    splitter.reset()
    assert splitter.pending_samples == 0


def test_splitter_rejects_a_non_positive_frame_size():
    with pytest.raises(ValueError):
        FrameSplitter(frame_samples=0)


# ---------------------------------------------------------------------------
# SpeechStateMachine
# ---------------------------------------------------------------------------
def make_state(**kwargs) -> SpeechStateMachine:
    defaults = dict(threshold=0.5, min_speech_ms=96, min_silence_ms=500)
    defaults.update(kwargs)
    return SpeechStateMachine(**defaults)


def feed(state: SpeechStateMachine, probabilities):
    return [state.push(p) for p in probabilities]


def test_state_machine_derives_frame_counts_from_durations():
    state = make_state()
    assert state.min_speech_frames == 3       # 96 ms / 32 ms
    assert state.min_silence_frames == 16     # 500 ms / 32 ms -> 15.6 -> 16


def test_segment_opens_only_after_the_minimum_speech_duration():
    state = make_state()
    decisions = feed(state, [0.9, 0.9, 0.9])
    assert [d.is_speech for d in decisions] == [False, False, True]
    assert decisions[2].event is VADEvent.SPEECH_START


def test_a_short_burst_never_opens_a_segment():
    """A keyboard click fires Silero for one or two frames only."""
    state = make_state()
    decisions = feed(state, [0.9, 0.9, 0.1, 0.9, 0.1])
    assert not any(d.is_speech for d in decisions)
    assert all(d.event is None for d in decisions)


def test_the_speech_run_resets_on_a_quiet_frame():
    state = make_state()
    feed(state, [0.9, 0.9, 0.1])
    assert not state.is_speech
    assert feed(state, [0.9, 0.9])[-1].is_speech is False   # counting restarted


def test_segment_stays_open_through_a_short_pause():
    state = make_state()
    feed(state, [0.9] * 3)
    decisions = feed(state, [0.1] * 10)                     # 320 ms < 500 ms
    assert all(d.is_speech for d in decisions)
    assert all(d.event is None for d in decisions)
    assert state.trailing_silence_ms == 320


def test_segment_closes_after_the_full_silence_hangover():
    state = make_state()
    feed(state, [0.9] * 3)
    decisions = feed(state, [0.1] * 16)
    assert decisions[-1].event is VADEvent.SPEECH_END
    assert decisions[-1].is_speech is False
    assert all(d.is_speech for d in decisions[:-1])


def test_hangover_restarts_when_speech_resumes():
    state = make_state()
    feed(state, [0.9] * 3)
    feed(state, [0.1] * 10)
    state.push(0.9)
    assert state.trailing_silence_ms == 0
    assert all(d.is_speech for d in feed(state, [0.1] * 15))


def test_hysteresis_keeps_a_segment_open_between_the_two_thresholds():
    """0.4 is below the open threshold but above the close threshold."""
    state = make_state()
    assert state.neg_threshold == pytest.approx(0.35)
    feed(state, [0.9] * 3)
    assert all(d.is_speech for d in feed(state, [0.4] * 30))


def test_two_sentences_produce_two_segments():
    state = make_state()
    events = [d.event for d in feed(state, [0.9] * 3 + [0.1] * 16 + [0.9] * 3 + [0.1] * 16)]
    assert [e for e in events if e] == [
        VADEvent.SPEECH_START,
        VADEvent.SPEECH_END,
        VADEvent.SPEECH_START,
        VADEvent.SPEECH_END,
    ]


def test_close_ends_an_open_segment_and_is_idempotent():
    state = make_state()
    feed(state, [0.9] * 3)
    assert state.close() is VADEvent.SPEECH_END
    assert state.close() is None


def test_close_on_silence_reports_nothing():
    assert make_state().close() is None


def test_reset_clears_the_segment_state():
    state = make_state()
    feed(state, [0.9] * 3)
    state.reset()
    assert not state.is_speech
    assert state.frames_seen == 0


def test_state_machine_rejects_an_out_of_range_threshold():
    with pytest.raises(ValueError):
        make_state(threshold=0.0)
    with pytest.raises(ValueError):
        make_state(threshold=1.0)


# ---------------------------------------------------------------------------
# VADSegmenter
# ---------------------------------------------------------------------------
def make_segmenter(probabilities, **kwargs) -> VADSegmenter:
    defaults = dict(threshold=0.5, min_speech_ms=96, min_silence_ms=500,
                    speech_pad_ms=256)
    defaults.update(kwargs)
    return VADSegmenter(vad=ScriptedVAD(probabilities), **defaults)


def test_segmenter_forwards_nothing_while_the_room_is_quiet():
    segmenter = make_segmenter([0.02])
    out = segmenter.push(pcm_frames(20))
    assert out.pcm == b""
    assert out.events == []
    assert out.is_speech is False
    assert segmenter.stats.dropped_ratio == 1.0


def test_segmenter_forwards_audio_once_speech_opens():
    segmenter = make_segmenter([0.9])
    out = segmenter.push(pcm_frames(10))
    assert out.event_kinds == [VADEvent.SPEECH_START]
    assert out.is_speech is True
    assert out.has_audio


def test_segmenter_prepends_the_preroll_so_the_word_onset_survives():
    """8 quiet frames, then speech: the pad must bring the earlier audio back."""
    segmenter = make_segmenter([0.02] * 8 + [0.9] * 5, speech_pad_ms=256)
    out = segmenter.push(pcm_frames(13))
    # The 8 pad frames (which already hold the 2 pre-trigger candidate frames)
    # plus the 3 frames from the trigger onwards.
    assert len(out.pcm) // VAD_FRAME_BYTES == 8 + 3
    assert out.event_kinds == [VADEvent.SPEECH_START]


def test_the_preroll_is_bounded_by_speech_pad_ms():
    segmenter = make_segmenter([0.02] * 40 + [0.9] * 3, speech_pad_ms=256)
    out = segmenter.push(pcm_frames(43))
    # Only the last 8 frames before the trigger are kept, not all 40.
    assert len(out.pcm) // VAD_FRAME_BYTES == 8 + 1


def test_segmenter_forwards_the_trailing_silence_the_buffer_manager_needs():
    """The buffer manager finalises on a pause > 400 ms, so it must see one."""
    segmenter = make_segmenter([0.9] * 3 + [0.02] * 20)
    out = segmenter.push(pcm_frames(23))
    forwarded_frames = len(out.pcm) // VAD_FRAME_BYTES
    trailing_silence_ms = (forwarded_frames - 3) * VAD_FRAME_MS
    assert trailing_silence_ms >= FINALIZE_PAUSE_MS
    assert out.event_kinds == [VADEvent.SPEECH_START, VADEvent.SPEECH_END]
    assert out.is_speech is False


def test_segmenter_drops_the_silence_after_a_segment_closed():
    segmenter = make_segmenter([0.9] * 3 + [0.02] * 16 + [0.02] * 50)
    first = segmenter.push(pcm_frames(19))
    second = segmenter.push(pcm_frames(50))
    assert first.has_audio
    assert second.pcm == b""
    assert segmenter.stats.dropped_ratio > 0.5


def test_segmenter_works_across_client_chunk_boundaries():
    """Frames must not be lost when speech straddles two 200 ms chunks."""
    segmenter = make_segmenter([0.9])
    outputs = [segmenter.push(bytes(CHUNK_BYTES)) for _ in range(4)]
    total_frames = sum(len(o.pcm) for o in outputs) // VAD_FRAME_BYTES
    assert segmenter.stats.frames_total == 25          # 4 * 3200 // 512
    assert total_frames == 25                          # nothing dropped mid-segment


def test_segmenter_reports_only_the_probabilities_of_this_chunk():
    segmenter = make_segmenter([0.9])
    out = segmenter.push(pcm_frames(4))
    assert len(out.probabilities) == 4
    assert out.max_probability == pytest.approx(0.9)


def test_segmenter_counts_segments_and_speech_frames():
    segmenter = make_segmenter([0.9] * 3 + [0.02] * 16 + [0.9] * 3 + [0.02] * 16)
    segmenter.push(pcm_frames(38))
    assert segmenter.stats.segments == 2
    assert segmenter.stats.frames_total == 38
    assert 0.0 < segmenter.stats.speech_ratio < 1.0


def test_segmenter_close_emits_a_final_speech_end():
    segmenter = make_segmenter([0.9])
    segmenter.push(pcm_frames(5))
    assert segmenter.close().event_kinds == [VADEvent.SPEECH_END]


def test_segmenter_close_on_silence_emits_nothing():
    segmenter = make_segmenter([0.02])
    segmenter.push(pcm_frames(5))
    assert segmenter.close().events == []


def test_segmenter_reset_clears_everything_including_the_model_state():
    vad = ScriptedVAD([0.9])
    segmenter = VADSegmenter(vad=vad)
    segmenter.push(pcm_frames(5))
    segmenter.reset()
    assert vad.resets == 1
    assert segmenter.stats.frames_total == 0
    assert segmenter.state.is_speech is False
    assert segmenter.splitter.pending_samples == 0
    assert segmenter.position_ms == 0


def test_segmenter_preserves_the_audio_it_forwards():
    """Whatever comes out must be a byte-exact slice of what went in."""
    segmenter = make_segmenter([0.9], speech_pad_ms=0)
    pcm = np.arange(3 * VAD_FRAME_SAMPLES, dtype="<i2").tobytes()
    out = segmenter.push(pcm)
    assert out.pcm == pcm[-len(out.pcm):]


# ---------------------------------------------------------------------------
# Timestamps - what the Stream Buffer Manager consumes
# ---------------------------------------------------------------------------
def test_position_advances_by_the_frames_actually_consumed():
    segmenter = make_segmenter([0.02])
    out = segmenter.push(bytes(CHUNK_BYTES))       # 6 whole frames, 128 samples left
    assert out.position_ms == 6 * VAD_FRAME_MS
    assert segmenter.position_ms == 6 * VAD_FRAME_MS


def test_speech_start_is_timestamped_at_the_first_forwarded_sample():
    """Not at the frame that triggered: the pre-roll comes before it."""
    segmenter = make_segmenter([0.02] * 8 + [0.9] * 3, speech_pad_ms=256)
    out = segmenter.push(pcm_frames(11))
    start = out.events[0]
    assert start.kind is VADEvent.SPEECH_START
    # Trigger lands on frame 10 (0-indexed), 8 pad frames precede it.
    assert start.at_ms == (10 - 8) * VAD_FRAME_MS


def test_speech_end_is_timestamped_just_past_the_last_forwarded_sample():
    segmenter = make_segmenter([0.9] * 3 + [0.02] * 16)
    out = segmenter.push(pcm_frames(19))
    end = out.events[-1]
    assert end.kind is VADEvent.SPEECH_END
    assert end.at_ms == 18 * VAD_FRAME_MS          # frame 18 closed the segment


def test_segment_bounds_match_the_forwarded_audio_duration():
    segmenter = make_segmenter([0.9] * 3 + [0.02] * 16, speech_pad_ms=0)
    out = segmenter.push(pcm_frames(19))
    start, end = out.events
    forwarded_ms = len(out.pcm) / 2 / 16_000 * 1000
    assert end.at_ms - start.at_ms == pytest.approx(forwarded_ms)


def test_timestamps_keep_running_across_chunks():
    segmenter = make_segmenter([0.02] * 30 + [0.9] * 3, speech_pad_ms=0)
    segmenter.push(pcm_frames(30))
    out = segmenter.push(pcm_frames(3))
    assert out.events[0].at_ms == 32 * VAD_FRAME_MS
    assert out.position_ms == 33 * VAD_FRAME_MS


def test_close_timestamps_the_final_event_at_the_stream_end():
    segmenter = make_segmenter([0.9])
    segmenter.push(pcm_frames(5))
    out = segmenter.close()
    assert out.events == [
        SegmentEvent(kind=VADEvent.SPEECH_END, at_ms=5 * VAD_FRAME_MS)
    ]


def test_empty_segmenter_output_defaults_are_safe():
    out = SegmenterOutput()
    assert not out.has_audio
    assert out.max_probability == 0.0
    assert out.event_kinds == []


# ---------------------------------------------------------------------------
# Audio spans - what the buffer manager consumes
# ---------------------------------------------------------------------------
def test_a_segment_that_fits_in_one_chunk_produces_one_span():
    segmenter = make_segmenter([0.9] * 3 + [0.02] * 16 + [0.02], speech_pad_ms=0)
    out = segmenter.push(pcm_frames(20))
    assert len(out.spans) == 1
    assert out.spans[0].opens_segment and out.spans[0].closes_segment


def test_spans_concatenate_back_to_the_flat_pcm():
    segmenter = make_segmenter([0.9])
    out = segmenter.push(pcm_frames(10))
    assert b"".join(span.pcm for span in out.spans) == out.pcm


def test_a_span_reports_its_own_position_and_length():
    segmenter = make_segmenter([0.9], speech_pad_ms=0)
    out = segmenter.push(pcm_frames(5))
    span = out.spans[0]
    assert span.start_ms == 2 * VAD_FRAME_MS         # opens on the third frame
    assert span.duration_ms == pytest.approx(3 * VAD_FRAME_MS)
    assert span.end_ms == pytest.approx(5 * VAD_FRAME_MS)


def test_a_segment_spanning_chunks_opens_once_and_closes_once():
    segmenter = make_segmenter([0.9] * 40 + [0.02])
    spans = []
    for _ in range(10):
        spans += segmenter.push(bytes(CHUNK_BYTES)).spans
    assert sum(1 for s in spans if s.opens_segment) == 1
    assert sum(1 for s in spans if s.closes_segment) == 1
    assert spans[0].opens_segment and spans[-1].closes_segment


def test_the_middle_spans_of_a_long_segment_carry_no_flags():
    segmenter = make_segmenter([0.9] * 40 + [0.02])
    spans = []
    for _ in range(10):
        spans += segmenter.push(bytes(CHUNK_BYTES)).spans
    for span in spans[1:-1]:
        assert not span.opens_segment and not span.closes_segment


def test_spans_are_contiguous_in_time():
    segmenter = make_segmenter([0.9] * 40 + [0.02])
    spans = []
    for _ in range(10):
        spans += segmenter.push(bytes(CHUNK_BYTES)).spans
    for earlier, later in zip(spans, spans[1:]):
        assert later.start_ms == pytest.approx(earlier.end_ms)


def test_a_close_on_an_audioless_chunk_still_carries_the_marker():
    """Regression: the buffer manager never saw the sentence end.

    When the hangover runs out on the first frame of a chunk, that chunk
    forwards no audio at all. The span is empty, but it still has to say
    closes_segment or the utterance stays open forever.
    """
    segmenter = make_segmenter([0.9] * 3 + [0.02] * 16 + [0.02])
    spans = []
    for _ in range(4):
        spans += segmenter.push(bytes(CHUNK_BYTES)).spans
    closing = [s for s in spans if s.closes_segment]
    assert len(closing) == 1
    assert closing[0].pcm == b""
    assert closing[0].start_ms == 18 * VAD_FRAME_MS


def test_silence_produces_no_spans_at_all():
    assert make_segmenter([0.02]).push(pcm_frames(20)).spans == []


def test_an_audio_span_is_immutable():
    span = AudioSpan(pcm=bytes(2), start_ms=0.0)
    with pytest.raises(AttributeError):
        span.pcm = b""
