"""Unit tests for the client Silero VAD gating logic.

The Silero model itself is replaced by a scripted stub, so these tests run on
the Dev PC without torch and without downloading a model.  The real model
behaviour is covered by ``client/tests_real/test_real_vad.py`` on the Windows
Client PC.

Run with::

    .venv\\Scripts\\python.exe -m pytest client/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.audio.vad import (
    VAD_FRAME_BYTES,
    VAD_FRAME_MS,
    VAD_FRAME_SAMPLES,
    Frame,
    FrameSplitter,
    GateOutput,
    SpeechStateMachine,
    VADEvent,
    VADGate,
)
from client.config import CHUNK_BYTES


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
# Frame contract
# ---------------------------------------------------------------------------
def test_silero_frame_contract():
    assert VAD_FRAME_SAMPLES == 512          # mandated by Silero v5 at 16 kHz
    assert VAD_FRAME_BYTES == 1024
    assert VAD_FRAME_MS == 32


def test_a_capture_chunk_is_not_a_whole_number_of_vad_frames():
    """The reason FrameSplitter has to keep a remainder between calls."""
    assert CHUNK_BYTES % VAD_FRAME_BYTES != 0


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
# VADGate
# ---------------------------------------------------------------------------
def make_gate(probabilities, **kwargs) -> VADGate:
    defaults = dict(threshold=0.5, min_speech_ms=96, min_silence_ms=500,
                    speech_pad_ms=256)
    defaults.update(kwargs)
    return VADGate(vad=ScriptedVAD(probabilities), **defaults)


def test_gate_forwards_nothing_while_the_room_is_quiet():
    gate = make_gate([0.02])
    out = gate.push(pcm_frames(20))
    assert out.pcm == b""
    assert out.events == []
    assert out.is_speech is False
    assert gate.stats.bandwidth_saved == 1.0


def test_gate_forwards_audio_once_speech_opens():
    gate = make_gate([0.9])
    out = gate.push(pcm_frames(10))
    assert out.events == [VADEvent.SPEECH_START]
    assert out.is_speech is True
    assert out.has_audio


def test_gate_prepends_the_preroll_so_the_word_onset_survives():
    """8 quiet frames, then speech: the pad must bring the earlier audio back."""
    gate = make_gate([0.02] * 8 + [0.9] * 5, speech_pad_ms=256)
    out = gate.push(pcm_frames(13))
    # The 8 pad frames (which already hold the 2 pre-trigger candidate frames)
    # plus the 3 frames from the trigger onwards.
    assert len(out.pcm) // VAD_FRAME_BYTES == 8 + 3
    assert out.events == [VADEvent.SPEECH_START]


def test_the_preroll_is_bounded_by_speech_pad_ms():
    gate = make_gate([0.02] * 40 + [0.9] * 3, speech_pad_ms=256)
    out = gate.push(pcm_frames(43))
    # Only the last 8 frames before the trigger are kept, not all 40.
    assert len(out.pcm) // VAD_FRAME_BYTES == 8 + 1


def test_gate_forwards_the_trailing_silence_the_server_needs():
    """The server finalises on a pause > 400 ms, so it must receive one."""
    gate = make_gate([0.9] * 3 + [0.02] * 20)
    out = gate.push(pcm_frames(23))
    forwarded_frames = len(out.pcm) // VAD_FRAME_BYTES
    trailing_silence_ms = (forwarded_frames - 3) * VAD_FRAME_MS
    assert trailing_silence_ms >= 400
    assert out.events == [VADEvent.SPEECH_START, VADEvent.SPEECH_END]
    assert out.is_speech is False


def test_gate_drops_the_silence_after_a_segment_closed():
    gate = make_gate([0.9] * 3 + [0.02] * 16 + [0.02] * 50)
    first = gate.push(pcm_frames(19))
    second = gate.push(pcm_frames(50))
    assert first.has_audio
    assert second.pcm == b""
    assert gate.stats.bandwidth_saved > 0.5


def test_gate_works_across_capture_chunk_boundaries():
    """Frames must not be lost when speech straddles two 200 ms chunks."""
    gate = make_gate([0.9])
    outputs = [gate.push(bytes(CHUNK_BYTES)) for _ in range(4)]
    total_frames = sum(len(o.pcm) for o in outputs) // VAD_FRAME_BYTES
    assert gate.stats.frames_total == 25          # 4 * 3200 // 512
    assert total_frames == 25                     # nothing dropped mid-segment


def test_gate_reports_only_the_probabilities_of_this_chunk():
    gate = make_gate([0.9])
    out = gate.push(pcm_frames(4))
    assert len(out.probabilities) == 4
    assert out.max_probability == pytest.approx(0.9)


def test_gate_counts_segments_and_speech_frames():
    gate = make_gate([0.9] * 3 + [0.02] * 16 + [0.9] * 3 + [0.02] * 16)
    gate.push(pcm_frames(38))
    assert gate.stats.segments == 2
    assert gate.stats.frames_total == 38
    assert 0.0 < gate.stats.speech_ratio < 1.0


def test_gate_close_emits_a_final_speech_end():
    gate = make_gate([0.9])
    gate.push(pcm_frames(5))
    assert gate.close().events == [VADEvent.SPEECH_END]


def test_gate_close_on_silence_emits_nothing():
    gate = make_gate([0.02])
    gate.push(pcm_frames(5))
    assert gate.close().events == []


def test_gate_reset_clears_everything_including_the_model_state():
    vad = ScriptedVAD([0.9])
    gate = VADGate(vad=vad)
    gate.push(pcm_frames(5))
    gate.reset()
    assert vad.resets == 1
    assert gate.stats.frames_total == 0
    assert gate.state.is_speech is False
    assert gate.splitter.pending_samples == 0


def test_gate_preserves_the_audio_it_forwards():
    """Whatever comes out must be a byte-exact slice of what went in."""
    gate = make_gate([0.9], speech_pad_ms=0)
    pcm = np.arange(3 * VAD_FRAME_SAMPLES, dtype="<i2").tobytes()
    out = gate.push(pcm)
    assert out.pcm == pcm[-len(out.pcm):]


def test_empty_gate_output_defaults_are_safe():
    out = GateOutput()
    assert not out.has_audio
    assert out.max_probability == 0.0
