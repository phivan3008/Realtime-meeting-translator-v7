"""Unit tests for the client audio capture module.

These tests never touch a sound card: they cover the pure-Python plumbing
(chunk assembly, PCM conversion, resampling) so the logic can be validated on
the Dev PC.  The hardware behaviour is covered separately by
``tests_real/test_real_audio_capture.py`` on the Windows Client PC.

Run with::

    .venv\\Scripts\\python.exe -m pytest client/tests -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.audio.capture import ChunkAssembler, DeviceInfo, pcm16_to_wav_bytes
from client.audio.resampler import (
    StreamResampler,
    bytes_to_float32,
    downmix_to_mono,
    float32_to_int16_bytes,
)
from client.config import CHUNK_BYTES, CHUNK_SAMPLES, TARGET_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Configuration contract
# ---------------------------------------------------------------------------
def test_chunk_constants_match_the_design_contract():
    assert TARGET_SAMPLE_RATE == 16_000
    assert CHUNK_SAMPLES == 3_200            # 200 ms
    assert CHUNK_BYTES == 6_400              # 16-bit mono


# ---------------------------------------------------------------------------
# ChunkAssembler
# ---------------------------------------------------------------------------
def test_assembler_splits_a_large_buffer_into_exact_chunks():
    assembler = ChunkAssembler(chunk_bytes=100)
    chunks = assembler.push(bytes(250))
    assert [len(c) for c in chunks] == [100, 100]
    assert assembler.pending_bytes == 50


def test_assembler_joins_small_buffers_across_calls():
    assembler = ChunkAssembler(chunk_bytes=100)
    assert assembler.push(bytes(60)) == []
    chunks = assembler.push(bytes(60))
    assert len(chunks) == 1
    assert assembler.pending_bytes == 20


def test_assembler_preserves_byte_order_across_chunks():
    assembler = ChunkAssembler(chunk_bytes=4)
    payload = bytes(range(12))
    chunks = assembler.push(payload)
    assert b"".join(chunks) == payload


def test_assembler_flush_pads_the_trailing_chunk_with_silence():
    assembler = ChunkAssembler(chunk_bytes=10)
    assembler.push(b"\x01\x02\x03")
    tail = assembler.flush(pad=True)
    assert tail == b"\x01\x02\x03" + b"\x00" * 7
    assert assembler.flush() is None         # buffer is empty afterwards


def test_assembler_flush_without_padding_returns_the_raw_remainder():
    assembler = ChunkAssembler(chunk_bytes=10)
    assembler.push(b"\x01\x02\x03")
    assert assembler.flush(pad=False) == b"\x01\x02\x03"


def test_assembler_rejects_a_non_positive_chunk_size():
    with pytest.raises(ValueError):
        ChunkAssembler(chunk_bytes=0)


# ---------------------------------------------------------------------------
# PCM conversion
# ---------------------------------------------------------------------------
def test_int16_bytes_survive_a_decode_encode_round_trip():
    original = np.array([0, 1000, -1000, 32767, -32767], dtype="<i2")
    decoded = bytes_to_float32(original.tobytes(), "int16")
    restored = np.frombuffer(float32_to_int16_bytes(decoded), dtype="<i2")
    assert np.max(np.abs(restored - original)) <= 1


def test_float32_input_is_clipped_to_the_int16_range():
    loud = np.array([2.0, -2.0], dtype="<f4")
    restored = np.frombuffer(
        float32_to_int16_bytes(bytes_to_float32(loud.tobytes(), "float32")),
        dtype="<i2",
    )
    assert restored.tolist() == [32767, -32767]


def test_unknown_sample_format_is_rejected():
    with pytest.raises(ValueError):
        bytes_to_float32(b"\x00\x00", "int24")


def test_downmix_averages_the_stereo_channels():
    interleaved = np.array([1.0, 0.0, 0.5, 0.5, -1.0, 1.0], dtype=np.float32)
    assert downmix_to_mono(interleaved, 2).tolist() == [0.5, 0.5, 0.0]


def test_downmix_is_a_no_op_for_mono():
    mono = np.array([0.1, 0.2], dtype=np.float32)
    assert downmix_to_mono(mono, 1) is mono


def test_downmix_drops_an_incomplete_trailing_frame():
    interleaved = np.array([1.0, 1.0, 0.0], dtype=np.float32)   # 1.5 frames
    assert downmix_to_mono(interleaved, 2).tolist() == [1.0]


# ---------------------------------------------------------------------------
# StreamResampler
# ---------------------------------------------------------------------------
def _sine(freq: float, seconds: float, rate: int, channels: int) -> bytes:
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    mono = 0.5 * np.sin(2 * math.pi * freq * t)
    frames = np.repeat(mono[:, None], channels, axis=1).reshape(-1)
    return (frames * 32767).astype("<i2").tobytes()


def _dominant_frequency(pcm: bytes, rate: int) -> float:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    return float(np.fft.rfftfreq(samples.size, 1.0 / rate)[int(np.argmax(spectrum))])


def test_resampler_uses_a_real_backend_when_rates_differ():
    resampler = StreamResampler(48_000, 2, "int16")
    assert resampler.backend in {"soxr", "scipy"}


def test_resampler_is_a_passthrough_when_the_rate_already_matches():
    resampler = StreamResampler(16_000, 1, "int16")
    assert resampler.backend == "passthrough"
    pcm = _sine(440, 0.1, 16_000, 1)
    assert resampler.process(pcm) == pcm


def test_resampler_converts_48k_stereo_to_16k_mono_with_the_right_length():
    resampler = StreamResampler(48_000, 2, "int16")
    out = resampler.process(_sine(440, 1.0, 48_000, 2))
    out += resampler.flush()
    samples = len(out) // 2
    # The filter delay costs a handful of samples; 1 % tolerance is plenty.
    assert abs(samples - 16_000) < 160


def test_resampler_preserves_the_tone_frequency():
    resampler = StreamResampler(48_000, 2, "int16")
    out = resampler.process(_sine(440, 1.0, 48_000, 2)) + resampler.flush()
    assert abs(_dominant_frequency(out, 16_000) - 440.0) < 5.0


def test_resampler_output_is_continuous_across_chunk_boundaries():
    """A tone fed in 20 ms slices must not gain clicks at the seams."""
    resampler = StreamResampler(48_000, 1, "int16")
    pcm = _sine(440, 1.0, 48_000, 1)
    slice_bytes = 48_000 * 2 * 20 // 1000
    out = b"".join(
        resampler.process(pcm[i : i + slice_bytes])
        for i in range(0, len(pcm), slice_bytes)
    )
    out += resampler.flush()
    assert abs(_dominant_frequency(out, 16_000) - 440.0) < 5.0
    # A click at a seam shows up as a sample-to-sample jump larger than the
    # steepest slope the tone itself can produce: 2*pi*f/rate * amplitude.
    max_slope = 2 * math.pi * 440 / 16_000 * (0.5 * 32767)
    samples = np.frombuffer(out, dtype="<i2").astype(np.int32)
    assert int(np.max(np.abs(np.diff(samples)))) < max_slope * 1.2


def test_resampler_handles_44100_hz_devices():
    resampler = StreamResampler(44_100, 2, "int16")
    out = resampler.process(_sine(1000, 0.5, 44_100, 2)) + resampler.flush()
    assert abs(len(out) // 2 - 8_000) < 160
    assert abs(_dominant_frequency(out, 16_000) - 1000.0) < 10.0


def test_resampler_ignores_empty_buffers():
    assert StreamResampler(48_000, 2, "int16").process(b"") == b""


def test_resampler_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        StreamResampler(0, 1)
    with pytest.raises(ValueError):
        StreamResampler(48_000, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_device_info_is_built_from_the_pyaudio_dict():
    device = DeviceInfo.from_dict(
        {
            "index": 7,
            "name": "Speakers (Realtek) [Loopback]",
            "maxInputChannels": 2,
            "defaultSampleRate": 48000.0,
            "isLoopbackDevice": True,
        }
    )
    assert (device.index, device.channels, device.default_sample_rate) == (7, 2, 48_000)
    assert device.is_loopback


def test_wav_wrapper_writes_a_playable_16k_mono_header():
    import io
    import wave

    pcm = _sine(440, 0.25, 16_000, 1)
    with wave.open(io.BytesIO(pcm16_to_wav_bytes(pcm)), "rb") as wav:
        assert wav.getframerate() == 16_000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == len(pcm) // 2
