"""Streaming PCM conversion helpers.

WASAPI loopback always hands us the audio in the *device* mix format, which on
a normal Windows machine is 44.1 kHz or 48 kHz stereo.  The server pipeline
expects 16 kHz mono 16-bit PCM, so every captured buffer goes through this
module first.

Two resampler backends are supported:

``soxr``
    Preferred.  ``soxr.ResampleStream`` keeps the filter state between calls,
    so consecutive chunks join without clicks at the boundaries.

``scipy.signal.resample_poly``
    Fallback used when soxr is not installed.  It is stateless, which means a
    very small discontinuity can appear at chunk boundaries, but the output is
    still perfectly usable for ASR.
"""

from __future__ import annotations

import math

import numpy as np

try:                                        # pragma: no cover - import guard
    import soxr

    _HAS_SOXR = True
except ImportError:                         # pragma: no cover - import guard
    soxr = None
    _HAS_SOXR = False

try:                                        # pragma: no cover - import guard
    from scipy.signal import resample_poly

    _HAS_SCIPY = True
except ImportError:                         # pragma: no cover - import guard
    resample_poly = None
    _HAS_SCIPY = False


INT16_MAX = 32767.0


def bytes_to_float32(raw: bytes, sample_format: str) -> np.ndarray:
    """Decode a raw PCM buffer into a float32 array normalised to [-1.0, 1.0]."""
    if sample_format == "int16":
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / INT16_MAX
    elif sample_format == "int32":
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_format == "float32":
        data = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"Unsupported sample format: {sample_format!r}")
    return data


def float32_to_int16_bytes(samples: np.ndarray) -> bytes:
    """Encode a float32 array in [-1.0, 1.0] as little endian 16-bit PCM."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * INT16_MAX).astype("<i2").tobytes()


def downmix_to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Average interleaved channels down to a single mono track."""
    if channels == 1:
        return samples
    usable = (samples.size // channels) * channels
    if usable != samples.size:
        # A partial frame at the end of the buffer would shift every channel by
        # one sample, so it is dropped rather than silently corrupting the mix.
        samples = samples[:usable]
    return samples.reshape(-1, channels).mean(axis=1)


class StreamResampler:
    """Convert a device audio stream to 16 kHz mono 16-bit PCM, chunk by chunk.

    The instance is stateful and must be used for one capture session only:
    call :meth:`process` for every captured buffer and :meth:`flush` once at
    the end of the session.
    """

    def __init__(
        self,
        input_rate: int,
        input_channels: int,
        sample_format: str = "int16",
        output_rate: int = 16_000,
        quality: str = "HQ",
    ) -> None:
        if input_rate <= 0:
            raise ValueError("input_rate must be positive")
        if input_channels < 1:
            raise ValueError("input_channels must be >= 1")

        self.input_rate = int(input_rate)
        self.input_channels = int(input_channels)
        self.sample_format = sample_format
        self.output_rate = int(output_rate)
        self.backend = "passthrough"
        self._stream = None

        if self.input_rate != self.output_rate:
            if _HAS_SOXR:
                self._stream = soxr.ResampleStream(
                    self.input_rate,
                    self.output_rate,
                    1,                       # already down-mixed to mono
                    dtype="float32",
                    quality=quality,
                )
                self.backend = "soxr"
            elif _HAS_SCIPY:
                self.backend = "scipy"
            else:  # pragma: no cover - both backends missing
                raise RuntimeError(
                    "Neither soxr nor scipy is installed; cannot resample "
                    f"{self.input_rate} Hz -> {self.output_rate} Hz"
                )

    # -- public API ---------------------------------------------------------
    def process(self, raw: bytes) -> bytes:
        """Feed one raw device buffer, return the converted 16 kHz mono PCM."""
        if not raw:
            return b""
        interleaved = bytes_to_float32(raw, self.sample_format)
        mono = downmix_to_mono(interleaved, self.input_channels)
        return float32_to_int16_bytes(self._resample(mono))

    def flush(self) -> bytes:
        """Drain whatever is still held inside the resampler filter state."""
        if self.backend != "soxr" or self._stream is None:
            return b""
        tail = self._stream.resample_chunk(
            np.zeros(0, dtype=np.float32), last=True
        )
        return float32_to_int16_bytes(np.asarray(tail, dtype=np.float32))

    def expected_output_samples(self, input_samples: int) -> int:
        """Rough number of output samples for ``input_samples`` input frames.

        Used by the tests to assert the conversion ratio; the real count can
        differ by a couple of samples because of the resampler filter delay.
        """
        return int(math.floor(input_samples * self.output_rate / self.input_rate))

    # -- internals ----------------------------------------------------------
    def _resample(self, mono: np.ndarray) -> np.ndarray:
        if self.backend == "passthrough":
            return mono
        if self.backend == "soxr":
            out = self._stream.resample_chunk(mono)
            return np.asarray(out, dtype=np.float32)
        gcd = math.gcd(self.output_rate, self.input_rate)
        up = self.output_rate // gcd
        down = self.input_rate // gcd
        return resample_poly(mono, up, down).astype(np.float32)
