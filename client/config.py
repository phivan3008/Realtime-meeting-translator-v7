"""Central configuration for the client application.

Every value here is a plain constant so that unit tests and the real hardware
tests share exactly the same audio contract as the production client.
"""

from __future__ import annotations

# --- Audio contract sent to the server (DESIGN.md section 2) -----------------
TARGET_SAMPLE_RATE = 16_000     # Hz
TARGET_CHANNELS = 1             # mono
TARGET_SAMPLE_WIDTH = 2         # bytes, 16-bit signed little endian PCM

# --- Streaming chunk size ----------------------------------------------------
# DESIGN.md asks for ~200-500 ms packets. 200 ms keeps the perceived latency low
# while still being large enough for Silero VAD (which works on 512 sample
# frames at 16 kHz, i.e. 32 ms).
CHUNK_DURATION_MS = 200
CHUNK_SAMPLES = TARGET_SAMPLE_RATE * CHUNK_DURATION_MS // 1000     # 3200
CHUNK_BYTES = CHUNK_SAMPLES * TARGET_SAMPLE_WIDTH                  # 6400

# --- Capture device ----------------------------------------------------------
# Number of frames pyaudiowpatch reads per callback at the *device* rate.
# Small enough to keep latency down, large enough to avoid callback overruns.
CAPTURE_FRAMES_PER_BUFFER = 1024

# Maximum number of 16 kHz chunks buffered before the oldest ones are dropped.
# 250 chunks * 200 ms = 50 seconds of backlog.
MAX_QUEUED_CHUNKS = 250
