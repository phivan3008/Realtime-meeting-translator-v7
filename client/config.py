"""Central configuration for the client application.

Every value here is a plain constant so that unit tests and the real hardware
tests share exactly the same audio contract as the production client.
"""

from __future__ import annotations

# --- Audio contract ----------------------------------------------------------
# Defined once in common/protocol.py and re-exported here, because a mismatch
# between the two sides corrupts audio silently rather than raising.
from common.protocol import (            # noqa: F401  (re-exported)
    CHUNK_BYTES,
    CHUNK_DURATION_MS,
    CHUNK_SAMPLES,
    CHANNELS as TARGET_CHANNELS,
    SAMPLE_RATE as TARGET_SAMPLE_RATE,
    SAMPLE_WIDTH as TARGET_SAMPLE_WIDTH,
)

# --- Capture device ----------------------------------------------------------
# Number of frames pyaudiowpatch reads per callback at the *device* rate.
# Small enough to keep latency down, large enough to avoid callback overruns.
CAPTURE_FRAMES_PER_BUFFER = 1024

# Maximum number of 16 kHz chunks buffered before the oldest ones are dropped.
# 250 chunks * 200 ms = 50 seconds of backlog.
MAX_QUEUED_CHUNKS = 250

# NOTE: Voice Activity Detection lives on the GPU server (server/pipeline/vad.py),
# not here. Silero drags torch + torchaudio onto the Windows client and they
# refused to load there; the server also needs the pause boundaries anyway for
# its Stream Buffer Manager. The client streams every chunk, unfiltered.
