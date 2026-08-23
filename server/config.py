"""Central configuration for the server pipeline.

The audio contract has to match ``client/config.py`` byte for byte: the client
produces exactly this format and the server assumes it without re-checking on
every chunk.
"""

from __future__ import annotations

# --- Audio contract received from the client (DESIGN.md section 2) -----------
SAMPLE_RATE = 16_000            # Hz
CHANNELS = 1                    # mono
SAMPLE_WIDTH = 2                # bytes, 16-bit signed little endian PCM

CHUNK_DURATION_MS = 200
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION_MS // 1000     # 3200
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH                  # 6400

# --- Voice Activity Detection (Silero, step 0 of the pipeline) ---------------
# Silero v5 only accepts 512 sample frames at 16 kHz, i.e. 32 ms per decision.
VAD_THRESHOLD = 0.5
# A segment opens only after this much speech-like audio, which filters out the
# short bursts Silero produces on keyboard clicks and door slams.
VAD_MIN_SPEECH_MS = 96
# Hangover before a segment closes. Kept above FINALIZE_PAUSE_MS so a closing
# segment always carries enough trailing silence for the buffer manager to see
# the pause that caused it.
VAD_MIN_SILENCE_MS = 500
# Audio kept in front of a segment so word onsets are not clipped before ASR.
VAD_SPEECH_PAD_MS = 256

# --- Stream Buffer Manager (DESIGN.md section 3.1) ---------------------------
# A sentence is finalised on a pause longer than this.
FINALIZE_PAUSE_MS = 400
# ... or when the segment has simply run too long.
FINALIZE_MAX_DURATION_MS = 7_000
