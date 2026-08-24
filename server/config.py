"""Central configuration for the server pipeline.

Machine-specific tuning lives here; the wire-level audio contract lives in
``common/protocol.py`` and is re-exported below so both sides cannot drift.
"""

from __future__ import annotations

import os

# --- Audio contract ----------------------------------------------------------
# Defined once in common/protocol.py and re-exported here, because a mismatch
# between the two sides corrupts audio silently rather than raising.
from common.protocol import (            # noqa: F401  (re-exported)
    CHANNELS,
    CHUNK_BYTES,
    CHUNK_DURATION_MS,
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)

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
# How often the open utterance is handed to the partial ASR pass while the
# speaker keeps talking. Shorter feels more live but costs GPU on text that is
# about to be replaced anyway.
PARTIAL_INTERVAL_MS = 600
# When a max-duration cut is forced, look back this far for the quietest
# moment and cut there instead, so the split lands between words rather than
# through one.
SPLIT_SEARCH_MS = 500

# --- 3. Deep Noise Filter (AST, DESIGN.md section 3.3) ----------------------
# DESIGN.md allows "YAMNet or a slimmed AST". AST wins on this pod: YAMNet
# means TensorFlow, and TF 2.17 pins numpy < 2.1 and protobuf 4.x while vllm,
# whisperx and pyannote all need numpy >= 2 and protobuf >= 5.29. There is no
# version of both. AST runs on the torch the pod already has, over the same
# AudioSet label space, so the filter policy is unchanged.
AST_MODEL_ID = os.environ.get(
    "AST_MODEL_ID", "MIT/ast-finetuned-audioset-10-10-0.4593"
)
# "cuda", "cpu", or "" to pick automatically. The model is ~350 MB of VRAM,
# which is nothing next to vLLM, and on CPU a 7 s utterance would eat most of
# the latency budget.
NOISE_DEVICE = os.environ.get("NOISE_DEVICE", "")

# An utterance survives unless the classifier is confident it holds no speech
# at all. The filter is deliberately timid: dropping real speech loses a
# sentence for good, while letting a cough through only costs one wasted ASR
# call.
NOISE_MIN_SPEECH_SCORE = 0.2
# ... and even then, only when the classifier is confident about what it heard
# instead. Comparing two near-zero scores is not evidence: a real keyboard
# scores 0.87 and a real cough 0.83, while audio the model cannot place at all
# peaks around 0.1. Below this bar the answer is "no idea", and no idea means
# keep.
NOISE_MIN_NOISE_SCORE = 0.3
# AST reads a fixed 10.24 s window. Longer audio is scored one window at a
# time and the best score for each label wins.
NOISE_WINDOW_SECONDS = 10.0
