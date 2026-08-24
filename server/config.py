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

# --- 4. Overlap Resolver (DSP, DESIGN.md section 3.4) -----------------------
# Thresholds are relative to the utterance's own loudness, never absolute:
# "quiet" only means quiet compared to whoever is dominating this sentence, and
# meeting recordings arrive at wildly different levels.
#
# That loudness is a high percentile of the short-term envelope, not the global
# RMS. An utterance deliberately carries the VAD's hangover silence and every
# pause between words; measured on a real recording the median 20 ms frame sat
# 28 dB below the speaking level. A gate built on the global RMS therefore
# lands far too low and never removes what it was meant to.
#
# The percentile is taken over the *peak* envelope, because pedalboard's gate
# compares its threshold against the signal peak. Measured against a second
# voice 20 dB down: an RMS-based threshold attenuated it by 0.1 dB, a
# peak-based one by 24 dB, with the dominant voice untouched either way.
OVERLAP_ENVELOPE_MS = 20
OVERLAP_LEVEL_PERCENTILE = 90.0

OVERLAP_GATE_BELOW_DB = 12.0        # gate anything this far under the speaker
OVERLAP_GATE_RATIO = 4.0
OVERLAP_GATE_ATTACK_MS = 2.0
OVERLAP_GATE_RELEASE_MS = 120.0     # long enough not to chop word tails

# The compressor only tames peaks above the speaking level; on real speech they
# sit barely 3 dB up, so a lower threshold would squash the voice itself.
OVERLAP_COMPRESSOR_ABOVE_DB = 3.0
OVERLAP_COMPRESSOR_RATIO = 3.0
OVERLAP_COMPRESSOR_ATTACK_MS = 5.0
OVERLAP_COMPRESSOR_RELEASE_MS = 120.0

# An utterance quieter than this has nothing to shape: gating it would only eat
# the little signal there is. Pass it through untouched instead.
OVERLAP_MIN_LEVEL_DBFS = -55.0

# --- 5. Speaker Diarization (DESIGN.md section 3.5) -------------------------
# ECAPA-TDNN voiceprints, matched by cosine similarity. The checkpoint is
# public, so no HuggingFace token is needed.
SPEAKER_EMBEDDING_MODEL = os.environ.get(
    "SPEAKER_EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb"
)
SPEAKER_DEVICE = os.environ.get("SPEAKER_DEVICE", "")
# Where SpeechBrain unpacks the checkpoint it downloads.
SPEAKER_CACHE_DIR = os.environ.get("SPEAKER_CACHE_DIR", "models/speaker")

# Cosine similarity above which two voiceprints are called the same person.
# 0.25 is SpeechBrain's own default for this checkpoint - see the `threshold`
# argument of SpeakerRecognition.verify_batch - which is a tuned operating
# point rather than a guess, and a long way below the 0.55 first put here.
# Measured on two single-speaker recordings (45 s each, different people):
# same voice ranged 0.394 to 0.994, different voices -0.129 to 0.199, so any
# threshold in (0.199, 0.394) separates them and 0.25 sits inside it.
#
# Kept at SpeechBrain's 0.25 rather than moved to the measured midpoint of
# 0.296. Two speakers is not enough to tune on, and the two failures are not
# equal: too low merges two people under one name, too high splits one person
# into Speaker_01 through Speaker_06, which is what the first broken run
# looked like and is far uglier. 0.25 leans towards merging on purpose.
SPEAKER_MATCH_THRESHOLD = 0.25

# Shorter than this there is not enough voice for a trustworthy print, and a
# wrong speaker label is worse than an honest "unknown".
SPEAKER_MIN_DURATION_MS = 600

# Beyond this many distinct voices, stop inventing new ones: a meeting with
# 30 "speakers" means the threshold is wrong, not that 30 people are talking.
SPEAKER_MAX_SPEAKERS = 12

# How much each new utterance moves a speaker's stored voiceprint. Keeping
# most of the old centroid stops one noisy sentence from redefining someone.
SPEAKER_CENTROID_MOMENTUM = 0.7

#: Label used when an utterance is too short to identify.
SPEAKER_UNKNOWN = "Speaker_unknown"
