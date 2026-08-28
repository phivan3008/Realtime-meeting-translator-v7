"""Central configuration for the server pipeline.

Every number here is explained in ``docs/TUNING.md``: what it means, the
measurement behind it, and what breaks if it moves. Read that before changing
one.

The blocked-sentence lists are editable text files in ``server/data/``; see
``server/data/README.md``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# --- Audio contract ---------------------------------------------------------
# Re-exported from common/protocol.py so the two sides cannot drift.
from common.protocol import (            # noqa: F401  (re-exported)
    CHANNELS,
    CHUNK_BYTES,
    CHUNK_DURATION_MS,
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)

# --- 1. Voice Activity Detection (Silero) -----------------------------------
# Silero only accepts 512-sample frames at 16 kHz: 32 ms per decision.
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 96
# Must stay above FINALIZE_PAUSE_MS, or a segment closes before the buffer can
# see the pause that closed it.
VAD_MIN_SILENCE_MS = 500
VAD_SPEECH_PAD_MS = 256

# --- 2. Stream Buffer Manager (DESIGN.md 3.2) -------------------------------
FINALIZE_PAUSE_MS = 400
FINALIZE_MAX_DURATION_MS = 7_000
PARTIAL_INTERVAL_MS = 600
SPLIT_SEARCH_MS = 500
# The running prediction decodes only this much of the open utterance.
PARTIAL_WINDOW_SECONDS = 4.0

# --- 3. Deep Noise Filter (AST, DESIGN.md 3.3) ------------------------------
AST_MODEL_ID = os.environ.get(
    "AST_MODEL_ID", "MIT/ast-finetuned-audioset-10-10-0.4593"
)
# CPU by default: DESIGN.md wants this stage off the GPU so the VRAM belongs
# to Whisper and vLLM, and on the pod AST on CUDA killed the server outright.
NOISE_DEVICE = os.environ.get("NOISE_DEVICE", "cpu")

# An utterance is dropped only when both hold: little speech, and something
# else recognised confidently. Either alone is not evidence.
NOISE_MIN_SPEECH_SCORE = 0.2
NOISE_MIN_NOISE_SCORE = 0.3
NOISE_WINDOW_SECONDS = 10.0

# --- 4. Overlap Resolver (DSP, DESIGN.md 3.4) -------------------------------
# Thresholds are relative to the utterance's own loudness, measured as a high
# percentile of the *peak* envelope - pedalboard's detectors compare against
# peak, not RMS.
OVERLAP_ENVELOPE_MS = 20
OVERLAP_LEVEL_PERCENTILE = 90.0

OVERLAP_GATE_BELOW_DB = 12.0
OVERLAP_GATE_RATIO = 4.0
OVERLAP_GATE_ATTACK_MS = 2.0
OVERLAP_GATE_RELEASE_MS = 120.0

OVERLAP_COMPRESSOR_ABOVE_DB = 3.0
OVERLAP_COMPRESSOR_RATIO = 3.0
OVERLAP_COMPRESSOR_ATTACK_MS = 5.0
OVERLAP_COMPRESSOR_RELEASE_MS = 120.0

# Quieter than this and there is nothing to favour; pass it through untouched.
OVERLAP_MIN_LEVEL_DBFS = -55.0

# --- 5. Speaker Diarization (DESIGN.md 3.5) ---------------------------------
SPEAKER_EMBEDDING_MODEL = os.environ.get(
    "SPEAKER_EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb"
)
SPEAKER_DEVICE = os.environ.get("SPEAKER_DEVICE", "")
SPEAKER_CACHE_DIR = os.environ.get("SPEAKER_CACHE_DIR", "models/speaker")

# Sits between the measured same-voice floor (0.361) and different-voice
# ceiling (0.232).
SPEAKER_MATCH_THRESHOLD = 0.30
SPEAKER_MIN_DURATION_MS = 600
SPEAKER_MAX_SPEAKERS = 12
SPEAKER_CENTROID_MOMENTUM = 0.7

#: Used when an utterance is too short to identify. Never guessed from the
#: previous speaker: a short interjection usually comes from whoever is
#: listening.
SPEAKER_UNKNOWN = "Speaker_unknown"

# Second thoughts: the whole meeting is clustered again every this many
# sentences, and corrected labels are sent back. Cost grows with the square
# of the number of voiceprints kept.
SPEAKER_RECLUSTER_EVERY = 15
SPEAKER_RECLUSTER_MAX = 300

# --- 5b. Speaker change boundary (DESIGN.md 3.2) ----------------------------
# Two people whose turns are less than VAD_MIN_SILENCE_MS apart land in one
# utterance, which then gets one voiceprint, one language and one ASR pass.
#
# OFF: measured on a real meeting, one-second voiceprints do not separate
# speakers - the comparisons form one unbroken cluster with no threshold to
# put between them. See docs/TUNING.md 5b. Set SPEAKER_CHANGE_ENABLED=1 to
# experiment.
SPEAKER_CHANGE_ENABLED = os.environ.get("SPEAKER_CHANGE_ENABLED", "0") == "1"
SPEAKER_CHANGE_WINDOW_MS = 1_000
SPEAKER_CHANGE_THRESHOLD = 0.25

# --- 6. Language ID (DESIGN.md 3.6) -----------------------------------------
LID_MODEL = os.environ.get("LID_MODEL", "speechbrain/lang-id-voxlingua107-ecapa")
LID_DEVICE = os.environ.get("LID_DEVICE", "")
LID_CACHE_DIR = os.environ.get("LID_CACHE_DIR", "models/lid")

# Only these two are scored, then renormalised between them. The model knows
# 107 languages and would answer Korean for Japanese given the chance.
LID_LANGUAGES = ("vi", "ja")
LID_MIN_MARGIN = 0.30
LID_MIN_DURATION_MS = 600

#: Undecided. The session then reuses the meeting's last known language
#: rather than letting Whisper choose from 99.
LID_UNKNOWN = ""

# --- 6b. Two languages in one utterance (DESIGN.md 3.2) ---------------------
# People answer each other faster than VAD_MIN_SILENCE_MS, so a reply in the
# other language lands in the same utterance and is lost rather than
# mistranslated. Probe both ends; only a confident disagreement cuts.
LANGUAGE_SPLIT_ENABLED = os.environ.get("LANGUAGE_SPLIT", "1") != "0"
LANGUAGE_SPLIT_PROBE_MS = LID_MIN_DURATION_MS
LANGUAGE_SPLIT_STEPS = 3

# --- 7. ASR (faster-whisper, DESIGN.md 3.7) ---------------------------------
ASR_MODEL = os.environ.get("ASR_MODEL", "large-v3")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "")
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "")
ASR_CACHE_DIR = os.environ.get("ASR_CACHE_DIR", "models/whisper")

# A partial is replaced within a second; a final is what the viewer keeps.
ASR_BEAM_SIZE_PARTIAL = 1
ASR_BEAM_SIZE_FINAL = 5

# Whisper answers near-silence with invented text. These three catch the
# unconfident kind; server/data/ catches the rest.
ASR_NO_SPEECH_THRESHOLD = 0.6

# Below this, no_speech_prob alone is enough to refuse a segment. Above it,
# avg_logprob has to agree - which is Whisper's own rule, and what stops a
# real seven-second sentence being thrown away. Duration is what separates
# the two cases: the sentence that rule was built for ran 6.8 s, and every
# confirmed invention on a real meeting was under 2 s. The number is the
# floor the speaker model and the LID already refuse to answer below.
ASR_SHORT_UTTERANCE_MS = SPEAKER_MIN_DURATION_MS

# Whisper reads only the start of initial_prompt, and a stuffed one makes it
# produce those very words over silence. See server/data/vocabulary.txt.
ASR_PROMPT_MAX_CHARS = 400
ASR_LOG_PROB_THRESHOLD = -1.0
ASR_MAX_COMPRESSION_RATIO = 2.4

# Feeding the previous sentence back in as a prompt is how one invention
# becomes a paragraph of them.
ASR_CONDITION_ON_PREVIOUS = False

# --- 8. Translation (vLLM, DESIGN.md 3.8) -----------------------------------
TRANSLATE_BASE_URL = os.environ.get("TRANSLATE_BASE_URL",
                                    "http://127.0.0.1:8001/v1")
TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "Qwen/Qwen3.5-9B")
TRANSLATE_TIMEOUT_S = float(os.environ.get("TRANSLATE_TIMEOUT_S", "20"))

TRANSLATE_TEMPERATURE = 0.0
# Qwen3 reasons aloud unless told not to, spending the whole token budget on
# it and returning no translation.
TRANSLATE_ENABLE_THINKING = False
TRANSLATE_MAX_TOKENS = 512

# Enough history to resolve a pronoun, not enough to summarise the meeting.
TRANSLATE_HISTORY = 3

#: How the history is written into the prompt: "plain", "labelled" or
#: "sources". With its translations in it the history reads as worked
#: examples and the model copies their language.
HISTORY_STYLE = "sources"

#: Spell out that a one-word line is still a line to translate.
SHORT_LINE_HINT_ENABLED = True

# Translation runs off the audio thread. A sentence waiting longer than the
# budget is dropped: the answer has stopped belonging to the sentence above it.
TRANSLATION_MAX_LAG_SECONDS = 10.0
TRANSLATION_QUEUE_DEPTH = 16

# Keyed by target language. Japanese carries the same meaning in far fewer
# characters, so one shared limit is wrong in both directions at once.
TRANSLATE_MAX_EXPANSION = {"vi": 2.0, "ja": 1.0}
TRANSLATE_EXPANSION_SLACK = 50

# Fraction of letters that may be kana or kanji. Vietnamese and Japanese share
# no script, which makes this a cheap test of whether the answer came out in
# the language it was asked for.
TRANSLATE_MAX_WRONG_SCRIPT = 0.30

#: Which language each one becomes, and the names used in the prompt.
TRANSLATE_PAIR = {"vi": "ja", "ja": "vi"}
LANGUAGE_NAMES = {"vi": "Vietnamese", "ja": "Japanese"}


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------
def known_variables() -> list[str]:
    """Every environment variable this file reads, taken from the file.

    Derived rather than listed by hand: a list that has to be kept in step
    with the code is a list that stops being true.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    return sorted(set(re.findall(r'os\.environ\.get\("(\w+)"', source)))


def overrides() -> dict[str, str]:
    """The ones actually set right now.

    Startup reports these, because a variable left over from an earlier
    terminal changes what the pipeline does and says nothing about it. Three
    measurements in this project were taken against a configuration nobody
    had meant to be running.
    """
    return {name: os.environ[name]
            for name in known_variables() if name in os.environ}
