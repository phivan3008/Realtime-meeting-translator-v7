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
# Measured on three single-speaker recordings, 45 s each, two of them the
# same gender:
#
#   same voice        0.361 .. 0.994
#   different voices -0.129 .. 0.232
#
# so any threshold in (0.232, 0.361) separates them. 0.30 is close to the
# midpoint of 0.296 that two separate runs agreed on.
#
# Sitting in the middle rather than at either edge is the point. That window
# can only shrink as more people join: adding a third voice, one of the same
# gender as the first, moved the different-voice ceiling from 0.199 to 0.232
# and the same-voice floor from 0.394 to 0.361. SpeechBrain's own default of
# 0.25 was inside the window but left only 0.018 of room above the
# different-voice ceiling - one more similar pair and it would merge two
# people. 0.30 leaves about 0.06 on both sides instead.
SPEAKER_MATCH_THRESHOLD = 0.30

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

# --- 6. Language ID (DESIGN.md section 3.6) ---------------------------------
LID_MODEL = os.environ.get("LID_MODEL", "speechbrain/lang-id-voxlingua107-ecapa")
LID_DEVICE = os.environ.get("LID_DEVICE", "")
LID_CACHE_DIR = os.environ.get("LID_CACHE_DIR", "models/lid")

# The meeting is Vietnamese and Japanese, so the decision is between those two
# and nothing else. VoxLingua107 knows 107 languages, and letting it pick
# freely means a Japanese sentence can come back as Korean or Chinese - a
# plausible mistake for the model and a useless answer for us, because the
# only thing downstream does with this is force Whisper's language.
LID_LANGUAGES = ("vi", "ja")

# How far apart the two have to be before the answer is trusted. Below this
# the languages are reported as unknown and Whisper detects for itself, which
# is better than forcing the wrong one: forced Japanese on Vietnamese audio
# does not fail, it quietly transcribes nonsense.
LID_MIN_MARGIN = 0.30

# Shorter than this there is not enough speech to tell the languages apart.
LID_MIN_DURATION_MS = 600

#: Reported when the languages cannot be told apart; Whisper then auto-detects.
LID_UNKNOWN = ""

# --- 7. ASR (DESIGN.md section 3.7) -----------------------------------------
ASR_MODEL = os.environ.get("ASR_MODEL", "large-v3")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "")
# float16 halves the memory and the latency on an H100 and costs nothing that
# survives being turned back into text. CPU falls back to int8.
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "")
ASR_CACHE_DIR = os.environ.get("ASR_CACHE_DIR", "models/whisper")

# A partial is thrown away as soon as the next one arrives, so it is decoded
# greedily; the final answer is worth a beam search.
ASR_BEAM_SIZE_PARTIAL = 1
ASR_BEAM_SIZE_FINAL = 5

# Whisper's own guards, passed through so they are visible here rather than
# buried in a default.
ASR_NO_SPEECH_THRESHOLD = 0.6
ASR_LOG_PROB_THRESHOLD = -1.0

# Repetition guard. Whisper answers near-silence with confident invented text
# and sometimes locks into a loop; a segment whose text compresses far better
# than real speech is that loop. gzip on natural speech lands near 1.5-2.0.
ASR_MAX_COMPRESSION_RATIO = 2.4

# Sentences Whisper invents out of near-silence, verbatim. Every guard above is
# statistical, and these defeat all of them: Whisper is *confident* when it
# writes them - low no_speech_prob, high avg_logprob, ordinary compression -
# because they close a large share of the videos it was trained on.
#
# Only entries this project has actually seen are listed. Each is matched in
# full, after punctuation and spacing are stripped, so a real sentence that
# merely contains one of these phrases is kept.
ASR_HALLUCINATIONS = (
    # Seen in the 60 s end-to-end run, as running text over a Vietnamese
    # meeting about task tables.
    "C\u1ea3m \u01a1n c\u00e1c b\u1ea1n \u0111\u00e3 theo d\u00f5i "
    "v\u00e0 h\u1eb9n g\u1eb7p l\u1ea1i.",
    # Seen in the Module 9 ASR test, and refused there by no_speech_prob.
    # Listed because that refusal was luck, not policy.
    "C\u1ea3m \u01a1n c\u00e1c b\u1ea1n \u0111\u00e3 theo d\u00f5i.",
    "\u3054\u8996\u8074\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044"
    "\u307e\u3057\u305f\u3002",
    # The same sign-off in the non-past. One character apart from the line
    # above, and Whisper picks between them freely.
    "\u3054\u8996\u8074\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044"
    "\u307e\u3059\u3002",
    # Whisper's answer to silence in English. A meeting where somebody says
    # exactly "you" and nothing else loses one word; that trade is worth it.
    "you",
    # Committed as a whole sentence at 55.3 s of the second end-to-end run,
    # attributed to Speaker_02 and translated to さようなら。 The recording
    # was played back over 54-56 s and nobody says it.
    #
    # This entry is a different trade from the ones above. Those are video
    # sign-offs that no meeting produces; this is an ordinary Vietnamese
    # sentence, and blocking it means a real goodbye said in exactly these
    # words and no others is deleted. Whole-segment matching is what keeps
    # that narrow: "Chào tạm biệt nhé", "Thôi chào tạm biệt mọi người" and
    # anything else with a word attached all survive.
    "Ch\u00e0o t\u1ea1m bi\u1ec7t.",
    # Running text at 107.3 s of the eighth end-to-end run, over a meeting
    # about task tables. A Vietnamese YouTube channel's subscribe pitch,
    # named channel and all - not something a meeting says, and the surest
    # entry on this list.
    #
    # It also shows what the list cannot do. It never became a sentence, so
    # nothing failed; the check "No running text is a Whisper sign-off"
    # passed, because the line was not on the list to be caught by. Only a
    # person reading the output found it.
    "H\u00e3y subscribe cho k\u00eanh Ghi\u1ec1n M\u00ec G\u00f5 "
    "\u0110\u1ec3 kh\u00f4ng b\u1ecf l\u1ee1 nh\u1eefng video h\u1ea5p d\u1eabn",
)

# What this list cannot do: it only knows what has already been seen. Every
# entry above was found by a person listening to a recording and reporting
# that nobody said it. A confident invention this project has not met yet
# will still reach the screen, because nothing in Whisper's own numbers
# separates one from real speech.

# Whisper carries the previous sentence into the next by default, which is
# where streaming hallucination loops come from: one invented sentence becomes
# the prompt for the next. Each utterance here is already a complete thought,
# so it is decoded alone.
ASR_CONDITION_ON_PREVIOUS = False

# --- 8. Translation (DESIGN.md section 3.8) ---------------------------------
# vLLM runs as its own process behind its OpenAI-compatible API, and this
# talks to it over HTTP rather than importing it.
#
# The reason is memory, not taste. vLLM profiles the GPU at load and reserves
# a fraction of it up front; in-process it would do that alongside Whisper
# large-v3, AST, ECAPA and VoxLingua, and the two allocators would have to be
# tuned against each other by hand. Out of process each side sees a GPU it can
# reason about, the LLM can be restarted or swapped without dropping a
# meeting, and a crash in it does not take the audio pipeline with it. The
# cost is one more process to start.
TRANSLATE_BASE_URL = os.environ.get("TRANSLATE_BASE_URL",
                                    "http://127.0.0.1:8001/v1")
# The checkpoint DESIGN.md section 3.8 names. It has to match what vLLM was
# started with: the client checks at connect time and refuses a server that is
# serving something else, because a quietly substituted model is a difference
# nobody would see in the logs and everybody would see in the translations.
# Set to empty to accept whatever the server happens to be serving.
TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "Qwen/Qwen3.5-9B")
TRANSLATE_TIMEOUT_S = float(os.environ.get("TRANSLATE_TIMEOUT_S", "20"))

# Translation is not a creative task and the same sentence twice should give
# the same answer twice.
TRANSLATE_TEMPERATURE = 0.0

# Qwen3 reasons before it answers, emitting a <think> block first. For a
# sentence-length translation that is all cost and no benefit: the first run
# against Qwen3.5-9B spent its entire 512-token budget thinking and returned
# no translation at all, at 3.5 s a sentence. Turned off through the chat
# template; the <think> stripping in translate.py stays as a second line of
# defence for a server that ignores the flag.
TRANSLATE_ENABLE_THINKING = False

# DESIGN.md asks for two or three previous sentences of context. Enough for
# pronouns and carried-over subjects, short enough that the model cannot drift
# into summarising the meeting.
TRANSLATE_HISTORY = 3

# How the history is written into the prompt.
#
# Written with its translations, the history reads as worked examples. When
# several turns run the same way every example ends in the same language, and
# the model follows the examples over the instruction. Measured against the
# live model on two sentences the pipeline lost, three Vietnamese turns of
# history behind each:
#
#     none      translated      translated
#     plain     REFUSED         REFUSED
#     labelled  REFUSED         REFUSED
#     sources   translated      translated
#
# The "none" column is what rules out the other explanation: both sentences
# had also been cut mid-sentence by the 7 s limit, and a model cannot
# translate half a sentence - but with no history at all it translated both.
#
# Naming each translation's language ("labelled") was not enough; at three
# turns deep it failed exactly as the original did. Removing the translations
# works, because there is then nothing to imitate. The history is kept to say
# what "that one" refers to, and the source lines carry that alone - the
# answers still change with the history, they just stop copying its language.
HISTORY_STYLE = "sources"

# Whether the system prompt spells out that a one-word line is still a line.
# はい came back as はい on the sixth end-to-end run - a whole turn of a
# Japanese meeting, and one of the commonest lines in one. Longer short lines
# on the same run were fine (えっ -> Eh?, いや違います -> Không, tôi nhầm rồi),
# so it is single words the model treats as nothing to do.
SHORT_LINE_HINT_ENABLED = True

# A translation runs a little longer than its source, never many times longer.
# Far past this and the model has started explaining itself or looping.
#
# One number for both directions was wrong, and it was wrong in both
# directions at once. Measured over 21 real pairs, as len(output)/len(source)
# in characters:
#
#     ja -> vi   1.17 - 4.44      Japanese is dense; a nine-character
#                                 fragment becomes a forty-character sentence
#     vi -> ja   0.44 - 0.70      the same information, written shorter
#
# The old shared 4.0 therefore refused a correct Vietnamese translation of
# あれこれ今下の方に (4.44) while being unreachable in the other direction -
# a Japanese answer would have had to run six times the length of a correct
# one before anything noticed.
#
# Keyed by the TARGET language. The slack is what makes short sources
# survive: at nine characters a ratio is mostly noise, and every measured
# pair fits inside its own limit with room to spare (tightest: 74 against 92).
TRANSLATE_MAX_EXPANSION = {"vi": 2.0, "ja": 1.0}
TRANSLATE_EXPANSION_SLACK = 50
TRANSLATE_MAX_TOKENS = 512

# A translation has to be written in the target language's script. Measured on
# every translation the 60 s end-to-end run produced, as a fraction of letters
# that are kana or kanji:
#
#     into Vietnamese, correct      0.00  (7 sentences)
#     into Vietnamese, NOT translated 1.00  (1 sentence: はい、今の画面の
#                                            came back as はい、現在の画面の)
#     into Japanese, correct        0.86 - 1.00  (4 sentences)
#
# Everything from 0.00 to 0.86 is empty, so this sits in the middle of both
# gaps rather than on the edge of either. The 0.86 is a Japanese sentence
# opening with the Latin initialism "FCG", which is what the headroom is for:
# a name kept as-is must not fail its own translation.
TRANSLATE_MAX_WRONG_SCRIPT = 0.30

#: Which language each one becomes.
TRANSLATE_PAIR = {"vi": "ja", "ja": "vi"}
#: Human names, for the prompt.
LANGUAGE_NAMES = {"vi": "Vietnamese", "ja": "Japanese"}
