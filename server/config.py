"""Central configuration for the server pipeline.

Machine-specific tuning lives here; the wire-level audio contract lives in
``common/protocol.py`` and is re-exported below so both sides cannot drift.

Every number is also explained in ``docs/TUNING.md``: what it means, the
measurement behind it, and what breaks if it moves. A test keeps the two in
step. The blocked-sentence lists and the vocabulary prompt are editable text
files in ``server/data/``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _flag(name: str, default: str = "0") -> bool:
    """An on/off environment variable. Anything but empty or "0" is on."""
    return os.environ.get(name, default) not in ("", "0")


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

# How much of the open utterance the running prediction decodes. It re-decodes
# from the start every PARTIAL_INTERVAL_MS, so an uncapped window costs about
# six times the audio it covers: a seven-second sentence is decoded eleven
# times at 0.6, 1.2 ... 7.0 seconds.
#
# Measured over a ten-minute meeting, that came to 97.8 s of decoding against
# 21.6 s for every committed sentence put together - 16% of the run spent on
# text that is replaced 600 ms later, on the same thread that reads audio. The
# worst single pass took 4.7 s while the slowest committed sentence took 0.4 s.
#
# Four seconds. Measured over the same ten minutes before and after:
#
#     worst event lag        8448 ms  ->  1398 ms
#     slowest partial pass    4.7 s   ->   3.9 s
#     partial_asr total      97.8 s   ->  88.7 s
#
# The stall is gone - the 8.4 s spike sat at the same point in the audio on
# two consecutive runs and is now 0.6 s - which is what the cap was for.
#
# The total barely moved, and the first version of this comment claimed a
# third off. That was arithmetic done on a seven-second sentence and never
# checked against the distribution: most utterances in a real meeting are
# one to three seconds and never reach the cap at all. The cap bounds the
# worst case; it does not reduce the common one.
#
# Which leaves partial decoding at 88.7 s of 600, still the largest single
# cost on the audio thread. Nothing is failing for it now - worst lag sits
# inside the 1500 ms budget - so it stays as it is until something needs the
# headroom.
#
# The cost paid: on a long sentence the grey text shows what is being said
# now rather than the sentence from its beginning. The committed sentence is
# decoded once, in full, and is unaffected.
PARTIAL_WINDOW_SECONDS = 4.0

# --- 3. Deep Noise Filter (AST, DESIGN.md section 3.3) ----------------------
# DESIGN.md allows "YAMNet or a slimmed AST". AST wins on this pod: YAMNet
# means TensorFlow, and TF 2.17 pins numpy < 2.1 and protobuf 4.x while vllm,
# whisperx and pyannote all need numpy >= 2 and protobuf >= 5.29. There is no
# version of both. AST runs on the torch the pod already has, over the same
# AudioSet label space, so the filter policy is unchanged.
AST_MODEL_ID = os.environ.get(
    "AST_MODEL_ID", "MIT/ast-finetuned-audioset-10-10-0.4593"
)
# The whole stage is OFF unless ENABLE_NOISE_FILTER=1. Measured over two real
# meetings from the venv, 237 utterances: it dropped nothing at all, and cost
# a fixed 1.31 s per utterance on CPU - a quarter of the thread that reads the
# socket. Turning it off took the slowest sentence from 2.1 s to 0.4 s.
ENABLE_NOISE_FILTER = _flag("ENABLE_NOISE_FILTER")
# CPU by default. DESIGN.md wants this stage off the GPU so the VRAM belongs
# to Whisper and vLLM, and on the pod AST on CUDA raised in cuDNN and the next
# call into it segfaulted the process. NOISE_DEVICE=cuda opts back in.
NOISE_DEVICE = os.environ.get("NOISE_DEVICE", "cpu")

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

# DISABLE_OVERLAP=1 feeds Whisper raw audio. The resolver's only consumer is
# the ASR, and what shaping does to transcription accuracy has never been
# measured - only what it does to voiceprints, where it costs 0.06 cosine.
DISABLE_OVERLAP = _flag("DISABLE_OVERLAP")

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

# Second thoughts: the whole meeting is clustered again every this many
# sentences, and corrected labels are sent back. The live matcher answers in
# meeting order and never revisits an answer; on a real four-minute meeting
# every sentence came out Speaker_01.
SPEAKER_RECLUSTER_EVERY = 15
# Voiceprints kept. Clustering cost grows with the square of this.
SPEAKER_RECLUSTER_MAX = 300
# Where average linkage stops merging. The same number as the live matcher
# for now, but it is NOT the same measurement: 0.30 was measured on single
# pairs, and the average between two clusters is a different quantity. A
# thirty-minute run found 22 speakers with it. Every run logs the merge
# scores so this can be placed from a real meeting.
SPEAKER_RECLUSTER_THRESHOLD = SPEAKER_MATCH_THRESHOLD
# A label only moves after two reclustering runs in a row agree on the new
# one. The same run corrected 329 labels across 313 sentences - names that
# would not sit still on screen.
SPEAKER_RECLUSTER_CONFIRMATIONS = 2

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

#: Reported when the languages cannot be told apart; the session then reuses
#: the meeting's last known language rather than letting Whisper pick one of 99.
LID_UNKNOWN = ""

# --- 6b. Splitting an utterance that holds two languages --------------------
# The VAD closes a segment on VAD_MIN_SILENCE_MS of quiet and people answer
# each other faster than that, so a reply in the other language lands inside
# the same utterance. The LID then names one language, the ASR is forced into
# it for all of it, and the turn in the other language is not mistranslated -
# it is gone. Measured: four turns lost per ten-minute meeting.

#: Off with LANGUAGE_SPLIT=0. The ordinary case costs two LID probes.
LANGUAGE_SPLIT = _flag("LANGUAGE_SPLIT", "1")

#: Audio each probe reads. Comfortably over LID_MIN_DURATION_MS: an undecided
#: probe ends the search, and on 600 ms probes the real margins ran near 0.13.
LANGUAGE_SPLIT_PROBE_MS = 1_000.0

#: Skipped at each end before probing. The first VAD_SPEECH_PAD_MS is
#: pre-roll and the last part is the VAD hangover. Probing the outermost
#: second gave 26 of 41 visible splits the same language on both halves.
LANGUAGE_SPLIT_EDGE_MS = 300.0

#: Margin a probe must clear to count towards a cut, well over LID_MIN_MARGIN.
#: A missed cut loses a turn; an unnecessary one manufactures a fragment, and
#: Whisper fills a fragment in ("All right, they will.").
LANGUAGE_SPLIT_MIN_MARGIN = 0.50

#: The shortest half a cut may leave, in speech. The tail floor adds the
#: utterance's own hangover on top. 800 ms left 34 of 41 second halves at
#: twenty-five characters or less, which is where the inventions were.
LANGUAGE_SPLIT_MIN_PART_MS = 1_200.0

#: Longest window a review probe reads, from the middle of its half. Without
#: the cap the slowest sentence went from 0.7 s to 1.2 s.
LANGUAGE_SPLIT_REVIEW_MS = 2_500.0

#: Narrowest window the quiet-frame search uses once the binary search has
#: converged.
LANGUAGE_SPLIT_SNAP_MS = 200.0

#: Halvings of the search range. Three narrow a five-second utterance to about
#: 600 ms, which is inside the snap search.
LANGUAGE_SPLIT_MAX_STEPS = 3

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
#
# no_speech_prob over this refuses a segment only when avg_logprob agrees -
# faster-whisper's own rule. Read alone, over thirty minutes of real meeting,
# it refused 68 segments that were all decoded confidently (median avg_logprob
# -0.37), including a 6.8 s sentence that matched what was said word for word.
ASR_NO_SPEECH_THRESHOLD = 0.6
ASR_LOG_PROB_THRESHOLD = -1.0

# ... except on audio shorter than this, where no_speech_prob alone refuses
# again. Every invention confirmed on a real meeting came from a scrap under
# 2 s, many of them Speaker_unknown, and the sentence the rule above was built
# for ran 6.8 s. 600 ms is the floor where the speaker model and the LID both
# already decline to answer; Whisper is the only model that answers below it.
ASR_SHORT_UTTERANCE_MS = SPEAKER_MIN_DURATION_MS

# ... and at any length, a no_speech_prob this high refuses on its own. Kept
# from the streaming rewrite; it has not been measured against real meetings,
# so the refusals it makes are logged with both scores.
ASR_NO_SPEECH_CERTAIN = 0.95

# Repetition guard. Whisper answers near-silence with confident invented text
# and sometimes locks into a loop; a segment whose text compresses far better
# than real speech is that loop. gzip on natural speech lands near 1.5-2.0.
ASR_MAX_COMPRESSION_RATIO = 2.4

# Sentences Whisper invents out of near-silence are not here any more. They
# are three editable text files in server/data/ (hallucinations.txt,
# hallucination_patterns.txt, keep.txt), read by server/wordlists.py, so a
# line can be added without a redeploy. server/data/README.md says why every
# statistical guard above lets them through and how to test an entry first.
MEETING_DATA_DIR = os.environ.get(
    "MEETING_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

# Whisper carries the previous sentence into the next by default, which is
# where streaming hallucination loops come from: one invented sentence becomes
# the prompt for the next. Each utterance here is already a complete thought,
# so it is decoded alone.
ASR_CONDITION_ON_PREVIOUS = False

# --- 7b. The meeting's own vocabulary ---------------------------------------
# server/data/vocabulary.txt becomes Whisper's initial_prompt. It tilts the
# model where it is undecided between two readings of one sound: the running
# text hears "solution" and the committed sentence writes "sau lưu sinh". A
# run of the same meeting that carried such a list read both correctly.
#
# Capped on a term boundary. Whisper reads only the start, and a non-empty
# prompt makes it fill near-silence rather than leave it, so a long list
# costs something merely by existing. A list seeded with thirty guesses made
# inventions rise sharply.
ASR_PROMPT_MAX_CHARS = 200
# The running text does not get it: it is decoded six times more often, which
# would multiply the one cost a prompt has. Set to 1 to measure the other way.
ASR_PROMPT_ON_PARTIALS = _flag("ASR_PROMPT_ON_PARTIALS")

# --- 7c. Streaming stabilisation (server/pipeline/asr.py) -------------------
# The running text is decoded again every PARTIAL_INTERVAL_MS on the last
# PARTIAL_WINDOW_SECONDS. A word that consecutive decodes agree on, at about
# the same time, is committed and never rewritten; the sentence at the end
# decodes only what was not committed yet. Measured on the debug logs, this
# took the share of on-screen text surviving an update from 33% to 67%.

#: Consecutive decodes that must contain a word before it is committed. Two is
#: the floor - one decode agreeing with itself is not evidence.
ASR_STREAM_MIN_AGREEMENT = 2
#: Words closer than this to the newest audio are never committed; Whisper
#: revises the end of a window more than anything else.
ASR_STREAM_COMMIT_MARGIN_SECONDS = 1.0
#: Committed audio decoded again in front of the final tail, for left context.
ASR_STREAM_FINAL_OVERLAP_SECONDS = 1.2
#: Audio kept after the VAD's last speech frame when the tail is decoded. The
#: rest of the hangover is silence, and Whisper answers silence with words: a
#: clause was added over 400 ms nobody spoke into.
ASR_STREAM_FINAL_POST_ROLL_SECONDS = 0.20
#: How far apart two decodes may place the same word and still agree.
ASR_STREAM_WORD_TOLERANCE_SECONDS = 0.45
#: Decodes kept per open utterance for agreement.
ASR_STREAM_HISTORY = 5

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
#
# The name the server reports is whatever --model was given, so if vLLM is
# started from a local directory, set this to that path instead.
TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "google/gemma-4-12b-it")
TRANSLATE_TIMEOUT_S = float(os.environ.get("TRANSLATE_TIMEOUT_S", "20"))

# How vLLM itself is started for that checkpoint. The audio server never
# starts vLLM - it runs as its own process - so these are read by
# ``server/launch_vllm.py``, which builds the command from them, and nothing
# else. Kept here so the engine and the requests are configured in one place.
#
# bfloat16 is not a speed choice. Gemma was trained in bfloat16 and float16
# overflows in its activations: the symptom is garbage text or NaN, not an
# error, so it is set explicitly rather than left to --dtype auto.
VLLM_DTYPE = "bfloat16"
# One sentence plus three sentences of history is a few hundred tokens. The
# context vLLM does not reserve is KV cache it does not have to allocate.
VLLM_MAX_MODEL_LEN = 4096
# Fraction of the WHOLE card vLLM claims at start-up, and vLLM refuses to start
# if that much is not free. Whisper, AST, ECAPA and VoxLingua share the card,
# so vLLM has to be started before the audio server loads them.
VLLM_GPU_MEMORY_UTILIZATION = float(
    os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85")
)
VLLM_TRUST_REMOTE_CODE = True
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8001"))

# Translation is not a creative task: the answer should follow the sentence
# closely, not paraphrase it. Low, but not zero, as recommended for Gemma.
TRANSLATE_TEMPERATURE = 0.1
TRANSLATE_TOP_P = 0.95
# At a temperature above zero the same sentence twice is no longer guaranteed
# the same answer. A fixed per-request seed puts that guarantee back, which
# the real test checks.
TRANSLATE_SEED = 0

# Gemma's end-of-turn markers. Without them the model can run on past its
# answer and start the next turn itself, which would be shown as translation.
TRANSLATE_STOP = ("<end_of_turn>", "<eos>")

# Qwen3 reasons before it answers, emitting a <think> block first. For a
# sentence-length translation that is all cost and no benefit: the first run
# against Qwen3.5-9B spent its entire 512-token budget thinking and returned
# no translation at all, at 3.5 s a sentence. Turned off through the chat
# template; the <think> stripping in translate.py stays as a second line of
# defence for a server that ignores the flag. A chat template that has no such
# switch ignores it.
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
#
# Off for gemma-4-12b-it. The first real run put all eight short lines to it
# with and without the hint, and the plain prompt translated every one - はい
# -> Vâng included - so the hint was doing nothing. Off, the message is also
# exactly the agreed prompt. The real test still tries both, so turn it back
# on if a later run shows the plain prompt handing a line back.
SHORT_LINE_HINT_ENABLED = False

# Translation runs off the audio path, so a sentence appears as soon as it is
# transcribed and its translation follows. That needs a queue, and a queue
# needs limits - not because it cannot cope today, but because "it copes" is
# a fact about vLLM's current speed rather than a property of the design.
#
# Measured over three real runs, 66 gaps between committed sentences:
#
#     median gap        3.58 s
#     busiest 8 gaps    0.74 s mean, so 1.35 sentences/s
#     one translation   0.15 s mean, 0.17 s worst
#
# That is 3.7% utilisation: a ten second stall leaves 13.5 sentences behind
# and clears them in two, and delays do not accumulate across stalls because
# the queue empties in between.
#
# The budget is therefore set by when an answer stops being useful, not by
# when the queue stops coping. At a 3.58 s median gap, ten seconds is three
# sentences ago - a translation appearing under a sentence the reader has
# scrolled past reads as a translation of something else.
TRANSLATION_MAX_LAG_SECONDS = 10.0

# A ceiling so a pathological stall cannot grow the queue without bound. At
# the busiest rate observed this is twelve seconds of solid backlog, which
# the budget above should have emptied long before.
TRANSLATION_QUEUE_DEPTH = 16

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
#: Human names, as the prompt writes them after "tiếng": "tiếng Việt".
LANGUAGE_NAMES = {"vi": "Việt", "ja": "Nhật"}


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------
def known_variables() -> list[str]:
    """Every environment variable this file reads, taken from the file.

    Derived rather than listed by hand: a list that has to be kept in step
    with the code is a list that stops being true.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    return sorted(set(re.findall(r'(?:os\.environ\.get|_flag)\(\s*"(\w+)"',
                                 source)))


def overrides() -> dict[str, str]:
    """The ones actually set right now.

    Startup reports these, because a variable left over from an earlier
    terminal changes what the pipeline does and says nothing about it. Three
    measurements in this project were taken against a configuration nobody
    had meant to be running.
    """
    return {name: os.environ[name]
            for name in known_variables() if name in os.environ}
