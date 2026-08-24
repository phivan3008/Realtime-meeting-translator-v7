# Real tests - GPU Server

These run on the GPU pod (VSCode SSH into the Kubernetes pod). Do not run them
on the Dev PC agent loop.

The pod has no sound card, so every offline audio test is driven by WAV files
recorded on the Windows Client PC with
`client/tests_real/test_real_audio_capture.py`. Those files are already
16 kHz mono 16-bit - the exact format the client streams - so no conversion is
needed.

## One-time setup on the pod

```bash
git pull
python3.11 -m venv .venv
source .venv/bin/activate
python --version          # must print 3.11.x

# The lock file is the exact resolved set for linux x86_64 / cp311.
# Prefer it: server/requirements.txt lists intent, the lock is reproducible.
python3.11 -m pip install -r server/requirements.lock.txt
```

Check the resolution before committing to the download:

```bash
python3.11 -m pip install --dry-run -r server/requirements.lock.txt
```

Never install into the system interpreter as root. That downgrades numpy and
protobuf underneath anything else sharing it, which is how this project lost
an afternoon once already.

## Offline: the pipeline on recorded audio

| Script | What it proves |
| --- | --- |
| `test_real_vad.py` | Silero VAD: model loads, runs far faster than real time, no false trigger on a quiet recording, correct speech segments and timestamps |
| `test_real_buffer.py` | Stream Buffer Manager: sentences partition the speech exactly, none outstays 7 s, timestamps line up, partials keep cadence |
| `test_real_noise.py` | Deep Noise Filter (AST): loads on the pod, keeps real speech, drops recorded keyboard and coughing, costs almost nothing |
| `test_real_overlap.py` | Overlap Resolver: a clean voice survives untouched, a voice 20 dB under it is squashed |
| `test_real_diarization.py` | Speaker voiceprints: measures the same-speaker and different-speaker cosine distributions. **Every `--voice` file must hold exactly one person** |
| `test_real_lid.py` | Language ID: Vietnamese against Japanese, per sentence, with the margin behind every verdict |
| `test_real_asr.py` | Whisper large-v3: decodes far faster than real time, keeps the forced language, and prints the transcripts for you to read |
| `test_real_translate.py` | Translation through vLLM: answers are translations rather than conversations, deterministic, history reaches the model, and what one costs |

```bash
python3.11 server/tests_real/test_real_vad.py \
    --speech recordings/meeting_speech.wav \
    --silence recordings/quiet_room.wav

python3.11 server/tests_real/test_real_buffer.py \
    --speech recordings/meeting_speech.wav
```

Add `--onnx` to either one to run the ONNX model instead of the torch jit one.

```bash
python3.11 server/tests_real/test_real_noise.py \
    --speech recordings/meeting_speech.wav \
    --noise recordings/keyboard.wav \
    --noise recordings/cough.wav
```

If the pod has no internet access it cannot pull the AST checkpoint from
HuggingFace. Download it once elsewhere, copy the directory over, and either
pass `--model-id <dir>` or export `AST_MODEL_ID=<dir>` before starting the
server.

`test_real_buffer.py` writes one WAV per sentence into
`server/tests_real/output/<name>_utterances/`. Listen to any file named
`*_max_duration*.wav`: the cut must fall between words, not through one.

## The translation server

Translation runs in its own vLLM process, not inside the audio server. vLLM
reserves a slice of the GPU at load, and in-process it would have to be tuned
by hand against Whisper's allocation; out of process each side sees a GPU it
can reason about, and the LLM can be restarted without dropping a meeting.

```bash
python3.11 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B \
    --port 8001 --gpu-memory-utilization 0.55
```

Leave headroom: Whisper large-v3 and the three small models want a few GB of
the same card. `TRANSLATE_BASE_URL` points the audio server at it.

The checkpoint must match `TRANSLATE_MODEL` in `server/config.py`. The client
checks at connect time and refuses a server running something else: vLLM would
answer a wrong-model request perfectly happily, and the only symptom would be
translations quietly worse than they should be.

## Online: serve the client

Start the WebSocket server, then run `client/tests_real/test_real_stream.py`
on the Windows Client PC against it.

```bash
python3.11 -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Check it locally first:

```bash
curl -s http://127.0.0.1:8000/health
```

Expect `{"status": "ok", ..., "vad_loaded": true, "session_active": false}`.
The server handles **one meeting at a time**; a second connection is refused
with WebSocket code 1013.

Exit code `0` means every check passed. Output files land in
`server/tests_real/output/`.
