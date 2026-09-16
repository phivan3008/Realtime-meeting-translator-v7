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
| `test_real_streaming.py` | The whole pipeline on a recorded meeting, fed in 200 ms chunks exactly as the client sends it: Japanese without spaces between characters, no word shown twice at a join, running text that does not rewrite what is being read, no stage failures, inside real time |

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

```bash
# Start with three minutes; a full thirty-minute meeting takes a few minutes
# of H100 time. --translate needs vLLM running.
python3.11 server/tests_real/test_real_streaming.py     --wav recordings/meeting_30min.wav --limit-seconds 180     --baseline recordings/meeting-20260911-144023.debug.txt     --baseline recordings/meeting-20260910-144120.debug.txt
```

`test_real_streaming.py` writes `server/tests_real/output/streaming.debug.txt`
in the client's own format, so it can be compared with any client log of the
same meeting (`python3.11 -m server.analysis.compare_logs ...`).

`recordings/` is not in git. The `--baseline` logs are the ones the Windows
client wrote; copy them to the pod first (`scp`), or leave `--baseline` out and
copy `streaming.debug.txt` back to run `compare_logs` where the logs are. The WAV must
be 16 kHz mono 16-bit; convert with
`ffmpeg -i meeting.m4a -ac 1 -ar 16000 -sample_fmt s16 meeting_16k.wav`.

`test_real_buffer.py` writes one WAV per sentence into
`server/tests_real/output/<name>_utterances/`. Listen to any file named
`*_max_duration*.wav`: the cut must fall between words, not through one.

## The translation server

Translation runs in its own vLLM process, not inside the audio server. vLLM
reserves a slice of the GPU at load, and in-process it would have to be tuned
by hand against Whisper's allocation; out of process each side sees a GPU it
can reason about, and the LLM can be restarted without dropping a meeting.

```bash
python3.11 server/launch_vllm.py          # add --print to see the command only
```

It builds the command from `server/config.py`:

```bash
python3.11 -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-12b-it --port 8001 --dtype bfloat16 \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --trust-remote-code
```

`bfloat16` is required for Gemma: float16 gives garbage or NaN, not an error.
vLLM claims 85% of the whole card and will not start if that much is not
free, so start it **before** the audio server loads Whisper and the three small
models into the rest. `TRANSLATE_BASE_URL` points the audio server at it.

Gemma's chat template has no system role, so the app sends one `user` message
holding the instruction and the sentence, with `temperature` 0.1, `top_p` 0.95,
a fixed `seed`, and `stop` `["<end_of_turn>", "<eos>"]`.

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
