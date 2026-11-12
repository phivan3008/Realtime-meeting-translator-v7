# Real tests - the pipeline, on the Windows machine

The pipeline runs on the Windows machine (CPU) and calls the two models on the
GPU pod over HTTP: Whisper behind `gpu_server/whisper_server.py` (port 8002)
and Gemma 4 behind vLLM (port 8001). So every script here runs **on Windows**,
and the ones that need Whisper or Gemma need the pod running and both ports
forwarded to `127.0.0.1` (see `gpu_server/README.md` and the root
`README.md`). Do not run them on the Dev PC agent loop.

The offline tests are driven by WAV files recorded with
`client/tests_real/test_real_audio_capture.py`. Those files are already
16 kHz mono 16-bit - the exact format the client streams - so no conversion is
needed.

## One-time setup on Windows

```powershell
git pull
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version          # must print 3.11.x

# The lock is the exact resolved set for Windows x86_64 / cp311, CPU torch.
py -3.11 -m pip install -r server\requirements.lock.txt
py -3.11 -m pip install -r client\requirements.lock.txt
```

Check the resolution before committing to the download:

```powershell
py -3.11 -m pip install --dry-run -r server\requirements.lock.txt
```

Never install into the system interpreter. That downgrades numpy and protobuf
underneath anything else sharing it, which is how this project lost an
afternoon once already.

## First: can this machine reach the pod?

```powershell
py -3.11 server\tests_real\test_real_remote_models.py --wav recordings\meeting_speech.wav
```

Both servers answer and serve the expected checkpoints, Whisper returns words
with timestamps and real scores, the same audio decodes to the same answer
twice, a partial round trip fits inside 600 ms, a final fits its budget, and
one translation comes back. Everything below depends on this passing.

## Offline: the pipeline on recorded audio

| Script | Needs the pod | What it proves |
| --- | --- | --- |
| `test_real_remote_models.py` | yes | The seam to the pod: reachability, checkpoints, word timestamps, determinism, round-trip latency |
| `test_real_vad.py` | no | Silero VAD on this CPU: model loads, runs far faster than real time, no false trigger on a quiet recording, correct speech segments and timestamps |
| `test_real_buffer.py` | no | Stream Buffer Manager: sentences partition the speech exactly, none outstays 7 s, timestamps line up, partials keep cadence |
| `test_real_noise.py` | no | Deep Noise Filter (AST) on CPU: loads, keeps real speech, drops recorded keyboard and coughing, costs almost nothing |
| `test_real_overlap.py` | no | Overlap Resolver: a clean voice survives untouched, a voice 20 dB under it is squashed |
| `test_real_diarization.py` | no | Speaker voiceprints on this CPU: measures the same-speaker and different-speaker cosine distributions. **Every `--voice` file must hold exactly one person** |
| `test_real_lid.py` | no | Language ID on this CPU: Vietnamese against Japanese, per sentence, with the margin behind every verdict |
| `test_real_asr.py` | yes | Whisper large-v3 on the pod: decodes far faster than real time **round trip included**, keeps the forced language, and prints the transcripts for you to read |
| `test_real_translate.py` | yes | Translation through vLLM: answers are translations rather than conversations, deterministic, history reaches the model, and what one costs |
| `test_real_streaming.py` | yes | The whole pipeline on a recorded meeting, fed in 200 ms chunks exactly as the client sends it: Japanese without spaces between characters, no word shown twice at a join, running text that does not rewrite what is being read, no stage failures, inside real time |

```powershell
py -3.11 server\tests_real\test_real_vad.py `
    --speech recordings\meeting_speech.wav `
    --silence recordings\quiet_room.wav

py -3.11 server\tests_real\test_real_buffer.py `
    --speech recordings\meeting_speech.wav
```

Add `--onnx` to either one to run the ONNX model instead of the torch jit one.

```powershell
py -3.11 server\tests_real\test_real_noise.py `
    --speech recordings\meeting_speech.wav `
    --noise recordings\keyboard.wav `
    --noise recordings\cough.wav
```

```powershell
# Start with three minutes. --translate needs vLLM on the pod.
py -3.11 server\tests_real\test_real_streaming.py --wav recordings\meeting_30min.wav --limit-seconds 180 `
    --baseline recordings\meeting-20260911-144023.debug.txt `
    --baseline recordings\meeting-20260910-144120.debug.txt
```

`test_real_asr.py` and `test_real_streaming.py` take `--asr-url` (default
`ASR_SERVER_URL`, `http://127.0.0.1:8002`) and `--backend remote|local`.

`test_real_streaming.py` writes `server/tests_real/output/streaming.debug.txt`
in the client's own format, so it can be compared with any client log of the
same meeting (`py -3.11 -m server.analysis.compare_logs ...`). The logs and
the replay are on the same machine now, so `--baseline` needs no copying.

`recordings/` is not in git. The WAV must be 16 kHz mono 16-bit; convert with
`ffmpeg -i meeting.m4a -ac 1 -ar 16000 -sample_fmt s16 meeting_16k.wav`.

`test_real_buffer.py` writes one WAV per sentence into
`server/tests_real/output/<name>_utterances/`. Listen to any file named
`*_max_duration*.wav`: the cut must fall between words, not through one.

## The translation server

vLLM runs on the pod and is started there (`gpu_server/README.md`). The
command is built from `server/config.py`; print it here with:

```powershell
py -3.11 server\launch_vllm.py --print
```

`bfloat16` is required for Gemma: float16 gives garbage or NaN, not an error.
vLLM claims 85% of the whole card and will not start if that much is not
free, so start it **before** `whisper_server.py`. `TRANSLATE_BASE_URL` points
the pipeline at it.

Gemma's chat template has no system role, so the app sends one `user` message
holding the instruction and the sentence, with `temperature` 0.1, `top_p` 0.95,
a fixed `seed`, and `stop` `["<end_of_turn>", "<eos>"]`.

The checkpoint must match `TRANSLATE_MODEL` in `server/config.py`. The
pipeline checks at connect time and refuses a server running something else:
vLLM would answer a wrong-model request perfectly happily, and the only
symptom would be translations quietly worse than they should be.

## Online: serve the client

Start the pipeline, then run `client/tests_real/test_real_stream.py` or the
meeting window against it, on the same machine.

```powershell
py -3.11 -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Check it first:

```powershell
curl.exe -s http://127.0.0.1:8000/health
```

Expect `{"status": "ok", ..., "vad_loaded": true, "asr_loaded": true,
"asr_source": "whisper large-v3 at http://127.0.0.1:8002 on cuda (float16)",
"session_active": false}`. The pipeline handles **one meeting at a time**; a
second connection is refused with WebSocket code 1013.

Exit code `0` means every check passed. Output files land in
`server/tests_real/output/`.
