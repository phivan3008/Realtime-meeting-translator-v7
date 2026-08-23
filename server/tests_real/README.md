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
python3.11 -m pip install -r server/requirements.txt
python --version          # must print 3.11.x
```

## Offline: the pipeline on recorded audio

| Script | What it proves |
| --- | --- |
| `test_real_vad.py` | Silero VAD: model loads, runs far faster than real time, no false trigger on a quiet recording, correct speech segments and timestamps |
| `test_real_buffer.py` | Stream Buffer Manager: sentences partition the speech exactly, none outstays 7 s, timestamps line up, partials keep cadence |

```bash
python3.11 server/tests_real/test_real_vad.py \
    --speech recordings/meeting_speech.wav \
    --silence recordings/quiet_room.wav

python3.11 server/tests_real/test_real_buffer.py \
    --speech recordings/meeting_speech.wav
```

Add `--onnx` to either one to run the ONNX model instead of the torch jit one.

`test_real_buffer.py` writes one WAV per sentence into
`server/tests_real/output/<name>_utterances/`. Listen to any file named
`*_max_duration*.wav`: the cut must fall between words, not through one.

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
