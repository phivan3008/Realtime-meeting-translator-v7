# Real tests - GPU Server

These run on the GPU pod (VSCode SSH into the Kubernetes pod). Do not run them
on the Dev PC agent loop.

The pod has no sound card, so every audio test is driven by WAV files recorded
on the Windows Client PC with `client/tests_real/test_real_audio_capture.py`.
Those files are already 16 kHz mono 16-bit - the exact format the client
streams - so no conversion is needed.

## One-time setup on the pod

```bash
git pull
python3.11 -m venv .venv
source .venv/bin/activate
python3.11 -m pip install -r server/requirements.txt
python --version          # must print 3.11.x
```

## Available tests

| Script | What it proves |
| --- | --- |
| `test_real_vad.py` | Silero VAD: model loads, runs far faster than real time, no false trigger on a quiet recording, correct speech segments and timestamps |

```bash
python3.11 server/tests_real/test_real_vad.py \
    --speech recordings/meeting_speech.wav \
    --silence recordings/quiet_room.wav
```

Add `--onnx` to run the ONNX model instead of the torch jit one.

Exit code `0` means every check passed. The gated audio is written to
`server/tests_real/output/` - copy it back to a machine with speakers and
listen: every word must still be there, with the long pauses cut out.
