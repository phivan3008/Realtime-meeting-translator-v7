# Real hardware tests - Client

These tests need a real Windows machine with a real sound card. Do not run
them on the Dev PC agent loop or on the GPU server.

## One-time setup on the Windows Client PC

```powershell
git pull
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.11 -m pip install -r client\requirements.txt
```

## Available tests

| Script | What it proves |
| --- | --- |
| `test_real_audio_capture.py` | WASAPI loopback capture, 16 kHz mono PCM conversion, 200 ms chunking |
| `test_real_vad.py` | Silero VAD on live audio: CPU speed, no false trigger on silence, speech segments, bandwidth saved |

```powershell
# Module 1 - audio capture
python client\tests_real\test_real_audio_capture.py --list
python client\tests_real\test_real_audio_capture.py --seconds 10

# Module 2 - Silero VAD (interactive: one silent phase, one speech phase)
python client\tests_real\test_real_vad.py

# Module 2 - replay an existing recording instead of capturing live
python client\tests_real\test_real_vad.py --wav client\tests_real\output\loopback_YYYYmmdd_HHMMSS.wav
```

Exit code `0` means every check passed. Recordings land in
`client/tests_real/output/` so they can be played back afterwards.
