# Real hardware tests - Client

These tests need a real Windows machine with a real sound card. Do not run
them on the Dev PC agent loop or on the GPU server.

## One-time setup on the Windows Client PC

```powershell
git pull
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.11 -m pip install -r client\requirements.txt
python --version          # must print 3.11.x
```

## Available tests

| Script | What it proves |
| --- | --- |
| `test_real_audio_capture.py` | WASAPI loopback capture, 16 kHz mono PCM conversion, 200 ms chunking |

```powershell
python client\tests_real\test_real_audio_capture.py --list
python client\tests_real\test_real_audio_capture.py --seconds 10
```

The client runs no ML. Silero VAD lives on the GPU server now - see
`server/tests_real/README.md`. The WAV files this test writes are the input
for the server-side VAD test.

Exit code `0` means every check passed. Recordings land in
`client/tests_real/output/` so they can be played back afterwards.
