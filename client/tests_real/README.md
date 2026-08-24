# Real hardware tests - Client

These tests need a real Windows machine with a real sound card. Do not run
them on the Dev PC agent loop or on the GPU server.

## One-time setup on the Windows Client PC

```powershell
git pull
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version          # must print 3.11.x

# The lock file is the exact resolved set for Windows x86_64 / cp311.
# client\requirements.txt lists intent; the lock is what reproduces.
py -3.11 -m pip install -r client\requirements.lock.txt
```

The client runs no ML. Silero VAD lives on the GPU server - see
`server/tests_real/README.md`.

## Available tests

| Script | What it proves |
| --- | --- |
| `test_real_audio_capture.py` | WASAPI loopback capture, 16 kHz mono PCM conversion, 200 ms chunking |
| `test_real_stream.py` | End to end: live audio from this machine reaches the GPU server, and VAD events come back fast enough |

```powershell
# Module 1 - audio capture only
python client\tests_real\test_real_audio_capture.py --list
python client\tests_real\test_real_audio_capture.py --seconds 10

# Module 3 - stream to the server (the server must already be running)
python client\tests_real\test_real_stream.py --url ws://127.0.0.1:8000 --seconds 25
```

### Reaching the GPU pod

The pod is usually not routable from this machine. Open an SSH tunnel first,
in a separate PowerShell window, and leave it running:

```powershell
ssh -N -L 8000:127.0.0.1:8000 <user>@<pod-ssh-host>
```

Then `--url ws://127.0.0.1:8000` reaches the pod through the tunnel. If the
pod is directly routable, point `--url` at its address instead.

Exit code `0` means every check passed. Recordings land in
`client/tests_real/output/` - the WAV files from `test_real_audio_capture.py`
are also the input for the server-side VAD test.
