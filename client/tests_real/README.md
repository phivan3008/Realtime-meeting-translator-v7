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
python -m pip install -r client\requirements.lock.txt
```

After activating the venv, run everything with `python`, never `py -3.11`:
the launcher given an explicit version ignores the active venv.

The client runs no ML. The pipeline (`server/`) runs on this same machine as
a second process on `127.0.0.1:8000`; only Whisper and Gemma run on the GPU
pod - see the root `README.md` and `server/tests_real/README.md`.

## Available tests

| Script | What it proves |
| --- | --- |
| `test_real_audio_capture.py` | WASAPI loopback capture, 16 kHz mono PCM conversion, 200 ms chunking |
| `test_real_stream.py` | End to end: live audio from this machine goes through all eight server stages and comes back as transcripts and translations |
| `python -m client.ui.main` | The meeting window: running text, committed sentences, translations, corrected speaker labels, and the two files each meeting writes |

```powershell
# Module 1 - audio capture only
python client\tests_real\test_real_audio_capture.py --list
python client\tests_real\test_real_audio_capture.py --seconds 10

# Module 3 - stream to the server (the server must already be running)
python client\tests_real\test_real_stream.py --url ws://127.0.0.1:8000 --seconds 25
```

The pipeline must have every stage loaded before this test means anything -
`/health` reports each one, and the test names whichever is missing.

### The pipeline and the pod

`--url ws://127.0.0.1:8000` is the pipeline on this machine, started with
`python -m uvicorn server.app:app --host 127.0.0.1 --port 8000`. The pipeline
in turn needs the pod's two model ports forwarded here, in a separate
PowerShell window left running:

```powershell
ssh -N -L 8001:127.0.0.1:8001 -L 8002:127.0.0.1:8002 <user>@<pod-ssh-host>
```

or the VSCode **Ports** tab. Open the tunnel before starting the pipeline:
ASR and translation connect once, at startup.

Exit code `0` means every check passed. Recordings land in
`client/tests_real/output/` - the WAV files from `test_real_audio_capture.py`
are also the input for the server-side VAD test.

## The meeting window, on a real meeting

```powershell
python -m client.ui.main --url ws://127.0.0.1:8000
```

Press **Bắt đầu** and play the meeting recording that the earlier runs used,
from the start, for at least ten minutes. Close the window when it is done -
closing sends `bye` and waits for the last translation.

Each run writes `recordings/meeting-<date>-<time>.txt` and
`recordings/meeting-<date>-<time>.debug.txt`. Compare the new debug log with
the earlier runs of the same meeting:

```powershell
python -m server.analysis.compare_logs recordings\meeting-<new>.debug.txt `
    recordings\meeting-20260911-144023.debug.txt `
    recordings\meeting-20260910-144120.debug.txt
```

It says first whether the logs are the same meeting; the numbers below that
only mean something if they are. Rates matter, not counts, when the new run is
shorter than the old ones.
