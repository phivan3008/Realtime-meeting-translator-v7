r"""Every tunable constant has an entry in docs/TUNING.md.

A number nobody can explain gets changed by guess, and the guesses in this
project have been expensive. This keeps the document from drifting behind the
code silently.

Run with::

    .venv\Scripts\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TUNING = ROOT / "docs" / "TUNING.md"

#: Names that are not tuning knobs: the audio contract (fixed by
#: common/protocol.py) and pure labels.
NOT_TUNABLE = {
    "SAMPLE_RATE", "SAMPLE_WIDTH", "CHANNELS", "CHUNK_BYTES",
    "CHUNK_SAMPLES", "CHUNK_DURATION_MS", "PROTOCOL_VERSION",
    "SPEAKER_UNKNOWN", "LID_UNKNOWN", "TRANSLATE_PAIR", "LANGUAGE_NAMES",
    "TRANSLATE_MAX_TOKENS", "TARGET_SAMPLE_RATE", "TARGET_CHANNELS",
    "TARGET_SAMPLE_WIDTH",
    # Paths and device pickers, all set by environment variable.
    "AST_MODEL_ID", "NOISE_DEVICE", "SPEAKER_EMBEDDING_MODEL",
    "SPEAKER_DEVICE", "SPEAKER_CACHE_DIR", "LID_MODEL", "LID_DEVICE",
    "LID_CACHE_DIR", "ASR_MODEL", "ASR_DEVICE", "ASR_COMPUTE_TYPE",
    "ASR_CACHE_DIR", "TRANSLATE_BASE_URL", "TRANSLATE_MODEL",
    "TRANSLATE_TIMEOUT_S",
}


def constants(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*) = ", line)
        if match and match.group(1) not in NOT_TUNABLE:
            names.append(match.group(1))
    return names


TUNABLE = (constants(ROOT / "server" / "config.py")
           + constants(ROOT / "client" / "config.py"))


def test_there_are_constants_to_check():
    """An empty list would make the test below pass for nothing."""
    assert len(TUNABLE) > 20


@pytest.mark.parametrize("name", TUNABLE)
def test_every_tunable_constant_is_documented(name):
    assert name in TUNING.read_text(encoding="utf-8"), (
        f"{name} has no entry in docs/TUNING.md. A number nobody can explain "
        f"gets changed by guess."
    )


def test_the_word_lists_are_documented_where_they_live():
    """They are data files now, so their guide sits beside them."""
    assert (ROOT / "server" / "data" / "README.md").exists()
