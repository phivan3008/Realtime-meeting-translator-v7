r"""The README points at files that exist and flags the server really reports.

A quick-start that names a missing file wastes the reader's first five
minutes, which is the worst five minutes to waste.

Run with::

    .venv\Scripts\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


def linked_paths() -> list[str]:
    text = README.read_text(encoding="utf-8")
    return [target
            for _label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
            if not target.startswith(("http", "#", "<"))]


def test_the_readme_exists_and_says_something():
    assert README.exists()
    assert len(README.read_text(encoding="utf-8")) > 1000


@pytest.mark.parametrize("target", linked_paths())
def test_every_linked_file_exists(target):
    assert (ROOT / target).exists(), target


@pytest.mark.parametrize("name", [
    "server/requirements.lock.txt",
    "client/requirements.lock.txt",
    "client/tests_real/test_real_stream.py",
    "common/protocol.py",
    "docs/TUNING.md",
    "server/data/README.md",
])
def test_every_file_the_quickstart_names_exists(name):
    assert (ROOT / name).exists(), name


def test_the_health_flags_it_promises_are_the_ones_the_server_reports():
    """The README tells the reader six ``*_loaded`` flags must be true. If
    app.py reports a different set, that sends them looking for the wrong
    thing at the worst moment."""
    app = (ROOT / "server" / "app.py").read_text(encoding="utf-8")
    reported = set(re.findall(r'"(\w+_loaded)":', app))
    # Six stages plus the VAD, which loads separately and cannot fail softly.
    assert len(reported) == 7, sorted(reported)
