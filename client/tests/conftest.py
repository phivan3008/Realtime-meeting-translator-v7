"""One Qt application for the whole test session.

A process may hold exactly one, and it must be a ``QApplication`` rather than
a bare ``QCoreApplication`` because the window tests build widgets. Two files
each making their own left pytest hanging at teardown with no summary line,
which is a confusing way to find that out.

Offscreen, so nothing needs a display.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_app():
    """The application object. Signals do not deliver without one."""
    pytest.importorskip("PySide6", reason="the UI needs PySide6 installed")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
