"""Entry point for the meeting window.

    py -3.11 -m client.ui.main --url ws://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_URL = "ws://127.0.0.1:8000"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"server WebSocket base URL (default {DEFAULT_URL})")
    parser.add_argument("--device", default="",
                        help="part of the loopback device name, if the wrong "
                             "one is picked")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Imported here so --help works on a machine with no Qt platform plugin.
    from PySide6.QtWidgets import QApplication

    from client.ui.window import MeetingWindow

    app = QApplication(sys.argv[:1])
    window = MeetingWindow(args.url, args.device or None)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
