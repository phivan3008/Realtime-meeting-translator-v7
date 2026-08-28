"""Start the audio server, naming the translation model's family.

    python3.11 -m server.run --profile gemma

``uvicorn server.app:app`` still works and still defaults to the profile this
project shipped with, so nothing that was running keeps running differently.
This exists because the profile has to be chosen before the app imports its
models, and a uvicorn command line has nowhere to say so.

The client is not told which model is behind the translation. It sends audio
and receives sentences either way.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.pipeline.profiles import DEFAULT_PROFILE, PROFILES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", default=os.environ.get("TRANSLATE_PROFILE",
                                            DEFAULT_PROFILE),
        choices=sorted(PROFILES),
        help="which translation model vLLM is serving (default: "
             f"{DEFAULT_PROFILE})")
    parser.add_argument(
        "--model", default="",
        help="checkpoint name, when it differs from the profile's default")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Set before the app is imported: the backend reads the profile when it
    # builds, and the models load on the first request.
    os.environ["TRANSLATE_PROFILE"] = args.profile
    if args.model:
        os.environ["TRANSLATE_MODEL"] = args.model

    import uvicorn

    uvicorn.run("server.app:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
