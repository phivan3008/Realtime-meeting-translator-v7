"""Build the vLLM command for the translation model from config.py.

vLLM runs on the GPU Server pod, and the pod may not hold this repository's
pipeline code - so this script is run on the Windows machine with
``--print``, and the line it prints is what gets pasted on the pod. The same
command is written out in ``gpu_server/README.md``; a unit test keeps the
two in step.

Start vLLM BEFORE ``gpu_server/whisper_server.py``. vLLM claims
``VLLM_GPU_MEMORY_UTILIZATION`` of the whole card at start-up and refuses to
start if that much is not free; started after Whisper, it can fail for want
of memory that it would have fitted around had it gone first.

Usage
-----
    py -3.11 server/launch_vllm.py --print              # show the command
    py -3.11 server/launch_vllm.py --print -- --host 127.0.0.1

Anything after ``--`` is passed to vLLM unchanged. Without ``--print`` the
command is started here, which only works on a machine that has vLLM.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.config import (  # noqa: E402
    TRANSLATE_MODEL,
    VLLM_DTYPE,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_MODEL_LEN,
    VLLM_PORT,
    VLLM_TRUST_REMOTE_CODE,
)


#: The interpreter on the pod, for the printed command.
POD_PYTHON = "python3.11"


def command(model: str = TRANSLATE_MODEL, extra: Iterable[str] = (),
            python: str = sys.executable) -> list[str]:
    """The vLLM command line, as a list so nothing is re-parsed by a shell."""
    if not model:
        raise ValueError("TRANSLATE_MODEL is empty; pass --model")
    args = [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(VLLM_PORT),
        "--dtype", VLLM_DTYPE,
        "--max-model-len", str(VLLM_MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(VLLM_GPU_MEMORY_UTILIZATION),
    ]
    if VLLM_TRUST_REMOTE_CODE:
        args.append("--trust-remote-code")
    return args + list(extra)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=TRANSLATE_MODEL,
                        help="checkpoint id or local path")
    parser.add_argument("--print", action="store_true", dest="print_only",
                        help="print the command and exit")
    parser.add_argument("--python", default="",
                        help="interpreter in the printed command; default "
                             "python3.11 with --print, this one otherwise")
    parser.add_argument("extra", nargs="*", help="passed to vLLM after --")
    args = parser.parse_args(argv)

    # Printed for the pod, where this machine's interpreter path means
    # nothing.
    python = args.python or (POD_PYTHON if args.print_only else sys.executable)
    cmd = command(args.model, args.extra, python=python)
    print(shlex.join(cmd), flush=True)
    if args.print_only:
        return 0
    os.execv(cmd[0], cmd)
    return 0                                        # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
