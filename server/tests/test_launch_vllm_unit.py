"""Unit tests for ``server/launch_vllm.py``.

Only the command line is checked; nothing here starts vLLM.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server import launch_vllm  # noqa: E402
from server.config import TRANSLATE_BASE_URL, TRANSLATE_MODEL  # noqa: E402


def flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def test_the_engine_arguments_are_the_agreed_ones():
    cmd = launch_vllm.command(python="python3.11")
    assert cmd[:3] == ["python3.11", "-m", "vllm.entrypoints.openai.api_server"]
    assert flag(cmd, "--model") == TRANSLATE_MODEL
    assert flag(cmd, "--dtype") == "bfloat16"
    assert flag(cmd, "--max-model-len") == "4096"
    assert 0.85 <= float(flag(cmd, "--gpu-memory-utilization")) <= 0.9
    assert "--trust-remote-code" in cmd


def test_the_port_is_the_one_the_audio_server_calls():
    cmd = launch_vllm.command(python="python3.11")
    assert TRANSLATE_BASE_URL.endswith(f":{flag(cmd, '--port')}/v1")


def test_extra_arguments_are_passed_through_last():
    cmd = launch_vllm.command(extra=["--host", "127.0.0.1"], python="p")
    assert cmd[-2:] == ["--host", "127.0.0.1"]


def test_a_local_checkpoint_path_can_be_given():
    cmd = launch_vllm.command(model="/models/gemma", python="p")
    assert flag(cmd, "--model") == "/models/gemma"


def test_an_empty_model_is_refused():
    with pytest.raises(ValueError):
        launch_vllm.command(model="", python="p")


def test_print_only_starts_nothing(monkeypatch, capsys):
    def refuse(*_args):
        raise AssertionError("vLLM must not be started with --print")

    monkeypatch.setattr(launch_vllm.os, "execv", refuse)
    assert launch_vllm.main(["--print"]) == 0
    out = capsys.readouterr().out
    assert "--dtype bfloat16" in out
    assert "--max-model-len 4096" in out
