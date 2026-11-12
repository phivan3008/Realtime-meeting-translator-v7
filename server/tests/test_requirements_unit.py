"""Unit tests for the pins of the Windows pipeline and of the GPU pod.

``server/requirements*.txt`` is the pipeline on the Windows machine (CPU),
``gpu_server/requirements*.txt`` the two model servers on the pod.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[1]
GPU_SERVER = SERVER.parent / "gpu_server"


def pins(path: Path) -> dict[str, tuple[int, ...]]:
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        # Only the leading release numbers: "1.0.post1" and "2.11.0+cu128"
        # compare as (1, 0) and (2, 11, 0).
        match = re.match(
            r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==(\d+(?:\.\d+)*)", line)
        if match:
            found[match.group(1).lower()] = tuple(
                int(part) for part in match.group(2).split("."))
    return found


@pytest.mark.parametrize("name", ["requirements.txt", "requirements.lock.txt"])
def test_vllm_that_reads_head_dim_globally_gets_a_transformers_that_allows_it(name):
    """Transformers 5.15 made Gemma 4's head_dim per-layer; vLLM before 0.28
    reads it from the global config and fails with
    AmbiguousGlobalPerLayerAttributeError before loading anything."""
    found = pins(GPU_SERVER / name)
    if found["vllm"] < (0, 28):
        assert found["transformers"] < (5, 15), found["transformers"]


@pytest.mark.parametrize("name", ["requirements.txt", "requirements.lock.txt"])
def test_transformers_is_new_enough_for_the_gemma4_unified_checkpoint(name):
    """gemma-4-12b-it is a gemma4_unified checkpoint, which vLLM 0.26.0
    imports from transformers.models.gemma4_unified (first in 5.10.0)."""
    assert pins(GPU_SERVER / name)["transformers"] >= (5, 10)


def test_the_pod_lock_agrees_with_the_direct_pins():
    direct = pins(GPU_SERVER / "requirements.txt")
    locked = pins(GPU_SERVER / "requirements.lock.txt")
    for name in direct:
        assert locked[name] == direct[name], name


def test_the_pipeline_lock_agrees_with_the_direct_pins():
    direct = pins(SERVER / "requirements.txt")
    locked = pins(SERVER / "requirements.lock.txt")
    for name in direct:
        assert locked[name] == direct[name], name


@pytest.mark.parametrize("name", ["requirements.txt", "requirements.lock.txt"])
@pytest.mark.parametrize("package", ["vllm", "faster-whisper",
                                     "ctranslate2", "pyannote.audio"])
def test_the_windows_pipeline_installs_no_gpu_server(name, package):
    """The two big models are served by the pod; the pipeline only calls."""
    assert package not in pins(SERVER / name)


@pytest.mark.parametrize("package", ["vllm", "faster-whisper"])
def test_the_pod_serves_both_models(package):
    assert package in pins(GPU_SERVER / "requirements.lock.txt")


def test_the_pod_holds_no_pipeline_library():
    """The pod may only serve models."""
    pod = pins(GPU_SERVER / "requirements.txt")
    for package in ("speechbrain", "silero-vad", "pedalboard"):
        assert package not in pod, package


def test_packages_on_both_machines_are_the_same_version():
    """The pipeline was measured with the pod's versions; moving it to
    Windows changes the machine, not the libraries."""
    windows = pins(SERVER / "requirements.lock.txt")
    pod = pins(GPU_SERVER / "requirements.lock.txt")
    shared = set(windows) & set(pod)
    assert {"numpy", "torch", "transformers"} <= shared
    for name in shared:
        assert windows[name] == pod[name], name
