"""Both SpeechBrain loaders copy checkpoints out of the cache, not symlink.

On Windows a symlink needs admin rights or Developer Mode; without them
``from_hparams`` fails with WinError 1314 before any model loads. torch and
speechbrain are not installed on the Dev PC, so both are faked and the fake
``from_hparams`` records what it was asked and then refuses to load.
"""

from __future__ import annotations

import enum
import sys
import types

import pytest

from server.pipeline.diarization import DiarizationError, EcapaEmbedder
from server.pipeline.lid import LanguageIdError, VoxLinguaClassifier


class LocalStrategy(enum.Enum):
    SYMLINK = 1
    COPY = 2


@pytest.fixture
def requests(monkeypatch):
    seen = []

    class EncoderClassifier:
        @classmethod
        def from_hparams(cls, **kwargs):
            seen.append(kwargs)
            raise RuntimeError("fake: no model here")

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    classifiers = types.ModuleType("speechbrain.inference.classifiers")
    classifiers.EncoderClassifier = EncoderClassifier
    fetching = types.ModuleType("speechbrain.utils.fetching")
    fetching.LocalStrategy = LocalStrategy
    for name, module in {
        "torch": torch,
        "speechbrain": types.ModuleType("speechbrain"),
        "speechbrain.inference": types.ModuleType("speechbrain.inference"),
        "speechbrain.inference.classifiers": classifiers,
        "speechbrain.utils": types.ModuleType("speechbrain.utils"),
        "speechbrain.utils.fetching": fetching,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return seen


@pytest.mark.parametrize("loader, error", [
    (EcapaEmbedder, DiarizationError),
    (VoxLinguaClassifier, LanguageIdError),
])
def test_checkpoints_are_copied_not_symlinked(requests, loader, error):
    with pytest.raises(error, match="fake: no model here"):
        loader(device="cpu")
    assert requests[0]["local_strategy"] is LocalStrategy.COPY
    assert requests[0]["run_opts"] == {"device": "cpu"}
