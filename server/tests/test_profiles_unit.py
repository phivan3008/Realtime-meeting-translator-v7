"""Unit tests for the per-model translation profiles.

No network and no model: what is tested is the shape of the request, which is
where the two families differ in ways that fail outright rather than fail
quietly.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.pipeline.profiles import (
    DEFAULT_PROFILE,
    GEMMA,
    PROFILES,
    QWEN,
    ModelProfile,
    profile_for,
    profile_for_model,
)

SYSTEM = "You translate meeting speech."
USER = "Translate this line into Japanese:\nXin chào."


# ---------------------------------------------------------------------------
# The system role
# ---------------------------------------------------------------------------
def test_qwen_gets_a_system_turn():
    messages = QWEN.messages(SYSTEM, USER)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM


def test_gemma_gets_no_system_turn():
    """Gemma's chat template has only user and model turns. A request with a
    system message is a template error, not a warning."""
    messages = GEMMA.messages(SYSTEM, USER)
    assert [m["role"] for m in messages] == ["user"]


def test_folding_keeps_both_halves_and_their_order():
    """The instructions have to survive the fold, and come first."""
    content = GEMMA.messages(SYSTEM, USER)[0]["content"]
    assert SYSTEM in content
    assert USER in content
    assert content.index(SYSTEM) < content.index(USER)


def test_the_two_halves_are_separated():
    content = GEMMA.messages(SYSTEM, USER)[0]["content"]
    assert content != SYSTEM + USER, "they ran together"


# ---------------------------------------------------------------------------
# Thinking
# ---------------------------------------------------------------------------
def test_qwen_is_told_not_to_think_aloud():
    """The first live run spent all 512 tokens on a <think> block and
    returned no translation."""
    assert QWEN.extra_body["chat_template_kwargs"]["enable_thinking"] is False


def test_gemma_is_sent_no_template_kwargs():
    """Its template declares none, and an undeclared one is an error on some
    builds rather than being ignored."""
    assert GEMMA.extra_body == {}


# ---------------------------------------------------------------------------
# Choosing one
# ---------------------------------------------------------------------------
def test_the_default_is_what_the_project_shipped_with(monkeypatch):
    """Nothing set anywhere still runs the model this project was measured
    on. A branch that quietly changed the default would make every earlier
    measurement not apply."""
    monkeypatch.delenv("TRANSLATE_PROFILE", raising=False)
    assert profile_for().name == DEFAULT_PROFILE == "qwen"


def test_a_profile_can_be_named():
    assert profile_for("gemma") is GEMMA


def test_the_name_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("TRANSLATE_PROFILE", "gemma")
    assert profile_for().name == "gemma"


def test_an_explicit_name_beats_the_environment(monkeypatch):
    monkeypatch.setenv("TRANSLATE_PROFILE", "gemma")
    assert profile_for("qwen") is QWEN


@pytest.mark.parametrize("name", ["GEMMA", " gemma ", "Gemma"])
def test_the_name_is_forgiving_about_case_and_space(name):
    assert profile_for(name) is GEMMA


def test_an_unknown_name_is_refused_by_name():
    """Falling back to a default here would serve a whole meeting through the
    wrong template with nothing to show for it."""
    with pytest.raises(ValueError) as caught:
        profile_for("llama")
    assert "llama" in str(caught.value)
    assert "gemma" in str(caught.value)


# ---------------------------------------------------------------------------
# Catching a mismatch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("served,expected", [
    ("google/gemma-4-12b-it", "gemma"),
    ("Qwen/Qwen3.5-9B", "qwen"),
    ("meta-llama/Llama-3-8B", None),
])
def test_a_served_model_is_traced_to_its_family(served, expected):
    found = profile_for_model(served)
    assert (found.name if found else None) == expected


# ---------------------------------------------------------------------------
# What a profile may hold
# ---------------------------------------------------------------------------
def test_every_profile_is_json_serialisable():
    """It goes straight into the request body."""
    for profile in PROFILES.values():
        json.dumps(profile.extra_body)


def test_no_profile_carries_prompt_wording():
    """Deliberately shared. A profile that starts holding prompt text is one
    that will drift away from the model it is meant to be compared against."""
    for profile in PROFILES.values():
        for field_name in vars(profile):
            value = getattr(profile, field_name)
            if isinstance(value, str):
                assert "translate" not in value.lower(), field_name


def test_a_custom_profile_needs_nothing_but_a_name_and_a_model():
    made = ModelProfile(name="test", model_id="x/y")
    assert made.messages("s", "u")[0]["role"] == "system"
