"""What each translation model needs done differently.

One place for the model-specific quirks, so the prompt, the guards and the
pipeline stay the same whichever checkpoint vLLM is serving. The client never
learns which one it is.

Two differences matter, and both are failures rather than degradations - the
request is rejected or the answer is empty, not merely worse.

**The system role.** Qwen's chat template takes one. Gemma's does not: its
template has only ``user`` and ``model`` turns, and vLLM answers a request
carrying a system message with a template error. The instructions have to be
folded into the first user turn instead.

**Thinking.** Qwen3 reasons aloud before answering unless the template is told
not to, and on the first run here it spent its entire token budget doing so
and returned no translation. That switch is ``chat_template_kwargs`` and it is
Qwen's. Sending it to a model whose template does not declare it is at best
ignored and at worst an error, so it goes in the profile rather than in the
request builder.

Everything else - the system prompt itself, the history, the refusal guards -
is deliberately shared. A profile that starts carrying prompt wording is a
profile that will drift away from the one it is meant to be compared against.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelProfile:
    """How to talk to one family of translation models."""

    name: str
    #: What to start vLLM with. The profile does not enforce it - the backend
    #: asks the server what it is serving and matches - but it is what the
    #: error messages suggest and what the README documents.
    model_id: str
    #: False folds the system prompt into the first user turn.
    supports_system_role: bool = True
    #: Extra top-level fields for the chat-completions request.
    extra_body: dict = field(default_factory=dict)
    #: How the two turns are joined when there is no system role.
    system_separator: str = "\n\n"

    def messages(self, system: str, user: str) -> list[dict]:
        """The chat messages for one translation."""
        if self.supports_system_role:
            return [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        return [{"role": "user",
                 "content": f"{system}{self.system_separator}{user}"}]


QWEN = ModelProfile(
    name="qwen",
    model_id="Qwen/Qwen3.5-9B",
    supports_system_role=True,
    # Qwen3 reasons before answering unless the template is told not to. The
    # first live run spent all 512 tokens on a <think> block and returned no
    # translation at all.
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

GEMMA = ModelProfile(
    name="gemma",
    model_id=os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-12b-it"),
    # Gemma's chat template has only user and model turns. A request carrying
    # a system message is a template error, not a warning.
    supports_system_role=False,
    # No thinking switch to set: the template does not declare one, and
    # sending an undeclared chat_template_kwargs is an error on some builds.
    extra_body={},
)

PROFILES = {profile.name: profile for profile in (QWEN, GEMMA)}
DEFAULT_PROFILE = QWEN.name


def profile_for(name: str = "") -> ModelProfile:
    """The profile by name, defaulting to the one this project shipped with."""
    wanted = (name or os.environ.get("TRANSLATE_PROFILE")
              or DEFAULT_PROFILE).strip().lower()
    if wanted not in PROFILES:
        raise ValueError(
            f"unknown translation profile {wanted!r}; "
            f"pick one of {', '.join(sorted(PROFILES))}"
        )
    return PROFILES[wanted]


def profile_for_model(model: str) -> Optional[ModelProfile]:
    """The profile whose family a served model name belongs to, if any.

    Used to catch the mismatch that has no symptom: vLLM answers a request for
    the wrong template happily enough, and the only sign is that the
    translations are worse than they should be.
    """
    lowered = model.lower()
    for profile in PROFILES.values():
        if profile.name in lowered:
            return profile
    return None
