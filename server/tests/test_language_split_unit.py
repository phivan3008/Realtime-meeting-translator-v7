"""Unit tests for splitting an utterance that holds two languages.

The language identifier is a stub that reads the audio's amplitude, so a
"language" here is a volume. That is enough to test the policy - when a probe
is taken, what is compared, and where the boundary lands - which has to be
right before a model is involved.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import SAMPLE_RATE, SAMPLE_WIDTH
from server.pipeline.language_split import LanguageSplitter

PROBE_MS = 600
VI, JA = 8_000, 500


class LidByVolume:
    """Loud is Vietnamese, quiet is Japanese, silence is undecided."""

    def __init__(self) -> None:
        self.calls = 0

    def identify(self, pcm: bytes):
        from server.pipeline.lid import LanguageDecision
        self.calls += 1
        level = float(np.abs(np.frombuffer(pcm, dtype="<i2")).mean())
        if level < 100:
            code = ""
        else:
            code = "vi" if level > 4_000 else "ja"
        return LanguageDecision(lang_code=code, confidence=0.9, margin=0.9,
                                reason="stub")


def tone(ms: float, amplitude: int) -> bytes:
    return np.full(int(ms * SAMPLE_RATE / 1000.0), amplitude,
                   dtype="<i2").tobytes()


def splitter(**kwargs) -> LanguageSplitter:
    defaults = dict(identifier=LidByVolume(), probe_ms=PROBE_MS, steps=3)
    defaults.update(kwargs)
    return LanguageSplitter(**defaults)


# ---------------------------------------------------------------------------
# When it splits at all
# ---------------------------------------------------------------------------
def test_one_language_throughout_is_not_split():
    assert splitter().find(tone(4_000, VI)) is None


def test_two_languages_are_found():
    assert splitter().find(tone(2_000, VI) + tone(2_000, JA)) is not None


def test_an_utterance_too_short_to_probe_twice_is_left_alone():
    watcher = splitter()
    assert watcher.find(tone(900, VI)) is None
    assert watcher.identifier.calls == 0, "it probed audio it could not use"


def test_silence_at_an_end_is_not_a_second_language():
    """An undecided probe is not evidence of anything."""
    assert splitter().find(tone(2_000, VI) + tone(2_000, 0)) is None


def test_nothing_at_all_is_not_an_error():
    assert splitter().find(b"") is None


# ---------------------------------------------------------------------------
# Where the boundary lands
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("first_ms", [1_500, 2_000, 3_000, 4_500])
def test_the_boundary_is_found_near_where_the_language_changes(first_ms):
    found = splitter(steps=5).find(tone(first_ms, VI) + tone(2_000, JA))
    assert found is not None
    assert abs(found.at_ms - first_ms) < 700, f"landed at {found.at_ms:.0f} ms"


def test_the_boundary_names_both_languages():
    found = splitter().find(tone(2_000, VI) + tone(2_000, JA))
    assert (found.before, found.after) == ("vi", "ja")


def test_it_works_the_other_way_round_too():
    found = splitter().find(tone(2_000, JA) + tone(2_000, VI))
    assert (found.before, found.after) == ("ja", "vi")


def test_more_steps_land_closer():
    audio = tone(4_000, VI) + tone(2_000, JA)
    coarse = splitter(steps=1).find(audio)
    fine = splitter(steps=5).find(audio)
    assert abs(fine.at_ms - 4_000) <= abs(coarse.at_ms - 4_000)


def test_the_boundary_is_inside_the_utterance():
    audio = tone(2_000, VI) + tone(2_000, JA)
    found = splitter().find(audio)
    assert 0 < found.at_ms < 4_000


# ---------------------------------------------------------------------------
# What it costs
# ---------------------------------------------------------------------------
def test_a_single_language_costs_two_probes():
    """The common case, and it must stay cheap: this runs on the thread that
    reads the socket."""
    watcher = splitter()
    watcher.find(tone(4_000, VI))
    assert watcher.identifier.calls == 2


def test_a_split_costs_the_two_probes_plus_the_search():
    watcher = splitter(steps=3)
    watcher.find(tone(2_000, VI) + tone(2_000, JA))
    assert watcher.identifier.calls <= 5


def test_the_counts_are_kept_for_the_summary():
    watcher = splitter()
    watcher.find(tone(4_000, VI))
    watcher.find(tone(2_000, VI) + tone(2_000, JA))
    assert watcher.stats.checked == 2
    assert watcher.stats.split == 1
    assert watcher.stats.probes >= 4


def test_reset_forgets_the_meeting():
    watcher = splitter()
    watcher.find(tone(4_000, VI))
    watcher.reset()
    assert watcher.stats.checked == 0


# ---------------------------------------------------------------------------
# Refusing nonsense
# ---------------------------------------------------------------------------
def test_a_probe_of_no_length_is_refused():
    with pytest.raises(ValueError):
        splitter(probe_ms=0)


def test_a_negative_number_of_steps_is_refused():
    with pytest.raises(ValueError):
        splitter(steps=-1)


def test_zero_steps_still_answers():
    """Without the search it is a coarse boundary, not a broken one."""
    found = splitter(steps=0).find(tone(2_000, VI) + tone(2_000, JA))
    assert found is not None


def test_the_probe_is_measured_in_whole_samples():
    assert splitter().probe_bytes == PROBE_MS * SAMPLE_RATE // 1000 * SAMPLE_WIDTH


# ---------------------------------------------------------------------------
# No slivers
# ---------------------------------------------------------------------------
def duration_ms(pcm: bytes) -> float:
    return len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH) * 1000.0


def test_a_boundary_is_never_so_early_that_the_head_is_a_sliver():
    """Seen on a real meeting: two finals 31 ms apart, the second of them a
    Whisper invention over near-silence. A fragment too short to transcribe
    is a fragment Whisper fills in.

    A probe that straddles the change reads as the second language, which
    walks the search's upper bound down past the first language entirely.
    """
    found = splitter(steps=5).find(tone(600, VI) + tone(3_000, JA))
    assert found is None or found.at_ms >= PROBE_MS,         f"head is {found.at_ms:.0f} ms"


def test_a_boundary_is_never_so_late_that_the_tail_is_a_sliver():
    audio = tone(3_000, VI) + tone(600, JA)
    found = splitter(steps=5).find(audio)
    assert found is None or duration_ms(audio) - found.at_ms >= PROBE_MS,         f"tail is {duration_ms(audio) - found.at_ms:.0f} ms"


def test_an_utterance_that_cannot_hold_two_probes_and_a_gap_is_left_whole():
    """Both halves have to be long enough for the LID to have meant it."""
    watcher = splitter()
    assert watcher.find(tone(1_100, VI) + tone(100, JA)) is None


# ---------------------------------------------------------------------------
# Saying why it declined
# ---------------------------------------------------------------------------
def test_it_says_when_both_ends_are_one_language():
    """The case it was built for went unsplit for a whole meeting with
    nothing in the log about it."""
    watcher = splitter()
    watcher.find(tone(4_000, VI))
    assert watcher.stats.one_language == 1
    assert "vi" in watcher.stats.last


def test_it_says_when_an_end_was_undecided():
    """A 600 ms probe rarely earns LID_MIN_MARGIN; real margins run near
    0.13, and the LID answers with nothing at all."""
    watcher = splitter()
    watcher.find(tone(2_000, VI) + tone(2_000, 0))
    assert watcher.stats.undecided == 1
    assert "undecided" in watcher.stats.last


def test_it_says_when_the_audio_was_too_short():
    watcher = splitter()
    watcher.find(tone(900, VI))
    assert watcher.stats.too_short == 1


def test_a_split_says_where_it_cut():
    watcher = splitter()
    watcher.find(tone(2_000, VI) + tone(2_000, JA))
    assert "ms" in watcher.stats.last


def test_the_reasons_add_up_to_what_was_seen():
    watcher = splitter()
    watcher.find(tone(4_000, VI))
    watcher.find(tone(2_000, VI) + tone(2_000, JA))
    watcher.find(tone(2_000, VI) + tone(2_000, 0))
    assert watcher.stats.checked == 3
    assert (watcher.stats.split + watcher.stats.one_language
            + watcher.stats.undecided) == 3
