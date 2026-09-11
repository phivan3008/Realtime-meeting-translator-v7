"""Unit tests for cutting an utterance that holds two languages.

The LID is stubbed, so these say nothing about whether VoxLingua107 can tell
Vietnamese from Japanese - that is
``server/tests_real/test_real_lid.py`` on the pod. What they pin is the
policy: when to cut, where, and above all when to refuse, because the failure
mode of this stage is not a missed cut. It is a sliver of audio too short to
transcribe, which Whisper does not return empty - it fills it in.

Run with::

    .venv\\Scripts\\python.exe -m pytest server/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.config import (  # noqa: E402
    LANGUAGE_SPLIT_MIN_PART_MS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from server.pipeline.buffer import bytes_to_ms, ms_to_bytes  # noqa: E402
from server.pipeline.language_split import find_split  # noqa: E402
from server.pipeline.lid import LanguageDecision  # noqa: E402


def decided(lang: str) -> LanguageDecision:
    return LanguageDecision(lang, 0.9, 0.5, "clear")


UNDECIDED = LanguageDecision("", 0.4, 0.05, "too close to call")


class TimedProber:
    """Says one language before a moment in the audio, another after it.

    Answers by *where* the window it is given sits, which it recovers from
    the audio itself: the fixture below writes a different amplitude either
    side of the boundary. That is how a stub can behave like a model whose
    answer depends on which span it was handed.
    """

    def __init__(self, before: str, after: str, quiet_is_before: bool = True):
        self.before, self.after = before, after
        self.quiet_is_before = quiet_is_before
        self.calls: list[int] = []

    def identify(self, pcm: bytes) -> LanguageDecision:
        self.calls.append(len(pcm))
        samples = np.frombuffer(pcm, dtype="<i2")
        if samples.size == 0:
            return UNDECIDED
        loud = float(np.mean(np.abs(samples.astype(np.float32))))
        first = loud < 6_000 if self.quiet_is_before else loud >= 6_000
        return decided(self.before if first else self.after)


class FixedProber:
    """Always says the same thing, however much audio it is handed."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def identify(self, pcm: bytes) -> LanguageDecision:
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return answer


def audio(seconds: float, amplitude: int = 3_000) -> bytes:
    return np.full(int(seconds * SAMPLE_RATE), amplitude, dtype="<i2").tobytes()


def two_turns(first_seconds: float, second_seconds: float,
              gap_ms: float = 100.0) -> bytes:
    """Quiet speech, a pause, then loud speech - a boundary a stub can see."""
    return (audio(first_seconds, 3_000)
            + audio(gap_ms / 1000.0, 5)
            + audio(second_seconds, 12_000))


class TestWhenNotToCut:
    def test_one_language_is_left_whole(self):
        assert find_split(audio(6.0), FixedProber(decided("vi"))) is None

    def test_an_undecided_head_is_not_evidence_of_two(self):
        """The LID saying it cannot tell is not the LID saying "both"."""
        prober = FixedProber(UNDECIDED, decided("ja"))
        assert find_split(audio(6.0), prober) is None

    def test_an_undecided_tail_is_not_either(self):
        prober = FixedProber(decided("vi"), UNDECIDED)
        assert find_split(audio(6.0), prober) is None

    def test_an_utterance_too_short_to_hold_two_turns_is_left_whole(self):
        """Below two probe windows the two probes would read the same audio,
        which compares a span with itself."""
        prober = FixedProber(decided("vi"), decided("ja"))
        assert find_split(audio(1.5), prober) is None
        assert prober.calls == 0

    def test_the_ordinary_case_costs_two_probes(self):
        prober = FixedProber(decided("vi"))
        find_split(audio(6.0), prober)
        assert prober.calls == 2


class TestWhereTheCutLands:
    def test_a_boundary_is_found_between_the_two_turns(self):
        pcm = two_turns(3.0, 3.0)
        split = find_split(pcm, TimedProber("vi", "ja"))
        assert split is not None
        assert split.first == "vi" and split.second == "ja"
        # Within a probe window of the real change at 3.0 s.
        assert 2.0 <= split.at_ms / 1000.0 <= 4.0

    def test_the_cut_lands_on_a_whole_sample(self):
        split = find_split(two_turns(3.0, 3.0), TimedProber("vi", "ja"))
        assert split.at % SAMPLE_WIDTH == 0

    def test_the_cut_is_snapped_to_the_quiet_between_the_turns(self):
        """Whisper turns half a word into a different word."""
        pcm = two_turns(3.0, 3.0, gap_ms=200.0)
        split = find_split(pcm, TimedProber("vi", "ja"))
        head = np.frombuffer(pcm[:split.at], dtype="<i2")
        assert float(np.mean(np.abs(head[-160:].astype(np.float32)))) < 100

    def test_it_costs_a_handful_of_probes_when_it_does_cut(self):
        prober = TimedProber("vi", "ja")
        split = find_split(two_turns(3.0, 3.0), prober)
        assert split.probes == len(prober.calls)
        assert split.probes <= 6

    def test_the_two_halves_together_are_the_whole_utterance(self):
        pcm = two_turns(3.0, 3.0)
        split = find_split(pcm, TimedProber("vi", "ja"))
        assert pcm[:split.at] + pcm[split.at:] == pcm


class TestTheSliverGuard:
    """The failure this stage has already produced once.

    A probe straddling the change reads as the second language, so the search
    walked past the language it started in and reported a boundary a few
    hundred milliseconds from the start; the quiet-frame snap took another
    500 ms off it. The second half came back as a subscribe line 31 ms after
    the first half was committed.
    """

    def test_a_cut_that_would_leave_a_sliver_is_refused(self):
        # The change sits 300 ms in - inside the floor on the left.
        pcm = two_turns(0.3, 5.0, gap_ms=60.0)
        assert find_split(pcm, TimedProber("vi", "ja")) is None

    def test_a_cut_near_the_end_is_refused_too(self):
        pcm = two_turns(5.0, 0.3, gap_ms=60.0)
        assert find_split(pcm, TimedProber("vi", "ja")) is None

    def test_every_half_it_does_return_clears_the_floor(self):
        for first in (2.0, 3.0, 4.0, 5.0):
            pcm = two_turns(first, 7.0 - first)
            split = find_split(pcm, TimedProber("vi", "ja"))
            if split is None:
                continue
            assert bytes_to_ms(split.at) >= LANGUAGE_SPLIT_MIN_PART_MS
            assert bytes_to_ms(len(pcm) - split.at) >= LANGUAGE_SPLIT_MIN_PART_MS


class TestReview:
    """The check that was missing, and what it cost to leave out.

    Over one real meeting the screening probes disagreed and the two halves
    then came back in the same language 26 times out of 41 - the probes had
    been wrong about audio the whole halves agree on, and every one of those
    cuts manufactured a fragment for nothing.
    """

    def test_a_cut_whose_halves_agree_is_refused(self):
        prober = FixedProber(decided("vi"), decided("ja"), decided("ja"),
                             decided("vi"), decided("vi"))
        assert find_split(audio(8.0), prober) is None

    def test_a_cut_whose_halves_are_not_confident_is_refused(self):
        prober = FixedProber(decided("vi"), decided("ja"), decided("ja"),
                             UNDECIDED, UNDECIDED)
        assert find_split(audio(8.0), prober) is None

    def test_the_languages_reported_are_the_halves_own(self):
        """Not the screening probes'. The halves are what gets decoded."""
        split = find_split(two_turns(3.0, 3.0), TimedProber("vi", "ja"))
        assert (split.first, split.second) == ("vi", "ja")

    def test_a_low_margin_is_not_enough_to_cut_on(self):
        """`known` is the bar for forcing a language, which a person can see
        is wrong. Cutting is not reversible."""
        weak = LanguageDecision("vi", 0.6, 0.31, "close")
        other = LanguageDecision("ja", 0.6, 0.31, "close")
        assert find_split(audio(6.0), FixedProber(weak, other)) is None


class TestOddAnswers:
    def test_a_third_language_stops_the_search_rather_than_guessing(self):
        prober = FixedProber(decided("vi"), decided("ja"), decided("ko"))
        split = find_split(audio(6.0), prober)
        # Two screening probes, one search probe that cannot be placed, then
        # the two review probes - and the review settles it.
        assert prober.calls == 5
        assert split is None

    def test_an_undecided_probe_mid_search_stops_it(self):
        prober = FixedProber(decided("vi"), decided("ja"), UNDECIDED)
        find_split(audio(6.0), prober)
        assert prober.calls == 5

    def test_a_custom_floor_is_honoured(self):
        pcm = two_turns(3.0, 3.0)
        assert find_split(pcm, TimedProber("vi", "ja"),
                          min_part_ms=4_000.0) is None
