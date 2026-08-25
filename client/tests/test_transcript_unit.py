"""Unit tests for what the meeting window shows.

No Qt here: the model is pure Python precisely so it can be tested on a
machine with no display.

Run with::

    .venv\\Scripts\\python.exe -m pytest client/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.ui.transcript import Sentence, TranscriptModel


def final(sentence_id: int = 1, text: str = "xin chào",
          speaker: str = "Speaker_01", lang: str = "vi",
          speech_score: float = 0.85) -> dict:
    return {"type": "final", "sentence_id": sentence_id,
            "speaker_id": speaker, "lang_code": lang,
            "transcript": text, "speech_score": speech_score}


def translation(sentence_id: int = 1, text: str = "こんにちは",
                reason: str = "", raw: str = "") -> dict:
    return {"type": "translation", "sentence_id": sentence_id,
            "translation": text, "reason": reason, "raw": raw}


def partial(text: str = "xin ch", lang: str = "vi") -> dict:
    return {"type": "partial", "speaker_id": "", "lang_code": lang,
            "transcript": text}


def utterance(kept: bool = True, label: str = "") -> dict:
    return {"type": "utterance", "index": 0, "start_ms": 0.0, "end_ms": 100.0,
            "reason": "pause", "kept": kept, "label": label}


# ---------------------------------------------------------------------------
# Committed sentences
# ---------------------------------------------------------------------------
def test_a_final_becomes_a_row():
    model = TranscriptModel()
    model.apply(final(text="Xin chào mọi người"))
    assert len(model.sentences) == 1
    assert model.sentences[0].text == "Xin chào mọi người"
    assert model.sentences[0].speaker_id == "Speaker_01"


def test_a_new_sentence_clears_the_running_text():
    """The grey line predicted the sentence that just arrived; leaving it up
    would show the same words twice."""
    model = TranscriptModel()
    model.apply(partial("Xin chào mọi ng"))
    assert model.partial
    model.apply(final(text="Xin chào mọi người."))
    assert model.partial == ""


def test_the_running_text_is_replaced_not_appended():
    model = TranscriptModel()
    model.apply(partial("xin"))
    model.apply(partial("xin chào"))
    assert model.partial == "xin chào"


def test_sentences_keep_the_order_they_arrived():
    model = TranscriptModel()
    for index in range(1, 4):
        model.apply(final(sentence_id=index, text=f"câu {index}"))
    assert [s.text for s in model.sentences] == ["câu 1", "câu 2", "câu 3"]


# ---------------------------------------------------------------------------
# Translations arriving separately
# ---------------------------------------------------------------------------
def test_a_translation_lands_on_its_own_sentence():
    model = TranscriptModel()
    model.apply(final(sentence_id=7))
    model.apply(translation(sentence_id=7, text="こんにちは"))
    assert model.sentences[0].translation == "こんにちは"


def test_translations_are_matched_by_id_not_by_order():
    """They arrive when the model finishes, which need not be in order."""
    model = TranscriptModel()
    model.apply(final(sentence_id=1, text="first"))
    model.apply(final(sentence_id=2, text="second"))
    model.apply(translation(sentence_id=2, text="二番目"))
    model.apply(translation(sentence_id=1, text="一番目"))
    assert [s.translation for s in model.sentences] == ["一番目", "二番目"]


def test_a_translation_for_an_unknown_sentence_is_ignored():
    """Its row scrolled away, or it belongs to a session that ended."""
    model = TranscriptModel()
    model.apply(final(sentence_id=1))
    assert model.apply(translation(sentence_id=99)) is None
    assert model.sentences[0].translation == ""


def test_a_refusal_is_recorded_with_its_reason_and_the_model_answer():
    """An empty translation with nothing beside it looks like a bug in the
    client rather than an answer from the server."""
    model = TranscriptModel()
    model.apply(final(sentence_id=1))
    model.apply(translation(sentence_id=1, text="",
                            reason="the answer is not written in ja",
                            raw="OK. OK. I will view it."))
    sentence = model.sentences[0]
    assert sentence.translated is False
    assert sentence.answered is True
    assert sentence.reason == "the answer is not written in ja"
    assert sentence.raw == "OK. OK. I will view it."


def test_a_sentence_with_no_word_yet_is_neither_translated_nor_answered():
    model = TranscriptModel()
    model.apply(final(sentence_id=1))
    assert model.sentences[0].answered is False
    assert model.waiting == 1


def test_waiting_falls_to_zero_once_every_sentence_is_answered():
    model = TranscriptModel()
    model.apply(final(sentence_id=1))
    model.apply(final(sentence_id=2))
    model.apply(translation(sentence_id=1))
    model.apply(translation(sentence_id=2, text="", reason="too late"))
    assert model.waiting == 0
    assert model.translated == 1


# ---------------------------------------------------------------------------
# Dropped sentences
# ---------------------------------------------------------------------------
def test_a_dropped_utterance_is_counted_but_shows_no_text():
    """There is no transcript for it - the filter stopped it before the ASR.
    The count is how a filter that starts eating speech becomes visible."""
    model = TranscriptModel()
    model.apply(utterance(kept=False, label="Breathing"))
    assert model.dropped == 1
    assert model.sentences == []


def test_a_kept_utterance_is_not_counted_as_dropped():
    model = TranscriptModel()
    model.apply(utterance(kept=True))
    assert model.dropped == 0


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
def test_old_rows_scroll_out_of_a_long_meeting():
    model = TranscriptModel(max_rows=3)
    for index in range(1, 6):
        model.apply(final(sentence_id=index, text=f"câu {index}"))
    assert [s.text for s in model.sentences] == ["câu 3", "câu 4", "câu 5"]


def test_a_translation_for_a_scrolled_row_does_not_resurrect_it():
    model = TranscriptModel(max_rows=2)
    model.apply(final(sentence_id=1))
    model.apply(final(sentence_id=2))
    model.apply(final(sentence_id=3))
    assert model.apply(translation(sentence_id=1)) is None
    assert len(model.sentences) == 2


def test_the_speakers_are_listed_in_the_order_they_first_spoke():
    model = TranscriptModel()
    model.apply(final(sentence_id=1, speaker="Speaker_02"))
    model.apply(final(sentence_id=2, speaker="Speaker_01"))
    model.apply(final(sentence_id=3, speaker="Speaker_02"))
    assert model.speakers() == ["Speaker_02", "Speaker_01"]


def test_an_unknown_speaker_is_not_listed_as_a_person():
    model = TranscriptModel()
    model.apply(final(sentence_id=1, speaker=""))
    assert model.speakers() == []


def test_clearing_starts_a_fresh_meeting():
    model = TranscriptModel()
    model.apply(final(sentence_id=1))
    model.apply(partial("nửa câu"))
    model.apply(utterance(kept=False))
    model.clear()
    assert model.sentences == []
    assert model.partial == ""
    assert model.dropped == 0
    assert model.apply(translation(sentence_id=1)) is None


def test_a_message_type_it_does_not_know_is_ignored():
    """A newer server may send something this client has not learned yet."""
    model = TranscriptModel()
    assert model.apply({"type": "something_new", "field": 1}) is None
    assert model.sentences == []


def test_apply_returns_the_row_that_changed():
    """The window redraws one row rather than the whole meeting."""
    model = TranscriptModel()
    added = model.apply(final(sentence_id=4))
    assert isinstance(added, Sentence) and added.sentence_id == 4
    updated = model.apply(translation(sentence_id=4))
    assert updated is added


# ---------------------------------------------------------------------------
# Second thoughts about who said what
# ---------------------------------------------------------------------------
def test_a_corrected_speaker_reaches_the_row_already_on_screen():
    model = TranscriptModel()
    model.apply({"type": "final", "sentence_id": 4, "speaker_id": "Speaker_01",
                 "lang_code": "vi", "transcript": "Xin chào."})
    model.apply({"type": "speakers", "labels": {"4": "Speaker_03"}})
    assert model.sentences[0].speaker_id == "Speaker_03"


def test_a_correction_for_a_row_that_scrolled_away_is_ignored():
    model = TranscriptModel(max_rows=1)
    model.apply({"type": "final", "sentence_id": 1, "speaker_id": "Speaker_01",
                 "lang_code": "vi", "transcript": "Câu một."})
    model.apply({"type": "final", "sentence_id": 2, "speaker_id": "Speaker_01",
                 "lang_code": "vi", "transcript": "Câu hai."})
    model.apply({"type": "speakers", "labels": {"1": "Speaker_09"}})
    assert [s.speaker_id for s in model.sentences] == ["Speaker_01"]


def test_corrections_change_the_speaker_count():
    """Two people merged into one is what this exists to undo."""
    model = TranscriptModel()
    for index in (1, 2, 3):
        model.apply({"type": "final", "sentence_id": index,
                     "speaker_id": "Speaker_01", "lang_code": "vi",
                     "transcript": f"Câu {index}."})
    assert model.speakers() == ["Speaker_01"]
    model.apply({"type": "speakers", "labels": {"2": "Speaker_02"}})
    assert model.speakers() == ["Speaker_01", "Speaker_02"]
