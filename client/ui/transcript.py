"""What the meeting window shows, without any Qt.

The server sends four kinds of message that end up on screen, and they do not
arrive in the order they are displayed:

``partial``     running text, replaced every 600 ms
``final``       a committed sentence, with a ``sentence_id``
``translation`` its translation, arriving separately and matched by that id
``utterance``   a sentence boundary, including ones the noise filter dropped
``speakers``    corrected labels for sentences already on screen

A translation can arrive before the reader has finished the sentence, after
several later sentences, or never - the server always answers, but the answer
is sometimes "no translation, and here is why". So the rows are keyed by
``sentence_id`` and updated in place rather than appended in arrival order.

Keeping this separate from the widget is what lets it be tested on a machine
with no display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

#: Rows kept in the window. A long meeting is 90+ sentences; older ones scroll
#: out of reach and only cost memory.
MAX_ROWS = 500


@dataclass
class Sentence:
    """One committed sentence and whatever is known about it so far."""

    sentence_id: int
    speaker_id: str
    lang_code: str
    text: str
    speech_score: float = 0.0
    translation: str = ""
    #: Why there is no translation, when there is none.
    reason: str = ""
    #: What the model said when it was refused, for reading in the log pane.
    raw: str = ""

    @property
    def answered(self) -> bool:
        """Has the server said anything about the translation yet?"""
        return bool(self.translation or self.reason)

    @property
    def translated(self) -> bool:
        return bool(self.translation.strip())


@dataclass
class TranscriptModel:
    """The state of one meeting, driven entirely by server messages."""

    sentences: list[Sentence] = field(default_factory=list)
    #: The greyed-out running text, or empty when nobody is mid-sentence.
    partial: str = ""
    partial_lang: str = ""
    #: Sentences the noise filter dropped, so the count is visible even
    #: though the text never existed.
    dropped: int = 0
    max_rows: int = MAX_ROWS

    def __post_init__(self) -> None:
        self._by_id: dict[int, Sentence] = {}

    # -- incoming -----------------------------------------------------------
    def apply(self, message: dict) -> Optional[Sentence]:
        """Absorb one server message. Returns the row that changed, if any."""
        kind = message.get("type")
        if kind == "partial":
            self.partial = message.get("transcript", "")
            self.partial_lang = message.get("lang_code", "")
            return None
        if kind == "final":
            return self._add_sentence(message)
        if kind == "translation":
            return self._attach_translation(message)
        if kind == "speakers":
            self._relabel(message.get("labels", {}))
            return None
        if kind == "utterance" and not message.get("kept", True):
            self.dropped += 1
        return None

    def _relabel(self, labels: dict) -> None:
        """Second thoughts from the server about who said what.

        The live matcher answers each sentence from what it had heard by then;
        clustering the whole meeting gives a better answer and can revise one
        it already gave. Rows are keyed by id at both ends, so this is an
        update to what is already on screen.
        """
        for key, speaker_id in labels.items():
            sentence = self._by_id.get(int(key))
            if sentence is not None:
                sentence.speaker_id = speaker_id

    def _add_sentence(self, message: dict) -> Sentence:
        sentence = Sentence(
            sentence_id=message.get("sentence_id", 0),
            speaker_id=message.get("speaker_id", ""),
            lang_code=message.get("lang_code", ""),
            text=message.get("transcript", ""),
            speech_score=message.get("speech_score", 0.0),
        )
        self.sentences.append(sentence)
        self._by_id[sentence.sentence_id] = sentence
        # A committed sentence replaces the running text that predicted it.
        self.partial = ""
        self.partial_lang = ""
        self._trim()
        return sentence

    def _attach_translation(self, message: dict) -> Optional[Sentence]:
        sentence = self._by_id.get(message.get("sentence_id"))
        if sentence is None:
            # Its row has already scrolled out of the window, or the id is
            # from a session that ended. Nothing to update.
            return None
        sentence.translation = message.get("translation", "")
        sentence.reason = message.get("reason", "")
        sentence.raw = message.get("raw", "")
        return sentence

    def _trim(self) -> None:
        while len(self.sentences) > self.max_rows:
            self._by_id.pop(self.sentences.pop(0).sentence_id, None)

    # -- what the window asks it --------------------------------------------
    @property
    def waiting(self) -> int:
        """Sentences still waiting for word about their translation."""
        return sum(1 for s in self.sentences if not s.answered)

    @property
    def translated(self) -> int:
        return sum(1 for s in self.sentences if s.translated)

    def speakers(self) -> list[str]:
        """Everyone heard so far, in the order they first spoke."""
        seen = []
        for sentence in self.sentences:
            if sentence.speaker_id and sentence.speaker_id not in seen:
                seen.append(sentence.speaker_id)
        return seen

    def clear(self) -> None:
        self.sentences.clear()
        self._by_id.clear()
        self.partial = ""
        self.partial_lang = ""
        self.dropped = 0
