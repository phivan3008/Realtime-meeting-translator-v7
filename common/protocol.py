"""The wire contract between the Windows client and the GPU server.

The single source of truth for both sides, kept outside ``client/`` and
``server/`` because the audio format is the one thing that must never drift.
A client sending 48 kHz stereo to a server assuming 16 kHz mono does not
crash: Whisper transcribes garbage at a third of the speed, and no log shows
why. So the format is defined once and checked during the handshake.

Transport: one WebSocket per meeting. Text frames carry JSON both ways;
binary frames carry raw PCM client-to-server only - no header, no framing,
exactly ``CHUNK_BYTES`` of little-endian 16-bit mono, each chunk continuing
where the last stopped.

Handshake:

1. Client sends ``hello`` before any audio.
2. Server validates the format and replies ``ready``, or ``error`` and closes.
3. Client streams binary chunks until it sends ``bye`` or drops.
4. Server pushes ``vad`` / ``partial`` / ``final`` / ``translation`` at any
   time after ``ready``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

PROTOCOL_VERSION = 1

# --- Audio contract ---------------------------------------------------------
# Whisper, Silero and PyAnnote all want 16 kHz mono, so the client resamples
# once at the source rather than making every server stage do it.
SAMPLE_RATE = 16_000            # Hz
CHANNELS = 1                    # mono
SAMPLE_WIDTH = 2                # bytes, 16-bit signed little endian PCM

# DESIGN.md asks for 200-500 ms packets. 200 ms keeps the perceived latency
# low while still being a whole number of Silero VAD frames plus a remainder
# the server buffers.
CHUNK_DURATION_MS = 200
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION_MS // 1000     # 3200
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH                  # 6400


class ProtocolError(ValueError):
    """A message that does not match this contract."""


# ---------------------------------------------------------------------------
# Message kinds
# ---------------------------------------------------------------------------
class ClientMessage(str, Enum):
    HELLO = "hello"
    BYE = "bye"


class ServerMessage(str, Enum):
    READY = "ready"
    VAD = "vad"
    UTTERANCE = "utterance"
    PARTIAL = "partial"
    FINAL = "final"
    #: The translation of a sentence already sent as FINAL, matched by
    #: ``sentence_id``. Separate because it arrives later: a sentence is worth
    #: showing the moment it is transcribed, and waiting for an LLM to answer
    #: before saying anything holds up the whole connection.
    TRANSLATION = "translation"
    SPEAKERS = "speakers"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Client -> server
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Hello:
    """The first message on every connection, describing the audio to come."""

    session_id: str
    protocol_version: int = PROTOCOL_VERSION
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    sample_width: int = SAMPLE_WIDTH
    chunk_ms: int = CHUNK_DURATION_MS
    client: str = ""                     # free-form, for logs only

    def to_json(self) -> str:
        return json.dumps({"type": ClientMessage.HELLO.value, **self.__dict__})

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Hello":
        if payload.get("type") != ClientMessage.HELLO.value:
            raise ProtocolError(
                f"expected a {ClientMessage.HELLO.value!r} message, "
                f"got {payload.get('type')!r}"
            )
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ProtocolError("hello.session_id must be a non-empty string")

        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in payload.items() if k in known}
        try:
            return cls(**kwargs)
        except TypeError as exc:            # pragma: no cover - defensive
            raise ProtocolError(f"malformed hello: {exc}") from exc

    def audio_mismatch(self) -> Optional[str]:
        """Return a human-readable reason the audio format is unusable."""
        expected = {
            "protocol_version": PROTOCOL_VERSION,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "sample_width": SAMPLE_WIDTH,
            "chunk_ms": CHUNK_DURATION_MS,
        }
        wrong = [
            f"{name}={getattr(self, name)!r} (server wants {want!r})"
            for name, want in expected.items()
            if getattr(self, name) != want
        ]
        return "; ".join(wrong) if wrong else None


def make_bye(reason: str = "") -> str:
    return json.dumps({"type": ClientMessage.BYE.value, "reason": reason})


# ---------------------------------------------------------------------------
# Server -> client
# ---------------------------------------------------------------------------
def make_ready(session_id: str, **extra: Any) -> str:
    return json.dumps(
        {
            "type": ServerMessage.READY.value,
            "session_id": session_id,
            "protocol_version": PROTOCOL_VERSION,
            "sample_rate": SAMPLE_RATE,
            "chunk_bytes": CHUNK_BYTES,
            **extra,
        }
    )


def make_vad(event: str, at_ms: float, **extra: Any) -> str:
    """A speech boundary, as produced by the server VAD segmenter."""
    return json.dumps(
        {
            "type": ServerMessage.VAD.value,
            "event": event,
            "at_ms": round(float(at_ms), 1),
            **extra,
        }
    )


def make_utterance(index: int, start_ms: float, end_ms: float, reason: str,
                   continues_previous: bool = False, kept: bool = True,
                   label: str = "", speech_score: float = 0.0,
                   speaker_id: str = "", lang_code: str = "") -> str:
    """A sentence boundary decided by the Stream Buffer Manager.

    Sent before any transcript exists, so the UI can open a row for the
    sentence and the client can see why it was cut.

    ``kept`` is the Deep Noise Filter's verdict. A dropped sentence is still
    announced, with ``label`` naming what it sounded like: the indexes stay
    consecutive, and a filter that starts eating real speech is visible on
    the client instead of silently losing sentences.

    ``speaker_id`` is the diarization label, empty for a dropped sentence and
    ``Speaker_unknown`` when the sentence was too short to identify.

    ``lang_code`` is ``"vi"`` or ``"ja"``, and empty when the two were too
    close to call - in which case the ASR detects the language itself rather
    than being forced into the wrong one.
    """
    return json.dumps(
        {
            "type": ServerMessage.UTTERANCE.value,
            "index": index,
            "start_ms": round(float(start_ms), 1),
            "end_ms": round(float(end_ms), 1),
            "duration_ms": round(float(end_ms) - float(start_ms), 1),
            "reason": reason,
            "continues_previous": bool(continues_previous),
            "kept": bool(kept),
            "label": label,
            "speech_score": round(float(speech_score), 3),
            "speaker_id": speaker_id,
            "lang_code": lang_code,
        }
    )


def make_partial(speaker_id: str, lang_code: str, transcript: str) -> str:
    """DESIGN.md section 4: the greyed-out running prediction."""
    return json.dumps(
        {
            "type": ServerMessage.PARTIAL.value,
            "speaker_id": speaker_id,
            "lang_code": lang_code,
            "transcript": transcript,
        },
        ensure_ascii=False,
    )


def make_final(sentence_id: int, speaker_id: str, lang_code: str,
               transcript: str, speech_score: float = 0.0) -> str:
    """DESIGN.md section 4: a committed sentence, as soon as it exists.

    No translation here. The sentence is worth showing the moment Whisper
    commits it, and an LLM call takes long enough that waiting for one before
    saying anything holds up the connection - on one run a slow answer put
    every VAD event 12 s late. The translation follows as its own message,
    matched by ``sentence_id``.

    ``sentence_id`` counts sentences within a session and never repeats.
    Utterance indexes restart with each speech segment, so they cannot be
    used to match anything up.
    """
    return json.dumps(
        {
            "type": ServerMessage.FINAL.value,
            "sentence_id": sentence_id,
            "speaker_id": speaker_id,
            "lang_code": lang_code,
            "transcript": transcript,
            # What the noise filter thought of the audio this came from.
            # Whisper invents sentences out of near-silence and does it
            # confidently, so nothing in its own numbers separates one from
            # real speech - but the classifier scored the audio before Whisper
            # ever saw it. A sign-off that reached the screen came from an
            # utterance scored 0.03 while every real sentence in the same run
            # scored 0.66 or better. Carried here so that pairing can be
            # measured over real runs rather than guessed at.
            "speech_score": speech_score,
        },
        ensure_ascii=False,
    )


def make_translation(sentence_id: int, translation: str, reason: str = "",
                     raw: str = "") -> str:
    """The translation of a sentence sent earlier as ``final``.

    ``reason`` carries why there is no translation when there is none, and
    ``raw`` carries what the model actually said. A reason alone answers "was
    it refused" but not "should it have been": "the answer is far longer than
    the sentence" reads identically whether the model rambled or produced a
    good translation the limit was too tight for. Only the text tells them
    apart, and reaching for it in the server log cost this project two round
    trips.

    Both are empty on success, and a UI has no reason to show either.
    """
    return json.dumps(
        {
            "type": ServerMessage.TRANSLATION.value,
            "sentence_id": sentence_id,
            "translation": translation,
            "reason": reason,
            "raw": raw,
        },
        ensure_ascii=False,
    )


def make_speakers(labels: dict[int, str]) -> str:
    """Corrected speaker labels for sentences already sent.

    The live matcher answers each sentence from what it had heard by then, so
    its answer depends on the order the meeting happened in and can never be
    revised. Clustering the whole meeting again gives a better one, and this
    carries it back: ``{sentence_id: speaker_id}``, only for the rows that
    changed.
    """
    return json.dumps(
        {
            "type": ServerMessage.SPEAKERS.value,
            "labels": {str(key): value for key, value in labels.items()},
        },
        ensure_ascii=False,
    )


def make_error(message: str, fatal: bool = True) -> str:
    return json.dumps(
        {"type": ServerMessage.ERROR.value, "message": message, "fatal": fatal}
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_message(raw: str) -> dict[str, Any]:
    """Decode a JSON control frame, rejecting anything without a known type."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"expected a JSON object, got {type(payload).__name__}")
    kind = payload.get("type")
    known = {m.value for m in ClientMessage} | {m.value for m in ServerMessage}
    if kind not in known:
        raise ProtocolError(f"unknown message type {kind!r}")
    return payload


def validate_audio_chunk(data: bytes) -> None:
    """Reject a binary frame that is not exactly one chunk of PCM."""
    if len(data) != CHUNK_BYTES:
        raise ProtocolError(
            f"audio chunk must be exactly {CHUNK_BYTES} bytes "
            f"({CHUNK_DURATION_MS} ms at {SAMPLE_RATE} Hz mono 16-bit), "
            f"got {len(data)}"
        )
