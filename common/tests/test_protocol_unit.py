"""Unit tests for the client/server wire contract.

Run with::

    .venv\\Scripts\\python.exe -m pytest common/tests -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.protocol import (
    CHANNELS,
    CHUNK_BYTES,
    CHUNK_DURATION_MS,
    CHUNK_SAMPLES,
    PROTOCOL_VERSION,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    ClientMessage,
    Hello,
    ProtocolError,
    ServerMessage,
    make_bye,
    make_error,
    make_final,
    make_partial,
    make_ready,
    make_vad,
    parse_message,
    validate_audio_chunk,
)


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------
def test_audio_contract_matches_design_md():
    assert (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH) == (16_000, 1, 2)
    assert CHUNK_DURATION_MS == 200
    assert CHUNK_SAMPLES == 3_200
    assert CHUNK_BYTES == 6_400


def test_both_sides_import_the_same_numbers():
    """The whole reason common/protocol.py exists."""
    from client import config as client_config
    from server import config as server_config

    assert client_config.TARGET_SAMPLE_RATE == server_config.SAMPLE_RATE
    assert client_config.TARGET_CHANNELS == server_config.CHANNELS
    assert client_config.TARGET_SAMPLE_WIDTH == server_config.SAMPLE_WIDTH
    assert client_config.CHUNK_BYTES == server_config.CHUNK_BYTES
    assert client_config.CHUNK_DURATION_MS == server_config.CHUNK_DURATION_MS


# ---------------------------------------------------------------------------
# Hello
# ---------------------------------------------------------------------------
def test_hello_round_trips_through_json():
    hello = Hello(session_id="abc123", client="windows-client")
    restored = Hello.from_dict(json.loads(hello.to_json()))
    assert restored == hello


def test_hello_defaults_to_the_server_audio_format():
    assert Hello(session_id="x").audio_mismatch() is None


def test_hello_json_is_tagged_with_its_type():
    assert json.loads(Hello(session_id="x").to_json())["type"] == "hello"


def test_hello_rejects_a_message_of_the_wrong_type():
    with pytest.raises(ProtocolError, match="expected a 'hello'"):
        Hello.from_dict({"type": "bye"})


@pytest.mark.parametrize("session_id", ["", None, 42])
def test_hello_requires_a_session_id(session_id):
    with pytest.raises(ProtocolError, match="session_id"):
        Hello.from_dict({"type": "hello", "session_id": session_id})


def test_hello_ignores_fields_it_does_not_know():
    """A newer client may send extra keys; that must not break the server."""
    hello = Hello.from_dict(
        {"type": "hello", "session_id": "x", "future_feature": True}
    )
    assert hello.session_id == "x"


@pytest.mark.parametrize(
    "field, value",
    [
        ("sample_rate", 48_000),
        ("channels", 2),
        ("sample_width", 4),
        ("chunk_ms", 500),
        ("protocol_version", PROTOCOL_VERSION + 1),
    ],
)
def test_audio_mismatch_names_the_offending_field(field, value):
    hello = Hello(session_id="x", **{field: value})
    mismatch = hello.audio_mismatch()
    assert mismatch is not None
    assert field in mismatch
    assert str(value) in mismatch


def test_audio_mismatch_reports_every_wrong_field_at_once():
    hello = Hello(session_id="x", sample_rate=44_100, channels=2)
    mismatch = hello.audio_mismatch()
    assert "sample_rate" in mismatch and "channels" in mismatch


# ---------------------------------------------------------------------------
# Server messages
# ---------------------------------------------------------------------------
def test_ready_carries_what_the_client_needs_to_verify():
    payload = json.loads(make_ready("abc"))
    assert payload["type"] == ServerMessage.READY.value
    assert payload["session_id"] == "abc"
    assert payload["chunk_bytes"] == CHUNK_BYTES
    assert payload["protocol_version"] == PROTOCOL_VERSION


def test_vad_message_rounds_the_timestamp_but_keeps_it_useful():
    payload = json.loads(make_vad("speech_start", 1234.5678))
    assert payload["type"] == ServerMessage.VAD.value
    assert payload["event"] == "speech_start"
    assert payload["at_ms"] == 1234.6


def test_partial_matches_the_design_md_shape():
    payload = json.loads(make_partial("Speaker_01", "vi", "hôm nay chúng ta họp về"))
    assert payload == {
        "type": "partial",
        "speaker_id": "Speaker_01",
        "lang_code": "vi",
        "transcript": "hôm nay chúng ta họp về",
    }


def test_final_matches_the_design_md_shape():
    payload = json.loads(
        make_final("Speaker_01", "vi", "Xin chào.", "こんにちは。")
    )
    assert payload["type"] == "final"
    assert payload["translation"] == "こんにちは。"


def test_transcripts_keep_their_original_characters_on_the_wire():
    """ensure_ascii would turn Vietnamese and Japanese into \\uXXXX noise."""
    raw = make_final("Speaker_01", "ja", "会議", "cuộc họp")
    assert "会議" in raw
    assert "cuộc họp" in raw


def test_error_defaults_to_fatal():
    payload = json.loads(make_error("nope"))
    assert payload == {"type": "error", "message": "nope", "fatal": True}


def test_bye_carries_a_reason():
    payload = json.loads(make_bye("client stopped"))
    assert payload["type"] == ClientMessage.BYE.value
    assert payload["reason"] == "client stopped"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_parse_message_accepts_every_defined_type():
    for raw in (
        Hello(session_id="x").to_json(),
        make_bye(),
        make_ready("x"),
        make_vad("speech_end", 0),
        make_partial("s", "vi", "t"),
        make_final("s", "vi", "t", "tr"),
        make_error("e"),
    ):
        assert "type" in parse_message(raw)


def test_parse_message_rejects_broken_json():
    with pytest.raises(ProtocolError, match="not valid JSON"):
        parse_message("{oops")


def test_parse_message_rejects_a_bare_json_value():
    with pytest.raises(ProtocolError, match="expected a JSON object"):
        parse_message('"hello"')


def test_parse_message_rejects_an_unknown_type():
    with pytest.raises(ProtocolError, match="unknown message type"):
        parse_message('{"type": "shutdown"}')


def test_parse_message_rejects_a_missing_type():
    with pytest.raises(ProtocolError, match="unknown message type"):
        parse_message('{"session_id": "x"}')


# ---------------------------------------------------------------------------
# Binary frames
# ---------------------------------------------------------------------------
def test_a_full_chunk_is_accepted():
    validate_audio_chunk(bytes(CHUNK_BYTES))


@pytest.mark.parametrize("size", [0, 1, CHUNK_BYTES - 1, CHUNK_BYTES + 1])
def test_any_other_size_is_rejected(size):
    with pytest.raises(ProtocolError, match="exactly 6400 bytes"):
        validate_audio_chunk(bytes(size))
