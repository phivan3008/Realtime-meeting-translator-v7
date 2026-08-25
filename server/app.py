"""FastAPI application: the WebSocket front door of the GPU server.

Run on the pod::

    python3.11 -m uvicorn server.app:app --host 0.0.0.0 --port 8000

The heavy lifting lives elsewhere - this module only owns the socket, the
model's lifetime and the single-session rule.

**One streaming session at a time.**  Silero is recurrent: its hidden state
belongs to one audio stream.  Sharing a single model instance across two
concurrent meetings would mix their states and quietly degrade both, so a
second connection is refused rather than served badly.  Loading a model per
connection instead would cost ~1.4 s on every reconnect, which a flaky
network makes routine.  One meeting per pod is the current scope; when that
changes, the fix is a pool of pre-loaded models, not a shared one.
"""

from __future__ import annotations

import contextlib
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from common.protocol import (
    CHUNK_BYTES,
    CHUNK_DURATION_MS,
    PROTOCOL_VERSION,
    SAMPLE_RATE,
    make_error,
)
from server.net.session import Response, ServerSession
from server.pipeline.asr import AsrError, Transcriber
from server.pipeline.diarization import DiarizationError, SpeakerIdentifier
from server.pipeline.lid import LanguageIdError, LanguageIdentifier
from server.pipeline.noise import AstClassifier, NoiseFilter, NoiseFilterError
from server.pipeline.overlap import OverlapError, OverlapResolver
from server.pipeline.translate import TranslationError, Translator
from server.pipeline.vad import SileroVAD, VADSegmenter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("server.app")


class AppState:
    """Process-wide singletons: the VAD model and the active session."""

    def __init__(self) -> None:
        self.vad: Optional[SileroVAD] = None
        self.noise_filter: Optional[NoiseFilter] = None
        self.noise_error: str = ""
        self.overlap_resolver: Optional[OverlapResolver] = None
        self.speaker_identifier: Optional[SpeakerIdentifier] = None
        self.speaker_error: str = ""
        self.language_identifier: Optional[LanguageIdentifier] = None
        self.language_error: str = ""
        self.transcriber: Optional[Transcriber] = None
        self.asr_error: str = ""
        self.translator: Optional[Translator] = None
        self.translate_error: str = ""
        self.active_session_id: Optional[str] = None

    def load_models(self) -> None:
        if self.vad is None:
            log.info("Loading Silero VAD ...")
            self.vad = SileroVAD()
            log.info("Silero VAD ready")
        if self.noise_filter is None and not self.noise_error:
            if os.environ.get("DISABLE_NOISE_FILTER"):
                self.noise_error = "disabled by DISABLE_NOISE_FILTER"
                log.warning("Deep Noise Filter disabled by environment")
                return
            try:
                log.info("Loading the audio classifier ...")
                self.noise_filter = NoiseFilter(classifier=AstClassifier())
                log.info("Audio classifier ready")
            except NoiseFilterError as exc:
                # Serving without the filter is worse than serving with it,
                # but far better than refusing the meeting. /health says which
                # mode this process is in so nobody has to guess.
                self.noise_error = str(exc)
                log.error("Deep Noise Filter unavailable: %s", exc)
        if self.overlap_resolver is None:
            try:
                self.overlap_resolver = OverlapResolver()
                log.info("Overlap resolver ready")
            except OverlapError as exc:
                log.error("Overlap resolver unavailable: %s", exc)
        if self.speaker_identifier is None and not self.speaker_error:
            try:
                log.info("Loading the speaker embedding model ...")
                self.speaker_identifier = SpeakerIdentifier()
                log.info("Speaker embedding ready")
            except DiarizationError as exc:
                # Without it every sentence is unattributed, which is worse
                # than the alternative but far better than no meeting at all.
                self.speaker_error = str(exc)
                log.error("Speaker identification unavailable: %s", exc)
        if self.language_identifier is None and not self.language_error:
            try:
                log.info("Loading the language ID model ...")
                self.language_identifier = LanguageIdentifier()
                log.info("Language ID ready")
            except LanguageIdError as exc:
                # The ASR can detect the language itself, just more slowly.
                self.language_error = str(exc)
                log.error("Language ID unavailable: %s", exc)
        if self.transcriber is None and not self.asr_error:
            try:
                log.info("Loading Whisper ...")
                self.transcriber = Transcriber()
                log.info("Whisper ready")
            except AsrError as exc:
                # Nothing downstream works without this one. The server still
                # starts, so /health can say why instead of the pod going
                # silent, but a meeting served in this state carries no text.
                self.asr_error = str(exc)
                log.error("ASR unavailable: %s", exc)
        if self.translator is None and not self.translate_error:
            try:
                log.info("Connecting to the translation server ...")
                self.translator = Translator()
                log.info("Translation ready")
            except TranslationError as exc:
                # vLLM runs as its own process, so it may simply not be up
                # yet. The meeting still gets transcripts; /health says why
                # there are no translations.
                self.translate_error = str(exc)
                log.error("Translation unavailable: %s", exc)

    def make_segmenter(self) -> VADSegmenter:
        if self.vad is None:                    # pragma: no cover - startup order
            self.load_models()
        return VADSegmenter(vad=self.vad)


state = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pay the model load once, at boot, so the first meeting does not.
    state.load_models()
    yield


app = FastAPI(
    title="Realtime VI-JA Meeting Translator",
    version="0.3.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Cheap reachability probe - the client real test calls this first."""
    return {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "sample_rate": SAMPLE_RATE,
        "chunk_bytes": CHUNK_BYTES,
        "chunk_ms": CHUNK_DURATION_MS,
        "vad_loaded": state.vad is not None,
        "noise_filter_loaded": state.noise_filter is not None,
        "overlap_resolver_loaded": state.overlap_resolver is not None,
        "speaker_model_loaded": state.speaker_identifier is not None,
        "speaker_model_error": state.speaker_error,
        "language_model_loaded": state.language_identifier is not None,
        "language_model_error": state.language_error,
        "asr_loaded": state.transcriber is not None,
        "asr_error": state.asr_error,
        "translation_loaded": state.translator is not None,
        "translation_error": state.translate_error,
        "noise_filter_error": state.noise_error,
        "session_active": state.active_session_id is not None,
    }


async def _send(socket: WebSocket, response: Response) -> None:
    for message in response.messages:
        await socket.send_text(message)


@app.websocket("/ws/stream")
async def stream(socket: WebSocket) -> None:
    await socket.accept()

    if state.active_session_id is not None:
        await socket.send_text(
            make_error(
                "another session is already streaming "
                f"({state.active_session_id}); this server handles one meeting "
                "at a time"
            )
        )
        await socket.close(code=1013)           # try again later
        return

    session = ServerSession(segmenter_factory=state.make_segmenter,
                            noise_filter=state.noise_filter,
                            overlap_resolver=state.overlap_resolver,
                            speaker_identifier=state.speaker_identifier,
                            language_identifier=state.language_identifier,
                            transcriber=state.transcriber,
                            translator=state.translator)
    claimed = False
    try:
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if (text := message.get("text")) is not None:
                response = session.handle_text(text)
            elif (data := message.get("bytes")) is not None:
                response = session.handle_binary(data)
            else:                               # pragma: no cover - empty frame
                continue

            if not claimed and session.session_id:
                state.active_session_id = session.session_id
                claimed = True

            await _send(socket, response)
            if response.close:
                await socket.close(code=1000 if "bye" in response.close_reason
                                   else 1008)
                break

    except WebSocketDisconnect:
        log.info("Session %s disconnected", session.session_id or "?")
    finally:
        # A dropped connection mid-sentence still has to close the segment,
        # or the buffer manager waits forever for an end that never comes.
        final = session.finish()
        if final.messages:
            try:
                await _send(socket, final)
            except Exception as exc:
                # Not suppressed silently: these are the last sentence of the
                # meeting and its translation, and a socket already closed is
                # exactly how one of them went missing once.
                log.warning(
                    "Session %s: %d closing message(s) could not be sent "
                    "(%s: %s)", session.session_id or "?",
                    len(final.messages), type(exc).__name__, exc)
        if claimed and state.active_session_id == session.session_id:
            state.active_session_id = None
        log.info(
            "Session %s finished: %d chunks, %.1f s audio, %d segments, "
            "%d utterances (%d dropped as noise, %d shaped, %d identified, "
            "%d with a language), %d transcripts, %d translations, "
            "%d partials, "
            "%d events, %d protocol errors, %d pipeline errors; "
            "slowest sentence %.1f s, stages %s; "
            "%d translations dropped, worst translation lag %.1f s",
            session.session_id or "?",
            session.stats.chunks,
            session.stats.audio_seconds,
            session.stats.speech_segments,
            session.stats.utterances,
            session.stats.utterances_dropped,
            session.stats.utterances_shaped,
            session.stats.utterances_identified,
            session.stats.utterances_with_language,
            session.stats.transcripts,
            session.stats.translations,
            session.stats.partials,
            session.stats.events_sent,
            session.stats.protocol_errors,
            session.stats.pipeline_errors,
            session.stats.slowest_utterance_seconds,
            {stage: round(value, 1)
             for stage, value in sorted(session.stats.stage_seconds.items(),
                                        key=lambda kv: -kv[1])},
            session.stats.translations_dropped,
            session.stats.worst_translation_lag,
        )
