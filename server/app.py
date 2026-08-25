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
from server.net.session import Response, ServerSession, Stages
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

    @property
    def stages(self) -> Stages:
        """The models a new session should run audio through."""
        return Stages(
            noise_filter=self.noise_filter,
            overlap_resolver=self.overlap_resolver,
            speaker_identifier=self.speaker_identifier,
            language_identifier=self.language_identifier,
            transcriber=self.transcriber,
            translator=self.translator,
        )

    def load_models(self) -> None:
        """Load every stage. A stage that fails is reported, not fatal.

        Serving without one is worse than serving with it and far better than
        refusing the meeting, so ``/health`` names whichever is missing and
        why rather than the pod going silent.
        """
        if self.vad is None:
            log.info("Loading Silero VAD ...")
            self.vad = SileroVAD()
            log.info("Silero VAD ready")

        if os.environ.get("DISABLE_NOISE_FILTER"):
            self.noise_error = "disabled by DISABLE_NOISE_FILTER"
            log.warning("Deep Noise Filter disabled by environment")
        else:
            self._load("noise_filter", "noise_error", "audio classifier",
                       lambda: NoiseFilter(classifier=AstClassifier()),
                       NoiseFilterError)

        self._load("overlap_resolver", "", "overlap resolver",
                   OverlapResolver, OverlapError)
        self._load("speaker_identifier", "speaker_error",
                   "speaker embedding model", SpeakerIdentifier,
                   DiarizationError)
        self._load("language_identifier", "language_error",
                   "language ID model", LanguageIdentifier, LanguageIdError)
        self._load("transcriber", "asr_error", "Whisper", Transcriber,
                   AsrError)
        self._load("translator", "translate_error", "translation server",
                   Translator, TranslationError)

    def _load(self, attribute: str, error_attribute: str, name: str,
              build, failure: type[Exception]) -> None:
        """Build one stage, recording the reason if it cannot be built."""
        if getattr(self, attribute) is not None:
            return
        if error_attribute and getattr(self, error_attribute):
            return                              # already tried and failed
        try:
            log.info("Loading the %s ...", name)
            setattr(self, attribute, build())
            log.info("%s ready", name.capitalize())
        except failure as exc:
            if error_attribute:
                setattr(self, error_attribute, str(exc))
            log.error("%s unavailable: %s", name.capitalize(), exc)

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
        await _refuse_second_meeting(socket)
        return

    session = ServerSession(segmenter_factory=state.make_segmenter,
                            stages=state.stages)
    claimed = False
    try:
        claimed = await _run_session(socket, session)
    except WebSocketDisconnect:
        log.info("Session %s disconnected", session.session_id or "?")
    finally:
        await _close_session(socket, session, claimed)


async def _refuse_second_meeting(socket: WebSocket) -> None:
    await socket.send_text(make_error(
        f"another session is already streaming ({state.active_session_id}); "
        "this server handles one meeting at a time"
    ))
    await socket.close(code=1013)               # try again later


async def _run_session(socket: WebSocket, session: ServerSession) -> bool:
    """Pump frames until the client leaves. True if the slot was claimed."""
    claimed = False
    while True:
        message = await socket.receive()
        if message["type"] == "websocket.disconnect":
            return claimed

        if (text := message.get("text")) is not None:
            response = session.handle_text(text)
        elif (data := message.get("bytes")) is not None:
            response = session.handle_binary(data)
        else:                                   # pragma: no cover - empty frame
            continue

        if not claimed and session.session_id:
            state.active_session_id = session.session_id
            claimed = True

        await _send(socket, response)
        if response.close:
            await socket.close(
                code=1000 if "bye" in response.close_reason else 1008)
            return claimed


async def _close_session(socket: WebSocket, session: ServerSession,
                         claimed: bool) -> None:
    """Finish the last sentence, release the slot, log what happened.

    A dropped connection mid-sentence still has to close the segment, or the
    buffer manager waits forever for an end that never comes.
    """
    final = session.finish()
    if final.messages:
        try:
            await _send(socket, final)
        except Exception as exc:
            # Not swallowed: these are the last sentence of the meeting and
            # its translation, and a closed socket is how one went missing.
            log.warning("Session %s: %d closing message(s) could not be sent "
                        "(%s: %s)", session.session_id or "?",
                        len(final.messages), type(exc).__name__, exc)

    if claimed and state.active_session_id == session.session_id:
        state.active_session_id = None
    _log_summary(session)


def _log_summary(session: ServerSession) -> None:
    stats = session.stats
    log.info(
        "Session %s finished: %d chunks, %.1f s audio, %d segments, "
        "%d utterances (%d dropped as noise, %d shaped, %d identified, "
        "%d with a language), %d transcripts, %d translations, %d partials, "
        "%d events, %d protocol errors, %d pipeline errors; "
        "slowest sentence %.1f s, slowest running text %.1f s, stages %s; "
        "%d translations dropped, worst translation lag %.1f s",
        session.session_id or "?",
        stats.chunks, stats.audio_seconds, stats.speech_segments,
        stats.utterances, stats.utterances_dropped, stats.utterances_shaped,
        stats.utterances_identified, stats.utterances_with_language,
        stats.transcripts, stats.translations, stats.partials,
        stats.events_sent, stats.protocol_errors, stats.pipeline_errors,
        stats.slowest_utterance_seconds, stats.slowest_partial_seconds,
        {stage: round(value, 1)
         for stage, value in sorted(stats.stage_seconds.items(),
                                    key=lambda kv: -kv[1])},
        stats.translations_dropped, stats.worst_translation_lag,
    )
    if session.speaker_history is not None:
        history = session.speaker_history.stats
        log.info("Session %s speakers: %d after %d reclustering runs, "
                 "%d labels corrected",
                 session.session_id or "?", history.speakers, history.runs,
                 history.corrections)
    _log_voice_scores(session)


def _log_voice_scores(session: ServerSession) -> None:
    """The distribution behind SPEAKER_CHANGE_THRESHOLD.

    The threshold was placed between ranges measured on whole sentences, not
    on the one-second windows this compares, so it needs a real meeting to
    settle. Printed as deciles: a threshold belongs in the gap between the
    same-voice cluster and the different-voice one, and deciles show whether
    there is a gap at all.
    """
    watcher = session.speaker_change
    if watcher is None or not watcher.stats.scores:
        return
    scores = sorted(watcher.stats.scores)
    deciles = [round(scores[min(len(scores) - 1, len(scores) * n // 10)], 3)
               for n in range(11)]
    log.info("Session %s voice comparisons: %d checks, %d cuts, "
             "threshold %.2f, deciles %s",
             session.session_id or "?", watcher.stats.checks,
             watcher.stats.changes, watcher.threshold, deciles)
