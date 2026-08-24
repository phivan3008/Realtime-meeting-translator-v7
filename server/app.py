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
from server.pipeline.noise import AstClassifier, NoiseFilter, NoiseFilterError
from server.pipeline.overlap import OverlapError, OverlapResolver
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
                            overlap_resolver=state.overlap_resolver)
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
        with contextlib.suppress(Exception):
            await _send(socket, final)
        if claimed and state.active_session_id == session.session_id:
            state.active_session_id = None
        log.info(
            "Session %s finished: %d chunks, %.1f s audio, %d segments, "
            "%d utterances (%d dropped as noise, %d shaped), %d partials, "
            "%d events, %d protocol errors",
            session.session_id or "?",
            session.stats.chunks,
            session.stats.audio_seconds,
            session.stats.speech_segments,
            session.stats.utterances,
            session.stats.utterances_dropped,
            session.stats.utterances_shaped,
            session.stats.partials,
            session.stats.events_sent,
            session.stats.protocol_errors,
        )
