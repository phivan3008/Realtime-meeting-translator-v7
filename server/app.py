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
import sys
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
from server import config
from server.config import overrides
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
    """Process-wide singletons: the models and the active session."""

    def __init__(self) -> None:
        self.vad: Optional[SileroVAD] = None
        self.noise_filter: Optional[NoiseFilter] = None
        self.noise_error: str = ""
        self.overlap_resolver: Optional[OverlapResolver] = None
        self.overlap_error: str = ""
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
        why rather than the pod going silent. Every stage is attempted: an
        early ``return`` here once left a pod with no ASR because the noise
        filter had been switched off.
        """
        log.info("Python: %s", which_environment())
        set_now = overrides()
        if set_now:
            log.warning("Environment overrides in effect: %s. Anything left "
                        "over from an earlier terminal changes what this run "
                        "does; the defaults need no variables at all.",
                        set_now)
        else:
            log.info("No environment overrides; running on the defaults")
        if not in_venv():
            log.warning(
                "Not running in a virtual environment. The pipeline will "
                "start anyway from whatever this interpreter has, which is "
                "not the set of versions this project pins. Activate it: "
                "source .venv/bin/activate")
        if self.vad is None:
            log.info("Loading Silero VAD ...")
            self.vad = SileroVAD()
            log.info("Silero VAD ready")

        # Off unless asked for. Measured over two real meetings it dropped
        # nothing and cost a quarter of the thread that reads the socket.
        if config.ENABLE_NOISE_FILTER:
            self._load("noise_filter", "noise_error", "audio classifier",
                       lambda: NoiseFilter(classifier=AstClassifier()),
                       NoiseFilterError)
        elif self.noise_filter is None:
            self.noise_error = "off by default; set ENABLE_NOISE_FILTER=1"
            log.info("Deep Noise Filter off (ENABLE_NOISE_FILTER not set)")

        if config.DISABLE_OVERLAP:
            self.overlap_error = "disabled by DISABLE_OVERLAP"
            log.warning("Overlap resolver disabled by environment; the ASR "
                        "will be given raw audio")
        else:
            self._load("overlap_resolver", "overlap_error", "overlap resolver",
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
        if getattr(self, error_attribute):
            return                              # already tried and failed
        try:
            log.info("Loading the %s ...", name)
            setattr(self, attribute, build())
            log.info("%s ready", name.capitalize())
        except failure as exc:
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
    version="0.4.0",
    lifespan=lifespan,
)


def in_venv() -> bool:
    """Whether this interpreter is the project's virtual environment.

    Running the server from the system or conda interpreter is silent: the
    heavy packages are there too, so it starts, loads most of the pipeline
    and serves meetings. It cost the project several measurements taken
    against a pipeline that was not the one under test.
    """
    return sys.prefix != sys.base_prefix


def which_environment() -> str:
    """One line naming the interpreter, for the startup log."""
    where = "venv" if in_venv() else "NOT a venv"
    return f"{sys.executable} ({where}, python {sys.version.split()[0]})"


@app.get("/health")
def health() -> dict:
    """Cheap reachability probe - the client real test calls this first."""
    return {
        "status": "ok",
        "python": sys.executable,
        "in_venv": in_venv(),
        "overrides": overrides(),
        "protocol_version": PROTOCOL_VERSION,
        "sample_rate": SAMPLE_RATE,
        "chunk_bytes": CHUNK_BYTES,
        "chunk_ms": CHUNK_DURATION_MS,
        "vad_loaded": state.vad is not None,
        "noise_filter_loaded": state.noise_filter is not None,
        "noise_filter_error": state.noise_error,
        "overlap_resolver_loaded": state.overlap_resolver is not None,
        "overlap_error": state.overlap_error,
        "speaker_model_loaded": state.speaker_identifier is not None,
        "speaker_model_error": state.speaker_error,
        "language_model_loaded": state.language_identifier is not None,
        "language_model_error": state.language_error,
        "asr_loaded": state.transcriber is not None,
        "asr_error": state.asr_error,
        "translation_loaded": state.translator is not None,
        "translation_error": state.translate_error,
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
    try:
        await _run_session(socket, session)
    except WebSocketDisconnect:
        log.info("Session %s disconnected", session.session_id or "?")
    finally:
        await _close_session(socket, session)


async def _refuse_second_meeting(socket: WebSocket) -> None:
    await socket.send_text(make_error(
        f"another session is already streaming ({state.active_session_id}); "
        "this server handles one meeting at a time"
    ))
    await socket.close(code=1013)               # try again later


async def _run_session(socket: WebSocket, session: ServerSession) -> None:
    """Pump frames until the client leaves."""
    while True:
        message = await socket.receive()
        if message["type"] == "websocket.disconnect":
            return

        if (text := message.get("text")) is not None:
            response = session.handle_text(text)
        elif (data := message.get("bytes")) is not None:
            response = session.handle_binary(data)
        else:                                   # pragma: no cover - empty frame
            continue

        if session.session_id and state.active_session_id is None:
            state.active_session_id = session.session_id

        await _send(socket, response)
        if response.close:
            await socket.close(
                code=1000 if "bye" in response.close_reason else 1008)
            return


async def _close_session(socket: WebSocket, session: ServerSession) -> None:
    """Finish the last sentence, release the slot, log what happened.

    A dropped connection mid-sentence still has to close the segment, or the
    buffer manager waits forever for an end that never comes. The slot is
    released by id rather than by a flag the receive loop returns: a
    disconnect raised inside the loop never returns one, and the pod would
    then refuse every later meeting.
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

    if session.session_id and state.active_session_id == session.session_id:
        state.active_session_id = None
    _log_summary(session)


def _log_summary(session: ServerSession) -> None:
    stats = session.stats
    log.info(
        "Session %s finished: %d chunks, %.1f s audio, %d segments, "
        "%d utterances (%d dropped as noise, %d shaped, %d identified, "
        "%d with a language, %d split on a language change), "
        "%d sentences, %d running texts, %d translations, %d partials, "
        "%d events, %d protocol errors, %d pipeline errors; "
        "slowest sentence %.1f s, slowest running text %.1f s, stages %s; "
        "%d translations dropped, worst translation lag %.1f s",
        session.session_id or "?",
        stats.chunks, stats.audio_seconds, stats.speech_segments,
        stats.utterances, stats.utterances_dropped, stats.utterances_shaped,
        stats.utterances_identified, stats.utterances_with_language,
        stats.language_splits,
        stats.transcripts, stats.running_texts, stats.translations,
        stats.partials, stats.events_sent, stats.protocol_errors,
        stats.pipeline_errors,
        stats.slowest_utterance_seconds, stats.slowest_partial_seconds,
        {stage: round(value, 1)
         for stage, value in sorted(stats.stage_seconds.items(),
                                    key=lambda kv: -kv[1])},
        stats.translations_dropped, stats.worst_translation_lag,
    )
    if session.language_splitter is not None:
        split = session.language_splitter.stats
        log.info("Session %s language splits: %d of %d utterances held two "
                 "languages (%d one language, %d undecided at an end, "
                 "%d too short, %d would leave a fragment, %d refused on "
                 "review), %d probes",
                 session.session_id or "?", split.split, split.checked,
                 split.one_language, split.undecided, split.too_short,
                 split.sliver, split.refused_on_review, split.probes)
    if stats.language_flips:
        log.warning("Session %s: %d sentence(s) were asked for one language "
                    "while their running text had committed words in the "
                    "other; the running text's language was kept",
                    session.session_id or "?", stats.language_flips)
    if session.transcriber is not None:
        asr = session.transcriber.stats
        log.info("Session %s ASR: %d finals (%d empty), %d commits, "
                 "%d repeated words removed at a join, dropped %s",
                 session.session_id or "?", asr.finals, asr.empty,
                 asr.committed_events, asr.duplicates_removed,
                 asr.dropped_reasons)
    if session.worker is not None:
        tr = session.worker.translator.stats
        if tr.retried:
            log.info("Session %s echoes: %d retried without the history, "
                     "%d rescued", session.session_id or "?",
                     tr.retried, tr.rescued)
    if stats.stage_failures:
        log.warning("Session %s: stages that raised %s; switched off %s",
                    session.session_id or "?", stats.stage_failures,
                    stats.stages_disabled or "none")
    if session.speaker_history is not None:
        history = session.speaker_history.stats
        scores = sorted(history.stop_scores)
        log.info("Session %s speakers: %d after %d reclustering runs, "
                 "%d labels corrected, %d merges forced by the speaker cap, "
                 "refused merges %s",
                 session.session_id or "?", history.speakers, history.runs,
                 history.corrections, history.forced_merges,
                 _deciles(scores) if scores else "none")


def _deciles(values: list) -> list:
    """Eleven points across a sorted list, for reading a distribution."""
    return [round(values[min(len(values) - 1, len(values) * n // 10)], 3)
            for n in range(11)]
