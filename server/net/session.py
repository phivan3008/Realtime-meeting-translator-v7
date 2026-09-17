"""Per-connection server logic, independent of the web framework.

Everything here is synchronous and returns the messages to send rather than
sending them, so the whole protocol can be unit tested on the Dev PC without
a socket, an event loop or a GPU.  ``server/app.py`` is the thin layer that
owns the actual WebSocket.

The session is a small state machine::

    AWAITING_HELLO --hello ok--> STREAMING --bye/disconnect--> CLOSED
          |                          |
          +--- bad hello ---> CLOSED +--- bad chunk ---> CLOSED

Streaming ASR
-------------
The open utterance is decoded every ``PARTIAL_INTERVAL_MS`` on its last
``PARTIAL_WINDOW_SECONDS``, and words consecutive decodes agree on are
committed (:mod:`server.pipeline.asr`). The partial and the final of one
utterance share an ASR id built from the session id and the buffer's
utterance index, which the buffer keeps the same for both.

The running text's language is fixed once ``ASR_STREAM_LANGUAGE_VOTES``
windows in a row confidently agree on it, and changed - the running text
started again - when as many agree on the other one. The committed sentence
is decoded in the language the LID found for the whole utterance whenever the
LID is sure of it; only when it is not does the running text's language
stand.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Optional

import numpy as np

from common.protocol import (
    ClientMessage,
    Hello,
    ProtocolError,
    make_error,
    make_final,
    make_partial,
    make_ready,
    make_speakers,
    make_translation,
    make_utterance,
    make_vad,
    parse_message,
    validate_audio_chunk,
)
from server.config import (
    ASR_LANGUAGE_OVERRIDE_MARGIN,
    ASR_STREAM_LANGUAGE_VOTES,
    LANGUAGE_SPLIT,
    PARTIAL_WINDOW_SECONDS,
    SAMPLE_RATE,
    SPEAKER_RECLUSTER,
)
from server.pipeline.asr import Transcriber, Transcript
from server.pipeline.buffer import BufferManager, BufferOutput, FinalizeReason
from server.pipeline.diarization import SpeakerIdentifier
from server.pipeline.language_split import LanguageSplitter
from server.pipeline.lid import LanguageIdentifier
from server.pipeline.noise import NoiseFilter
from server.pipeline.overlap import OverlapResolver
from server.pipeline.reclustering import SpeakerHistory
from server.pipeline.translate import Translator, Turn
from server.pipeline.translation_queue import TranslationWorker
from server.pipeline.vad import VADSegmenter

log = logging.getLogger(__name__)


class SessionState(str, Enum):
    AWAITING_HELLO = "awaiting_hello"
    STREAMING = "streaming"
    CLOSED = "closed"


@dataclass
class Response:
    """What the transport layer should do after one incoming frame."""

    messages: list[str] = field(default_factory=list)
    close: bool = False
    close_reason: str = ""


@dataclass
class ServerSessionStats:
    chunks: int = 0
    bytes_received: int = 0
    speech_segments: int = 0
    events_sent: int = 0
    utterances: int = 0
    utterances_dropped: int = 0
    utterances_shaped: int = 0
    utterances_identified: int = 0
    utterances_with_language: int = 0
    #: Utterances cut in two because they held two languages.
    language_splits: int = 0
    #: Sentences decoded whole again because the LID was sure of a language
    #: other than the one their running text was decoded in.
    language_flips: int = 0
    #: Open utterances whose running text changed language part way, and
    #: how many of those were cut there into two sentences.
    running_language_changes: int = 0
    running_language_cuts: int = 0
    #: Cuts at the end of a sentence refused because the running text's sure
    #: windows heard another language over one of the halves.
    language_splits_refused: int = 0
    #: Labels the live matcher got wrong and clustering put right.
    speaker_corrections: int = 0
    #: Committed sentences with text. Running texts are counted apart:
    #: folded together this read as several sentences a second.
    transcripts: int = 0
    running_texts: int = 0
    translations: int = 0
    #: Sentences that went out without a translation: refused by the model,
    #: or given up on because the answer would have arrived too late to read.
    translations_dropped: int = 0
    #: Longest a translation took from sentence to wire.
    worst_translation_lag: float = 0.0
    partials: int = 0
    protocol_errors: int = 0
    #: Batches lost to a bug outside any stage. Not protocol errors: the
    #: client did nothing wrong and the meeting carries on without them.
    pipeline_errors: int = 0
    #: How many times each stage raised, and which were switched off.
    stage_failures: dict = field(default_factory=dict)
    stages_disabled: list = field(default_factory=list)
    #: Seconds spent in each stage, summed over the meeting, and the worst
    #: single sentence. Every stage runs on the thread that reads audio, so
    #: this is exactly the time the socket was not being read.
    stage_seconds: dict = field(default_factory=dict)
    slowest_utterance_seconds: float = 0.0
    #: The same for the running text, counted apart. It runs every 600 ms,
    #: so it does far more decoding than the finals do, and folding the two
    #: together hides which one is slow.
    slowest_partial_seconds: float = 0.0

    @property
    def audio_seconds(self) -> float:
        return self.bytes_received / 2 / 16_000


@dataclass
class Stages:
    """The models a session runs audio through.

    Every one is optional: a pod missing one still runs without it.
    ``/health`` reports which are loaded.
    """

    noise_filter: Optional[NoiseFilter] = None
    overlap_resolver: Optional[OverlapResolver] = None
    speaker_identifier: Optional[SpeakerIdentifier] = None
    language_identifier: Optional[LanguageIdentifier] = None
    transcriber: Optional[Transcriber] = None
    translator: Optional[Translator] = None


@dataclass
class Analysis:
    """What the pipeline worked out about one utterance."""

    audio: bytes                    # shaped, for the ASR
    keep: bool = True
    label: str = ""                 # what it sounded like, when dropped
    speech_score: float = 0.0
    speaker_id: str = ""
    #: Kept for reclustering the meeting later, not used by any other stage.
    voiceprint: Optional[np.ndarray] = None
    lang_code: str = ""
    #: Whether ``lang_code`` is a decision rather than a fallback, and whether
    #: it is sure enough to overrule the running text.
    lang_known: bool = False
    lang_sure: bool = False
    transcript: Optional[Transcript] = None


#: A sentence taking longer than this stalls the whole connection, because
#: every stage runs on the thread that reads audio. At 200 ms per chunk, one
#: second is five chunks the socket did not get to.
SLOW_UTTERANCE_SECONDS = 1.0

#: A stage that raises this many times in a row is switched off for the rest
#: of the meeting. One failure is a bad sentence; the same failure on every
#: sentence is a broken stage, and calling it again only buries the evidence.
STAGE_FAILURE_LIMIT = 3

#: A failure naming any of these is a broken device, not a bad sentence, and
#: the stage goes off on the first one. On the pod a cuDNN error raised
#: cleanly, CUDA kept working for five more seconds, and the process died on
#: the next call into the same stage - so going back in is what kills it.
DEVICE_FAILURES = ("cuda", "cudnn", "cublas", "out of memory")

#: Which attributes a broken stage switches off. Everything that reads the
#: same model goes with it.
STAGE_ATTRIBUTES = {
    "noise": ("noise_filter",),
    "overlap": ("overlap_resolver",),
    "speaker": ("speaker_identifier", "speaker_history"),
    "recluster": ("speaker_history",),
    "language": ("language_identifier", "language_splitter"),
    "partial_language": ("language_identifier", "language_splitter"),
    "language_split": ("language_splitter",),
    "asr": ("transcriber",),
    "partial_asr": ("transcriber",),
}


class ServerSession:
    """Drive one client connection: handshake, then audio into the VAD."""

    def __init__(
        self,
        segmenter_factory: Callable[[], VADSegmenter],
        buffer_factory: Callable[[], BufferManager] = BufferManager,
        stages: Optional[Stages] = None,
        translation_inline: bool = False,
        strict_chunk_size: bool = True,
        **models,
    ) -> None:
        self._segmenter_factory = segmenter_factory
        self._buffer_factory = buffer_factory
        self.stages = stages if stages is not None else Stages(**models)
        self.noise_filter = self.stages.noise_filter
        self.overlap_resolver = self.stages.overlap_resolver
        self.speaker_identifier = self.stages.speaker_identifier
        #: Second thoughts about the labels the live matcher gave out. Sent to
        #: the client only with SPEAKER_RECLUSTER=1; otherwise measured and
        #: logged.
        self.speaker_history: Optional[SpeakerHistory] = (
            SpeakerHistory() if self.speaker_identifier is not None else None)
        self.send_speaker_corrections = SPEAKER_RECLUSTER
        self.language_identifier = self.stages.language_identifier
        # Two languages in one utterance means one of them is lost rather
        # than mistranslated.
        self.language_splitter: Optional[LanguageSplitter] = (
            LanguageSplitter(self.language_identifier)
            if self.language_identifier is not None and LANGUAGE_SPLIT
            else None
        )
        self.transcriber = self.stages.transcriber
        self.translator = self.stages.translator
        # `inline` runs translation on the calling thread instead of its own,
        # which is what the unit tests use.
        self.worker: Optional[TranslationWorker] = (
            TranslationWorker(self.translator, inline=translation_inline)
            if self.translator is not None else None
        )
        self._strict_chunk_size = strict_chunk_size
        self.state = SessionState.AWAITING_HELLO
        self.hello: Optional[Hello] = None
        self.segmenter: Optional[VADSegmenter] = None
        self.buffer: Optional[BufferManager] = None
        self.stats = ServerSessionStats()
        #: The last language this meeting was confidently in, used when the
        #: LID cannot decide. See :meth:`_language_for`.
        self._last_language = ""
        #: Consecutive failures per stage, and what to tell the client about
        #: the ones that were switched off.
        self._stage_failures: dict[str, int] = {}
        self._stage_notices: list[str] = []
        #: Counts sentences within this session so a translation can be
        #: matched to its sentence.
        self._sentences = 0
        self._clear_open()

    @property
    def session_id(self) -> str:
        return self.hello.session_id if self.hello else ""

    # -- incoming text ------------------------------------------------------
    def handle_text(self, raw: str) -> Response:
        try:
            payload = parse_message(raw)
        except ProtocolError as exc:
            return self._fail(f"bad control message: {exc}")

        kind = payload.get("type")
        if kind == ClientMessage.BYE.value:
            reason = payload.get("reason", "")
            log.info("Session %s said bye: %s", self.session_id, reason)
            # A clean goodbye can still land mid-sentence. Close the segment
            # here, while the socket is open and the event can still be
            # delivered, and wait for its translation too, because after
            # this the socket is gone.
            messages = self._finalise()
            self.state = SessionState.CLOSED
            return Response(messages=messages, close=True,
                            close_reason=f"client bye: {reason}")

        if kind == ClientMessage.HELLO.value:
            if self.state is not SessionState.AWAITING_HELLO:
                return self._fail("hello sent twice on one connection")
            return self._handle_hello(payload)

        return self._fail(f"unexpected {kind!r} from a client")

    def _handle_hello(self, payload: dict) -> Response:
        try:
            hello = Hello.from_dict(payload)
        except ProtocolError as exc:
            return self._fail(f"bad hello: {exc}")

        mismatch = hello.audio_mismatch()
        if mismatch is not None:
            # Refusing here is the whole point of the handshake: mismatched
            # audio does not crash anything downstream, it just quietly makes
            # every transcript wrong.
            return self._fail(f"unsupported audio format: {mismatch}")

        self.hello = hello
        self.segmenter = self._segmenter_factory()
        self.segmenter.reset()
        self.buffer = self._buffer_factory()
        for resettable in (self.speaker_identifier, self.speaker_history,
                           self.language_identifier, self.language_splitter,
                           self.transcriber, self.translator):
            # A new meeting starts with nobody known and nothing said.
            if resettable is not None:
                resettable.reset()
        self._clear_open()
        if self.worker is not None:
            self.worker.start()
        self.state = SessionState.STREAMING
        log.info("Session %s ready (client=%r)", hello.session_id, hello.client)
        return Response(messages=[make_ready(hello.session_id)])

    # -- incoming audio -----------------------------------------------------
    def handle_binary(self, data: bytes) -> Response:
        if self.state is not SessionState.STREAMING:
            return self._fail("audio arrived before a valid hello")

        if self._strict_chunk_size:
            try:
                validate_audio_chunk(data)
            except ProtocolError as exc:
                return self._fail(str(exc))

        self.stats.chunks += 1
        self.stats.bytes_received += len(data)

        assert self.segmenter is not None       # guaranteed by STREAMING
        assert self.buffer is not None
        out = self.segmenter.push(data)
        messages = [
            make_vad(event.kind.value, event.at_ms) for event in out.events
        ]
        self.stats.events_sent += len(messages)
        self.stats.speech_segments = self.segmenter.stats.segments
        messages += self._safely_announce(self.buffer.push(out))
        # Translations finished since the last chunk. Audio arrives every
        # 200 ms, so this is a cheap and regular heartbeat to hand them back
        # on - no extra timer, and at most 200 ms of extra delay.
        messages += self._collect_translations()
        return Response(messages=messages)

    def _language_for(self, decision) -> str:
        """The language to force on the ASR when the LID could not decide.

        Passing "" hands the choice to Whisper's own detector, which is
        choosing between ninety-nine - on a live run it answered Swedish,
        Finnish, Chinese and English for a Vietnamese-Japanese meeting.
        Falling back to the last language this meeting was confidently in is
        wrong at worst half the time, and only when a speaker switches.
        """
        if decision.known:
            return decision.lang_code
        if self._last_language:
            log.debug("Language undecided (%s); falling back to the meeting's "
                      "last known language %r", decision.reason,
                      self._last_language)
        return self._last_language

    def _safely_announce(self, result: BufferOutput) -> list[str]:
        """Run the pipeline, and let the meeting outlive a bug outside a stage.

        Each stage already survives its own failures (:meth:`_stage`). What
        reaches here is a bug in the glue between them; it is logged with its
        traceback and counted, and it must not end the connection.
        """
        try:
            return self._announce(result)
        except Exception:
            self.stats.pipeline_errors += 1
            log.exception(
                "Session %s: the pipeline raised on %d sentence(s); "
                "dropping them and carrying on",
                self.session_id or "?", len(result.finals))
            return []

    @contextmanager
    def _stage(self, name: str, into: dict):
        """Time one stage, and let the meeting outlive a stage that breaks.

        A stage that raises leaves :class:`Analysis` at its defaults, which
        are what the pipeline does when that stage is absent - so the sentence
        carries on through the rest of it instead of being thrown away. On the
        pod, a cuDNN failure in one stage killed every sentence for two
        minutes and the client had no way to tell that from silence.
        """
        try:
            with self._timed(name, into):
                yield
        except Exception as exc:
            self._stage_broke(name, exc)
        else:
            self._stage_failures.pop(name, None)

    def _stage_broke(self, name: str, exc: BaseException) -> None:
        count = self._stage_failures.get(name, 0) + 1
        self._stage_failures[name] = count
        self.stats.stage_failures[name] = (
            self.stats.stage_failures.get(name, 0) + 1)
        log.exception("Session %s: stage %r raised (%d in a row)",
                      self.session_id or "?", name, count)

        message = str(exc).lower()
        device = any(word in message for word in DEVICE_FAILURES)
        if name in self.stats.stages_disabled:
            return
        if not device and count < STAGE_FAILURE_LIMIT:
            return
        for attribute in STAGE_ATTRIBUTES.get(name, ()):
            setattr(self, attribute, None)
        self.stats.stages_disabled.append(name)
        self._stage_notices.append(
            f"tầng {name} đã bị tắt vì lỗi CUDA/thiết bị; cuộc họp vẫn chạy "
            f"nhưng thiếu tầng này"
            if device else
            f"tầng {name} đã bị tắt sau {count} lần lỗi liên tiếp; "
            f"cuộc họp vẫn chạy nhưng thiếu tầng này")

    @contextmanager
    def _timed(self, stage: str, into: dict):
        """Charge the wall time of one stage to that stage.

        Everything here runs on the thread that reads the socket, so these
        numbers are the time the connection spent not reading audio.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            spent = time.perf_counter() - started
            into[stage] = into.get(stage, 0.0) + spent
            self.stats.stage_seconds[stage] = (
                self.stats.stage_seconds.get(stage, 0.0) + spent)

    # -- the open utterance ---------------------------------------------------
    def _asr_id(self, index: int) -> str:
        """One id for the partials and the final of an utterance.

        Buffer indexes restart for every session, so the session id keeps
        two meetings apart.
        """
        return f"{self.session_id or 'session'}:{index}"

    def _clear_open(self) -> None:
        self._open_asr_id = ""
        #: The running text's language, once enough windows agreed on it.
        self._open_language = ""
        #: The run of confident window answers: (language, how many in a row).
        self._language_run: tuple = ("", 0)
        #: Sure LID answers on the running text's windows, as (start, end in
        #: ms from the utterance start, language).
        self._open_windows: list = []
        #: What the screen is showing as running text, so an emptied one is
        #: cleared once rather than sent on every interval.
        self._open_running = ""

    def _forget(self, asr_id: str) -> None:
        """Drop any streaming state kept for an utterance that is done."""
        if self.transcriber is not None:
            self.transcriber.cancel_utterance(asr_id)
        if asr_id == self._open_asr_id:
            self._clear_open()

    # -- committed sentences ----------------------------------------------------
    def _announce(self, result: BufferOutput) -> list[str]:
        """Run each finished sentence through the pipeline and send it."""
        messages = []
        for whole in result.finals:
            whole_id = self._asr_id(whole.index)
            parts = self._by_language(whole)
            streamed = self._stream_language(whole_id) if len(parts) > 1 else ""
            for utterance, language in parts:
                # A half in the running text's language is finished from the
                # running text, like a whole sentence; the other half has
                # nothing there and is decoded on its own.
                use_stream = len(parts) == 1 or (bool(streamed)
                                                 and language == streamed)
                if use_stream:
                    streamed = ""
                messages += self._emit(
                    utterance, whole_id if use_stream else None, language,
                    offset_ms=utterance.start_ms - whole.start_ms)
            self._forget(whole_id)

        if result.partial is not None:
            self.stats.partials += 1
            messages += self._transcribe_partial(result.partial)

        messages += self._recluster_speakers()
        while self._stage_notices:
            messages.append(make_error(self._stage_notices.pop(0), fatal=False))
        self.stats.utterances += len(result.finals)
        self.stats.events_sent += len(messages)
        return messages

    def _stream_language(self, asr_id: str) -> str:
        if self.transcriber is None:
            return ""
        return self.transcriber.stream_language(asr_id)

    def _emit(self, utterance, asr_id: Optional[str], language: str,
              offset_ms: float = 0.0) -> list[str]:
        """One committed utterance through the pipeline and onto the wire.

        ``offset_ms`` is where the utterance starts on the timeline of the
        running text named by ``asr_id``: non-zero for the second half of a
        cut.
        """
        spent: dict[str, float] = {}
        found = self._analyse(utterance, spent, asr_id=asr_id,
                              language=language, offset_ms=offset_ms)
        messages = [self._utterance_message(utterance, found)]
        if found.transcript is not None and found.transcript.has_text:
            messages += self._commit_sentence(found)
        elif language and found.transcript is not None:
            # A half of a cut that says nothing is a turn lost to the cut.
            log.info("Session %s: utterance %d (%.0f ms, %s, %s) came back "
                     "empty; dropped %s", self.session_id or "?",
                     utterance.index, utterance.duration_ms, language,
                     utterance.reason.value,
                     [reason for _piece, reason in found.transcript.dropped])
        self._report_if_slow(utterance, spent)
        return messages

    def _by_language(self, utterance) -> list[tuple]:
        """One utterance, or its two halves when it holds two languages.

        Each part comes with the language already decided for it, or "" to
        let the LID decide. A split hands each half the language the review
        probes found for that half.
        """
        if self.language_splitter is None:
            return [(utterance, "")]
        split = None
        with self._stage("language_split", {}):
            split = self.language_splitter.find(
                utterance.pcm, hangover_ms=utterance.trailing_silence_ms)
        if split is None:
            return [(utterance, "")]
        heard = self._window_languages(utterance.index, split.at_ms)
        if heard[0] not in ("", split.first) or heard[1] not in ("", split.second):
            # The running text heard something else over one of the halves,
            # window after window. On a real meeting a cut like this put six
            # seconds of Vietnamese into a half decoded as Japanese
            # ("はい、で、ウェル") and nothing came out of the other.
            self.stats.language_splits_refused += 1
            log.info("Session %s: utterance %d: the cut heard %s then %s but "
                     "the running text heard %s then %s; left whole",
                     self.session_id or "?", utterance.index, split.first,
                     split.second, heard[0] or "?", heard[1] or "?")
            return [(utterance, "")]

        self.stats.language_splits += 1
        log.info("Session %s: utterance %d holds two languages (%s then %s); "
                 "cutting at %.0f ms of %.0f",
                 self.session_id or "?", utterance.index, split.first,
                 split.second, split.at_ms, utterance.duration_ms)
        return [
            (replace(utterance, pcm=utterance.pcm[:split.at],
                     trailing_silence_ms=0.0), split.first),
            (replace(utterance, pcm=utterance.pcm[split.at:],
                     start_ms=utterance.start_ms + split.at_ms,
                     continues_previous=False), split.second),
        ]

    def _window_languages(self, index: int, at_ms: float) -> tuple:
        """What the running text's sure windows heard before and after a cut.

        Each window counts for the side its middle falls on. Returns the
        majority on each side, or "" where no sure window fell or the vote
        was tied.
        """
        if self._open_asr_id != self._asr_id(index):
            return "", ""
        sides: tuple = ({}, {})
        for start, end, lang in self._open_windows:
            side = sides[0] if (start + end) / 2.0 < at_ms else sides[1]
            side[lang] = side.get(lang, 0) + 1
        found = []
        for side in sides:
            ranked = sorted(side.items(), key=lambda kv: -kv[1])
            if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
                found.append("")
            else:
                found.append(ranked[0][0])
        return tuple(found)

    def _recluster_speakers(self) -> list[str]:
        """Cluster the meeting again and send back the labels that moved."""
        if self.speaker_history is None or not self.speaker_history.due:
            return []
        if not self.send_speaker_corrections:
            # Measured, not sent: on a real meeting of four people the
            # corrections made the labels worse. See SPEAKER_RECLUSTER.
            with self._stage("recluster", {}):
                self.speaker_history.survey()
            return []
        corrections: dict = {}
        with self._stage("recluster", {}):
            corrections = self.speaker_history.recluster()
        if not corrections:
            return []
        self.stats.speaker_corrections += len(corrections)
        return [make_speakers(corrections)]

    def _analyse(self, utterance, spent: dict, asr_id: Optional[str] = None,
                 language: str = "", offset_ms: float = 0.0) -> Analysis:
        """Every stage that reads audio, in order, for one utterance.

        Each stage owns the whole of its block, result included: a stage that
        raises must leave :class:`Analysis` at the defaults, which are what
        the pipeline does when that stage is absent.

        ``asr_id`` names the streaming state the running text left for this
        utterance; None decodes it whole. ``language`` is a decision already
        made, which the LID is then not asked to repeat.
        """
        found = Analysis(audio=utterance.pcm)

        if self.noise_filter is not None:
            with self._stage("noise", spent):
                verdict = self.noise_filter.judge(utterance.pcm)
                found.keep = verdict.keep
                found.speech_score = verdict.classification.speech_score
                if not verdict.keep:
                    found.label = verdict.classification.noise_label
                    self.stats.utterances_dropped += 1
        if not found.keep:
            return found

        # Shaping is for the ASR. Speaker and language read the raw audio:
        # the gate removes quiet syllables, and those carry both voice (0.06
        # cosine, measured) and the cues that tell the two languages apart.
        if self.overlap_resolver is not None:
            with self._stage("overlap", spent):
                shaped = self.overlap_resolver.resolve(found.audio)
                found.audio = shaped.pcm
                if shaped.shaped:
                    self.stats.utterances_shaped += 1

        if self.speaker_identifier is not None:
            with self._stage("speaker", spent):
                assignment = self.speaker_identifier.identify(utterance.pcm)
                found.speaker_id = assignment.speaker_id
                found.voiceprint = assignment.embedding
                self.stats.utterances_identified += 1
                # The score is what decides whether two turns are one person.
                log.info("utterance %d speaker %s similarity %.3f (%s)",
                         utterance.index, assignment.speaker_id,
                         assignment.similarity, assignment.reason)

        if language:
            found.lang_code = language
            found.lang_known = found.lang_sure = True
            self.stats.utterances_with_language += 1
            self._last_language = language
        elif self.language_identifier is not None:
            with self._stage("language", spent):
                decision = self.language_identifier.identify(utterance.pcm)
                if decision.known:
                    self.stats.utterances_with_language += 1
                    self._last_language = decision.lang_code
                found.lang_code = self._language_for(decision)
                found.lang_known = decision.known
                found.lang_sure = self._sure(decision)
        else:
            found.lang_code = self._last_language

        if self.transcriber is not None:
            with self._stage("asr", spent):
                found.transcript = self._decode(utterance, found, asr_id,
                                                offset_ms)
                if found.transcript.has_text:
                    self.stats.transcripts += 1
        return found

    def _decode(self, utterance, found: Analysis, asr_id: Optional[str],
                offset_ms: float = 0.0) -> Transcript:
        """The committed sentence: the running text's work, finished."""
        assert self.transcriber is not None
        if asr_id is None:
            return self.transcriber.transcribe(found.audio, found.lang_code,
                                               is_final=True)
        streamed = self.transcriber.stream_language(asr_id)
        if streamed and not found.lang_sure:
            # The LID is not sure enough. The running text's language was
            # agreed by several windows, which beats a weak answer or a
            # fallback to the last sentence.
            found.lang_code = streamed
        transcript = self.transcriber.finish_utterance(
            found.audio,
            utterance_id=asr_id,
            utterance_start_seconds=offset_ms / 1000.0,
            # Where the speech ended, not where the VAD's hangover did.
            # Whisper answers that half second of silence with words.
            speech_end_seconds=(offset_ms + utterance.speech_end_ms
                                - utterance.start_ms) / 1000.0,
            lang_code=found.lang_code,
        )
        if transcript.discarded_language:
            # The LID on the whole sentence is sure, and the running text was
            # decoded in the other language: on a real meeting that was
            # Japanese speech shown as Vietnamese inventions, with the whole
            # sentence's LID right. So the sentence is decoded whole again.
            self.stats.language_flips += 1
            why = (self.language_splitter.stats.last
                   if self.language_splitter else "no splitter")
            log.info("utterance %d: the running text was %r, the LID is sure "
                     "of %r; decoded the whole sentence again. The splitter "
                     "said: %s", utterance.index,
                     transcript.discarded_language, transcript.lang_code,
                     why or "nothing")
        return transcript

    def _utterance_message(self, utterance, found: Analysis) -> str:
        """The verdict on one utterance, sent whether it was kept or not."""
        return make_utterance(
            index=utterance.index,
            start_ms=utterance.start_ms,
            end_ms=utterance.end_ms,
            reason=utterance.reason.value,
            continues_previous=utterance.continues_previous,
            kept=found.keep,
            label=found.label,
            speech_score=found.speech_score,
            speaker_id=found.speaker_id,
            lang_code=found.lang_code,
        )

    def _commit_sentence(self, found: Analysis) -> list[str]:
        """Send the sentence now and queue its translation to follow."""
        self._sentences += 1
        if self.speaker_history is not None and found.voiceprint is not None:
            self.speaker_history.add(self._sentences, found.voiceprint,
                                     found.speaker_id)
        transcript = found.transcript
        assert transcript is not None
        messages = [make_final(
            sentence_id=self._sentences,
            speaker_id=found.speaker_id,
            lang_code=transcript.lang_code,
            transcript=transcript.text,
            speech_score=found.speech_score,
        )]
        if self.worker is not None:
            # The history this sentence is translated with is the meeting as
            # it stands now - taken before the sentence joins it, because the
            # translation runs later, on another thread, when the history
            # would already hold this sentence and the ones after it. The
            # session owns the history; the translator adds nothing to it, so
            # a dropped translation still leaves its source behind.
            context = self.worker.translator.context
            history = context.snapshot()
            context.remember(Turn(
                speaker_id=found.speaker_id,
                lang_code=transcript.lang_code,
                source=transcript.text,
                translation="",
            ))
            self.worker.submit(self._sentences, transcript.text,
                               transcript.lang_code, found.speaker_id,
                               history=history)
        return messages

    def _report_if_slow(self, utterance, spent: dict) -> None:
        """Name the stage that stalled the connection, while it is still known."""
        total = sum(spent.values())
        self.stats.slowest_utterance_seconds = max(
            self.stats.slowest_utterance_seconds, total)
        if total < SLOW_UTTERANCE_SECONDS:
            return
        worst = max(spent, key=spent.get)
        log.warning(
            "Session %s: sentence %d (%.1f s of audio) held the socket for "
            "%.1f s - %s took %.1f s of it. Breakdown: %s",
            self.session_id or "?", utterance.index,
            (utterance.end_ms - utterance.start_ms) / 1000.0,
            total, worst, spent[worst],
            {stage: round(value, 2) for stage, value in spent.items()},
        )

    # -- running text -----------------------------------------------------------
    def _transcribe_partial(self, partial) -> list[str]:
        """Decode the rolling window and send the running text.

        The protocol has one replace-only running text, so it carries the
        committed text and the unstable text together. No speaker label goes
        out with it: showing a name and then correcting it reads worse than
        showing none.
        """
        if self.transcriber is None:
            return []

        utterance_id = self._asr_id(partial.index)
        if utterance_id != self._open_asr_id:
            if self._open_asr_id:
                # The buffer finishes one utterance before opening the next,
                # so this is stale state that must not leak.
                self._forget(self._open_asr_id)
            self._open_asr_id = utterance_id

        window = partial.tail(PARTIAL_WINDOW_SECONDS)
        window_start_seconds = max(
            0.0, (window.start_ms - partial.start_ms) / 1000.0)

        spent: dict[str, float] = {}
        lang_code = self._open_language or self._last_language
        changed_from = ""
        if self.language_identifier is not None:
            with self._stage("partial_language", spent):
                decision = self.language_identifier.identify(window.pcm)
                if self._sure(decision):
                    self._open_windows.append((
                        window.start_ms - partial.start_ms,
                        window.end_ms - partial.start_ms,
                        decision.lang_code))
                lang_code, changed_from = self._running_language(
                    decision, utterance_id)

        if changed_from:
            cut = self._cut_on_language_change(partial, changed_from,
                                               lang_code, spent)
            if cut is not None:
                return cut

        events: tuple = ()
        with self._stage("partial_asr", spent):
            events = self.transcriber.process_partial(
                window.pcm,
                utterance_id=utterance_id,
                window_start_seconds=window_start_seconds,
                lang_code=lang_code,
            )
        self._report_if_partial_slow(window, spent)

        latest = next((event for event in reversed(events)
                       if event.kind == "partial"), None)
        if latest is None:
            return []

        running = latest.running_text
        if not running and not self._open_running:
            return []
        self._open_running = running
        if running:
            self.stats.running_texts += 1
        # An empty one is sent once, to clear what the screen still shows.
        return [make_partial("", latest.lang_code, running)]

    @staticmethod
    def _sure(decision) -> bool:
        """Sure enough to overrule what the running text agreed on."""
        return (decision.known
                and decision.margin >= ASR_LANGUAGE_OVERRIDE_MARGIN)

    def _running_language(self, decision, utterance_id: str) -> tuple:
        """The language for this window, and the one it replaced, if any.

        Fixed once ``ASR_STREAM_LANGUAGE_VOTES`` sure answers in a row
        agree, and changed when as many agree on another language. Before it
        is fixed a window uses its own answer only when that answer is sure;
        otherwise the meeting's last language, because a weak answer forced
        on Whisper produced inventions in a language nobody was speaking.
        """
        changed_from = ""
        if self._sure(decision):
            # Only sure answers vote. Two weak ones fixed Vietnamese over a
            # Japanese speaker once, and the running text invented a sentence
            # in it ("Bên mặt của nó sẽ là").
            language, count = self._language_run
            count = count + 1 if language == decision.lang_code else 1
            self._language_run = (decision.lang_code, count)
            if (count >= ASR_STREAM_LANGUAGE_VOTES
                    and decision.lang_code != self._open_language):
                if self._open_language:
                    changed_from = self._open_language
                    self.stats.running_language_changes += 1
                    log.info("utterance %s: the running text changes from %r "
                             "to %r", utterance_id, self._open_language,
                             decision.lang_code)
                self._open_language = decision.lang_code
        if self._open_language:
            return self._open_language, changed_from
        if self._sure(decision):
            return decision.lang_code, changed_from
        return self._last_language, changed_from

    def _cut_on_language_change(self, partial, before: str, after: str,
                                spent: dict) -> Optional[list]:
        """Cut the open utterance where its language changed, if it can.

        The running text is the only stage that sees a second turn while it
        is happening: on a real meeting it changed language 35 times inside a
        sentence, and the cut at the end found few of those - its probes near
        the end of the whole sentence had the other turn to contend with. Asked
        now, the tail of the audio is the turn that has just begun.

        Returns the messages for the committed head, or None to carry on with
        the whole utterance.
        """
        if self.language_splitter is None or self.buffer is None:
            return None
        split = None
        with self._stage("language_split", spent):
            split = self.language_splitter.find(partial.pcm, hangover_ms=0.0)
        if split is None or split.first != before or split.second != after:
            return None
        extra = self.buffer.cut_at(split.at_ms)
        if not extra.finals:
            return None
        self.stats.running_language_cuts += 1
        log.info("Session %s: utterance %d changed from %s to %s; committing "
                 "the first %.0f ms as its own sentence",
                 self.session_id or "?", partial.index, before, after,
                 split.at_ms)
        old_id = self._asr_id(partial.index)
        messages = []
        for utterance in extra.finals:
            # The running text is still in the language of the head - it has
            # not been decoded in the new one yet - so the head is finished
            # from it. Decoded again from nothing, a head once came back as
            # an invented sign-off and was dropped.
            streamed = self._stream_language(old_id) == before
            messages += self._emit(utterance, old_id if streamed else None,
                                   before)
        self.stats.utterances += len(extra.finals)
        self._forget(old_id)
        # The turn that has just begun keeps the votes it already won, and
        # the windows that heard it, moved to its own timeline.
        windows = [(start - split.at_ms, end - split.at_ms, lang)
                   for start, end, lang in self._open_windows
                   if (start + end) / 2.0 >= split.at_ms]
        self._open_language = after
        self._language_run = (after, ASR_STREAM_LANGUAGE_VOTES)
        self._open_windows = windows
        self._report_if_partial_slow(partial, spent)
        return messages

    def _report_if_partial_slow(self, partial, spent: dict) -> None:
        """Say so when the running text is what held up the connection."""
        total = sum(spent.values())
        self.stats.slowest_partial_seconds = max(
            self.stats.slowest_partial_seconds, total)
        if total < SLOW_UTTERANCE_SECONDS:
            return
        worst = max(spent, key=spent.get)
        log.warning(
            "Session %s: running text for %.1f s of audio held the socket "
            "for %.1f s - %s took %.1f s of it. Breakdown: %s",
            self.session_id or "?", len(partial.pcm) / 2 / SAMPLE_RATE,
            total, worst, spent[worst],
            {stage: round(value, 2) for stage, value in spent.items()},
        )

    # -- teardown -----------------------------------------------------------
    def finish(self) -> Response:
        """Close an open speech segment when the connection goes away.

        Safe to call after a ``bye`` has already closed it: the segmenter
        reports no events the second time, so no duplicate end is emitted.
        """
        messages = self._finalise()
        self.state = SessionState.CLOSED
        return Response(messages=messages)

    def _finalise(self) -> list[str]:
        """Close the last segment and settle every outstanding translation.

        Used by both ``bye`` and ``finish``, because the socket closes as
        soon as ``bye`` is answered: the last sentence of a meeting is
        committed here and its translation collected here, or never sent.
        """
        messages = self._close_segment()
        if self.worker is not None:
            messages += [self._as_message(done)
                         for done in self.worker.stop()]
        return messages

    def _collect_translations(self) -> list[str]:
        """Whatever the worker has finished, in the order it finished it."""
        if self.worker is None:
            return []
        return [self._as_message(done) for done in self.worker.collect()]

    def _as_message(self, done) -> str:
        if done.translation:
            self.stats.translations += 1
        else:
            self.stats.translations_dropped += 1
        self.stats.worst_translation_lag = max(
            self.stats.worst_translation_lag, done.lag_seconds)
        return make_translation(
            sentence_id=done.sentence_id,
            translation=done.translation,
            reason=done.reason,
            raw=done.raw,
        )

    def _close_segment(self) -> list[str]:
        """End an in-progress speech segment and commit what it held."""
        if self.state is not SessionState.STREAMING or self.segmenter is None:
            return []
        out = self.segmenter.close()
        messages = [make_vad(event.kind.value, event.at_ms) for event in out.events]
        self.stats.events_sent += len(messages)
        if self.buffer is not None:
            messages += self._safely_announce(
                self.buffer.flush(FinalizeReason.END_OF_STREAM,
                                  out.trailing_silence_ms)
            )
        return messages

    # -- helpers ------------------------------------------------------------
    def _fail(self, message: str) -> Response:
        self.stats.protocol_errors += 1
        self.state = SessionState.CLOSED
        log.warning("Session %s rejected: %s", self.session_id or "?", message)
        return Response(messages=[make_error(message)], close=True,
                        close_reason=message)
