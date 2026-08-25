"""Per-connection server logic, independent of the web framework.

Everything here is synchronous and returns the messages to send rather than
sending them, so the whole protocol can be unit tested on the Dev PC without
a socket, an event loop or a GPU.  ``server/app.py`` is the thin layer that
owns the actual WebSocket.

The session is a small state machine::

    AWAITING_HELLO --hello ok--> STREAMING --bye/disconnect--> CLOSED
          |                          |
          +--- bad hello ---> CLOSED +--- bad chunk ---> CLOSED
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from common.protocol import (
    ClientMessage,
    Hello,
    ProtocolError,
    make_error,
    make_final,
    make_translation,
    make_partial,
    make_ready,
    make_utterance,
    make_vad,
    parse_message,
    validate_audio_chunk,
)
from server.config import PARTIAL_WINDOW_SECONDS, SAMPLE_RATE
from server.pipeline.asr import Transcriber
from server.pipeline.buffer import BufferManager, BufferOutput, FinalizeReason
from server.pipeline.diarization import SpeakerIdentifier
from server.pipeline.lid import LanguageIdentifier
from server.pipeline.noise import NoiseFilter
from server.pipeline.overlap import OverlapResolver
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
    transcripts: int = 0
    translations: int = 0
    #: Sentences that went out without a translation: refused by the model,
    #: or given up on because the answer would have arrived too late to read.
    translations_dropped: int = 0
    #: Longest a translation took from sentence to wire.
    worst_translation_lag: float = 0.0
    partials: int = 0
    protocol_errors: int = 0
    #: Sentences lost because a pipeline stage raised. Not protocol errors:
    #: the client did nothing wrong and the meeting carries on without them.
    pipeline_errors: int = 0
    #: Seconds spent in each stage, summed over the meeting, and the worst
    #: single sentence. Every stage runs on the thread that reads audio, so
    #: this is exactly the time the socket was not being read.
    stage_seconds: dict = field(default_factory=dict)
    slowest_utterance_seconds: float = 0.0
    #: The same for the running text, counted apart. It runs every 600 ms on
    #: the whole open utterance, so it does far more decoding than the finals
    #: do, and folding the two together hides which one is slow.
    slowest_partial_seconds: float = 0.0

    @property
    def audio_seconds(self) -> float:
        return self.bytes_received / 2 / 16_000


#: A sentence taking longer than this stalls the whole connection, because
#: every stage runs on the thread that reads audio. At 200 ms per chunk, one
#: second is five chunks the socket did not get to.
SLOW_UTTERANCE_SECONDS = 1.0


class ServerSession:
    """Drive one client connection: handshake, then audio into the VAD."""

    def __init__(
        self,
        segmenter_factory: Callable[[], VADSegmenter],
        buffer_factory: Callable[[], BufferManager] = BufferManager,
        noise_filter: Optional[NoiseFilter] = None,
        overlap_resolver: Optional[OverlapResolver] = None,
        speaker_identifier: Optional[SpeakerIdentifier] = None,
        language_identifier: Optional[LanguageIdentifier] = None,
        transcriber: Optional[Transcriber] = None,
        translator: Optional[Translator] = None,
        translation_inline: bool = False,
        strict_chunk_size: bool = True,
    ) -> None:
        self._segmenter_factory = segmenter_factory
        self._buffer_factory = buffer_factory
        # Optional: a pod without the classifier still runs, it just
        # transcribes the coughs too. /health reports which mode it is in.
        self.noise_filter = noise_filter
        self.overlap_resolver = overlap_resolver
        self.speaker_identifier = speaker_identifier
        self.language_identifier = language_identifier
        self.transcriber = transcriber
        self.translator = translator
        # Translation runs off the thread that reads audio. `inline` puts it
        # back on, which is what the unit tests use so nothing here depends on
        # thread scheduling.
        self.worker: Optional[TranslationWorker] = (
            TranslationWorker(translator, inline=translation_inline)
            if translator is not None else None
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
        #: Counts sentences within this session so a translation can be
        #: matched to its sentence. Utterance indexes restart with each speech
        #: segment, so they cannot be used for it.
        self._sentences = 0

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
            # delivered, or the last sentence of the meeting never gets an
            # end and never gets finalised - and wait for its translation
            # too, because after this the socket is gone.
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
        if self.speaker_identifier is not None:
            # A new meeting starts with nobody known.
            self.speaker_identifier.reset()
        if self.language_identifier is not None:
            self.language_identifier.reset()
        if self.transcriber is not None:
            self.transcriber.reset()
        if self.translator is not None:
            # A new meeting carries none of the last one's context.
            self.translator.reset()
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

        A meeting holds two languages, and the LID is asked about clips as
        short as 400 ms, where it often cannot separate them. Passing "" hands
        the choice to Whisper's own detector, which is choosing between
        ninety-nine - on a live run it answered Swedish (0.66), Finnish (0.50),
        Chinese (0.18) and English (0.29) for a Vietnamese-Japanese meeting,
        and a sentence decoded as Swedish is worthless.

        Falling back to the last language this meeting was confidently in is
        wrong at worst half the time, and only at the moment a speaker
        switches. Whisper's guess was wrong every time.
        """
        if decision.known:
            return decision.lang_code
        if self._last_language:
            log.debug("Language undecided (%s); falling back to the meeting's "
                      "last known language %r", decision.reason,
                      self._last_language)
        return self._last_language

    def _safely_announce(self, result: BufferOutput) -> list[str]:
        """Run the pipeline, and let the meeting outlive a bug in one sentence.

        A bug that reaches here is a bug in this project, not bad input, so it
        is logged with its traceback and counted rather than swallowed. What it
        must not do is end the connection: an ``AttributeError`` on one line
        once killed a 60 s meeting seven times over, and each reconnect threw
        away the audio queued behind it. Losing one sentence beats losing the
        rest of the meeting.
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
    def _timed(self, stage: str, into: dict):
        """Charge the wall time of one stage to that stage.

        Everything here runs on the thread that reads the socket, so these
        numbers are not curiosity: they are the time the connection spent not
        reading audio. A run where one sentence took 12 s showed up on the
        client as VAD events arriving 12 s late, and nothing in the pipeline
        said which stage had taken it.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            spent = time.perf_counter() - started
            into[stage] = into.get(stage, 0.0) + spent
            self.stats.stage_seconds[stage] = (
                self.stats.stage_seconds.get(stage, 0.0) + spent)

    def _announce(self, result: BufferOutput) -> list[str]:
        """Classify each finished sentence, then put it on the wire."""
        messages = []
        for utterance in result.finals:
            spent: dict[str, float] = {}
            keep, label, score = True, "", 0.0
            if self.noise_filter is not None:
                with self._timed("noise", spent):
                    verdict = self.noise_filter.judge(utterance.pcm)
                keep = verdict.keep
                label = verdict.classification.noise_label if not keep else ""
                score = verdict.classification.speech_score
                if not keep:
                    self.stats.utterances_dropped += 1
            audio = utterance.pcm
            if keep and self.overlap_resolver is not None:
                # Only what survives is worth shaping; a dropped sentence goes
                # nowhere. The shaped audio is what the ASR stage will read.
                with self._timed("overlap", spent):
                    shaped = self.overlap_resolver.resolve(audio)
                audio = shaped.pcm
                if shaped.shaped:
                    self.stats.utterances_shaped += 1

            speaker_id = ""
            if keep and self.speaker_identifier is not None:
                # Identified from the *raw* utterance, not the shaped one.
                # Measured on two single-speaker recordings, gating first cost
                # 0.06 of same-speaker cosine (0.677 raw against 0.616 shaped):
                # the gate removes quiet syllables inside a sentence, and those
                # carry voice. The resolver is there to help the ASR, and the
                # bleed it removes is not worth a known loss of identity.
                with self._timed("speaker", spent):
                    assignment = self.speaker_identifier.identify(utterance.pcm)
                speaker_id = assignment.speaker_id
                self.stats.utterances_identified += 1

            lang_code = ""
            if keep and self.language_identifier is not None:
                # Raw audio again, for the same reason: the gate removes quiet
                # phonemes, and those carry the cues that tell the two
                # languages apart.
                with self._timed("language", spent):
                    decision = self.language_identifier.identify(utterance.pcm)
                if decision.known:
                    self.stats.utterances_with_language += 1
                    self._last_language = decision.lang_code
                lang_code = self._language_for(decision)

            transcript = None
            if keep and self.transcriber is not None:
                # The shaped audio, not the raw: gating is what the overlap
                # resolver is for, and this is the stage it was for.
                with self._timed("asr", spent):
                    transcript = self.transcriber.transcribe(audio, lang_code,
                                                             is_final=True)
                if transcript.has_text:
                    self.stats.transcripts += 1

            messages.append(
                make_utterance(
                    index=utterance.index,
                    start_ms=utterance.start_ms,
                    end_ms=utterance.end_ms,
                    reason=utterance.reason.value,
                    continues_previous=utterance.continues_previous,
                    kept=keep,
                    label=label,
                    speech_score=score,
                    speaker_id=speaker_id,
                    lang_code=lang_code,
                )
            )
            if transcript is not None and transcript.has_text:
                self._sentences += 1
                sentence_id = self._sentences
                # The sentence goes out now. Waiting for a translation before
                # saying anything is what put every VAD event 12 s late on the
                # seventh end-to-end run: an LLM call sat on the thread that
                # reads audio, and everything behind it waited too.
                messages.append(make_final(
                    sentence_id=sentence_id,
                    speaker_id=speaker_id,
                    lang_code=transcript.lang_code,
                    transcript=transcript.text,
                    speech_score=score,
                ))
                if self.worker is not None:
                    # The history is remembered here rather than inside the
                    # translator, so a sentence whose translation is dropped
                    # still leaves its source behind for the next one to read.
                    # HISTORY_STYLE is "sources", so that is all it needs.
                    self.worker.translator.context.remember(Turn(
                        speaker_id=speaker_id,
                        lang_code=transcript.lang_code,
                        source=transcript.text,
                        translation="",
                    ))
                    self.worker.submit(sentence_id, transcript.text,
                                       transcript.lang_code, speaker_id)
            self._report_if_slow(utterance, spent)
        self.stats.utterances += len(result.finals)
        self.stats.events_sent += len(messages)
        if result.partial is not None:
            self.stats.partials += 1
            messages += self._transcribe_partial(result.partial)
        return messages

    def _report_if_slow(self, utterance, spent: dict) -> None:
        """Name the stage that stalled the connection, while it is still known.

        Without this a slow sentence is invisible on the server and shows up
        on the client only as VAD events arriving late - which says the
        connection stalled but not on what.
        """
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

    def _transcribe_partial(self, partial) -> list[str]:
        """The grey running text, replaced by the next one a second later.

        No speaker label goes out with it. Identity can wait for the final:
        showing a name and then correcting it reads worse than showing none.
        The language cannot wait, because it changes the text itself.

        Timed separately from the committed sentences, and it has to be. This
        runs every 600 ms on the *whole* open utterance, so a seven-second
        sentence is decoded eleven times at growing lengths - far more work
        than the one final decode. Measured only on finals, a ten-minute run
        reported "slowest sentence 0.4 s" while the connection stalled for
        eleven seconds inside this function.
        """
        if self.transcriber is None:
            return []
        # Only the tail of a long sentence. See PARTIAL_WINDOW_SECONDS: the
        # full window cost more than every committed sentence put together.
        partial = partial.tail(PARTIAL_WINDOW_SECONDS)
        spent: dict[str, float] = {}
        lang_code = ""
        if self.language_identifier is not None:
            # A partial is a fragment of a sentence and shorter still than the
            # utterances the LID already struggles with, so the fallback below
            # matters more here, not less.
            with self._timed("partial_language", spent):
                decision = self.language_identifier.identify(partial.pcm)
            lang_code = self._language_for(decision)
        with self._timed("partial_asr", spent):
            transcript = self.transcriber.transcribe(partial.pcm, lang_code,
                                                     is_final=False)
        self._report_if_partial_slow(partial, spent)
        if not transcript.has_text:
            return []
        self.stats.transcripts += 1
        return [make_partial("", transcript.lang_code, transcript.text)]

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

        Used by both ``bye`` and ``finish``, and it has to be, because the
        socket closes as soon as ``bye`` is answered. The last sentence of a
        meeting is committed here and queued for translation here; if the
        answer is collected any later there is nowhere left to send it. On the
        run that found this, the final sentence arrived and its translation
        never did - the connection was already shut.

        Stopping the worker waits for the sentence in flight (about 0.2 s) and
        accounts for anything still queued, so every sentence gets an answer
        even if the answer is "the meeting ended first".
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
                self.buffer.flush(FinalizeReason.END_OF_STREAM)
            )
        return messages

    # -- helpers ------------------------------------------------------------
    def _fail(self, message: str) -> Response:
        self.stats.protocol_errors += 1
        self.state = SessionState.CLOSED
        log.warning("Session %s rejected: %s", self.session_id or "?", message)
        return Response(messages=[make_error(message)], close=True,
                        close_reason=message)
