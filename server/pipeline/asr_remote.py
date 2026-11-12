"""The Whisper decoder on the GPU pod, called over HTTP.

The pod may only hold and serve models, so faster-whisper runs there behind
``gpu_server/whisper_server.py`` and the whole streaming policy in
:mod:`server.pipeline.asr` runs here. This class is the seam: it satisfies
the same :class:`~server.pipeline.asr.Decoder` protocol the in-process
``WhisperDecoder`` does, sending the very options that one passed to
faster-whisper, so the policy cannot tell the two apart.

Audio goes as PCM16. The pipeline's samples are PCM16 divided by 32768, so
turning them back is exact and the pod decodes the same numbers it would
have decoded in-process, at half the bytes of float32.

One persistent connection, opened again once if it drops: a partial decode
every 600 ms over a fresh TCP (and possibly tunnel) handshake each time
would add a round trip to every one of them.
"""

from __future__ import annotations

import http.client
import json
import logging
import time
import urllib.parse
from typing import Optional

import numpy as np

from server.config import (
    ASR_BEAM_SIZE_PARTIAL,
    ASR_CONDITION_ON_PREVIOUS,
    ASR_LOG_PROB_THRESHOLD,
    ASR_MAX_COMPRESSION_RATIO,
    ASR_MODEL,
    ASR_NO_SPEECH_THRESHOLD,
    ASR_SERVER_URL,
    ASR_TIMEOUT_S,
    SAMPLE_RATE,
)
from server.pipeline.asr import AsrError, Piece, Word

log = logging.getLogger(__name__)

#: The whisper_server.py API this client speaks.
API_VERSION = 1


def samples_to_pcm(samples: np.ndarray) -> bytes:
    """float32 in [-1, 1) back to PCM16LE. Exact for samples that came from
    PCM16, which is every sample the pipeline decodes."""
    scaled = np.rint(np.asarray(samples, dtype=np.float64) * 32768.0)
    return np.clip(scaled, -32768, 32767).astype("<i2").tobytes()


def decode_query(lang_code: str, beam_size: int,
                 prompt: Optional[str]) -> dict[str, str]:
    """The options faster-whisper was given in-process, as a query string."""
    query = {
        "beam_size": str(beam_size),
        "temperature": "0",
        "condition_on_previous_text": "1" if ASR_CONDITION_ON_PREVIOUS else "0",
        "no_speech_threshold": repr(ASR_NO_SPEECH_THRESHOLD),
        "log_prob_threshold": repr(ASR_LOG_PROB_THRESHOLD),
        "compression_ratio_threshold": repr(ASR_MAX_COMPRESSION_RATIO),
    }
    if lang_code:
        query["language"] = lang_code
    if prompt:
        query["prompt"] = prompt
    return query


def pieces_from_answer(payload: dict) -> tuple[list[Piece], str]:
    """The server's JSON as the Pieces the in-process decoder built."""
    try:
        pieces = [
            Piece(
                text=str(segment["text"]),
                avg_logprob=float(segment["avg_logprob"]),
                no_speech_prob=float(segment["no_speech_prob"]),
                compression_ratio=float(segment["compression_ratio"]),
                start=float(segment["start"]),
                end=float(segment["end"]),
                words=tuple(
                    Word(
                        text=str(word["text"]),
                        start=float(word["start"]),
                        end=float(word["end"]),
                        probability=float(word.get("probability", 1.0)),
                    )
                    for word in segment.get("words") or ()
                    # The in-process decoder dropped blank words too.
                    if str(word["text"]).strip()
                ),
            )
            for segment in payload["segments"]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise AsrError(f"Whisper server answered in an unknown shape: {exc}") \
            from exc
    return pieces, str(payload.get("language") or "")


class RemoteWhisperDecoder:
    """faster-whisper on the GPU pod, behind whisper_server.py."""

    def __init__(self, base_url: str = "", model_id: str = "",
                 timeout: float = ASR_TIMEOUT_S,
                 connection_factory=None) -> None:
        self.base_url = (base_url or ASR_SERVER_URL).rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise AsrError(f"ASR_SERVER_URL is not an http URL: {self.base_url!r}")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._prefix = parsed.path.rstrip("/")
        self.timeout = timeout
        self._connection_factory = connection_factory or self._connect
        self._connection: Optional[http.client.HTTPConnection] = None

        health = self.health()
        if health.get("api_version") != API_VERSION:
            raise AsrError(
                f"Whisper server at {self.base_url} speaks API version "
                f"{health.get('api_version')!r}, this pipeline speaks "
                f"{API_VERSION}; update gpu_server/whisper_server.py on the pod")
        self.model_id = str(health.get("model", ""))
        self.device = str(health.get("device", ""))
        self.compute_type = str(health.get("compute_type", ""))
        wanted = model_id or ASR_MODEL
        if wanted and self.model_id != wanted:
            # A quietly substituted checkpoint is a difference nobody sees in
            # the logs and everybody sees in the transcripts.
            raise AsrError(
                f"Whisper server at {self.base_url} serves {self.model_id!r}, "
                f"not ASR_MODEL={wanted!r}. Start it with --model {wanted} or "
                "set ASR_MODEL to what it serves.")

        started = time.perf_counter()
        self.warmup()
        log.info("Whisper ready at %s: %s on %s (%s), first round trip %.0f ms",
                 self.base_url, self.model_id, self.device, self.compute_type,
                 (time.perf_counter() - started) * 1000)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _connect(self) -> http.client.HTTPConnection:
        kind = (http.client.HTTPSConnection if self._scheme == "https"
                else http.client.HTTPConnection)
        return kind(self._host, self._port, timeout=self.timeout)

    def _close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except OSError:                          # pragma: no cover
                pass
        self._connection = None

    def _request(self, method: str, path: str, body: Optional[bytes] = None,
                 headers: Optional[dict[str, str]] = None) -> dict:
        """One request, on a fresh connection once if the kept one dropped."""
        last: Optional[Exception] = None
        for _attempt in range(2):
            reused = self._connection is not None
            if not reused:
                self._connection = self._connection_factory()
            try:
                self._connection.request(method, self._prefix + path,
                                         body=body, headers=headers or {})
                response = self._connection.getresponse()
                raw = response.read()
                status = response.status
            except (OSError, http.client.HTTPException) as exc:
                # A kept-alive connection the server or a tunnel closed shows
                # up here on first use: once more, on a new one. A new one
                # failing is the server being gone - or a timeout, which a
                # retry would only double on the thread that reads audio.
                self._close()
                last = exc
                if reused and not isinstance(exc, TimeoutError):
                    continue
                break
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError as exc:
                self._close()
                raise AsrError(
                    f"Whisper server answered {status} with non-JSON: "
                    f"{raw[:200]!r}") from exc
            if status != 200:
                raise AsrError(
                    f"Whisper server answered {status}: "
                    f"{payload.get('error', payload)}")
            return payload
        raise AsrError(f"No Whisper server at {self.base_url}: {last}. Start "
                       "gpu_server/whisper_server.py on the pod and check the "
                       "port-forward.") from last

    def health(self) -> dict:
        return self._request("GET", "/health")

    # ------------------------------------------------------------------
    # Decoder protocol
    # ------------------------------------------------------------------

    def warmup(self, seconds: float = 1.0) -> None:
        self.decode(np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32),
                    "", ASR_BEAM_SIZE_PARTIAL)

    def decode(self, samples: np.ndarray, lang_code: str, beam_size: int,
               prompt: Optional[str] = None) -> tuple[list[Piece], str]:
        query = urllib.parse.urlencode(decode_query(lang_code, beam_size,
                                                    prompt))
        payload = self._request(
            "POST", f"/v1/decode?{query}", body=samples_to_pcm(samples),
            headers={"Content-Type": "application/octet-stream"})
        return pieces_from_answer(payload)

    @property
    def source(self) -> str:
        return (f"whisper {self.model_id} at {self.base_url} on "
                f"{self.device} ({self.compute_type})")
