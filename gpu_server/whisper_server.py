"""Whisper decode server - the only project file that runs on the GPU pod.

MUST RUN ON: the GPU Server pod, AFTER vLLM has started (see README.md here).

The pod may hold models and serve them, nothing more. The meeting pipeline -
VAD, sentence boundaries, speakers, language ID, the streaming policy that
turns repeated decodes into stable text, the hallucination guards - runs on
the Windows machine and calls this over HTTP. This file therefore holds no
policy at all: it loads faster-whisper, decodes the audio it is sent with
the options it is sent, and returns every segment and word with its scores.
Every decision about what those scores mean is made by the caller.

That is also why it is self-contained. It imports nothing from ``server/``
or ``common/``: copy this one file to the pod, install
``gpu_server/requirements.txt``, and run it.

Why not vLLM's own transcription endpoint
-----------------------------------------
The streaming policy needs word timestamps (to commit words two decodes
agree on), ``avg_logprob`` / ``no_speech_prob`` / ``compression_ratio`` per
segment (to refuse invented sentences) and a beam search for the final
decode. vLLM's ``/v1/audio/transcriptions`` decodes greedily and returns no
word timestamps. Decoding here with the exact call the pipeline used to make
in-process keeps its output identical to what it was.

API
---
``GET /health``
    ``{"status": "ok", "api_version": 1, "model": ..., "device": ...,
    "compute_type": ..., "decodes": n}``

``POST /v1/decode?language=ja&beam_size=5&...``
    Body: raw PCM16 little-endian, mono, 16 kHz. The query string carries the
    decode options (see :class:`DecodeOptions`). Answers with the language
    faster-whisper settled on and every segment::

        {"language": "ja", "language_probability": 0.98,
         "decode_seconds": 0.21,
         "segments": [{"text": ..., "start": 0.0, "end": 1.9,
                       "avg_logprob": -0.21, "no_speech_prob": 0.01,
                       "compression_ratio": 1.3,
                       "words": [{"text": ..., "start": ..., "end": ...,
                                  "probability": ...}]}]}

Usage
-----
    python3.11 whisper_server.py                       # large-v3, cuda, :8002
    python3.11 whisper_server.py --model /models/whisper-large-v3 --port 8002

Bind it to 127.0.0.1 (the default) and reach it through a port-forward or an
SSH tunnel. It has no authentication: anything that can reach the port can
make the GPU decode.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

log = logging.getLogger("whisper_server")

#: The audio contract. Duplicated from common/protocol.py on purpose: this
#: file must run without the rest of the repository.
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2

#: Bumped when the request or the answer changes shape. The caller refuses a
#: server that speaks another version rather than misreading its answers.
API_VERSION = 1

#: Longest clip accepted. The pipeline sends at most a 7 s sentence plus its
#: pre-roll and hangover; Whisper itself reads 30 s.
MAX_SECONDS = 30.0

MAX_BEAM_SIZE = 10


class RequestError(ValueError):
    """The request was malformed. Answered with 400, never with a decode."""


@dataclass(frozen=True)
class DecodeOptions:
    """What the caller controls. Everything else is fixed below."""

    language: Optional[str] = None
    beam_size: int = 1
    prompt: Optional[str] = None
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    no_speech_threshold: Optional[float] = None
    log_prob_threshold: Optional[float] = None
    compression_ratio_threshold: Optional[float] = None

    @classmethod
    def from_query(cls, query: dict[str, str]) -> "DecodeOptions":
        unknown = set(query) - set(cls.__dataclass_fields__)
        if unknown:
            # A misspelt option silently ignored is a decode that differs
            # from the one asked for, with nothing in any log to say so.
            raise RequestError(f"unknown options: {sorted(unknown)}")
        try:
            beam_size = int(query.get("beam_size", "1"))
            options = cls(
                language=query.get("language") or None,
                beam_size=beam_size,
                # None, not "". An empty string is itself a prompt as far as
                # Whisper is concerned.
                prompt=query.get("prompt") or None,
                temperature=float(query.get("temperature", "0")),
                condition_on_previous_text=_boolean(
                    query.get("condition_on_previous_text", "0")),
                no_speech_threshold=_optional_float(
                    query.get("no_speech_threshold")),
                log_prob_threshold=_optional_float(
                    query.get("log_prob_threshold")),
                compression_ratio_threshold=_optional_float(
                    query.get("compression_ratio_threshold")),
            )
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
        if not 1 <= beam_size <= MAX_BEAM_SIZE:
            raise RequestError(f"beam_size must be 1..{MAX_BEAM_SIZE}")
        return options


def _boolean(value: str) -> bool:
    if value.lower() in ("1", "true"):
        return True
    if value.lower() in ("0", "false", ""):
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def pcm_to_samples(body: bytes) -> np.ndarray:
    """PCM16LE bytes to float32 in [-1, 1), exactly as the pipeline did."""
    if not body:
        raise RequestError("empty audio")
    if len(body) % SAMPLE_WIDTH:
        raise RequestError("audio length is not a whole number of samples")
    if len(body) / SAMPLE_WIDTH / SAMPLE_RATE > MAX_SECONDS:
        raise RequestError(f"audio longer than {MAX_SECONDS:.0f} s")
    samples = np.frombuffer(body, dtype="<i2").astype(np.float32)
    samples /= 32768.0
    return samples


class Engine:
    """One faster-whisper model, decoding one clip at a time."""

    def __init__(self, model: Any, model_id: str, device: str,
                 compute_type: str) -> None:
        self.model = model
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self.decodes = 0
        # One GPU decode at a time. The pipeline sends one request at a time
        # anyway; this keeps a second caller from interleaving with it.
        self._lock = threading.Lock()

    def decode(self, samples: np.ndarray, options: DecodeOptions) -> dict:
        started = time.perf_counter()
        with self._lock:
            segments, info = self.model.transcribe(
                samples,
                language=options.language,
                task="transcribe",
                beam_size=options.beam_size,
                temperature=options.temperature,
                initial_prompt=options.prompt,
                condition_on_previous_text=options.condition_on_previous_text,
                no_speech_threshold=options.no_speech_threshold,
                log_prob_threshold=options.log_prob_threshold,
                compression_ratio_threshold=options.compression_ratio_threshold,
                # The caller's streaming policy is built on word timestamps.
                without_timestamps=False,
                word_timestamps=True,
                # The caller has run Silero already; a second VAD here eats
                # the pre-roll that holds the word onsets.
                vad_filter=False,
            )
            # The segments are a generator: the decode happens here, inside
            # the lock, not in transcribe().
            answer = [segment_to_dict(segment) for segment in segments]
            self.decodes += 1
        return {
            "language": str(getattr(info, "language", "") or ""),
            "language_probability": float(
                getattr(info, "language_probability", 0.0) or 0.0),
            "decode_seconds": round(time.perf_counter() - started, 4),
            "segments": answer,
        }

    def health(self) -> dict:
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "model": self.model_id,
            "device": self.device,
            "compute_type": self.compute_type,
            "decodes": self.decodes,
        }


def segment_to_dict(segment: Any) -> dict:
    """Everything the caller's policy reads, and nothing filtered out."""
    return {
        "text": segment.text,
        "start": float(segment.start),
        "end": float(segment.end),
        "avg_logprob": float(segment.avg_logprob),
        "no_speech_prob": float(segment.no_speech_prob),
        "compression_ratio": float(segment.compression_ratio),
        "words": [
            {
                "text": word.word,
                "start": float(word.start),
                "end": float(word.end),
                "probability": float(word.probability),
            }
            for word in (segment.words or ())
        ],
    }


def load_engine(model_id: str, device: str, compute_type: str,
                download_root: Optional[str] = None) -> Engine:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_id, device=device, compute_type=compute_type,
                         download_root=download_root)
    engine = Engine(model, model_id, device, compute_type)
    # Pay the first-decode cost before the first meeting does.
    engine.decode(np.zeros(SAMPLE_RATE, dtype=np.float32),
                  DecodeOptions(beam_size=1))
    log.info("Whisper ready on %s: %s (%s)", device, model_id, compute_type)
    return engine


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="Whisper decode server", version=str(API_VERSION))

    @app.get("/health")
    def health() -> dict:
        return engine.health()

    @app.post("/v1/decode")
    async def decode(request: Request):
        body = await request.body()
        try:
            options = DecodeOptions.from_query(dict(request.query_params))
            samples = pcm_to_samples(body)
        except RequestError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        try:
            return await run_in_threadpool(engine.decode, samples, options)
        except Exception as exc:                     # noqa: BLE001
            log.exception("decode failed")
            return JSONResponse(
                {"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=os.environ.get(
        "WHISPER_MODEL", "large-v3"), help="checkpoint name or local path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--download-root", default=os.environ.get(
        "WHISPER_CACHE_DIR", "models/whisper"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = load_engine(args.model, args.device, args.compute_type,
                         args.download_root)

    import uvicorn
    uvicorn.run(create_app(engine), host=args.host, port=args.port,
                workers=1, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
