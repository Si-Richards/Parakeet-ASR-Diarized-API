from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import ORJSONResponse

from app.core.config import settings
from app.core.logging import setup_logging
from app.schemas import HealthResponse, TranscribeResponse, Segment, WordTS
from app.services.audio import ensure_dir, ffmpeg_to_wav_16k_mono, probe_duration_seconds, safe_unlink
from app.services.parakeet_asr import ParakeetASR
from app.services.diarization import Diarizer
from app.services.align import assign_speakers_to_words, words_to_speaker_segments
from app.services.pii import redact_basic

log = logging.getLogger("api")

app = FastAPI(title=settings.app_name, default_response_class=ORJSONResponse)

# Simple GPU safety: semaphore to avoid VRAM thrash
import asyncio
_sem = asyncio.Semaphore(settings.max_concurrent_requests)

asr = ParakeetASR(
    model_name=settings.parakeet_model,
    device=settings.device,
    nemo_cache_dir=settings.nemo_cache_dir,
)

diar = Diarizer(
    model_name=settings.pyannote_model,
    device=settings.device,
    hf_token=settings.hf_token,
    hf_home=settings.hf_home,
)


@app.on_event("startup")
def _startup() -> None:
    setup_logging(settings.log_level)
    ensure_dir(settings.tmp_dir)
    ensure_dir(settings.nemo_cache_dir)
    ensure_dir(settings.hf_home)

    # Load ASR always
    asr.load()

    # Load diarization if token provided (otherwise service still runs, diarization disabled)
    if settings.hf_token:
        try:
            diar.load()
        except Exception as e:
            log.warning("Diarization not available: %s", e)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=settings.device,
        parakeet_model=settings.parakeet_model,
        diarization_model=settings.pyannote_model,
        diarization_ready=diar.ready,
        word_timestamps_default=settings.enable_word_timestamps,
    )


def _enforce_upload_size(upload: UploadFile) -> None:
    # Best-effort: content-length not always set. Still helpful.
    # You can also enforce this at reverse proxy (nginx) level.
    max_bytes = settings.max_upload_mb * 1024 * 1024
    cl = upload.headers.get("content-length")
    if cl:
        try:
            if int(cl) > max_bytes:
                raise HTTPException(status_code=413, detail=f"File too large (max {settings.max_upload_mb}MB)")
        except ValueError:
            pass


@app.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(default=True),
    timestamps: bool = Form(default=True),
    min_speakers: int | None = Form(default=None),
    max_speakers: int | None = Form(default=None),
    redact_pii: bool = Form(default=settings.pii_redaction_default),
):
    _enforce_upload_size(file)

    req_id = str(uuid.uuid4())[:8]
    src_path = str(Path(settings.tmp_dir) / f"{req_id}-{file.filename}")
    wav_path = str(Path(settings.tmp_dir) / f"{req_id}.wav")

    async with _sem:
        try:
            # Save upload
            content = await file.read()
            max_bytes = settings.max_upload_mb * 1024 * 1024
            if len(content) > max_bytes:
                raise HTTPException(status_code=413, detail=f"File too large (max {settings.max_upload_mb}MB)")
            with open(src_path, "wb") as f:
                f.write(content)

            # Convert to 16k mono WAV
            ffmpeg_to_wav_16k_mono(src_path, wav_path)
            duration = probe_duration_seconds(wav_path)

            # ASR
            ts_enabled = bool(timestamps and settings.enable_word_timestamps)
            asr_out = asr.transcribe(wav_path, timestamps=ts_enabled)

            # Optional diarization + alignment
            segments_out: list[Segment] = []
            words_out: list[WordTS] | None = None
            diar_used = False
            diar_model_used = None

            text = asr_out.text

            if ts_enabled and asr_out.words:
                # start with word timestamps
                words = [dict(x) for x in asr_out.words]
            else:
                words = []

            if diarize:
                if not diar.ready:
                    raise HTTPException(
                        status_code=400,
                        detail="Diarization requested but not available. Set HF_TOKEN and accept pyannote model terms.",
                    )

                turns = diar.diarize(wav_path, min_speakers=min_speakers, max_speakers=max_speakers)
                turns_dict = [{"start": t.start, "end": t.end, "speaker": t.speaker} for t in turns]

                if words:
                    words = assign_speakers_to_words(words, turns_dict)
                    spk_segments = words_to_speaker_segments(words)
                    segments_out = [
                        Segment(start=s.start, end=s.end, speaker=s.speaker, text=s.text) for s in spk_segments
                    ]
                    words_out = [WordTS(**w) for w in words]
                else:
                    # No word timestamps: return diarization turns as empty text segments
                    segments_out = [
                        Segment(start=t["start"], end=t["end"], speaker=t["speaker"], text="") for t in turns_dict
                    ]
                    words_out = None

                diar_used = True
                diar_model_used = settings.pyannote_model

            else:
                # no diarization: provide parakeet segment timestamps if present
                if asr_out.segments:
                    segments_out = [
                        Segment(start=s["start"], end=s["end"], speaker=None, text=s["segment"]) for s in asr_out.segments
                    ]
                words_out = [WordTS(**w) for w in words] if words else None

            redacted = False
            if redact_pii and text:
                text = redact_basic(text)
                redacted = True
                if segments_out:
                    for i in range(len(segments_out)):
                        segments_out[i].text = redact_basic(segments_out[i].text)

            return TranscribeResponse(
                text=text,
                segments=segments_out,
                words=words_out,
                diarization=diar_used,
                model=settings.parakeet_model,
                diarization_model=diar_model_used,
                duration_sec=duration,
                redacted=redacted,
            )

        except HTTPException:
            raise
        except Exception as e:
            log.exception("transcribe failed (req_id=%s): %s", req_id, e)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            safe_unlink(src_path)
            safe_unlink(wav_path)
