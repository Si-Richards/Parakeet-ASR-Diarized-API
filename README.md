# ASR Parakeet + pyannote diarization API

FastAPI webservice that:
- transcribes audio with NVIDIA Parakeet TDT 0.6B v3 (NeMo)
- optionally diarizes speakers using pyannote speaker-diarization-3.1
- aligns Parakeet word timestamps to diarization turns to produce speaker-labelled segments

## Requirements

- NVIDIA GPU + nvidia-container-toolkit on the host (for CUDA)
- Docker + Compose

Model notes:
- Parakeet is loaded via NeMo `ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")`.
- pyannote diarization typically requires accepting model conditions + providing a HF token.

## Run

```bash
cp .env.example .env
# edit .env and set HF_TOKEN
docker compose up -d --build

API will be at: http://localhost:8010

Endpoints
Health

GET /health

Transcribe

POST /v1/transcribe (multipart/form-data)

Fields:

file: audio file

diarize (bool, default true)

timestamps (bool, default true)

min_speakers / max_speakers (optional)

redact_pii (bool, default false): basic masking of emails + phone numbers in returned text

Example:

curl -s -X POST "http://localhost:8010/v1/transcribe" \
  -F "file=@./sample.wav" \
  -F "diarize=true" \
  -F "timestamps=true" \
  -F "redact_pii=false" | jq

Output

text: full transcript

segments: speaker-labelled merged segments (when diarize=true and timestamps=true)

words: per-word timestamps (+ speaker when diarized)

Operational notes

This service uses a semaphore (MAX_CONCURRENT_REQUESTS) to avoid GPU VRAM thrash.

For large files, enforce limits in your reverse proxy as well (e.g. nginx client_max_body_size).


---

### What you’ll likely want next (based on your “summary service” style)

1) **Return format parity**: if your summary API already returns `request_id`, `processing_ms`, etc., I can mirror that exactly.  
2) **Queueing**: if you want >1 inbound concurrency but only 1 GPU worker, we can add an internal queue (or Redis/Celery) so callers don’t blow VRAM.  
3) **Telephony defaults**: if your inputs are mostly 8kHz mono, we can optimize conversion and chunking for long calls.

If you paste your current “summary webservice” folder structure (or just its `main.py` + config) I’ll reshape the above to match it *exactly* (routes, response schema, logging, folders).
::contentReference[oaicite:4]{index=4}