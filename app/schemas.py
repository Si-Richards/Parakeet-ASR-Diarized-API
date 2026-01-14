from pydantic import BaseModel, Field
from typing import Literal


class WordTS(BaseModel):
    start: float
    end: float
    word: str
    speaker: str | None = None


class Segment(BaseModel):
    start: float
    end: float
    speaker: str | None = None
    text: str


class TranscribeResponse(BaseModel):
    text: str
    language: str | None = None
    segments: list[Segment] = Field(default_factory=list)
    words: list[WordTS] | None = None
    diarization: bool = False
    model: str
    diarization_model: str | None = None
    duration_sec: float | None = None
    redacted: bool = False


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    device: str
    parakeet_model: str
    diarization_model: str
    diarization_ready: bool
    word_timestamps_default: bool
