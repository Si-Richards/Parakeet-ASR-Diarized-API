import logging
import os
from dataclasses import dataclass

import torch
from pyannote.audio import Pipeline

log = logging.getLogger("diarization")


@dataclass
class DiarizationTurn:
    start: float
    end: float
    speaker: str


class Diarizer:
    def __init__(self, model_name: str, device: str = "cuda", hf_token: str | None = None, hf_home: str = "/models/hf"):
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token
        self.hf_home = hf_home
        self._pipeline = None

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        if self._pipeline is not None:
            return

        # HF cache location (keeps container restarts fast)
        os.environ.setdefault("HF_HOME", self.hf_home)

        if not self.hf_token:
            raise RuntimeError(
                "HF_TOKEN not set. pyannote diarization models usually require accepting conditions + token."
            )

        log.info("Loading pyannote pipeline: %s (device=%s)", self.model_name, self.device)
        pipeline = Pipeline.from_pretrained(self.model_name, use_auth_token=self.hf_token)

        if self.device == "cuda" and torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        else:
            pipeline.to(torch.device("cpu"))

        self._pipeline = pipeline
        log.info("pyannote loaded.")

    def diarize(self, wav_path: str, min_speakers: int | None = None, max_speakers: int | None = None) -> list[DiarizationTurn]:
        if self._pipeline is None:
            raise RuntimeError("Diarization pipeline not loaded")

        kwargs = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            kwargs["max_speakers"] = int(max_speakers)

        diarization = self._pipeline(wav_path, **kwargs)

        turns: list[DiarizationTurn] = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(DiarizationTurn(start=float(segment.start), end=float(segment.end), speaker=str(speaker)))

        turns.sort(key=lambda t: t.start)
        return turns
