import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import nemo.collections.asr as nemo_asr

log = logging.getLogger("parakeet")


@dataclass
class ParakeetResult:
    text: str
    words: list[dict[str, Any]] | None  # [{"start":..., "end":..., "word":...}]
    segments: list[dict[str, Any]] | None  # [{"start":..., "end":..., "segment":...}]


class ParakeetASR:
    def __init__(self, model_name: str, device: str = "cuda", nemo_cache_dir: str = "/models/nemo"):
        self.model_name = model_name
        self.device = device
        self.nemo_cache_dir = nemo_cache_dir
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return

        os.environ.setdefault("NEMO_CACHE_DIR", self.nemo_cache_dir)
        log.info("Loading Parakeet model: %s (device=%s)", self.model_name, self.device)

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
        else:
            model = model.to(torch.device("cpu"))

        model.eval()
        self._model = model
        log.info("Parakeet loaded.")

    def transcribe(self, wav_path: str, timestamps: bool = True) -> ParakeetResult:
        if self._model is None:
            raise RuntimeError("ASR model not loaded")

        # NeMo expects list of paths
        output = self._model.transcribe([wav_path], timestamps=timestamps)

        # output[0] is a Hypothesis-like object (text + optional timestamp dict)
        hyp = output[0]
        text = getattr(hyp, "text", "") or ""

        words = None
        segments = None
        if timestamps and hasattr(hyp, "timestamp") and isinstance(hyp.timestamp, dict):
            # per NVIDIA example: hyp.timestamp['word'] and ['segment'] exist when timestamps=True
            # :contentReference[oaicite:3]{index=3}
            w = hyp.timestamp.get("word")
            s = hyp.timestamp.get("segment")
            if isinstance(w, list):
                words = [{"start": float(x["start"]), "end": float(x["end"]), "word": str(x["word"])} for x in w]
            if isinstance(s, list):
                segments = [{"start": float(x["start"]), "end": float(x["end"]), "segment": str(x["segment"])} for x in s]

        return ParakeetResult(text=text, words=words, segments=segments)
