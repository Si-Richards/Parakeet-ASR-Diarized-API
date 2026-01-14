from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    text: str


def _midpoint(start: float, end: float) -> float:
    return (start + end) / 2.0


def assign_speakers_to_words(words: list[dict[str, Any]], turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Assign each word to the diarization speaker whose turn contains the word midpoint.
    If no turn matches, speaker=None.
    """
    if not words:
        return []

    # turns: [{"start":..,"end":..,"speaker":..}]
    t_idx = 0
    for w in words:
        mid = _midpoint(w["start"], w["end"])
        # advance pointer
        while t_idx < len(turns) and turns[t_idx]["end"] < mid:
            t_idx += 1
        if t_idx < len(turns) and turns[t_idx]["start"] <= mid <= turns[t_idx]["end"]:
            w["speaker"] = turns[t_idx]["speaker"]
        else:
            w["speaker"] = None
    return words


def words_to_speaker_segments(words: list[dict[str, Any]], max_gap: float = 0.6) -> list[SpeakerSegment]:
    """
    Merge consecutive words into speaker-labelled segments.
    """
    if not words:
        return []

    segments: list[SpeakerSegment] = []
    cur = None

    for w in words:
        spk = w.get("speaker") or "UNKNOWN"
        ws, we = float(w["start"]), float(w["end"])
        token = (w.get("word") or "").strip()

        if cur is None:
            cur = SpeakerSegment(start=ws, end=we, speaker=spk, text=token)
            continue

        gap = ws - cur.end
        if spk == cur.speaker and gap <= max_gap:
            cur.end = max(cur.end, we)
            if token:
                if cur.text and not cur.text.endswith(" "):
                    cur.text += " "
                cur.text += token
        else:
            segments.append(cur)
            cur = SpeakerSegment(start=ws, end=we, speaker=spk, text=token)

    if cur is not None:
        segments.append(cur)

    return segments
