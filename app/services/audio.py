import os
import subprocess
from pathlib import Path


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def ffmpeg_to_wav_16k_mono(src_path: str, dst_path: str) -> None:
    """
    Convert arbitrary audio to 16kHz mono WAV (PCM s16le) for consistent diarization & ASR.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        dst_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {p.stderr.decode('utf-8', errors='ignore')}")


def probe_duration_seconds(path: str) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return None
    try:
        return float(p.stdout.decode().strip())
    except Exception:
        return None


def safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except Exception:
        return
