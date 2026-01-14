from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Service
    app_name: str = "asr-parakeet-diarize-api"
    log_level: str = "INFO"

    # Concurrency control (important for GPU stability)
    max_concurrent_requests: int = Field(default=1, ge=1, le=32)

    # Upload limits / temp
    max_upload_mb: int = Field(default=200, ge=1, le=2048)
    tmp_dir: str = Field(default="/tmp/asr")

    # Models
    device: str = Field(default="cuda")  # "cuda" or "cpu"
    parakeet_model: str = Field(default="nvidia/parakeet-tdt-0.6b-v3")
    pyannote_model: str = Field(default="pyannote/speaker-diarization-3.1")

    # Tokens / caches
    hf_token: str | None = Field(default=None)  # required for pyannote gated models
    nemo_cache_dir: str = Field(default="/models/nemo")
    hf_home: str = Field(default="/models/hf")

    # Output controls
    enable_word_timestamps: bool = True

    # Optional basic PII redaction in returned text (OFF by default)
    pii_redaction_default: bool = False


settings = Settings()
