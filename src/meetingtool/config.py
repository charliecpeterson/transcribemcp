from pathlib import Path
from typing import Literal

from platformdirs import user_data_dir
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    transcription_backend: Literal["whisperx", "qwen3_asr", "cohere", "stub"] = "stub"

    whisper_model: str = "base"
    whisperx_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    whisperx_compute_type: Literal["int8", "float16", "float32"] = "int8"
    whisperx_align: bool = False

    qwen3_mode: Literal["local", "vllm"] = "local"
    qwen3_size: Literal["0.6B", "1.7B"] = "0.6B"
    qwen3_vllm_url: str = ""
    qwen3_vllm_api_key: str = ""

    # Cohere Transcribe (LLM-decoded — timestamps come from silero-vad, not the model).
    cohere_backend: Literal["auto", "transformers", "mlx"] = "auto"
    cohere_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    cohere_lang: str = "en"

    # Diarization is opt-in: pyannote needs an HF_TOKEN with the
    # speaker-diarization-community-1 license accepted, and there's no point
    # demanding that for a smoke test against the stub backend.
    diarize: bool = False
    hf_token: str = ""

    meetingtool_data_dir: Path = Field(
        default_factory=lambda: Path(user_data_dir("meetingtool"))
    )

    @property
    def db_path(self) -> Path:
        return self.meetingtool_data_dir / "meetingtool.db"

    @property
    def transcripts_dir(self) -> Path:
        return self.meetingtool_data_dir / "transcripts"

    def ensure_dirs(self) -> None:
        self.meetingtool_data_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
