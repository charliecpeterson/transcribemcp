from pathlib import Path
from typing import Literal

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

    # Where transcript JSON is written. None = beside the source audio
    # (<audio>.transcript.json). Set OUTPUT_DIR to redirect when the source
    # dir is read-only/shared, or to collect transcripts in one place.
    output_dir: Path | None = None

    @property
    def active_model(self) -> str:
        """Model id the active backend will use, for transcript metadata.

        Backend-specific because each names its model differently; recording
        it lets a transcript carry which model produced it.
        """
        b = self.transcription_backend
        if b == "whisperx":
            return self.whisper_model
        if b == "qwen3_asr":
            return f"qwen3-asr-{self.qwen3_size}"
        if b == "cohere":
            return "cohere-transcribe"
        return "stub"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
