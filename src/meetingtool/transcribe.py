"""Backend router. Reads TRANSCRIPTION_BACKEND from config and returns a transcribe fn."""
from __future__ import annotations

from typing import Callable

from .backends.base import ProgressCB, TranscriptionResult
from .config import get_settings


def get_backend_fn() -> Callable[..., TranscriptionResult]:
    """Return a callable (audio_path, *, progress, window=None) -> TranscriptionResult.

    Backend instance is constructed lazily so heavy ML imports don't happen at
    server startup — only when the first transcription job runs. The returned
    function pins a single backend instance so its model weights are cached
    across windows within a job (and across jobs within a process).
    """
    settings = get_settings()
    backend_name = settings.transcription_backend

    if backend_name == "stub":
        from .backends.stub import StubBackend
        backend = StubBackend()
    elif backend_name == "whisperx":
        from .backends.whisperx_backend import WhisperXBackend
        backend = WhisperXBackend()
    elif backend_name == "qwen3_asr":
        from .backends.qwen3_backend import Qwen3Backend
        backend = Qwen3Backend()
    elif backend_name == "cohere":
        from .backends.cohere_backend import CohereBackend
        backend = CohereBackend()
    else:
        raise ValueError(f"unknown TRANSCRIPTION_BACKEND: {backend_name}")

    def _fn(
        audio_path: str,
        *,
        progress: ProgressCB,
        window: tuple[float, float] | None = None,
    ) -> TranscriptionResult:
        return backend.transcribe(audio_path, progress=progress, window=window)

    return _fn
