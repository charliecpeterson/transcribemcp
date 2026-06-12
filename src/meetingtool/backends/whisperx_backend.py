"""WhisperX / faster-whisper backend.

Lazy imports — heavy ML deps are only loaded when this backend is selected
and a transcription job is actually run.

Pure ASR: returns segments with absolute timestamps. Diarization is
orchestrated by jobs.py (runs once on the full file after all windows
finish) and applied to persisted chunks, so this file never calls pyannote.
"""
from __future__ import annotations

import logging
import platform

from ..config import get_settings
from .base import ProgressCB, Segment, TranscriptionResult

logger = logging.getLogger(__name__)


def _pick_device(requested: str) -> tuple[str, str]:
    """Return (device, compute_type). faster-whisper int8 works everywhere."""
    settings = get_settings()
    compute = settings.whisperx_compute_type
    if requested != "auto":
        return requested, compute

    # Auto-detect.
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", compute if compute != "int8" else "float16"
    except Exception:
        pass
    # faster-whisper does not support MPS. CPU int8 is the right Mac default.
    return "cpu", "int8" if platform.machine() in ("arm64", "aarch64") else compute


class WhisperXBackend:
    name = "whisperx"

    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        settings = get_settings()
        device, compute = _pick_device(settings.whisperx_device)
        logger.info("loading faster-whisper: model=%s device=%s compute=%s",
                    settings.whisper_model, device, compute)
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            settings.whisper_model,
            device=device,
            compute_type=compute,
        )
        return self._model

    def transcribe(
        self,
        audio_path: str,
        *,
        progress: ProgressCB,
        window: tuple[float, float] | None = None,
    ) -> TranscriptionResult:
        progress("loading_model", 0.05)
        model = self._load_model()

        progress("asr", 0.15)
        # faster-whisper clips the input to [start, end] — segment timestamps
        # it emits are already absolute to the file, not relative to the
        # clip, so we don't need to offset.
        kwargs: dict = {"beam_size": 5}
        if window is not None:
            kwargs["clip_timestamps"] = list(window)
        segments_iter, info = model.transcribe(audio_path, **kwargs)

        segments: list[Segment] = []
        for s in segments_iter:
            segments.append(Segment(start=s.start, end=s.end, text=s.text.strip()))

        progress("asr", 0.85)
        return TranscriptionResult(
            segments=segments,
            language=info.language,
            duration=info.duration,
            backend_name=self.name,
        )
