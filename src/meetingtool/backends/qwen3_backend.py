"""Qwen3-ASR backend. Two modes selected by QWEN3_MODE:

- local: load model via transformers (CUDA/CPU) or MLX on Apple Silicon
- vllm : POST audio to an OpenAI-compatible /v1/audio/transcriptions endpoint

Pure ASR: diarization is orchestrated by jobs.py. Qwen3-ASR doesn't emit
speaker labels anyway.
"""
from __future__ import annotations

import logging
import platform
import subprocess
import tempfile
from pathlib import Path

import httpx

from ..audio import DEFAULT_SR, decode_pcm
from ..config import get_settings
from .base import ProgressCB, Segment, TranscriptionResult

logger = logging.getLogger(__name__)

_MODEL_IDS = {
    "0.6B": "Qwen/Qwen3-ASR-0.6B",
    "1.7B": "Qwen/Qwen3-ASR-1.7B",
}


class Qwen3Backend:
    name = "qwen3_asr"

    def __init__(self) -> None:
        self._pipe = None  # transformers pipeline, cached after first load

    # ---- mode dispatch ----------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        *,
        progress: ProgressCB,
        window: tuple[float, float] | None = None,
    ) -> TranscriptionResult:
        settings = get_settings()
        if settings.qwen3_mode == "vllm":
            segments, duration, language = self._transcribe_vllm(
                audio_path, progress, window,
            )
        else:
            segments, duration, language = self._transcribe_local(
                audio_path, progress, window,
            )

        return TranscriptionResult(
            segments=segments,
            language=language,
            duration=duration,
            backend_name=self.name,
        )

    # ---- local ------------------------------------------------------------

    def _transcribe_local(
        self,
        audio_path: str,
        progress: ProgressCB,
        window: tuple[float, float] | None,
    ) -> tuple[list[Segment], float, str | None]:
        progress("loading_model", 0.05)
        settings = get_settings()
        model_id = _MODEL_IDS[settings.qwen3_size]

        if self._pipe is None:
            self._pipe = self._build_local_pipeline(model_id)

        progress("asr", 0.15)
        # The HF pipeline accepts numpy arrays directly, which is how we do
        # windowing without writing temp files.
        if window is None:
            pipe_input = audio_path
            offset = 0.0
        else:
            waveform = decode_pcm(audio_path, sample_rate=DEFAULT_SR)
            ws, we = window
            start_i = int(ws * DEFAULT_SR)
            end_i = int(we * DEFAULT_SR)
            pipe_input = waveform[start_i:end_i].numpy()
            offset = ws

        # return_timestamps=True gives chunk-level timestamps; the HF pipeline
        # returns dicts with {"text", "chunks": [{"text", "timestamp": (start, end)}]}.
        result = self._pipe(pipe_input, return_timestamps=True)

        segments: list[Segment] = []
        duration = 0.0
        if isinstance(result, dict) and "chunks" in result:
            for chunk in result["chunks"]:
                start, end = chunk.get("timestamp") or (0.0, 0.0)
                text = (chunk.get("text") or "").strip()
                if text and start is not None and end is not None:
                    segments.append(Segment(
                        start=float(start) + offset,
                        end=float(end) + offset,
                        text=text,
                    ))
                    duration = max(duration, float(end) + offset)
        else:
            # Fallback: single blob with no timestamps.
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            if text.strip():
                segments.append(Segment(
                    start=offset, end=offset, text=text.strip(),
                ))

        progress("asr", 0.85)
        return segments, duration, None

    def _build_local_pipeline(self, model_id: str):
        import torch
        from transformers import pipeline

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() and platform.system() == "Darwin"
            else "cpu"
        )
        logger.info("loading Qwen3-ASR: model=%s device=%s", model_id, device)
        return pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )

    # ---- vLLM -------------------------------------------------------------

    def _transcribe_vllm(
        self,
        audio_path: str,
        progress: ProgressCB,
        window: tuple[float, float] | None,
    ) -> tuple[list[Segment], float, str | None]:
        progress("loading_model", 0.05)
        settings = get_settings()
        if not settings.qwen3_vllm_url:
            raise RuntimeError("QWEN3_MODE=vllm but QWEN3_VLLM_URL is not set")

        url = settings.qwen3_vllm_url.rstrip("/") + "/audio/transcriptions"
        headers = {}
        if settings.qwen3_vllm_api_key:
            headers["Authorization"] = f"Bearer {settings.qwen3_vllm_api_key}"

        model_id = _MODEL_IDS[settings.qwen3_size]
        progress("asr", 0.15)

        source_path, cleanup = self._prepare_upload(audio_path, window)
        offset = window[0] if window is not None else 0.0
        try:
            with source_path.open("rb") as f:
                files = {"file": (source_path.name, f, "application/octet-stream")}
                data = {
                    "model": model_id,
                    "response_format": "verbose_json",  # returns segments + timestamps
                    "timestamp_granularities[]": "segment",
                }
                with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
                    resp = client.post(url, headers=headers, files=files, data=data)
        finally:
            cleanup()
        resp.raise_for_status()
        body = resp.json()
        progress("asr", 0.85)

        segments: list[Segment] = []
        for s in body.get("segments") or []:
            segments.append(Segment(
                start=float(s.get("start", 0.0)) + offset,
                end=float(s.get("end", 0.0)) + offset,
                text=(s.get("text") or "").strip(),
            ))

        if not segments and body.get("text"):
            # Endpoint returned no per-segment breakdown; keep as one blob.
            segments.append(Segment(
                start=offset,
                end=offset + float(body.get("duration", 0.0)),
                text=body["text"].strip(),
            ))

        duration = float(body.get("duration", 0.0))
        language = body.get("language")
        return segments, duration, language

    def _prepare_upload(
        self,
        audio_path: str,
        window: tuple[float, float] | None,
    ) -> tuple[Path, "callable[[], None]"]:
        """Return (path_to_upload, cleanup_callback).

        With window=None the original file is uploaded directly. With a
        window set, we use the bundled ffmpeg to slice a mono 16 kHz wav
        into a tempfile and upload that. Cleanup deletes the tempfile.
        """
        if window is None:
            return Path(audio_path), lambda: None

        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        ws, we = window
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="qwen3-win-")
        # Close the fd immediately; ffmpeg will write to the path.
        import os
        os.close(fd)
        subprocess.run(
            [
                ffmpeg, "-nostdin", "-loglevel", "error", "-y",
                "-i", audio_path,
                "-ss", f"{ws:.3f}", "-to", f"{we:.3f}",
                "-ac", "1", "-ar", str(DEFAULT_SR),
                tmp_path,
            ],
            check=True,
        )

        def _cleanup() -> None:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

        return Path(tmp_path), _cleanup
