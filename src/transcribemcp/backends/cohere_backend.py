"""Cohere Transcribe backend.

Cohere Labs' cohere-transcribe-03-2026 is a 2B-parameter LLM-decoded ASR
model. It outputs plain text with no segment timestamps, so we preprocess
audio with silero-vad and infer one utterance per voiced span. Each
`Segment.start/end` comes from the VAD, not the model.

Two runtimes, picked by COHERE_BACKEND:

- ``transformers`` — the reference HF path (CUDA / CPU).
- ``mlx`` — the mlx-audio port for Apple Silicon. Much faster than the
  transformers path on an M-series Mac.

English-only for now (COHERE_LANG=en). The model itself supports 14
languages; we'll expose that once the single-language path is validated.

Pure ASR: diarization is orchestrated by jobs.py and applied to persisted
chunks after all windows finish.
"""
from __future__ import annotations

import logging
import platform

from ..audio import DEFAULT_SR, decode_pcm
from ..config import get_settings
from ..vad import VoicedSpan, detect_voiced
from .base import ProgressCB, Segment, TranscriptionResult

logger = logging.getLogger(__name__)

_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


class CohereBackend:
    name = "cohere"

    def __init__(self) -> None:
        self._model = None       # transformers model OR mlx handle
        self._processor = None   # transformers processor (None in mlx mode)
        self._runtime: str | None = None  # "transformers" | "mlx"

    def transcribe(
        self,
        audio_path: str,
        *,
        progress: ProgressCB,
        window: tuple[float, float] | None = None,
    ) -> TranscriptionResult:
        settings = get_settings()

        progress("loading_model", 0.05)
        self._ensure_loaded()

        progress("vad", 0.15)
        spans = detect_voiced(audio_path)
        if window is not None:
            ws, we = window
            spans = [s for s in spans if s.end > ws and s.start < we]
        if not spans:
            progress("writing", 0.95)
            return TranscriptionResult(
                segments=[], language=settings.cohere_lang,
                duration=0.0, backend_name=self.name,
            )

        progress("asr", 0.25)
        waveform = decode_pcm(audio_path, sample_rate=DEFAULT_SR)
        total = float(waveform.numel()) / DEFAULT_SR

        segments: list[Segment] = []
        n = len(spans)
        for i, span in enumerate(spans):
            text = self._transcribe_span(waveform, span, lang=settings.cohere_lang)
            text = text.strip()
            if text:
                segments.append(Segment(start=span.start, end=span.end, text=text))
            # 0.25 → 0.9 over the ASR loop
            progress("asr", 0.25 + 0.65 * ((i + 1) / n))

        return TranscriptionResult(
            segments=segments,
            language=settings.cohere_lang,
            duration=total,
            backend_name=self.name,
        )

    # ---- model loading ----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._runtime is not None:
            return
        runtime = self._resolve_runtime()
        if runtime == "mlx":
            self._load_mlx()
        else:
            self._load_transformers()
        self._runtime = runtime

    def _resolve_runtime(self) -> str:
        configured = get_settings().cohere_backend
        if configured != "auto":
            return configured
        # On Apple Silicon, prefer mlx if importable; else transformers.
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                import mlx_audio  # noqa: F401
                return "mlx"
            except ImportError:
                pass
        return "transformers"

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        token = get_settings().hf_token or None
        if not token:
            raise RuntimeError(
                "Cohere Transcribe is a gated HF repo. Set HF_TOKEN and accept the "
                "license at https://huggingface.co/CohereLabs/cohere-transcribe-03-2026"
            )
        device = self._resolve_torch_device()
        logger.info("loading Cohere Transcribe (transformers): device=%s", device)
        self._processor = AutoProcessor.from_pretrained(_MODEL_ID, token=token)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            token=token,
        ).to(device)
        self._model.eval()

    def _load_mlx(self) -> None:
        from mlx_audio.stt import load_model  # type: ignore
        logger.info("loading Cohere Transcribe (mlx)")
        # mlx-audio's load_model resolves gated repos via the ambient
        # huggingface_hub credentials (HF_TOKEN env var or ~/.huggingface).
        self._model = load_model(_MODEL_ID)

    def _resolve_torch_device(self) -> str:
        configured = get_settings().cohere_device
        if configured != "auto":
            return configured
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ---- inference --------------------------------------------------------

    def _transcribe_span(self, waveform, span: VoicedSpan, *, lang: str) -> str:
        """Run the model on a single VAD span. Returns the decoded text."""
        start = int(span.start * DEFAULT_SR)
        end = int(span.end * DEFAULT_SR)
        slice_ = waveform[start:end]
        if self._runtime == "mlx":
            return self._infer_mlx(slice_, lang=lang)
        return self._infer_transformers(slice_, lang=lang)

    def _infer_transformers(self, slice_, *, lang: str) -> str:
        import torch

        inputs = self._processor(
            slice_.numpy(), sampling_rate=DEFAULT_SR, return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                language=lang,
                task="transcribe",
            )
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text

    def _infer_mlx(self, slice_, *, lang: str) -> str:
        # mlx-audio's Cohere handle is kwargs-only and batch-oriented. It returns
        # a list[str], one entry per audio_arrays item. Future optimization:
        # batch every VAD span in a single call via audio_arrays=[...].
        out = self._model.transcribe(
            language=lang,
            audio_arrays=[slice_.numpy()],
            sample_rates=[DEFAULT_SR],
        )
        return out[0] if out else ""
