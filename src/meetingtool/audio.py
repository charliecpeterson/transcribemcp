"""Shared audio decoding utilities.

One place to decode any audio format to mono 16 kHz float32 PCM via the
ffmpeg binary bundled by `imageio-ffmpeg`. Used by:

- diarize.py (pass `{waveform, sample_rate}` dict to pyannote, bypassing
  torchcodec which needs FFmpeg dylibs we don't ship)
- vad.py (silero-vad wants a 1D waveform tensor)
- backends/cohere_backend.py (Cohere Transcribe auto-resamples internally
  but we pre-decode so every backend goes through the same ffmpeg path)

Keeping this in one module avoids per-backend drift in decode settings.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # pragma: no cover

DEFAULT_SR = 16000


def decode_pcm(audio_path: str, sample_rate: int = DEFAULT_SR) -> "torch.Tensor":
    """Decode audio to a 1D mono float32 torch.Tensor at the requested rate.

    Shells out to the `imageio-ffmpeg`-bundled ffmpeg binary so no system
    ffmpeg install is required.
    """
    import subprocess

    import imageio_ffmpeg
    import numpy as np
    import torch

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        proc = subprocess.run(
            [
                ffmpeg, "-nostdin", "-loglevel", "error",
                "-i", audio_path,
                "-f", "f32le", "-ac", "1", "-ar", str(sample_rate),
                "-",
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed to decode {audio_path!r} (exit {e.returncode}): "
            f"{stderr or '<no stderr>'}"
        ) from e
    pcm = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return torch.from_numpy(pcm)


def load_waveform(audio_path: str, sample_rate: int = DEFAULT_SR) -> dict:
    """Return a pyannote-compatible `{waveform, sample_rate}` dict.

    `waveform` is a torch.Tensor of shape (1, T).
    """
    return {
        "waveform": decode_pcm(audio_path, sample_rate).unsqueeze(0),
        "sample_rate": sample_rate,
    }
