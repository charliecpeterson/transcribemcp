"""Tests for shared audio decode helpers. Requires tests/fixtures/sample.wav."""
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "sample.wav"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="sample.wav fixture missing (run scripts/fetch_sample_audio.sh)",
)


def test_decode_pcm_returns_1d_float32_tensor():
    torch = pytest.importorskip("torch")
    from transcribemcp.audio import DEFAULT_SR, decode_pcm

    wav = decode_pcm(str(FIXTURE))
    assert isinstance(wav, torch.Tensor)
    assert wav.ndim == 1
    assert wav.dtype == torch.float32
    # Sample clip is ~10 s; at 16 kHz that's ~160k frames. Sanity only.
    assert wav.numel() > DEFAULT_SR // 2


def test_decode_pcm_respects_sample_rate():
    pytest.importorskip("torch")
    from transcribemcp.audio import decode_pcm

    slow = decode_pcm(str(FIXTURE), sample_rate=8000)
    fast = decode_pcm(str(FIXTURE), sample_rate=16000)
    # Roughly 2x frames at the higher rate; allow 10% slack for ffmpeg rounding.
    assert abs(fast.numel() / slow.numel() - 2.0) < 0.2


def test_load_waveform_shape_is_pyannote_compatible():
    torch = pytest.importorskip("torch")
    from transcribemcp.audio import DEFAULT_SR, load_waveform

    out = load_waveform(str(FIXTURE))
    assert set(out.keys()) == {"waveform", "sample_rate"}
    assert out["sample_rate"] == DEFAULT_SR
    assert isinstance(out["waveform"], torch.Tensor)
    # pyannote wants (channels, frames) — mono is (1, T).
    assert out["waveform"].ndim == 2
    assert out["waveform"].shape[0] == 1
