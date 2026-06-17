"""Unit test for the Qwen3 vLLM HTTP mode. No ML deps needed — mocks the endpoint."""
from unittest.mock import patch

import httpx
import pytest

from transcribemcp import config as config_mod
from transcribemcp.backends.qwen3_backend import Qwen3Backend


@pytest.fixture
def vllm_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "qwen3_asr")
    monkeypatch.setenv("QWEN3_MODE", "vllm")
    monkeypatch.setenv("QWEN3_VLLM_URL", "http://mock-vllm.invalid/v1")
    monkeypatch.setenv("QWEN3_VLLM_API_KEY", "secret-123")
    monkeypatch.setenv("DIARIZE", "false")
    # Force settings reload
    config_mod._settings = None
    yield
    config_mod._settings = None


def _mock_response(request: httpx.Request) -> httpx.Response:
    assert request.url.path.endswith("/v1/audio/transcriptions")
    assert request.headers.get("Authorization") == "Bearer secret-123"
    return httpx.Response(
        200,
        json={
            "text": "full text here",
            "language": "en",
            "duration": 12.5,
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "hello world"},
                {"start": 5.0, "end": 12.5, "text": "another utterance"},
            ],
        },
    )


def test_vllm_mode_parses_segments(vllm_settings, tmp_path):
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"RIFFfake-wav-bytes")

    backend = Qwen3Backend()
    transport = httpx.MockTransport(_mock_response)
    real_client = httpx.Client  # capture before patching

    def fake_client(**kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=transport, **kwargs)

    with patch("transcribemcp.backends.qwen3_backend.httpx.Client", side_effect=fake_client):
        result = backend.transcribe(str(audio), progress=lambda *_: None)

    assert result.backend_name == "qwen3_asr"
    assert result.language == "en"
    assert result.duration == 12.5
    assert [s.text for s in result.segments] == ["hello world", "another utterance"]
    assert result.segments[0].speaker is None  # backend never sets speakers


def test_vllm_mode_missing_url_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "qwen3_asr")
    monkeypatch.setenv("QWEN3_MODE", "vllm")
    monkeypatch.setenv("QWEN3_VLLM_URL", "")
    config_mod._settings = None
    try:
        audio = tmp_path / "fake.wav"
        audio.write_bytes(b"x")
        backend = Qwen3Backend()
        with pytest.raises(RuntimeError, match="QWEN3_VLLM_URL"):
            backend.transcribe(str(audio), progress=lambda *_: None)
    finally:
        config_mod._settings = None
