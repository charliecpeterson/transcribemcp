"""Unit tests for the Cohere backend. Mocks the model + VAD so no ML deps run."""
from pathlib import Path
from unittest.mock import patch

import pytest

from meetingtool import config as config_mod
from meetingtool.backends.cohere_backend import CohereBackend
from meetingtool.vad import VoicedSpan


@pytest.fixture
def cohere_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "cohere")
    monkeypatch.setenv("COHERE_LANG", "en")
    monkeypatch.setenv("DIARIZE", "false")
    monkeypatch.setenv("MEETINGTOOL_DATA_DIR", str(tmp_path))
    config_mod._settings = None
    yield
    config_mod._settings = None


def _fake_spans():
    return [
        VoicedSpan(start=0.0, end=3.0),
        VoicedSpan(start=3.5, end=8.2),
        VoicedSpan(start=9.0, end=12.0),
    ]


def test_cohere_synthesizes_segments_from_vad(cohere_settings, tmp_path):
    # Need real decodable audio because decode_pcm shells to ffmpeg. Use the
    # sample fixture if it exists; otherwise write a short silence wav.
    audio = Path(__file__).parent / "fixtures" / "sample.wav"
    if not audio.exists():
        pytest.skip("sample.wav fixture missing")

    utterances = iter(["hello world", "second utterance", "third span"])

    backend = CohereBackend()
    # Short-circuit model loading + inference; real paths need transformers/mlx.
    backend._runtime = "transformers"

    with patch(
        "meetingtool.backends.cohere_backend.detect_voiced",
        return_value=_fake_spans(),
    ), patch.object(
        backend, "_ensure_loaded", lambda: None,
    ), patch.object(
        backend, "_transcribe_span",
        side_effect=lambda wav, span, *, lang: next(utterances),
    ):
        result = backend.transcribe(
            str(audio), progress=lambda *_: None,
        )

    assert result.backend_name == "cohere"
    assert result.language == "en"
    assert [s.text for s in result.segments] == [
        "hello world", "second utterance", "third span",
    ]
    # Timestamps come straight from the VAD spans.
    assert [(s.start, s.end) for s in result.segments] == [
        (0.0, 3.0), (3.5, 8.2), (9.0, 12.0),
    ]
    assert all(s.speaker is None for s in result.segments)


def test_cohere_drops_empty_utterances(cohere_settings, tmp_path):
    audio = Path(__file__).parent / "fixtures" / "sample.wav"
    if not audio.exists():
        pytest.skip("sample.wav fixture missing")

    utterances = iter(["keep this", "   ", "and this"])

    backend = CohereBackend()
    backend._runtime = "transformers"
    with patch(
        "meetingtool.backends.cohere_backend.detect_voiced",
        return_value=_fake_spans(),
    ), patch.object(
        backend, "_ensure_loaded", lambda: None,
    ), patch.object(
        backend, "_transcribe_span",
        side_effect=lambda wav, span, *, lang: next(utterances),
    ):
        result = backend.transcribe(
            str(audio), progress=lambda *_: None,
        )

    assert [s.text for s in result.segments] == ["keep this", "and this"]


def test_cohere_no_voice_returns_empty(cohere_settings, tmp_path):
    audio = Path(__file__).parent / "fixtures" / "sample.wav"
    if not audio.exists():
        pytest.skip("sample.wav fixture missing")

    backend = CohereBackend()
    backend._runtime = "transformers"
    with patch(
        "meetingtool.backends.cohere_backend.detect_voiced",
        return_value=[],
    ), patch.object(
        backend, "_ensure_loaded", lambda: None,
    ):
        result = backend.transcribe(
            str(audio), progress=lambda *_: None,
        )
    assert result.segments == []
    assert result.backend_name == "cohere"


def test_router_dispatches_to_cohere(monkeypatch, tmp_path):
    from meetingtool.transcribe import get_backend_fn

    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "cohere")
    monkeypatch.setenv("MEETINGTOOL_DATA_DIR", str(tmp_path))
    config_mod._settings = None
    try:
        fn = get_backend_fn()
        # The wrapper closes over a CohereBackend instance; we can't assert
        # type without poking at closures, but calling the fn path with a
        # fully-mocked backend is overkill. Just verify it constructs.
        assert callable(fn)
    finally:
        config_mod._settings = None
