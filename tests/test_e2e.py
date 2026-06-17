"""End-to-end pipeline test against a real audio fixture.

Skipped unless:
- tests/fixtures/sample.wav exists (run scripts/fetch_sample_audio.sh)
- MEETINGTOOL_E2E=1 is set

Diarization runs only when pyannote is installed AND HF_TOKEN is set;
otherwise the no-speakers path is exercised.
"""
import os
from pathlib import Path

import pytest

from meetingtool import config as config_mod
from meetingtool.scribe_tools import read_transcript, transcribe

FIXTURE = Path(__file__).parent / "fixtures" / "sample.wav"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists() or os.getenv("MEETINGTOOL_E2E") != "1",
    reason="e2e disabled: set MEETINGTOOL_E2E=1 and fetch fixture via scripts/fetch_sample_audio.sh",
)


def _run_real_backend(tmp_path, monkeypatch, *, backend: str, extra_env: dict | None = None):
    """Transcribe the fixture with a real backend; return the rendered text."""
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", backend)
    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)

    try:
        import pyannote.audio  # noqa: F401
        have_pyannote = True
    except ImportError:
        have_pyannote = False
    # Override the autouse DIARIZE=false when we can actually diarize.
    monkeypatch.setenv(
        "DIARIZE", "true" if (have_pyannote and os.getenv("HF_TOKEN")) else "false"
    )
    # Keep the transcript out of the fixtures dir.
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    config_mod._settings = None

    try:
        result = transcribe(str(FIXTURE))
        assert result["segments"] > 0, result
        return read_transcript(result["transcript_path"])
    finally:
        config_mod._settings = None


def test_whisperx_real_audio(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper", reason="whisperx extra not installed")
    transcript = _run_real_backend(
        tmp_path, monkeypatch, backend="whisperx", extra_env={"WHISPER_MODEL": "base"},
    )
    assert len(transcript) > 20


def test_cohere_real_audio(tmp_path, monkeypatch):
    pytest.importorskip("silero_vad", reason="vad extra not installed")
    has_transformers = False
    has_mlx = False
    try:
        import transformers  # noqa: F401
        has_transformers = True
    except ImportError:
        pass
    try:
        import mlx_audio  # noqa: F401
        has_mlx = True
    except ImportError:
        pass
    if not (has_transformers or has_mlx):
        pytest.skip("cohere extra not installed (need transformers or mlx-audio)")

    transcript = _run_real_backend(
        tmp_path, monkeypatch, backend="cohere", extra_env={"COHERE_LANG": "en"},
    )
    assert len(transcript) > 20
