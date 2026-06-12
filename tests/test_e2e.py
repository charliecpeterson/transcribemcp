"""End-to-end pipeline test against a real audio fixture.

Skipped unless:
- tests/fixtures/sample.wav exists (run scripts/fetch_sample_audio.sh)
- TRANSCRIPTION_BACKEND is not 'stub' (set MEETINGTOOL_E2E=1 to opt in)

The pyannote diarization step is skipped if HF_TOKEN is not set.
"""
import os
import time
from pathlib import Path

import pytest

from meetingtool import config as config_mod
from meetingtool import jobs as jobs_mod
from meetingtool.tools.jobs import get_status, transcribe_meeting
from meetingtool.tools.meetings import add_meeting, get_transcript
from meetingtool.tools.projects import create_project

FIXTURE = Path(__file__).parent / "fixtures" / "sample.wav"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists() or os.getenv("MEETINGTOOL_E2E") != "1",
    reason="e2e disabled: set MEETINGTOOL_E2E=1 and fetch fixture via scripts/fetch_sample_audio.sh",
)


def _wait_for_done(job_id, timeout: float = 300.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = get_status(job_id=job_id)
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.5)
    raise AssertionError(f"job did not finish in {timeout}s: last={s}")


def _run_real_backend(tmp_path, monkeypatch, *, backend: str, extra_env: dict | None = None):
    """Shared harness: set up a runner against a real backend, run one job, return the transcript."""
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", backend)
    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)
    # Diarize only if both pyannote is installed AND HF_TOKEN is set —
    # otherwise fall back to no-speakers instead of erroring.
    try:
        import pyannote.audio  # noqa: F401
        have_pyannote = True
    except ImportError:
        have_pyannote = False
    if not (have_pyannote and os.getenv("HF_TOKEN")):
        monkeypatch.setenv("DIARIZE", "false")
    monkeypatch.setenv("MEETINGTOOL_DATA_DIR", str(tmp_path))
    config_mod._settings = None

    from meetingtool.transcribe import get_backend_fn
    runner = jobs_mod.JobRunner(config_mod.get_settings().db_path, get_backend_fn())
    jobs_mod.reset_runner_for_tests(runner)

    try:
        p = create_project("E2E")
        m = add_meeting(p["id"], "JFK", str(FIXTURE), auto_transcribe=False)
        resp = transcribe_meeting(m["id"])
        final = _wait_for_done(resp["job_id"])
        assert final["status"] == "done", final.get("error")

        t = get_transcript(m["id"])
        assert t["status"] == "ready"
        return t["transcript"]
    finally:
        runner.shutdown()
        jobs_mod.reset_runner_for_tests(None)
        config_mod._settings = None


def test_whisperx_real_audio(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper", reason="whisperx extra not installed")
    transcript = _run_real_backend(
        tmp_path, monkeypatch,
        backend="whisperx",
        extra_env={"WHISPER_MODEL": "base"},
    )
    assert len(transcript) > 20


def test_cohere_real_audio(tmp_path, monkeypatch):
    pytest.importorskip("silero_vad", reason="vad extra not installed")
    # Either transformers or mlx-audio must be present. Let the backend pick.
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
        tmp_path, monkeypatch,
        backend="cohere",
        extra_env={"COHERE_LANG": "en"},
    )
    assert len(transcript) > 20
