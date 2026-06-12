"""Tests for the windowed transcription pipeline + checkpoint/resume.

Uses the stub backend with an injected multi-window plan so we don't need
silero-vad or real audio. The stub's fake output covers 0..10s with three
segments: [0.0-3.2], [3.2-7.5], [7.5-10.0]. Splitting the plan on a 5s
boundary puts the first two segments in window 1 and the third in window 2.
"""
import time

import pytest

from meetingtool import jobs as jobs_mod
from meetingtool.backends.stub import StubBackend
from meetingtool.tools.jobs import (
    cancel_job,
    get_status,
    list_jobs,
    resume_job,
    transcribe_meeting,
)
from meetingtool.tools.meetings import add_meeting, get_transcript
from meetingtool.tools.projects import create_project


def _wait_for_status(job_id, want: set[str], timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = get_status(job_id=job_id)
        if s["status"] in want:
            return s
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {want}, last={s}")


def _install_multi_window_runner(db_path, *, delay=0.01, windows):
    stub = StubBackend(delay=delay)

    def fn(audio_path, *, progress, window=None):
        return stub.transcribe(audio_path, progress=progress, window=window)

    runner = jobs_mod.JobRunner(
        db_path, fn,
        plan_windows_fn=lambda _p: list(windows),
    )
    jobs_mod.reset_runner_for_tests(runner)
    return runner


# ---- multi-window happy path ----------------------------------------------


def test_multi_window_persists_all_chunks(conn, tmp_path):
    """Two-window plan should transcribe both and end with all 3 stub segments."""
    runner = _install_multi_window_runner(
        tmp_path / "test.db", windows=[(0.0, 5.0), (5.0, 10.0)],
    )
    try:
        p = create_project("P")
        f = tmp_path / "a.wav"; f.write_bytes(b"x")
        m = add_meeting(p["id"], "t", str(f), auto_transcribe=False)
        job_id = transcribe_meeting(m["id"])["job_id"]
        final = _wait_for_status(job_id, {"done", "error"})
        assert final["status"] == "done", final.get("error")

        # Checkpoint should be at the last window's end.
        assert final["checkpoint_seconds"] == 5.0 * 2

        # All three stub segments should have persisted.
        chunks = conn.execute(
            "SELECT text, start_time FROM chunks WHERE meeting_id=? ORDER BY start_time",
            (m["id"],),
        ).fetchall()
        assert len(chunks) == 3
        assert "fake transcript" in chunks[0]["text"].lower()
    finally:
        runner.shutdown()


def test_checkpoint_advances_after_each_window(conn, tmp_path):
    """Checkpoint should be visible at 5.0s after window 1 finishes."""
    # Slow per-window delay so we can catch the mid-way checkpoint.
    runner = _install_multi_window_runner(
        tmp_path / "test.db", delay=0.1,
        windows=[(0.0, 5.0), (5.0, 10.0)],
    )
    try:
        p = create_project("P")
        f = tmp_path / "a.wav"; f.write_bytes(b"x")
        m = add_meeting(p["id"], "t", str(f), auto_transcribe=False)
        job_id = transcribe_meeting(m["id"])["job_id"]

        saw_mid = False
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            s = get_status(job_id=job_id)
            if 0.0 < s["checkpoint_seconds"] < 10.0:
                saw_mid = True
            if s["status"] in ("done", "error"):
                break
            time.sleep(0.03)
        assert saw_mid, "never observed a mid-way checkpoint"
    finally:
        runner.shutdown()


# ---- resume ---------------------------------------------------------------


def test_resume_skips_completed_windows(conn, tmp_path):
    """A job with a checkpoint at 5.0s should only process the second window on resume."""
    runner = _install_multi_window_runner(
        tmp_path / "test.db", windows=[(0.0, 5.0), (5.0, 10.0)],
    )
    try:
        p = create_project("P")
        f = tmp_path / "a.wav"; f.write_bytes(b"x")
        m = add_meeting(p["id"], "t", str(f), auto_transcribe=False)
        job_id = transcribe_meeting(m["id"])["job_id"]
        _wait_for_status(job_id, {"done", "error"})

        # Simulate a crash-style reset: flip job to 'error' and pretend only
        # the first window committed. Leave the first window's chunks in
        # place; delete anything after 5.0s.
        conn.execute(
            "UPDATE jobs SET status='error', error='simulated crash', "
            "checkpoint_seconds=5.0 WHERE id=?",
            (job_id,),
        )
        conn.execute(
            "DELETE FROM chunks WHERE meeting_id=? AND start_time >= 5.0",
            (m["id"],),
        )
        conn.execute(
            "UPDATE meetings SET status='error' WHERE id=?", (m["id"],),
        )

        # Track which windows the backend sees post-resume.
        seen_windows: list = []

        stub = StubBackend(delay=0.01)
        def spying_fn(audio_path, *, progress, window=None):
            seen_windows.append(window)
            return stub.transcribe(audio_path, progress=progress, window=window)

        runner._transcribe = spying_fn  # swap in place for the same executor

        out = resume_job(job_id)
        assert out["checkpoint_seconds"] == 5.0
        assert out["status"] == "queued"

        final = _wait_for_status(job_id, {"done", "error"})
        assert final["status"] == "done", final.get("error")

        # Only window 2 should have been called on the backend.
        assert seen_windows == [(5.0, 10.0)]

        # Chunks from both halves should now be present.
        total = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE meeting_id=?", (m["id"],)
        ).fetchone()[0]
        assert total == 3
    finally:
        runner.shutdown()


def test_resume_rejects_done_jobs(conn, tmp_path):
    runner = _install_multi_window_runner(
        tmp_path / "test.db", windows=[(0.0, 10.0)],
    )
    try:
        p = create_project("P")
        f = tmp_path / "a.wav"; f.write_bytes(b"x")
        m = add_meeting(p["id"], "t", str(f), auto_transcribe=False)
        job_id = transcribe_meeting(m["id"])["job_id"]
        _wait_for_status(job_id, {"done", "error"})

        with pytest.raises(ValueError, match="already done"):
            resume_job(job_id)
    finally:
        runner.shutdown()


def test_resume_rejects_running_jobs(conn, tmp_path):
    """Can't resume a job that's still running / queued — cancel first."""
    runner = _install_multi_window_runner(
        tmp_path / "test.db", delay=0.2,
        windows=[(0.0, 10.0)],
    )
    try:
        p = create_project("P")
        f = tmp_path / "a.wav"; f.write_bytes(b"x")
        m = add_meeting(p["id"], "t", str(f), auto_transcribe=False)
        job_id = transcribe_meeting(m["id"])["job_id"]
        _wait_for_status(job_id, {"running", "done", "error"})

        # If it's still running, resume should refuse.
        status = get_status(job_id=job_id)["status"]
        if status == "running":
            with pytest.raises(ValueError, match="cancel it first"):
                resume_job(job_id)
        _wait_for_status(job_id, {"done", "error", "cancelled"})
    finally:
        runner.shutdown()


def test_resume_unknown_job(conn, tmp_path):
    with pytest.raises(ValueError, match="unknown job_id"):
        resume_job("does-not-exist")


# ---- no windows -----------------------------------------------------------


def test_empty_plan_marks_ready_with_no_chunks(conn, tmp_path):
    """If VAD returns no speech, the job should still complete successfully."""
    runner = _install_multi_window_runner(
        tmp_path / "test.db", windows=[],
    )
    try:
        p = create_project("P")
        f = tmp_path / "a.wav"; f.write_bytes(b"x")
        m = add_meeting(p["id"], "t", str(f), auto_transcribe=False)
        job_id = transcribe_meeting(m["id"])["job_id"]
        final = _wait_for_status(job_id, {"done", "error"})
        assert final["status"] == "done"

        count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE meeting_id=?", (m["id"],)
        ).fetchone()[0]
        assert count == 0

        t = get_transcript(m["id"])
        assert t["status"] == "ready"
    finally:
        runner.shutdown()


# ---- VAD grouping unit tests ---------------------------------------------


def test_group_into_windows_packs_under_target():
    from meetingtool.vad import VoicedSpan, group_into_windows
    spans = [VoicedSpan(0, 60), VoicedSpan(70, 120), VoicedSpan(200, 260)]
    windows = group_into_windows(spans, target_seconds=300.0)
    assert len(windows) == 1
    assert windows[0].start == 0
    assert windows[0].end == 260


def test_group_into_windows_breaks_on_target():
    from meetingtool.vad import VoicedSpan, group_into_windows
    spans = [
        VoicedSpan(0, 100),
        VoicedSpan(110, 250),  # inclusion keeps window under 300
        VoicedSpan(260, 400),  # including this would exceed 300 → new window
        VoicedSpan(410, 500),
    ]
    windows = group_into_windows(spans, target_seconds=300.0)
    assert len(windows) == 2
    assert windows[0].start == 0 and windows[0].end == 250
    assert windows[1].start == 260 and windows[1].end == 500


def test_group_into_windows_empty():
    from meetingtool.vad import group_into_windows
    assert group_into_windows([]) == []
