"""Async job runner tests using the stub backend."""
import time

from meetingtool import db as db_mod
from meetingtool import jobs as jobs_mod
from meetingtool.backends.stub import StubBackend
from meetingtool.tools.meetings import add_meeting, get_transcript
from meetingtool.tools.projects import create_project
from meetingtool.tools.jobs import cancel_job, get_status, list_jobs, transcribe_meeting


def _install_runner(db_path, delay: float = 0.01, windows=None):
    """Install a stub-backed runner. Windows default to a single 0..10s window
    (the stub's fake audio range). Pass a list for multi-window tests."""
    stub = StubBackend(delay=delay)

    def fn(audio_path, *, progress, window=None):
        return stub.transcribe(audio_path, progress=progress, window=window)

    def plan_fn(_audio_path):
        return list(windows) if windows is not None else [(0.0, 10.0)]

    runner = jobs_mod.JobRunner(db_path, fn, plan_windows_fn=plan_fn)
    jobs_mod.reset_runner_for_tests(runner)
    return runner


def _wait_for_status(job_id, want: set[str], timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = get_status(job_id=job_id)
        if s["status"] in want:
            return s
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {want}, last={s}")


def test_full_async_flow(conn, tmp_path):
    runner = _install_runner(tmp_path / "test.db")  # matches conftest db path? no — use conn's
    # conftest created the DB at tmp_path / "test.db" — same path.

    p = create_project("P")
    f = tmp_path / "a.wav"; f.write_bytes(b"x")
    m = add_meeting(p["id"], "m", str(f), auto_transcribe=False)

    resp = transcribe_meeting(m["id"])
    assert resp["status"] == "queued"
    job_id = resp["job_id"]

    final = _wait_for_status(job_id, {"done", "error"})
    assert final["status"] == "done", final.get("error")
    assert final["progress"] == 1.0
    assert final["progress_updated_at"] is not None
    assert final["progress_updated_at"] >= final["started_at"]

    t = get_transcript(m["id"])
    assert t["status"] == "ready"
    assert "fake transcript" in t["transcript"].lower()
    assert "SPEAKER_00" in t["transcript"] or "SPEAKER_01" in t["transcript"]

    jobs_list = list_jobs()
    assert any(j["id"] == job_id and j["status"] == "done" for j in jobs_list)

    runner.shutdown()


def test_status_transitions_queued_running_done(conn, tmp_path):
    runner = _install_runner(tmp_path / "test.db", delay=0.15)

    p = create_project("P")
    f = tmp_path / "a.wav"; f.write_bytes(b"x")
    m = add_meeting(p["id"], "m", str(f), auto_transcribe=False)
    job_id = transcribe_meeting(m["id"])["job_id"]

    # Should observe 'running' at least once
    saw_running = False
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        s = get_status(job_id=job_id)
        if s["status"] == "running":
            saw_running = True
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.03)
    assert saw_running, "never observed running state"
    final = _wait_for_status(job_id, {"done", "error"})
    assert final["status"] == "done"

    runner.shutdown()


def test_progress_updated_at_advances_between_stages(conn, tmp_path):
    """progress_updated_at should bump on every stage/progress write."""
    runner = _install_runner(tmp_path / "test.db", delay=0.1)

    p = create_project("P")
    f = tmp_path / "a.wav"; f.write_bytes(b"x")
    m = add_meeting(p["id"], "m", str(f), auto_transcribe=False)
    job_id = transcribe_meeting(m["id"])["job_id"]

    seen_timestamps: set[str] = set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        s = get_status(job_id=job_id)
        if s["progress_updated_at"]:
            seen_timestamps.add(s["progress_updated_at"])
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.05)

    # At minimum: queued insert + running + 4 stage updates + done ≥ 3 distinct
    assert len(seen_timestamps) >= 3, seen_timestamps
    runner.shutdown()


def test_cancel_unknown_job(conn):
    import pytest

    with pytest.raises(ValueError, match="unknown job_id"):
        cancel_job("does-not-exist")


def test_cancel_terminal_job_is_idempotent(conn, tmp_path):
    """Cancelling a job that's already done returns the existing status."""
    runner = _install_runner(tmp_path / "test.db", delay=0.0)

    p = create_project("P")
    f = tmp_path / "a.wav"; f.write_bytes(b"x")
    m = add_meeting(p["id"], "m", str(f), auto_transcribe=False)
    job_id = transcribe_meeting(m["id"])["job_id"]
    _wait_for_status(job_id, {"done", "error"})

    out = cancel_job(job_id)
    assert out["cancelled"] is False
    assert out["status"] == "done"
    assert out["was_running"] is False
    # Meeting should stay 'ready' — cancel must not revert a finished job.
    meeting_status = conn.execute(
        "SELECT status FROM meetings WHERE id=?", (m["id"],)
    ).fetchone()["status"]
    assert meeting_status == "ready"

    runner.shutdown()


def test_cancel_running_job_observed_by_worker(conn, tmp_path):
    """A running job should flip to 'cancelled' and reset the meeting."""
    # Longer delay so the worker is mid-stage when we cancel; ensures the
    # progress() callback sees the cancel before the stub's next sleep.
    runner = _install_runner(tmp_path / "test.db", delay=0.15)

    p = create_project("P")
    f = tmp_path / "a.wav"; f.write_bytes(b"x")
    m = add_meeting(p["id"], "m", str(f), auto_transcribe=False)
    job_id = transcribe_meeting(m["id"])["job_id"]

    # Wait until worker has entered 'running' so we cancel a live job.
    _wait_for_status(job_id, {"running", "done", "error"})
    out = cancel_job(job_id)
    assert out["cancelled"] is True
    assert out["status"] == "cancelled"

    # Worker should observe cancel at next progress() and exit.
    final = _wait_for_status(job_id, {"cancelled", "done", "error"})
    assert final["status"] == "cancelled"

    # Meeting reverts to pending so a fresh transcribe can be kicked off.
    meeting_status = conn.execute(
        "SELECT status FROM meetings WHERE id=?", (m["id"],)
    ).fetchone()["status"]
    assert meeting_status == "pending"

    # No chunks persisted for a cancelled job.
    chunk_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE meeting_id=?", (m["id"],)
    ).fetchone()[0]
    assert chunk_count == 0

    runner.shutdown()


def test_cancelled_meeting_can_be_resubmitted(conn, tmp_path):
    """After cancel, transcribe_meeting should accept a fresh job."""
    runner = _install_runner(tmp_path / "test.db", delay=0.15)

    p = create_project("P")
    f = tmp_path / "a.wav"; f.write_bytes(b"x")
    m = add_meeting(p["id"], "m", str(f), auto_transcribe=False)

    job1 = transcribe_meeting(m["id"])["job_id"]
    _wait_for_status(job1, {"running"})
    cancel_job(job1)
    _wait_for_status(job1, {"cancelled"})

    # Fresh submit should succeed because meeting.status went back to pending.
    job2 = transcribe_meeting(m["id"])["job_id"]
    assert job2 != job1
    final = _wait_for_status(job2, {"done", "error"})
    assert final["status"] == "done"

    runner.shutdown()


def test_preflight_diarize_without_token_raises(monkeypatch):
    """preflight_diarize(True) without HF_TOKEN should raise. We set
    hf_token on the live settings object instead of via env, because the
    project's .env may legitimately have a real token (and pydantic-settings
    reads .env regardless of monkeypatch.delenv)."""
    import pytest

    from meetingtool import config
    from meetingtool.jobs import preflight_diarize

    config._settings = None
    settings = config.get_settings()
    monkeypatch.setattr(settings, "hf_token", "")

    # diarize=False is always a no-op regardless of token state.
    preflight_diarize(False)

    with pytest.raises(ValueError, match="HF_TOKEN"):
        preflight_diarize(True)


def test_preflight_blocks_submit_before_state_change(conn, tmp_path, monkeypatch):
    """submit_transcribe with diarize=True and no token must NOT create a
    job row or flip the meeting to 'transcribing' — the raise has to happen
    before any DB mutation."""
    import pytest

    from meetingtool import config

    config._settings = None
    settings = config.get_settings()
    monkeypatch.setattr(settings, "hf_token", "")

    runner = _install_runner(tmp_path / "test.db")
    try:
        p = create_project("P")
        f = tmp_path / "a.wav"; f.write_bytes(b"x")
        m = add_meeting(p["id"], "m", str(f), auto_transcribe=False)

        with pytest.raises(ValueError, match="HF_TOKEN"):
            runner.submit_transcribe(m["id"], diarize=True)

        status = conn.execute(
            "SELECT status FROM meetings WHERE id=?", (m["id"],)
        ).fetchone()["status"]
        assert status == "pending"
        job_count = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE meeting_id=?", (m["id"],)
        ).fetchone()[0]
        assert job_count == 0
    finally:
        runner.shutdown()


def test_startup_reconciliation(tmp_path):
    """A crashed job should be flipped to error on next init()."""
    db_path = tmp_path / "r.db"
    c1 = db_mod.init(db_path)

    pid, mid, jid = db_mod.new_id(), db_mod.new_id(), db_mod.new_id()
    c1.execute(
        "INSERT INTO projects(id, name, created_at) VALUES (?, ?, ?)",
        (pid, "P", db_mod.now_iso()),
    )
    c1.execute(
        "INSERT INTO meetings(id, project_id, title, audio_path, status, created_at) "
        "VALUES (?, ?, ?, ?, 'transcribing', ?)",
        (mid, pid, "M", "/tmp/a.wav", db_mod.now_iso()),
    )
    c1.execute(
        "INSERT INTO jobs(id, meeting_id, kind, status, created_at) "
        "VALUES (?, ?, 'transcribe', 'running', ?)",
        (jid, mid, db_mod.now_iso()),
    )
    c1.close()

    c2 = db_mod.init(db_path)
    job = c2.execute("SELECT status, error FROM jobs WHERE id=?", (jid,)).fetchone()
    meeting = c2.execute("SELECT status FROM meetings WHERE id=?", (mid,)).fetchone()
    assert job["status"] == "error"
    assert job["error"] == "interrupted"
    assert meeting["status"] == "error"
    c2.close()
