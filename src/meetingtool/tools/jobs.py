from ..config import get_settings
from ..db import get_conn, now_iso, tx
from ..jobs import get_runner, preflight_diarize
from ..server import mcp

_TERMINAL_STATUSES = {"done", "error", "cancelled"}
_RESUMABLE_STATUSES = {"error", "cancelled"}


@mcp.tool()
def transcribe_meeting(meeting_id: str) -> dict:
    """Enqueue an async transcription job. Returns immediately with a job_id.

    Poll status with get_status(job_id=...) or get_status(meeting_id=...).
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT id, status FROM meetings WHERE id=?", (meeting_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown meeting_id: {meeting_id}")
    if row["status"] in ("transcribing",):
        raise ValueError(f"meeting {meeting_id} already has a job in flight")

    runner = get_runner()
    job_id = runner.submit_transcribe(meeting_id, diarize=get_settings().diarize)
    return {"job_id": job_id, "meeting_id": meeting_id, "status": "queued"}


@mcp.tool()
def get_status(job_id: str | None = None, meeting_id: str | None = None) -> dict:
    """Return the current state of a transcription job.

    Pass either job_id (most recent if multiple) or meeting_id.

    `progress_updated_at` is the timestamp of the last progress/stage write.
    Long-running ASR backends can sit on the same `progress` value for
    minutes mid-stage; compare `progress_updated_at` to now to distinguish
    "slow" from "stuck" (heuristic: >3-5 min without an update on a job that
    still reports status='running' is suspicious).
    """
    if not job_id and not meeting_id:
        raise ValueError("provide either job_id or meeting_id")
    conn = get_conn()
    if job_id:
        row = conn.execute(
            "SELECT id, meeting_id, kind, status, stage, progress, error, "
            "started_at, finished_at, created_at, progress_updated_at, "
            "checkpoint_seconds FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, meeting_id, kind, status, stage, progress, error, "
            "started_at, finished_at, created_at, progress_updated_at, "
            "checkpoint_seconds FROM jobs "
            "WHERE meeting_id=? ORDER BY created_at DESC LIMIT 1",
            (meeting_id,),
        ).fetchone()
    if row is None:
        raise ValueError("no matching job")
    return dict(row)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Cancel a queued or running transcription job.

    Cooperative: the worker only notices cancellation at stage boundaries
    (when it calls the progress callback). A whisperx/pyannote stage that
    runs straight through native code won't be interrupted until it returns
    and the next stage begins. For a truly wedged job, expect to wait out
    the current stage.

    Idempotent on terminal jobs (done / error / already cancelled) — returns
    the current status without error. On cancel, the meeting's status goes
    back to 'pending' so the user can kick off a fresh transcription job.
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT id, meeting_id, status FROM jobs WHERE id=?", (job_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown job_id: {job_id}")

    if row["status"] in _TERMINAL_STATUSES:
        return {
            "job_id": job_id,
            "meeting_id": row["meeting_id"],
            "status": row["status"],
            "cancelled": False,
            "was_running": False,
        }

    was_running = row["status"] == "running"
    ts = now_iso()
    with tx(conn):
        conn.execute(
            "UPDATE jobs SET status='cancelled', finished_at=?, progress_updated_at=? "
            "WHERE id=? AND status NOT IN ('done','error','cancelled')",
            (ts, ts, job_id),
        )
        # If the worker hasn't picked the job up yet (status was 'queued') or
        # is queued behind another job, the meeting's status update below
        # makes the cancel visible to list/get tools immediately. If the
        # worker is running, it will also reset the meeting on its way out —
        # but doing it here is idempotent and keeps the UI consistent before
        # the worker observes the cancel.
        if row["meeting_id"]:
            conn.execute(
                "UPDATE meetings SET status='pending' WHERE id=? AND status='transcribing'",
                (row["meeting_id"],),
            )
    return {
        "job_id": job_id,
        "meeting_id": row["meeting_id"],
        "status": "cancelled",
        "cancelled": True,
        "was_running": was_running,
    }


@mcp.tool()
def list_jobs(status: str | None = None, limit: int = 20) -> list[dict]:
    """List recent jobs. Optionally filter by status (queued|running|done|error|cancelled)."""
    conn = get_conn()
    if status:
        rows = conn.execute(
            "SELECT id, meeting_id, kind, status, stage, progress, error, "
            "started_at, finished_at, created_at, progress_updated_at, "
            "checkpoint_seconds FROM jobs "
            "WHERE status=? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, meeting_id, kind, status, stage, progress, error, "
            "started_at, finished_at, created_at, progress_updated_at, "
            "checkpoint_seconds FROM jobs "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
def resume_job(job_id: str) -> dict:
    """Resume an errored or cancelled transcription job from its checkpoint.

    Long transcriptions are split into ~5-minute windows (aligned to VAD
    silence boundaries). Each window's chunks are persisted before the
    next one starts and `jobs.checkpoint_seconds` is advanced. On crash or
    cancel, those persisted chunks stay put; resume re-plans the same VAD
    windows (deterministic), skips any whose end is <= checkpoint, and
    picks up from there. A three-hour recording that crashed at window 12
    of 36 resumes at window 13, not at zero.

    Only jobs in 'error' or 'cancelled' status can be resumed. 'done' jobs
    are no-ops; 'running' / 'queued' jobs must be cancelled first.

    Returns the job's post-resubmit state: `{job_id, meeting_id, status,
    checkpoint_seconds}`.
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT id, meeting_id, status, checkpoint_seconds FROM jobs WHERE id=?",
        (job_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown job_id: {job_id}")
    if row["status"] == "done":
        raise ValueError(f"job {job_id} is already done — nothing to resume")
    if row["status"] not in _RESUMABLE_STATUSES:
        raise ValueError(
            f"job {job_id} has status '{row['status']}'; "
            "cancel it first if you want to resume"
        )
    if not row["meeting_id"]:
        raise ValueError(f"job {job_id} has no meeting attached")

    # Fail BEFORE we mutate jobs/meetings state, so a misconfigured resume
    # leaves the row in its prior 'error'/'cancelled' state instead of a
    # half-applied 'queued'.
    preflight_diarize(get_settings().diarize)

    ts = now_iso()
    with tx(conn):
        # Preserve checkpoint_seconds and started_at; reset transient state.
        conn.execute(
            "UPDATE jobs SET status='queued', stage='resuming', error=NULL, "
            "finished_at=NULL, progress_updated_at=? WHERE id=?",
            (ts, job_id),
        )
        conn.execute(
            "UPDATE meetings SET status='transcribing' WHERE id=?",
            (row["meeting_id"],),
        )

    get_runner().resubmit(job_id)
    return {
        "job_id": job_id,
        "meeting_id": row["meeting_id"],
        "status": "queued",
        "checkpoint_seconds": float(row["checkpoint_seconds"] or 0.0),
    }
