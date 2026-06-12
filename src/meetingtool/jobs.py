"""Background job runner for transcription.

Design:
- One process-wide ThreadPoolExecutor with max_workers=1. ASR is GPU/CPU heavy;
  running jobs in parallel would thrash. If that ever changes, bump this number.
- Each worker opens its own SQLite connection (sqlite3 connections are not
  safe to share across threads). Progress is persisted to the `jobs` table so
  callers read current status via get_status().
- On startup, db.init() reconciles any jobs left as queued/running (from a
  previous crash) to status='error'.

Windowed pipeline (supports resume on long recordings):
1. VAD pass → group voiced spans into ~5-min Windows aligned to silence
   boundaries. The plan is deterministic so resume can recompute and skip.
2. Per-window ASR via the configured backend. Each window's chunks are
   persisted before moving on, and `jobs.checkpoint_seconds` is advanced.
3. After all windows complete, diarization runs **once** on the full file
   and speaker labels are applied to already-persisted chunks.

A crash mid-pipeline leaves chunks from completed windows in place. The
`resume_job` tool re-submits the same job_id; the worker reads the
checkpoint, recomputes the same VAD plan, and skips any window whose end
is <= checkpoint.
"""
from __future__ import annotations

import logging
import sqlite3
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol

from .db import connect, new_id, now_iso, tx

logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    """Raised inside a worker when the DB shows the job has been cancelled.

    Cancellation is cooperative: the worker only notices between windows or
    when it calls progress(), so a stage that runs straight through C code
    (faster-whisper ASR, pyannote diarization) won't be interrupted until it
    returns and the next boundary fires.
    """


class TranscribeFn(Protocol):
    """Backend-agnostic ASR callable. Returns segments with absolute
    timestamps (relative to the full file, not the window)."""

    def __call__(
        self,
        audio_path: str,
        *,
        progress: Callable[[str, float], None],
        window: tuple[float, float] | None = None,
    ) -> "TranscribeOutput": ...


class TranscribeOutput(Protocol):
    segments: list
    language: str | None
    duration: float
    backend_name: str


# A Window is (start_seconds, end_seconds). We use plain tuples in the
# runner to avoid an import cycle with vad.py (which is optional).
Window = tuple[float, float]
PlanWindowsFn = Callable[[str], list[Window]]


# ---- job table helpers (work with any connection) --------------------------

def _insert_job(conn: sqlite3.Connection, meeting_id: str, kind: str = "transcribe") -> str:
    jid = new_id()
    ts = now_iso()
    with tx(conn):
        conn.execute(
            "INSERT INTO jobs(id, meeting_id, kind, status, created_at, progress_updated_at) "
            "VALUES (?, ?, ?, 'queued', ?, ?)",
            (jid, meeting_id, kind, ts, ts),
        )
    return jid


def _set_running(conn: sqlite3.Connection, job_id: str) -> None:
    ts = now_iso()
    conn.execute(
        "UPDATE jobs SET status='running', started_at=COALESCE(started_at, ?), "
        "stage='starting', progress=0.0, error=NULL, progress_updated_at=? "
        "WHERE id = ?",
        (ts, ts, job_id),
    )


def _set_progress(conn: sqlite3.Connection, job_id: str, stage: str, progress: float) -> None:
    conn.execute(
        "UPDATE jobs SET stage=?, progress=?, progress_updated_at=? WHERE id=?",
        (stage, max(0.0, min(1.0, progress)), now_iso(), job_id),
    )


def _set_checkpoint(conn: sqlite3.Connection, job_id: str, seconds: float) -> None:
    conn.execute(
        "UPDATE jobs SET checkpoint_seconds=?, progress_updated_at=? WHERE id=?",
        (seconds, now_iso(), job_id),
    )


def _get_checkpoint(conn: sqlite3.Connection, job_id: str) -> float:
    row = conn.execute(
        "SELECT checkpoint_seconds FROM jobs WHERE id=?", (job_id,)
    ).fetchone()
    return float(row["checkpoint_seconds"]) if row else 0.0


def _set_done(conn: sqlite3.Connection, job_id: str) -> None:
    ts = now_iso()
    conn.execute(
        "UPDATE jobs SET status='done', stage='done', progress=1.0, finished_at=?, "
        "progress_updated_at=? WHERE id=?",
        (ts, ts, job_id),
    )


def _set_error(conn: sqlite3.Connection, job_id: str, err: str) -> None:
    ts = now_iso()
    conn.execute(
        "UPDATE jobs SET status='error', error=?, finished_at=?, progress_updated_at=? "
        "WHERE id=?",
        (err, ts, ts, job_id),
    )


def _is_cancelled(conn: sqlite3.Connection, job_id: str) -> bool:
    row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
    return row is not None and row["status"] == "cancelled"


# ---- preflight -------------------------------------------------------------

def preflight_diarize(diarize: bool) -> None:
    """Validate diarization preconditions before any state change.

    If diarize=True is requested but HF_TOKEN isn't set, raise immediately so
    the caller (typically a tool handler) returns an error to the user before
    we waste an ASR pass that's only going to fail at the diarize step. See
    diarize._get_pipeline for the eventual failure site this preempts.
    """
    if not diarize:
        return
    from .config import get_settings
    if not get_settings().hf_token:
        raise ValueError(
            "DIARIZE=true but HF_TOKEN is not set. Set HF_TOKEN to a HuggingFace "
            "token with the pyannote/speaker-diarization-community-1 license accepted, "
            "or set DIARIZE=false to skip diarization."
        )


# ---- default window planner -----------------------------------------------

def _default_plan_windows(audio_path: str) -> list[Window]:
    """Plan by running silero-vad and grouping into ~5-min windows.

    If silero-vad isn't installed, fall back to a single window covering
    the full file — the backend will transcribe the whole thing and
    resume is effectively disabled. We don't hard-fail because the stub
    backend doesn't need VAD (tests inject their own planner anyway).
    """
    try:
        from .vad import detect_voiced, group_into_windows
    except ImportError:
        logger.warning(
            "silero-vad not installed; windowing disabled. "
            "Install the [vad] extra for chunked + resumable transcription."
        )
        return [(0.0, float("inf"))]
    spans = detect_voiced(audio_path)
    windows = group_into_windows(spans)
    return [(w.start, w.end) for w in windows]


# ---- runner -----------------------------------------------------------------

class JobRunner:
    """Owns the executor. One instance per process."""

    def __init__(
        self,
        db_path: Path,
        transcribe_fn: TranscribeFn,
        *,
        plan_windows_fn: PlanWindowsFn | None = None,
    ):
        self._db_path = db_path
        self._transcribe = transcribe_fn
        self._plan_windows = plan_windows_fn or _default_plan_windows
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meetingtool-job")

    def submit_transcribe(self, meeting_id: str, *, diarize: bool) -> str:
        """Enqueue a transcription job and return job_id immediately."""
        preflight_diarize(diarize)
        from .db import get_conn
        conn = get_conn()
        job_id = _insert_job(conn, meeting_id)
        conn.execute(
            "UPDATE meetings SET status='transcribing' WHERE id=?", (meeting_id,)
        )
        self._executor.submit(self._run_transcribe, job_id, meeting_id, diarize)
        return job_id

    def resubmit(self, job_id: str) -> None:
        """Re-queue an existing job (for resume_job). Worker picks up from
        the job's existing checkpoint_seconds."""
        from .config import get_settings
        preflight_diarize(get_settings().diarize)
        self._executor.submit(self._run_resubmitted_transcribe, job_id)

    # ---- worker entry points ---------------------------------------------

    def _run_transcribe(self, job_id: str, meeting_id: str, diarize: bool) -> None:
        conn = connect(self._db_path)
        try:
            if _is_cancelled(conn, job_id):
                conn.execute(
                    "UPDATE meetings SET status='pending' WHERE id=?", (meeting_id,)
                )
                return
            self._do_transcribe(conn, job_id, meeting_id, diarize, resume=False)
        except JobCancelled:
            logger.info("transcription job cancelled: %s", job_id)
            try:
                conn.execute(
                    "UPDATE meetings SET status='pending' WHERE id=?", (meeting_id,)
                )
            except Exception:
                logger.exception("failed to reset meeting after cancel")
        except Exception as exc:  # noqa: BLE001
            self._record_failure(conn, job_id, meeting_id, exc)
        finally:
            conn.close()

    def _run_resubmitted_transcribe(self, job_id: str) -> None:
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT meeting_id, status FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                logger.error("resubmit: job %s vanished", job_id)
                return
            meeting_id = row["meeting_id"]
            if _is_cancelled(conn, job_id):
                conn.execute(
                    "UPDATE meetings SET status='pending' WHERE id=?", (meeting_id,)
                )
                return
            from .config import get_settings
            diarize = get_settings().diarize
            self._do_transcribe(conn, job_id, meeting_id, diarize, resume=True)
        except JobCancelled:
            logger.info("resubmitted transcription cancelled: %s", job_id)
            mid = conn.execute(
                "SELECT meeting_id FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if mid and mid["meeting_id"]:
                try:
                    conn.execute(
                        "UPDATE meetings SET status='pending' WHERE id=?",
                        (mid["meeting_id"],),
                    )
                except Exception:
                    logger.exception("failed to reset meeting after cancel")
        except Exception as exc:  # noqa: BLE001
            mid = conn.execute(
                "SELECT meeting_id FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            self._record_failure(
                conn, job_id, mid["meeting_id"] if mid else None, exc,
            )
        finally:
            conn.close()

    # ---- core pipeline ---------------------------------------------------

    def _do_transcribe(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        meeting_id: str,
        diarize: bool,
        *,
        resume: bool,
    ) -> None:
        _set_running(conn, job_id)

        audio_path = conn.execute(
            "SELECT audio_path FROM meetings WHERE id=?", (meeting_id,)
        ).fetchone()["audio_path"]

        def check_cancel() -> None:
            if _is_cancelled(conn, job_id):
                raise JobCancelled()

        checkpoint = _get_checkpoint(conn, job_id)
        if not resume or checkpoint <= 0:
            # Fresh: wipe prior transcript artifacts for the meeting.
            _clear_meeting_transcript(conn, meeting_id)
            checkpoint = 0.0

        _set_progress(conn, job_id, "planning", 0.02)
        check_cancel()
        windows = self._plan_windows(audio_path)

        if not windows:
            # No speech detected. Record a "ready" but empty meeting.
            _mark_meeting_ready(conn, meeting_id, backend_name="", duration=0.0)
            _set_done(conn, job_id)
            return

        total_windows = len(windows)
        backend_name = ""
        last_window_end = windows[-1][1]

        for i, (win_start, win_end) in enumerate(windows):
            check_cancel()
            if win_end <= checkpoint:
                # Already done on a prior run.
                continue

            # Allocate 0.05 → 0.8 to the ASR loop; diarize gets 0.8 → 0.95.
            base = 0.05 + 0.75 * (i / total_windows)
            span = 0.75 / total_windows

            def win_progress(stage: str, pct: float, _base=base, _span=span) -> None:
                if _is_cancelled(conn, job_id):
                    raise JobCancelled()
                _set_progress(
                    conn, job_id,
                    f"asr:{stage}(win {i + 1}/{total_windows})",
                    _base + _span * max(0.0, min(1.0, pct)),
                )

            result = self._transcribe(
                audio_path, progress=win_progress, window=(win_start, win_end),
            )
            check_cancel()
            backend_name = result.backend_name or backend_name
            _persist_window_chunks(
                conn, meeting_id, result.segments, keep_speakers=not diarize,
            )
            _set_checkpoint(conn, job_id, win_end)
            checkpoint = win_end

        # All windows done. Diarize once on the full audio if requested.
        if diarize:
            _set_progress(conn, job_id, "diarize", 0.85)
            check_cancel()
            from .diarize import diarize as run_diarize
            turns = run_diarize(audio_path)
            check_cancel()
            _apply_speakers_to_chunks(conn, meeting_id, turns)

        _mark_meeting_ready(
            conn, meeting_id, backend_name=backend_name, duration=last_window_end,
        )
        _set_done(conn, job_id)

    def _record_failure(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        meeting_id: str | None,
        exc: BaseException,
    ) -> None:
        logger.exception("transcription job failed: %s", job_id)
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        try:
            _set_error(conn, job_id, err)
            if meeting_id:
                conn.execute(
                    "UPDATE meetings SET status='error' WHERE id=?", (meeting_id,)
                )
        except Exception:
            logger.exception("failed to record job error")

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


# ---- persistence helpers ---------------------------------------------------

def _clear_meeting_transcript(conn: sqlite3.Connection, meeting_id: str) -> None:
    """Wipe chunks + speakers for a fresh transcription run."""
    with tx(conn):
        conn.execute("DELETE FROM chunks WHERE meeting_id=?", (meeting_id,))
        conn.execute("DELETE FROM speakers WHERE meeting_id=?", (meeting_id,))


def _persist_window_chunks(
    conn: sqlite3.Connection,
    meeting_id: str,
    segments: list,
    *,
    keep_speakers: bool,
) -> None:
    """Append a window's segments to the chunks table.

    If `keep_speakers` is True, segment.speaker labels (if any) are
    materialized as speakers rows and linked via chunks.speaker_id. This is
    the no-diarize path — the backend is the source of truth for speakers.

    If False, chunks are inserted with speaker_id=NULL; the diarize-at-end
    stage will fill them in via `_apply_speakers_to_chunks` after all
    windows finish.
    """
    with tx(conn):
        label_to_sid: dict[str, str] = {}
        if keep_speakers:
            # Existing rows from prior windows (if any) should be reused.
            existing = conn.execute(
                "SELECT id, label FROM speakers WHERE meeting_id=?",
                (meeting_id,),
            ).fetchall()
            label_to_sid = {r["label"]: r["id"] for r in existing}

        for seg in segments:
            sid = None
            if keep_speakers and seg.speaker:
                sid = label_to_sid.get(seg.speaker)
                if sid is None:
                    sid = new_id()
                    conn.execute(
                        "INSERT INTO speakers(id, meeting_id, label) VALUES (?, ?, ?)",
                        (sid, meeting_id, seg.speaker),
                    )
                    label_to_sid[seg.speaker] = sid
            conn.execute(
                "INSERT INTO chunks(id, meeting_id, speaker_id, text, start_time, end_time) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (new_id(), meeting_id, sid, seg.text, seg.start, seg.end),
            )


def _apply_speakers_to_chunks(
    conn: sqlite3.Connection,
    meeting_id: str,
    turns: list,
) -> None:
    """Assign speaker labels to already-persisted chunks via max-overlap.

    Drops any speakers rows from prior runs (with no chunks linking to them
    after the reassign), creates fresh rows for every label pyannote found,
    and updates chunks.speaker_id in place.
    """
    chunks_rows = conn.execute(
        "SELECT id, start_time, end_time FROM chunks "
        "WHERE meeting_id=? ORDER BY start_time",
        (meeting_id,),
    ).fetchall()
    if not chunks_rows:
        return

    # Compute the best-matching speaker label for each chunk.
    assignments: dict[str, str] = {}  # chunk_id -> speaker_label
    labels_needed: set[str] = set()
    for chunk in chunks_rows:
        cs, ce = float(chunk["start_time"] or 0.0), float(chunk["end_time"] or 0.0)
        best_label: str | None = None
        best_overlap = 0.0
        for turn in turns:
            if turn.start >= ce:
                break
            if turn.end <= cs:
                continue
            overlap = min(ce, turn.end) - max(cs, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = turn.speaker
        if best_label:
            assignments[chunk["id"]] = best_label
            labels_needed.add(best_label)

    with tx(conn):
        # Delete old speakers rows. FK ON DELETE SET NULL will clear
        # chunks.speaker_id, then we reassign below. persons_ad trigger
        # preserves persons; only speakers rows get wiped.
        conn.execute("DELETE FROM speakers WHERE meeting_id=?", (meeting_id,))

        label_to_sid: dict[str, str] = {}
        for label in sorted(labels_needed):
            sid = new_id()
            conn.execute(
                "INSERT INTO speakers(id, meeting_id, label) VALUES (?, ?, ?)",
                (sid, meeting_id, label),
            )
            label_to_sid[label] = sid

        for chunk_id, label in assignments.items():
            conn.execute(
                "UPDATE chunks SET speaker_id=? WHERE id=?",
                (label_to_sid[label], chunk_id),
            )


def _mark_meeting_ready(
    conn: sqlite3.Connection,
    meeting_id: str,
    *,
    backend_name: str,
    duration: float,
) -> None:
    conn.execute(
        "UPDATE meetings SET status='ready', backend=?, duration_seconds=? WHERE id=?",
        (backend_name, int(duration) if duration != float("inf") else None, meeting_id),
    )


# ---- module-level singleton -------------------------------------------------

_runner: JobRunner | None = None


def get_runner() -> JobRunner:
    global _runner
    if _runner is None:
        from .config import get_settings
        from .transcribe import get_backend_fn
        _runner = JobRunner(get_settings().db_path, get_backend_fn())
    return _runner


def reset_runner_for_tests(runner: JobRunner | None) -> None:
    global _runner
    _runner = runner
