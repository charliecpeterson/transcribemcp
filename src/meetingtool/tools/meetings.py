from pathlib import Path

from ..db import get_conn, new_id, now_iso, tx
from ..server import mcp


@mcp.tool()
def add_meeting(
    project_id: str,
    title: str,
    audio_path: str,
    date: str | None = None,
    auto_transcribe: bool = True,
) -> dict:
    """Register an audio file as a meeting under a project.

    `audio_path` must be an absolute path that exists on disk.
    `date` is a free-form ISO date (YYYY-MM-DD) or None.

    By default, a transcription job is enqueued immediately — the returned
    dict includes `job_id` and `status: "queued"`. Pass `auto_transcribe=False`
    to register the file without kicking off work (e.g. for bulk import or when
    you plan to call `transcribe_meeting` later with different settings).
    """
    p = Path(audio_path).expanduser()
    if not p.is_absolute():
        raise ValueError(f"audio_path must be absolute: {audio_path}")
    if not p.exists():
        raise FileNotFoundError(f"audio file not found: {p}")
    if not p.is_file():
        raise ValueError(f"audio_path is not a file: {p}")

    conn = get_conn()
    row = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown project_id: {project_id}")

    mid = new_id()
    with tx(conn):
        conn.execute(
            "INSERT INTO meetings(id, project_id, title, date, audio_path, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (mid, project_id, title, date, str(p), now_iso()),
        )

    result = {
        "id": mid,
        "project_id": project_id,
        "title": title,
        "audio_path": str(p),
        "status": "pending",
    }
    if auto_transcribe:
        from ..config import get_settings
        from ..jobs import get_runner

        job_id = get_runner().submit_transcribe(mid, diarize=get_settings().diarize)
        result["job_id"] = job_id
        result["status"] = "queued"
    return result


@mcp.tool()
def update_meeting(
    meeting_id: str,
    title: str | None = None,
    date: str | None = None,
) -> dict:
    """Rename a meeting or change its date. At least one field required.

    Only user-owned fields are editable here — `audio_path`, `status`,
    `backend`, and `transcript_path` are system-owned (set by the job runner)
    and must not change after creation. To re-run transcription, use
    `retranscribe_meeting`.
    """
    if title is None and date is None:
        raise ValueError("must provide title and/or date")
    if title is not None and not title.strip():
        raise ValueError("title must be non-empty if provided")

    conn = get_conn()
    row = conn.execute("SELECT id FROM meetings WHERE id=?", (meeting_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown meeting_id: {meeting_id}")

    sets: list[str] = []
    params: list = []
    if title is not None:
        sets.append("title = ?")
        params.append(title.strip())
    if date is not None:
        sets.append("date = ?")
        params.append(date)
    params.append(meeting_id)
    with tx(conn):
        conn.execute(f"UPDATE meetings SET {', '.join(sets)} WHERE id = ?", params)

    return {"id": meeting_id, "updated_fields": [s.split(" = ")[0] for s in sets]}


@mcp.tool()
def get_meeting(meeting_id: str) -> dict:
    """Rich single-call overview of a meeting — metadata, project, speakers,
    series memberships, saved summary kinds, and document counts.

    Does NOT include transcript text (use `get_transcript`) or summary text
    (use `get_summary`). This tool is for "what do I know about this meeting
    before I decide what to fetch next?" — the cheapest orientation call.
    """
    conn = get_conn()
    m = conn.execute(
        """
        SELECT m.id, m.project_id, m.title, m.date, m.status, m.backend,
               m.duration_seconds, m.audio_path, m.transcript_path, m.created_at,
               p.name AS project_name
        FROM meetings m
        JOIN projects p ON p.id = m.project_id
        WHERE m.id = ?
        """,
        (meeting_id,),
    ).fetchone()
    if m is None:
        raise ValueError(f"unknown meeting_id: {meeting_id}")

    speakers = conn.execute(
        """
        SELECT s.label, s.name, s.person_id, p.name AS person_name,
               COUNT(c.id) AS segment_count
        FROM speakers s
        LEFT JOIN chunks  c ON c.speaker_id = s.id
        LEFT JOIN persons p ON p.id = s.person_id
        WHERE s.meeting_id = ?
        GROUP BY s.id
        ORDER BY s.label
        """,
        (meeting_id,),
    ).fetchall()

    series = conn.execute(
        """
        SELECT s.id, s.name
        FROM meeting_series_members sm
        JOIN meeting_series s ON s.id = sm.series_id
        WHERE sm.meeting_id = ?
        ORDER BY s.name
        """,
        (meeting_id,),
    ).fetchall()

    summary_kinds = [
        r["kind"]
        for r in conn.execute(
            "SELECT kind FROM summaries WHERE meeting_id = ? ORDER BY kind",
            (meeting_id,),
        ).fetchall()
    ]

    document_count = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE meeting_id = ?", (meeting_id,)
    ).fetchone()[0]

    chunk_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE meeting_id = ?", (meeting_id,)
    ).fetchone()[0]

    return {
        **dict(m),
        "chunk_count": chunk_count,
        "speakers": [dict(s) for s in speakers],
        "series": [dict(s) for s in series],
        "summary_kinds": summary_kinds,
        "document_count": document_count,
    }


@mcp.tool()
def list_meetings(project_id: str) -> list[dict]:
    """List all meetings in a project."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT id, title, date, status, backend, audio_path, duration_seconds, created_at
        FROM meetings
        WHERE project_id = ?
        ORDER BY COALESCE(date, created_at) DESC
        """,
        (project_id,),
    ).fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
def retranscribe_meeting(meeting_id: str) -> dict:
    """Clear existing transcript/speakers/chunks for a meeting and enqueue a fresh job.

    Useful after switching backends, correcting a mis-assigned audio file, or
    recovering from a failed transcription.
    """
    from ..config import get_settings
    from ..jobs import get_runner

    conn = get_conn()
    row = conn.execute(
        "SELECT id, status FROM meetings WHERE id=?", (meeting_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown meeting_id: {meeting_id}")
    if row["status"] == "transcribing":
        raise ValueError(f"meeting {meeting_id} already has a job in flight")

    with tx(conn):
        conn.execute("DELETE FROM chunks WHERE meeting_id=?", (meeting_id,))
        conn.execute("DELETE FROM speakers WHERE meeting_id=?", (meeting_id,))
        conn.execute(
            "UPDATE meetings SET status='pending', transcript_path=NULL, backend=NULL "
            "WHERE id=?",
            (meeting_id,),
        )
        # Saved summaries now describe an outdated transcript. Flag both the
        # per-meeting summaries and any series summaries whose rollup would
        # include this meeting. save_summary resets the flag on rewrite.
        conn.execute(
            "UPDATE summaries SET transcript_stale = 1 WHERE meeting_id = ?",
            (meeting_id,),
        )
        conn.execute(
            "UPDATE summaries SET transcript_stale = 1 WHERE series_id IN "
            "(SELECT series_id FROM meeting_series_members WHERE meeting_id = ?)",
            (meeting_id,),
        )

    runner = get_runner()
    job_id = runner.submit_transcribe(meeting_id, diarize=get_settings().diarize)
    return {"job_id": job_id, "meeting_id": meeting_id, "status": "queued"}


@mcp.tool()
def delete_meeting(meeting_id: str) -> dict:
    """Delete a meeting and all its transcripts/speakers/chunks/jobs.

    The audio file on disk is NOT deleted — the user owns it.
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT id, audio_path FROM meetings WHERE id = ?", (meeting_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown meeting_id: {meeting_id}")
    with tx(conn):
        conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    return {"deleted": meeting_id, "audio_path_left_on_disk": row["audio_path"]}


@mcp.tool()
def get_transcript(
    meeting_id: str,
    format: str = "text",
    speaker_labels: list[str] | None = None,
    speaker_names: list[str] | None = None,
    time_range: list[float] | None = None,
    max_chars: int | None = None,
) -> dict:
    """Return the transcript for a meeting. All filters are optional.

    Filters (combine with AND):
    - speaker_labels: only include these diarization labels, e.g. ["SPEAKER_00"]
    - speaker_names:  only include speakers whose assigned name is in this list
    - time_range:     [start_sec, end_sec] — only segments whose window overlaps

    Output shaping:
    - format: 'text' (speaker-labeled lines) or 'json' (segment list)
    - max_chars: truncate `text` output at this many chars; returned payload
      includes {"truncated": true, "total_chars": N} when it kicks in. For
      json format, max_chars caps the total length of concatenated segment
      text. Use this when you only need a sample, not the full content.

    For speaker-ID work, prefer get_speaker_cameos over calling this with
    filters — cameos are ~10x smaller for the same information.
    """
    if format not in ("text", "json"):
        raise ValueError("format must be 'text' or 'json'")
    if time_range is not None:
        if len(time_range) != 2 or time_range[0] >= time_range[1]:
            raise ValueError("time_range must be [start, end] with start < end")
    conn = get_conn()
    meeting = conn.execute(
        "SELECT id, title, status FROM meetings WHERE id = ?", (meeting_id,)
    ).fetchone()
    if meeting is None:
        raise ValueError(f"unknown meeting_id: {meeting_id}")
    if meeting["status"] != "ready":
        return {"status": meeting["status"], "transcript": None}

    where = ["c.meeting_id = ?"]
    params: list = [meeting_id]
    if speaker_labels:
        where.append(f"s.label IN ({','.join('?' * len(speaker_labels))})")
        params.extend(speaker_labels)
    if speaker_names:
        where.append(f"s.name IN ({','.join('?' * len(speaker_names))})")
        params.extend(speaker_names)
    if time_range is not None:
        where.append("c.end_time > ? AND c.start_time < ?")
        params.extend(time_range)

    rows = conn.execute(
        f"""
        SELECT c.text, c.start_time, c.end_time, s.label, s.name
        FROM chunks c
        LEFT JOIN speakers s ON s.id = c.speaker_id
        WHERE {' AND '.join(where)}
        ORDER BY c.start_time
        """,
        params,
    ).fetchall()

    segments = [
        {
            "start": r["start_time"],
            "end": r["end_time"],
            "speaker_label": r["label"],
            "speaker_name": r["name"],
            "text": r["text"],
        }
        for r in rows
    ]

    if format == "json":
        truncated = False
        total_chars = 0
        out_segs = []
        for seg in segments:
            total_chars += len(seg["text"])
            if max_chars is not None and total_chars > max_chars:
                truncated = True
                break
            out_segs.append(seg)
        payload = {"status": "ready", "segments": out_segs, "segment_count": len(out_segs)}
        if truncated:
            payload["truncated"] = True
            payload["total_chars"] = sum(len(s["text"]) for s in segments)
        return payload

    lines = []
    for s in segments:
        who = s["speaker_name"] or s["speaker_label"] or "UNKNOWN"
        ts = _fmt_ts(s["start"])
        lines.append(f"[{ts}] {who}: {s['text']}")
    text = "\n".join(lines)
    payload = {"status": "ready", "transcript": text, "segment_count": len(segments)}
    if max_chars is not None and len(text) > max_chars:
        payload["transcript"] = text[:max_chars].rstrip() + "\n... [truncated]"
        payload["truncated"] = True
        payload["total_chars"] = len(text)
    return payload


def _fmt_ts(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
