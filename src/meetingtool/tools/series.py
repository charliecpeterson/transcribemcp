"""Meeting series — weekly 1:1s, interview loops, project stages.

A series is a named group of meetings within a project. Meetings appear in
a series in chronological order (by meeting date, falling back to created_at).
One meeting can belong to multiple series (e.g. both "Weekly leadership" and
"Q4 planning").
"""
from __future__ import annotations

from ..db import get_conn, new_id, now_iso, tx
from ..server import mcp


@mcp.tool()
def create_series(
    project_id: str,
    name: str,
    description: str | None = None,
) -> dict:
    """Create a named meeting series within a project.

    Examples: "Sarah 1:1 weekly", "Candidate: Alex Kim interview loop",
    "Platform migration standups".
    """
    if not name or not name.strip():
        raise ValueError("name must be non-empty")
    conn = get_conn()
    if conn.execute("SELECT 1 FROM projects WHERE id=?", (project_id,)).fetchone() is None:
        raise ValueError(f"unknown project_id: {project_id}")
    sid = new_id()
    with tx(conn):
        conn.execute(
            "INSERT INTO meeting_series(id, project_id, name, description, created_at) "
            "VALUES (?,?,?,?,?)",
            (sid, project_id, name.strip(), description, now_iso()),
        )
    return {"id": sid, "project_id": project_id, "name": name.strip()}


@mcp.tool()
def list_series(project_id: str | None = None) -> list[dict]:
    """List meeting series. Pass `project_id` to scope; otherwise returns all.

    Each row includes `meeting_count` — how many meetings are in the series.
    Metadata only, no transcript content.
    """
    conn = get_conn()
    sql = """
        SELECT s.id, s.project_id, s.name, s.description, s.created_at,
               COUNT(m.meeting_id) AS meeting_count
        FROM meeting_series s
        LEFT JOIN meeting_series_members m ON m.series_id = s.id
        {where}
        GROUP BY s.id
        ORDER BY s.name
    """
    params: list = []
    if project_id:
        sql = sql.format(where="WHERE s.project_id = ?")
        params.append(project_id)
    else:
        sql = sql.format(where="")
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


@mcp.tool()
def get_series(series_id: str) -> dict:
    """Rich single-call overview of a series — metadata, members, canonical
    persons appearing across members, saved summary kinds (series-scope),
    and a document count for anything attached to a member meeting.

    Meetings are ordered by `COALESCE(date, created_at)` ascending — oldest
    first, which matches the natural reading order for 1:1 histories and
    interview loops. Each meeting row includes `summary_kinds` and
    `chunk_count` so the caller can decide which meetings to pull full text
    from without an extra call per meeting.

    Does NOT include transcript text or summary text. Use `get_chat_context`
    for a chat-ready bundle, or `get_transcript` / `get_summary` for the
    specific piece you need.
    """
    conn = get_conn()
    s = conn.execute(
        "SELECT id, project_id, name, description, created_at "
        "FROM meeting_series WHERE id=?",
        (series_id,),
    ).fetchone()
    if s is None:
        raise ValueError(f"unknown series_id: {series_id}")

    meeting_rows = conn.execute(
        """
        SELECT m.id, m.title, m.date, m.status, m.duration_seconds,
               m.backend, m.created_at,
               (SELECT COUNT(*) FROM chunks c WHERE c.meeting_id = m.id) AS chunk_count
        FROM meeting_series_members sm
        JOIN meetings m ON m.id = sm.meeting_id
        WHERE sm.series_id = ?
        ORDER BY COALESCE(m.date, m.created_at) ASC
        """,
        (series_id,),
    ).fetchall()
    meetings: list[dict] = []
    total_duration = 0
    for r in meeting_rows:
        kinds = [
            row["kind"]
            for row in conn.execute(
                "SELECT kind FROM summaries WHERE meeting_id = ? ORDER BY kind",
                (r["id"],),
            ).fetchall()
        ]
        row = dict(r)
        row["summary_kinds"] = kinds
        meetings.append(row)
        if r["duration_seconds"]:
            total_duration += r["duration_seconds"]

    summary_kinds = [
        r["kind"]
        for r in conn.execute(
            "SELECT kind FROM summaries WHERE series_id = ? ORDER BY kind",
            (series_id,),
        ).fetchall()
    ]

    # Canonical persons appearing across any member meeting. A person
    # appears in the series if any speaker row for a member meeting links
    # to them. `meeting_count` is the count of distinct member meetings
    # they show up in — useful for "who are the regulars?" queries.
    persons = conn.execute(
        """
        SELECT p.id, p.name, p.email, p.role,
               COUNT(DISTINCT s.meeting_id) AS meeting_count
        FROM meeting_series_members sm
        JOIN speakers s ON s.meeting_id = sm.meeting_id AND s.person_id IS NOT NULL
        JOIN persons  p ON p.id = s.person_id
        WHERE sm.series_id = ?
        GROUP BY p.id
        ORDER BY meeting_count DESC, p.name
        """,
        (series_id,),
    ).fetchall()

    document_count = conn.execute(
        """
        SELECT COUNT(*) FROM documents d
        JOIN meeting_series_members sm ON sm.meeting_id = d.meeting_id
        WHERE sm.series_id = ?
        """,
        (series_id,),
    ).fetchone()[0]

    return {
        **dict(s),
        "meeting_count": len(meetings),
        "total_duration_seconds": total_duration,
        "summary_kinds": summary_kinds,
        "document_count": document_count,
        "persons": [dict(p) for p in persons],
        "meetings": meetings,
    }


@mcp.tool()
def add_meeting_to_series(series_id: str, meeting_id: str) -> dict:
    """Add a meeting to a series. Idempotent — re-adding an existing member
    is a no-op (UNIQUE PK prevents duplicates; we swallow the integrity error).

    The meeting and series must belong to the same project.
    """
    conn = get_conn()
    s = conn.execute(
        "SELECT id, project_id FROM meeting_series WHERE id=?", (series_id,)
    ).fetchone()
    if s is None:
        raise ValueError(f"unknown series_id: {series_id}")
    m = conn.execute(
        "SELECT id, project_id FROM meetings WHERE id=?", (meeting_id,)
    ).fetchone()
    if m is None:
        raise ValueError(f"unknown meeting_id: {meeting_id}")
    if m["project_id"] != s["project_id"]:
        raise ValueError(
            f"meeting {meeting_id} belongs to project {m['project_id']}, "
            f"series {series_id} belongs to project {s['project_id']}"
        )
    with tx(conn):
        conn.execute(
            "INSERT OR IGNORE INTO meeting_series_members(series_id, meeting_id, added_at) "
            "VALUES (?,?,?)",
            (series_id, meeting_id, now_iso()),
        )
    return {"series_id": series_id, "meeting_id": meeting_id}


@mcp.tool()
def remove_meeting_from_series(series_id: str, meeting_id: str) -> dict:
    """Remove a meeting from a series. Neither the series nor the meeting
    is deleted — only the membership row. No error if it wasn't a member."""
    conn = get_conn()
    with tx(conn):
        cur = conn.execute(
            "DELETE FROM meeting_series_members WHERE series_id=? AND meeting_id=?",
            (series_id, meeting_id),
        )
    return {"series_id": series_id, "meeting_id": meeting_id, "removed": cur.rowcount > 0}


@mcp.tool()
def delete_series(series_id: str) -> dict:
    """Delete a series. Meetings are untouched — only the series and its
    membership rows are removed."""
    conn = get_conn()
    row = conn.execute(
        "SELECT id FROM meeting_series WHERE id=?", (series_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown series_id: {series_id}")
    with tx(conn):
        conn.execute("DELETE FROM meeting_series WHERE id=?", (series_id,))
    return {"deleted": series_id}
