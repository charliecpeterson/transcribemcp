"""Cached summaries, scoped to either a single meeting or a series.

Claude Code does the actual summarization — reading transcripts, picking out
decisions, action items, topics. These tools just persist the result so
subsequent sessions don't need to re-read the full transcript. `kind` is
free-form ("overview", "action_items", "decisions", "notes", ...) — we don't
enforce a taxonomy, but upsert is keyed by (scope, kind).

Two scopes:
- **meeting summary**: one meeting's output. Cascades-delete with the meeting.
- **series summary**: a rollup across every meeting in a series. Useful for
  "state of the Weekly 1:1 as of week N" without stitching per-meeting
  summaries at retrieval time. Cascades-delete with the series.

Exactly one of (meeting_id, series_id) must be supplied on write/delete.
On read (list/search), either can be used as a filter, or both omitted to
return everything.

FTS5 indexes summary text so `search_summaries("budget freeze")` is O(log n)
across hundreds of meetings — the cheapest way to answer "which meetings
touched on X?" without scanning raw transcripts.
"""
from __future__ import annotations

from ..db import get_conn, new_id, now_iso, require_one_of, tx
from ..server import mcp


def _resolve_scope(
    meeting_id: str | None, series_id: str | None
) -> tuple[str, str, str]:
    """Returns (scope_col, scope_id, other_col). Validates exactly-one + existence."""
    require_one_of(meeting_id, series_id)
    conn = get_conn()
    if meeting_id is not None:
        if conn.execute("SELECT 1 FROM meetings WHERE id=?", (meeting_id,)).fetchone() is None:
            raise ValueError(f"unknown meeting_id: {meeting_id}")
        return "meeting_id", meeting_id, "series_id"
    if conn.execute("SELECT 1 FROM meeting_series WHERE id=?", (series_id,)).fetchone() is None:
        raise ValueError(f"unknown series_id: {series_id}")
    return "series_id", series_id, "meeting_id"


@mcp.tool()
def save_summary(
    meeting_id: str | None = None,
    kind: str = "",
    text: str = "",
    series_id: str | None = None,
) -> dict:
    """Upsert a summary for a meeting or a series. Replaces any existing
    summary with the same `kind` on the same scope; keeps `created_at` stable
    across edits.

    Exactly one of `meeting_id` or `series_id` must be provided.

    `kind` is a free-form label — suggested values: 'overview' (3-5 sentence
    TL;DR), 'action_items', 'decisions', 'notes'. Not enforced.
    """
    if not kind or not kind.strip():
        raise ValueError("kind must be non-empty")
    if not text or not text.strip():
        raise ValueError("text must be non-empty")

    scope_col, scope_id, _ = _resolve_scope(meeting_id, series_id)
    kind = kind.strip()
    ts = now_iso()

    conn = get_conn()
    existing = conn.execute(
        f"SELECT id, created_at FROM summaries WHERE {scope_col}=? AND kind=?",
        (scope_id, kind),
    ).fetchone()

    with tx(conn):
        if existing:
            conn.execute(
                "UPDATE summaries SET text=?, updated_at=?, transcript_stale=0 WHERE id=?",
                (text, ts, existing["id"]),
            )
            sid = existing["id"]
            created_at = existing["created_at"]
        else:
            sid = new_id()
            created_at = ts
            conn.execute(
                "INSERT INTO summaries(id, meeting_id, series_id, kind, text, "
                "created_at, updated_at, transcript_stale) "
                "VALUES (?,?,?,?,?,?,?,0)",
                (
                    sid,
                    scope_id if scope_col == "meeting_id" else None,
                    scope_id if scope_col == "series_id" else None,
                    kind,
                    text,
                    created_at,
                    ts,
                ),
            )
    return {
        "id": sid,
        "meeting_id": scope_id if scope_col == "meeting_id" else None,
        "series_id": scope_id if scope_col == "series_id" else None,
        "kind": kind,
        "created_at": created_at,
        "updated_at": ts,
        "transcript_stale": False,
        "replaced_existing": existing is not None,
    }


@mcp.tool()
def get_summary(
    meeting_id: str | None = None,
    kind: str | None = None,
    series_id: str | None = None,
) -> dict:
    """Retrieve summaries for a meeting or a series.

    Exactly one of `meeting_id` or `series_id` must be provided.

    - `kind` given: return that single summary (raises if missing).
    - `kind` omitted: return `{kind: text}` for every summary on this scope
      (empty dict if none have been saved yet).
    """
    scope_col, scope_id, _ = _resolve_scope(meeting_id, series_id)
    conn = get_conn()

    if kind is not None:
        row = conn.execute(
            f"SELECT id, kind, text, created_at, updated_at, transcript_stale "
            f"FROM summaries WHERE {scope_col}=? AND kind=?",
            (scope_id, kind),
        ).fetchone()
        if row is None:
            scope_label = "meeting" if scope_col == "meeting_id" else "series"
            raise ValueError(
                f"no summary of kind {kind!r} for {scope_label} {scope_id}"
            )
        payload = {scope_col: scope_id, **dict(row)}
        payload["transcript_stale"] = bool(payload["transcript_stale"])
        return payload

    rows = conn.execute(
        f"SELECT kind, text, created_at, updated_at, transcript_stale FROM summaries "
        f"WHERE {scope_col}=? ORDER BY kind",
        (scope_id,),
    ).fetchall()
    return {
        scope_col: scope_id,
        "summaries": {
            r["kind"]: {
                "text": r["text"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "transcript_stale": bool(r["transcript_stale"]),
            }
            for r in rows
        },
    }


@mcp.tool()
def list_summaries(
    project_id: str | None = None,
    meeting_id: str | None = None,
    series_id: str | None = None,
    kind: str | None = None,
) -> list[dict]:
    """List summary metadata (no text). Filter by project, meeting, series, and/or kind.

    Scope precedence — narrowest wins: meeting > series > project. If
    `meeting_id` is given, `series_id` and `project_id` are ignored; if
    `series_id` is given, `project_id` is ignored. `kind` is orthogonal and
    always combines as AND. Unfiltered returns both meeting and series
    summaries. Use `get_summary` / `search_summaries` to retrieve actual text.
    """
    conn = get_conn()
    where: list[str] = []
    params: list = []
    if meeting_id:
        where.append("s.meeting_id = ?")
        params.append(meeting_id)
    elif series_id:
        where.append("s.series_id = ?")
        params.append(series_id)
    elif project_id:
        where.append("(m.project_id = ? OR ms.project_id = ?)")
        params.extend([project_id, project_id])
    if kind:
        where.append("s.kind = ?")
        params.append(kind)

    sql = """
        SELECT
            s.id,
            s.meeting_id,
            s.series_id,
            m.title  AS meeting_title,
            ms.name  AS series_name,
            s.kind,
            LENGTH(s.text) AS char_count,
            s.transcript_stale,
            s.created_at,
            s.updated_at
        FROM summaries s
        LEFT JOIN meetings       m  ON m.id  = s.meeting_id
        LEFT JOIN meeting_series ms ON ms.id = s.series_id
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY s.updated_at DESC"
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    for r in rows:
        r["transcript_stale"] = bool(r["transcript_stale"])
    return rows


@mcp.tool()
def delete_summary(
    meeting_id: str | None = None,
    kind: str = "",
    series_id: str | None = None,
) -> dict:
    """Delete a single (scope, kind) summary. No error if it doesn't exist.

    Exactly one of `meeting_id` or `series_id` must be provided.
    """
    if not kind or not kind.strip():
        raise ValueError("kind must be non-empty")
    scope_col, scope_id, _ = _resolve_scope(meeting_id, series_id)

    conn = get_conn()
    with tx(conn):
        cur = conn.execute(
            f"DELETE FROM summaries WHERE {scope_col}=? AND kind=?",
            (scope_id, kind),
        )
    return {
        "meeting_id": scope_id if scope_col == "meeting_id" else None,
        "series_id": scope_id if scope_col == "series_id" else None,
        "kind": kind,
        "deleted": cur.rowcount > 0,
    }


@mcp.tool()
def search_summaries(
    query: str,
    project_id: str | None = None,
    meeting_id: str | None = None,
    series_id: str | None = None,
    kind: str | None = None,
    limit: int = 10,
    snippet_chars: int = 120,
) -> dict:
    """FTS5 search over saved summaries (meeting-level AND series-level).
    Returns snippets ranked by bm25.

    Scope precedence — narrowest wins: meeting > series > project. `kind`
    combines as AND if provided.

    Summaries are short, so snippet_chars defaults a bit higher (120) than
    for transcript search. Each hit gives you scope + kind, so followups go
    through get_summary for full text.
    """
    if not query or not query.strip():
        raise ValueError("query must be non-empty")
    where = ["summaries_fts MATCH ?"]
    params: list = [query]
    if meeting_id:
        where.append("s.meeting_id = ?")
        params.append(meeting_id)
    elif series_id:
        where.append("s.series_id = ?")
        params.append(series_id)
    elif project_id:
        where.append("(m.project_id = ? OR ms.project_id = ?)")
        params.extend([project_id, project_id])
    if kind:
        where.append("s.kind = ?")
        params.append(kind)
    params.append(limit)

    token_budget = max(4, min(64, snippet_chars // 5))
    sql = f"""
        SELECT
            s.id AS summary_id,
            s.meeting_id,
            s.series_id,
            m.title  AS meeting_title,
            ms.name  AS series_name,
            COALESCE(m.project_id, ms.project_id) AS project_id,
            s.kind,
            snippet(summaries_fts, 0, '<<', '>>', '...', ?) AS snippet
        FROM summaries_fts
        JOIN summaries s         ON s.rowid = summaries_fts.rowid
        LEFT JOIN meetings       m  ON m.id  = s.meeting_id
        LEFT JOIN meeting_series ms ON ms.id = s.series_id
        WHERE {' AND '.join(where)}
        ORDER BY bm25(summaries_fts)
        LIMIT ?
    """
    conn = get_conn()
    hits = conn.execute(sql, [token_budget, *params]).fetchall()
    return {
        "query": query,
        "count": len(hits),
        "hits": [dict(h) for h in hits],
    }
