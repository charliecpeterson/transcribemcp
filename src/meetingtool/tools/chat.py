"""One-call context bundle for chat sessions.

`get_chat_context` is the "I want to chat about this meeting / series —
give me everything Claude needs in a single payload" shortcut. It exists
because the alternative is orchestrating five-plus calls (`get_meeting` +
`get_transcript` + `get_summary` + `list_documents` + `get_document` ×N)
every time a conversation starts, which is both slow and easy to get
wrong.

Design notes:
- Summaries are always included — they're small and high-density.
- Transcripts are optional because a series with 20 meetings easily blows
  a context window. Default on for meeting scope, off for series scope.
- Documents are optional (full text). Off by default because docs can be
  large and aren't always load-bearing for a chat.
- Stale summary flags and per-meeting summary_kinds are preserved so the
  LLM knows when to ignore a rotted summary.

Exactly one of (meeting_id, series_id) must be provided.
"""
from __future__ import annotations

from ..db import get_conn, require_one_of
from ..server import mcp
from .meetings import get_transcript


def _collect_speakers(conn, meeting_id: str) -> list[dict]:
    rows = conn.execute(
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
    return [dict(r) for r in rows]


def _collect_summaries_for_meeting(conn, meeting_id: str) -> dict:
    rows = conn.execute(
        "SELECT kind, text, transcript_stale, created_at, updated_at "
        "FROM summaries WHERE meeting_id = ? ORDER BY kind",
        (meeting_id,),
    ).fetchall()
    return {
        r["kind"]: {
            "text": r["text"],
            "transcript_stale": bool(r["transcript_stale"]),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    }


def _collect_summaries_for_series(conn, series_id: str) -> dict:
    rows = conn.execute(
        "SELECT kind, text, transcript_stale, created_at, updated_at "
        "FROM summaries WHERE series_id = ? ORDER BY kind",
        (series_id,),
    ).fetchall()
    return {
        r["kind"]: {
            "text": r["text"],
            "transcript_stale": bool(r["transcript_stale"]),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    }


def _collect_documents(conn, meeting_ids: list[str], *, with_text: bool) -> list[dict]:
    """Docs attached to any of the given meetings. If with_text=False,
    returns metadata only (cheap orientation call)."""
    if not meeting_ids:
        return []
    placeholders = ",".join("?" * len(meeting_ids))
    rows = conn.execute(
        f"""
        SELECT id, meeting_id, title, kind, char_count, chunk_count, created_at
        FROM documents WHERE meeting_id IN ({placeholders})
        ORDER BY created_at
        """,
        meeting_ids,
    ).fetchall()
    docs: list[dict] = []
    for r in rows:
        doc = dict(r)
        if with_text:
            chunk_rows = conn.execute(
                "SELECT text FROM document_chunks WHERE document_id=? ORDER BY ord",
                (r["id"],),
            ).fetchall()
            doc["text"] = "\n\n".join(c["text"] for c in chunk_rows)
        docs.append(doc)
    return docs


@mcp.tool()
def get_chat_context(
    meeting_id: str | None = None,
    series_id: str | None = None,
    include_transcripts: bool | None = None,
    include_documents: bool = False,
    transcript_max_chars: int | None = None,
) -> dict:
    """Bundle the context Claude needs to chat about a meeting or a series.

    Exactly one of `meeting_id` or `series_id` must be provided.

    Always returned:
    - Scope metadata (project, speakers / persons, series membership)
    - All saved summaries on this scope, with `transcript_stale` flags
    - For meeting scope: summaries from any series the meeting belongs to
      (useful broader context — small surface, often load-bearing)
    - For series scope: per-member-meeting summaries (the high-value bit)

    Optional (default chosen to avoid accidental context-window blowups):
    - `include_transcripts`: pull speaker-labeled transcript text.
      Default True for meeting scope, False for series scope (a 20-meeting
      weekly 1:1 easily runs to hundreds of KB). Pass True explicitly for
      a series when you actually want the raw text.
    - `include_documents`: full text of documents attached to the
      meeting(s). Default False. When False, you still get document
      metadata (title, kind, char_count) so you know what to ask for.
    - `transcript_max_chars`: per-transcript char cap. Useful when pulling
      a series with `include_transcripts=True`.

    If you only need metadata (no text), call `get_meeting` / `get_series`.
    If you need one specific piece (transcript-only, summary-only), call
    the targeted tool. `get_chat_context` is the "starting a chat session"
    shortcut.
    """
    require_one_of(meeting_id, series_id)
    conn = get_conn()

    if meeting_id is not None:
        return _meeting_context(
            conn,
            meeting_id,
            include_transcripts=True if include_transcripts is None else include_transcripts,
            include_documents=include_documents,
            transcript_max_chars=transcript_max_chars,
        )
    return _series_context(
        conn,
        series_id,  # type: ignore[arg-type]
        include_transcripts=False if include_transcripts is None else include_transcripts,
        include_documents=include_documents,
        transcript_max_chars=transcript_max_chars,
    )


def _meeting_context(
    conn,
    meeting_id: str,
    *,
    include_transcripts: bool,
    include_documents: bool,
    transcript_max_chars: int | None,
) -> dict:
    m = conn.execute(
        """
        SELECT m.id, m.project_id, m.title, m.date, m.status, m.backend,
               m.duration_seconds, m.created_at,
               p.name AS project_name
        FROM meetings m
        JOIN projects p ON p.id = m.project_id
        WHERE m.id = ?
        """,
        (meeting_id,),
    ).fetchone()
    if m is None:
        raise ValueError(f"unknown meeting_id: {meeting_id}")

    series_rows = conn.execute(
        """
        SELECT s.id, s.name
        FROM meeting_series_members sm
        JOIN meeting_series s ON s.id = sm.series_id
        WHERE sm.meeting_id = ?
        ORDER BY s.name
        """,
        (meeting_id,),
    ).fetchall()
    series = [dict(s) for s in series_rows]

    # Summaries from any series this meeting belongs to — surprisingly
    # load-bearing context ("last week you said X in the rollup").
    series_summaries: dict[str, dict] = {}
    for s in series:
        s_sums = _collect_summaries_for_series(conn, s["id"])
        if s_sums:
            series_summaries[s["id"]] = {"series_name": s["name"], "summaries": s_sums}

    bundle: dict = {
        "scope": "meeting",
        "meeting": dict(m),
        "speakers": _collect_speakers(conn, meeting_id),
        "series": series,
        "summaries": _collect_summaries_for_meeting(conn, meeting_id),
        "series_summaries": series_summaries,
    }

    if include_transcripts:
        t = get_transcript(meeting_id, max_chars=transcript_max_chars)
        bundle["transcript"] = t

    bundle["documents"] = _collect_documents(
        conn, [meeting_id], with_text=include_documents
    )
    return bundle


def _series_context(
    conn,
    series_id: str,
    *,
    include_transcripts: bool,
    include_documents: bool,
    transcript_max_chars: int | None,
) -> dict:
    s = conn.execute(
        """
        SELECT s.id, s.project_id, s.name, s.description, s.created_at,
               p.name AS project_name
        FROM meeting_series s
        JOIN projects p ON p.id = s.project_id
        WHERE s.id = ?
        """,
        (series_id,),
    ).fetchone()
    if s is None:
        raise ValueError(f"unknown series_id: {series_id}")

    meeting_rows = conn.execute(
        """
        SELECT m.id, m.title, m.date, m.status, m.duration_seconds, m.backend,
               m.created_at,
               (SELECT COUNT(*) FROM chunks c WHERE c.meeting_id = m.id) AS chunk_count
        FROM meeting_series_members sm
        JOIN meetings m ON m.id = sm.meeting_id
        WHERE sm.series_id = ?
        ORDER BY COALESCE(m.date, m.created_at) ASC
        """,
        (series_id,),
    ).fetchall()

    meetings: list[dict] = []
    for r in meeting_rows:
        entry = dict(r)
        entry["speakers"] = _collect_speakers(conn, r["id"])
        entry["summaries"] = _collect_summaries_for_meeting(conn, r["id"])
        if include_transcripts:
            entry["transcript"] = get_transcript(
                r["id"], max_chars=transcript_max_chars
            )
        meetings.append(entry)

    persons = conn.execute(
        """
        SELECT p.id, p.name, p.email, p.role,
               COUNT(DISTINCT sp.meeting_id) AS meeting_count
        FROM meeting_series_members sm
        JOIN speakers sp ON sp.meeting_id = sm.meeting_id AND sp.person_id IS NOT NULL
        JOIN persons  p  ON p.id = sp.person_id
        WHERE sm.series_id = ?
        GROUP BY p.id
        ORDER BY meeting_count DESC, p.name
        """,
        (series_id,),
    ).fetchall()

    member_ids = [r["id"] for r in meeting_rows]
    documents = _collect_documents(conn, member_ids, with_text=include_documents)

    return {
        "scope": "series",
        "series": dict(s),
        "summaries": _collect_summaries_for_series(conn, series_id),
        "persons": [dict(p) for p in persons],
        "meetings": meetings,
        "documents": documents,
    }
