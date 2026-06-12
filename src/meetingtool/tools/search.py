"""Full-text search over transcript chunks.

Uses SQLite FTS5 (already wired via triggers in db.py). Returns snippets
with match highlighting, not full segments — keeps tokens tight when
exploring large meeting archives.
"""
from __future__ import annotations

from ..db import get_conn
from ..server import mcp


@mcp.tool()
def search_transcripts(
    query: str,
    project_id: str | None = None,
    meeting_id: str | None = None,
    person_id: str | None = None,
    series_id: str | None = None,
    limit: int = 10,
    snippet_chars: int = 80,
) -> dict:
    """Search transcript chunks with SQLite FTS5. Returns snippets, not full segments.

    `query` uses FTS5 syntax: simple terms ("budget"), phrases ("key decisions"),
    boolean ("budget AND timeline"), prefix ("hiring*"). See SQLite FTS5 docs
    for full grammar.

    Scope precedence — narrowest wins: meeting > series > project. If
    multiple are given, broader ones are silently ignored. `person_id` is
    orthogonal and always combines as AND.

    Scope filters:
    - meeting_id: single meeting
    - project_id: all meetings in a project
    - series_id:  all meetings in a given series
    - person_id:  only chunks spoken by the given cross-meeting person
      (requires that speakers have been linked via link_speaker_to_person)

    Each hit includes a short snippet (~snippet_chars chars around the match),
    speaker attribution, and timestamp, so you can decide whether to pull the
    full segment via get_transcript(time_range=[...]).
    """
    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    conn = get_conn()
    where = ["chunks_fts MATCH ?"]
    params: list = [query]
    if meeting_id:
        where.append("c.meeting_id = ?")
        params.append(meeting_id)
    elif series_id:
        where.append(
            "c.meeting_id IN (SELECT meeting_id FROM meeting_series_members "
            "WHERE series_id = ?)"
        )
        params.append(series_id)
    elif project_id:
        where.append("m.project_id = ?")
        params.append(project_id)
    if person_id:
        where.append("s.person_id = ?")
        params.append(person_id)
    params.append(limit)

    sql = f"""
        SELECT
            m.id      AS meeting_id,
            m.title   AS meeting_title,
            m.project_id,
            c.id      AS chunk_id,
            c.start_time,
            c.end_time,
            s.label   AS speaker_label,
            s.name    AS speaker_name,
            snippet(chunks_fts, 0, '<<', '>>', '...', ?) AS snippet
        FROM chunks_fts
        JOIN chunks  c ON c.rowid = chunks_fts.rowid
        JOIN meetings m ON m.id = c.meeting_id
        LEFT JOIN speakers s ON s.id = c.speaker_id
        WHERE {' AND '.join(where)}
        ORDER BY bm25(chunks_fts)
        LIMIT ?
        """
    # snippet() token-count arg goes first; we compute it from snippet_chars (~5 chars/token)
    token_budget = max(4, min(64, snippet_chars // 5))
    hits = conn.execute(sql, [token_budget, *params]).fetchall()

    return {
        "query": query,
        "count": len(hits),
        "hits": [
            {
                "meeting_id": h["meeting_id"],
                "meeting_title": h["meeting_title"],
                "project_id": h["project_id"],
                "chunk_id": h["chunk_id"],
                "start": h["start_time"],
                "end": h["end_time"],
                "speaker": h["speaker_name"] or h["speaker_label"],
                "snippet": h["snippet"],
            }
            for h in hits
        ],
    }
