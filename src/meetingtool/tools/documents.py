"""Document ingestion + retrieval.

Supplemental docs (agendas, meeting notes, plans) are chunked at ingest time
and indexed in SQLite FTS5. Retrieval defaults to metadata; full text must
be asked for explicitly. search_documents returns snippets — the token-lean
path for "does this doc mention X?" questions.
"""
from __future__ import annotations

from pathlib import Path

from ..db import get_conn, new_id, now_iso, tx
from ..documents import chunk_text, extract_text
from ..server import mcp


@mcp.tool()
def add_document(
    project_id: str,
    title: str,
    path: str,
    meeting_id: str | None = None,
) -> dict:
    """Ingest a supplemental document (.txt, .md, .pdf, .docx) under a project.

    The file is read, extracted to text, and chunked (~800 chars, paragraph
    boundaries preserved) for FTS retrieval. Optionally bind to a specific
    meeting (e.g. the agenda or notes for that session).

    Returns metadata only — never the full text. Use get_document or
    search_documents to retrieve content.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        raise ValueError(f"path must be absolute: {path}")
    if not p.exists():
        raise FileNotFoundError(f"document not found: {p}")
    if not p.is_file():
        raise ValueError(f"path is not a file: {p}")

    conn = get_conn()
    if conn.execute("SELECT 1 FROM projects WHERE id=?", (project_id,)).fetchone() is None:
        raise ValueError(f"unknown project_id: {project_id}")
    if meeting_id is not None:
        if conn.execute("SELECT 1 FROM meetings WHERE id=?", (meeting_id,)).fetchone() is None:
            raise ValueError(f"unknown meeting_id: {meeting_id}")

    text, kind = extract_text(p)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"document is empty after extraction: {p}")

    doc_id = new_id()
    ts = now_iso()
    with tx(conn):
        conn.execute(
            "INSERT INTO documents(id, project_id, meeting_id, title, kind, "
            "source_path, char_count, chunk_count, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, project_id, meeting_id, title, kind, str(p),
             len(text), len(chunks), ts),
        )
        conn.executemany(
            "INSERT INTO document_chunks(id, document_id, ord, text) VALUES (?,?,?,?)",
            [(new_id(), doc_id, i, chunk) for i, chunk in enumerate(chunks)],
        )
    return {
        "id": doc_id,
        "project_id": project_id,
        "meeting_id": meeting_id,
        "title": title,
        "kind": kind,
        "char_count": len(text),
        "chunk_count": len(chunks),
    }


@mcp.tool()
def list_documents(
    project_id: str | None = None,
    meeting_id: str | None = None,
) -> list[dict]:
    """List documents, optionally scoped to a project and/or meeting.

    Metadata only — no text is returned. Use search_documents or get_document
    to retrieve content.
    """
    conn = get_conn()
    where: list[str] = []
    params: list = []
    if project_id:
        where.append("project_id = ?")
        params.append(project_id)
    if meeting_id:
        where.append("meeting_id = ?")
        params.append(meeting_id)
    sql = (
        "SELECT id, project_id, meeting_id, title, kind, char_count, "
        "chunk_count, created_at FROM documents"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


@mcp.tool()
def get_document(
    document_id: str,
    format: str = "meta",
    chunk_ords: list[int] | None = None,
    max_chars: int | None = None,
) -> dict:
    """Retrieve a document.

    Formats:
    - 'meta'   (default): metadata only — title, kind, char_count, chunk_count.
      The lowest-cost call; start here.
    - 'text'  : concatenated full text. Pass `max_chars` to cap size or
      `chunk_ords` to request specific chunk ordinals from search hits.
    - 'chunks': structured list of {ord, text} objects — useful when you need
      stable addressing back into a chunk after search_documents.

    For large docs prefer search_documents to find relevant chunks, then call
    get_document(..., format='chunks', chunk_ords=[...]) to pull only those.
    """
    if format not in ("meta", "text", "chunks"):
        raise ValueError("format must be 'meta', 'text', or 'chunks'")
    conn = get_conn()
    doc = conn.execute(
        "SELECT id, project_id, meeting_id, title, kind, char_count, "
        "chunk_count, source_path, created_at FROM documents WHERE id=?",
        (document_id,),
    ).fetchone()
    if doc is None:
        raise ValueError(f"unknown document_id: {document_id}")

    meta = dict(doc)
    if format == "meta":
        return meta

    where = ["document_id = ?"]
    params: list = [document_id]
    if chunk_ords is not None:
        if not chunk_ords:
            raise ValueError("chunk_ords must be a non-empty list or None")
        where.append(f"ord IN ({','.join('?' * len(chunk_ords))})")
        params.extend(chunk_ords)

    rows = conn.execute(
        f"SELECT ord, text FROM document_chunks WHERE {' AND '.join(where)} "
        "ORDER BY ord",
        params,
    ).fetchall()

    if format == "chunks":
        out: list[dict] = []
        total = 0
        truncated = False
        for r in rows:
            total += len(r["text"])
            if max_chars is not None and total > max_chars:
                truncated = True
                break
            out.append({"ord": r["ord"], "text": r["text"]})
        payload = {**meta, "chunks": out}
        if truncated:
            payload["truncated"] = True
        return payload

    text = "\n\n".join(r["text"] for r in rows)
    payload = {**meta, "text": text}
    if max_chars is not None and len(text) > max_chars:
        payload["text"] = text[:max_chars].rstrip() + "\n... [truncated]"
        payload["truncated"] = True
        payload["total_chars"] = len(text)
    return payload


@mcp.tool()
def delete_document(document_id: str) -> dict:
    """Delete a document and its chunks. The source file on disk is untouched."""
    conn = get_conn()
    row = conn.execute(
        "SELECT id, source_path FROM documents WHERE id=?", (document_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown document_id: {document_id}")
    with tx(conn):
        conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
    return {"deleted": document_id, "source_path_left_on_disk": row["source_path"]}


@mcp.tool()
def search_documents(
    query: str,
    project_id: str | None = None,
    meeting_id: str | None = None,
    document_id: str | None = None,
    limit: int = 10,
    snippet_chars: int = 80,
) -> dict:
    """FTS5 search across document chunks. Returns ranked snippets, not full text.

    `query` uses FTS5 syntax (see search_transcripts).

    Scope precedence — narrowest wins: document > meeting > project. If
    multiple are given, broader ones are silently ignored.

    Each hit carries a `chunk_ord` you can feed back to
    get_document(format='chunks', chunk_ords=[...]) to pull the full chunk
    if the snippet isn't enough.
    """
    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    conn = get_conn()
    where = ["document_chunks_fts MATCH ?"]
    params: list = [query]
    if document_id:
        where.append("c.document_id = ?")
        params.append(document_id)
    elif meeting_id:
        where.append("d.meeting_id = ?")
        params.append(meeting_id)
    elif project_id:
        where.append("d.project_id = ?")
        params.append(project_id)
    params.append(limit)

    token_budget = max(4, min(64, snippet_chars // 5))
    sql = f"""
        SELECT
            d.id      AS document_id,
            d.title   AS document_title,
            d.kind    AS document_kind,
            d.project_id,
            d.meeting_id,
            c.id      AS chunk_id,
            c.ord     AS chunk_ord,
            snippet(document_chunks_fts, 0, '<<', '>>', '...', ?) AS snippet
        FROM document_chunks_fts
        JOIN document_chunks c ON c.rowid = document_chunks_fts.rowid
        JOIN documents       d ON d.id = c.document_id
        WHERE {' AND '.join(where)}
        ORDER BY bm25(document_chunks_fts)
        LIMIT ?
        """
    hits = conn.execute(sql, [token_budget, *params]).fetchall()

    return {
        "query": query,
        "count": len(hits),
        "hits": [
            {
                "document_id": h["document_id"],
                "document_title": h["document_title"],
                "document_kind": h["document_kind"],
                "project_id": h["project_id"],
                "meeting_id": h["meeting_id"],
                "chunk_id": h["chunk_id"],
                "chunk_ord": h["chunk_ord"],
                "snippet": h["snippet"],
            }
            for h in hits
        ],
    }
