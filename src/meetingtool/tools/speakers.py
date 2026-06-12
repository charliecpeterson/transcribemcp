"""Speaker-identity tools.

The intent is that Claude Code does the actual identification work — reading
the transcript (and any attached docs) and deciding who's who. Our job is to
(a) make that reasoning cheap by providing *cameos* with high signal per token,
and (b) persist the answer via assign_speaker.
"""
from __future__ import annotations

from ..db import get_conn, new_id, now_iso, tx
from ..server import mcp


@mcp.tool()
def list_speakers(meeting_id: str) -> list[dict]:
    """List all speaker labels detected in a meeting, with any human-assigned names.

    Cheap to call — returns metadata only, no transcript text.
    """
    conn = get_conn()
    row = conn.execute("SELECT id FROM meetings WHERE id=?", (meeting_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown meeting_id: {meeting_id}")
    rows = conn.execute(
        """
        SELECT s.id, s.label, s.name, s.notes,
               s.person_id, p.name AS person_name,
               COUNT(c.id)      AS segment_count,
               SUM(COALESCE(c.end_time, 0) - COALESCE(c.start_time, 0)) AS total_seconds
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


@mcp.tool()
def assign_speaker(
    meeting_id: str,
    label: str,
    name: str,
    notes: str | None = None,
) -> dict:
    """Assign a human-readable name to a diarization label (e.g. SPEAKER_00 -> 'Sarah - HR').

    `label` must match an existing speaker label in the meeting. Pass notes to
    record why (e.g. 'self-introduced at 0:12' or 'confirmed via attendee list').
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT id FROM speakers WHERE meeting_id=? AND label=?",
        (meeting_id, label),
    ).fetchone()
    if row is None:
        raise ValueError(
            f"no speaker with label={label!r} in meeting {meeting_id}. "
            f"Call list_speakers to see available labels."
        )
    with tx(conn):
        conn.execute(
            "UPDATE speakers SET name=?, notes=? WHERE id=?",
            (name, notes, row["id"]),
        )
    return {"speaker_id": row["id"], "label": label, "name": name}


@mcp.tool()
def get_speaker_cameos(
    meeting_id: str,
    n_per_speaker: int = 3,
    max_chars_per_cameo: int = 400,
    only_unnamed: bool = True,
    include_attached_docs: bool = False,
    max_chars_per_doc: int = 2000,
) -> dict:
    """Return the first N utterances per speaker — the high-signal evidence for speaker-ID.

    By default returns only speakers without an assigned name (`only_unnamed=True`),
    so the LLM can focus on what's unresolved.

    Pass `include_attached_docs=True` to also return the text of any documents
    linked to this meeting (attendee lists, agendas, notes). This is the
    one-shot call to make when you want to identify speakers using both
    transcript evidence and out-of-band context. Each doc is truncated to
    `max_chars_per_doc`; use get_document(format="text") for full text.

    This is deliberately small on tokens. For a typical meeting with 3 unnamed
    speakers and n_per_speaker=3, total output is ~1–2 KB (plus up to
    max_chars_per_doc per attached doc). Prefer this over dumping the full
    transcript when the question is 'who are these speakers'.

    Each cameo's text is truncated to max_chars_per_cameo; long utterances end with '...'.
    """
    if n_per_speaker < 1:
        raise ValueError("n_per_speaker must be >= 1")

    conn = get_conn()
    m = conn.execute(
        "SELECT id, title FROM meetings WHERE id=?", (meeting_id,)
    ).fetchone()
    if m is None:
        raise ValueError(f"unknown meeting_id: {meeting_id}")

    speaker_filter = "AND s.name IS NULL AND s.person_id IS NULL" if only_unnamed else ""
    speakers = conn.execute(
        f"""
        SELECT s.id, s.label, s.name, s.person_id, p.name AS person_name
        FROM speakers s
        LEFT JOIN persons p ON p.id = s.person_id
        WHERE s.meeting_id = ? {speaker_filter}
        ORDER BY s.label
        """,
        (meeting_id,),
    ).fetchall()

    cameos = []
    for sp in speakers:
        rows = conn.execute(
            """
            SELECT start_time, end_time, text FROM chunks
            WHERE meeting_id = ? AND speaker_id = ?
            ORDER BY start_time
            LIMIT ?
            """,
            (meeting_id, sp["id"], n_per_speaker),
        ).fetchall()
        utterances = []
        for r in rows:
            text = r["text"]
            if len(text) > max_chars_per_cameo:
                text = text[: max_chars_per_cameo - 3].rstrip() + "..."
            utterances.append({
                "start": r["start_time"],
                "end": r["end_time"],
                "text": text,
            })
        cameos.append({
            "label": sp["label"],
            "name": sp["name"],
            "person_id": sp["person_id"],
            "person_name": sp["person_name"],
            "utterances": utterances,
        })

    payload: dict = {
        "meeting_id": meeting_id,
        "meeting_title": m["title"],
        "speakers": cameos,
        "hint": (
            "To name a speaker locally: assign_speaker(meeting_id, label, name). "
            "To link to a known cross-meeting person: "
            "link_speaker_to_person(meeting_id, label, person_id) — see list_persons."
        ),
    }

    if include_attached_docs:
        docs = conn.execute(
            """
            SELECT id, title, kind, char_count
            FROM documents
            WHERE meeting_id = ?
            ORDER BY created_at
            """,
            (meeting_id,),
        ).fetchall()
        attached = []
        for d in docs:
            parts = conn.execute(
                "SELECT text FROM document_chunks WHERE document_id = ? ORDER BY ord",
                (d["id"],),
            ).fetchall()
            full = "\n\n".join(r["text"] for r in parts)
            truncated = len(full) > max_chars_per_doc
            attached.append({
                "id": d["id"],
                "title": d["title"],
                "kind": d["kind"],
                "text": full[:max_chars_per_doc] if truncated else full,
                "truncated": truncated,
                "total_chars": len(full),
            })
        payload["attached_documents"] = attached

    return payload
