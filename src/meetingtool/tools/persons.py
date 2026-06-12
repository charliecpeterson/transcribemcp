"""Cross-meeting speaker identity.

A `person` is a canonical identity (Sarah Chen, Diego Reyes) that can be
linked to one or more per-meeting `speaker` rows. Diarization labels are
meeting-local (`SPEAKER_00`); persons are global. Link them with
`link_speaker_to_person` once you've identified who's who (usually via
`get_speaker_cameos`).
"""
from __future__ import annotations

from ..db import get_conn, new_id, now_iso, tx
from ..server import mcp


@mcp.tool()
def create_person(
    name: str,
    email: str | None = None,
    role: str | None = None,
    notes: str | None = None,
) -> dict:
    """Register a canonical person for cross-meeting identity.

    `email` / `role` / `notes` are optional free-form fields — useful for
    disambiguating common names and for giving the LLM context ("Sarah Chen,
    Engineering Director, sarah@example.com") when it's reasoning about
    speaker identity.
    """
    if not name or not name.strip():
        raise ValueError("name must be non-empty")
    pid = new_id()
    conn = get_conn()
    with tx(conn):
        conn.execute(
            "INSERT INTO persons(id, name, email, role, notes, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (pid, name.strip(), email, role, notes, now_iso()),
        )
    return {"id": pid, "name": name.strip(), "email": email, "role": role}


@mcp.tool()
def list_persons() -> list[dict]:
    """List all known persons with meeting counts (how many distinct meetings
    each person has been linked to via `speakers.person_id`)."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT p.id, p.name, p.email, p.role, p.notes, p.created_at,
               COUNT(DISTINCT s.meeting_id) AS meeting_count
        FROM persons p
        LEFT JOIN speakers s ON s.person_id = p.id
        GROUP BY p.id
        ORDER BY p.name
        """
    ).fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
def get_person(person_id: str) -> dict:
    """Return a person plus every meeting they've been linked to.

    The `meetings` list includes meeting id, title, date, and the local
    diarization label used in that meeting — useful when you need to pull
    only that speaker's segments via
    `get_transcript(..., speaker_labels=[...])`.
    """
    conn = get_conn()
    p = conn.execute(
        "SELECT id, name, email, role, notes, created_at FROM persons WHERE id=?",
        (person_id,),
    ).fetchone()
    if p is None:
        raise ValueError(f"unknown person_id: {person_id}")
    meetings = conn.execute(
        """
        SELECT m.id AS meeting_id, m.title, m.date, m.project_id,
               s.label AS speaker_label
        FROM speakers s
        JOIN meetings m ON m.id = s.meeting_id
        WHERE s.person_id = ?
        ORDER BY COALESCE(m.date, m.created_at) DESC
        """,
        (person_id,),
    ).fetchall()
    return {**dict(p), "meetings": [dict(r) for r in meetings]}


@mcp.tool()
def delete_person(person_id: str) -> dict:
    """Delete a person. Any speakers currently linked to them are unlinked
    (speakers.person_id → NULL) via trigger; their local names are kept."""
    conn = get_conn()
    row = conn.execute("SELECT id FROM persons WHERE id=?", (person_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown person_id: {person_id}")
    with tx(conn):
        conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
    return {"deleted": person_id}


@mcp.tool()
def link_speaker_to_person(
    meeting_id: str,
    label: str,
    person_id: str,
    copy_name: bool = True,
) -> dict:
    """Link a per-meeting speaker (diarization label) to a canonical person.

    If `copy_name` is True (default) and the speaker row has no local name
    yet, the person's name is copied into `speakers.name` so transcript
    rendering shows the person name immediately. Pass False to keep the
    person link without touching the local name.
    """
    conn = get_conn()
    speaker = conn.execute(
        "SELECT id, name FROM speakers WHERE meeting_id=? AND label=?",
        (meeting_id, label),
    ).fetchone()
    if speaker is None:
        raise ValueError(f"unknown speaker: meeting_id={meeting_id} label={label}")
    person = conn.execute(
        "SELECT id, name FROM persons WHERE id=?", (person_id,)
    ).fetchone()
    if person is None:
        raise ValueError(f"unknown person_id: {person_id}")

    with tx(conn):
        if copy_name and not speaker["name"]:
            conn.execute(
                "UPDATE speakers SET person_id=?, name=? WHERE id=?",
                (person_id, person["name"], speaker["id"]),
            )
        else:
            conn.execute(
                "UPDATE speakers SET person_id=? WHERE id=?",
                (person_id, speaker["id"]),
            )
    return {
        "meeting_id": meeting_id,
        "label": label,
        "person_id": person_id,
        "person_name": person["name"],
    }
