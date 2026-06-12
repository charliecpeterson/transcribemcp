from ..db import get_conn, new_id, now_iso, tx
from ..server import mcp


@mcp.tool()
def create_project(name: str, description: str = "") -> dict:
    """Create a named project (e.g. 'Q2 Hiring', 'Jones Investigation')."""
    conn = get_conn()
    pid = new_id()
    with tx(conn):
        conn.execute(
            "INSERT INTO projects(id, name, description, created_at) VALUES (?, ?, ?, ?)",
            (pid, name, description, now_iso()),
        )
    return {"id": pid, "name": name, "description": description}


@mcp.tool()
def list_projects() -> list[dict]:
    """List all projects with meeting counts."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT p.id, p.name, p.description, p.created_at,
               (SELECT COUNT(*) FROM meetings m WHERE m.project_id = p.id) AS meeting_count
        FROM projects p
        ORDER BY p.created_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
def update_project(
    project_id: str,
    name: str | None = None,
    description: str | None = None,
) -> dict:
    """Rename a project or update its description. At least one field required.

    Pass `description=""` to clear the description; pass `None` (the default)
    to leave it unchanged.
    """
    if name is None and description is None:
        raise ValueError("must provide name and/or description")
    if name is not None and not name.strip():
        raise ValueError("name must be non-empty if provided")

    conn = get_conn()
    row = conn.execute("SELECT id FROM projects WHERE id=?", (project_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown project_id: {project_id}")

    sets: list[str] = []
    params: list = []
    if name is not None:
        sets.append("name = ?")
        params.append(name.strip())
    if description is not None:
        sets.append("description = ?")
        params.append(description)
    params.append(project_id)
    with tx(conn):
        conn.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", params)

    return {"id": project_id, "updated_fields": [s.split(" = ")[0] for s in sets]}


@mcp.tool()
def delete_project(project_id: str) -> dict:
    """Delete a project and all its meetings, transcripts, speakers, chunks,
    documents, series, and summaries. Audio files on disk are NOT touched.

    Returns counts of what was removed so the caller can sanity-check the
    blast radius before reporting to the user.
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT id, name FROM projects WHERE id=?", (project_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown project_id: {project_id}")

    counts = {
        "meetings": conn.execute(
            "SELECT COUNT(*) FROM meetings WHERE project_id=?", (project_id,)
        ).fetchone()[0],
        "documents": conn.execute(
            "SELECT COUNT(*) FROM documents WHERE project_id=?", (project_id,)
        ).fetchone()[0],
        "series": conn.execute(
            "SELECT COUNT(*) FROM meeting_series WHERE project_id=?", (project_id,)
        ).fetchone()[0],
    }
    with tx(conn):
        conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
    return {
        "deleted": project_id,
        "name": row["name"],
        "removed_counts": counts,
    }
