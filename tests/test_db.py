from meetingtool import db as db_mod


def test_schema_round_trip(conn):
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == db_mod.SCHEMA_VERSION

    pid = db_mod.new_id()
    conn.execute(
        "INSERT INTO projects(id, name, description, created_at) VALUES (?, ?, ?, ?)",
        (pid, "Hiring Q2", "desc", db_mod.now_iso()),
    )
    mid = db_mod.new_id()
    conn.execute(
        "INSERT INTO meetings(id, project_id, title, audio_path, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (mid, pid, "Interview", "/tmp/x.wav", "pending", db_mod.now_iso()),
    )
    rows = conn.execute(
        "SELECT title FROM meetings WHERE project_id = ?", (pid,)
    ).fetchall()
    assert [r["title"] for r in rows] == ["Interview"]


def test_fts_in_sync_with_chunks(conn):
    pid, mid = db_mod.new_id(), db_mod.new_id()
    conn.execute(
        "INSERT INTO projects(id, name, created_at) VALUES (?, ?, ?)",
        (pid, "P", db_mod.now_iso()),
    )
    conn.execute(
        "INSERT INTO meetings(id, project_id, title, audio_path, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (mid, pid, "M", "/tmp/a.wav", db_mod.now_iso()),
    )
    cid = db_mod.new_id()
    conn.execute(
        "INSERT INTO chunks(id, meeting_id, text, start_time, end_time) "
        "VALUES (?, ?, ?, ?, ?)",
        (cid, mid, "the budget overrun was the main concern", 0.0, 5.0),
    )

    hit = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'budget'"
    ).fetchall()
    assert len(hit) == 1

    conn.execute("UPDATE chunks SET text = 'nothing to see here' WHERE id = ?", (cid,))
    hit_after = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'budget'"
    ).fetchall()
    assert len(hit_after) == 0

    conn.execute("DELETE FROM chunks WHERE id = ?", (cid,))
    hit_final = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'nothing'"
    ).fetchall()
    assert len(hit_final) == 0


def test_cascade_delete_meeting(conn):
    pid, mid, sid = db_mod.new_id(), db_mod.new_id(), db_mod.new_id()
    conn.execute(
        "INSERT INTO projects(id, name, created_at) VALUES (?, ?, ?)",
        (pid, "P", db_mod.now_iso()),
    )
    conn.execute(
        "INSERT INTO meetings(id, project_id, title, audio_path, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (mid, pid, "M", "/tmp/a.wav", db_mod.now_iso()),
    )
    conn.execute(
        "INSERT INTO speakers(id, meeting_id, label) VALUES (?, ?, ?)",
        (sid, mid, "SPEAKER_00"),
    )
    conn.execute(
        "INSERT INTO chunks(id, meeting_id, speaker_id, text) VALUES (?, ?, ?, ?)",
        (db_mod.new_id(), mid, sid, "hello"),
    )

    conn.execute("DELETE FROM meetings WHERE id = ?", (mid,))

    assert conn.execute("SELECT COUNT(*) FROM speakers WHERE meeting_id = ?", (mid,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE meeting_id = ?", (mid,)).fetchone()[0] == 0


def test_reconciliation_on_init(tmp_path):
    db_path = tmp_path / "r.db"
    c1 = db_mod.init(db_path)
    pid, mid, jid = db_mod.new_id(), db_mod.new_id(), db_mod.new_id()
    c1.execute(
        "INSERT INTO projects(id, name, created_at) VALUES (?, ?, ?)",
        (pid, "P", db_mod.now_iso()),
    )
    c1.execute(
        "INSERT INTO meetings(id, project_id, title, audio_path, status, created_at) "
        "VALUES (?, ?, ?, ?, 'transcribing', ?)",
        (mid, pid, "M", "/tmp/a.wav", db_mod.now_iso()),
    )
    c1.execute(
        "INSERT INTO jobs(id, meeting_id, kind, status, created_at) "
        "VALUES (?, ?, 'transcribe', 'running', ?)",
        (jid, mid, db_mod.now_iso()),
    )
    c1.close()

    c2 = db_mod.init(db_path)
    job_status = c2.execute("SELECT status, error FROM jobs WHERE id = ?", (jid,)).fetchone()
    meeting_status = c2.execute("SELECT status FROM meetings WHERE id = ?", (mid,)).fetchone()
    assert job_status["status"] == "error"
    assert job_status["error"] == "interrupted"
    assert meeting_status["status"] == "error"
    c2.close()
