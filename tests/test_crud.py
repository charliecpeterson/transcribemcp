"""Tests for update_project / delete_project / update_meeting / get_meeting."""
import pytest

from meetingtool import db as db_mod
from meetingtool.tools.documents import add_document
from meetingtool.tools.meetings import get_meeting, update_meeting
from meetingtool.tools.persons import create_person, link_speaker_to_person
from meetingtool.tools.projects import (
    create_project,
    delete_project,
    list_projects,
    update_project,
)
from meetingtool.tools.series import add_meeting_to_series, create_series
from meetingtool.tools.summaries import save_summary


def _mkmeeting(conn, pid, title="M") -> str:
    mid = db_mod.new_id()
    conn.execute(
        "INSERT INTO meetings(id, project_id, title, audio_path, status, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (mid, pid, title, "/tmp/x.wav", "ready", db_mod.now_iso()),
    )
    return mid


def test_update_project_name(conn):
    p = create_project("Old Name")
    update_project(p["id"], name="New Name")
    assert list_projects()[0]["name"] == "New Name"


def test_update_project_requires_one_field(conn):
    p = create_project("X")
    with pytest.raises(ValueError, match="name and/or description"):
        update_project(p["id"])


def test_update_project_empty_name_rejected(conn):
    p = create_project("X")
    with pytest.raises(ValueError, match="non-empty"):
        update_project(p["id"], name="   ")


def test_update_project_unknown(conn):
    with pytest.raises(ValueError, match="unknown project_id"):
        update_project("nope", name="x")


def test_update_project_clear_description(conn):
    p = create_project("X", "keep this around")
    update_project(p["id"], description="")
    row = list_projects()[0]
    assert row["description"] == ""


def test_delete_project_cascades(conn, tmp_path):
    p = create_project("Doomed")
    m1 = _mkmeeting(conn, p["id"])
    m2 = _mkmeeting(conn, p["id"])
    create_series(p["id"], "Weekly")
    f = tmp_path / "notes.txt"
    f.write_text("anything")
    add_document(p["id"], "N", str(f))

    out = delete_project(p["id"])
    assert out["removed_counts"] == {"meetings": 2, "documents": 1, "series": 1}
    assert list_projects() == []
    # cascade worked via FK
    assert conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM meeting_series").fetchone()[0] == 0
    # unrelated mention — m1/m2 unused after this point
    del m1, m2


def test_delete_project_unknown(conn):
    with pytest.raises(ValueError, match="unknown project_id"):
        delete_project("nope")


def test_update_meeting_title(conn):
    p = create_project("P")
    mid = _mkmeeting(conn, p["id"], title="Typo")
    update_meeting(mid, title="Correct Title")
    row = conn.execute("SELECT title FROM meetings WHERE id=?", (mid,)).fetchone()
    assert row["title"] == "Correct Title"


def test_update_meeting_date(conn):
    p = create_project("P")
    mid = _mkmeeting(conn, p["id"])
    update_meeting(mid, date="2026-04-19")
    row = conn.execute("SELECT date FROM meetings WHERE id=?", (mid,)).fetchone()
    assert row["date"] == "2026-04-19"


def test_update_meeting_requires_one_field(conn):
    p = create_project("P")
    mid = _mkmeeting(conn, p["id"])
    with pytest.raises(ValueError, match="title and/or date"):
        update_meeting(mid)


def test_update_meeting_unknown(conn):
    with pytest.raises(ValueError, match="unknown meeting_id"):
        update_meeting("nope", title="x")


def test_get_meeting_rich_overview(conn, tmp_path):
    p = create_project("Project X")
    mid = _mkmeeting(conn, p["id"], title="Kickoff")
    # Speakers + person link
    conn.execute(
        "INSERT INTO speakers(id, meeting_id, label) VALUES (?,?,?)",
        (db_mod.new_id(), mid, "SPEAKER_00"),
    )
    conn.execute(
        "INSERT INTO speakers(id, meeting_id, label) VALUES (?,?,?)",
        (db_mod.new_id(), mid, "SPEAKER_01"),
    )
    person = create_person("Sarah Chen")
    link_speaker_to_person(mid, "SPEAKER_00", person["id"])

    # Chunks
    for _ in range(4):
        conn.execute(
            "INSERT INTO chunks(id, meeting_id, speaker_id, text, start_time, end_time) "
            "VALUES (?,?,?,?,?,?)",
            (db_mod.new_id(), mid, None, "text", 0.0, 1.0),
        )

    # Series membership
    s = create_series(p["id"], "Weekly Sync")
    add_meeting_to_series(s["id"], mid)

    # Summaries
    save_summary(mid, "overview", "tl;dr")
    save_summary(mid, "action_items", "- do X")

    # Document attached
    f = tmp_path / "agenda.md"
    f.write_text("# Agenda")
    add_document(p["id"], "Agenda", str(f), meeting_id=mid)

    out = get_meeting(mid)
    assert out["title"] == "Kickoff"
    assert out["project_name"] == "Project X"
    assert out["chunk_count"] == 4
    assert {s["label"] for s in out["speakers"]} == {"SPEAKER_00", "SPEAKER_01"}
    sarah = next(s for s in out["speakers"] if s["label"] == "SPEAKER_00")
    assert sarah["person_name"] == "Sarah Chen"
    assert out["series"][0]["name"] == "Weekly Sync"
    assert set(out["summary_kinds"]) == {"overview", "action_items"}
    assert out["document_count"] == 1
    # No transcript text bleeds into overview
    assert "transcript" not in out and "text" not in out


def test_get_meeting_unknown(conn):
    with pytest.raises(ValueError, match="unknown meeting_id"):
        get_meeting("nope")
