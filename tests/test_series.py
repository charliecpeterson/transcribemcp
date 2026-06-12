"""Tests for meeting_series tools."""
import pytest

from meetingtool import db as db_mod
from meetingtool.tools.search import search_transcripts
from meetingtool.tools.series import (
    add_meeting_to_series,
    create_series,
    delete_series,
    get_series,
    list_series,
    remove_meeting_from_series,
)


def _mkproject(conn, name="P") -> str:
    pid = db_mod.new_id()
    conn.execute(
        "INSERT INTO projects(id, name, created_at) VALUES (?,?,?)",
        (pid, name, db_mod.now_iso()),
    )
    return pid


def _mkmeeting(conn, pid, title, date=None) -> str:
    mid = db_mod.new_id()
    conn.execute(
        "INSERT INTO meetings(id, project_id, title, date, audio_path, status, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (mid, pid, title, date, "/tmp/x.wav", "ready", db_mod.now_iso()),
    )
    return mid


def test_create_series_requires_name(conn):
    pid = _mkproject(conn)
    with pytest.raises(ValueError):
        create_series(pid, "  ")


def test_create_series_unknown_project(conn):
    with pytest.raises(ValueError, match="unknown project_id"):
        create_series("nope", "X")


def test_list_series_scoped_by_project(conn):
    p1 = _mkproject(conn, "one")
    p2 = _mkproject(conn, "two")
    create_series(p1, "A")
    create_series(p1, "B")
    create_series(p2, "C")
    assert len(list_series()) == 3
    assert {r["name"] for r in list_series(project_id=p1)} == {"A", "B"}


def test_add_meeting_to_series(conn):
    pid = _mkproject(conn)
    m1 = _mkmeeting(conn, pid, "Jan 1:1", date="2026-01-07")
    m2 = _mkmeeting(conn, pid, "Jan 1:1 #2", date="2026-01-14")
    s = create_series(pid, "Weekly 1:1")
    add_meeting_to_series(s["id"], m1)
    add_meeting_to_series(s["id"], m2)

    series = get_series(s["id"])
    assert [m["id"] for m in series["meetings"]] == [m1, m2]
    # list_series reflects count
    assert list_series(project_id=pid)[0]["meeting_count"] == 2


def test_add_meeting_to_series_is_idempotent(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid, "M")
    s = create_series(pid, "S")
    add_meeting_to_series(s["id"], mid)
    add_meeting_to_series(s["id"], mid)  # second time: no-op
    assert list_series(project_id=pid)[0]["meeting_count"] == 1


def test_add_meeting_cross_project_rejected(conn):
    p1 = _mkproject(conn, "A")
    p2 = _mkproject(conn, "B")
    mid = _mkmeeting(conn, p1, "M")
    s = create_series(p2, "S")
    with pytest.raises(ValueError, match="belongs to project"):
        add_meeting_to_series(s["id"], mid)


def test_get_series_rich_overview(conn):
    """get_series should return summary_kinds, total_duration, document_count,
    persons, and per-meeting summary_kinds + chunk_count."""
    from meetingtool.tools.persons import create_person, link_speaker_to_person
    from meetingtool.tools.summaries import save_summary

    pid = _mkproject(conn)
    m1 = _mkmeeting(conn, pid, "M1", date="2026-01-01")
    m2 = _mkmeeting(conn, pid, "M2", date="2026-01-08")
    # Give them durations + a chunk each so the counts are non-zero.
    conn.execute("UPDATE meetings SET duration_seconds=1800 WHERE id=?", (m1,))
    conn.execute("UPDATE meetings SET duration_seconds=3600 WHERE id=?", (m2,))
    sp1 = db_mod.new_id()
    conn.execute(
        "INSERT INTO speakers(id, meeting_id, label) VALUES (?,?,?)",
        (sp1, m1, "SPEAKER_00"),
    )
    conn.execute(
        "INSERT INTO chunks(id, meeting_id, speaker_id, text, start_time, end_time) "
        "VALUES (?,?,?,?,?,?)",
        (db_mod.new_id(), m1, sp1, "hello", 0.0, 1.0),
    )

    s = create_series(pid, "Weekly")
    add_meeting_to_series(s["id"], m1)
    add_meeting_to_series(s["id"], m2)

    # Summaries at both scopes
    save_summary(meeting_id=m1, kind="overview", text="m1 overview")
    save_summary(meeting_id=m2, kind="overview", text="m2 overview")
    save_summary(series_id=s["id"], kind="rollup", text="series rollup")

    # Person linked to a member meeting
    person = create_person("Sarah")
    link_speaker_to_person(m1, "SPEAKER_00", person["id"])

    out = get_series(s["id"])
    assert out["meeting_count"] == 2
    assert out["total_duration_seconds"] == 5400
    assert out["summary_kinds"] == ["rollup"]
    assert out["document_count"] == 0
    assert len(out["persons"]) == 1
    assert out["persons"][0]["name"] == "Sarah"
    assert out["persons"][0]["meeting_count"] == 1

    # Per-meeting enrichments
    by_id = {m["id"]: m for m in out["meetings"]}
    assert by_id[m1]["summary_kinds"] == ["overview"]
    assert by_id[m1]["chunk_count"] == 1
    assert by_id[m2]["chunk_count"] == 0


def test_get_series_orders_chronologically(conn):
    pid = _mkproject(conn)
    # Insert out-of-order to prove the ORDER BY works
    later = _mkmeeting(conn, pid, "Feb", date="2026-02-15")
    earlier = _mkmeeting(conn, pid, "Jan", date="2026-01-10")
    middle = _mkmeeting(conn, pid, "Mid", date="2026-01-25")
    s = create_series(pid, "S")
    for mid in (later, earlier, middle):
        add_meeting_to_series(s["id"], mid)
    ordered = [m["id"] for m in get_series(s["id"])["meetings"]]
    assert ordered == [earlier, middle, later]


def test_remove_meeting_from_series(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid, "M")
    s = create_series(pid, "S")
    add_meeting_to_series(s["id"], mid)
    out = remove_meeting_from_series(s["id"], mid)
    assert out["removed"] is True
    assert get_series(s["id"])["meetings"] == []
    # second removal: no-op
    assert remove_meeting_from_series(s["id"], mid)["removed"] is False


def test_delete_series_leaves_meetings(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid, "M")
    s = create_series(pid, "S")
    add_meeting_to_series(s["id"], mid)
    delete_series(s["id"])
    assert list_series() == []
    # meeting still exists
    assert conn.execute("SELECT 1 FROM meetings WHERE id=?", (mid,)).fetchone() is not None


def test_meeting_delete_cascades_series_membership(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid, "M")
    s = create_series(pid, "S")
    add_meeting_to_series(s["id"], mid)
    conn.execute("DELETE FROM meetings WHERE id=?", (mid,))
    # membership row auto-cleaned by FK
    assert list_series(project_id=pid)[0]["meeting_count"] == 0


def test_search_transcripts_by_series(conn):
    pid = _mkproject(conn)
    m1 = _mkmeeting(conn, pid, "Jan", date="2026-01-01")
    m2 = _mkmeeting(conn, pid, "Feb", date="2026-02-01")
    m3 = _mkmeeting(conn, pid, "Unrelated", date="2026-03-01")
    for mid, text in [
        (m1, "we reviewed the hiring plan in January"),
        (m2, "we refined the hiring plan in February"),
        (m3, "hiring plan came up but different context"),
    ]:
        conn.execute(
            "INSERT INTO chunks(id, meeting_id, speaker_id, text, start_time, end_time) "
            "VALUES (?,?,?,?,?,?)",
            (db_mod.new_id(), mid, None, text, 0.0, 5.0),
        )
    s = create_series(pid, "Weekly 1:1")
    add_meeting_to_series(s["id"], m1)
    add_meeting_to_series(s["id"], m2)

    out = search_transcripts("hiring", series_id=s["id"])
    assert out["count"] == 2
    mids = {h["meeting_id"] for h in out["hits"]}
    assert mids == {m1, m2}
