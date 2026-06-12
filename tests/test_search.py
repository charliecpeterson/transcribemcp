"""Tests for search_transcripts FTS tool."""
import pytest

from meetingtool import db as db_mod
from meetingtool.tools.search import search_transcripts


def _seed_two_meetings(conn):
    pid = db_mod.new_id()
    ts = db_mod.now_iso()
    conn.execute("INSERT INTO projects(id, name, created_at) VALUES (?,?,?)", (pid, "P", ts))
    m1, m2 = db_mod.new_id(), db_mod.new_id()
    for mid, title in [(m1, "Jan"), (m2, "Feb")]:
        conn.execute(
            "INSERT INTO meetings(id, project_id, title, audio_path, status, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (mid, pid, title, "/tmp/x.wav", "ready", ts),
        )
    sid = db_mod.new_id()
    conn.execute("INSERT INTO speakers(id, meeting_id, label, name) VALUES (?,?,?,?)",
                 (sid, m1, "SPEAKER_00", "Sarah"))
    texts = [
        (m1, sid, "we discussed the Q1 budget overrun and projected timeline"),
        (m1, None, "no decision was reached on hiring plan"),
        (m2, None, "the Q2 budget is on track, timeline looks healthy"),
        (m2, None, "we need to revisit the hiring plan next week"),
    ]
    for mid, speaker_id, text in texts:
        conn.execute(
            "INSERT INTO chunks(id, meeting_id, speaker_id, text, start_time, end_time) "
            "VALUES (?,?,?,?,?,?)",
            (db_mod.new_id(), mid, speaker_id, text, 0.0, 5.0),
        )
    return pid, m1, m2


def test_basic_search_returns_snippets(conn):
    _seed_two_meetings(conn)
    out = search_transcripts("budget")
    assert out["count"] == 2
    for hit in out["hits"]:
        assert "<<budget>>" in hit["snippet"].lower() or "budget" in hit["snippet"].lower()


def test_scope_by_meeting_id(conn):
    _, m1, _ = _seed_two_meetings(conn)
    out = search_transcripts("budget", meeting_id=m1)
    assert out["count"] == 1
    assert out["hits"][0]["meeting_id"] == m1


def test_scope_by_project_id(conn):
    pid, _, _ = _seed_two_meetings(conn)
    out = search_transcripts("hiring", project_id=pid)
    assert out["count"] == 2


def test_phrase_search(conn):
    _seed_two_meetings(conn)
    out = search_transcripts('"hiring plan"')
    assert out["count"] == 2


def test_speaker_attribution_in_hits(conn):
    _, m1, _ = _seed_two_meetings(conn)
    out = search_transcripts("overrun", meeting_id=m1)
    assert out["count"] == 1
    assert out["hits"][0]["speaker"] == "Sarah"


def test_empty_query_rejected(conn):
    with pytest.raises(ValueError):
        search_transcripts("   ")
