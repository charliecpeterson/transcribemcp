"""Tests for speaker tooling. No ML deps — uses pre-seeded chunks."""
import pytest

from meetingtool import db as db_mod
from meetingtool.tools.speakers import (
    assign_speaker,
    get_speaker_cameos,
    list_speakers,
)


def _seed_meeting_with_speakers(conn):
    """Insert a meeting with 2 speakers, each with a few utterances."""
    pid, mid = db_mod.new_id(), db_mod.new_id()
    s0, s1 = db_mod.new_id(), db_mod.new_id()
    ts = db_mod.now_iso()
    conn.execute(
        "INSERT INTO projects(id, name, created_at) VALUES (?,?,?)",
        (pid, "P", ts),
    )
    conn.execute(
        "INSERT INTO meetings(id, project_id, title, audio_path, status, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (mid, pid, "Demo", "/tmp/a.wav", "ready", ts),
    )
    conn.execute(
        "INSERT INTO speakers(id, meeting_id, label) VALUES (?,?,?)",
        (s0, mid, "SPEAKER_00"),
    )
    conn.execute(
        "INSERT INTO speakers(id, meeting_id, label) VALUES (?,?,?)",
        (s1, mid, "SPEAKER_01"),
    )
    chunks = [
        (0.0, 3.0, s0, "Hi everyone, I'm Sarah and I'll be leading this."),
        (3.0, 5.0, s1, "Thanks Sarah. I'm Mike from Engineering."),
        (5.0, 9.0, s0, "Today we'll cover the Q2 hiring plan."),
        (9.0, 12.0, s1, "Sounds good. I have a question about the budget."),
        (12.0, 16.0, s0, "Great — let's start there."),
    ]
    for start, end, sid, text in chunks:
        conn.execute(
            "INSERT INTO chunks(id, meeting_id, speaker_id, text, start_time, end_time) "
            "VALUES (?,?,?,?,?,?)",
            (db_mod.new_id(), mid, sid, text, start, end),
        )
    return mid


def test_list_speakers_shows_counts_and_duration(conn):
    mid = _seed_meeting_with_speakers(conn)
    result = list_speakers(mid)
    assert [s["label"] for s in result] == ["SPEAKER_00", "SPEAKER_01"]
    assert all(s["name"] is None for s in result)
    assert result[0]["segment_count"] == 3
    assert result[1]["segment_count"] == 2
    assert result[0]["total_seconds"] > 0


def test_assign_speaker_updates_name_and_notes(conn):
    mid = _seed_meeting_with_speakers(conn)
    out = assign_speaker(mid, "SPEAKER_00", "Sarah - HR", notes="self-ID at 0:00")
    assert out["label"] == "SPEAKER_00"
    assert out["name"] == "Sarah - HR"

    listed = {s["label"]: s for s in list_speakers(mid)}
    assert listed["SPEAKER_00"]["name"] == "Sarah - HR"
    assert listed["SPEAKER_00"]["notes"] == "self-ID at 0:00"
    assert listed["SPEAKER_01"]["name"] is None


def test_assign_speaker_rejects_unknown_label(conn):
    mid = _seed_meeting_with_speakers(conn)
    with pytest.raises(ValueError, match="SPEAKER_99"):
        assign_speaker(mid, "SPEAKER_99", "Nobody")


def test_cameos_default_only_unnamed(conn):
    mid = _seed_meeting_with_speakers(conn)
    assign_speaker(mid, "SPEAKER_00", "Sarah")
    out = get_speaker_cameos(mid)
    labels = [sp["label"] for sp in out["speakers"]]
    assert labels == ["SPEAKER_01"]  # Sarah is named, hidden by default
    assert out["speakers"][0]["utterances"][0]["text"].startswith("Thanks Sarah")


def test_cameos_include_named_if_asked(conn):
    mid = _seed_meeting_with_speakers(conn)
    assign_speaker(mid, "SPEAKER_00", "Sarah")
    out = get_speaker_cameos(mid, only_unnamed=False)
    labels = [sp["label"] for sp in out["speakers"]]
    assert labels == ["SPEAKER_00", "SPEAKER_01"]


def test_cameos_respects_n_per_speaker(conn):
    mid = _seed_meeting_with_speakers(conn)
    out = get_speaker_cameos(mid, n_per_speaker=2)
    for sp in out["speakers"]:
        assert len(sp["utterances"]) <= 2


def test_cameos_include_attached_docs(conn, tmp_path):
    from meetingtool.tools.documents import add_document

    mid = _seed_meeting_with_speakers(conn)
    pid = conn.execute(
        "SELECT project_id FROM meetings WHERE id=?", (mid,)
    ).fetchone()["project_id"]
    f = tmp_path / "attendees.md"
    f.write_text(
        "# Attendees\n\n- Sarah Chen (HR, lead)\n- Mike Novak (Engineering)\n"
    )
    add_document(pid, "Attendee list", str(f), meeting_id=mid)

    # Default off — no attached_documents key
    out = get_speaker_cameos(mid)
    assert "attached_documents" not in out

    # On — returns the doc text so Claude Code can match speakers to names
    out = get_speaker_cameos(mid, include_attached_docs=True)
    assert "attached_documents" in out
    assert len(out["attached_documents"]) == 1
    doc = out["attached_documents"][0]
    assert doc["title"] == "Attendee list"
    assert doc["kind"] == "md"
    assert "Sarah Chen" in doc["text"]
    assert "Mike Novak" in doc["text"]
    assert doc["truncated"] is False


def test_cameos_attached_docs_truncated(conn, tmp_path):
    from meetingtool.tools.documents import add_document

    mid = _seed_meeting_with_speakers(conn)
    pid = conn.execute(
        "SELECT project_id FROM meetings WHERE id=?", (mid,)
    ).fetchone()["project_id"]
    f = tmp_path / "notes.md"
    f.write_text("x" * 5000)
    add_document(pid, "Long notes", str(f), meeting_id=mid)

    out = get_speaker_cameos(mid, include_attached_docs=True, max_chars_per_doc=500)
    doc = out["attached_documents"][0]
    assert doc["truncated"] is True
    assert len(doc["text"]) == 500
    assert doc["total_chars"] >= 5000


def test_cameos_no_docs_returns_empty_list(conn):
    mid = _seed_meeting_with_speakers(conn)
    out = get_speaker_cameos(mid, include_attached_docs=True)
    assert out["attached_documents"] == []


def test_cameos_truncates_long_utterances(conn):
    mid = _seed_meeting_with_speakers(conn)
    long = "x" * 1000
    sid = conn.execute(
        "SELECT id FROM speakers WHERE meeting_id=? AND label='SPEAKER_00'", (mid,)
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO chunks(id, meeting_id, speaker_id, text, start_time, end_time) "
        "VALUES (?,?,?,?,?,?)",
        (db_mod.new_id(), mid, sid, long, 100.0, 110.0),
    )
    out = get_speaker_cameos(mid, n_per_speaker=10, max_chars_per_cameo=50)
    for sp in out["speakers"]:
        for u in sp["utterances"]:
            assert len(u["text"]) <= 50
            if len(u["text"]) == 50:
                assert u["text"].endswith("...")
