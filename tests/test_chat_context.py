"""Tests for get_chat_context — the one-call bundle for chat sessions."""
import pytest

from meetingtool import db as db_mod
from meetingtool.tools.chat import get_chat_context
from meetingtool.tools.documents import add_document
from meetingtool.tools.series import add_meeting_to_series, create_series
from meetingtool.tools.summaries import save_summary


def _mkproject(conn, name="P") -> str:
    pid = db_mod.new_id()
    conn.execute(
        "INSERT INTO projects(id, name, created_at) VALUES (?,?,?)",
        (pid, name, db_mod.now_iso()),
    )
    return pid


def _mkmeeting_with_transcript(conn, pid, title="M", date=None) -> str:
    mid = db_mod.new_id()
    conn.execute(
        "INSERT INTO meetings(id, project_id, title, date, audio_path, status, "
        "created_at, duration_seconds) VALUES (?,?,?,?,?,?,?,?)",
        (mid, pid, title, date, "/tmp/x.wav", "ready", db_mod.now_iso(), 60),
    )
    sid = db_mod.new_id()
    conn.execute(
        "INSERT INTO speakers(id, meeting_id, label, name) VALUES (?,?,?,?)",
        (sid, mid, "SPEAKER_00", None),
    )
    conn.execute(
        "INSERT INTO chunks(id, meeting_id, speaker_id, text, start_time, end_time) "
        "VALUES (?,?,?,?,?,?)",
        (db_mod.new_id(), mid, sid, f"transcript body for {title}", 0.0, 5.0),
    )
    return mid


# --- argument validation ----------------------------------------------------


def test_requires_exactly_one_scope(conn):
    with pytest.raises(ValueError, match="exactly one"):
        get_chat_context()
    pid = _mkproject(conn)
    mid = _mkmeeting_with_transcript(conn, pid)
    s = create_series(pid, "S")
    with pytest.raises(ValueError, match="exactly one"):
        get_chat_context(meeting_id=mid, series_id=s["id"])


def test_unknown_meeting_raises(conn):
    with pytest.raises(ValueError, match="unknown meeting_id"):
        get_chat_context(meeting_id="nope")


def test_unknown_series_raises(conn):
    with pytest.raises(ValueError, match="unknown series_id"):
        get_chat_context(series_id="nope")


# --- meeting scope ----------------------------------------------------------


def test_meeting_context_default_includes_transcript(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting_with_transcript(conn, pid, "Meeting A")
    save_summary(meeting_id=mid, kind="overview", text="A brief overview.")

    out = get_chat_context(meeting_id=mid)
    assert out["scope"] == "meeting"
    assert out["meeting"]["id"] == mid
    assert out["meeting"]["project_name"] == "P"
    assert "overview" in out["summaries"]
    assert out["summaries"]["overview"]["transcript_stale"] is False
    # Transcript included by default for meeting scope.
    assert out["transcript"]["status"] == "ready"
    assert "transcript body for Meeting A" in out["transcript"]["transcript"]
    # Documents: metadata-less list (no docs attached).
    assert out["documents"] == []


def test_meeting_context_can_skip_transcript(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting_with_transcript(conn, pid)
    out = get_chat_context(meeting_id=mid, include_transcripts=False)
    assert "transcript" not in out


def test_meeting_context_includes_series_summaries(conn):
    """A meeting in a series should surface series-scope summaries."""
    pid = _mkproject(conn)
    mid = _mkmeeting_with_transcript(conn, pid)
    s = create_series(pid, "Weekly 1:1")
    add_meeting_to_series(s["id"], mid)
    save_summary(series_id=s["id"], kind="rollup", text="Across 5 weeks: hiring focus.")

    out = get_chat_context(meeting_id=mid)
    assert s["id"] in out["series_summaries"]
    bundle = out["series_summaries"][s["id"]]
    assert bundle["series_name"] == "Weekly 1:1"
    assert "hiring focus" in bundle["summaries"]["rollup"]["text"]


def test_meeting_context_documents_metadata_only_by_default(conn, tmp_path):
    pid = _mkproject(conn)
    mid = _mkmeeting_with_transcript(conn, pid)
    doc_path = tmp_path / "agenda.md"
    doc_path.write_text("# Agenda\n\nDiscuss Q1 plans.")
    add_document(pid, "Agenda", str(doc_path), meeting_id=mid)

    out = get_chat_context(meeting_id=mid)
    assert len(out["documents"]) == 1
    assert out["documents"][0]["title"] == "Agenda"
    assert "text" not in out["documents"][0]

    out_full = get_chat_context(meeting_id=mid, include_documents=True)
    assert "text" in out_full["documents"][0]
    assert "Discuss Q1 plans" in out_full["documents"][0]["text"]


def test_meeting_transcript_max_chars_propagates(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting_with_transcript(conn, pid, "Long")
    out = get_chat_context(meeting_id=mid, transcript_max_chars=5)
    assert out["transcript"].get("truncated") is True


# --- series scope -----------------------------------------------------------


def test_series_context_metadata_and_per_meeting_summaries(conn):
    pid = _mkproject(conn)
    m1 = _mkmeeting_with_transcript(conn, pid, "Week 1", date="2026-01-07")
    m2 = _mkmeeting_with_transcript(conn, pid, "Week 2", date="2026-01-14")
    s = create_series(pid, "Weekly")
    add_meeting_to_series(s["id"], m1)
    add_meeting_to_series(s["id"], m2)
    save_summary(meeting_id=m1, kind="overview", text="week 1 notes")
    save_summary(meeting_id=m2, kind="overview", text="week 2 notes")
    save_summary(series_id=s["id"], kind="rollup", text="series rollup")

    out = get_chat_context(series_id=s["id"])
    assert out["scope"] == "series"
    assert out["series"]["name"] == "Weekly"
    assert out["summaries"]["rollup"]["text"] == "series rollup"
    assert len(out["meetings"]) == 2
    # Chronological order.
    assert [m["title"] for m in out["meetings"]] == ["Week 1", "Week 2"]
    # Per-meeting summaries are included by default (high value / low cost).
    assert out["meetings"][0]["summaries"]["overview"]["text"] == "week 1 notes"
    # Transcripts default OFF for series.
    assert "transcript" not in out["meetings"][0]


def test_series_context_can_include_transcripts(conn):
    pid = _mkproject(conn)
    m1 = _mkmeeting_with_transcript(conn, pid, "W1", date="2026-01-07")
    s = create_series(pid, "Weekly")
    add_meeting_to_series(s["id"], m1)

    out = get_chat_context(series_id=s["id"], include_transcripts=True)
    assert out["meetings"][0]["transcript"]["status"] == "ready"
    assert "transcript body for W1" in out["meetings"][0]["transcript"]["transcript"]


def test_series_context_empty_series(conn):
    pid = _mkproject(conn)
    s = create_series(pid, "New")
    out = get_chat_context(series_id=s["id"])
    assert out["meetings"] == []
    assert out["persons"] == []
    assert out["documents"] == []
    assert out["summaries"] == {}
