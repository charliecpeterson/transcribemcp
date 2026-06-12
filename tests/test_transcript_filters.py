"""Tests for get_transcript filter params."""
import pytest

from meetingtool import db as db_mod
from meetingtool.tools.meetings import get_transcript
from meetingtool.tools.speakers import assign_speaker


def _seed(conn):
    pid, mid = db_mod.new_id(), db_mod.new_id()
    s0, s1 = db_mod.new_id(), db_mod.new_id()
    ts = db_mod.now_iso()
    conn.execute("INSERT INTO projects(id, name, created_at) VALUES (?,?,?)", (pid, "P", ts))
    conn.execute(
        "INSERT INTO meetings(id, project_id, title, audio_path, status, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (mid, pid, "Demo", "/tmp/a.wav", "ready", ts),
    )
    conn.execute("INSERT INTO speakers(id, meeting_id, label) VALUES (?,?,?)", (s0, mid, "SPEAKER_00"))
    conn.execute("INSERT INTO speakers(id, meeting_id, label) VALUES (?,?,?)", (s1, mid, "SPEAKER_01"))
    chunks = [
        (0.0, 2.0, s0, "intro alpha"),
        (2.0, 5.0, s1, "response beta"),
        (5.0, 10.0, s0, "middle gamma"),
        (10.0, 15.0, s1, "middle delta"),
        (15.0, 20.0, s0, "closing epsilon"),
    ]
    for start, end, sid, text in chunks:
        conn.execute(
            "INSERT INTO chunks(id, meeting_id, speaker_id, text, start_time, end_time) "
            "VALUES (?,?,?,?,?,?)",
            (db_mod.new_id(), mid, sid, text, start, end),
        )
    return mid


def test_no_filters_returns_everything(conn):
    mid = _seed(conn)
    out = get_transcript(mid, format="json")
    assert out["segment_count"] == 5


def test_filter_by_speaker_label(conn):
    mid = _seed(conn)
    out = get_transcript(mid, format="json", speaker_labels=["SPEAKER_00"])
    assert out["segment_count"] == 3
    assert all(s["speaker_label"] == "SPEAKER_00" for s in out["segments"])


def test_filter_by_speaker_name_after_assign(conn):
    mid = _seed(conn)
    assign_speaker(mid, "SPEAKER_00", "Sarah")
    out = get_transcript(mid, format="json", speaker_names=["Sarah"])
    assert out["segment_count"] == 3


def test_filter_by_time_range(conn):
    mid = _seed(conn)
    out = get_transcript(mid, format="json", time_range=[4.0, 12.0])
    # Should pick up segments that overlap [4,12]: (2-5), (5-10), (10-15)
    texts = [s["text"] for s in out["segments"]]
    assert texts == ["response beta", "middle gamma", "middle delta"]


def test_time_range_validation(conn):
    mid = _seed(conn)
    with pytest.raises(ValueError, match="time_range"):
        get_transcript(mid, time_range=[10.0, 5.0])


def test_max_chars_truncates_text_format(conn):
    mid = _seed(conn)
    out = get_transcript(mid, format="text", max_chars=30)
    assert out.get("truncated") is True
    assert out["total_chars"] > 30
    assert len(out["transcript"]) <= 30 + len("\n... [truncated]") + 1


def test_max_chars_truncates_json_format(conn):
    mid = _seed(conn)
    out = get_transcript(mid, format="json", max_chars=20)
    # Should stop before going over 20 chars cumulative
    total = sum(len(s["text"]) for s in out["segments"])
    assert total <= 20
    assert out.get("truncated") is True


def test_combined_filters(conn):
    mid = _seed(conn)
    out = get_transcript(
        mid,
        format="json",
        speaker_labels=["SPEAKER_00"],
        time_range=[4.0, 16.0],
    )
    texts = [s["text"] for s in out["segments"]]
    assert texts == ["middle gamma", "closing epsilon"]
