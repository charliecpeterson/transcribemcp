"""Exercise the CRUD tools directly (not through the MCP wire protocol)."""
import pytest

from meetingtool.tools.projects import create_project, list_projects
from meetingtool.tools.meetings import (
    add_meeting,
    delete_meeting,
    get_transcript,
    list_meetings,
)


def test_create_and_list_projects(conn):
    p1 = create_project("Hiring Q2")
    p2 = create_project("Jones Inv", "ongoing")
    listed = list_projects()
    names = {p["name"] for p in listed}
    assert {"Hiring Q2", "Jones Inv"} <= names
    ids = {p["id"] for p in listed}
    assert p1["id"] in ids and p2["id"] in ids


def test_add_meeting_requires_existing_file(conn, tmp_path):
    p = create_project("P")
    missing = tmp_path / "nope.wav"
    with pytest.raises(FileNotFoundError):
        add_meeting(p["id"], "t", str(missing))


def test_add_meeting_rejects_relative_path(conn, tmp_path):
    p = create_project("P")
    with pytest.raises(ValueError, match="absolute"):
        add_meeting(p["id"], "t", "relative/path.wav")


def test_add_meeting_rejects_unknown_project(conn, tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"fake")
    with pytest.raises(ValueError, match="unknown project_id"):
        add_meeting("no-such-id", "t", str(f))


def test_list_meetings_and_delete(conn, tmp_path):
    p = create_project("P")
    f1, f2 = tmp_path / "a.wav", tmp_path / "b.wav"
    f1.write_bytes(b"x"); f2.write_bytes(b"x")
    m1 = add_meeting(p["id"], "first", str(f1), auto_transcribe=False)
    m2 = add_meeting(p["id"], "second", str(f2), auto_transcribe=False)

    listed = list_meetings(p["id"])
    assert {m["id"] for m in listed} == {m1["id"], m2["id"]}

    deleted = delete_meeting(m1["id"])
    assert deleted["deleted"] == m1["id"]
    assert deleted["audio_path_left_on_disk"] == str(f1)
    assert f1.exists(), "audio file must not be deleted from disk"

    remaining = list_meetings(p["id"])
    assert {m["id"] for m in remaining} == {m2["id"]}


def test_get_transcript_before_ready(conn, tmp_path):
    p = create_project("P")
    f = tmp_path / "a.wav"; f.write_bytes(b"x")
    m = add_meeting(p["id"], "t", str(f), auto_transcribe=False)
    result = get_transcript(m["id"])
    assert result["status"] == "pending"
    assert result["transcript"] is None


def test_add_meeting_auto_transcribe_enqueues_job(conn, tmp_path, stub_runner_factory):
    """Default behavior: add_meeting enqueues a transcription job and returns job_id."""
    import time

    stub_runner_factory(db_path=tmp_path / "test.db")

    p = create_project("P")
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    m = add_meeting(p["id"], "t", str(f))

    assert m["status"] == "queued"
    assert "job_id" in m

    from meetingtool.tools.jobs import get_status

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        s = get_status(job_id=m["job_id"])
        if s["status"] in ("done", "error"):
            assert s["status"] == "done", s.get("error")
            break
        time.sleep(0.02)
    else:
        raise AssertionError("auto-transcribe job did not complete")


def test_add_meeting_auto_transcribe_false_skips_job(conn, tmp_path):
    p = create_project("P")
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    m = add_meeting(p["id"], "t", str(f), auto_transcribe=False)
    assert m["status"] == "pending"
    assert "job_id" not in m
