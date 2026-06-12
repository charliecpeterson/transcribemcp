"""Tests for cross-meeting person identity."""
import pytest

from meetingtool import db as db_mod
from meetingtool.tools.persons import (
    create_person,
    delete_person,
    get_person,
    link_speaker_to_person,
    list_persons,
)
from meetingtool.tools.search import search_transcripts
from meetingtool.tools.speakers import get_speaker_cameos, list_speakers


def _seed_two_meetings_with_sarah(conn):
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
    s1 = db_mod.new_id()
    s2 = db_mod.new_id()
    s3 = db_mod.new_id()
    # Sarah is SPEAKER_00 in m1, SPEAKER_01 in m2. Diego is SPEAKER_00 in m2.
    conn.execute("INSERT INTO speakers(id, meeting_id, label) VALUES (?,?,?)", (s1, m1, "SPEAKER_00"))
    conn.execute("INSERT INTO speakers(id, meeting_id, label) VALUES (?,?,?)", (s2, m2, "SPEAKER_00"))
    conn.execute("INSERT INTO speakers(id, meeting_id, label) VALUES (?,?,?)", (s3, m2, "SPEAKER_01"))
    for mid, sid, text in [
        (m1, s1, "Sarah in January said we need to revisit the budget"),
        (m2, s2, "Diego opened February with platform migration updates"),
        (m2, s3, "Sarah returned to the budget topic in February"),
    ]:
        conn.execute(
            "INSERT INTO chunks(id, meeting_id, speaker_id, text, start_time, end_time) "
            "VALUES (?,?,?,?,?,?)",
            (db_mod.new_id(), mid, sid, text, 0.0, 5.0),
        )
    return pid, m1, m2


def test_create_person_requires_name(conn):
    with pytest.raises(ValueError):
        create_person("   ")


def test_create_and_list_persons(conn):
    p = create_person("Sarah Chen", email="sarah@example.com", role="EM")
    assert p["name"] == "Sarah Chen"
    all_persons = list_persons()
    assert len(all_persons) == 1
    assert all_persons[0]["meeting_count"] == 0


def test_link_speaker_to_person(conn):
    _, m1, _ = _seed_two_meetings_with_sarah(conn)
    p = create_person("Sarah Chen")
    out = link_speaker_to_person(m1, "SPEAKER_00", p["id"])
    assert out["person_name"] == "Sarah Chen"

    speakers = list_speakers(m1)
    sarah = next(s for s in speakers if s["label"] == "SPEAKER_00")
    assert sarah["person_id"] == p["id"]
    assert sarah["person_name"] == "Sarah Chen"
    # copy_name default True → local name is also filled
    assert sarah["name"] == "Sarah Chen"


def test_link_speaker_copy_name_false(conn):
    _, m1, _ = _seed_two_meetings_with_sarah(conn)
    p = create_person("Sarah Chen")
    link_speaker_to_person(m1, "SPEAKER_00", p["id"], copy_name=False)
    speakers = list_speakers(m1)
    sarah = next(s for s in speakers if s["label"] == "SPEAKER_00")
    assert sarah["person_id"] == p["id"]
    assert sarah["name"] is None


def test_get_person_lists_meetings(conn):
    _, m1, m2 = _seed_two_meetings_with_sarah(conn)
    p = create_person("Sarah Chen")
    link_speaker_to_person(m1, "SPEAKER_00", p["id"])
    link_speaker_to_person(m2, "SPEAKER_01", p["id"])

    details = get_person(p["id"])
    assert len(details["meetings"]) == 2
    labels = {(m["meeting_id"], m["speaker_label"]) for m in details["meetings"]}
    assert (m1, "SPEAKER_00") in labels
    assert (m2, "SPEAKER_01") in labels


def test_list_persons_meeting_count(conn):
    _, m1, m2 = _seed_two_meetings_with_sarah(conn)
    p = create_person("Sarah Chen")
    link_speaker_to_person(m1, "SPEAKER_00", p["id"])
    link_speaker_to_person(m2, "SPEAKER_01", p["id"])
    persons = list_persons()
    assert persons[0]["meeting_count"] == 2


def test_delete_person_unlinks_speakers(conn):
    _, m1, _ = _seed_two_meetings_with_sarah(conn)
    p = create_person("Sarah Chen")
    link_speaker_to_person(m1, "SPEAKER_00", p["id"])
    delete_person(p["id"])
    assert list_persons() == []
    speakers = list_speakers(m1)
    sarah = next(s for s in speakers if s["label"] == "SPEAKER_00")
    assert sarah["person_id"] is None
    # local name survives the unlink (trigger only touches person_id)
    assert sarah["name"] == "Sarah Chen"


def test_cameos_skip_person_linked_speakers(conn):
    _, m1, _ = _seed_two_meetings_with_sarah(conn)
    p = create_person("Sarah Chen")
    link_speaker_to_person(m1, "SPEAKER_00", p["id"])
    out = get_speaker_cameos(m1, only_unnamed=True)
    # SPEAKER_00 has both name and person link → should be skipped
    labels = [s["label"] for s in out["speakers"]]
    assert "SPEAKER_00" not in labels


def test_search_transcripts_by_person_id(conn):
    _, m1, m2 = _seed_two_meetings_with_sarah(conn)
    p = create_person("Sarah Chen")
    link_speaker_to_person(m1, "SPEAKER_00", p["id"])
    link_speaker_to_person(m2, "SPEAKER_01", p["id"])

    out = search_transcripts("budget", person_id=p["id"])
    # Both Sarah chunks mention budget; Diego's Feb chunk does not.
    assert out["count"] == 2
    for hit in out["hits"]:
        assert hit["speaker"] == "Sarah Chen"


def test_link_to_unknown_person_rejected(conn):
    _, m1, _ = _seed_two_meetings_with_sarah(conn)
    with pytest.raises(ValueError, match="unknown person_id"):
        link_speaker_to_person(m1, "SPEAKER_00", "does-not-exist")


def test_link_unknown_speaker_rejected(conn):
    _seed_two_meetings_with_sarah(conn)
    p = create_person("Sarah Chen")
    with pytest.raises(ValueError, match="unknown speaker"):
        link_speaker_to_person("no-such-meeting", "SPEAKER_99", p["id"])
