"""Tests for summary storage + FTS search."""
import pytest

from meetingtool import db as db_mod
from meetingtool.tools.summaries import (
    delete_summary,
    get_summary,
    list_summaries,
    save_summary,
    search_summaries,
)


def _mkproject(conn, name="P") -> str:
    pid = db_mod.new_id()
    conn.execute(
        "INSERT INTO projects(id, name, created_at) VALUES (?,?,?)",
        (pid, name, db_mod.now_iso()),
    )
    return pid


def _mkmeeting(conn, pid, title="M") -> str:
    mid = db_mod.new_id()
    conn.execute(
        "INSERT INTO meetings(id, project_id, title, audio_path, status, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (mid, pid, title, "/tmp/x.wav", "ready", db_mod.now_iso()),
    )
    return mid


def _mkseries(conn, pid, name="S") -> str:
    sid = db_mod.new_id()
    conn.execute(
        "INSERT INTO meeting_series(id, project_id, name, created_at) "
        "VALUES (?,?,?,?)",
        (sid, pid, name, db_mod.now_iso()),
    )
    return sid


def test_save_summary_requires_kind_and_text(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    with pytest.raises(ValueError, match="kind"):
        save_summary(mid, "", "hello")
    with pytest.raises(ValueError, match="text"):
        save_summary(mid, "overview", "  ")


def test_save_summary_unknown_meeting(conn):
    with pytest.raises(ValueError, match="unknown meeting_id"):
        save_summary("nope", "overview", "hi")


def test_save_and_get_summary(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    save_summary(mid, "overview", "Q4 budget review was the focus.")
    out = get_summary(mid, "overview")
    assert out["text"] == "Q4 budget review was the focus."
    assert out["kind"] == "overview"


def test_get_summary_all_kinds(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    save_summary(mid, "overview", "Big picture.")
    save_summary(mid, "action_items", "- Send agenda by Friday")
    out = get_summary(mid)
    assert set(out["summaries"].keys()) == {"overview", "action_items"}
    assert out["summaries"]["action_items"]["text"].startswith("- Send")


def test_save_summary_upsert_preserves_created_at(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    first = save_summary(mid, "overview", "v1")
    second = save_summary(mid, "overview", "v2 with more detail")
    assert second["replaced_existing"] is True
    assert second["id"] == first["id"]
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] >= first["updated_at"]
    assert get_summary(mid, "overview")["text"] == "v2 with more detail"


def test_get_summary_missing_kind_raises(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    with pytest.raises(ValueError, match="no summary"):
        get_summary(mid, "overview")


def test_get_summary_no_kind_returns_empty_dict(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    out = get_summary(mid)
    assert out["summaries"] == {}


def test_list_summaries_scope_filters(conn):
    pid1 = _mkproject(conn, "one")
    pid2 = _mkproject(conn, "two")
    m1 = _mkmeeting(conn, pid1)
    m2 = _mkmeeting(conn, pid1)
    m3 = _mkmeeting(conn, pid2)
    save_summary(m1, "overview", "a")
    save_summary(m2, "overview", "b")
    save_summary(m2, "decisions", "c")
    save_summary(m3, "overview", "d")

    assert len(list_summaries()) == 4
    assert len(list_summaries(project_id=pid1)) == 3
    assert len(list_summaries(meeting_id=m2)) == 2
    assert len(list_summaries(project_id=pid1, kind="overview")) == 2
    # metadata only — no text
    for row in list_summaries():
        assert "text" not in row
        assert "char_count" in row


def test_delete_summary(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    save_summary(mid, "overview", "content")
    out = delete_summary(mid, "overview")
    assert out["deleted"] is True
    assert delete_summary(mid, "overview")["deleted"] is False


def test_search_summaries_basic(conn):
    pid = _mkproject(conn)
    m1 = _mkmeeting(conn, pid, "Jan")
    m2 = _mkmeeting(conn, pid, "Feb")
    m3 = _mkmeeting(conn, pid, "Mar")
    save_summary(m1, "overview", "Discussed Q1 budget overrun and hiring freeze.")
    save_summary(m2, "overview", "Green-lit the platform migration roadmap.")
    save_summary(m3, "decisions", "Deferred budget review to Q2.")

    out = search_summaries("budget")
    assert out["count"] == 2
    titles = {h["meeting_title"] for h in out["hits"]}
    assert titles == {"Jan", "Mar"}


def test_search_summaries_kind_filter(conn):
    pid = _mkproject(conn)
    m1 = _mkmeeting(conn, pid)
    m2 = _mkmeeting(conn, pid)
    save_summary(m1, "overview", "budget")
    save_summary(m2, "decisions", "budget")
    out = search_summaries("budget", kind="decisions")
    assert out["count"] == 1
    assert out["hits"][0]["kind"] == "decisions"


def test_search_summaries_empty_query_rejected(conn):
    with pytest.raises(ValueError):
        search_summaries("   ")


def test_summary_updates_reflect_in_fts(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    save_summary(mid, "overview", "old content about carrots")
    save_summary(mid, "overview", "updated content about budget")
    assert search_summaries("carrots")["count"] == 0
    assert search_summaries("budget")["count"] == 1


def test_meeting_delete_cascades_summaries(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    save_summary(mid, "overview", "budget discussion")
    conn.execute("DELETE FROM meetings WHERE id=?", (mid,))
    assert search_summaries("budget")["count"] == 0


# --- series-scoped summaries -------------------------------------------------


def test_save_summary_requires_exactly_one_scope(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    sid = _mkseries(conn, pid)
    with pytest.raises(ValueError, match="exactly one"):
        save_summary(kind="overview", text="x")
    with pytest.raises(ValueError, match="exactly one"):
        save_summary(meeting_id=mid, kind="overview", text="x", series_id=sid)


def test_save_summary_unknown_series(conn):
    with pytest.raises(ValueError, match="unknown series_id"):
        save_summary(kind="overview", text="x", series_id="nope")


def test_save_and_get_series_summary(conn):
    pid = _mkproject(conn)
    sid = _mkseries(conn, pid, "Weekly 1:1")
    saved = save_summary(series_id=sid, kind="rollup", text="Across 8 meetings: hiring focus.")
    assert saved["series_id"] == sid
    assert saved["meeting_id"] is None

    out = get_summary(series_id=sid, kind="rollup")
    assert out["text"].startswith("Across 8")
    assert out["kind"] == "rollup"


def test_series_summary_upsert_is_independent_of_meeting_scope(conn):
    """Same kind can exist for a meeting and a series without collision."""
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    sid = _mkseries(conn, pid)
    save_summary(meeting_id=mid, kind="overview", text="meeting-level")
    save_summary(series_id=sid, kind="overview", text="series-level")
    assert get_summary(meeting_id=mid, kind="overview")["text"] == "meeting-level"
    assert get_summary(series_id=sid, kind="overview")["text"] == "series-level"


def test_list_summaries_includes_both_scopes(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    sid = _mkseries(conn, pid, "Roadmap")
    save_summary(meeting_id=mid, kind="overview", text="a")
    save_summary(series_id=sid, kind="rollup", text="b")

    rows = list_summaries()
    assert len(rows) == 2
    by_kind = {r["kind"]: r for r in rows}
    assert by_kind["rollup"]["series_id"] == sid
    assert by_kind["rollup"]["series_name"] == "Roadmap"
    assert by_kind["rollup"]["meeting_id"] is None
    assert by_kind["overview"]["meeting_id"] == mid
    assert by_kind["overview"]["series_id"] is None


def test_list_summaries_project_filter_covers_series(conn):
    pid1 = _mkproject(conn, "one")
    pid2 = _mkproject(conn, "two")
    m1 = _mkmeeting(conn, pid1)
    s1 = _mkseries(conn, pid1)
    s2 = _mkseries(conn, pid2)
    save_summary(meeting_id=m1, kind="overview", text="a")
    save_summary(series_id=s1, kind="rollup", text="b")
    save_summary(series_id=s2, kind="rollup", text="c")

    assert len(list_summaries(project_id=pid1)) == 2
    assert len(list_summaries(project_id=pid2)) == 1


def test_list_summaries_series_filter(conn):
    pid = _mkproject(conn)
    sid = _mkseries(conn, pid)
    mid = _mkmeeting(conn, pid)
    save_summary(series_id=sid, kind="rollup", text="x")
    save_summary(meeting_id=mid, kind="overview", text="y")
    rows = list_summaries(series_id=sid)
    assert len(rows) == 1
    assert rows[0]["series_id"] == sid


def test_delete_series_summary(conn):
    pid = _mkproject(conn)
    sid = _mkseries(conn, pid)
    save_summary(series_id=sid, kind="rollup", text="x")
    out = delete_summary(series_id=sid, kind="rollup")
    assert out["deleted"] is True
    assert out["series_id"] == sid
    assert out["meeting_id"] is None


def test_search_summaries_hits_series_scope(conn):
    pid = _mkproject(conn)
    sid = _mkseries(conn, pid, "Weekly 1:1")
    save_summary(series_id=sid, kind="rollup", text="Quarterly budget freeze recurring theme.")
    out = search_summaries("budget")
    assert out["count"] == 1
    hit = out["hits"][0]
    assert hit["series_id"] == sid
    assert hit["series_name"] == "Weekly 1:1"
    assert hit["meeting_id"] is None


def test_search_summaries_series_filter(conn):
    pid = _mkproject(conn)
    s1 = _mkseries(conn, pid, "A")
    s2 = _mkseries(conn, pid, "B")
    save_summary(series_id=s1, kind="rollup", text="budget")
    save_summary(series_id=s2, kind="rollup", text="budget")
    out = search_summaries("budget", series_id=s1)
    assert out["count"] == 1
    assert out["hits"][0]["series_id"] == s1


def test_series_delete_cascades_summaries(conn):
    pid = _mkproject(conn)
    sid = _mkseries(conn, pid)
    save_summary(series_id=sid, kind="rollup", text="budget talk")
    conn.execute("DELETE FROM meeting_series WHERE id=?", (sid,))
    assert search_summaries("budget")["count"] == 0


# --- transcript staleness ---------------------------------------------------


def test_fresh_summary_is_not_stale(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    saved = save_summary(meeting_id=mid, kind="overview", text="x")
    assert saved["transcript_stale"] is False
    assert get_summary(meeting_id=mid, kind="overview")["transcript_stale"] is False
    assert list_summaries(meeting_id=mid)[0]["transcript_stale"] is False


def test_retranscribe_flags_meeting_summary_stale(conn, tmp_path, monkeypatch):
    """Retranscribing a meeting should mark its saved summaries stale."""
    from meetingtool import jobs as jobs_mod
    from meetingtool.backends.stub import StubBackend
    from meetingtool.tools.meetings import retranscribe_meeting

    # retranscribe_meeting submits a job; install a stub runner so it runs.
    stub = StubBackend(delay=0.0)

    def fn(audio_path, *, progress, window=None):
        return stub.transcribe(audio_path, progress=progress, window=window)

    runner = jobs_mod.JobRunner(
        tmp_path / "test.db", fn,
        plan_windows_fn=lambda _p: [(0.0, 10.0)],
    )
    jobs_mod.reset_runner_for_tests(runner)
    try:
        pid = _mkproject(conn)
        mid = _mkmeeting(conn, pid)
        save_summary(meeting_id=mid, kind="overview", text="pre-retranscribe")

        retranscribe_meeting(mid)
        out = get_summary(meeting_id=mid, kind="overview")
        assert out["transcript_stale"] is True
    finally:
        runner.shutdown()
        jobs_mod.reset_runner_for_tests(None)


def test_retranscribe_flags_series_summary_stale(conn, tmp_path):
    """Retranscribing any member meeting should mark series summaries stale."""
    from meetingtool import jobs as jobs_mod
    from meetingtool.backends.stub import StubBackend
    from meetingtool.tools.meetings import retranscribe_meeting

    stub = StubBackend(delay=0.0)

    def fn(audio_path, *, progress, window=None):
        return stub.transcribe(audio_path, progress=progress, window=window)

    runner = jobs_mod.JobRunner(
        tmp_path / "test.db", fn,
        plan_windows_fn=lambda _p: [(0.0, 10.0)],
    )
    jobs_mod.reset_runner_for_tests(runner)
    try:
        pid = _mkproject(conn)
        mid = _mkmeeting(conn, pid)
        sid = _mkseries(conn, pid)
        conn.execute(
            "INSERT INTO meeting_series_members(series_id, meeting_id, added_at) "
            "VALUES (?,?,?)",
            (sid, mid, db_mod.now_iso()),
        )
        save_summary(series_id=sid, kind="rollup", text="series-level")

        retranscribe_meeting(mid)
        out = get_summary(series_id=sid, kind="rollup")
        assert out["transcript_stale"] is True
    finally:
        runner.shutdown()
        jobs_mod.reset_runner_for_tests(None)


def test_save_summary_clears_stale_flag(conn):
    pid = _mkproject(conn)
    mid = _mkmeeting(conn, pid)
    save_summary(meeting_id=mid, kind="overview", text="v1")
    # Simulate a retranscribe marking it stale
    conn.execute(
        "UPDATE summaries SET transcript_stale=1 WHERE meeting_id=?", (mid,)
    )
    assert get_summary(meeting_id=mid, kind="overview")["transcript_stale"] is True

    save_summary(meeting_id=mid, kind="overview", text="v2 — rewritten")
    assert get_summary(meeting_id=mid, kind="overview")["transcript_stale"] is False
