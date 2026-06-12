"""End-to-end workflow test — exercises the canonical Claude Code session.

This is the cross-module regression guard: per-module tests validate each
tool in isolation, this one verifies they compose correctly. It mirrors the
session flow a real user would drive:

    1. Create a project, register two recordings.
    2. Transcribe both (stub backend → deterministic fake output).
    3. Inspect speakers, assign local names, promote to canonical persons,
       link across both meetings.
    4. Group the two meetings into a series.
    5. Attach a supplemental document, save summaries, search across
       transcripts / documents / summaries.
    6. Pull a rich meeting overview that should reflect all of the above in
       one call.
    7. Update titles, then tear down via delete_project and verify cascade.
"""
import time
from pathlib import Path

import pytest

from meetingtool import jobs as jobs_mod
from meetingtool.backends.stub import StubBackend
from meetingtool.tools.documents import add_document, search_documents
from meetingtool.tools.jobs import get_status, transcribe_meeting
from meetingtool.tools.meetings import (
    add_meeting,
    get_meeting,
    get_transcript,
    update_meeting,
)
from meetingtool.tools.persons import (
    create_person,
    get_person,
    link_speaker_to_person,
    list_persons,
)
from meetingtool.tools.projects import (
    create_project,
    delete_project,
    list_projects,
    update_project,
)
from meetingtool.tools.search import search_transcripts
from meetingtool.tools.series import (
    add_meeting_to_series,
    create_series,
    get_series,
)
from meetingtool.tools.speakers import (
    assign_speaker,
    get_speaker_cameos,
    list_speakers,
)
from meetingtool.tools.summaries import (
    get_summary,
    save_summary,
    search_summaries,
)


def _install_stub_runner(db_path: Path):
    stub = StubBackend(delay=0.01)

    def fn(audio_path, *, progress, window=None):
        return stub.transcribe(audio_path, progress=progress, window=window)

    runner = jobs_mod.JobRunner(
        db_path, fn,
        plan_windows_fn=lambda _p: [(0.0, 10.0)],
    )
    jobs_mod.reset_runner_for_tests(runner)
    return runner


def _wait_done(job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = get_status(job_id=job_id)
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.02)
    pytest.fail(f"timed out waiting for job {job_id}")


def test_canonical_end_to_end_workflow(conn, tmp_path):
    runner = _install_stub_runner(tmp_path / "test.db")
    try:
        # --- 1. Project + meetings ---
        proj = create_project("Hiring 2026", "Weekly sync + interview loop")
        audio1 = tmp_path / "jan1.wav"
        audio1.write_bytes(b"fake audio bytes")
        audio2 = tmp_path / "jan8.wav"
        audio2.write_bytes(b"fake audio bytes")
        m1 = add_meeting(proj["id"], "Jan 1 sync", str(audio1), date="2026-01-01", auto_transcribe=False)
        m2 = add_meeting(proj["id"], "Jan 8 sync", str(audio2), date="2026-01-08", auto_transcribe=False)

        # --- 2. Transcribe both asynchronously ---
        for mid in (m1["id"], m2["id"]):
            resp = transcribe_meeting(mid)
            assert resp["status"] == "queued"
            final = _wait_done(resp["job_id"])
            assert final["status"] == "done", final.get("error")

        # Both transcripts are now usable
        t1 = get_transcript(m1["id"])
        assert t1["status"] == "ready"
        assert t1["segment_count"] >= 2

        # --- 3. Speaker identification ---
        speakers_m1 = list_speakers(m1["id"])
        assert len(speakers_m1) >= 2, "stub should produce multi-speaker output"

        cameos = get_speaker_cameos(m1["id"])
        # Every unnamed speaker should have utterances we could reason over
        assert all(len(s["utterances"]) >= 1 for s in cameos["speakers"])

        # Pretend Claude Code decided the first two labels are Sarah + Diego
        first_label = speakers_m1[0]["label"]
        second_label = speakers_m1[1]["label"]
        assign_speaker(m1["id"], first_label, "Sarah (local guess)")

        # --- 4. Promote to canonical persons, link across both meetings ---
        sarah = create_person("Sarah Chen", role="EM")
        diego = create_person("Diego Reyes", role="Staff Eng")
        assert len(list_persons()) == 2

        link_speaker_to_person(m1["id"], first_label, sarah["id"])
        link_speaker_to_person(m2["id"], first_label, sarah["id"])
        link_speaker_to_person(m1["id"], second_label, diego["id"])

        # Sarah should now appear in two meetings
        sarah_view = get_person(sarah["id"])
        assert len(sarah_view["meetings"]) == 2
        # Each entry carries the per-meeting label so you can pull her segments
        assert all("speaker_label" in m for m in sarah_view["meetings"])

        # Person-scoped search: find Sarah's words across all her meetings
        out = search_transcripts("fake", person_id=sarah["id"])
        mids_hit = {h["meeting_id"] for h in out["hits"]}
        assert mids_hit == {m1["id"], m2["id"]}

        # --- 5. Group into a series, search scoped to it ---
        series = create_series(proj["id"], "Weekly 1:1 Sarah", "Jan–Mar")
        add_meeting_to_series(series["id"], m1["id"])
        add_meeting_to_series(series["id"], m2["id"])
        s = get_series(series["id"])
        assert [m["id"] for m in s["meetings"]] == [m1["id"], m2["id"]]

        scoped = search_transcripts("fake", series_id=series["id"])
        assert scoped["count"] >= 2

        # --- 6. Document + summaries ---
        agenda = tmp_path / "agenda.md"
        agenda.write_text(
            "# Agenda — Jan 8\n\n"
            "- Review Q1 hiring pipeline\n\n"
            "- Discuss compensation bands for staff+ roles\n"
        )
        doc = add_document(
            proj["id"], "Jan 8 agenda", str(agenda), meeting_id=m2["id"]
        )
        assert doc["chunk_count"] >= 1

        hits = search_documents("compensation")
        assert hits["count"] >= 1
        assert hits["hits"][0]["document_title"] == "Jan 8 agenda"

        save_summary(m1["id"], "overview", "Kickoff discussed Q1 hiring targets.")
        save_summary(m1["id"], "action_items", "- Sarah: draft rubric by Friday")
        save_summary(m2["id"], "overview", "Follow-up on hiring rubric and comp.")

        sum_hits = search_summaries("hiring", project_id=proj["id"])
        assert sum_hits["count"] == 2
        assert get_summary(m1["id"], "action_items")["text"].startswith("- Sarah")

        # --- 7. Rich single-call overview ---
        overview = get_meeting(m1["id"])
        assert overview["project_name"] == "Hiring 2026"
        assert {s["label"] for s in overview["speakers"]} >= {first_label, second_label}
        named = next(s for s in overview["speakers"] if s["label"] == first_label)
        assert named["person_name"] == "Sarah Chen"
        assert any(sr["id"] == series["id"] for sr in overview["series"])
        assert set(overview["summary_kinds"]) == {"overview", "action_items"}
        # document was linked to m2, not m1
        assert overview["document_count"] == 0

        overview_m2 = get_meeting(m2["id"])
        assert overview_m2["document_count"] == 1

        # --- 8. Typo fixes via update tools ---
        update_project(proj["id"], name="Hiring FY26")
        update_meeting(m1["id"], title="Jan 1 sync (week 1)")
        assert list_projects()[0]["name"] == "Hiring FY26"
        assert get_meeting(m1["id"])["title"] == "Jan 1 sync (week 1)"

        # --- 9. Cascade teardown ---
        report = delete_project(proj["id"])
        assert report["removed_counts"]["meetings"] == 2
        assert report["removed_counts"]["documents"] == 1
        assert report["removed_counts"]["series"] == 1
        # Nothing left
        assert list_projects() == []
        assert conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM meeting_series").fetchone()[0] == 0
        # Persons survive project deletion (they're cross-project identities)
        # but their speaker links were cascaded away via meeting deletion.
        assert len(list_persons()) == 2
        assert get_person(sarah["id"])["meetings"] == []
    finally:
        runner.shutdown()
