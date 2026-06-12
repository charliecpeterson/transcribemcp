# CLAUDE.md — meetingtool

## What this is

A Python MCP server that exposes meeting-transcription tools to Claude Code.
Claude Code is the agent — **do not rebuild the chat loop, UI, or agent
orchestration in this repo**. This project ships tools + SQLite storage,
nothing more.

Design doc: `meeting-assistant-plan.md` (in repo root)
Phase 1 plan: `/Users/charlie/.claude/plans/quiet-sniffing-hoare.md`

## Environment

- **Packaging: `uv`** (not conda). Matches the MCP ecosystem; this overrides
  the user's global conda-per-project default.
- Python 3.12 pinned via `.python-version`. `requires-python = ">=3.11,<3.14"`.
- Heavy ML deps are **optional extras** — the server runs with just
  `uv sync` (stub backend only). Real backends need:
  - `uv sync --extra whisperx --extra diarize` (WhisperX + pyannote)
  - `uv sync --extra qwen3 --extra diarize` (Qwen3-ASR + pyannote)
  - `--extra qwen3-mlx` adds MLX on Apple Silicon
- `.venv` contains an `imageio-ffmpeg`-bundled ffmpeg binary — scripts that
  need ffmpeg should resolve it via `uv run python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`.
  Do **not** ask the user to `brew install ffmpeg`.

## Key architectural choices

- **FastMCP** (`mcp[cli]`) over stdio — Claude Code's standard transport.
- **SQLite + WAL + FTS5**. Schema in `src/meetingtool/db.py`. Two FTS tables
  (`chunks_fts` for transcripts, `document_chunks_fts` for supplemental docs),
  both kept in sync by triggers. `search_transcripts` and `search_documents`
  return `snippet(...)`-highlighted hits ranked by `bm25()`.
- **Schema migrations**: `PRAGMA user_version` drives the migration ladder in
  `db.init()`. Current version is 8 (v8 adds `jobs.checkpoint_seconds` for
  windowed transcribe/resume). New tables go in a new `SCHEMA_V{n}` block +
  a `current < n` branch — never mutate prior blocks.
- **Async jobs: single `ThreadPoolExecutor(max_workers=1)`**. ASR is heavy;
  parallel jobs would thrash. Each worker opens its own SQLite connection
  (sqlite3 connections aren't thread-safe). Progress is written to the
  `jobs` table — callers read state via `get_status` rather than streaming.
- **Crash recovery**: `db.init()` reconciles any job left as `queued`/`running`
  to `error` on startup. Meetings stuck `transcribing` also go to `error`.
- **Lazy ML imports**: `transcribe.py` only imports the selected backend on
  first job. Never import torch/pyannote/transformers at module top level.
- **Backends are pure ASR**: `TranscriptionBackend.transcribe(audio_path, *,
  progress, window=None)` returns segments with absolute timestamps. Backends
  do **not** diarize — `jobs.py` owns the diarize-at-end step and applies
  speakers to already-persisted chunks. This keeps backends simple and makes
  resume work: a crashed job's pre-diarize chunks are still valid ASR output.
- **Windowed transcribe + resume**: `jobs.py` plans VAD-derived windows
  (`vad.group_into_windows`, target 5 min, break only on silence — no overlap)
  and drives the backend one window at a time. Each window persists its chunks
  and advances `jobs.checkpoint_seconds` in a single transaction. On crash/
  cancel, `resume_job(job_id)` re-plans the *same* windows (silero-vad is
  deterministic on identical audio) and skips any with `end <= checkpoint`.
  Backends that can slice natively (faster-whisper via `clip_timestamps`,
  Cohere via VAD-span filter, Qwen3 local via tensor slice, Qwen3 vLLM via
  ffmpeg `-ss`/`-to` temp file) accept a `window=(start, end)` kwarg; others
  fall back to full-file transcribe with a single `(0, inf)` window.
- **Shared audio + VAD preprocessors**: `audio.py` decodes any format to
  mono 16 kHz float32 PCM via the `imageio-ffmpeg` bundled binary — used
  by `diarize.py` (pyannote load), `vad.py` (silero-vad), and the Cohere
  backend. `vad.py` exists because LLM-decoded ASR models (Cohere, and
  future Granite / Phi-4 / Canary-Qwen) emit bare text with no segment
  boundaries; we synthesize `Segment(start, end)` from voiced spans. It also
  exposes `group_into_windows(spans, target_seconds=300.0)` for the
  transcribe pipeline.

## Tool surface (Phases 1 + 2)

```
# core CRUD
ping, create_project, list_projects, update_project, delete_project,
add_meeting, list_meetings, update_meeting, get_meeting, delete_meeting,
retranscribe_meeting, get_transcript,
transcribe_meeting, get_status, cancel_job, resume_job, list_jobs

# speakers (per-meeting diarization labels)
list_speakers, assign_speaker, get_speaker_cameos

# persons (canonical cross-meeting identity)
create_person, list_persons, get_person, delete_person, link_speaker_to_person

# meeting series (weekly 1:1s, interview loops, project stages)
create_series, list_series, get_series,
add_meeting_to_series, remove_meeting_from_series, delete_series

# one-call chat bundle — everything Claude needs to chat about a meeting
# or series. Metadata + summaries always; transcripts and docs are gated
# by flags so long series don't blow the context window.
get_chat_context

# cached summaries — scopable to a meeting OR a series (Claude Code writes
# them; these tools persist + search). save_summary/get_summary/delete_summary
# take exactly one of (meeting_id, series_id).
save_summary, get_summary, list_summaries, delete_summary, search_summaries

# search over transcripts (scope by meeting / series / project / person)
search_transcripts

# supplemental documents (.txt, .md, .pdf, .docx)
add_document, list_documents, get_document, delete_document, search_documents
```

**Speakers vs persons.** A `speaker` row is meeting-local (one row per
diarization label per meeting, with an optional `name` string). A `person`
is a canonical cross-meeting identity. A speaker links to a person via
`speakers.person_id`. `assign_speaker` sets the local name; if you want the
same human's utterances to be findable across every meeting they attended,
`link_speaker_to_person` is the tool.

**Speaker identification as a one-shot call.** `get_speaker_cameos` is the
speaker-ID primitive. With `include_attached_docs=True` it also returns the
text of any documents linked to the meeting — so the canonical flow is:
user drops an attendee list / agenda / notes via `add_document(..., meeting_id=mid)`,
then `get_speaker_cameos(mid, include_attached_docs=True)` hands Claude Code
cameos + doc text in one payload for reasoning. Don't make Claude Code
juggle `list_documents` + `get_document` manually for this — that's what
the flag is for. See README "Speaker identification workflow" for the full
playbook.

`transcribe_meeting` is **async** — it returns `{job_id, status: "queued"}`
immediately. Callers poll `get_status(job_id=...)`. Do not add blocking
variants; the whole design depends on this.

**`add_meeting` auto-transcribes by default.** Returns `{job_id, status:
"queued", ...}` exactly like `transcribe_meeting`. Pass
`auto_transcribe=False` for bulk import or when you plan to transcribe
later with different settings — the returned dict then has `status:
"pending"` and no `job_id`. This is the normal Claude Code flow: "add this
recording" should mean "add it and start transcribing," not "add it and
wait for me to ask again."

**`cancel_job` is cooperative.** The worker only notices cancellation at
stage boundaries (inside the `progress()` closure). A whisperx/pyannote
stage that runs straight through native code won't be interrupted until it
returns — this is a Python threading constraint, not a design choice.
Cancel flips `jobs.status='cancelled'` immediately; the worker resets
`meetings.status='pending'` on its way out. Terminal jobs (done/error/
already cancelled) are no-ops. For truly wedged jobs, the only real
escape today is restarting the MCP server process.

**`resume_job` picks up where a crash or cancel left off.** Only accepts
jobs in `error` or `cancelled` state — `done` jobs raise, `running`/`queued`
jobs raise with "cancel it first". Flips the job back to `queued` (preserving
`started_at` and `checkpoint_seconds`), marks the meeting `transcribing`, and
resubmits to the runner. The worker re-plans VAD windows, skips any with
`end <= checkpoint_seconds`, and continues from the next window. Diarization
runs fresh at the end over all persisted chunks (cheap compared to ASR).

**Progress heartbeat.** `jobs.progress_updated_at` is stamped on every
progress/stage write (insert, running, each `_set_progress`, done, error).
Callers use it to tell "slow" from "stuck" — whisperx ASR sits on a single
`progress` value for minutes at a time, so the timestamp is the only
reliable signal that the worker is still alive. `get_status` returns it.

**Summary staleness.** `summaries.transcript_stale` is flipped to 1 by
`retranscribe_meeting` on every summary whose scope covers the meeting
being retranscribed (the meeting itself, plus any series the meeting
belongs to). `save_summary` resets the flag to 0 on every write. Surface
it in `get_summary` / `list_summaries` so callers know when to regenerate.

**Series-scoped summaries.** `save_summary` takes exactly one of
`meeting_id` or `series_id`. Meeting-scope and series-scope summaries live
in the same table and FTS index — `search_summaries` hits both. Use series
scope for rollups across a weekly 1:1 / interview loop / project stage
("state of the Weekly 1:1 as of week N") without restitching per-meeting
summaries at retrieval time. Cascade-deletes with the series.

**`get_chat_context` is the chat-start shortcut.** One call returns the
scope metadata, every summary on the scope (with `transcript_stale`
flags), and — for meeting scope — summaries from any series the meeting
belongs to. Transcripts and full document text are opt-in so long series
don't blow the context window: transcripts default ON for meeting scope,
OFF for series scope (pass `include_transcripts=True` to force). Use this
to start a chat; use `get_meeting` / `get_series` when you only need
orientation metadata; use `get_transcript` / `get_summary` / `get_document`
for targeted pulls.

**Token-lean retrieval** is a first-class design goal. `search_*` returns
FTS snippets with stable back-references (`chunk_id` for transcripts,
`chunk_ord` for documents). The caller pulls only the matching pieces with
`get_transcript(..., time_range=[...])` or `get_document(..., format="chunks",
chunk_ords=[...])`. Default list/get responses are metadata-only; full text
must be asked for explicitly. Preserve this pattern when adding new
retrieval tools.

## Testing

- `uv run pytest` — ~167 fast tests, no ML deps required (uses the stub backend).
  `tests/test_workflow.py` is the cross-module regression guard — touches every
  tool in the canonical Claude Code flow; break it and you've broken the API.
- `MEETINGTOOL_E2E=1 uv run pytest tests/test_e2e.py` — real WhisperX against
  `tests/fixtures/sample.wav`. Fixture fetched by `scripts/fetch_sample_audio.sh`.
- Tests use `monkeypatch.setenv` + `config._settings = None` to reload config
  with per-test overrides. The `conn` fixture in `tests/conftest.py` creates
  a fresh SQLite DB and rebinds the `db.get_conn()` singleton.

## Do / don't

**Do:**
- Keep the MCP tool surface small and well-named. Claude Code reads tool
  docstrings at runtime — they matter.
- Add input validation (absolute paths, existing files, known IDs) at the
  tool boundary. Internals can assume validated input.
- Stage progress updates in `jobs.py` with meaningful names:
  `loading_model`, `asr`, `diarize`, `writing`, `resuming`.
- For new backends: accept an optional `window=(start, end)` kwarg and return
  segments with absolute timestamps. If the backend can't slice natively,
  fall back to full-file transcribe when `window` is `None` or `(0, inf)` —
  `jobs.py` will default to a single `(0, inf)` window when silero-vad is
  unavailable.

**Don't:**
- Don't add a UI, web server, or chat endpoint. Claude Code is the UI.
- Don't add per-user auth or shared storage — single-user local-first
  is a first-class constraint.
- Don't import heavy ML libraries at module scope. Lazy-load in backend
  classes only.
- Don't write synchronous `transcribe_and_return_transcript()` tools.
  Everything goes through jobs.

## Phase 3 (backlog — don't build yet)

- **Web-facing packaging**: a `uvx meetingtool` publish so other users on the
  team can wire the server without cloning this repo.
- **Per-kind summary templating hints**: optional prompt skeletons that
  Claude Code can request via `get_summary_template(kind)` to keep
  summaries across meetings structurally consistent.
- **Live-session validation**: drive the full 40-tool surface through a real
  Claude Code session against a multi-speaker recording. Document any
  friction that surfaces.

The schema and job model already support these additively; none require a
refactor of existing tools.
