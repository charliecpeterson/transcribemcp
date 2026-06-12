# meetingtool

A local-first meeting transcription MCP server for [Claude Code](https://claude.com/claude-code).
Claude Code is the agent; this project provides the tools (projects, meetings, transcription, search) and the SQLite + local-file storage.

Audio never leaves the machine. LLM endpoint is pluggable (Anthropic / internal vLLM / Ollama) via `ANTHROPIC_BASE_URL`.

---

## Status

Phase 1 + Phase 2 (core plumbing + speakers + search + documents): **complete on macOS** (Apple Silicon). Windows validation pending.

- ✅ FastMCP server, 41 MCP tools, async job runner, SQLite + FTS5 schema
- ✅ Windowed transcribe + checkpoint/resume — crashes mid-multi-hour recording
  don't lose committed work; `resume_job` picks up from the last completed window
- ✅ Stub backend (no ML deps; for smoke-testing the plumbing)
- ✅ WhisperX backend — validated end-to-end on a real JFK audio clip
- ✅ Qwen3-ASR backend (local + vLLM) — code complete; vLLM mode unit-tested with mock
- ✅ pyannote.audio diarization wrapper with max-overlap merge
- ✅ End-to-end test with real audio (`MEETINGTOOL_E2E=1 pytest tests/test_e2e.py`)
- ✅ Speaker ID assist (`get_speaker_cameos`), filtered transcript retrieval, FTS5 search over chunks
- ✅ Supplemental documents (.txt / .md / .pdf / .docx) with paragraph chunking + FTS
- ✅ Cross-meeting person identity (`create_person`, `link_speaker_to_person`)
  — search every quote a person has ever said across all meetings
- ✅ Meeting series (weekly 1:1s, interview loops) — scoped transcript search
- ✅ Cached summaries with FTS — "which meetings talked about X?" without
  re-reading transcripts

---

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
cd meetingtool
uv sync --extra dev
cp .env.example .env
# edit .env — set TRANSCRIPTION_BACKEND=stub to smoke-test without ML deps
```

Optional heavy ML deps (install only what you need):

```bash
uv sync --extra dev --extra whisperx --extra diarize                # WhisperX + pyannote
uv sync --extra dev --extra qwen3    --extra diarize                # Qwen3-ASR + pyannote
uv sync --extra dev --extra qwen3-mlx                               # Apple Silicon MLX (Qwen3)
uv sync --extra dev --extra cohere   --extra vad --extra diarize    # Cohere Transcribe + silero-vad + pyannote
uv sync --extra dev --extra cohere-mlx                              # Apple Silicon MLX (Cohere)
```

Backend quick reference:

| Backend | Strengths | Trade-offs |
|---|---|---|
| `whisperx` | Best-tested path; native segment timestamps; optional word-level alignment | Large models are slow on CPU |
| `qwen3_asr` | Modern LLM-decoded ASR; local or vLLM HTTP | No native timestamps in some decoder paths |
| `cohere` | Best-in-class AMI accuracy (8.15% WER); Apache 2.0; very fast on Apple Silicon via mlx-audio | LLM-decoded — timestamps come from silero-vad, not the model. English only for now. |

The `cohere` backend requires the `[vad]` extra because Cohere Transcribe emits
plain text with no segment boundaries; we run silero-vad first and synthesize
`Segment(start, end)` from the voiced spans.

`HF_TOKEN` in `.env` is required for pyannote to download its model weights.

---

## Wire into Claude Code

Add to `~/.claude.json` (or your Claude Code MCP config):

```json
{
  "mcpServers": {
    "meetingtool": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/charlie/projects/meetingtool",
        "run",
        "meetingtool"
      ]
    }
  }
}
```

Restart Claude Code. In a fresh session, tool calls like `create_project` will appear under the `meetingtool` server.

---

## Smoke test (stub backend, no ML deps)

With `TRANSCRIPTION_BACKEND=stub` in `.env`, try this in Claude Code:

```
Create a project called "Smoke Test".
Add a meeting titled "demo" with audio_path /tmp/dummy.wav (any file works — stub ignores it).
Transcribe it.
Poll get_status until it says done.
Show me the transcript.
```

You should see a fake 3-segment transcript with SPEAKER_00 / SPEAKER_01 labels. That confirms the async plumbing works end-to-end before you commit to installing WhisperX or Qwen3-ASR.

---

## Real audio test fixture

```bash
./scripts/fetch_sample_audio.sh    # downloads a short public-domain speech clip
```

Uses the ffmpeg bundled by `imageio-ffmpeg` (pulled in by `uv sync --extra dev`) —
no system-level ffmpeg install required. Writes to `tests/fixtures/sample.wav`.

Then run the full end-to-end test with a real ASR backend:

```bash
MEETINGTOOL_E2E=1 uv run pytest tests/test_e2e.py -v
```

First run downloads the WhisperX `base` model (~145 MB) and takes ~30 s on CPU.
Subsequent runs are a few seconds.

---

## Tools exposed to Claude Code

**Projects & meetings**

| Tool | Purpose |
|---|---|
| `create_project(name, description="")` | Create a project |
| `list_projects()` | List projects + meeting counts |
| `update_project(project_id, name?, description?)` | Rename / edit description |
| `delete_project(project_id)` | Cascade-delete everything (returns removed counts) |
| `add_meeting(project_id, title, audio_path, date?, auto_transcribe=True)` | Register an audio file; by default also enqueues a transcription job and returns `{job_id, status: "queued"}` |
| `list_meetings(project_id)` | Meetings in a project + status |
| `get_meeting(meeting_id)` | Rich overview: project, speakers, series, summary kinds, doc count |
| `update_meeting(meeting_id, title?, date?)` | Fix a typo or backfill the date |
| `delete_meeting(meeting_id)` | Cascade-delete rows (leaves audio on disk) |

**Transcription**

| Tool | Purpose |
|---|---|
| `transcribe_meeting(meeting_id)` | Enqueue async transcription, returns `{job_id}` |
| `retranscribe_meeting(meeting_id)` | Clear existing transcript + enqueue fresh job; flags existing summaries stale |
| `get_status(job_id?, meeting_id?)` | Poll a job's stage/progress/error + `progress_updated_at` heartbeat + `checkpoint_seconds` |
| `cancel_job(job_id)` | Cooperatively cancel a queued/running job; idempotent on terminal jobs; resets meeting to `pending` |
| `resume_job(job_id)` | Resume a crashed/cancelled job from its last completed window — skips re-transcribing anything already persisted |
| `list_jobs(status?, limit=20)` | Recent jobs |

`get_status` returns `progress_updated_at` alongside `progress`. Heavy ASR
backends (whisperx, qwen3-asr local) can sit on the same `progress` value
for minutes mid-stage — compare `progress_updated_at` to now to tell
"stuck" from "slow." Rule of thumb: >3–5 min without an update on a
`running` job is suspicious. `cancel_job(job_id)` is the escape hatch —
note that it's cooperative (the worker only notices between stages), so a
genuinely wedged stage won't abort until it returns to native Python.

Summaries carry a `transcript_stale` flag. `retranscribe_meeting` flips it
to `true` on every summary whose scope covers the meeting (the meeting
itself, plus any series it belongs to). `save_summary` resets it on every
write — so the flag means "this saved text describes an earlier version of
the transcript; consider regenerating."

### Long recordings: windowed transcribe + resume

`jobs.py` chunks audio into ~5-minute windows derived from silero-vad
(breaks only on silence — no fixed boundaries, no overlap). Each window's
chunks are persisted in a single transaction before the next window
starts, and `jobs.checkpoint_seconds` advances to that window's end. If
the process crashes or the job errors midway through a 3-hour recording,
work up to the last completed window is already in the DB.

`resume_job(job_id)` re-plans the *same* VAD windows on the same audio
(silero-vad is deterministic) and skips any with `end <= checkpoint_seconds`.
Diarization runs fresh at the end over the full set of persisted chunks
— it's cheap compared to ASR, and doing it once at the end avoids stitching
speaker labels across window seams.

Only `error` and `cancelled` jobs are resumable. `done` jobs raise; jobs
still `queued` or `running` raise with "cancel it first." Truly wedged
native-code stages (whisperx/pyannote inside a C call) won't respond to
`cancel_job` until they return — restart the MCP server to force the
situation, then `resume_job(job_id)` picks up cleanly.

**Transcript retrieval & search**

| Tool | Purpose |
|---|---|
| `get_transcript(meeting_id, format, speaker_labels?, speaker_names?, time_range?, max_chars?)` | Speaker-labeled text or JSON segments, with optional filters |
| `search_transcripts(query, project_id?, meeting_id?, limit=10)` | FTS5 search with snippets + bm25 ranking |

**Speakers (per-meeting) & persons (cross-meeting)**

| Tool | Purpose |
|---|---|
| `list_speakers(meeting_id)` | Per-speaker metadata (segment count, seconds spoken, person link) |
| `get_speaker_cameos(meeting_id, n_per_speaker=3, only_unnamed=True, include_attached_docs=False)` | High-density evidence for LLM speaker ID; pass `include_attached_docs=True` to bundle attendee lists / notes in one call |
| `assign_speaker(meeting_id, label, name, notes?)` | Set a meeting-local name |
| `create_person(name, email?, role?, notes?)` | Register a canonical person |
| `list_persons()` | All persons + how many meetings each appears in |
| `get_person(person_id)` | Person details + every meeting they're linked to |
| `link_speaker_to_person(meeting_id, label, person_id)` | Link a diarization label to a canonical person |
| `delete_person(person_id)` | Remove a person; speakers are unlinked (names preserved) |

Search can then scope by `person_id`: `search_transcripts("budget", person_id=...)`
returns every quote Sarah has ever given containing "budget" across all meetings.

**Meeting series**

| Tool | Purpose |
|---|---|
| `create_series(project_id, name, description?)` | Create a series ("Weekly 1:1", "Candidate X interview loop") |
| `list_series(project_id?)` | All series, optionally scoped to a project, with meeting counts |
| `get_series(series_id)` | Rich overview: members (each with `summary_kinds` + `chunk_count`), total duration, series-scope summary kinds, persons appearing across the series, document count |
| `add_meeting_to_series(series_id, meeting_id)` | Add a meeting (idempotent) |
| `remove_meeting_from_series(series_id, meeting_id)` | Remove a meeting (membership only; meeting is kept) |
| `delete_series(series_id)` | Delete the series; meetings are kept |

Scope a search to a series: `search_transcripts("action items", series_id=...)`
returns hits only from meetings in that series.

**Cached summaries**

Claude Code writes the summaries (it reads the transcript + context and
decides what to capture); these tools just persist and retrieve them.

| Tool | Purpose |
|---|---|
Summaries scope to either a single meeting **or** a series — pass exactly
one of `meeting_id` / `series_id` on write.

| Tool | Purpose |
|---|---|
| `save_summary(meeting_id?, kind, text, series_id?)` | Upsert by (scope, kind). `kind` is free-form — `overview`, `action_items`, `decisions`, `notes`, `rollup`, etc. |
| `get_summary(meeting_id?, kind?, series_id?)` | One kind or all kinds for the scope |
| `list_summaries(project_id?, meeting_id?, series_id?, kind?)` | Metadata list (no text); unfiltered returns both meeting- and series-scope rows |
| `delete_summary(meeting_id?, kind, series_id?)` | Remove one summary |
| `search_summaries(query, project_id?, meeting_id?, series_id?, kind?, limit=10)` | FTS5 over saved summaries (both scopes) |

Typical flows:
- After a transcript lands, Claude Code writes an `overview` + `action_items`
  summary on the meeting. Weeks later, `search_summaries("offsite")` surfaces
  every meeting where the offsite was discussed without re-reading transcripts.
- For a weekly 1:1, Claude Code can also write a `rollup` summary on the
  **series** covering every meeting to date — one query for "state of the
  Weekly 1:1 as of week N" without restitching per-meeting summaries.

**Chat context**

| Tool | Purpose |
|---|---|
| `get_chat_context(meeting_id? OR series_id?, include_transcripts?, include_documents=False, transcript_max_chars?)` | Bundle everything Claude needs to start a chat about a meeting or series in one call |

Always returned: scope metadata, speakers / persons, series memberships,
every summary on the scope (with `transcript_stale` flags). For meeting
scope, also returned: summaries from any series the meeting belongs to
— often load-bearing context ("last week you said X in the rollup").

Opt-in knobs (default chosen to avoid context-window blowups):
- `include_transcripts` defaults **True for meeting scope, False for
  series scope**. A 20-meeting weekly 1:1 series easily runs to hundreds
  of KB of transcript text — pass `True` explicitly when you want it.
- `include_documents` defaults **False**. When False you still get
  document metadata (titles, char counts) so you can decide what to pull.
- `transcript_max_chars` caps each transcript — pair with
  `include_transcripts=True` on a long series.

Use `get_meeting` / `get_series` when you only need metadata; use
`get_transcript` / `get_summary` / `get_document` for targeted pulls.
`get_chat_context` is specifically the "starting a chat session" shortcut.

**Supplemental documents**

| Tool | Purpose |
|---|---|
| `add_document(project_id, title, path, meeting_id?)` | Ingest .txt / .md / .pdf / .docx; chunks + FTS-indexes |
| `list_documents(project_id?, meeting_id?)` | Metadata only |
| `get_document(document_id, format="meta"\|"text"\|"chunks", chunk_ords?, max_chars?)` | Retrieve on demand |
| `search_documents(query, project_id?, meeting_id?, document_id?, limit=10)` | FTS5 over document chunks |
| `delete_document(document_id)` | Cascade-delete chunks (leaves source file on disk) |

**Misc**

| Tool | Purpose |
|---|---|
| `ping()` | Server reachable |

> Token-lean retrieval pattern: `search_*` tools return snippets with stable
> IDs (`chunk_id` for transcripts, `chunk_ord` for documents). Pull only the
> matching pieces back with `get_transcript(..., time_range=[...])` or
> `get_document(..., format="chunks", chunk_ords=[...])`.

---

## Speaker identification workflow

After transcription, diarization gives you labels (`SPEAKER_00`, `SPEAKER_01`,
…) but not names. There are three ways to fix that, composable:

**1. From transcript context alone.** Works when speakers self-introduce
("Hi, I'm Sarah from HR…"). One call, no extra input:

```
get_speaker_cameos(meeting_id)
```

Returns ~1 KB of high-signal utterances per unnamed speaker. Claude Code
reads it, proposes a mapping, and applies it with `assign_speaker`
(meeting-local) or `create_person` + `link_speaker_to_person` (cross-meeting).

**2. With an attendee list / agenda / meeting notes.** Attach the file
first, then do the identification in one round-trip:

```
add_document(project_id, "Attendee list", "/path/to/attendees.md", meeting_id=<mid>)
get_speaker_cameos(meeting_id, include_attached_docs=True)
```

The second call returns cameos **plus** the full text of every document
linked to the meeting. Claude Code can now cross-reference utterance
content against names on the list ("Sarah Chen — HR" ↔ "I'm Sarah from HR").

**3. User-driven rename in chat.** For any speaker — known, unknown, or
half-guessed — `assign_speaker(meeting_id, label, name)` accepts any free-form
string, so placeholders like `"Unknown guest (marketing?)"` are fine. To
make the identity searchable across every meeting that person attends:

```
p = create_person("Sarah Chen", role="EM")
link_speaker_to_person(meeting_id, "SPEAKER_00", p["id"])
```

After that, `search_transcripts("budget", person_id=p["id"])` returns every
budget-related quote Sarah has given across every meeting in the DB.

---

## Tests

```bash
uv run pytest
```

---

## Layout

```
src/meetingtool/
├── server.py                FastMCP instance
├── __main__.py              entry point (uv run meetingtool)
├── config.py                pydantic-settings from .env
├── db.py                    SQLite schema + migrations + FTS triggers
├── jobs.py                  ThreadPoolExecutor-based job runner
├── transcribe.py            backend router
├── diarize.py               pyannote wrapper + max-overlap merge
├── audio.py                 shared ffmpeg-based PCM decode (pyannote + VAD + Cohere)
├── vad.py                   silero-vad wrapper → [(start, end)] voiced spans
├── documents.py             text extractors (txt/md/pdf/docx) + paragraph chunker
├── backends/
│   ├── base.py              Segment / TranscriptionResult / Protocol
│   ├── stub.py              dev + test backend (no ML deps)
│   ├── whisperx_backend.py  faster-whisper + alignment
│   ├── qwen3_backend.py     local transformers OR vLLM HTTP
│   └── cohere_backend.py    Cohere Transcribe; VAD-derived timestamps; transformers or mlx-audio
└── tools/
    ├── projects.py          create_project, list_projects
    ├── meetings.py          add_meeting, list_meetings, delete_meeting,
    │                        get_transcript, retranscribe_meeting
    ├── jobs.py              transcribe_meeting, get_status, list_jobs
    ├── speakers.py          list_speakers, assign_speaker, get_speaker_cameos
    ├── persons.py           create/list/get/delete_person, link_speaker_to_person
    ├── series.py            create/list/get/delete_series, add/remove_meeting_to_series
    ├── summaries.py         save/get/list/delete/search_summary(ies)
    ├── search.py            search_transcripts (FTS5)
    └── documents.py         add/list/get/delete/search_document(s)
```
