# Meeting Assistant — Project Plan

> **Purpose:** A local-first, privacy-focused meeting transcription and analysis tool for internal IT/HR use. Designed to be fully self-hostable, with no mandatory cloud dependencies.

---

## Background & Motivation

HR staff currently use Zoom's AI features for meeting transcription but want:
- Control over where the LLM runs (Anthropic API, internal vLLM server, or local Ollama)
- Less exposure to third-party services (Zoom AI, etc.)
- The ability to **chain related meetings** across time (e.g. an ongoing investigation or hiring process)
- Upload supplemental documents (job postings, HR policies, prior notes) as context
- A chat interface to query hours of transcripts naturally

This tool is built and maintained by an HPC sysadmin for internal org use. It is not a SaaS product. Privacy and data locality are first-class requirements.

---

## Architecture Overview

```
User's Laptop
├── Audio files (local, user-managed)
├── Transcription backend (pluggable — see Transcription Stack)
│     ├── Option A: WhisperX / faster-whisper + pyannote
│     └── Option B: Qwen3-ASR + pyannote
├── Claude Code  (agent brain — not maintained by us)
├── MCP Server (Python)  (tools + SQLite storage — what we build)
│
└── → Remote LLM API
        ├── Anthropic API  (default)
        ├── Internal vLLM server  (for full privacy)
        └── Local Ollama  (fully air-gapped option)
```

### Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Agent / orchestration | Claude Code | We don't maintain this; it handles the hard reasoning |
| MCP server language | Python | Native to WhisperX, faster-whisper, Qwen3-ASR, pyannote |
| Storage | SQLite + local files | Simple, no server, per-user data locality |
| Transcription | Pluggable backend | WhisperX (stable) or Qwen3-ASR (newer, higher accuracy) |
| Diarization | pyannote.audio | Best open-source option regardless of ASR backend |
| UI (v1) | Claude Code terminal | Validate tools first before building UI |
| UI (v2) | Thin web wrapper | For non-technical users (HR) |
| LLM endpoint | Configurable | Anthropic / vLLM / Ollama via `ANTHROPIC_BASE_URL` |

### What Leaves the Machine

Only transcript text sent to the LLM API. Audio files, raw transcripts, documents, and SQLite database never leave the user's machine. If pointed at a local Ollama instance, **nothing leaves the machine at all**.

---

## Phase 1 — Core Plumbing

Goal: Get Claude Code talking to real tools with a real audio file end-to-end.

### Tools to build

- `create_project(name, description)` — create a named project (e.g. "Q1 Hiring", "Jones Investigation")
- `add_meeting(project_id, title, date, audio_path)` — register a meeting file
- `transcribe_meeting(meeting_id)` — run configured transcription backend (WhisperX or Qwen3-ASR) + pyannote diarization as subprocess, store result, update status
- `get_transcript(meeting_id)` — return transcript text to Claude Code
- `list_projects()` — list all projects
- `list_meetings(project_id)` — list meetings within a project

### Milestone

Claude Code can transcribe a real audio file, store the result, and return the transcript on request.

---

## Phase 2 — Usefulness

Goal: Tools that make Claude Code actually useful for HR workflows.

### Tools to build

- `assign_speaker(meeting_id, label, name)` — map `SPEAKER_00` → "Sarah from HR"
- `get_summary(meeting_id, type)` — prompt templates for: `minutes` | `decisions` | `action_items` | `findings`
- `add_document(project_id, file_path, title)` — ingest supplemental docs (policy PDFs, job postings, etc.)
- `search_transcripts(query, project_id?)` — SQLite FTS full-text search across transcripts
- `link_meetings(project_id, meeting_ids)` — explicitly chain related meetings

### Milestone

HR can ask Claude Code questions like:
- *"Summarize the key decisions from all meetings in the Jones project"*
- *"What concerns have been raised across these check-ins?"*
- *"Does anything in the interview recordings conflict with the job posting I uploaded?"*

---

## Phase 3 — Power Features (v2 backlog)

These are noted now so they don't get re-debated later. Do not build these until Phase 2 is validated with real users.

| Feature | Notes |
|---|---|
| Semantic / embedding search | Upgrade from SQLite FTS; needs embedding model |
| Cross-meeting speaker identity | Same person appearing in multiple meeting files |
| Async transcription with progress | Long meetings (1hr+) need status polling, not blocking |
| Pluggable transcription backend | Config-driven in v1 (WhisperX vs Qwen3-ASR); remote/API transcription service option for v2 |
| Shared / network storage | For teams sharing meeting projects |
| Thin web UI | Wraps Claude Code for non-technical HR users |
| Chunked document indexing | Better retrieval for large supplemental docs |

---

## Data Model (SQLite)

```sql
-- A "case" or topic grouping related meetings
CREATE TABLE projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TEXT NOT NULL
);

-- Individual meeting recordings
CREATE TABLE meetings (
    id               TEXT PRIMARY KEY,
    project_id       TEXT REFERENCES projects(id),
    title            TEXT NOT NULL,
    date             TEXT,
    duration_seconds INTEGER,
    audio_path       TEXT NOT NULL,
    transcript_path  TEXT,
    status           TEXT DEFAULT 'pending',  -- pending|transcribing|ready|error
    created_at       TEXT NOT NULL
);

-- Speaker labels within a meeting
CREATE TABLE speakers (
    id         TEXT PRIMARY KEY,
    meeting_id TEXT REFERENCES meetings(id),
    label      TEXT NOT NULL,   -- e.g. SPEAKER_00 (from WhisperX)
    name       TEXT,            -- e.g. "Sarah - HR" (human assigned)
    notes      TEXT
);

-- Supplemental context documents
CREATE TABLE documents (
    id         TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id),
    title      TEXT NOT NULL,
    file_path  TEXT NOT NULL,
    file_type  TEXT,
    created_at TEXT NOT NULL
);

-- Transcript segments (for search; embeddings added later)
CREATE TABLE chunks (
    id         TEXT PRIMARY KEY,
    meeting_id TEXT REFERENCES meetings(id),
    speaker_id TEXT REFERENCES speakers(id),
    text       TEXT NOT NULL,
    start_time REAL,
    end_time   REAL
    -- embedding BLOB  -- reserved for v2 semantic search
);
```

---

## Repo Structure

```
meeting-mcp/
├── mcp_server.py          # Entry point, tool registration
├── db.py                  # SQLite setup, schema, migrations
├── transcribe.py          # Backend router — picks WhisperX or Qwen3-ASR based on config
├── backends/
│   ├── whisperx.py        # WhisperX / faster-whisper implementation
│   └── qwen3_asr.py       # Qwen3-ASR implementation
├── diarize.py             # pyannote.audio wrapper (shared by both backends)
├── search.py              # FTS search logic
├── summarize.py           # Prompt templates for summary types
├── models/
│   ├── project.py
│   ├── meeting.py
│   ├── speaker.py
│   └── document.py
├── data/                  # SQLite DB lives here (gitignored)
├── audio/                 # Optional symlink or config path
├── pyproject.toml
├── .env.example           # LLM endpoint + transcription backend config
└── README.md              # Claude Code wiring instructions
```

---

## Transcription Stack

The transcription backend is **pluggable by design** — selected via a config flag. The MCP tool interface (`transcribe_meeting()`) stays identical regardless of which backend is active. This allows switching backends without touching any other code.

```bash
# In .env or config
TRANSCRIPTION_BACKEND=whisperx   # or: qwen3_asr
```

---

### Option A: WhisperX + faster-whisper (Stable, Recommended for v1)

**ASR:** `faster-whisper` — used under the hood by WhisperX; faster and lower memory than original Whisper

**Diarization:** `pyannote.audio` — bolted on after transcription to assign speaker labels

**Why choose this:**
- Battle-tested, large community, well-documented
- Works reliably on CPU (slower) and GPU
- WhisperX handles chunking of long audio files well
- More mature tooling around edge cases (noise, accents, overlapping speech)

**Downsides:**
- Accuracy ceiling is lower than newer models
- WhisperX is a community project, not officially maintained

---

### Option B: Qwen3-ASR + pyannote (Newer, Higher Accuracy)

**ASR:** `Qwen3-ASR` — released January 2026 by Alibaba/Qwen team. State-of-the-art among open-source ASR models, competitive with strongest proprietary APIs. Two size options:
- `Qwen3-ASR-0.6B` — best accuracy-to-size tradeoff, strong choice for on-device/laptop use
- `Qwen3-ASR-1.7B` — SOTA accuracy, needs more VRAM

**Diarization:** `pyannote.audio` — same as Option A. Qwen3-ASR does not do native diarization; speaker diarization is an open feature request on the official toolkit. The architecture ends up being: Qwen3-ASR for transcription → pyannote for "who spoke when" → merge results. Same pipeline shape as WhisperX.

**vLLM support:** Qwen3-ASR can be served via vLLM, meaning if you have a vLLM server already running internally, you could offload transcription there rather than running it on each user's laptop.

**Why choose this:**
- Noticeably better accuracy, especially on noisy audio, accents, and mixed-language content
- 52-language support
- vLLM-compatible — aligns with your existing infrastructure
- Apple Silicon MLX port exists (with pyannote diarization) for Mac users

**Downsides:**
- Very new (weeks old as of writing) — ecosystem still forming
- Less community troubleshooting resources
- Slightly more setup complexity

---

### Diarization: pyannote.audio (Both Options)

Both backends rely on `pyannote.audio` for speaker diarization. This is still the best open-source option for "who spoke when."

**Important:** pyannote has a **non-commercial license** for the standard weights. For internal org use this is typically fine, but verify with your org before deploying. The license requires accepting terms on Hugging Face and using an `HF_TOKEN`.

---

### CPU vs GPU Summary

| Hardware | Option A (WhisperX) | Option B (Qwen3-ASR 0.6B) |
|---|---|---|
| CPU only | ✅ Works, slow | ✅ Works, slow |
| Modest GPU (4-8GB VRAM) | ✅ Fast | ✅ Fast |
| Good GPU (8GB+ VRAM) | ✅ Very fast | ✅ Very fast (1.7B viable) |
| Apple Silicon | ✅ via MPS | ✅ via MLX port |
| vLLM server (remote) | ❌ | ✅ Offload transcription |

---

### Recommended Approach

Start with **Option A (WhisperX)** for v1 — it's stable and you can get end-to-end working faster. Design the `transcribe.py` module with a clean internal interface so swapping to **Option B (Qwen3-ASR)** later is just a new backend implementation, not a refactor. By the time you're ready for v2, Qwen3-ASR's ecosystem will be more mature.

---

## LLM Configuration

Claude Code supports `ANTHROPIC_BASE_URL` override. This means the LLM endpoint is fully configurable:

```bash
# Use Anthropic directly (default)
ANTHROPIC_API_KEY=sk-ant-...

# Use internal vLLM server (OpenAI-compatible)
ANTHROPIC_BASE_URL=https://your-vllm-server/v1
ANTHROPIC_API_KEY=your-internal-key

# Use local Ollama (fully air-gapped)
ANTHROPIC_BASE_URL=http://localhost:11434/v1
ANTHROPIC_API_KEY=ollama
```

The MCP server itself does not call the LLM — Claude Code does. The MCP server only provides tools and data.

---

## v1 Usage Flow (Claude Code)

1. User has audio file at `/path/to/meeting.m4a`
2. Opens Claude Code, configured with this MCP server
3. Types naturally: *"Transcribe this meeting and add it to the Jones project"*
4. Claude Code calls `add_meeting()` then `transcribe_meeting()`
5. After transcription: *"Assign the speakers — Sarah is SPEAKER_00, Mike is SPEAKER_01"*
6. Claude Code calls `assign_speaker()` for each
7. *"Give me an action items summary"* → Claude Code calls `get_summary(meeting_id, 'action_items')`
8. *"Upload this job posting for context"* → `add_document()`
9. *"Search all Jones meetings for any mention of the budget"* → `search_transcripts()`

No UI. No buttons. Just Claude Code and tools that work.

---

## What We Are NOT Building (v1)

- Real-time transcription / live meeting joining
- Audio recording (user provides their own files)
- User authentication / multi-user access control
- Cloud sync or shared storage
- A traditional web app with forms and state management
- Our own chat/agent loop (Claude Code handles this)

---

## Open Questions for Later

- Do HR users need to **share** meeting projects with each other? (Determines if local-only is enough or if a lightweight local server is needed)
- What models are available on the internal vLLM server? (Affects summary quality; also relevant if offloading Qwen3-ASR transcription there)
- Is pyannote's license acceptable for internal org use? (Required regardless of ASR backend choice)
- What's the realistic hardware profile of HR user laptops? (GPU or CPU-only — affects whether Qwen3-ASR 1.7B is viable vs 0.6B)
- Should speaker identity persist across meetings? (e.g. "Sarah" in meeting 1 = "Sarah" in meeting 5)
- When to switch from WhisperX to Qwen3-ASR? (Revisit once Qwen3-ASR ecosystem matures — likely 3-6 months)
