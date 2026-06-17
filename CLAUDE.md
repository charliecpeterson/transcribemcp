# CLAUDE.md — transcribemcp

## What this is

A thin, local MCP server that transcribes audio and writes the result to a
JSON file on disk. That is the whole job: **audio in, transcript file out.**

The driving LLM (Claude Code, Goose, a local vLLM/Ollama loop, whatever) is
the orchestrator. It chats about transcripts, chains several together, adds
outside context, and writes summaries. This server does **not** do any of
that, and must not grow to. There is no database, no "meeting" ontology, no
projects/series/persons, no summaries-as-rows. The filesystem holds the
artifacts; the LLM holds the working memory.

This is the result of a deliberate restructure (see `PROJECT_PLAN.md`) that
deleted ~6,000 lines of DB + ontology code. The engine underneath
(backends, diarization, VAD, audio decode) is hard-won and survived intact.
Read `PROJECT_PLAN.md` before proposing structural changes.

## Environment

- **Packaging: `uv`** (not conda). Python 3.12 pinned via `.python-version`;
  `requires-python = ">=3.11,<3.14"`.
- The server runs with just `uv sync` (stub backend only). Real backends are
  optional extras:
  - `uv sync --extra whisperx --extra diarize` (WhisperX + pyannote)
  - `uv sync --extra qwen3 --extra diarize` (Qwen3-ASR + pyannote)
  - `uv sync --extra cohere --extra vad` (Cohere Transcribe + silero-vad)
  - `--extra qwen3-mlx` / `--extra cohere-mlx` add MLX on Apple Silicon
- ffmpeg is bundled via `imageio-ffmpeg` — never tell the user to
  `brew install ffmpeg`. Resolve the binary with
  `uv run python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`.
- Launched client-agnostically via `~/mcps/bin/transcribemcp-run`. Point
  Claude Code and Goose at that one path.

## Architecture

- **FastMCP** (`mcp[cli]`) over stdio. `server.py` holds the shared `mcp`
  instance + `ping`; `scribe_tools.py` registers `transcribe` and
  `read_transcript`; `__main__.py` imports `scribe_tools` and runs the server.
- **Synchronous + idempotent.** `transcribe` blocks until the transcript is
  written, then returns its path. If `<audio>.transcript.json` already exists
  it returns immediately without recomputing (`overwrite=True` forces a
  rerun). There is **no** job queue, no async polling, no checkpoint/resume —
  that all died with the DB. Don't add it back.
- **Filesystem as state.** Output is `<audio>.transcript.json` beside the
  source by default, or under `OUTPUT_DIR` if set. "Chaining transcripts" is
  the LLM holding two files in context — not a join.
- **Lazy ML imports.** `transcribe.py` is the backend router; it imports the
  selected backend only on first use. Never import torch/pyannote/
  transformers/whisperx at module top level.
- **Backends are pure ASR.** A backend's `transcribe(audio_path, *, progress,
  window=None)` returns segments with absolute timestamps. Diarization is
  **not** a backend concern: `pipeline.run_transcribe` runs `diarize()` once
  over the full file at the end and merges speakers via
  `diarize.assign_speakers`. (`window` is a vestigial kwarg from the old
  windowed pipeline; the thin pipeline always passes `None` = whole file.)
- **Shared audio + VAD.** `audio.py` decodes any format to mono 16 kHz
  float32 PCM via the bundled ffmpeg. `vad.py` exists because LLM-decoded ASR
  models (Cohere, future Granite/Phi-4) emit bare text with no segment
  boundaries; it synthesizes `Segment(start, end)` from voiced spans.

## Tool surface

```
ping                                   # reachability check
transcribe(audio_path, diarize=None,   # → writes JSON, returns {transcript_path, cached, ...}
           output_dir=None, overwrite=False)
read_transcript(path, format=text|srt|json,  # render + filter server-side
                time_range=[s,e], speaker="SPEAKER_01")
```

`transcribe` is synchronous. `read_transcript` does the token-lean retrieval:
filter by time range or speaker and render to text/SRT/json server-side, so
the LLM pulls only the slice it needs instead of reading the whole JSON. There
is intentionally **no** `list_transcripts` — the orchestrating LLM globs
`*.transcript.json` with its own filesystem tools.

## Long jobs and client timeouts

An hour of audio on CPU blocks the `transcribe` call for minutes. MCP clients
have tool-call timeouts, and **neither Claude Code nor Goose resets that
timeout on progress notifications** (verified 2026-06; Claude Code closed the
request as "not planned"). So:

- The fix for long files is to **raise the client's hard timeout**: Claude
  Code `MCP_TOOL_TIMEOUT` (env, ms); Goose per-extension `timeout`
  (config.yaml, seconds, default 300).
- If a call times out anyway, the **idempotent re-call** returns the finished
  file. Work isn't lost.
- The server logs progress stages to stderr for visibility; this is **not** a
  timeout defense, just local logging.

## Transcript schema

```json
{
  "schema_version": 1,
  "audio_path": "/abs/path/talk.wav",
  "backend": "whisperx", "model": "base", "language": "en",
  "diarized": false, "duration": 1234.5,
  "created_at": "2026-06-16T21:00:00-07:00",
  "segments": [{"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00", "text": "…"}]
}
```

Boring on purpose. Cross-file speaker identity ("SPEAKER_01 here is the same
person as SPEAKER_03 there") is the LLM's reasoning job, not a table.

## Config (`.env`)

`config.py` (pydantic-settings) reads `.env`. Keys: `TRANSCRIPTION_BACKEND`
(whisperx|qwen3_asr|cohere|stub), `WHISPER_MODEL`, `WHISPERX_DEVICE`/
`_COMPUTE_TYPE`, the `QWEN3_*` and `COHERE_*` knobs, `DIARIZE` + `HF_TOKEN`
(pyannote needs the community-1 license accepted), and `OUTPUT_DIR`.

## Testing

- `uv run --extra dev pytest` — fast, no ML deps (stub backend).
  `tests/test_thin_surface.py` covers the whole tool surface.
- `TRANSCRIBEMCP_E2E=1 uv run --extra whisperx --extra dev pytest tests/test_e2e.py`
  runs real backends against `tests/fixtures/sample.wav` (fetch via
  `scripts/fetch_sample_audio.sh`).
- Per-test config overrides use `monkeypatch.setenv` + `config._settings = None`.

## Do / don't

**Do:** keep the surface tiny and the docstrings sharp (the LLM reads them at
runtime). Validate inputs (absolute paths, existing files) at the tool
boundary; internals assume validated input. Lazy-load ML in backends only.

**Don't:** re-add a DB, an ontology, summaries, or a job queue. Don't add a
UI/web server/auth — the driving LLM is the UI, single-user local-first is a
constraint. Don't import heavy ML at module scope. Don't make `transcribe`
async or add polling. If a change grows the line count instead of shrinking
it, stop and re-read `PROJECT_PLAN.md`.
