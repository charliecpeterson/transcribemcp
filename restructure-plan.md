# Restructure plan: thin local transcription MCP

Design snapshot captured 2026-06-16, to be picked up later with the
`deep-planner` skill. This is a refactor of the *surface* of the existing
MCP, not a rewrite of the transcription engine. The engine works; the goal
is to shed the layer that grew around it.

## History (why we're here)

1. Started as a full app: people upload meetings, the app saves them, they
   browse back and forth. Grew too complex, never shipped, abandoned.
2. Stripped to an MCP (`meetingmcp`): still persistent. SQLite DB, an
   ontology of projects / series / meetings / persons, speaker-to-person
   linking, FTS search, summaries-as-rows, job checkpoint/resume.
3. This plan: step further back. Drop the database and the ontology. Keep
   only the part that has value (audio in, transcript + diarization out)
   and let the LLM in the loop do the rest.

This is the third move toward *less*. Each prior move was correct. The tell
that this one is also correct: it is mostly deletion.

## North star

A completely local path to transcribe audio and reason over it with a local
LLM. The MCP transcribes and diarizes. The LLM (Claude Code, Goose, local
vLLM/Ollama, whatever is driving) stays in the loop to chat, read
transcripts, chain several together, and add outside context. Nothing about
this is meeting-specific; "meeting" was an assumption baked into the
ontology we're removing.

## The reframe: filesystem-as-state, LLM-as-orchestrator

"No persistence" and "see various transcripts and chain them" are in tension
only if persistence means a database. It doesn't have to.

- The LLM holds the working memory (the conversation).
- The filesystem holds the artifacts (transcript files on disk).
- "Chaining" is just having two transcript files in context and reasoning
  over them. No table, no join, no app state.

Transcription is expensive (a 244 MB file grinds for minutes on CPU), so the
result must survive past the chat. That means files on disk, written once,
idempotent on re-request. What we are deleting is the DB and the ontology,
not durability.

## What survives, what dies

Keep (this is the hard-won core, do not rewrite):

- `audio.py` (ffmpeg decode via bundled imageio-ffmpeg)
- `backends/` (whisperx, cohere, qwen3, stub)
- `diarize.py`, `vad.py`
- the per-machine config (device/compute, `.env`, per-machine extras,
  the launch wrapper)

Delete or collapse:

- `db.py`
- most of `tools/`: meetings, series, persons, projects, summaries,
  speakers, search, documents, jobs
- the per-5-minute-window checkpoint/resume machinery (it only earned its
  keep because of the DB)

## Proposed tool surface (small on purpose)

- `transcribe(audio_path, diarize=…, model=…, …)` -> runs the engine,
  writes `<audio>.transcript.json`, returns the path. Idempotent: if the
  output already exists, return it without recomputing.
- `read_transcript(path, format=text|json, time_range=…, speaker=…)` ->
  reads a transcript file with light filtering/formatting.
- `list_transcripts(dir)` (optional) -> enumerate transcript files.

Summarization leaves the MCP entirely. The old `save_summary` never
generated anything; it stored text the LLM produced. The in-loop LLM is
already the summarizer, so there is no reason to embed an LLM call in a tool
that is meant to stay local and dumb.

"Meeting" disappears for free once the ontology is gone. The engine does not
care whether the audio is a meeting, a lecture, or a voice memo.

## The one load-bearing decision: long-running transcription

An hour of audio on CPU blocks the tool call for many minutes. This is what
originally justified the checkpoint/resume code. Options, in order of how
much to reach for them:

1. Idempotent-on-file, synchronous. `transcribe` returns instantly if
   `<audio>.transcript.json` exists, else runs and writes it. Resume
   granularity is the whole file: if it dies, re-run. Good enough up to
   ~20-30 min clips. **Start here.**
2. Add a sidecar progress file for long jobs: write
   `<audio>.transcript.partial.json` as chunks finish and return a path the
   LLM can poll by reading the file. A one-file job system, not a DB. Keeps
   the long-job survivability we actually used at a fraction of the
   complexity. **Add only when a real file makes you wait.**
3. Split engine from MCP: a local `transcribe` CLI does the heavy work
   (terminal, cron, or `$SCRATCH` batch on HPC), and the MCP only reads
   transcript files. Cleanest separation, most reusable. **"Wait for the
   third use"; do not build preemptively.**

Guardrail: do not rebuild per-window DB checkpointing. Without the DB it is
not worth it.

## Smaller decisions to settle before coding

- Transcript JSON schema: segments with `start`, `end`, `speaker`, `text`
  is essentially the whole thing. Keep it boring and exportable to plain
  text and SRT.
- Cross-file speaker identity (for chaining): make it the LLM's job.
  "SPEAKER_01 here is the same person as SPEAKER_03 there" is reasoning over
  two files in context, not a `persons` table.
- Output location: next to the audio (`<audio>.transcript.json`) vs a
  dedicated output dir vs `$SCRATCH` on HPC. Decide per the deployment
  target.

## Constraints to preserve (already working, keep them)

- Client-agnostic launch via the `~/mcps/bin/<name>-run` wrapper, so Claude
  Code and Goose both point at one command.
- Per-repo `.env` as the single source of config (backend, device, diarize,
  HF token).
- Per-machine extras selection (mlx / cuda / cpu) and the
  ctranslate2-has-no-Metal fact: whisperx is CPU-only on Apple Silicon; use
  the MLX cohere backend if GPU speed on the Mac matters.

## Risks and guardrails

- This is round three of restructuring. The risk is re-growing. Hold the
  line: a handful of pure-ish tools, files on disk, no DB, no ontology.
- Do not rewrite the engine. It took several rounds to get whisperx + VAD +
  diarization installed and configured correctly across machines. The
  restructure is the tool/persistence surface only.
- A good restructure here is mostly `rm`. If the diff starts growing instead
  of shrinking, stop and re-check.

## Open questions for deep-planner

- New name? "meeting" is no longer accurate (e.g. `transcribemcp`,
  `localscribe`). Renaming touches the repo, the wrapper, and client
  registrations.
- Deployment targets to support explicitly: Mac (local), 4090 (local), HPC
  batch. Each implies different output locations and job handling.
- Does `transcribe` ever need a non-file return for very long jobs (the
  sidecar in option 2), or is synchronous-with-idempotence enough for the
  real workload?
- Keep multiple backends (whisperx / cohere / qwen3) or commit to one now
  that the ontology is gone?
