# Project Plan: thin local transcription MCP (meetingmcp restructure)

> Living document. Updated incrementally by the deep-planner skill.
> Last updated: 2026-06-16
> Current phase: planning complete

## Goal                                                    (always)
Strip meetingmcp down to a thin local transcription MCP — no DB, no
ontology — that takes audio in and writes transcript files out, with
the driving LLM doing everything the ontology used to do.

## Archetype                                               (always)
- **Primary**: library/CLI (a local MCP server / tool).
- **Secondary**: none.
- **Expertise calibration**: user owns MCP design as "joint" and
  HPC/deployment as "defer." Conductor leads on MCP-shape and code
  structure; defers to user on deployment reality.

## Scope                                                   (always)
### In scope
- Keep the transcription engine: `audio.py`, `backends/`,
  `diarize.py`, `vad.py`, `config.py`, backend-facing half of
  `transcribe.py` (~1,000 LOC). Do **not** rewrite it.
- New thin tool surface: `transcribe` (writes transcript file,
  idempotent), `read_transcript`, optional `list_transcripts`.
- Filesystem-as-state, LLM-as-orchestrator.
- Harness-agnostic (Claude Code, Goose, any MCP client) via the
  `~/mcps/bin/<name>-run` wrapper.
- Deployment: Mac (mX, MLX/CPU) and CUDA GPU with CPU fallback.

### Out of scope
- SQLite DB (`db.py`), the ontology (projects/series/meetings/
  persons/summaries/documents/speakers), all of `tools/`, `jobs.py`,
  per-window checkpoint/resume. ~3,300 LOC deleted.
- HPC batch / Slurm path (option 3, split engine into standalone CLI).
- Embedded summarization (the LLM in the loop summarizes).
- UI, web server, auth, shared storage.

## Decision Log                                            (always)
- **[2026-06-16] Decision-mapping mode**
  - **Choice**: run this session as decision-mapping; direction is
    settled, four genuine decisions remain.
  - **Why**: restructure-plan.md already fixes the architecture;
    open questions are name, output location, long-job handling,
    backend count.

- **[2026-06-16] Deployment targets**
  - **Choice**: Mac (mX) + CUDA GPU with CPU fallback. Harness-agnostic.
  - **Why**: user's real targets. HPC batch explicitly excluded.
  - **Alternatives considered**: HPC batch (Stampede3/Anvil) — out;
    would force option-3 engine/MCP split.
  - **Revisit if**: a real need to transcribe hour-plus audio on the
    cluster appears.

- **[2026-06-16] Long-job handling**
  - **Choice**: `transcribe` is synchronous, streams
    `ctx.report_progress()` (engine already produces a progress
    stream), writes `<audio>.transcript.json`, returns the path.
    Idempotent: re-call returns the existing file instantly.
  - **Why**: smallest tool surface — no poll tool, no job handle, no
    status sidecar. `ctx.report_progress()` is sent for UX (stage
    visibility), **not** as a timeout defense — see the timeout note.
  - **Timeout reality (verified 2026-06-16)**: neither Claude Code nor
    Goose resets the tool-call timeout on progress notifications.
    Claude Code lacks a `progressToken` and closed the request as "not
    planned" (issue #58687); Goose's per-extension `timeout` is a hard
    cap. Both expose a configurable hard timeout — Claude Code
    `MCP_TOOL_TIMEOUT` (env, ms), Goose per-extension `timeout`
    (config.yaml, seconds, default 300). The real long-file mitigation
    is **raise the client timeout + idempotent re-call**, both
    client-side config the user sets per harness in realtime.
  - **Alternatives considered**: background-thread + sidecar poll
    (survives any timeout without client config, but reintroduces a
    single-worker thread + status file — deferred behind a concrete
    trigger); start-sync-add-later.
  - **Revisit if**: raising the client timeout proves impractical for a
    real workflow (e.g. a harness with no timeout knob). Then add
    background + sidecar (option 2).

- **[2026-06-16] Backend count**
  - **Choice**: keep all — whisperx, cohere, qwen3, stub. Backend
    interface (`TranscriptionBackend`) preserved.
  - **Why**: already working across machines; covers Mac-GPU (MLX)
    and CUDA corners; the pluggable interface costs nothing to retain.
  - **Alternatives considered**: trim to two + stub (less config
    surface); one + stub (loses Mac GPU path).
  - **Revisit if**: cross-machine install permutations become a
    maintenance burden on a solo-maintained tool.

- **[2026-06-16] Name**
  - **Choice**: `transcribemcp`. Repo dir, Python package
    (`src/meetingtool/` → `src/transcribemcp/`), wrapper
    (`~/mcps/bin/transcribemcp-run`), and client registrations rename.
  - **Why**: literal, matches `~/mcps/<name>` convention, sheds the
    inaccurate "meeting" ontology implication.
  - **Alternatives considered**: scribe/localscribe (generic/overloaded);
    keep meetingmcp (zero churn but misleading).

- **[2026-06-16] Output location**
  - **Choice**: write `<audio>.transcript.json` beside the source by
    default; allow `OUTPUT_DIR` override in `.env`. Idempotence checks
    the resolved path.
  - **Why**: simple default, but survives read-only/shared source dirs
    (Zoom folders, mounted shares).
  - **Alternatives considered**: strictly next-to-audio (breaks on
    unwritable dirs); always-dedicated-dir (needs a stable audio→file
    key, more code).

- **[2026-06-16] Transcript schema**
  - **Choice**: `{audio_path, model, diarized, duration, segments:
    [{start, end, speaker, text}]}`. `read_transcript` renders
    text / json / SRT, with optional `time_range` and `speaker` filters.
  - **Why**: boring, exportable, mirrors the kept engine's segment
    output. Cross-file speaker identity stays the LLM's job (no
    persons table).

## Deferred Register                                       (always)
| Item | Why deferred | Trigger to revisit |
|------|--------------|--------------------|
| HPC batch / standalone transcribe CLI (option 3) | No present need; not a deployment target | Real cluster transcription workload |

## Open Questions                                          (always)
All Phase-1 decisions resolved. Timeout behavior verified (see the
Long-job decision): neither Claude Code nor Goose resets on progress;
both have a configurable hard timeout the user raises per harness.
Remaining item is a wiring check, not a decision:
- Confirm `ctx.report_progress()` plumbs cleanly from the existing
  backend `progress` closure through FastMCP (UX only).
- Document the per-harness timeout bump in the README (Claude Code
  `MCP_TOOL_TIMEOUT`; Goose extension `timeout`) so long-file users
  aren't surprised.

## Roadmap                                                 (if Phase 9 ran)

Sequencing principle for a deletion-heavy refactor: **build the new
thin surface alongside the old, get it green, then delete in one
sweep, then rename.** Never sit in a broken state. The `stub` backend
+ `tests/test_e2e.py` (real whisperx on `sample.wav`) are what guard
the engine while the ontology tests get deleted.

### Phase 1: build the thin surface alongside the old
The new code lands in new modules; the old DB/ontology server still
runs untouched until Phase 2.
- [ ] `pipeline.py` — one synchronous function: resolve output path →
      idempotent early-return if it exists → decode → backend ASR over
      the **full file** (no windows) with progress → if diarize:
      `diarize()` + `assign_speakers()` → write JSON. Reuses
      `diarize.assign_speakers` (already pure) and `preflight_diarize`
      (HF-token boundary check).
- [ ] Transcript schema: `{audio_path, model, diarized, duration,
      segments: [{start, end, speaker, text}]}`. One writer, one reader.
- [ ] `transcribe(audio_path, diarize=…, backend=…, output_dir=…)`
      MCP tool — wires the backend `progress` closure to
      `ctx.report_progress()`; returns the resolved path.
- [ ] `read_transcript(path, format=text|json|srt, time_range=…,
      speaker=…)` — text/SRT rendering + light filtering.
- [ ] `list_transcripts(dir)` (optional, build only if a real flow
      needs it).
- [ ] Output-path resolution: `<audio>.transcript.json` next to
      source, `OUTPUT_DIR` override in `.env`. Idempotence = resolved
      path exists.
- [ ] New fast tests against `stub`: idempotence (second call no
      recompute), schema round-trip, the three render formats,
      output-path resolution, read-time filters.
**Out of scope for this phase**: deleting anything; the rename.
**Effort**: ~1 focused day. This is the only phase that writes net-new code.

### Phase 2: delete the DB and the ontology
Now the new surface is green, remove the old one in one sweep.
- [ ] Delete `db.py`, `documents.py`, `jobs.py`, all of `tools/`
      (chat, documents, jobs, meetings, persons, projects, search,
      series, speakers, summaries).
- [ ] Delete the matching tests (`test_workflow.py` and every
      ontology/DB/jobs test). Keep `test_e2e.py`, `conftest.py`
      (trim DB fixtures), and the new Phase-1 tests.
- [ ] Strip `transcribe.py` to the backend router only (it already
      mostly is) or fold it into `pipeline.py`.
- [ ] Rewrite `server.py` / `tools/__init__.py` to register only
      `ping`, `transcribe`, `read_transcript`, `list_transcripts`.
- [ ] Prune `pyproject.toml`: drop extras/deps used only by the
      ontology (PDF/DOCX parsers from `documents.py`, any FTS-only
      bits). Keep the engine extras (whisperx/cohere/qwen3/diarize/mlx).
- [ ] **Data migration note**: existing SQLite data is dev-era; decide
      abandon vs. one-off export script. Default: abandon (no
      production data). Confirm before `rm` the DB file.
**Out of scope for this phase**: the rename.
**Effort**: ~half a day. Mostly `rm` + registration rewiring. The diff
should be overwhelmingly deletions — if it grows, stop and re-check.

### Phase 3: rename to transcribemcp + rewire clients + docs
- [ ] Rename `src/meetingtool/` → `src/transcribemcp/`, repo dir,
      `pyproject` name, console-script entry point.
- [ ] Rename the wrapper to `~/mcps/bin/transcribemcp-run`.
- [ ] Re-register in Claude Code and Goose client configs under the
      new name; remove the old `meetingmcp` registration.
- [ ] Rewrite `CLAUDE.md` and `README.md` for the thin surface; delete
      `meeting-assistant-plan.md` and the old phase plan references.
- [ ] Verify end-to-end: `MEETINGTOOL_E2E=1 uv run pytest
      tests/test_e2e.py` against `sample.wav` through the new
      `transcribe` tool, on both Mac (MLX) and CUDA if available.
- [ ] **Verification gate**: confirm Goose resets its tool-call
      timeout on progress notifications. If not, re-open the
      background+sidecar decision before relying on long-file sync.
**Out of scope for this phase**: nothing — this closes the restructure.
**Effort**: ~half a day, mostly mechanical.

## Dependencies & Risks                                    (always)
- **Re-growth risk**: third restructure; the failure mode is
  re-accreting machinery. Guardrail: a handful of pure-ish tools,
  files on disk, no DB, no ontology. If the diff grows instead of
  shrinks, stop.
- **Engine fragility**: whisperx + VAD + diarization took several
  rounds to configure across machines. Restructure touches the
  tool/persistence surface only.
