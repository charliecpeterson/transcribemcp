# Project Plan: thin local transcription MCP (meetingmcp restructure)

> Living document. Updated incrementally by the deep-planner skill.
> Last updated: 2026-06-16
> Current phase: Phase 1 complete; Phase 2 (delete DB + ontology) next

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

- **[2026-06-16] Repo move (done early, ahead of the rename)**
  - **Choice**: relocated to `~/mcps/transcribemcp` and re-pointed
    `origin` to `github.com/charliecpeterson/transcribemcp`. Wrapper
    `~/mcps/bin/meetingmcp-run` `--directory` path updated. Plan +
    restructure-plan pushed (commit b85bd29).
  - **Deferred to Phase 3** (unchanged from roadmap): in-code package
    rename `src/meetingtool/` to `src/transcribemcp/`, the `meetingtool`
    console script, the wrapper *filename*, and the `.claude.json`
    registration key. Live MCP keeps working because the wrapper
    filename and script name are untouched.
  - **Note**: old `github.com/charliecpeterson/meetingmcp` repo now
    orphaned; user's call whether to archive/delete it.

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
runs untouched until Phase 2. **DONE 2026-06-16** (commit pending).
- [x] `pipeline.py` (pure, no MCP import) — `run_transcribe`: resolve
      path → idempotent early-return → backend ASR over the **full file**
      (no windows) → if diarize: `diarize()` + `assign_speakers()` →
      write JSON. Plus `transcript_path_for` and the render helpers.
- [x] Transcript schema: `{schema_version, audio_path, backend, model,
      language, diarized, duration, created_at, segments:[{start, end,
      speaker, text}]}`. Written by `_build_doc`, read by
      `render_transcript`. `active_model` property added to config.
- [x] `transcribe(audio_path, diarize=None, output_dir=None,
      overwrite=False)` MCP tool in `scribe_tools.py`. Returns
      `{transcript_path, cached, diarized, model, duration, segments}`.
- [x] `read_transcript(path, format=text|json|srt, time_range, speaker)`
      — render + filter server-side.
- [x] Output-path resolution: `<audio>.transcript.json` beside source,
      `OUTPUT_DIR` override (added to config). Idempotence = path exists.
- [x] 9 fast tests in `tests/test_thin_surface.py` against `stub`:
      path resolution (×3), schema, idempotence+overwrite, 3 formats,
      filters, tool validation, tool roundtrip+cache flag. Full suite
      green: 169 passed / 11 skipped.

**Deviations from the original checklist (deliberate, see notes):**
- **No `ctx.report_progress()` streaming.** Bridging a sync ASR
  callback to the async loop from a worker thread fights "thin", and
  progress was already established as UX-only (not a timeout defense).
  `transcribe` is a sync tool with a stderr-logging progress callback,
  matching every other tool in this repo. Live client progress is a
  clean later add if wanted.
- **`list_transcripts` skipped.** The orchestrating LLM has its own
  filesystem tools; globbing `*.transcript.json` is free for it.
  `read_transcript` earns its place (server-side format + filter =
  token-lean); `list_transcripts` would just reimplement `ls`.
- New tools registered via `__main__.py` importing `scribe_tools`
  alongside the legacy `tools` package (removed in Phase 2). 45 tools
  registered total during the overlap.

**Out of scope for this phase**: deleting anything; the rename.
**Effort**: done in well under the ~1 day estimate (the engine reuse
paid off — `pipeline.py` is ~170 lines, `scribe_tools.py` ~100).

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
