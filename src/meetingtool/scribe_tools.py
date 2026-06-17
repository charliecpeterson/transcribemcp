"""The thin MCP tool surface: transcribe + read_transcript.

Boundary layer over `pipeline.py`. Tools validate inputs (absolute paths,
existing files) and hand off to the pure pipeline functions. Importing this
module registers the tools on the shared FastMCP instance.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import get_settings
from .pipeline import render_transcript, run_transcribe, transcript_path_for
from .server import mcp

logger = logging.getLogger(__name__)


def _log_progress(stage: str, pct: float) -> None:
    # Server logs go to stderr; stdout is the MCP stdio channel and must stay
    # clean. This is local stage visibility, not a client progress stream.
    logger.info("transcribe: %s %.0f%%", stage, pct * 100)


@mcp.tool()
def transcribe(
    audio_path: str,
    diarize: bool | None = None,
    output_dir: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Transcribe an audio file to a JSON transcript on disk; return its path.

    Synchronous and idempotent: if `<audio>.transcript.json` already exists it
    is returned without recomputing (pass `overwrite=True` to force). Long
    files block for minutes — raise your MCP client's tool-call timeout
    (Claude Code: MCP_TOOL_TIMEOUT; Goose: extension `timeout`) for big audio.

    `diarize` defaults to the DIARIZE setting; pass True/False to override per
    call (True needs HF_TOKEN). `output_dir` overrides where the transcript is
    written (default: beside the audio). Pull transcript text with
    `read_transcript` rather than reading the JSON directly.
    """
    p = Path(audio_path)
    if not p.is_absolute():
        raise ValueError(f"audio_path must be absolute, got {audio_path!r}")
    if not p.is_file():
        raise ValueError(f"no file at {audio_path!r}")

    diar = get_settings().diarize if diarize is None else diarize
    out_path = transcript_path_for(audio_path, output_dir)
    cached = out_path.exists() and not overwrite

    run_transcribe(
        audio_path,
        diarize=diar,
        progress=_log_progress,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    doc = render_transcript(out_path, format="json")
    return {
        "transcript_path": str(out_path),
        "cached": cached,
        "diarized": doc["diarized"],
        "model": doc["model"],
        "duration": doc["duration"],
        "segments": len(doc["segments"]),
    }


@mcp.tool()
def read_transcript(
    path: str,
    format: str = "text",
    time_range: list[float] | None = None,
    speaker: str | None = None,
) -> str | dict:
    """Read a transcript file, rendered and filtered server-side.

    `format` is text (default), srt, or json. `time_range=[start, end]` keeps
    only segments overlapping that window (seconds); `speaker="SPEAKER_01"`
    keeps only that speaker's segments. Filtering here keeps large transcripts
    from flooding the context window — pull the slice you need, not the whole
    file.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no transcript at {path!r}")
    if time_range is not None and len(time_range) != 2:
        raise ValueError("time_range must be [start, end]")
    return render_transcript(
        p, format=format, time_range=time_range, speaker=speaker
    )
