"""Synchronous transcription pipeline: audio in, transcript JSON on disk.

The thin core of the MCP. `run_transcribe` resolves an output path, returns it
untouched if a transcript already exists (idempotent), otherwise runs the
configured backend over the whole file, optionally diarizes, and writes the
result. No DB, no windowing, no job queue — the engine modules (backends,
diarize) do the work. Rendering helpers turn a written transcript back into
text / SRT / filtered JSON for `read_transcript`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import get_settings

SCHEMA_VERSION = 1

# (stage_name, fraction_done) — fraction is approximate and for display only.
ProgressCB = Callable[[str, float], None]


def transcript_path_for(
    audio_path: str | Path, output_dir: str | Path | None = None
) -> Path:
    """Resolve where a transcript for `audio_path` lives.

    `output_dir` arg wins; else the OUTPUT_DIR setting; else beside the audio.
    The basename keeps the audio's extension so `talk.wav` and `talk.mp3` in
    one OUTPUT_DIR don't collide.
    """
    audio = Path(audio_path)
    if output_dir is not None:
        out_dir = Path(output_dir)
    elif get_settings().output_dir is not None:
        out_dir = get_settings().output_dir
    else:
        out_dir = audio.parent
    return out_dir / f"{audio.name}.transcript.json"


def run_transcribe(
    audio_path: str,
    *,
    diarize: bool,
    progress: ProgressCB | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Transcribe `audio_path` to a JSON file and return its path.

    Idempotent: if the target exists and `overwrite` is False, returns it
    without recomputing. When `diarize` is True, pyannote runs once over the
    full file and its speaker labels replace the backend's; otherwise the
    backend's own speaker labels (if any) are kept verbatim.
    """
    report = progress or (lambda stage, pct: None)
    out_path = transcript_path_for(audio_path, output_dir)
    if out_path.exists() and not overwrite:
        report("cached", 1.0)
        return out_path

    from .transcribe import get_backend_fn  # lazy: avoids ML import at startup

    report("loading_model", 0.05)
    backend_fn = get_backend_fn()
    result = backend_fn(audio_path, progress=report)  # window=None → whole file

    segments = result.segments
    if diarize:
        report("diarize", 0.85)
        from .diarize import assign_speakers, diarize as run_diarize

        turns = run_diarize(audio_path)
        segments = assign_speakers(segments, turns)

    report("writing", 0.95)
    doc = _build_doc(audio_path, result, segments, diarized=diarize)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    report("done", 1.0)
    return out_path


def _build_doc(audio_path: str, result, segments, *, diarized: bool) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "audio_path": str(Path(audio_path).resolve()),
        "backend": result.backend_name,
        "model": get_settings().active_model,
        "language": result.language,
        "diarized": diarized,
        "duration": result.duration,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "segments": [
            {"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text}
            for s in segments
        ],
    }


def render_transcript(
    path: str | Path,
    *,
    format: str = "text",
    time_range: tuple[float, float] | list[float] | None = None,
    speaker: str | None = None,
) -> str | dict:
    """Read a transcript file and render it filtered to the requested shape."""
    doc = json.loads(Path(path).read_text())
    segs = _filter_segments(doc["segments"], time_range, speaker)
    if format == "json":
        return {**doc, "segments": segs}
    if format == "srt":
        return _to_srt(segs)
    if format == "text":
        return _to_text(segs)
    raise ValueError(f"unknown format: {format!r} (use text|json|srt)")


def _filter_segments(segments: list[dict], time_range, speaker) -> list[dict]:
    out = segments
    if time_range is not None:
        lo, hi = time_range
        out = [s for s in out if s["end"] > lo and s["start"] < hi]
    if speaker is not None:
        out = [s for s in out if s.get("speaker") == speaker]
    return out


def _fmt_ts(seconds: float, *, srt: bool = False) -> str:
    seconds = max(seconds, 0.0)
    whole = int(seconds)
    ms = int((seconds - whole) * 1000)
    h, rem = divmod(whole, 3600)
    m, sec = divmod(rem, 60)
    if srt:
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _to_text(segments: list[dict]) -> str:
    return "\n".join(
        f"[{_fmt_ts(s['start'])}] {s.get('speaker') or '?'}: {s['text']}"
        for s in segments
    )


def _to_srt(segments: list[dict]) -> str:
    blocks = []
    for i, s in enumerate(segments, 1):
        spk = s.get("speaker")
        body = f"{spk}: {s['text']}" if spk else s["text"]
        stamp = f"{_fmt_ts(s['start'], srt=True)} --> {_fmt_ts(s['end'], srt=True)}"
        blocks.append(f"{i}\n{stamp}\n{body}")
    return "\n\n".join(blocks) + "\n"
