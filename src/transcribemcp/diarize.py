"""pyannote.audio wrapper — produces speaker turns and merges them into ASR segments.

Shared by all ASR backends. The model is loaded lazily and cached on the module
after first use (pyannote models take several seconds to load; we don't want
to pay that on every job).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .audio import load_waveform
from .backends.base import Segment
from .config import get_settings

if TYPE_CHECKING:
    from pyannote.audio import Pipeline  # pragma: no cover


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


_pipeline: "Pipeline | None" = None


def _get_pipeline() -> "Pipeline":
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    settings = get_settings()
    if not settings.hf_token:
        raise RuntimeError(
            "DIARIZE=true but HF_TOKEN is not set. pyannote requires a HuggingFace "
            "token with the pyannote/speaker-diarization-community-1 license accepted."
        )

    from pyannote.audio import Pipeline  # heavy import, deferred

    _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=settings.hf_token,
    )

    # Move to the best available device.
    try:
        import torch
        if torch.cuda.is_available():
            _pipeline = _pipeline.to(torch.device("cuda"))
        elif torch.backends.mps.is_available():
            _pipeline = _pipeline.to(torch.device("mps"))
    except Exception:
        # If torch import or device selection fails, fall back to CPU silently.
        pass

    return _pipeline


def diarize(audio_path: str) -> list[SpeakerTurn]:
    """Return a list of (start, end, speaker_label) tuples covering the audio."""
    pipeline = _get_pipeline()
    output = pipeline(load_waveform(audio_path))
    # pyannote 4.x returns a DiarizeOutput with `.speaker_diarization` (the
    # Annotation we want) and `.exclusive_speaker_diarization` (non-overlapping
    # variant). We use the full one so overlap can be resolved in assign_speakers.
    turns: list[SpeakerTurn] = []
    for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
        turns.append(SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker))
    turns.sort(key=lambda t: t.start)
    return turns


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    """Merge diarization turns into ASR segments by max-overlap.

    Each ASR segment gets the speaker label of whichever turn overlaps it the
    most. Segments with no overlapping turn keep speaker=None.
    """
    if not turns:
        return segments

    out: list[Segment] = []
    for seg in segments:
        best_label: str | None = None
        best_overlap = 0.0
        for turn in turns:
            # Skip turns that can't possibly overlap (they're sorted).
            if turn.start >= seg.end:
                break
            if turn.end <= seg.start:
                continue
            overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = turn.speaker
        out.append(Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker=best_label,
        ))
    return out
