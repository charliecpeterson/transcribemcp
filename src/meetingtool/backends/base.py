from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None  # SPEAKER_00 style; None if diarization disabled/failed


@dataclass
class TranscriptionResult:
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    duration: float = 0.0
    backend_name: str = ""


ProgressCB = Callable[[str, float], None]


class TranscriptionBackend(Protocol):
    name: str

    def transcribe(
        self,
        audio_path: str,
        *,
        progress: ProgressCB,
        window: tuple[float, float] | None = None,
    ) -> TranscriptionResult: ...
    # Returns segments with ABSOLUTE timestamps (relative to the full audio
    # file, not the window). Diarization is orchestrated by jobs.py and
    # applied to persisted chunks after all windows finish — backends are
    # pure ASR. window=None means "transcribe the whole file".
