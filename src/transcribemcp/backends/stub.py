"""Stub backend: no ML deps, writes a fake transcript.

Used for development, tests, and smoke-checking the async plumbing before
committing to any heavyweight ASR install.
"""
import time

from .base import ProgressCB, Segment, TranscriptionResult


# Three hardcoded segments covering the first 10 seconds of "audio". Each
# test relies on their shape; keep them stable.
_FAKE_SEGMENTS = [
    Segment(0.0, 3.2, "Hello, this is a fake transcript.", "SPEAKER_00"),
    Segment(3.2, 7.5, "The stub backend produced these segments for testing.", "SPEAKER_01"),
    Segment(7.5, 10.0, "Real backends will replace this output.", "SPEAKER_00"),
]


class StubBackend:
    name = "stub"

    def __init__(self, delay: float = 0.5) -> None:
        self._delay = delay

    def transcribe(
        self,
        audio_path: str,
        *,
        progress: ProgressCB,
        window: tuple[float, float] | None = None,
    ) -> TranscriptionResult:
        # Two "stages" let jobs.py observe progress ticks and gives cancel a
        # boundary to fire at. Real backends have more stages; the stub is
        # deliberately thin.
        progress("loading_model", 0.1)
        time.sleep(self._delay)
        progress("asr", 0.8)
        time.sleep(self._delay)

        if window is None:
            segments = list(_FAKE_SEGMENTS)
            duration = 10.0
        else:
            ws, we = window
            # Each segment "belongs" to the window whose range contains its
            # start — matches how faster-whisper's clip_timestamps assigns
            # segments to clips. Keeps dedupe simple.
            segments = [s for s in _FAKE_SEGMENTS if ws <= s.start < we]
            duration = we - ws

        return TranscriptionResult(
            segments=segments,
            language="en",
            duration=duration,
            backend_name=self.name,
        )
