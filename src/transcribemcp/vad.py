"""Voice-activity detection via silero-vad.

Produces `[VoicedSpan(start, end)]` covering speech regions in an audio file,
with configurable minimum silence / speech thresholds. Also groups those
spans into ~N-minute `Window`s that jobs.py uses to chunk long recordings
for incremental transcription + resume.

This exists because several LLM-decoded ASR backends (Cohere Transcribe,
Granite Speech, Phi-4 multimodal, Canary-Qwen) emit bare text with no
segment boundaries. Our backend Protocol wants `Segment(start, end, text,
speaker)` so diarization can merge by timestamp. Pattern:

    1. vad.detect_voiced(path) → voiced spans
    2. for each span, slice audio, run ASR on the slice
    3. produce a Segment using the span's (start, end) as timestamps

The model is loaded lazily and cached after first use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .audio import DEFAULT_SR, decode_pcm

if TYPE_CHECKING:
    import torch  # pragma: no cover


@dataclass
class VoicedSpan:
    start: float
    end: float


@dataclass
class Window:
    """A contiguous transcription window anchored to silence boundaries.

    Windows don't overlap. A window spans from the start of its first
    voiced span to the end of its last voiced span, so any audio between
    windows is guaranteed silent — breaks happen in the gaps between
    detected speech, not mid-utterance.
    """
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    from silero_vad import load_silero_vad  # heavy import, deferred
    _model = load_silero_vad()
    return _model


def detect_voiced(
    audio_path: str,
    min_silence_s: float = 0.3,
    min_speech_s: float = 0.25,
    max_speech_s: float | None = 30.0,
) -> list[VoicedSpan]:
    """Detect voiced regions using silero-vad.

    - `min_silence_s`: a silence shorter than this doesn't split voiced spans
    - `min_speech_s`: voiced regions shorter than this are dropped
    - `max_speech_s`: longest voiced span; silero splits on internal pauses if
      a span would exceed this. Defaults to 30 s so chunks fit LLM-decoded
      ASR models (Cohere auto-chunks internally but shorter chunks help
      accuracy on Granite/Canary-Qwen which cap out around 40 s).
    """
    from silero_vad import get_speech_timestamps

    wav = decode_pcm(audio_path, sample_rate=DEFAULT_SR)
    model = _get_model()
    kwargs = dict(
        sampling_rate=DEFAULT_SR,
        return_seconds=True,
        min_silence_duration_ms=int(min_silence_s * 1000),
        min_speech_duration_ms=int(min_speech_s * 1000),
    )
    if max_speech_s is not None:
        kwargs["max_speech_duration_s"] = max_speech_s
    spans = get_speech_timestamps(wav, model, **kwargs)
    return [VoicedSpan(start=float(s["start"]), end=float(s["end"])) for s in spans]


def group_into_windows(
    spans: list[VoicedSpan],
    target_seconds: float = 300.0,
) -> list[Window]:
    """Pack voiced spans into ~`target_seconds` windows, breaking on silence.

    Greedy: start a new window when the current one's cumulative duration
    (first span.start → last span.end) meets or exceeds `target_seconds`.
    Breaks always land in silence (between consecutive spans), never
    mid-utterance. Returns [] if `spans` is empty.

    The default 300s (5 min) balances ASR context (WhisperX handles long
    windows fine) against crash-resume granularity (a 5-min redo is
    acceptable; a 3-hour redo is not).
    """
    if not spans:
        return []

    windows: list[Window] = []
    win_start = spans[0].start
    last_end = spans[0].end
    for span in spans[1:]:
        # Current window spans win_start .. last_end. If we'd exceed target
        # by including `span`, close the window at last_end and start a new
        # one at `span.start` (the silence gap is discarded — backends will
        # have already skipped it via their own VAD or return nothing for
        # silent regions).
        if (span.end - win_start) > target_seconds:
            windows.append(Window(start=win_start, end=last_end))
            win_start = span.start
        last_end = span.end
    windows.append(Window(start=win_start, end=last_end))
    return windows
