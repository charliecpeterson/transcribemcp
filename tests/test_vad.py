"""Tests for silero-vad wrapper. Skipped if the [vad] extra isn't installed."""
from pathlib import Path

import pytest

pytest.importorskip(
    "silero_vad",
    reason="silero-vad not installed (uv sync --extra vad)",
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.wav"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="sample.wav fixture missing (run scripts/fetch_sample_audio.sh)",
)


def test_detect_voiced_returns_spans():
    from meetingtool.vad import VoicedSpan, detect_voiced

    spans = detect_voiced(str(FIXTURE))
    assert isinstance(spans, list)
    assert len(spans) >= 1
    for s in spans:
        assert isinstance(s, VoicedSpan)
        assert s.end > s.start
        assert s.start >= 0.0


def test_detect_voiced_spans_are_monotonic():
    from meetingtool.vad import detect_voiced

    spans = detect_voiced(str(FIXTURE))
    for a, b in zip(spans, spans[1:]):
        assert b.start >= a.end  # silero never returns overlapping spans


def test_max_speech_cap_is_respected():
    from meetingtool.vad import detect_voiced

    spans = detect_voiced(str(FIXTURE), max_speech_s=5.0)
    # silero uses max_speech_duration_s as a soft cap — it splits on interior
    # pauses when it can. Allow a small slack for a tail silence that silero
    # sometimes appends inside the final span.
    for s in spans:
        assert (s.end - s.start) <= 6.0
