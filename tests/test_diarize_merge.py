"""Pure-logic tests for assign_speakers. Doesn't load pyannote."""
from transcribemcp.backends.base import Segment
from transcribemcp.diarize import SpeakerTurn, assign_speakers


def test_empty_turns_leaves_speakers_none():
    segs = [Segment(0.0, 2.0, "hi"), Segment(2.0, 4.0, "there")]
    out = assign_speakers(segs, [])
    assert all(s.speaker is None for s in out)
    assert [s.text for s in out] == ["hi", "there"]


def test_max_overlap_wins():
    segs = [Segment(0.0, 5.0, "mostly A but a bit of B")]
    turns = [
        SpeakerTurn(0.0, 4.0, "SPEAKER_00"),  # 4s overlap
        SpeakerTurn(4.0, 5.0, "SPEAKER_01"),  # 1s overlap
    ]
    out = assign_speakers(segs, turns)
    assert out[0].speaker == "SPEAKER_00"


def test_no_overlap_stays_none():
    segs = [Segment(10.0, 12.0, "silence filler")]
    turns = [SpeakerTurn(0.0, 5.0, "SPEAKER_00")]
    out = assign_speakers(segs, turns)
    assert out[0].speaker is None


def test_sorted_turns_early_break_still_correct():
    segs = [
        Segment(0.0, 1.0, "a"),
        Segment(1.0, 2.0, "b"),
        Segment(2.0, 3.0, "c"),
    ]
    turns = [
        SpeakerTurn(0.0, 1.0, "S0"),
        SpeakerTurn(1.0, 2.0, "S1"),
        SpeakerTurn(2.0, 3.0, "S0"),
    ]
    out = assign_speakers(segs, turns)
    assert [s.speaker for s in out] == ["S0", "S1", "S0"]
