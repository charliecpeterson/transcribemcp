"""Phase-1 thin surface: transcribe + read_transcript against the stub backend.

No ML deps, no DB. Covers output-path resolution, idempotence, the transcript
schema, the three render formats, read-time filtering, and tool validation.
"""
import json

import pytest

from transcribemcp.pipeline import (
    render_transcript,
    run_transcribe,
    transcript_path_for,
)


@pytest.fixture
def fast_backend(monkeypatch):
    """Route the pipeline through a zero-delay stub, regardless of .env."""
    from transcribemcp import config

    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "stub")
    config._settings = None

    from transcribemcp.backends.stub import StubBackend

    stub = StubBackend(delay=0.0)

    def fn(audio_path, *, progress, window=None):
        return stub.transcribe(audio_path, progress=progress, window=window)

    monkeypatch.setattr("transcribemcp.transcribe.get_backend_fn", lambda: fn)
    return fn


def _audio(tmp_path, name="talk.wav"):
    p = tmp_path / name
    p.write_bytes(b"not real audio; the stub never decodes it")
    return p


def test_path_default_is_beside_audio(tmp_path):
    audio = tmp_path / "x.wav"
    assert transcript_path_for(str(audio)) == tmp_path / "x.wav.transcript.json"


def test_path_arg_overrides(tmp_path):
    audio = tmp_path / "x.wav"
    out = tmp_path / "out"
    assert transcript_path_for(str(audio), str(out)) == out / "x.wav.transcript.json"


def test_path_env_setting(tmp_path, monkeypatch):
    from transcribemcp import config

    out = tmp_path / "env_out"
    monkeypatch.setenv("OUTPUT_DIR", str(out))
    config._settings = None
    audio = tmp_path / "x.wav"
    assert transcript_path_for(str(audio)) == out / "x.wav.transcript.json"


def test_schema_shape(tmp_path, fast_backend):
    audio = _audio(tmp_path)
    doc = json.loads(run_transcribe(str(audio), diarize=False).read_text())

    assert doc["schema_version"] == 1
    assert doc["audio_path"] == str(audio.resolve())
    assert doc["model"] == "stub"
    assert doc["diarized"] is False
    assert doc["duration"] == 10.0
    assert len(doc["segments"]) == 3
    assert set(doc["segments"][0]) == {"start", "end", "speaker", "text"}


def test_idempotent_unless_overwrite(tmp_path, fast_backend):
    audio = _audio(tmp_path)
    p1 = run_transcribe(str(audio), diarize=False)
    p1.write_text(json.dumps({"sentinel": True, "segments": []}))

    p2 = run_transcribe(str(audio), diarize=False)
    assert p2 == p1
    assert json.loads(p1.read_text()).get("sentinel") is True  # not recomputed

    p3 = run_transcribe(str(audio), diarize=False, overwrite=True)
    assert "sentinel" not in json.loads(p3.read_text())
    assert len(json.loads(p3.read_text())["segments"]) == 3


def test_render_formats(tmp_path, fast_backend):
    audio = _audio(tmp_path)
    p = run_transcribe(str(audio), diarize=False)

    text = render_transcript(p, format="text")
    assert "[00:00:00] SPEAKER_00: Hello, this is a fake transcript." in text

    srt = render_transcript(p, format="srt")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:03,200\n")

    js = render_transcript(p, format="json")
    assert len(js["segments"]) == 3

    with pytest.raises(ValueError):
        render_transcript(p, format="xml")


def test_render_filters(tmp_path, fast_backend):
    audio = _audio(tmp_path)
    p = run_transcribe(str(audio), diarize=False)

    by_speaker = render_transcript(p, format="json", speaker="SPEAKER_01")
    assert [s["speaker"] for s in by_speaker["segments"]] == ["SPEAKER_01"]

    # Stub segments: (0,3.2), (3.2,7.5), (7.5,10). [8,11] overlaps only the last.
    by_time = render_transcript(p, format="json", time_range=[8.0, 11.0])
    assert [s["start"] for s in by_time["segments"]] == [7.5]


def test_tool_validates_inputs(tmp_path, fast_backend):
    from transcribemcp.scribe_tools import transcribe

    with pytest.raises(ValueError):
        transcribe("relative/path.wav")
    with pytest.raises(ValueError):
        transcribe(str(tmp_path / "missing.wav"))


def test_tool_roundtrip_and_cache_flag(tmp_path, fast_backend):
    from transcribemcp.scribe_tools import read_transcript, transcribe

    audio = _audio(tmp_path)
    r1 = transcribe(str(audio))
    assert r1["cached"] is False
    assert r1["segments"] == 3
    assert r1["model"] == "stub"
    assert r1["diarized"] is False

    r2 = transcribe(str(audio))
    assert r2["cached"] is True
    assert r2["transcript_path"] == r1["transcript_path"]

    assert "Hello" in read_transcript(r1["transcript_path"])
