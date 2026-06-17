# transcribemcp

A thin, local MCP server that transcribes audio and writes the result to a
JSON file on disk. Audio in, transcript file out — nothing more.

The agent driving the MCP (Claude Code, Goose, a local vLLM/Ollama loop) does
everything else: reading transcripts, chaining several together, identifying
speakers across files, summarizing. There's no database and no "meeting"
model. The filesystem holds the transcripts; the agent holds the context.

Runs fully locally. WhisperX/Qwen3/Cohere for ASR, pyannote for diarization,
on CPU, CUDA, or Apple Silicon (MLX).

## Install

Uses [`uv`](https://docs.astral.sh/uv/). The server runs with no ML deps
against a stub backend:

```bash
uv sync
```

For real transcription, add the extras for the backend you want:

```bash
uv sync --extra whisperx --extra diarize      # WhisperX + pyannote
uv sync --extra qwen3 --extra diarize         # Qwen3-ASR + pyannote
uv sync --extra cohere --extra vad            # Cohere Transcribe + silero-vad
# Apple Silicon GPU: add --extra qwen3-mlx or --extra cohere-mlx
```

WhisperX uses CTranslate2, which has no Metal backend — on Apple Silicon it
runs **CPU-only**. For GPU speed on a Mac, use a `*-mlx` backend. On a CUDA
box, WhisperX is the fast path.

ffmpeg is bundled (via `imageio-ffmpeg`); no system ffmpeg needed.

## Configure

Copy `.env.example` to `.env` and set what you need:

```bash
TRANSCRIPTION_BACKEND=whisperx     # whisperx | qwen3_asr | cohere | stub
WHISPER_MODEL=base                 # tiny | base | small | medium | large-v3
WHISPERX_DEVICE=auto               # auto | cpu | cuda | mps
DIARIZE=true                       # speaker labels via pyannote
HF_TOKEN=hf_...                    # required if DIARIZE=true
OUTPUT_DIR=                        # blank = write beside the source audio
```

Diarization needs a HuggingFace token with the
`pyannote/speaker-diarization-community-1` license accepted.

## Register with a client

Point your MCP client at the launch wrapper (one path, any client):

```bash
# Claude Code
claude mcp add transcribemcp ~/mcps/bin/transcribemcp-run

# Goose: add an stdio extension with command ~/mcps/bin/transcribemcp-run
```

## Tools

| Tool | What it does |
|------|--------------|
| `ping` | Reachability check. |
| `transcribe(audio_path, diarize=None, output_dir=None, overwrite=False)` | Transcribe to `<audio>.transcript.json`. Returns the path. Idempotent: returns the existing file unless `overwrite=True`. |
| `read_transcript(path, format="text", time_range=None, speaker=None)` | Render a transcript as `text`, `srt`, or `json`, optionally filtered to a time window or one speaker. |

`transcribe` returns metadata, not the transcript:
`{transcript_path, cached, diarized, model, duration, segments}`. Pull the
text with `read_transcript` so you fetch only the slice you need — filtering
by `time_range=[start, end]` or `speaker="SPEAKER_01"` keeps long transcripts
from flooding the context window.

To work across recordings, transcribe each and read them both: deciding that
`SPEAKER_01` in one file is the same person as `SPEAKER_03` in another is the
agent's reasoning, not something the server tracks.

## Long files and client timeouts

`transcribe` is synchronous — an hour of audio blocks the call for minutes.
MCP clients impose a tool-call timeout and **do not** extend it on progress
notifications (true for both Claude Code and Goose). For long files, raise the
client's timeout:

- **Claude Code:** set `MCP_TOOL_TIMEOUT` (milliseconds) in the environment.
- **Goose:** set the extension's `timeout` (seconds; default 300) in
  `config.yaml`.

If a call times out anyway, just call `transcribe` again — it returns the
finished file from disk without recomputing.

## Transcript format

```json
{
  "schema_version": 1,
  "audio_path": "/abs/path/talk.wav",
  "backend": "whisperx", "model": "base", "language": "en",
  "diarized": true, "duration": 1234.5,
  "created_at": "2026-06-16T21:00:00-07:00",
  "segments": [
    {"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00", "text": "Hello."}
  ]
}
```

## Development

```bash
uv run --extra dev pytest                     # fast suite, stub backend, no ML
TRANSCRIBEMCP_E2E=1 uv run --extra whisperx --extra dev pytest tests/test_e2e.py
```

The e2e test needs `tests/fixtures/sample.wav` — fetch it with
`scripts/fetch_sample_audio.sh`.

See `PROJECT_PLAN.md` for the design and the history of how this got thin.
