#!/usr/bin/env bash
# Downloads a short public-domain multi-speaker clip for testing.
#
# Uses the ffmpeg bundled by imageio-ffmpeg (installed via `uv sync --extra dev`)
# so no system-level ffmpeg install is required.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p tests/fixtures

OUT=tests/fixtures/sample.wav
if [[ -f "$OUT" ]]; then
    echo "already present: $OUT"
    exit 0
fi

FFMPEG=$(uv run --extra dev python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || true)
if [[ -z "$FFMPEG" || ! -x "$FFMPEG" ]]; then
    echo "error: could not locate ffmpeg via imageio-ffmpeg."
    echo "       run: uv sync --extra dev"
    echo "       or:  brew install ffmpeg  (and re-run this script)"
    exit 1
fi
echo "using ffmpeg at: $FFMPEG"

# Try a list of known-stable CC0/public-domain speech samples. The first one
# that downloads successfully wins. Single-speaker is fine for validating the
# full ASR + diarization pipeline (diarization will produce SPEAKER_00 only).
URLS=(
    # OpenAI Whisper test fixture — short JFK speech, public domain
    "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
    # pyannote-audio sample — multi-speaker telephone conversation
    "https://github.com/pyannote/pyannote-audio/raw/develop/tutorials/assets/sample.wav"
)

TMP=$(mktemp -t meetingtool_sample.XXXXXX)
trap 'rm -f "$TMP"' EXIT

FETCHED=""
for URL in "${URLS[@]}"; do
    echo "fetching $URL ..."
    if curl -fL --retry 2 --max-time 30 -o "$TMP" "$URL"; then
        FETCHED="$URL"
        break
    fi
    echo "  -> failed, trying next"
done

if [[ -z "$FETCHED" ]]; then
    echo "error: all sample URLs failed. Drop your own audio file at $OUT manually."
    exit 1
fi

echo "converting to 16 kHz mono wav (max 30s) ..."
"$FFMPEG" -y -hide_banner -loglevel error -i "$TMP" -t 30 -ac 1 -ar 16000 "$OUT"

echo "wrote $OUT"
ls -lh "$OUT"
