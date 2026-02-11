#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <huggingface-url>"
    exit 1
fi

URL="$1"
MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "Downloading model from $URL..."
curl -L -O "$URL"
echo "Download complete!"
