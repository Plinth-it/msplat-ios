#!/bin/bash
# Fetch benchmark data separately; SwiftPM consumers never execute this helper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DESTINATION="$ROOT_DIR/datasets/mipnerf360/garden"
DATASET_REPOSITORY="https://github.com/Plinth-it/msplat-ios"
# Immutable 2.1.1 source revision containing the original 188 dataset pointers.
DATASET_REVISION="b36f2aa3335293eda590132677cd0b55a6b5b4a6"
if [[ $# != 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi

dataset_ready() {
    python3 - "$1" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
images = [root / 'images' / f'DSC{number:05}.JPG' for number in range(7956, 8141)]
files = images + [root / 'sparse/0' / name for name in ('cameras.bin', 'images.bin', 'points3D.bin')]
for file in files:
    if not file.is_file() or file.stat().st_size == 0:
        sys.exit(1)
    with file.open('rb') as stream:
        if stream.read(43).startswith(b'version https://git-lfs.github.com/spec/v1'):
            sys.exit(1)
PY
}

if [[ -e "$DESTINATION" ]]; then
    if dataset_ready "$DESTINATION"; then
        echo "Garden benchmark data already available: $DESTINATION"
        exit 0
    fi
    echo "error: $DESTINATION exists but is incomplete. Move it aside before retrying." >&2
    exit 1
fi

git lfs version >/dev/null
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/msplat-datasets.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT
CHECKOUT="$TEMP_DIR/source"
unset GIT_LFS_SKIP_SMUDGE GIT_LFS_SKIP_DOWNLOAD_ERRORS
git init --quiet "$CHECKOUT"
git -C "$CHECKOUT" remote add origin "$DATASET_REPOSITORY"
GIT_LFS_SKIP_SMUDGE=1 git -C "$CHECKOUT" fetch --depth=1 origin "$DATASET_REVISION"
GIT_LFS_SKIP_SMUDGE=1 git -C "$CHECKOUT" checkout --detach "$DATASET_REVISION"
git -C "$CHECKOUT" lfs install --local
git -C "$CHECKOUT" config lfs.skipdownloaderrors false
git -C "$CHECKOUT" lfs pull --include='datasets/mipnerf360/garden/**' --exclude=''
SOURCE="$CHECKOUT/datasets/mipnerf360/garden"
if ! dataset_ready "$SOURCE"; then
    echo "error: Download did not produce the complete Garden benchmark dataset." >&2
    exit 1
fi
mkdir -p "$(dirname "$DESTINATION")"
mv "$SOURCE" "$DESTINATION"
echo "Downloaded Garden benchmark data to ignored directory: $DESTINATION"
