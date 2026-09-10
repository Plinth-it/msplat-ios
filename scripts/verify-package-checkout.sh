#!/bin/bash
# Build the distributed Swift package from a fresh checkout with LFS unavailable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REVISION="$(git -C "$ROOT_DIR" rev-parse "${1:-HEAD}^{commit}")"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/msplat-package-checkout.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT
CHECKOUT="$TEMP_DIR/checkout"

# Do not let developer LFS settings or cached objects hide a checkout regression.
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1
unset GIT_LFS_SKIP_SMUDGE GIT_LFS_SKIP_DOWNLOAD_ERRORS
git clone --no-local --no-checkout "$ROOT_DIR" "$CHECKOUT"
git -C "$CHECKOUT" lfs install --local --force
git -C "$CHECKOUT" config lfs.storage "$CHECKOUT/.git/lfs"
git -C "$CHECKOUT" config lfs.url http://127.0.0.1:1/unavailable-lfs
git -C "$CHECKOUT" config lfs.skipdownloaderrors false
git -C "$CHECKOUT" checkout --detach "$REVISION"

git -C "$CHECKOUT" lfs ls-files --name-only > "$TEMP_DIR/lfs-files"
if [[ ! -s "$TEMP_DIR/lfs-files" ]]; then
    echo "error: checkout test must exercise tracked LFS files" >&2
    exit 1
fi
while IFS= read -r path; do
    case "$path" in
        datasets/*) ;;
        *) echo "error: unexpected package LFS dependency: $path" >&2; exit 1 ;;
    esac
    git -C "$CHECKOUT" lfs pointer --check --strict --file="$CHECKOUT/$path"
done < "$TEMP_DIR/lfs-files"

if [[ -d "$CHECKOUT/.git/lfs/objects" ]] &&
    [[ -n "$(find "$CHECKOUT/.git/lfs/objects" -type f -print -quit)" ]]; then
    echo "error: package checkout downloaded LFS objects" >&2
    exit 1
fi

swift build --package-path "$CHECKOUT" \
    --scratch-path "$TEMP_DIR/build" \
    --cache-path "$TEMP_DIR/cache"

echo "Fresh package build passed with $(wc -l < "$TEMP_DIR/lfs-files" | tr -d ' ') dataset pointers and no LFS downloads."
