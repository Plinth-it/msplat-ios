#!/bin/bash
# Verifies that the semantic Swift package references one complete, immutable
# binary release and embeds the exact metallibs published with that release.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPOSITORY="${MSPLAT_RELEASE_REPOSITORY:-Plinth-it/msplat-ios}"
VERSION="$(tr -d '[:space:]' < "$ROOT_DIR/VERSION")"
BINARY_TAG="binary-v$VERSION"
ARCHIVE_NAME="MsplatCore.xcframework.zip"
EXPECTED_URL="https://github.com/$REPOSITORY/releases/download/$BINARY_TAG/$ARCHIVE_NAME"

MANIFEST_URL="$(awk '/url: "/ { line=$0; sub(/^.*url: "/, "", line); sub(/".*$/, "", line); print line; exit }' "$ROOT_DIR/Package.swift")"
MANIFEST_CHECKSUM="$(awk '/checksum: "/ { line=$0; sub(/^.*checksum: "/, "", line); sub(/".*$/, "", line); print line; exit }' "$ROOT_DIR/Package.swift")"

if [[ "$MANIFEST_URL" != "$EXPECTED_URL" ]]; then
    echo "error: Package.swift URL must be $EXPECTED_URL" >&2
    exit 1
fi
if [[ ! "$MANIFEST_CHECKSUM" =~ ^[0-9a-f]{64}$ ]]; then
    echo "error: Package.swift checksum is not a lowercase SHA-256 digest" >&2
    exit 1
fi

if [[ "$(gh api "repos/$REPOSITORY/releases/tags/$BINARY_TAG" --jq .immutable)" != true ]]; then
    echo "error: $BINARY_TAG is not an immutable GitHub release" >&2
    exit 1
fi

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/msplat-release-contract.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT

gh release download "$BINARY_TAG" \
    --repo "$REPOSITORY" \
    --dir "$TEMP_DIR" \
    --pattern "$ARCHIVE_NAME" \
    --pattern "$ARCHIVE_NAME.checksum" \
    --pattern 'SHA256SUMS' \
    --pattern 'provenance.txt' \
    --pattern 'default-*.metallib'

ACTUAL_CHECKSUM="$(swift package compute-checksum "$TEMP_DIR/$ARCHIVE_NAME")"
PUBLISHED_CHECKSUM="$(tr -d '[:space:]' < "$TEMP_DIR/$ARCHIVE_NAME.checksum")"
if [[ "$ACTUAL_CHECKSUM" != "$MANIFEST_CHECKSUM" || "$PUBLISHED_CHECKSUM" != "$MANIFEST_CHECKSUM" ]]; then
    echo "error: Package.swift and published XCFramework checksums differ" >&2
    exit 1
fi

(
    cd "$TEMP_DIR"
    shasum -a 256 -c SHA256SUMS
)

for platform in macos ios iossimulator; do
    cmp \
        "$TEMP_DIR/default-$platform.metallib" \
        "$ROOT_DIR/swift/Sources/Msplat/Resources/default-$platform.metallib"
done

grep -Fqx "version=$VERSION" "$TEMP_DIR/provenance.txt"
grep -Fqx "release_tag=$BINARY_TAG" "$TEMP_DIR/provenance.txt"

PROVENANCE_COMMIT="$(awk -F= '$1 == "source_commit" { print $2 }' "$TEMP_DIR/provenance.txt")"
BINARY_TAG_COMMIT="$(gh api "repos/$REPOSITORY/commits/$BINARY_TAG" --jq .sha)"
if [[ ! "$PROVENANCE_COMMIT" =~ ^[0-9a-f]{40}$ || "$PROVENANCE_COMMIT" != "$BINARY_TAG_COMMIT" ]]; then
    echo "error: binary release provenance does not match its protected tag" >&2
    exit 1
fi

if ! git -C "$ROOT_DIR" cat-file -e "$PROVENANCE_COMMIT^{commit}" 2>/dev/null; then
    echo "error: binary source commit is unavailable; use a full checkout" >&2
    exit 1
fi
if ! git -C "$ROOT_DIR" merge-base --is-ancestor "$PROVENANCE_COMMIT" HEAD; then
    echo "error: semantic tag does not descend from the binary source commit" >&2
    exit 1
fi
if ! git -C "$ROOT_DIR" diff --quiet "$PROVENANCE_COMMIT" HEAD -- \
    CMakeLists.txt \
    core \
    swift/Sources/Msplat \
    ':(exclude)swift/Sources/Msplat/Resources'; then
    echo "error: semantic tag changes native or Swift wrapper sources after $BINARY_TAG" >&2
    exit 1
fi

echo "Swift release contract verified: $BINARY_TAG"
