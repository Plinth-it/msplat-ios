#!/bin/bash
# Packages an already-built XCFramework and its matching platform metallibs.
# The output archive is created once; its SwiftPM checksum and provenance are
# derived from those exact bytes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${MSPLAT_RELEASE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUTPUT_DIR="${1:-$ROOT_DIR/build/release-swift}"

case "$OUTPUT_DIR" in
    /*) ;;
    *) OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR" ;;
esac

cd "$ROOT_DIR"

VERSION="$(tr -d '[:space:]' < VERSION)"
SOURCE_COMMIT="$(git rev-parse HEAD)"
RELEASE_TAG="${RELEASE_TAG:-v$VERSION}"
XCFRAMEWORK_PATH="$ROOT_DIR/MsplatCore.xcframework"
RESOURCE_DIR="${MSPLAT_RELEASE_RESOURCE_DIR:-$ROOT_DIR/swift/Sources/Msplat/Resources}"
ARCHIVE_NAME="MsplatCore.xcframework.zip"

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "error: output path already exists: $OUTPUT_DIR" >&2
    exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "error: tracked working tree changes make release provenance ambiguous" >&2
    git status --short --untracked-files=no >&2
    exit 1
fi

required_files=(
    "$XCFRAMEWORK_PATH/Info.plist"
    "$XCFRAMEWORK_PATH/macos-arm64/libmsplat_core.a"
    "$XCFRAMEWORK_PATH/ios-arm64/libmsplat_core.a"
    "$XCFRAMEWORK_PATH/ios-arm64_x86_64-simulator/libmsplat_core.a"
    "$RESOURCE_DIR/default-macos.metallib"
    "$RESOURCE_DIR/default-ios.metallib"
    "$RESOURCE_DIR/default-iossimulator.metallib"
)

for path in "${required_files[@]}"; do
    if [[ ! -s "$path" ]]; then
        echo "error: required release input is missing or empty: $path" >&2
        exit 1
    fi
done

cmp -s \
    "$XCFRAMEWORK_PATH/macos-arm64/Headers/msplat_c_api.h" \
    "$XCFRAMEWORK_PATH/ios-arm64/Headers/msplat_c_api.h"
cmp -s \
    "$XCFRAMEWORK_PATH/macos-arm64/Headers/msplat_c_api.h" \
    "$XCFRAMEWORK_PATH/ios-arm64_x86_64-simulator/Headers/msplat_c_api.h"

[[ "$(lipo -archs "$XCFRAMEWORK_PATH/macos-arm64/libmsplat_core.a")" == "arm64" ]]
[[ "$(lipo -archs "$XCFRAMEWORK_PATH/ios-arm64/libmsplat_core.a")" == "arm64" ]]
SIMULATOR_ARCHS="$(lipo -archs "$XCFRAMEWORK_PATH/ios-arm64_x86_64-simulator/libmsplat_core.a")"
[[ "$SIMULATOR_ARCHS" == *"arm64"* && "$SIMULATOR_ARCHS" == *"x86_64"* ]]

mkdir -p "$OUTPUT_DIR"
ditto -c -k --norsrc --noextattr --noqtn --noacl --keepParent \
    "$XCFRAMEWORK_PATH" "$OUTPUT_DIR/$ARCHIVE_NAME"

if ! unzip -Z1 "$OUTPUT_DIR/$ARCHIVE_NAME" | grep -qx \
    'MsplatCore.xcframework/Info.plist'; then
    echo "error: archive does not contain MsplatCore.xcframework at its root" >&2
    exit 1
fi

CHECKSUM="$(swift package compute-checksum "$OUTPUT_DIR/$ARCHIVE_NAME")"
printf '%s\n' "$CHECKSUM" > "$OUTPUT_DIR/$ARCHIVE_NAME.checksum"

cp "$RESOURCE_DIR/default-macos.metallib" "$OUTPUT_DIR/"
cp "$RESOURCE_DIR/default-ios.metallib" "$OUTPUT_DIR/"
cp "$RESOURCE_DIR/default-iossimulator.metallib" "$OUTPUT_DIR/"

XCODE_VERSION="$(xcodebuild -version | tr '\n' ';' | sed 's/;$//')"
{
    printf 'format=1\n'
    printf 'version=%s\n' "$VERSION"
    printf 'release_tag=%s\n' "$RELEASE_TAG"
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'xcode=%s\n' "$XCODE_VERSION"
    printf 'macos_sdk=%s\n' "$(xcrun --sdk macosx --show-sdk-version)"
    printf 'iphoneos_sdk=%s\n' "$(xcrun --sdk iphoneos --show-sdk-version)"
    printf 'iphonesimulator_sdk=%s\n' "$(xcrun --sdk iphonesimulator --show-sdk-version)"
    printf 'macos_deployment_target=%s\n' "${MACOS_DEPLOYMENT_TARGET:-15.0}"
    printf 'ios_deployment_target=%s\n' "${IOS_DEPLOYMENT_TARGET:-18.0}"
    printf 'metal_language_standard=%s\n' "${MSPLAT_METAL_LANGUAGE_STANDARD:-auto}"
} > "$OUTPUT_DIR/provenance.txt"

(
    cd "$OUTPUT_DIR"
    shasum -a 256 \
        "$ARCHIVE_NAME" \
        default-macos.metallib \
        default-ios.metallib \
        default-iossimulator.metallib \
        provenance.txt > SHA256SUMS
)

echo "Release payload: $OUTPUT_DIR"
echo "SwiftPM checksum: $CHECKSUM"
cat "$OUTPUT_DIR/SHA256SUMS"
