#!/bin/bash
# Builds MsplatCore.xcframework and the per-platform metallibs the Swift
# package ships as resources.
#
# A metallib is compiled against one SDK and will not load on another, so each
# slice is configured and built separately rather than lipo'd together. Device
# and simulator are arm64 only; an Intel Mac running the simulator needs
# x86_64 added to CMAKE_OSX_ARCHITECTURES for that slice.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT_DIR="$(pwd)"

RESOURCE_DIR="$ROOT_DIR/swift/Sources/Msplat/Resources"
HEADERS_DIR="$ROOT_DIR/build/xcf-headers"
XCFRAMEWORK_PATH="$ROOT_DIR/MsplatCore.xcframework"
IOS_DEPLOYMENT_TARGET="${IOS_DEPLOYMENT_TARGET:-18.0}"
MACOS_DEPLOYMENT_TARGET="${MACOS_DEPLOYMENT_TARGET:-15.0}"
JOBS="${BUILD_JOBS:-$(sysctl -n hw.ncpu)}"

if [[ -z "${DEVELOPER_DIR:-}" ]]; then
    if [[ -d "/Applications/Xcode.app" ]]; then
        export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
    elif [[ -d "/Applications/Xcode-beta.app" ]]; then
        export DEVELOPER_DIR="/Applications/Xcode-beta.app/Contents/Developer"
    fi
fi

# slice <build-dir> <metal-sdk> [<sdk-for-sysroot>]
build_slice() {
    local dir="build/$1" metal_sdk="$2" sdk="${3:-}"
    local args=(-B "$dir" -DCMAKE_BUILD_TYPE=Release -DMSPLAT_METAL_SDK="$metal_sdk")

    if [[ -n "$sdk" ]]; then
        args+=(-DCMAKE_SYSTEM_NAME=iOS
               -DCMAKE_OSX_ARCHITECTURES=arm64
               -DCMAKE_OSX_SYSROOT="$(xcrun --sdk "$sdk" --show-sdk-path)"
               -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET"
               -DMSPLAT_METAL_IOS_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET")
    else
        args+=(-DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOS_DEPLOYMENT_TARGET"
               -DMSPLAT_METAL_MACOS_DEPLOYMENT_TARGET="$MACOS_DEPLOYMENT_TARGET")
    fi

    echo "=== Configuring $1 ($metal_sdk) ==="
    cmake "${args[@]}"
    echo "=== Building $1 ==="
    cmake --build "$dir" --config Release --target msplat_core metallib -j "$JOBS"

    [[ -f "$dir/libmsplat_core.a" ]] || { echo "error: $dir/libmsplat_core.a missing" >&2; exit 1; }
    [[ -f "$dir/default.metallib" ]] || { echo "error: $dir/default.metallib missing" >&2; exit 1; }
}

build_slice macos         macosx
build_slice ios-device    iphoneos        iphoneos
build_slice ios-simulator iphonesimulator iphonesimulator

echo "=== Preparing XCFramework headers ==="
rm -rf "$HEADERS_DIR"
mkdir -p "$HEADERS_DIR"
cp core/include/msplat_c_api.h "$HEADERS_DIR/"
cat > "$HEADERS_DIR/module.modulemap" <<'MAP'
module MsplatCore {
    header "msplat_c_api.h"
    export *
}
MAP

echo "=== Creating XCFramework ==="
rm -rf "$XCFRAMEWORK_PATH"
xcodebuild -create-xcframework \
    -library build/macos/libmsplat_core.a         -headers "$HEADERS_DIR" \
    -library build/ios-device/libmsplat_core.a    -headers "$HEADERS_DIR" \
    -library build/ios-simulator/libmsplat_core.a -headers "$HEADERS_DIR" \
    -output "$XCFRAMEWORK_PATH"

echo "=== Copying metallibs ==="
mkdir -p "$RESOURCE_DIR"
cp build/macos/default.metallib         "$RESOURCE_DIR/default-macos.metallib"
cp build/ios-device/default.metallib    "$RESOURCE_DIR/default-ios.metallib"
cp build/ios-simulator/default.metallib "$RESOURCE_DIR/default-iossimulator.metallib"

echo "=== Done ==="
echo "  $XCFRAMEWORK_PATH"
ls -1 "$RESOURCE_DIR" | sed 's/^/  /'
