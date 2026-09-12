#!/bin/bash
# Resolve and build a fresh SwiftPM consumer with Git LFS deliberately unusable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REVISION="$(git -C "$ROOT_DIR" rev-parse "${1:-HEAD}^{commit}")"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/msplat-package-checkout.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT
XCODE_GIT="$(xcrun --find git)"
XCODE_SWIFT="$(xcrun --find swift)"
mkdir -p "$TEMP_DIR/no-lfs" "$TEMP_DIR/consumer/Sources/Consumer"

cat > "$TEMP_DIR/no-lfs/git-lfs" <<'SH'
#!/bin/sh
echo called >> "$MSPLAT_LFS_CALL_LOG"
echo 'error: Swift package consumers must not require Git LFS' >&2
exit 127
SH
chmod +x "$TEMP_DIR/no-lfs/git-lfs"
export MSPLAT_LFS_CALL_LOG="$TEMP_DIR/lfs-calls"
export PATH="$TEMP_DIR/no-lfs:$(dirname "$XCODE_SWIFT"):$(dirname "$XCODE_GIT"):/usr/bin:/bin:/usr/sbin:/sbin"
export GIT_CONFIG_GLOBAL="$TEMP_DIR/gitconfig"
export GIT_CONFIG_NOSYSTEM=1
unset GIT_LFS_SKIP_SMUDGE GIT_LFS_SKIP_DOWNLOAD_ERRORS GIT_CONFIG_COUNT GIT_CONFIG_PARAMETERS
# Also catch smudging on machines with global LFS filters already installed.
for filter in clean smudge filter-process; do
    key="$filter"
    [[ "$filter" != filter-process ]] || key=process
    git config --file "$GIT_CONFIG_GLOBAL" "filter.lfs.$key" "\"$TEMP_DIR/no-lfs/git-lfs\" $filter"
done
git config --file "$GIT_CONFIG_GLOBAL" filter.lfs.required true
if git lfs version 2>/dev/null || [[ ! -s "$MSPLAT_LFS_CALL_LOG" ]]; then
    echo 'error: Test failed to intercept Git LFS discovery.' >&2
    exit 1
fi
: > "$MSPLAT_LFS_CALL_LOG"

# A consumer exercises SwiftPM's own LFS detection and fresh source checkout;
# building the package directly would skip these dependency-resolution paths.
python3 - "$TEMP_DIR/consumer/Package.swift" "$ROOT_DIR" "$REVISION" <<'PY'
import json
import sys
from pathlib import Path

manifest, source, revision = sys.argv[1:]
Path(manifest).write_text('''// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "Consumer",
    platforms: [.macOS(.v15)],
    products: [.library(name: "Consumer", targets: ["Consumer"])],
    dependencies: [.package(url: %s, revision: %s)],
    targets: [.target(name: "Consumer", dependencies: [.product(name: "Msplat", package: %s)])]
)
''' % (json.dumps(source), json.dumps(revision), json.dumps(Path(source).name.lower())))
PY
printf 'import Msplat\n' > "$TEMP_DIR/consumer/Sources/Consumer/Consumer.swift"
swift build --package-path "$TEMP_DIR/consumer" \
    --scratch-path "$TEMP_DIR/build" \
    --cache-path "$TEMP_DIR/cache"
(
    cd "$TEMP_DIR/consumer"
    xcodebuild -resolvePackageDependencies -scheme Consumer \
        -clonedSourcePackagesDirPath "$TEMP_DIR/xcode-packages" \
        -packageCachePath "$TEMP_DIR/xcode-cache" \
        -derivedDataPath "$TEMP_DIR/xcode-build"
)

CHECKOUT="$TEMP_DIR/build/checkouts/$(basename "$ROOT_DIR")"
python3 - "$CHECKOUT" "$TEMP_DIR/build" <<'PY'
import subprocess
import sys
from pathlib import Path

root, build = map(Path, sys.argv[1:])
pointer_header = b'version https://git-lfs.github.com/spec/v1\n'
paths = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).split(b'\0')
for path in filter(None, paths):
    file = root / path.decode()
    if path.startswith(b'datasets/'):
        raise SystemExit(f'error: Benchmark dataset tracked in package: {path!r}')
    if file.is_file():
        with file.open('rb') as stream:
            if stream.read(len(pointer_header)) == pointer_header:
                raise SystemExit(f'error: LFS pointer tracked in package: {path!r}')
if any((root / '.git/lfs/objects').rglob('*')):
    raise SystemExit('error: Package checkout created LFS objects.')
expected = {'default-ios.metallib', 'default-iossimulator.metallib', 'default-macos.metallib'}
bundles = list(build.glob('**/Msplat_Msplat.bundle'))
if not bundles:
    raise SystemExit('error: Built Msplat resource bundle not found.')
for bundle in bundles:
    files = {str(file.relative_to(bundle)) for file in bundle.rglob('*') if file.is_file()}
    # macOS bundles wrap resources in Contents; iOS bundles use a flat layout.
    prefix = 'Contents/Resources/' if (bundle / 'Contents/Resources').is_dir() else ''
    payload = {
        file for file in files
        if file not in {'Info.plist', 'Contents/Info.plist'}
        and not file.startswith(('_CodeSignature/', 'Contents/_CodeSignature/'))
    }
    if payload != {prefix + name for name in expected}:
        raise SystemExit(f'error: Unexpected Msplat resources: {sorted(files)}')
PY
if [[ -s "$MSPLAT_LFS_CALL_LOG" ]]; then
    echo 'error: Package resolution or build invoked Git LFS.' >&2
    exit 1
fi

echo "Fresh SwiftPM build and Xcode resolution passed: no datasets, no LFS pointers or invocations, and only Metal shader resources."
