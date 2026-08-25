#!/bin/bash
# Runs the fixed-topology MsplatExample benchmark matrix on one physical iPhone.
#
# The app must already be able to resolve a dataset. Either select and persist a
# folder once in MsplatExample, or pass --dataset for a folder staged in the
# app's Documents directory. Each launch is scoped to the exact bundle ID and
# each console bridge is stopped as soon as the app emits a result or failure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DEVICE=""
APP_PATH=""
BUNDLE_ID=""
DATASET="${MSPLAT_BENCHMARK_DATASET:-}"
WARMUP_ITERATIONS=50
MEASURED_ITERATIONS=300
RUN_TIMEOUT_SECONDS=1800
RESULTS_DIR=""
ONLY_VARIANT=""
PROFILE_STAGES_ENABLED=0
CURRENT_LAUNCH_PID=""

usage() {
    cat <<'USAGE'
Usage:
  ./scripts/run-ios-benchmarks.sh --device <UDID> [options]

Required:
  --device <UDID>          Connected physical iPhone UDID.

Options:
  --app <path>             Signed MsplatExample.app to install. When omitted,
                           the script builds the Release app once.
  --bundle-id <id>         Installed app bundle ID. When omitted, read it from
                           the built/provided app's Info.plist.
  --dataset <name-or-path> Value passed as MSPLAT_BENCHMARK_DATASET. Leave
                           empty to restore the folder previously picked in UI.
  --warmup <count>         Warm-up steps per variant (default: 50).
  --measured <count>       Measured steps per variant (default: 300).
  --timeout <seconds>      Per-variant launch timeout (default: 1800).
  --results-dir <path>     Output directory. Default:
                           build/ios-benchmarks/<UTC timestamp>.
  --only <label>           Run one matrix label instead of all seven.
  --profile-stages         Enable per-stage Metal timestamp logging; requires
                           at least 500 total iterations.
  -h, --help               Show this help.

Prerequisites:
  - Xcode and xcrun devicectl are available.
  - The iPhone is unlocked, trusted, and has Developer Mode enabled.
  - Device signing is configured for the MsplatExample target.
  - MsplatCore.xcframework represents the core revision being benchmarked.
    Rebuild it with ./scripts/build-xcframework.sh when native code changed.
  - The benchmark dataset was selected once in MsplatExample, or --dataset
    identifies a dataset staged in the app's Documents directory.

The app is built and installed once, then these isolated variants run:
  baseline, arena-retry, difference-retry, gather, fused-ssim, raster-16x8,
  raster-16x16

Every run writes a console log and devicectl JSON result. Successful benchmark
summaries are appended to results.jsonl; runs.tsv records every run's status.
The script never removes app data and never sends a broad process kill.

The seven-run matrix is exploratory because its fixed order does not control
device thermals. Before promoting a candidate, use --only with fresh result
directories for a baseline/candidate/baseline comparison from matched states.
USAGE
}

die() {
    echo "error: $*" >&2
    exit 2
}

require_value() {
    local option="$1"
    local value="${2:-}"
    [[ -n "$value" ]] || die "$option requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            require_value "$1" "${2:-}"
            DEVICE="$2"
            shift 2
            ;;
        --app)
            require_value "$1" "${2:-}"
            APP_PATH="$2"
            shift 2
            ;;
        --bundle-id)
            require_value "$1" "${2:-}"
            BUNDLE_ID="$2"
            shift 2
            ;;
        --dataset)
            [[ $# -ge 2 ]] || die "$1 requires a value (use an empty string to clear it)"
            DATASET="$2"
            shift 2
            ;;
        --warmup)
            require_value "$1" "${2:-}"
            WARMUP_ITERATIONS="$2"
            shift 2
            ;;
        --measured)
            require_value "$1" "${2:-}"
            MEASURED_ITERATIONS="$2"
            shift 2
            ;;
        --timeout)
            require_value "$1" "${2:-}"
            RUN_TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --results-dir)
            require_value "$1" "${2:-}"
            RESULTS_DIR="$2"
            shift 2
            ;;
        --only)
            require_value "$1" "${2:-}"
            ONLY_VARIANT="$2"
            shift 2
            ;;
        --profile-stages)
            PROFILE_STAGES_ENABLED=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -n "$DEVICE" ]] || die "--device is required"
[[ "$WARMUP_ITERATIONS" =~ ^[0-9]+$ ]] || die "--warmup must be zero or a positive integer"
[[ "$MEASURED_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || die "--measured must be a positive integer"
[[ "$RUN_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "--timeout must be a positive integer"
(( WARMUP_ITERATIONS + MEASURED_ITERATIONS >= 2 )) || die \
    "Preview benchmarks require at least two total iterations"
if (( PROFILE_STAGES_ENABLED &&
      WARMUP_ITERATIONS + MEASURED_ITERATIONS < 500 )); then
    die "--profile-stages requires at least 500 total iterations"
fi
case "$ONLY_VARIANT" in
    ""|baseline|arena-retry|difference-retry|gather|fused-ssim|raster-16x8|raster-16x16) ;;
    *) die "unknown benchmark label: $ONLY_VARIANT" ;;
esac

command -v xcrun >/dev/null 2>&1 || die "xcrun was not found"
command -v xcodebuild >/dev/null 2>&1 || die "xcodebuild was not found"
xcrun --find devicectl >/dev/null 2>&1 || die "xcrun devicectl is unavailable"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$RESULTS_DIR" ]]; then
    RESULTS_DIR="$ROOT_DIR/build/ios-benchmarks/$TIMESTAMP"
fi
if [[ -d "$RESULTS_DIR" ]] && \
   [[ -n "$(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "results directory must be empty: $RESULTS_DIR"
fi
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"

if [[ -z "$APP_PATH" ]]; then
    [[ -d "$ROOT_DIR/MsplatCore.xcframework" ]] || die \
        "MsplatCore.xcframework is missing; run ./scripts/build-xcframework.sh first"

    DERIVED_DATA="$RESULTS_DIR/DerivedData"
    BUILD_LOG="$RESULTS_DIR/build.log"
    echo "Building Release MsplatExample..."
    if ! xcodebuild \
        -project "$ROOT_DIR/examples/ios/MsplatExample.xcodeproj" \
        -scheme MsplatExample \
        -configuration Release \
        -destination "generic/platform=iOS" \
        -derivedDataPath "$DERIVED_DATA" \
        build 2>&1 | tee "$BUILD_LOG"; then
        die "Release build failed; see $BUILD_LOG"
    fi
    APP_PATH="$DERIVED_DATA/Build/Products/Release-iphoneos/MsplatExample.app"
fi

[[ -d "$APP_PATH" ]] || die "app bundle does not exist: $APP_PATH"
APP_PATH="$(cd "$(dirname "$APP_PATH")" && pwd)/$(basename "$APP_PATH")"

if [[ -z "$BUNDLE_ID" ]]; then
    INFO_PLIST="$APP_PATH/Info.plist"
    [[ -f "$INFO_PLIST" ]] || die "Info.plist is missing; pass --bundle-id explicitly"
    BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$INFO_PLIST" 2>/dev/null || true)"
    [[ -n "$BUNDLE_ID" ]] || die "could not read CFBundleIdentifier; pass --bundle-id explicitly"
fi

INSTALL_LOG="$RESULTS_DIR/install.log"
INSTALL_JSON="$RESULTS_DIR/install.json"
if [[ -z "$ONLY_VARIANT" ]]; then
    echo "Note: repeat promising variants with --only in a counterbalanced order."
fi
echo "Installing $APP_PATH on $DEVICE..."
if ! xcrun devicectl device install app \
    --device "$DEVICE" \
    --timeout 300 \
    --json-output "$INSTALL_JSON" \
    "$APP_PATH" 2>&1 | tee "$INSTALL_LOG"; then
    die "installation failed; see $INSTALL_LOG"
fi

RESULTS_JSONL="$RESULTS_DIR/results.jsonl"
RUNS_TSV="$RESULTS_DIR/runs.tsv"
: > "$RESULTS_JSONL"
printf 'label\tstatus\tconsole_bridge_exit\telapsed_seconds\tconsole_log\tdevicectl_json\n' > "$RUNS_TSV"

json_escape() {
    local value="$1"
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//$'\n'/\\n}
    value=${value//$'\r'/\\r}
    value=${value//$'\t'/\\t}
    printf '%s' "$value"
}

stop_current_launcher() {
    local pid="$CURRENT_LAUNCH_PID"
    [[ -n "$pid" ]] || return 0
    kill -0 "$pid" 2>/dev/null || return 0

    # devicectl documents that catchable signals are forwarded to the app when
    # --console is active. This targets only the launcher created by this run.
    kill -TERM "$pid" 2>/dev/null || true
    local attempt
    for attempt in {1..20}; do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.25
    done
    kill -KILL "$pid" 2>/dev/null || true
}

trap stop_current_launcher EXIT
trap 'exit 130' INT TERM

run_variant() {
    local label="$1"
    local tile_count_mode="$2"
    local tile_layout_mode="$3"
    local arena_mode="$4"
    local intersection_attributes="$5"
    local ssim_mode="$6"
    local raster_variant="$7"
    local console_log="$RESULTS_DIR/$label.console.log"
    local launch_json="$RESULTS_DIR/$label.devicectl.json"
    local environment_json
    local start_seconds=$SECONDS
    local deadline=$((SECONDS + RUN_TIMEOUT_SECONDS))
    local bridge_timeout=$((RUN_TIMEOUT_SECONDS + 10))
    local outcome=""
    local launcher_exit=0
    local profile_stages_json=""

    if (( PROFILE_STAGES_ENABLED )); then
        profile_stages_json=',"PROFILE_STAGES":"1"'
    fi

    printf -v environment_json \
        '{"MSPLAT_BENCHMARK":"1","MSPLAT_BENCHMARK_LABEL":"%s","MSPLAT_BENCHMARK_WARMUP":"%s","MSPLAT_BENCHMARK_MEASURED":"%s","MSPLAT_BENCHMARK_DATASET":"%s","MSPLAT_TILE_COUNT_MODE":"%s","MSPLAT_TILE_LAYOUT_MODE":"%s","MSPLAT_TRAINING_ARENA_MODE":"%s","MSPLAT_INTERSECTION_ATTRIBUTES":"%s","MSPLAT_SSIM_MODE":"%s","MSPLAT_RASTER_VARIANT":"%s","MSPLAT_DENSIFY_RANDOM_MODE":"cpu"%s}' \
        "$(json_escape "$label")" \
        "$WARMUP_ITERATIONS" \
        "$MEASURED_ITERATIONS" \
        "$(json_escape "$DATASET")" \
        "$tile_count_mode" \
        "$tile_layout_mode" \
        "$arena_mode" \
        "$intersection_attributes" \
        "$ssim_mode" \
        "$raster_variant" \
        "$profile_stages_json"

    echo "[$label] launching; console: $console_log"
    : > "$console_log"
    xcrun devicectl device process launch \
        --device "$DEVICE" \
        --terminate-existing \
        --console \
        --timeout "$bridge_timeout" \
        --json-output "$launch_json" \
        --environment-variables "$environment_json" \
        "$BUNDLE_ID" > "$console_log" 2>&1 &
    CURRENT_LAUNCH_PID=$!

    while kill -0 "$CURRENT_LAUNCH_PID" 2>/dev/null; do
        if LC_ALL=C grep -Eq 'MSPLAT_BENCHMARK_RESULT \{.*\}[[:space:]]*$' "$console_log"; then
            outcome="success"
            stop_current_launcher
            break
        fi
        if LC_ALL=C grep -q 'MSPLAT_BENCHMARK_FAILURE ' "$console_log"; then
            outcome="app-failure"
            stop_current_launcher
            break
        fi
        if (( SECONDS >= deadline )); then
            outcome="timeout"
            stop_current_launcher
            break
        fi
        sleep 1
    done

    set +e
    wait "$CURRENT_LAUNCH_PID"
    launcher_exit=$?
    set -e
    CURRENT_LAUNCH_PID=""

    # Catch a final line written just before devicectl exited.
    if [[ -z "$outcome" ]] && \
       LC_ALL=C grep -Eq 'MSPLAT_BENCHMARK_RESULT \{.*\}[[:space:]]*$' "$console_log"; then
        outcome="success"
    elif [[ -z "$outcome" ]] && \
         LC_ALL=C grep -q 'MSPLAT_BENCHMARK_FAILURE ' "$console_log"; then
        outcome="app-failure"
    elif [[ -z "$outcome" ]] && (( SECONDS >= deadline )); then
        outcome="timeout"
    elif [[ -z "$outcome" ]]; then
        outcome="launch-failure"
    fi

    local elapsed=$((SECONDS - start_seconds))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$outcome" "$launcher_exit" "$elapsed" \
        "$console_log" "$launch_json" >> "$RUNS_TSV"

    if [[ "$outcome" == "success" ]]; then
        local result_line
        local summary_json
        result_line="$(LC_ALL=C grep -E -m1 'MSPLAT_BENCHMARK_RESULT \{.*\}[[:space:]]*$' "$console_log")"
        summary_json="${result_line#*MSPLAT_BENCHMARK_RESULT }"
        summary_json="${summary_json%$'\r'}"
        printf '%s\n' "$summary_json" >> "$RESULTS_JSONL"

        local report_name
        report_name="$(
            printf '%s' "$summary_json" |
                /usr/bin/plutil -extract resultFile raw -o - -- - 2>/dev/null || true
        )"
        if [[ "$report_name" == "$(basename "$report_name")" ]]; then
            case "$report_name" in
                msplat-benchmark-*.json)
                    mkdir -p "$RESULTS_DIR/raw-reports"
                    if ! xcrun devicectl device copy from \
                        --device "$DEVICE" \
                        --domain-type appDataContainer \
                        --domain-identifier "$BUNDLE_ID" \
                        --source "Documents/$report_name" \
                        --destination "$RESULTS_DIR/raw-reports/$report_name" \
                        --timeout 60 >/dev/null; then
                        echo "[$label] warning: could not retrieve $report_name" >&2
                    fi
                    ;;
                *)
                    echo "[$label] warning: resultFile has an unexpected name" >&2
                    ;;
            esac
        else
            echo "[$label] warning: resultFile is not a plain filename" >&2
        fi
        echo "[$label] complete in ${elapsed}s"
        return 0
    fi

    echo "[$label] $outcome after ${elapsed}s (launcher exit $launcher_exit)" >&2
    if LC_ALL=C grep -q 'MSPLAT_BENCHMARK_FAILURE ' "$console_log"; then
        LC_ALL=C grep -m1 'MSPLAT_BENCHMARK_FAILURE ' "$console_log" >&2 || true
    else
        tail -n 40 "$console_log" >&2 || true
    fi
    return 1
}

# Each experiment changes only the named optimization from the explicit
# baseline. Retry requires GPU-owned layout metadata; the separate
# difference-retry row measures the exact difference-grid counter on that same
# transactional path.
CONFIGURATIONS=(
    'baseline|enumerated|cpu|exact|packed|staged|8x8'
    'arena-retry|enumerated|gpu|retry|packed|staged|8x8'
    'difference-retry|difference|gpu|retry|packed|staged|8x8'
    'gather|enumerated|cpu|exact|gather|staged|8x8'
    'fused-ssim|enumerated|cpu|exact|packed|fused|8x8'
    'raster-16x8|enumerated|cpu|exact|packed|staged|16x8'
    'raster-16x16|enumerated|cpu|exact|packed|staged|16x16'
)

failures=0
for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r label tile_count tile_layout arena attributes ssim raster \
        <<< "$configuration"
    if [[ -n "$ONLY_VARIANT" && "$label" != "$ONLY_VARIANT" ]]; then
        continue
    fi
    if ! run_variant \
        "$label" "$tile_count" "$tile_layout" "$arena" "$attributes" "$ssim" "$raster"; then
        failures=$((failures + 1))
    fi
done

echo "Benchmark artifacts: $RESULTS_DIR"
echo "Successful summaries: $RESULTS_JSONL"
echo "Run manifest: $RUNS_TSV"

if (( failures > 0 )); then
    echo "error: $failures benchmark variant(s) failed; inspect runs.tsv and console logs" >&2
    exit 1
fi

if [[ -n "$ONLY_VARIANT" ]]; then
    echo "Benchmark variant '$ONLY_VARIANT' completed successfully."
else
    echo "All seven benchmark variants completed successfully."
fi
