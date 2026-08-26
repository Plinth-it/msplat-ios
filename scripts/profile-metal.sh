#!/bin/bash
# Capture and profile a bounded command-buffer window from a Metal CLI process.
set -euo pipefail

CAPTURE_COUNT=8
GPU_STATE="default"
EXECUTION_MODE="overlapping"
ATTACH_TIMEOUT_SECONDS=20
OUTPUT_DIR=""

usage() {
    cat <<'USAGE'
Usage:
  ./scripts/profile-metal.sh [options] -- <command> [arguments...]

Options:
  --output-dir <path>     New directory for the trace and reports. Default:
                          /private/tmp/msplat-metal-profile-<UTC timestamp>
  --count <count>         Command buffers to capture (default: 8).
  --gpu-state <state>     default, low, medium, or high (default: default).
  --execution <mode>      overlapping or serial (default: overlapping).
  --attach-timeout <sec>  Wait for GPU Tools registration (default: 20).
  -h, --help              Show this help.

The target is launched with MTL_CAPTURE_ENABLED=1 and
MTLCAPTURE_WAIT_FOR_SIGNAL=1. It pauses at MTLDevice creation until gpucapture
attaches. gpudebug then replays the trace, embeds a profile, and writes an NDJSON
report containing the command ranking and key occupancy/ALU counters.

Example:
  ./scripts/profile-metal.sh --output-dir /private/tmp/msplat-gpu -- \
    ./build/msplat datasets/mipnerf360/garden -n 80 --save-every=-1 \
    -o /private/tmp/msplat-gpu-result.ply
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
        --output-dir)
            require_value "$1" "${2:-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --count)
            require_value "$1" "${2:-}"
            CAPTURE_COUNT="$2"
            shift 2
            ;;
        --gpu-state)
            require_value "$1" "${2:-}"
            GPU_STATE="$2"
            shift 2
            ;;
        --execution)
            require_value "$1" "${2:-}"
            EXECUTION_MODE="$2"
            shift 2
            ;;
        --attach-timeout)
            require_value "$1" "${2:-}"
            ATTACH_TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --)
            shift
            break
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

[[ $# -gt 0 ]] || die "a target command is required after --"
[[ "$CAPTURE_COUNT" =~ ^[1-9][0-9]*$ ]] || die "--count must be positive"
[[ "$ATTACH_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    die "--attach-timeout must be positive"
case "$GPU_STATE" in
    default|low|medium|high) ;;
    *) die "--gpu-state must be default, low, medium, or high" ;;
esac
case "$EXECUTION_MODE" in
    overlapping|serial) ;;
    *) die "--execution must be overlapping or serial" ;;
esac

for tool in gpucapture gpudebug rg; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool was not found"
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="/private/tmp/msplat-metal-profile-$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ ! -e "$OUTPUT_DIR" ]] || die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

TRACE_PATH="$OUTPUT_DIR/msplat.gputrace"
TARGET_LOG="$OUTPUT_DIR/target.log"
CAPTURE_LOG="$OUTPUT_DIR/gpucapture.log"
PROFILE_RUN_LOG="$OUTPUT_DIR/gpudebug-profile-run.jsonl"
PROFILE_REPORT="$OUTPUT_DIR/gpudebug-profile.jsonl"
COMMAND_FILE="$OUTPUT_DIR/command.txt"

printf '%q ' "$@" > "$COMMAND_FILE"
printf '\n' >> "$COMMAND_FILE"

TARGET_PID=""
cleanup() {
    if [[ -n "$TARGET_PID" ]] && kill -0 "$TARGET_PID" 2>/dev/null; then
        kill -TERM "$TARGET_PID" 2>/dev/null || true
        wait "$TARGET_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Launching target and waiting for GPU Tools..."
MTL_CAPTURE_ENABLED=1 MTLCAPTURE_WAIT_FOR_SIGNAL=1 \
    "$@" > "$TARGET_LOG" 2>&1 &
TARGET_PID=$!

ATTACH_DEADLINE=$((SECONDS + ATTACH_TIMEOUT_SECONDS))
until gpucapture list 2>&1 | \
    rg -q "^[[:space:]]*${TARGET_PID}[[:space:]]"; do
    if ! kill -0 "$TARGET_PID" 2>/dev/null; then
        wait "$TARGET_PID" 2>/dev/null || true
        die "target exited before becoming capturable; see $TARGET_LOG"
    fi
    (( SECONDS < ATTACH_DEADLINE )) || \
        die "target did not become capturable within ${ATTACH_TIMEOUT_SECONDS}s"
    sleep 0.1
done

echo "Capturing $CAPTURE_COUNT command buffers from PID $TARGET_PID..."
if ! gpucapture start \
    --pid "$TARGET_PID" \
    --count "$CAPTURE_COUNT" \
    --disable-unused-recording \
    --output "$TRACE_PATH" > "$CAPTURE_LOG" 2>&1; then
    die "gpucapture failed; see $CAPTURE_LOG"
fi

TARGET_STATUS=0
if wait "$TARGET_PID"; then
    TARGET_STATUS=0
else
    TARGET_STATUS=$?
fi
TARGET_PID=""
(( TARGET_STATUS == 0 )) || \
    die "target exited with status $TARGET_STATUS; see $TARGET_LOG"

echo "Profiling captured work with gpudebug..."
gpudebug --oneshot --json \
    --gputrace "$TRACE_PATH" \
    -c "profile run --gpu-state $GPU_STATE --exec $EXECUTION_MODE --embed" \
    > "$PROFILE_RUN_LOG"

# A newly collected profile becomes navigable after reopening the trace and
# loading the embedded session.
gpudebug --oneshot --json \
    --gputrace "$TRACE_PATH" \
    -c "profile load" \
    -c "go performance/commands" \
    -c "go /performance/timeline" \
    -c "info" \
    -c "go /performance/timeline/counters/occupancy" \
    -c "info --all" \
    -c "go /performance/timeline/counters/alu" \
    -c "info --all" > "$PROFILE_REPORT"

rg -q '"type":"percentage"' "$PROFILE_REPORT" || \
    die "gpudebug produced no command-cost ranking; increase --count"

trap - EXIT
echo "Trace:   $TRACE_PATH"
echo "Profile: $PROFILE_REPORT"
echo "Target:  $TARGET_LOG"
