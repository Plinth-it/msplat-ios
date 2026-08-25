#import "bindings.h"
#include "intersection_layout.hpp"
#define BLOCK_X 16
#define BLOCK_Y 16

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <algorithm>
#import <chrono>
#import <atomic>
#import <cmath>
#import <condition_variable>
#import <cstdlib>
#import <cstring>
#import <dlfcn.h>
#import <unordered_map>
#import <functional>
#import <array>
#import <limits>
#import <memory>
#import <mutex>
#import <new>
#import <random>
#import <stdexcept>
#import <string>
#import <utility>
#import <vector>
#import <mach/mach_time.h>

// GPU profiling infrastructure.
// PROFILE_GPU=1: per-CB total GPU time via completion handlers.
// PROFILE_STAGES=1: per-stage GPU time via Metal timestamp counters + separate encoders.
static bool g_gpu_timing_enabled = false;
static bool g_gpu_timing_checked = false;
static std::mutex g_gpu_timing_mutex;
static std::vector<double> g_gpu_times_ms;

// Per-stage profiling
static bool g_profile_stages = false;
static bool g_profile_stages_checked = false;

// Stage names for training pipeline
static const char* g_train_stage_names[] = {
    "blit_zero", "proj_layout_validate", "scatter_sort_finalize", "pack",
    "rast_fwd", "loss_fwd_bwd", "rast_bwd", "proj_sh_bwd_adam"
};
static constexpr int N_TRAIN_STAGES = 8;

static const char* trainingStageName(int index) {
    return g_train_stage_names[index];
}

static std::mutex g_stage_timing_mutex;
// Per-stage accumulated times (ms), indexed by stage
static std::vector<double> g_stage_times[N_TRAIN_STAGES];
static int g_stage_report_count = 0;

// The command buffer and transient tensor cache are process-global. Serialize
// every operation that touches either so cleanup, rendering, and training cannot
// mutate their shared lifecycle concurrently.
static std::mutex g_engine_mutex;

namespace {

using TelemetryClock = std::chrono::steady_clock;
static constexpr size_t kTrainingReadbackAttemptMetadataWord =
    4;
static constexpr size_t kTrainingReadbackWordCount =
    kTrainingReadbackAttemptMetadataWord +
    msplat::kTileIntersectionLayoutMetadataWordCount;
static constexpr size_t kTrainingReadbackBytes =
    kTrainingReadbackWordCount * sizeof(uint32_t);
static constexpr NSUInteger kTrainingReadbackLossOffset = sizeof(uint32_t);
static constexpr size_t kTrainingReadbackIntersectionWord = 2;
static constexpr NSUInteger kTrainingReadbackIntersectionOffset =
    kTrainingReadbackIntersectionWord * sizeof(uint32_t);
static constexpr NSUInteger kTrainingReadbackAttemptMetadataOffset =
    kTrainingReadbackAttemptMetadataWord * sizeof(uint32_t);
static_assert(sizeof(std::array<float, 4>) == 16,
              "Metal constant float3 arguments require 16 bytes");

double elapsedMilliseconds(TelemetryClock::time_point start,
                           TelemetryClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct SynchronousGpuMetrics {
    double waitWallMs = 0.0;
    double gpuExecutionMs = 0.0;
    bool gpuTimingValid = false;
};

struct RetryAttemptSnapshot {
    uint32_t failureReasons = MSPLAT_TRAINING_OVERFLOW_NONE;
    std::array<uint32_t, msplat::kTileIntersectionLayoutMetadataWordCount>
        layoutMetadata{};
    int32_t finalInclusiveOffset = 0;
    size_t tileCount = 0;
};

} // namespace

struct MsplatTrainingTelemetryState {
    MTensor acquireReadback(id<MTLDevice> device) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!readbackPool.empty()) {
            MTensor result = std::move(readbackPool.back());
            readbackPool.pop_back();
            return result;
        }

        MTensor result = mtensor_empty(
            device, {static_cast<int64_t>(kTrainingReadbackWordCount)},
            DType::Int32);
        ++allocatedReadbackCount;
        return result;
    }

    void releaseReadback(MTensor&& readback) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            readbackPool.push_back(std::move(readback));
        } catch (...) {
            // Telemetry must never make a Metal completion callback throw.
        }
    }

    uint64_t currentGeneration() const {
        std::lock_guard<std::mutex> lock(mutex);
        return generation;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex);
        ++generation;
        snapshot = {};
        snapshot.generation = generation;
    }

    MsplatTrainingTelemetrySnapshot readSnapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        return snapshot;
    }

    size_t readbackBytes() const {
        std::lock_guard<std::mutex> lock(mutex);
        return allocatedReadbackCount * kTrainingReadbackBytes;
    }

    void publishSubmitted(uint64_t stepGeneration,
                          const MsplatTrainingStepDescriptor& descriptor,
                          double cpuSubmitMs) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (stepGeneration != generation) return;
            const bool hasSubmitted =
                (snapshot.flags & MSPLAT_TRAINING_TELEMETRY_HAS_SUBMITTED) != 0;
            if (hasSubmitted &&
                descriptor.iteration < snapshot.submittedStep.iteration) {
                return;
            }
            snapshot.flags |= MSPLAT_TRAINING_TELEMETRY_HAS_SUBMITTED;
            snapshot.submittedStep = descriptor;
            snapshot.submittedCpuSubmitMs = cpuSubmitMs;
        } catch (...) {
            // Completion telemetry is best-effort and must not terminate a host.
        }
    }

    void publishCompleted(uint64_t stepGeneration,
                          const MsplatCompletedTrainingStepMetrics& completed,
                          bool gpuTimingValid, bool lossValid,
                          bool intersectionCountValid,
                          bool countGpuTimingValid,
                          bool queueIdleTimingValid) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (stepGeneration != generation) return;

            if (completed.overflowReasons != MSPLAT_TRAINING_OVERFLOW_NONE) {
                ++snapshot.overflowedStepCount;
                if (completed.overflowReasons &
                    MSPLAT_TRAINING_OVERFLOW_TILE_CAP) {
                    ++snapshot.tileCapOverflowedStepCount;
                }
                if (completed.overflowReasons &
                    MSPLAT_TRAINING_OVERFLOW_PACKED_CAPACITY) {
                    ++snapshot.packedCapacityOverflowedStepCount;
                }
                snapshot.lastOverflowIteration = std::max(
                    snapshot.lastOverflowIteration, completed.step.iteration);
            }

            const bool hasCompleted =
                (snapshot.flags & MSPLAT_TRAINING_TELEMETRY_HAS_COMPLETED) != 0;
            if (hasCompleted &&
                completed.step.iteration < snapshot.completedStep.step.iteration) {
                return;
            }

            snapshot.flags |= MSPLAT_TRAINING_TELEMETRY_HAS_COMPLETED;
            snapshot.flags &= ~(
                MSPLAT_TRAINING_TELEMETRY_GPU_TIMING_VALID |
                MSPLAT_TRAINING_TELEMETRY_LOSS_VALID |
                MSPLAT_TRAINING_TELEMETRY_INTERSECTION_COUNT_VALID |
                MSPLAT_TRAINING_TELEMETRY_COUNT_GPU_TIMING_VALID |
                MSPLAT_TRAINING_TELEMETRY_QUEUE_IDLE_TIMING_VALID);
            if (gpuTimingValid)
                snapshot.flags |= MSPLAT_TRAINING_TELEMETRY_GPU_TIMING_VALID;
            if (lossValid)
                snapshot.flags |= MSPLAT_TRAINING_TELEMETRY_LOSS_VALID;
            if (intersectionCountValid) {
                snapshot.flags |=
                    MSPLAT_TRAINING_TELEMETRY_INTERSECTION_COUNT_VALID;
            }
            if (countGpuTimingValid) {
                snapshot.flags |=
                    MSPLAT_TRAINING_TELEMETRY_COUNT_GPU_TIMING_VALID;
            }
            if (queueIdleTimingValid) {
                snapshot.flags |=
                    MSPLAT_TRAINING_TELEMETRY_QUEUE_IDLE_TIMING_VALID;
            }
            snapshot.completedStep = completed;
        } catch (...) {
            // Completion telemetry is best-effort and must not terminate a host.
        }
    }

    void publishFailure(uint64_t stepGeneration, int64_t iteration) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (stepGeneration != generation) return;
            snapshot.flags |= MSPLAT_TRAINING_TELEMETRY_HAS_FAILED;
            ++snapshot.failedStepCount;
            snapshot.lastFailedIteration =
                std::max(snapshot.lastFailedIteration, iteration);
        } catch (...) {
            // Completion telemetry is best-effort and must not terminate a host.
        }
    }

private:
    mutable std::mutex mutex;
    uint64_t generation = 1;
    MsplatTrainingTelemetrySnapshot snapshot = [] {
        MsplatTrainingTelemetrySnapshot initial;
        initial.generation = 1;
        return initial;
    }();
    std::vector<MTensor> readbackPool;
    size_t allocatedReadbackCount = 0;
};

struct MsplatLogicalTrainingStep {
    MsplatLogicalTrainingStep(MsplatTrainingTelemetryHandle telemetryState,
                              uint64_t stepGeneration, int64_t stepIteration,
                              TelemetryClock::time_point stepWallStart,
                              TelemetryClock::time_point imagePrepareStart,
                              MTensor stepReadback)
        : telemetry(std::move(telemetryState)),
          generation(stepGeneration), iteration(stepIteration),
          wallStart(stepWallStart), imagePrepareStart(imagePrepareStart),
          readback(std::move(stepReadback)) {}

    MTensor& readbackBuffer() { return readback; }

    RetryAttemptSnapshot completedRetryAttemptSnapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        return completedRetryAttemptSnapshotLocked();
    }

private:
    RetryAttemptSnapshot completedRetryAttemptSnapshotLocked() const {
        validateRetryAttemptReadbackLocked();
        const auto* words = readback.data<uint32_t>();
        RetryAttemptSnapshot snapshot;
        snapshot.failureReasons = words[0];
        std::copy_n(
            words + kTrainingReadbackAttemptMetadataWord,
            snapshot.layoutMetadata.size(), snapshot.layoutMetadata.begin());
        std::memcpy(
            &snapshot.finalInclusiveOffset,
            words + kTrainingReadbackIntersectionWord,
            sizeof(snapshot.finalInclusiveOffset));
        snapshot.tileCount = retryAttemptTileCount;
        return snapshot;
    }

public:
    void markRetryAttemptSnapshotEncoded(size_t tileCount) {
        std::lock_guard<std::mutex> lock(mutex);
        retryAttemptTileCount = tileCount;
        retryAttemptSnapshotEncoded = true;
    }

private:
    void validateRetryAttemptReadbackLocked() const {
        if (!readback.defined() ||
            readback.numel() < static_cast<int64_t>(kTrainingReadbackWordCount)) {
            throw std::logic_error(
                "Training retry-attempt readback is unavailable");
        }
        if (!retryAttemptSnapshotEncoded || retryAttemptTileCount == 0) {
            throw std::logic_error(
                "Training retry-attempt snapshot was not encoded");
        }
    }

public:
    void markCpuStart() {
        std::lock_guard<std::mutex> lock(mutex);
        if (sealed || aborted)
            throw std::logic_error("Training telemetry step is no longer active");
        cpuStart = TelemetryClock::now();
        imagePrepareMs = elapsedMilliseconds(imagePrepareStart, cpuStart);
        cpuStarted = true;
    }

    void recordExactCountPass(const SynchronousGpuMetrics& metrics) {
        std::lock_guard<std::mutex> lock(mutex);
        if (sealed || aborted) return;
        countWaitWallMs += metrics.waitWallMs;
        if (metrics.gpuTimingValid)
            countGpuMs += metrics.gpuExecutionMs;
        countGpuTimingValid = countGpuTimingRecorded
            ? countGpuTimingValid && metrics.gpuTimingValid
            : metrics.gpuTimingValid;
        countGpuTimingRecorded = true;
    }

    void recordIntersectionLayout(
        const msplat::TileIntersectionLayout& layout,
        double arenaGrowDurationMs) {
        std::lock_guard<std::mutex> lock(mutex);
        if (sealed || aborted) return;
        intersectionArenaGrowMs += arenaGrowDurationMs;
        maximumTileCount = layout.maximumTileCount;
        activeTileCount = layout.activeTileCount;
        trivialTileCount = layout.trivialTileCount;
        smallTileCount = layout.smallTileCount;
        mediumTileCount = layout.mediumTileCount;
        largeTileCount = layout.largeTileCount;
    }

    void recordRecoveredIntersectionRetry(uint32_t overflowReasons,
                                          double arenaGrowDurationMs) {
        std::lock_guard<std::mutex> lock(mutex);
        if (sealed || aborted) return;
        recoveredOverflowReasons |= overflowReasons;
        intersectionArenaGrowMs += arenaGrowDurationMs;
    }

    void recordPostCountEncode(TelemetryClock::time_point encodeStart,
                               TelemetryClock::time_point encodeEnd) {
        std::lock_guard<std::mutex> lock(mutex);
        if (sealed || aborted) return;
        postCountEncodeMs += elapsedMilliseconds(encodeStart, encodeEnd);
    }

    void validateForSubmit(
        const MsplatTrainingStepDescriptor& submittedDescriptor) const {
        std::lock_guard<std::mutex> lock(mutex);
        if (sealed || aborted)
            throw std::logic_error("Training telemetry step is no longer active");
        if (!cpuStarted)
            throw std::logic_error("Training telemetry CPU timer was not started");
        if (submittedDescriptor.iteration != iteration)
            throw std::invalid_argument(
                "Training telemetry iteration does not match the active step");
        if (submittedDescriptor.splatCount < 0 ||
            submittedDescriptor.modelCapacity < submittedDescriptor.splatCount ||
            submittedDescriptor.effectiveWidth <= 0 ||
            submittedDescriptor.effectiveHeight <= 0 ||
            submittedDescriptor.activeShDegree < 0) {
            throw std::invalid_argument("Training telemetry descriptor is invalid");
        }
    }

    void beginCommandBuffer() {
        std::lock_guard<std::mutex> lock(mutex);
        ++pendingCommandBuffers;
        ++commandBufferCount;
    }

    void cancelCommandBuffer() noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (pendingCommandBuffers > 0) --pendingCommandBuffers;
            finishIfReadyLocked();
        } catch (...) {
            // Never throw through Objective-C exception recovery paths.
        }
    }

    void finishCommandBuffer(bool succeeded, double gpuStartSeconds,
                             double gpuEndSeconds) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (!succeeded) {
                commandBufferFailed = true;
            } else if (!std::isfinite(gpuStartSeconds) ||
                       !std::isfinite(gpuEndSeconds) ||
                       gpuEndSeconds <= gpuStartSeconds) {
                gpuTimingValid = false;
            } else {
                gpuExecutionMs +=
                    (gpuEndSeconds - gpuStartSeconds) * 1000.0;
                if (gpuIntervalCount < gpuIntervals.size()) {
                    gpuIntervals[gpuIntervalCount++] =
                        {gpuStartSeconds, gpuEndSeconds};
                } else {
                    queueIdleIntervalsValid = false;
                }
            }
            if (pendingCommandBuffers > 0) --pendingCommandBuffers;
            finishIfReadyLocked();
        } catch (...) {
            // A C++ exception must never escape a Metal completion handler.
        }
    }

    void markReadbackEncoded(uint64_t capacity,
                             uint64_t lossCoverageUnits) {
        std::lock_guard<std::mutex> lock(mutex);
        packedIntersectionCapacity = capacity;
        this->lossCoverageUnits = lossCoverageUnits;
        readbackEncoded = true;
    }

    void recordSynchronousGpuWait(TelemetryClock::time_point waitStart,
                                  TelemetryClock::time_point waitEnd) {
        std::lock_guard<std::mutex> lock(mutex);
        if (cpuStarted && !sealed && !aborted) {
            synchronousGpuWaitMs += elapsedMilliseconds(waitStart, waitEnd);
        }
    }

    double seal(const MsplatTrainingStepDescriptor& submittedDescriptor,
                TelemetryClock::time_point cpuEnd) {
        std::lock_guard<std::mutex> lock(mutex);
        descriptor = submittedDescriptor;
        // Exact intersection sizing deliberately waits for its count command
        // buffer. Report CPU encoding/submission time rather than charging that
        // synchronous GPU wait to the legacy CPU metric.
        cpuSubmitMs = std::max(
            0.0, elapsedMilliseconds(cpuStart, cpuEnd) - synchronousGpuWaitMs);
        sealed = true;
        telemetry->publishSubmitted(
            generation, descriptor, cpuSubmitMs);
        finishIfReadyLocked();
        return cpuSubmitMs;
    }

    void abort() noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (sealed) return;
            aborted = true;
            finishIfReadyLocked();
        } catch (...) {
            // Abort is used from catch paths and must remain noexcept.
        }
    }

private:
    bool calculateQueueIdleMsLocked(double& queueIdleMs) noexcept {
        queueIdleMs = 0.0;
        if (!gpuTimingValid || !queueIdleIntervalsValid ||
            commandBufferCount == 0 ||
            gpuIntervalCount != commandBufferCount) {
            return false;
        }

        std::sort(
            gpuIntervals.begin(), gpuIntervals.begin() + gpuIntervalCount,
            [](const auto& lhs, const auto& rhs) noexcept {
                return lhs.first < rhs.first;
            });
        double mergedEnd = gpuIntervals.front().second;
        for (size_t index = 1; index < gpuIntervalCount; ++index) {
            const auto& interval = gpuIntervals[index];
            if (interval.first > mergedEnd) {
                queueIdleMs += interval.first - mergedEnd;
            }
            mergedEnd = std::max(mergedEnd, interval.second);
        }
        queueIdleMs *= 1000.0;
        if (!std::isfinite(queueIdleMs) || queueIdleMs < 0.0) {
            queueIdleMs = 0.0;
            return false;
        }
        return true;
    }

    void finishIfReadyLocked() noexcept {
        if (published || pendingCommandBuffers != 0 || (!sealed && !aborted))
            return;
        published = true;

        if (commandBufferFailed) {
            telemetry->publishFailure(generation, iteration);
        } else if (sealed && !aborted) {
            MsplatCompletedTrainingStepMetrics completed;
            completed.step = descriptor;
            completed.cpuSubmitMs = cpuSubmitMs;
            completed.gpuExecutionMs = gpuExecutionMs;
            completed.endToEndMs =
                elapsedMilliseconds(wallStart, TelemetryClock::now());
            completed.commandBufferCount = commandBufferCount;
            completed.packedIntersectionCapacity =
                packedIntersectionCapacity;
            completed.imagePrepareMs = imagePrepareMs;
            completed.countGpuMs = countGpuMs;
            completed.countWaitWallMs = countWaitWallMs;
            completed.postCountEncodeMs = postCountEncodeMs;
            completed.intersectionArenaGrowMs = intersectionArenaGrowMs;
            completed.maximumTileCount = maximumTileCount;
            completed.activeTileCount = activeTileCount;
            completed.trivialTileCount = trivialTileCount;
            completed.smallTileCount = smallTileCount;
            completed.mediumTileCount = mediumTileCount;
            completed.largeTileCount = largeTileCount;

            if (retryAttemptSnapshotEncoded) {
                try {
                    const RetryAttemptSnapshot snapshot =
                        completedRetryAttemptSnapshotLocked();
                    const msplat::TileIntersectionLayout layout =
                        msplat::tileIntersectionLayoutFromGpuMetadata(
                            snapshot.layoutMetadata.data(),
                            snapshot.layoutMetadata.size(), snapshot.tileCount,
                            snapshot.finalInclusiveOffset);
                    msplat::validateTileIntersectionWorkLimit(layout);
                    completed.maximumTileCount = layout.maximumTileCount;
                    completed.activeTileCount = layout.activeTileCount;
                    completed.trivialTileCount = layout.trivialTileCount;
                    completed.smallTileCount = layout.smallTileCount;
                    completed.mediumTileCount = layout.mediumTileCount;
                    completed.largeTileCount = layout.largeTileCount;
                } catch (...) {
                    telemetry->publishFailure(generation, iteration);
                    telemetry->releaseReadback(std::move(readback));
                    return;
                }
            }

            bool lossValid = false;
            bool intersectionCountValid = false;
            if (readbackEncoded && readback.defined()) {
                const auto* words = readback.data<uint32_t>();
                completed.overflowReasons = recoveredOverflowReasons |
                    (words[0] & (
                    MSPLAT_TRAINING_OVERFLOW_TILE_CAP |
                    MSPLAT_TRAINING_OVERFLOW_PACKED_CAPACITY));

                float rawLoss = 0.0f;
                std::memcpy(&rawLoss, &words[1], sizeof(rawLoss));
                if (lossCoverageUnits > 0 && std::isfinite(rawLoss)) {
                    completed.loss =
                        static_cast<double>(rawLoss) * 255.0 /
                        static_cast<double>(lossCoverageUnits);
                    lossValid = std::isfinite(completed.loss);
                }

                int32_t exactIntersections = 0;
                std::memcpy(&exactIntersections, &words[2],
                            sizeof(exactIntersections));
                if (exactIntersections >= 0) {
                    completed.retainedPackedIntersections = std::min<uint64_t>(
                        static_cast<uint64_t>(exactIntersections),
                        packedIntersectionCapacity);
                    intersectionCountValid = true;
                }
            }

            const bool completedGpuTimingValid =
                commandBufferCount > 0 && gpuTimingValid;
            if (!completedGpuTimingValid)
                completed.gpuExecutionMs = 0.0;
            const bool queueIdleTimingValid =
                calculateQueueIdleMsLocked(completed.queueIdleMs);
            telemetry->publishCompleted(
                generation, completed, completedGpuTimingValid, lossValid,
                intersectionCountValid, countGpuTimingValid,
                queueIdleTimingValid);
        }

        telemetry->releaseReadback(std::move(readback));
    }

    MsplatTrainingTelemetryHandle telemetry;
    uint64_t generation = 0;
    int64_t iteration = 0;
    TelemetryClock::time_point wallStart;
    TelemetryClock::time_point imagePrepareStart;
    TelemetryClock::time_point cpuStart;
    MTensor readback;
    mutable std::mutex mutex;
    MsplatTrainingStepDescriptor descriptor;
    size_t pendingCommandBuffers = 0;
    uint32_t commandBufferCount = 0;
    uint64_t packedIntersectionCapacity = 0;
    uint64_t lossCoverageUnits = 0;
    double cpuSubmitMs = 0.0;
    double synchronousGpuWaitMs = 0.0;
    double gpuExecutionMs = 0.0;
    // Current exact and retry paths use at most three roots. Keep additional
    // headroom for later queue-depth experiments without allocating in a GPU
    // completion callback; larger future chains simply omit this metric.
    std::array<std::pair<double, double>, 8> gpuIntervals{};
    size_t gpuIntervalCount = 0;
    bool queueIdleIntervalsValid = true;
    double imagePrepareMs = 0.0;
    double countGpuMs = 0.0;
    double countWaitWallMs = 0.0;
    double postCountEncodeMs = 0.0;
    double intersectionArenaGrowMs = 0.0;
    uint32_t recoveredOverflowReasons = MSPLAT_TRAINING_OVERFLOW_NONE;
    uint32_t maximumTileCount = 0;
    uint32_t activeTileCount = 0;
    uint32_t trivialTileCount = 0;
    uint32_t smallTileCount = 0;
    uint32_t mediumTileCount = 0;
    uint32_t largeTileCount = 0;
    bool cpuStarted = false;
    bool gpuTimingValid = true;
    bool countGpuTimingValid = false;
    bool countGpuTimingRecorded = false;
    bool readbackEncoded = false;
    size_t retryAttemptTileCount = 0;
    bool retryAttemptSnapshotEncoded = false;
    bool commandBufferFailed = false;
    bool sealed = false;
    bool aborted = false;
    bool published = false;
};

struct ScopedObjCRelease {
    id object = nil;

    ~ScopedObjCRelease() {
        [object release];
    }
};

struct CommandCompletionState {
    void beginSubmission() {
        std::lock_guard<std::mutex> lock(mutex);
        ++pendingSubmissions;
    }

    void cancelSubmission() noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (pendingSubmissions > 0) --pendingSubmissions;
            completed.notify_all();
        } catch (...) {
            // Mutex failures indicate process-level corruption; there is no
            // useful recovery work a Metal completion callback can perform.
        }
    }

    void finishSubmission(id<MTLCommandBuffer> commandBuffer) noexcept {
        char failure[1024] = {};
        const MTLCommandBufferStatus status = commandBuffer.status;
        if (status != MTLCommandBufferStatusCompleted) {
            NSError* error = commandBuffer.error;
            const char* description = error.localizedDescription.UTF8String;
            if (description) {
                snprintf(failure, sizeof(failure),
                         "msplat: Metal command buffer did not complete: %s",
                         description);
            } else {
                snprintf(failure, sizeof(failure),
                         "msplat: Metal command buffer did not complete (status %ld)",
                         static_cast<long>(status));
            }
        }

        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (failure[0] && firstFailure[0] == '\0')
                snprintf(firstFailure, sizeof(firstFailure), "%s", failure);
            if (pendingSubmissions > 0) --pendingSubmissions;
            completed.notify_all();
        } catch (...) {
            // Do not let a C++ exception escape a Metal completion handler.
        }
    }

    void waitAndConsumeFailure(char* destination, size_t capacity) {
        std::unique_lock<std::mutex> lock(mutex);
        completed.wait(lock, [&] { return pendingSubmissions == 0; });
        if (destination[0] == '\0' && firstFailure[0] != '\0')
            snprintf(destination, capacity, "%s", firstFailure);
        firstFailure[0] = '\0';
    }

private:
    std::mutex mutex;
    std::condition_variable completed;
    size_t pendingSubmissions = 0;
    char firstFailure[1024] = {};
};

struct MetalContext {
    id<MTLDevice>       device = nil;
    id<MTLCommandQueue> queue = nil;
    dispatch_queue_t d_queue = nullptr;

    // Command buffer lifecycle using MPSCommandBuffer for commitAndContinue support.
    MPSCommandBuffer* _currentCB = nil;
    // Completion handlers accumulate the first error from every root submitted
    // through commitAndContinue without retaining thousands of completed
    // command buffers during long training runs.
    std::shared_ptr<CommandCompletionState> commandCompletionState =
        std::make_shared<CommandCompletionState>();
    // The active logical step is installed by Trainer::step. A root command
    // buffer is associated only when training first asks to encode into it;
    // this avoids charging an empty ordering fence from a prior
    // commitAndContinue to the next step.
    MsplatLogicalTrainingStepHandle activeTrainingStep;
    MsplatLogicalTrainingStepHandle currentCBTrainingStep;

    id<MTLCommandBuffer> getCommandBuffer() {
        if (!_currentCB) {
            _currentCB = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            if (!_currentCB) {
                throw std::runtime_error("msplat: failed to create a Metal command buffer");
            }
            [_currentCB retain];
        }
        if (activeTrainingStep) {
            if (currentCBTrainingStep &&
                currentCBTrainingStep != activeTrainingStep) {
                throw std::logic_error(
                    "A Metal command buffer spans two logical training steps");
            }
            currentCBTrainingStep = activeTrainingStep;
        }
        return _currentCB;
    }
    void commitCB() {
        if (_currentCB) {
            id<MTLCommandBuffer> submitted = _currentCB.rootCommandBuffer;
            if (!submitted) {
                throw std::runtime_error(
                    "msplat: MPS command buffer has no root command buffer");
            }
            auto completionState = commandCompletionState;
            auto logicalStep = currentCBTrainingStep;
            const bool collectTiming = g_gpu_timing_enabled;
            completionState->beginSubmission();
            if (logicalStep) logicalStep->beginCommandBuffer();
            @try {
                [submitted addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                    if (logicalStep) {
                        logicalStep->finishCommandBuffer(
                            cb.status == MTLCommandBufferStatusCompleted,
                            cb.GPUStartTime, cb.GPUEndTime);
                    }
                    completionState->finishSubmission(cb);
                    if (!collectTiming) return;
                    double gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                    if (gpu_ms > 0) {
                        std::lock_guard<std::mutex> lock(g_gpu_timing_mutex);
                        g_gpu_times_ms.push_back(gpu_ms);
                    }
                }];
                [_currentCB commitAndContinue];
                currentCBTrainingStep.reset();
            } @catch (NSException *exception) {
                const char* reason = exception.reason.UTF8String;
                std::string message = "msplat: Metal commitAndContinue failed";
                if (reason) message += std::string(": ") + reason;
                if (logicalStep) logicalStep->cancelCommandBuffer();
                completionState->cancelSubmission();
                throw std::runtime_error(message);
            }
        }
    }
    void discardCB() noexcept {
        if (_currentCB) {
            [_currentCB release];
            _currentCB = nil;
        }
        currentCBTrainingStep.reset();
    }
    SynchronousGpuMetrics syncCB() {
        if (!g_gpu_timing_checked) {
            g_gpu_timing_enabled = std::getenv("PROFILE_GPU") != nullptr;
            g_gpu_timing_checked = true;
        }
        const bool collectTiming = g_gpu_timing_enabled;
        SynchronousGpuMetrics metrics;
        char failure[1024] = {};
        MPSCommandBuffer* commandBuffer = _currentCB;
        auto logicalStep = currentCBTrainingStep;
        id<MTLCommandBuffer> finalCommandBuffer = nil;
        bool finalWasCommitted = false;
        if (commandBuffer) {
            finalCommandBuffer = commandBuffer.rootCommandBuffer;
            if (!finalCommandBuffer) {
                throw std::runtime_error(
                    "msplat: MPS command buffer has no root command buffer");
            }
            [finalCommandBuffer retain];
            _currentCB = nil;
            currentCBTrainingStep.reset();
            if (logicalStep) logicalStep->beginCommandBuffer();

            @try {
                [commandBuffer commit];
                finalWasCommitted = true;
            } @catch (NSException *exception) {
                const char* reason = exception.reason.UTF8String;
                snprintf(failure, sizeof(failure),
                         "msplat: Metal command buffer commit failed%s%s",
                         reason ? ": " : "", reason ? reason : "");
                if (logicalStep) logicalStep->cancelCommandBuffer();
            }
        }

        auto waitAndRecordFailure = [&](id<MTLCommandBuffer> submitted) {
            const auto waitStart = TelemetryClock::now();
            [submitted waitUntilCompleted];
            const auto waitEnd = TelemetryClock::now();
            metrics.waitWallMs += elapsedMilliseconds(waitStart, waitEnd);
            if (logicalStep) {
                logicalStep->recordSynchronousGpuWait(waitStart, waitEnd);
            }
            const MTLCommandBufferStatus status = submitted.status;
            if (status != MTLCommandBufferStatusCompleted && failure[0] == '\0') {
                NSError* error = submitted.error;
                const char* description = error.localizedDescription.UTF8String;
                if (description) {
                    snprintf(failure, sizeof(failure),
                             "msplat: Metal command buffer did not complete: %s",
                             description);
                } else {
                    snprintf(failure, sizeof(failure),
                             "msplat: Metal command buffer did not complete (status %ld)",
                             static_cast<long>(status));
                }
            }
            if (logicalStep) {
                logicalStep->finishCommandBuffer(
                    status == MTLCommandBufferStatusCompleted,
                    submitted.GPUStartTime, submitted.GPUEndTime);
            }
            if (status == MTLCommandBufferStatusCompleted &&
                std::isfinite(submitted.GPUStartTime) &&
                std::isfinite(submitted.GPUEndTime) &&
                submitted.GPUEndTime > submitted.GPUStartTime) {
                metrics.gpuExecutionMs =
                    (submitted.GPUEndTime - submitted.GPUStartTime) * 1000.0;
                metrics.gpuTimingValid = true;
            }
            if (collectTiming && status == MTLCommandBufferStatusCompleted) {
                const double gpuMs =
                    (submitted.GPUEndTime - submitted.GPUStartTime) * 1000.0;
                if (std::isfinite(gpuMs) && gpuMs > 0) {
                    std::lock_guard<std::mutex> lock(g_gpu_timing_mutex);
                    g_gpu_times_ms.push_back(gpuMs);
                }
            }
            [submitted release];
        };

        if (finalCommandBuffer) {
            if (finalWasCommitted)
                waitAndRecordFailure(finalCommandBuffer);
            else
                [finalCommandBuffer release];
        }
        if (commandBuffer)
            [commandBuffer release];

        // The final command buffer is ordered after all roots previously
        // submitted on this queue. Wait for their completion handlers too so
        // errors from any earlier step are deterministically visible here.
        commandCompletionState->waitAndConsumeFailure(failure, sizeof(failure));

        if (failure[0])
            throw std::runtime_error(failure);
        return metrics;
    }

    // Per-stage GPU timestamp profiling. Each logical training step owns its
    // own sample buffer so queued completion handlers cannot observe samples
    // overwritten by a later step.
    id<MTLCounterSet> timestampCounterSet = nil;
    bool counterSamplingAvailable = false;
    double ticksToMs = 0.0;  // conversion factor from GPU ticks to milliseconds

    id<MTLCounterSampleBuffer> newTrainingCounterSampleBuffer() {
        if (!timestampCounterSet) return nil;

        MTLCounterSampleBufferDescriptor *desc =
            [MTLCounterSampleBufferDescriptor new];
        ScopedObjCRelease descOwner{desc};
        desc.counterSet = timestampCounterSet;
        desc.sampleCount = N_TRAIN_STAGES * 2;
        desc.storageMode = MTLStorageModeShared;
        desc.label = @"msplat training stage profiling";

        NSError *error = nil;
        id<MTLCounterSampleBuffer> sampleBuffer =
            [device newCounterSampleBufferWithDescriptor:desc error:&error];
        if (!sampleBuffer) {
            const char *description = error.localizedDescription.UTF8String;
            fprintf(stderr,
                    "PROFILE_STAGES: Failed to create counter sample buffer%s%s\n",
                    description ? ": " : "", description ? description : "");
        }
        return sampleBuffer;
    }

    void initCounterSampling() {
        // Need 2 samples per stage (start + end)
        NSUInteger sampleCount = N_TRAIN_STAGES * 2;

        // Find timestamp counter set
        id<MTLCounterSet> timestampSet = nil;
        for (id<MTLCounterSet> cs in device.counterSets) {
            if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp]) {
                timestampSet = cs;
                break;
            }
        }
        if (!timestampSet) {
            fprintf(stderr, "PROFILE_STAGES: MTLCommonCounterSetTimestamp not available\n");
            return;
        }

        // Check if stage boundary sampling is supported (guaranteed on Apple Silicon)
        if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            fprintf(stderr, "PROFILE_STAGES: AtStageBoundary sampling not supported\n");
            return;
        }

        timestampCounterSet = [timestampSet retain];
        id<MTLCounterSampleBuffer> probe = newTrainingCounterSampleBuffer();
        if (!probe) {
            [timestampCounterSet release];
            timestampCounterSet = nil;
            return;
        }
        [probe release];

        // Derive the GPU tick duration from two paired CPU/GPU samples rather
        // than assuming GPU timestamps use the CPU's Mach timebase.
        mach_timebase_info_data_t timebase = {};
        MTLTimestamp cpuStart = 0;
        MTLTimestamp gpuStart = 0;
        MTLTimestamp cpuEnd = 0;
        MTLTimestamp gpuEnd = 0;
        if (mach_timebase_info(&timebase) != KERN_SUCCESS ||
            timebase.numer == 0 || timebase.denom == 0) {
            fprintf(stderr,
                    "PROFILE_STAGES: Failed to read the CPU timebase\n");
            [timestampCounterSet release];
            timestampCounterSet = nil;
            return;
        }
        [device sampleTimestamps:&cpuStart gpuTimestamp:&gpuStart];
        constexpr uint64_t calibrationNanoseconds = 5'000'000;
        const uint64_t calibrationCpuTicks =
            (calibrationNanoseconds * timebase.denom +
             timebase.numer - 1) / timebase.numer;
        mach_wait_until(mach_absolute_time() + calibrationCpuTicks);
        [device sampleTimestamps:&cpuEnd gpuTimestamp:&gpuEnd];
        if (cpuEnd > cpuStart && gpuEnd > gpuStart) {
            // Metal reports the paired CPU timestamps in nanoseconds.
            const double cpuElapsedMs =
                static_cast<double>(cpuEnd - cpuStart) / 1e6;
            ticksToMs = cpuElapsedMs /
                static_cast<double>(gpuEnd - gpuStart);
        }
        if (!std::isfinite(ticksToMs) || ticksToMs <= 0.0) {
            fprintf(stderr,
                    "PROFILE_STAGES: Failed to calibrate GPU timestamps\n");
            [timestampCounterSet release];
            timestampCounterSet = nil;
            ticksToMs = 0.0;
            return;
        }

        counterSamplingAvailable = true;
        fprintf(stderr, "PROFILE_STAGES: GPU timestamp profiling enabled (%lu sample slots)\n",
                (unsigned long)sampleCount);
    }

    // Forward pipeline kernels
    id<MTLComputePipelineState> project_and_sh_forward_kernel_cpso = nil;
    id<MTLComputePipelineState> tile_count_diff_horizontal_kernel_cpso = nil;
    id<MTLComputePipelineState> tile_count_diff_vertical_kernel_cpso = nil;
    bool difference_tile_counting = false;
    id<MTLComputePipelineState> build_tile_intersection_layout_kernel_cpso = nil;
    bool gpu_tile_layout = false;
    bool retry_intersection_attempts = false;
    bool gather_intersection_attributes = true;
    id<MTLComputePipelineState> validate_tile_intersection_attempt_kernel_cpso = nil;
    id<MTLComputePipelineState> finalize_tile_intersection_attempt_kernel_cpso = nil;
    id<MTLComputePipelineState> nd_rasterize_forward_kernel_cpso = nil;
    uint32_t monolithic_raster_block_x = 8;
    uint32_t monolithic_raster_block_y = 8;
    id<MTLComputePipelineState> float_rgb_to_preview_texture_kernel_cpso = nil;
    // Exact tile-intersection compaction and sorting
    id<MTLComputePipelineState> scatter_to_exact_bins_kernel_cpso = nil;
    id<MTLComputePipelineState> small_sort_per_tile_kernel_cpso = nil;
    id<MTLComputePipelineState> radix_sort_per_tile_kernel_cpso = nil;
    id<MTLComputePipelineState> pack_sorted_gaussians_kernel_cpso = nil;
    id<MTLComputePipelineState> block_reduce_kernel_cpso = nil;
    id<MTLComputePipelineState> block_scan_propagate_kernel_cpso = nil;
    // Depth-chunked rasterization
    id<MTLComputePipelineState> rasterize_forward_chunked_kernel_cpso = nil;
    id<MTLComputePipelineState> rasterize_forward_merge_kernel_cpso = nil;
    id<MTLComputePipelineState> compute_chunk_prefix_suffix_kernel_cpso = nil;
    id<MTLComputePipelineState> rasterize_backward_chunked_kernel_cpso = nil;
    id<MTLComputePipelineState> rasterize_backward_kernel_cpso = nil;
    // Separable SSIM loss kernels
    id<MTLComputePipelineState> ssim_h_fwd_kernel_cpso = nil;
    id<MTLComputePipelineState> ssim_v_fwd_kernel_cpso = nil;
    id<MTLComputePipelineState> ssim_fused_v_fwd_h_bwd_kernel_cpso = nil;
    id<MTLComputePipelineState> ssim_fused_v_fwd_bwd_kernel_cpso = nil;
    id<MTLComputePipelineState> ssim_v_bwd_kernel_cpso = nil;
    bool fused_ssim_backward = true;
    id<MTLComputePipelineState> photometric_adam_kernel_cpso = nil;
    id<MTLComputePipelineState> prepare_camera_pose_kernel_cpso = nil;
    id<MTLComputePipelineState> camera_pose_adam_kernel_cpso = nil;
    // Backward pipeline kernels
    id<MTLComputePipelineState> sh_opacity_backward_adam_kernel_cpso = nil;
    id<MTLComputePipelineState> project_backward_adam_kernel_cpso = nil;
    id<MTLComputePipelineState> reset_opacity_state_kernel_cpso = nil;
    // GPU densification kernels
    bool gpu_densify_random = false;
    id<MTLComputePipelineState> densify_classify_kernel_cpso = nil;
    id<MTLComputePipelineState> densify_append_split_kernel_cpso = nil;
    id<MTLComputePipelineState> densify_append_dup_kernel_cpso = nil;
    id<MTLComputePipelineState> densify_cull_classify_kernel_cpso = nil;
    id<MTLComputePipelineState> compact_scatter_kernel_cpso = nil;
    id<MTLComputePipelineState> compact_copy_back_kernel_cpso = nil;

    ~MetalContext() noexcept {
        try {
            syncCB();
        } catch (const std::exception& error) {
            fprintf(stderr, "msplat: Metal teardown sync failed: %s\n", error.what());
        }
        if (activeTrainingStep) {
            auto abandoned = std::move(activeTrainingStep);
            currentCBTrainingStep.reset();
            abandoned->abort();
        }

        id resources[] = {
            timestampCounterSet,
            project_and_sh_forward_kernel_cpso,
            tile_count_diff_horizontal_kernel_cpso,
            tile_count_diff_vertical_kernel_cpso,
            build_tile_intersection_layout_kernel_cpso,
            validate_tile_intersection_attempt_kernel_cpso,
            finalize_tile_intersection_attempt_kernel_cpso,
            nd_rasterize_forward_kernel_cpso,
            float_rgb_to_preview_texture_kernel_cpso,
            scatter_to_exact_bins_kernel_cpso,
            small_sort_per_tile_kernel_cpso,
            radix_sort_per_tile_kernel_cpso,
            pack_sorted_gaussians_kernel_cpso,
            block_reduce_kernel_cpso,
            block_scan_propagate_kernel_cpso,
            rasterize_forward_chunked_kernel_cpso,
            rasterize_forward_merge_kernel_cpso,
            compute_chunk_prefix_suffix_kernel_cpso,
            rasterize_backward_chunked_kernel_cpso,
            rasterize_backward_kernel_cpso,
            ssim_h_fwd_kernel_cpso,
            ssim_v_fwd_kernel_cpso,
            ssim_fused_v_fwd_h_bwd_kernel_cpso,
            ssim_fused_v_fwd_bwd_kernel_cpso,
            ssim_v_bwd_kernel_cpso,
            photometric_adam_kernel_cpso,
            prepare_camera_pose_kernel_cpso,
            camera_pose_adam_kernel_cpso,
            sh_opacity_backward_adam_kernel_cpso,
            project_backward_adam_kernel_cpso,
            reset_opacity_state_kernel_cpso,
            densify_classify_kernel_cpso,
            densify_append_split_kernel_cpso,
            densify_append_dup_kernel_cpso,
            densify_cull_classify_kernel_cpso,
            compact_scatter_kernel_cpso,
            compact_copy_back_kernel_cpso,
        };
        for (id resource : resources) {
            [resource release];
        }

        if (d_queue) dispatch_release(d_queue);
        [queue release];
        [device release];
    }
};

// Explicit metallib path (set by Swift/Python wrappers before first use)
static char* g_metallib_path = NULL;
static std::mutex g_context_config_mutex;
static bool g_context_config_frozen = false;
static std::atomic<MetalContext*> g_context_instance{nullptr};

struct ContextConfigurationSnapshot {
    bool hasExplicitMetallibPath = false;
    std::string metallibPath;
    bool succeeded = false;

    ContextConfigurationSnapshot() {
        std::lock_guard<std::mutex> lock(g_context_config_mutex);
        if (g_metallib_path) {
            metallibPath = g_metallib_path;
            hasExplicitMetallibPath = true;
        }
        g_context_config_frozen = true;
    }

    ContextConfigurationSnapshot(const ContextConfigurationSnapshot&) = delete;
    ContextConfigurationSnapshot& operator=(const ContextConfigurationSnapshot&) = delete;

    ~ContextConfigurationSnapshot() {
        std::lock_guard<std::mutex> lock(g_context_config_mutex);
        if (succeeded) {
            free(g_metallib_path);
            g_metallib_path = nullptr;
        } else {
            g_context_config_frozen = false;
        }
    }
};

void msplat_set_metallib_path_checked(const char* path) {
    char* replacement = path ? strdup(path) : NULL;
    if (path && !replacement) {
        throw std::bad_alloc();
    }

    std::lock_guard<std::mutex> lock(g_context_config_mutex);
    if (g_context_config_frozen) {
        free(replacement);
        throw std::invalid_argument(
            "msplat: metallib path must be set before first Metal use");
    }

    free(g_metallib_path);
    g_metallib_path = replacement;
}

extern "C" void msplat_set_metallib_path(const char* path) {
    try {
        msplat_set_metallib_path_checked(path);
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
    } catch (...) {
        fprintf(stderr, "msplat: failed to configure metallib path\n");
    }
}

MetalContext* init_msplat_metal_context() {
    ContextConfigurationSnapshot configuration;
    auto ctx = std::make_unique<MetalContext>();

    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        throw std::runtime_error("msplat: no Metal device is available");
    }

    ctx->queue  = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        throw std::runtime_error("msplat: failed to create the Metal command queue");
    }

    ctx->d_queue = dispatch_queue_create("com.msplat.metal", DISPATCH_QUEUE_SERIAL);
    if (!ctx->d_queue) {
        throw std::runtime_error("msplat: failed to create the Metal encoding queue");
    }

    // Find precompiled metallib: explicit path (XCFramework/Python) or auto-discover
    NSError *error = nil;
    id<MTLLibrary> metal_library = nil;

    if (configuration.hasExplicitMetallibPath) {
        // Explicit path (set by XCFramework / Swift wrapper)
        NSString *path = [NSString stringWithUTF8String:configuration.metallibPath.c_str()];
        if (!path) {
            throw std::runtime_error("msplat: metallib path is not valid UTF-8");
        }
        NSURL *url = [NSURL fileURLWithPath:path];
        metal_library = [ctx->device newLibraryWithURL:url error:&error];
    } else {
        // Auto-discover default.metallib next to this library or the executable
        NSFileManager *fm = [NSFileManager defaultManager];

        // 1. Next to this shared library (Python .so / linked .a)
        Dl_info dl_info;
        if (dladdr((void*)init_msplat_metal_context, &dl_info) && dl_info.dli_fname) {
            NSString *dir = [[NSString stringWithUTF8String:dl_info.dli_fname] stringByDeletingLastPathComponent];
            NSString *path = [dir stringByAppendingPathComponent:@"default.metallib"];
            if ([fm fileExistsAtPath:path]) {
                metal_library = [ctx->device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            }
        }
        // 2. Next to the main executable (CLI build)
        if (!metal_library) {
            NSString *dir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
            NSString *path = [dir stringByAppendingPathComponent:@"default.metallib"];
            if (dir && [fm fileExistsAtPath:path]) {
                metal_library = [ctx->device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            }
        }
    }

    if (!metal_library) {
        const char* description = error ? error.localizedDescription.UTF8String : nullptr;
        std::string failure = "msplat: failed to load metallib: ";
        failure += description ? description : "default.metallib not found";
        fprintf(stderr, "%s\n", failure.c_str());
        throw std::runtime_error(failure);
    }
    ScopedObjCRelease metalLibraryOwner{metal_library};

    bool pipelineLoadFailed = false;
    std::string pipelineFailure;
    auto load = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [metal_library newFunctionWithName:name];
        if (!fn) {
            fprintf(stderr, "msplat: kernel not found: %s\n", [name UTF8String]);
            if (!pipelineLoadFailed) {
                pipelineFailure = "msplat: kernel not found: ";
                pipelineFailure += name.UTF8String;
            }
            pipelineLoadFailed = true;
            return nil;
        }
        NSError* pipelineError = nil;
        id<MTLComputePipelineState> pso =
            [ctx->device newComputePipelineStateWithFunction:fn error:&pipelineError];
        [fn release];
        if (!pso) {
            fprintf(stderr, "msplat: failed to create pipeline for %s: %s\n",
                    name.UTF8String,
                    pipelineError ? pipelineError.localizedDescription.UTF8String : "unknown error");
            if (!pipelineLoadFailed) {
                pipelineFailure = "msplat: failed to create pipeline for ";
                pipelineFailure += name.UTF8String;
                const char* description = pipelineError.localizedDescription.UTF8String;
                if (description) {
                    pipelineFailure += ": ";
                    pipelineFailure += description;
                }
            }
            pipelineLoadFailed = true;
        }
        return pso;
    };

    auto loadBoolSpecialization = [&](NSString* name, NSUInteger index,
                                      bool value)
        -> id<MTLComputePipelineState> {
        MTLFunctionConstantValues* constants =
            [MTLFunctionConstantValues new];
        if (!constants) {
            throw std::bad_alloc();
        }
        [constants setConstantValue:&value type:MTLDataTypeBool atIndex:index];
        NSError* functionError = nil;
        id<MTLFunction> fn = [metal_library
            newFunctionWithName:name
            constantValues:constants
            error:&functionError];
        [constants release];
        if (!fn) {
            const char* description = functionError
                ? functionError.localizedDescription.UTF8String
                : "unknown error";
            fprintf(stderr, "msplat: failed to specialize kernel %s: %s\n",
                    name.UTF8String, description);
            if (!pipelineLoadFailed) {
                pipelineFailure = "msplat: failed to specialize kernel ";
                pipelineFailure += name.UTF8String;
                if (description) {
                    pipelineFailure += ": ";
                    pipelineFailure += description;
                }
            }
            pipelineLoadFailed = true;
            return nil;
        }
        NSError* pipelineError = nil;
        id<MTLComputePipelineState> pso =
            [ctx->device newComputePipelineStateWithFunction:fn
                                                       error:&pipelineError];
        [fn release];
        if (!pso) {
            const char* description = pipelineError
                ? pipelineError.localizedDescription.UTF8String
                : "unknown error";
            fprintf(stderr, "msplat: failed to create pipeline for %s: %s\n",
                    name.UTF8String, description);
            if (!pipelineLoadFailed) {
                pipelineFailure = "msplat: failed to create pipeline for ";
                pipelineFailure += name.UTF8String;
                if (description) {
                    pipelineFailure += ": ";
                    pipelineFailure += description;
                }
            }
            pipelineLoadFailed = true;
        }
        return pso;
    };

    const char* tileCountModeOverride =
        std::getenv("MSPLAT_TILE_COUNT_MODE");
    if (tileCountModeOverride) {
        if (std::strcmp(tileCountModeOverride, "enumerated") == 0) {
            // Keep the established per-tile atomic enumeration path.
        } else if (std::strcmp(tileCountModeOverride, "difference") == 0) {
            ctx->difference_tile_counting = true;
        } else {
            throw std::invalid_argument(
                "msplat: MSPLAT_TILE_COUNT_MODE must be enumerated or difference");
        }
    }

    const char* tileLayoutModeOverride =
        std::getenv("MSPLAT_TILE_LAYOUT_MODE");
    if (tileLayoutModeOverride) {
        if (std::strcmp(tileLayoutModeOverride, "cpu") == 0) {
            // Keep the established synchronized CPU prefix and classification.
        } else if (std::strcmp(tileLayoutModeOverride, "gpu") == 0) {
            ctx->gpu_tile_layout = true;
        } else {
            throw std::invalid_argument(
                "msplat: MSPLAT_TILE_LAYOUT_MODE must be cpu or gpu");
        }
    }

    const char* trainingArenaModeOverride =
        std::getenv("MSPLAT_TRAINING_ARENA_MODE");
    if (trainingArenaModeOverride) {
        if (std::strcmp(trainingArenaModeOverride, "exact") == 0) {
            // Keep the established synchronized exact-sizing path.
        } else if (std::strcmp(trainingArenaModeOverride, "retry") == 0) {
            ctx->retry_intersection_attempts = true;
            // Retry mode depends on GPU-resident offsets and classification even
            // when the standalone layout A/B override was not supplied.
            ctx->gpu_tile_layout = true;
        } else {
            throw std::invalid_argument(
                "msplat: MSPLAT_TRAINING_ARENA_MODE must be exact or retry");
        }
    }
    const char* ssimModeOverride = std::getenv("MSPLAT_SSIM_MODE");
    if (ssimModeOverride) {
        if (std::strcmp(ssimModeOverride, "staged") == 0) {
            ctx->fused_ssim_backward = false;
        } else if (std::strcmp(ssimModeOverride, "fused") == 0) {
            ctx->fused_ssim_backward = true;
        } else {
            throw std::invalid_argument(
                "msplat: MSPLAT_SSIM_MODE must be staged or fused");
        }
    }

    const char* intersectionAttributesOverride =
        std::getenv("MSPLAT_INTERSECTION_ATTRIBUTES");
    if (intersectionAttributesOverride) {
        if (std::strcmp(intersectionAttributesOverride, "packed") == 0) {
            ctx->gather_intersection_attributes = false;
        } else if (std::strcmp(
                       intersectionAttributesOverride, "gather") == 0) {
            ctx->gather_intersection_attributes = true;
        } else {
            throw std::invalid_argument(
                "msplat: MSPLAT_INTERSECTION_ATTRIBUTES must be packed or gather");
        }
    }

    const char* densifyRandomModeOverride =
        std::getenv("MSPLAT_DENSIFY_RANDOM_MODE");
    if (densifyRandomModeOverride) {
        if (std::strcmp(densifyRandomModeOverride, "cpu") == 0) {
            // Preserve the established libc++ normal-distribution stream.
        } else if (std::strcmp(densifyRandomModeOverride, "gpu") == 0) {
            ctx->gpu_densify_random = true;
        } else {
            throw std::invalid_argument(
                "msplat: MSPLAT_DENSIFY_RANDOM_MODE must be cpu or gpu");
        }
    }

    const char* rasterVariantOverride = std::getenv("MSPLAT_RASTER_VARIANT");
    NSString* monolithicRasterFunctionName = @"nd_rasterize_forward_kernel";
    if (rasterVariantOverride) {
        if (std::strcmp(rasterVariantOverride, "8x8") == 0) {
            // Keep the default function and dimensions.
        } else if (std::strcmp(rasterVariantOverride, "16x8") == 0) {
            monolithicRasterFunctionName = @"nd_rasterize_forward_16x8_kernel";
            ctx->monolithic_raster_block_x = 16;
            ctx->monolithic_raster_block_y = 8;
        } else if (std::strcmp(rasterVariantOverride, "16x16") == 0) {
            monolithicRasterFunctionName = @"nd_rasterize_forward_16x16_kernel";
            ctx->monolithic_raster_block_x = 16;
            ctx->monolithic_raster_block_y = 16;
        } else {
            throw std::invalid_argument(
                "msplat: MSPLAT_RASTER_VARIANT must be 8x8, 16x8, or 16x16");
        }
    }

    // Forward pipeline
    ctx->project_and_sh_forward_kernel_cpso       = load(@"project_and_sh_forward_kernel");
    if (ctx->difference_tile_counting) {
        ctx->tile_count_diff_horizontal_kernel_cpso =
            load(@"tile_count_diff_horizontal_kernel");
        ctx->tile_count_diff_vertical_kernel_cpso =
            load(@"tile_count_diff_vertical_kernel");
    }
    if (ctx->gpu_tile_layout) {
        ctx->build_tile_intersection_layout_kernel_cpso =
            load(@"build_tile_intersection_layout_kernel");
    }
    if (ctx->retry_intersection_attempts) {
        ctx->validate_tile_intersection_attempt_kernel_cpso =
            load(@"validate_tile_intersection_attempt_kernel");
        ctx->finalize_tile_intersection_attempt_kernel_cpso =
            load(@"finalize_tile_intersection_attempt_kernel");
    }
    ctx->nd_rasterize_forward_kernel_cpso         = load(monolithicRasterFunctionName);
    ctx->float_rgb_to_preview_texture_kernel_cpso = load(@"float_rgb_to_preview_texture_kernel");
    // Exact tile-intersection compaction and sorting
    ctx->scatter_to_exact_bins_kernel_cpso        = load(@"scatter_to_exact_bins_kernel");
    ctx->small_sort_per_tile_kernel_cpso          = load(@"small_sort_per_tile_kernel");
    ctx->radix_sort_per_tile_kernel_cpso          = load(@"radix_sort_per_tile_kernel");
    ctx->pack_sorted_gaussians_kernel_cpso        = load(@"pack_sorted_gaussians_kernel");
    ctx->block_reduce_kernel_cpso                 = load(@"block_reduce_kernel");
    ctx->block_scan_propagate_kernel_cpso         = load(@"block_scan_propagate_kernel");
    // Depth-chunked rasterization
    ctx->rasterize_forward_chunked_kernel_cpso    = load(@"rasterize_forward_chunked_kernel");
    ctx->rasterize_forward_merge_kernel_cpso      = load(@"rasterize_forward_merge_kernel");
    ctx->compute_chunk_prefix_suffix_kernel_cpso  = load(@"compute_chunk_prefix_suffix_kernel");
    ctx->rasterize_backward_chunked_kernel_cpso   = load(@"rasterize_backward_chunked_kernel");
    ctx->rasterize_backward_kernel_cpso           = load(@"rasterize_backward_kernel");
    // Separable SSIM loss
    ctx->ssim_h_fwd_kernel_cpso                   = load(@"ssim_h_fwd_kernel");
    ctx->ssim_v_fwd_kernel_cpso                   = load(@"ssim_v_fwd_kernel");
    if (ctx->fused_ssim_backward) {
        ctx->ssim_fused_v_fwd_bwd_kernel_cpso =
            load(@"ssim_fused_v_fwd_bwd_kernel");
    } else {
        ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso =
            load(@"ssim_fused_v_fwd_h_bwd_kernel");
        ctx->ssim_v_bwd_kernel_cpso = load(@"ssim_v_bwd_kernel");
    }
    ctx->photometric_adam_kernel_cpso              = load(@"photometric_adam_kernel");
    ctx->prepare_camera_pose_kernel_cpso            = load(@"prepare_camera_pose_kernel");
    ctx->camera_pose_adam_kernel_cpso               = load(@"camera_pose_adam_kernel");
    // Backward pipeline
    ctx->sh_opacity_backward_adam_kernel_cpso     = load(@"sh_opacity_backward_adam_kernel");
    ctx->project_backward_adam_kernel_cpso        = load(@"project_backward_adam_kernel");
    ctx->reset_opacity_state_kernel_cpso          = load(@"reset_opacity_state_kernel");
    // GPU densification
    ctx->densify_classify_kernel_cpso             = load(@"densify_classify_kernel");
    ctx->densify_append_split_kernel_cpso = loadBoolSpecialization(
        @"densify_append_split_kernel", 0, ctx->gpu_densify_random);
    ctx->densify_append_dup_kernel_cpso           = load(@"densify_append_dup_kernel");
    ctx->densify_cull_classify_kernel_cpso        = load(@"densify_cull_classify_kernel");
    ctx->compact_scatter_kernel_cpso              = load(@"compact_scatter_kernel");
    ctx->compact_copy_back_kernel_cpso            = load(@"compact_copy_back_kernel");

    if (pipelineLoadFailed) {
        throw std::runtime_error(pipelineFailure);
    }

    const NSUInteger monolithicRasterThreads =
        static_cast<NSUInteger>(ctx->monolithic_raster_block_x) *
        static_cast<NSUInteger>(ctx->monolithic_raster_block_y);
    const NSUInteger maximumMonolithicRasterThreads =
        ctx->nd_rasterize_forward_kernel_cpso.maxTotalThreadsPerThreadgroup;
    if (maximumMonolithicRasterThreads < monolithicRasterThreads) {
        throw std::runtime_error(
            "msplat: selected monolithic raster variant exceeds the Metal "
            "pipeline threadgroup limit");
    }
    if (ctx->fused_ssim_backward) {
        constexpr NSUInteger fusedSsimThreads = 16u * 8u;
        const id<MTLComputePipelineState> fusedSsimPipeline =
            ctx->ssim_fused_v_fwd_bwd_kernel_cpso;
        if (fusedSsimPipeline.maxTotalThreadsPerThreadgroup <
            fusedSsimThreads) {
            throw std::runtime_error(
                "msplat: fused SSIM requires 128 threads per threadgroup");
        }
        if (fusedSsimPipeline.staticThreadgroupMemoryLength >
            ctx->device.maxThreadgroupMemoryLength) {
            throw std::runtime_error(
                "msplat: fused SSIM exceeds the device threadgroup-memory limit");
        }
    }
    if (rasterVariantOverride) {
        fprintf(stderr,
                "msplat: MSPLAT_RASTER_VARIANT=%s (monolithic forward only)\n",
                rasterVariantOverride);
    }
    if (tileCountModeOverride) {
        fprintf(stderr, "msplat: MSPLAT_TILE_COUNT_MODE=%s\n",
                tileCountModeOverride);
    }
    if (tileLayoutModeOverride) {
        fprintf(stderr, "msplat: MSPLAT_TILE_LAYOUT_MODE=%s\n",
                tileLayoutModeOverride);
    }
    if (trainingArenaModeOverride) {
        fprintf(stderr, "msplat: MSPLAT_TRAINING_ARENA_MODE=%s\n",
                trainingArenaModeOverride);
    }
    if (ssimModeOverride) {
        fprintf(stderr, "msplat: MSPLAT_SSIM_MODE=%s\n", ssimModeOverride);
    }
    if (intersectionAttributesOverride) {
        fprintf(stderr, "msplat: MSPLAT_INTERSECTION_ATTRIBUTES=%s\n",
                intersectionAttributesOverride);
    }
    if (densifyRandomModeOverride) {
        fprintf(stderr, "msplat: MSPLAT_DENSIFY_RANDOM_MODE=%s\n",
                densifyRandomModeOverride);
    }
    if (ctx->small_sort_per_tile_kernel_cpso.maxTotalThreadsPerThreadgroup <
        msplat::kExactSmallTileMaximum) {
        throw std::runtime_error(
            "msplat: exact small-tile sort requires 32 threads per threadgroup");
    }
    if (ctx->radix_sort_per_tile_kernel_cpso.maxTotalThreadsPerThreadgroup <
        256) {
        throw std::runtime_error(
            "msplat: exact radix sort requires 256 threads per threadgroup");
    }

    // Initialize counter sampling if PROFILE_STAGES is set
    ctx->timestampCounterSet = nil;
    ctx->counterSamplingAvailable = false;
    ctx->ticksToMs = 0.0;
    if (std::getenv("PROFILE_STAGES")) {
        g_profile_stages = true;
        g_profile_stages_checked = true;
        ctx->initCounterSampling();
    }

    configuration.succeeded = true;
    return ctx.release();
}

struct GlobalMetalContext {
    std::unique_ptr<MetalContext> context;

    GlobalMetalContext() : context(init_msplat_metal_context()) {
        g_context_instance.store(context.get(), std::memory_order_release);
    }

    ~GlobalMetalContext() {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        g_context_instance.store(nullptr, std::memory_order_release);
        context.reset();
    }
};

MetalContext* get_global_context() {
    static GlobalMetalContext globalContext;
    return globalContext.context.get();
}



#define ENC_SCALAR(encoder, x, i) [encoder setBytes:&x length:sizeof(x) atIndex:i]
#define ENC_ARRAY(encoder, x, i) [encoder setBytes:x length:sizeof(x) atIndex:i]
#define ENC_BUF(encoder, x, i) [encoder setBuffer:x.buffer() offset:0 atIndex:i]

static void encode_tile_count_difference_scan(
    MetalContext* ctx,
    id<MTLComputeCommandEncoder> encoder,
    MTensor& tileCountDiff,
    MTensor& tileCounts,
    const uint32_t* tileBounds,
    const MTensor& coverageRenderTiles,
    uint32_t coverageRenderTileStride
) {
    if (!ctx->difference_tile_counting) return;

    const NSUInteger horizontalThreadCount =
        static_cast<NSUInteger>(tileBounds[1]) + 1;
    const NSUInteger horizontalThreadsPerGroup = MIN(
        ctx->tile_count_diff_horizontal_kernel_cpso
            .maxTotalThreadsPerThreadgroup,
        horizontalThreadCount);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:
        ctx->tile_count_diff_horizontal_kernel_cpso];
    ENC_BUF(encoder, tileCountDiff, 0);
    [encoder setBytes:tileBounds length:4 * sizeof(uint32_t) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(horizontalThreadCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(
            horizontalThreadsPerGroup, 1, 1)];

    const NSUInteger verticalThreadCount =
        static_cast<NSUInteger>(tileBounds[0]);
    const NSUInteger verticalThreadsPerGroup = MIN(
        ctx->tile_count_diff_vertical_kernel_cpso
            .maxTotalThreadsPerThreadgroup,
        verticalThreadCount);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:
        ctx->tile_count_diff_vertical_kernel_cpso];
    ENC_BUF(encoder, tileCountDiff, 0);
    ENC_BUF(encoder, tileCounts, 1);
    [encoder setBytes:tileBounds length:4 * sizeof(uint32_t) atIndex:2];
    ENC_BUF(encoder, coverageRenderTiles, 3);
    ENC_SCALAR(encoder, coverageRenderTileStride, 4);
    [encoder dispatchThreads:MTLSizeMake(verticalThreadCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(verticalThreadsPerGroup, 1, 1)];
}

id<MTLDevice> msplat_device() {
    return get_global_context()->device;
}

static void encode_gpu_tile_intersection_layout(
    MetalContext* ctx,
    id<MTLComputeCommandEncoder> encoder,
    MTensor& tileCounts,
    MTensor& tileOffsets,
    MTensor& tileBins,
    MTensor& sortableTileIndices,
    MTensor& metadata,
    uint32_t numTiles
) {
    if (!ctx->gpu_tile_layout) return;

    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:
        ctx->build_tile_intersection_layout_kernel_cpso];
    ENC_BUF(encoder, tileCounts, 0);
    ENC_BUF(encoder, tileOffsets, 1);
    ENC_BUF(encoder, tileBins, 2);
    ENC_BUF(encoder, sortableTileIndices, 3);
    ENC_BUF(encoder, metadata, 4);
    ENC_SCALAR(encoder, numTiles, 5);
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

static void encode_validate_tile_intersection_attempt(
    MetalContext* ctx,
    id<MTLComputeCommandEncoder> encoder,
    MTensor& tileBins,
    uint32_t numTiles,
    uint32_t arenaCapacity,
    uint32_t radixScratchAvailable,
    uint32_t plannedChunkCount,
    uint32_t packThreadsPerGroup,
    MTensor& attemptStatus,
    MTensor& layoutMetadata,
    MTensor& dispatchControl,
    MTensor& tileOffsets
) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:
        ctx->validate_tile_intersection_attempt_kernel_cpso];
    ENC_BUF(encoder, layoutMetadata, 0);
    ENC_BUF(encoder, tileBins, 1);
    ENC_SCALAR(encoder, numTiles, 2);
    ENC_SCALAR(encoder, arenaCapacity, 3);
    ENC_SCALAR(encoder, radixScratchAvailable, 4);
    ENC_SCALAR(encoder, plannedChunkCount, 5);
    ENC_BUF(encoder, attemptStatus, 6);
    ENC_BUF(encoder, dispatchControl, 7);
    ENC_SCALAR(encoder, packThreadsPerGroup, 8);
    ENC_BUF(encoder, tileOffsets, 9);
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype) {
    return mtensor_zeros(get_global_context()->device, std::move(shape), dtype);
}

MTensor gpu_empty(std::vector<int64_t> shape, DType dtype) {
    return mtensor_empty(get_global_context()->device, std::move(shape), dtype);
}

bool msplat_densify_uses_gpu_random() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    return get_global_context()->gpu_densify_random;
}

MsplatTrainingTelemetryHandle msplat_training_telemetry_create() {
    return std::make_shared<MsplatTrainingTelemetryState>();
}

void msplat_training_telemetry_reset(
    const MsplatTrainingTelemetryHandle& telemetry) {
    if (!telemetry)
        throw std::invalid_argument("Training telemetry state must not be null");
    telemetry->reset();
}

MsplatTrainingTelemetrySnapshot msplat_training_telemetry_snapshot(
    const MsplatTrainingTelemetryHandle& telemetry) {
    if (!telemetry)
        throw std::invalid_argument("Training telemetry state must not be null");
    return telemetry->readSnapshot();
}

size_t msplat_training_telemetry_readback_bytes(
    const MsplatTrainingTelemetryHandle& telemetry) {
    if (!telemetry) return 0;
    return telemetry->readbackBytes();
}

MsplatLogicalTrainingStepHandle msplat_training_step_begin(
    const MsplatTrainingTelemetryHandle& telemetry, int64_t iteration) {
    if (!telemetry)
        throw std::invalid_argument("Training telemetry state must not be null");
    if (iteration <= 0)
        throw std::invalid_argument("Training telemetry iteration must be positive");

    const auto wallStart = TelemetryClock::now();
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    MetalContext* context = get_global_context();
    if (context->activeTrainingStep)
        throw std::logic_error("A logical training step is already active");

    MTensor readback = telemetry->acquireReadback(context->device);
    const auto imagePrepareStart = TelemetryClock::now();
    auto step = std::make_shared<MsplatLogicalTrainingStep>(
        telemetry, telemetry->currentGeneration(), iteration, wallStart,
        imagePrepareStart, std::move(readback));
    context->activeTrainingStep = step;
    return step;
}

void msplat_training_step_mark_cpu_start(
    const MsplatLogicalTrainingStepHandle& step) {
    if (!step)
        throw std::invalid_argument("Logical training step must not be null");
    step->markCpuStart();
}

double msplat_training_step_submit(
    const MsplatLogicalTrainingStepHandle& step,
    const MsplatTrainingStepDescriptor& descriptor) {
    if (!step)
        throw std::invalid_argument("Logical training step must not be null");

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    MetalContext* context = get_global_context();
    if (context->activeTrainingStep != step)
        throw std::logic_error("Logical training step is not active");
    step->validateForSubmit(descriptor);

    if (!g_gpu_timing_checked) {
        g_gpu_timing_enabled = std::getenv("PROFILE_GPU") != nullptr;
        g_gpu_timing_checked = true;
    }
    context->commitCB();
    const auto cpuEnd = TelemetryClock::now();
    context->activeTrainingStep.reset();
    return step->seal(descriptor, cpuEnd);
}

void msplat_training_step_abort(
    const MsplatLogicalTrainingStepHandle& step) noexcept {
    if (!step) return;
    try {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        MetalContext* context = get_global_context();
        if (context->activeTrainingStep == step) {
            if (context->currentCBTrainingStep == step)
                context->discardCB();
            context->activeTrainingStep.reset();
        }
    } catch (...) {
        // Abort is called while propagating another error.
    }
    step->abort();
}

void msplat_commit() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    MetalContext* context = get_global_context();
    if (context->activeTrainingStep) {
        throw std::logic_error(
            "Use msplat_training_step_submit for an active logical step");
    }
    if (!g_gpu_timing_checked) {
        g_gpu_timing_enabled = std::getenv("PROFILE_GPU") != nullptr;
        g_gpu_timing_checked = true;
    }
    context->commitCB();
}

void msplat_gpu_sync() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    get_global_context()->syncCB();
}

void msplat_submit_preview_texture(
    MTensor& rgb, void* texture, MsplatPreviewCompletion completion) {
    if (!completion)
        throw std::invalid_argument("Preview completion must not be empty");
    if (!rgb.isGpu() || rgb.dtype() != DType::Float32 || rgb.ndim() != 3 ||
        rgb.size(0) <= 0 || rgb.size(1) <= 0 || rgb.size(2) != 3) {
        throw std::invalid_argument(
            "Preview input must be a non-empty GPU Float32 [H,W,3] tensor");
    }
    if (!texture)
        throw std::invalid_argument("Preview texture must not be null");

    id<MTLTexture> output = (__bridge id<MTLTexture>)texture;
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    MetalContext* context = get_global_context();
    if (context->activeTrainingStep)
        throw std::logic_error(
            "Preview submission cannot join an active logical training step");
    if (output.device.registryID != context->device.registryID)
        throw std::invalid_argument(
            "Preview texture must use the msplat Metal device");
    if (output.textureType != MTLTextureType2D || output.arrayLength != 1 ||
        output.sampleCount != 1 || output.depth != 1) {
        throw std::invalid_argument(
            "Preview texture must be a non-multisampled 2D texture");
    }
    if (output.pixelFormat != MTLPixelFormatBGRA8Unorm)
        throw std::invalid_argument(
            "Preview texture must use BGRA8Unorm (non-sRGB)");
    const MTLTextureUsage requiredUsage =
        MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    if ((output.usage & requiredUsage) != requiredUsage)
        throw std::invalid_argument(
            "Preview texture must allow shader read and write access");

    const NSUInteger width = static_cast<NSUInteger>(rgb.size(1));
    const NSUInteger height = static_cast<NSUInteger>(rgb.size(0));
    if (output.width != width || output.height != height)
        throw std::invalid_argument(
            "Preview texture dimensions must match the rendered image");

    id<MTLCommandBuffer> commandBuffer = context->getCommandBuffer();
    __block const char* encodingFailure = nullptr;
    const std::array<uint32_t, 2> imageSize = {
        static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    dispatch_sync(context->d_queue, ^{
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (!encoder) {
            encodingFailure =
                "msplat: failed to create the preview conversion encoder";
            return;
        }
        [encoder setComputePipelineState:
            context->float_rgb_to_preview_texture_kernel_cpso];
        [encoder setBuffer:rgb.buffer() offset:0 atIndex:0];
        [encoder setTexture:output atIndex:0];
        [encoder setBytes:imageSize.data()
                    length:sizeof(imageSize)
                   atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(width, height, 1)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    });
    if (encodingFailure) {
        context->discardCB();
        throw std::runtime_error(encodingFailure);
    }

    id<MTLCommandBuffer> rootCommandBuffer =
        context->_currentCB.rootCommandBuffer;
    if (!rootCommandBuffer) {
        context->discardCB();
        throw std::runtime_error(
            "msplat: preview command buffer has no root command buffer");
    }
    auto completionHolder =
        std::make_shared<MsplatPreviewCompletion>(std::move(completion));
    [rootCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        @autoreleasepool {
            try {
                if (completed.status == MTLCommandBufferStatusCompleted) {
                    (*completionHolder)(true, nullptr);
                    return;
                }
                NSError* error = completed.error;
                const char* description = error.localizedDescription.UTF8String;
                std::string message =
                    "msplat: preview command buffer failed";
                if (description) {
                    message += ": ";
                    message += description;
                }
                (*completionHolder)(false, message.c_str());
            } catch (...) {
                // Even allocation pressure while formatting an NSError must
                // transition the frame out of Pending.
                try {
                    (*completionHolder)(
                        false, "msplat: preview command buffer failed");
                } catch (...) {
                    // A C++ exception must never escape a Metal callback.
                }
            }
        }
    }];
    try {
        context->commitCB();
    } catch (...) {
        context->discardCB();
        throw;
    }
}

void msplat_enable_gpu_timing(bool enable) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    g_gpu_timing_enabled = enable;
    g_gpu_timing_checked = true;
}

void msplat_drain_gpu_times(std::vector<double>& out) {
    std::lock_guard<std::mutex> lock(g_gpu_timing_mutex);
    out = std::move(g_gpu_times_ms);
    g_gpu_times_ms.clear();
}

void msplat_drain_stage_times(std::vector<double> stage_times[], int max_stages, int& n_stages,
                              const char** stage_names) {
    std::lock_guard<std::mutex> lock(g_stage_timing_mutex);
    n_stages = std::min(max_stages, N_TRAIN_STAGES);
    for (int i = 0; i < n_stages; i++) {
        stage_times[i] = std::move(g_stage_times[i]);
        g_stage_times[i].clear();
        stage_names[i] = trainingStageName(i);
    }
}

#define RAST_BLOCK_X 8
#define RAST_BLOCK_Y 8

// Cached buffer pool — all intermediate GPU buffers are reused across iterations.
// Sizes only change at densification (every 100 steps); between densifications
// this eliminates all per-iteration GPU allocations.
struct FusedTensorCache {
    // -1 means "this group holds nothing usable" — the state a group is left in
    // when one of its allocations fails partway through, so the next call
    // rebuilds it instead of binding buffers that were never allocated.
    int fwd_num_points = -1, shared_img_height = -1, shared_img_width = -1;
    int num_tiles = -1;
    int tile_count_diff_width = -1, tile_count_diff_height = -1;
    int training_img_height = -1, training_img_width = -1;
    int bwd_num_points = -1;
    int64_t capacity = -1;

    // Forward intermediates
    MTensor xys, depths, radii_out, conics, colors, aabb;
    MTensor projected_opacities;
    MTensor packed_xy_opac, packed_conic, packed_rgb;
    MTensor out_img, final_Ts, final_idx;
    MTensor tile_bins;

    // Exact tile-intersection layout and sorting buffers
    MTensor tile_offsets, tile_scatter_counters, sortable_tile_indices;
    MTensor tile_count_diff;
    MTensor tile_layout_metadata;
    MTensor tile_attempt_dispatch_control;
    MTensor intersection_keys_a, intersection_keys_b;

    // Defensive invariant check: exact sizing should make this remain zero.
    MTensor overflow_flag;

    // Shared forward depth-chunked rasterization buffers
    int forward_chunk_K_max = -1, forward_chunk_height = -1, forward_chunk_width = -1;
    MTensor chunk_T, chunk_C, chunk_final_idx;

    // Training-only image and backward depth-chunked buffers
    MTensor ssim_deriv_h_buf, ssim_h_buf, loss_sum;
    MTensor photometric_gradient;
    MTensor pose_viewmat, pose_cam_pos, pose_gradient;
    int backward_chunk_K_max = -1, backward_chunk_height = -1, backward_chunk_width = -1;
    MTensor prefix_T, after_C;

    // Training-only backward gradient accumulators
    MTensor v_xy, v_conic, v_colors_rast, v_opacity;

    size_t sharedEstimatedBytes() const {
        const MTensor* tensors[] = {
            &xys, &depths, &radii_out, &conics, &colors, &aabb,
            &projected_opacities,
            &packed_xy_opac, &packed_conic, &packed_rgb,
            &out_img, &final_Ts, &final_idx,
            &tile_bins, &tile_offsets, &tile_scatter_counters,
            &sortable_tile_indices, &tile_count_diff, &tile_layout_metadata,
            &tile_attempt_dispatch_control,
            &intersection_keys_a, &intersection_keys_b, &overflow_flag,
            &chunk_T, &chunk_C, &chunk_final_idx
        };
        size_t bytes = 0;
        for (const MTensor* tensor : tensors) {
            if (tensor->defined()) bytes += tensor->nbytes();
        }
        return bytes;
    }

    size_t trainingEstimatedBytes() const {
        const MTensor* tensors[] = {
            &ssim_deriv_h_buf, &ssim_h_buf, &loss_sum,
            &photometric_gradient, &pose_viewmat, &pose_cam_pos,
            &pose_gradient,
            &prefix_T, &after_C,
            &v_xy, &v_conic, &v_colors_rast, &v_opacity
        };
        size_t bytes = 0;
        for (const MTensor* tensor : tensors) {
            if (tensor->defined()) bytes += tensor->nbytes();
        }
        return bytes;
    }

    size_t estimatedBytes() const {
        return sharedEstimatedBytes() + trainingEstimatedBytes();
    }

    // Each group releases the previous generation before allocating the new one.
    // At the last resolution step-up the new set is roughly 4x the old, so
    // holding both alive across the transition would raise the peak above what
    // the device can hand back. The size tracker is invalidated first, so an
    // allocation that throws partway leaves the group marked empty rather than
    // leaving stale sizes claiming buffers that no longer exist.
    void resetIntersectionArena() {
        capacity = -1;
        intersection_keys_a.reset(); intersection_keys_b.reset();
        packed_xy_opac.reset();
        packed_conic.reset(); packed_rgb.reset();
    }

    void resetTileCountDifference() {
        tile_count_diff_width = -1;
        tile_count_diff_height = -1;
        tile_count_diff.reset();
    }

    void ensure_tile_count_diff(int tileWidth, int tileHeight,
                                bool enabled, id<MTLDevice> dev) {
        if (!enabled) {
            resetTileCountDifference();
            return;
        }
        if (tileWidth == tile_count_diff_width &&
            tileHeight == tile_count_diff_height &&
            tile_count_diff.defined()) {
            return;
        }
        if (tileWidth <= 0 || tileHeight <= 0) {
            throw std::invalid_argument(
                "Tile-count difference dimensions must be positive");
        }

        resetTileCountDifference();
        try {
            tile_count_diff = mtensor_empty(
                dev,
                {static_cast<int64_t>(tileHeight) + 1,
                 static_cast<int64_t>(tileWidth) + 1},
                DType::Int32);
        } catch (...) {
            resetTileCountDifference();
            throw;
        }
        tile_count_diff_width = tileWidth;
        tile_count_diff_height = tileHeight;
    }

    void ensure_tile_layout_metadata(bool enabled, id<MTLDevice> dev) {
        if (!enabled) {
            tile_layout_metadata.reset();
            return;
        }
        if (tile_layout_metadata.defined()) return;
        tile_layout_metadata = mtensor_empty(
            dev,
            {static_cast<int64_t>(
                msplat::kTileIntersectionLayoutMetadataWordCount)},
            DType::Int32);
    }

    void ensure_tile_attempt_dispatch_control(bool enabled,
                                              id<MTLDevice> dev) {
        if (!enabled) {
            tile_attempt_dispatch_control.reset();
            return;
        }
        if (tile_attempt_dispatch_control.defined()) return;
        // Three MTLDispatchThreadgroupsIndirectArguments records followed by
        // the small count, general count, and general-list offset scalars.
        tile_attempt_dispatch_control =
            mtensor_empty(dev, {12}, DType::Int32);
    }

    void ensure_shared_forward(int np, int ih, int iw, int nt,
                               id<MTLDevice> dev) {
        const bool resolutionChanged =
            ih != shared_img_height || iw != shared_img_width || nt != num_tiles;
        if (resolutionChanged) resetIntersectionArena();

        if (np != fwd_num_points) {
            fwd_num_points = -1;
            xys.reset(); depths.reset(); radii_out.reset(); conics.reset();
            colors.reset(); aabb.reset(); projected_opacities.reset();

            try {
                xys = mtensor_empty(dev, {np, 2}, DType::Float32);
                depths = mtensor_empty(dev, {np}, DType::Float32);
                radii_out = mtensor_empty(dev, {np}, DType::Int32);
                conics = mtensor_empty(dev, {np, 3}, DType::Float32);
                colors = mtensor_empty(dev, {np, 3}, DType::Float32);
                aabb = mtensor_empty(dev, {np, 2}, DType::Float32);
                projected_opacities =
                    mtensor_empty(dev, {np}, DType::Float32);
            } catch (...) {
                xys.reset(); depths.reset(); radii_out.reset(); conics.reset();
                colors.reset(); aabb.reset(); projected_opacities.reset();
                throw;
            }
            fwd_num_points = np;
        }
        if (ih != shared_img_height || iw != shared_img_width) {
            shared_img_height = -1; shared_img_width = -1;
            out_img.reset(); final_Ts.reset(); final_idx.reset();

            try {
                out_img = mtensor_empty(dev, {ih, iw, 3}, DType::Float32);
                final_Ts = mtensor_empty(dev, {ih, iw}, DType::Float32);
                final_idx = mtensor_empty(dev, {ih, iw}, DType::Int32);
            } catch (...) {
                out_img.reset(); final_Ts.reset(); final_idx.reset();
                throw;
            }
            shared_img_height = ih; shared_img_width = iw;
        }
        if (nt != num_tiles) {
            num_tiles = -1;
            tile_bins.reset(); tile_offsets.reset();
            tile_scatter_counters.reset();
            sortable_tile_indices.reset();

            try {
                tile_bins = mtensor_empty(dev, {nt, 2}, DType::Int32);
                tile_offsets = mtensor_empty(dev, {nt}, DType::Int32);
                tile_scatter_counters =
                    mtensor_empty(dev, {nt}, DType::Int32);
                sortable_tile_indices =
                    mtensor_empty(dev, {nt}, DType::Int32);
            } catch (...) {
                tile_bins.reset(); tile_offsets.reset();
                tile_scatter_counters.reset();
                sortable_tile_indices.reset();
                throw;
            }
            num_tiles = nt;
        }
        if (!overflow_flag.defined()) {
            overflow_flag = mtensor_empty(dev, {1}, DType::Int32);
        }
    }

    bool ensure_intersection_arena(uint32_t requiredCount,
                                   bool needsRadixScratch,
                                   bool needsPackedAttributes,
                                   id<MTLDevice> dev) {
        if (!needsPackedAttributes) {
            packed_xy_opac.reset();
            packed_conic.reset();
            packed_rgb.reset();
        }
        const uint32_t currentCapacity = capacity > 0
            ? static_cast<uint32_t>(capacity)
            : 0;
        const uint32_t requestedCapacity =
            msplat::tileIntersectionArenaCapacity(
                requiredCount, currentCapacity, false);
        const bool packedAttributesReady = !needsPackedAttributes ||
            (packed_xy_opac.defined() && packed_conic.defined() &&
             packed_rgb.defined());
        const bool baseArenaReady = requestedCapacity == currentCapacity &&
            intersection_keys_a.defined() && packedAttributesReady;
        if (baseArenaReady) {
            if (!needsRadixScratch || intersection_keys_b.defined()) {
                return false;
            }
            intersection_keys_b =
                mtensor_empty(dev, {capacity}, DType::Int64);
            return true;
        }

        const uint64_t bytesPerLargestBuffer = needsPackedAttributes
            ? 3u * sizeof(float)
            : sizeof(uint64_t);
        const uint64_t largestBufferBytes =
            static_cast<uint64_t>(requestedCapacity) * bytesPerLargestBuffer;
        if (largestBufferBytes > static_cast<uint64_t>(dev.maxBufferLength)) {
            throw std::length_error(
                "Exact tile-intersection arena exceeds Metal's maximum buffer length");
        }

        resetIntersectionArena();
        const int64_t cap = requestedCapacity;
        try {
            intersection_keys_a = mtensor_empty(dev, {cap}, DType::Int64);
            if (needsPackedAttributes) {
                packed_xy_opac =
                    mtensor_empty(dev, {cap, 3}, DType::Float32);
                packed_conic =
                    mtensor_empty(dev, {cap, 3}, DType::Float32);
                packed_rgb =
                    mtensor_empty(dev, {cap, 3}, DType::Float32);
            }
            if (needsRadixScratch) {
                intersection_keys_b =
                    mtensor_empty(dev, {cap}, DType::Int64);
            }
        } catch (...) {
            resetIntersectionArena();
            throw;
        }
        capacity = cap;
        return true;
    }

    void ensure_training_image(int ih, int iw, bool needsDerivativeBuffer,
                               id<MTLDevice> dev) {
        const bool derivativeBufferReady =
            !needsDerivativeBuffer || ssim_deriv_h_buf.defined();
        if (ih == training_img_height && iw == training_img_width &&
            derivativeBufferReady && ssim_h_buf.defined() &&
            loss_sum.defined() && photometric_gradient.defined()) {
            return;
        }

        training_img_height = -1; training_img_width = -1;
        ssim_deriv_h_buf.reset(); ssim_h_buf.reset();
        loss_sum.reset(); photometric_gradient.reset();

        if (needsDerivativeBuffer) {
            ssim_deriv_h_buf = mtensor_empty(
                dev, {(int64_t)ih, (int64_t)iw, 9}, DType::Float32);
        }
        ssim_h_buf = mtensor_empty(
            dev, {(int64_t)ih, (int64_t)iw, 15}, DType::Float32);
        loss_sum = mtensor_empty(dev, {1}, DType::Float32);
        photometric_gradient = mtensor_empty(dev, {3}, DType::Float32);
        training_img_height = ih; training_img_width = iw;
    }

    void ensure_pose_refinement(id<MTLDevice> dev) {
        if (pose_viewmat.defined() && pose_cam_pos.defined() &&
            pose_gradient.defined()) {
            return;
        }
        pose_viewmat.reset(); pose_cam_pos.reset(); pose_gradient.reset();
        try {
            pose_viewmat = mtensor_empty(dev, {4, 4}, DType::Float32);
            // Metal constant float3 arguments occupy 16 bytes. Keep the
            // camera-center buffer padded to the shader ABI width.
            pose_cam_pos = mtensor_empty(dev, {4}, DType::Float32);
            pose_gradient = mtensor_empty(dev, {6}, DType::Float32);
        } catch (...) {
            pose_viewmat.reset(); pose_cam_pos.reset(); pose_gradient.reset();
            throw;
        }
    }

    void ensure_forward_chunks(int K, int ih, int iw, id<MTLDevice> dev) {
        if (K <= 1) {
            // Chunking is off at this resolution. Whatever the previous level
            // built can never be reused — on garden that is ~208MB, carried to
            // the end of training — so hand it back. Only on a resolution
            // change: within a level K_max climbs as the model densifies, so a
            // dip to 1 there would just churn the allocation.
            const bool anyDefined = chunk_T.defined() || chunk_C.defined() ||
                chunk_final_idx.defined();
            if (anyDefined &&
                (ih != forward_chunk_height || iw != forward_chunk_width)) {
                forward_chunk_K_max = -1;
                forward_chunk_height = -1; forward_chunk_width = -1;
                chunk_T.reset(); chunk_C.reset(); chunk_final_idx.reset();
            }
            return;
        }
        if (K <= forward_chunk_K_max && ih == forward_chunk_height &&
            iw == forward_chunk_width && chunk_T.defined() &&
            chunk_C.defined() && chunk_final_idx.defined()) {
            return;
        }
        forward_chunk_K_max = -1;
        forward_chunk_height = -1; forward_chunk_width = -1;
        chunk_T.reset(); chunk_C.reset(); chunk_final_idx.reset();

        chunk_T = mtensor_empty(dev, {K, ih, iw}, DType::Float32);
        chunk_C = mtensor_empty(dev, {K, ih, iw, 3}, DType::Float32);
        chunk_final_idx = mtensor_empty(dev, {K, ih, iw}, DType::Int32);
        forward_chunk_K_max = K;
        forward_chunk_height = ih; forward_chunk_width = iw;
    }

    void ensure_backward_chunks(int K, int ih, int iw, id<MTLDevice> dev) {
        if (K <= 1) {
            const bool anyDefined = prefix_T.defined() || after_C.defined();
            if (anyDefined &&
                (ih != backward_chunk_height || iw != backward_chunk_width)) {
                backward_chunk_K_max = -1;
                backward_chunk_height = -1; backward_chunk_width = -1;
                prefix_T.reset(); after_C.reset();
            }
            return;
        }
        if (K <= backward_chunk_K_max && ih == backward_chunk_height &&
            iw == backward_chunk_width && prefix_T.defined() && after_C.defined()) {
            return;
        }
        backward_chunk_K_max = -1;
        backward_chunk_height = -1; backward_chunk_width = -1;
        prefix_T.reset(); after_C.reset();

        prefix_T = mtensor_empty(dev, {K, ih, iw}, DType::Float32);
        after_C = mtensor_empty(dev, {K, ih, iw, 3}, DType::Float32);
        backward_chunk_K_max = K;
        backward_chunk_height = ih; backward_chunk_width = iw;
    }

    void ensure_backward(int np, id<MTLDevice> dev) {
        if (np != bwd_num_points || !v_xy.defined() || !v_conic.defined() ||
            !v_colors_rast.defined() || !v_opacity.defined()) {
            bwd_num_points = -1;
            v_xy.reset(); v_conic.reset(); v_colors_rast.reset(); v_opacity.reset();

            v_xy = mtensor_empty(dev, {np, 2}, DType::Float32);
            v_conic = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_colors_rast = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_opacity = mtensor_empty(dev, {np, 1}, DType::Float32);
            bwd_num_points = np;
        }
    }
};
static FusedTensorCache g_tcache;

static msplat::TileIntersectionLayout completed_gpu_tile_intersection_layout(
    int numTiles) {
    if (numTiles < 0) {
        throw std::invalid_argument("Tile count must not be negative");
    }
    if (!g_tcache.tile_layout_metadata.defined() ||
        g_tcache.tile_layout_metadata.numel() <
            static_cast<int64_t>(
                msplat::kTileIntersectionLayoutMetadataWordCount)) {
        throw std::logic_error(
            "GPU tile-intersection layout metadata is unavailable");
    }
    return msplat::tileIntersectionLayoutFromGpuMetadata(
        g_tcache.tile_layout_metadata.data<uint32_t>(),
        static_cast<size_t>(g_tcache.tile_layout_metadata.numel()),
        static_cast<size_t>(numTiles),
        g_tcache.tile_offsets.data<int32_t>());
}

size_t msplat_cached_tensor_bytes() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    return g_tcache.estimatedBytes();
}

size_t msplat_shared_cached_tensor_bytes() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    return g_tcache.sharedEstimatedBytes();
}

size_t msplat_training_cached_tensor_bytes() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    return g_tcache.trainingEstimatedBytes();
}

size_t msplat_packed_intersection_attribute_bytes() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    size_t bytes = 0;
    const MTensor* tensors[] = {
        &g_tcache.packed_xy_opac,
        &g_tcache.packed_conic,
        &g_tcache.packed_rgb,
    };
    for (const MTensor* tensor : tensors) {
        if (tensor->defined()) bytes += tensor->nbytes();
    }
    return bytes;
}

void cleanup_msplat_metal() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (MetalContext* ctx = g_context_instance.load(std::memory_order_acquire)) {
        auto activeStep = ctx->activeTrainingStep;
        try {
            ctx->syncCB();
        } catch (...) {
            ctx->activeTrainingStep.reset();
            if (activeStep) activeStep->abort();
            g_tcache = FusedTensorCache{};
            throw;
        }
        ctx->activeTrainingStep.reset();
        if (activeStep) activeStep->abort();
    }
    g_tcache = FusedTensorCache{};
}

// Render-only forward pipeline. Training has its own fused forward/backward
// encoder below and lazily allocates the additional buffers it needs.
static void render_pipeline(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background
) {
    MetalContext* ctx = get_global_context();
    int tile_bounds_x = std::get<0>(tile_bounds);
    int tile_bounds_y = std::get<1>(tile_bounds);
    int num_tiles = tile_bounds_x * tile_bounds_y;

    uint32_t channels = 3;

    // Geometry and tile-count buffers are known before projection. The exact
    // intersection arena is sized after the GPU has produced this frame's
    // per-tile counts.
    g_tcache.ensure_shared_forward(
        num_points, img_height, img_width, num_tiles, ctx->device);
    g_tcache.ensure_tile_count_diff(
        tile_bounds_x, tile_bounds_y, ctx->difference_tile_counting,
        ctx->device);
    g_tcache.ensure_tile_layout_metadata(ctx->gpu_tile_layout, ctx->device);
    MTensor& tileCountStorage = ctx->difference_tile_counting
        ? g_tcache.tile_count_diff
        : g_tcache.tile_scatter_counters;
    const uint32_t tileCountMode =
        ctx->difference_tile_counting ? 1u : 0u;
    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &conics = g_tcache.conics;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
    MTensor &projected_opacities = g_tcache.projected_opacities;
    MTensor &tile_bins = g_tcache.tile_bins;
    MTensor &out_img = g_tcache.out_img;
    MTensor &final_Ts = g_tcache.final_Ts;
    MTensor &final_idx = g_tcache.final_idx;

    // --- Constants (heap-allocated for Obj-C block) ---
    auto proj_intrins = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto tile_bounds_arr = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (uint32_t)tile_bounds_x, (uint32_t)tile_bounds_y,
        (uint32_t)std::get<2>(tile_bounds), 0xDEAD
    });
    // Metal constant float3 arguments occupy 16 bytes, despite carrying only
    // three values.
    auto cam_pos_arr = std::make_shared<std::array<float, 4>>(
        std::array<float, 4>{cam_pos[0], cam_pos[1], cam_pos[2], 0.0f});
    uint32_t num_points_u32 = (uint32_t)num_points;
    auto img_size_dim3 = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{img_width, img_height, 1, 0xDEAD});
    auto monolithic_block_size_dim2 =
        std::make_shared<std::array<int32_t, 2>>(
            std::array<int32_t, 2>{
                static_cast<int32_t>(ctx->monolithic_raster_block_x),
                static_cast<int32_t>(ctx->monolithic_raster_block_y)});
    auto chunked_block_size_dim2 =
        std::make_shared<std::array<int32_t, 2>>(
            std::array<int32_t, 2>{RAST_BLOCK_X, RAST_BLOCK_Y});

    // Helper lambdas to encode each stage onto a given encoder
    auto encode_proj_sh = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_forward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_forward_kernel_cpso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_intrins->data() length:sizeof(*proj_intrins) atIndex:7];
        [enc setBytes:proj_img_size->data() length:sizeof(*proj_img_size) atIndex:8];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:9];
        ENC_SCALAR(enc, clip_thresh, 10);
        ENC_BUF(enc, xys, 11); ENC_BUF(enc, depths, 12);
        ENC_BUF(enc, radii_out, 13); ENC_BUF(enc, conics, 14);
        ENC_BUF(enc, tileCountStorage, 15);
        ENC_SCALAR(enc, degree, 16); ENC_SCALAR(enc, degrees_to_use, 17);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:18];
        ENC_BUF(enc, features_dc, 19); ENC_BUF(enc, features_rest, 20);
        ENC_BUF(enc, colors, 21); ENC_BUF(enc, aabb, 22);
        const uint32_t poseDisabled = 0u;
        ENC_SCALAR(enc, poseDisabled, 23);
        // Render has no coverage objective; a zero stride disables pruning.
        ENC_BUF(enc, g_tcache.tile_scatter_counters, 24);
        const uint32_t coverageRenderTileStrideDisabled = 0u;
        ENC_SCALAR(enc, coverageRenderTileStrideDisabled, 25);
        ENC_SCALAR(enc, tileCountMode, 26);

        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    // Pass 1: project and count exact intersections per tile. Shared storage
    // makes the completed counts directly readable by the CPU.
    {
        id<MTLCommandBuffer> commandBuffer = ctx->getCommandBuffer();
        __block const char* encodingFailure = nullptr;
        dispatch_sync(ctx->d_queue, ^{
            id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
            if (!blit) {
                encodingFailure = "msplat: failed to create a Metal blit encoder";
                return;
            }
            [blit fillBuffer:tileCountStorage.buffer()
                       range:NSMakeRange(
                           0, tileCountStorage.nbytes())
                       value:0];
            [blit endEncoding];

            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (!encoder) {
                encodingFailure =
                    "msplat: failed to create a Metal compute encoder";
                return;
            }
            encode_proj_sh(encoder);
            const uint32_t coverageRenderTileStrideDisabled = 0u;
            encode_tile_count_difference_scan(
                ctx, encoder, g_tcache.tile_count_diff,
                g_tcache.tile_scatter_counters, tile_bounds_arr->data(),
                g_tcache.overflow_flag,
                coverageRenderTileStrideDisabled);
            encode_gpu_tile_intersection_layout(
                ctx, encoder, g_tcache.tile_scatter_counters,
                g_tcache.tile_offsets, g_tcache.tile_bins,
                g_tcache.sortable_tile_indices,
                g_tcache.tile_layout_metadata,
                static_cast<uint32_t>(num_tiles));
            [encoder endEncoding];
        });
        if (encodingFailure) {
            ctx->discardCB();
            throw std::runtime_error(encodingFailure);
        }
        ctx->syncCB();
    }

    const msplat::TileIntersectionLayout intersectionLayout =
        ctx->gpu_tile_layout
            ? completed_gpu_tile_intersection_layout(num_tiles)
            : msplat::buildTileIntersectionLayout(
                g_tcache.tile_scatter_counters.data<uint32_t>(),
                g_tcache.tile_offsets.data<int32_t>(),
                static_cast<size_t>(num_tiles),
                g_tcache.tile_bins.data<int32_t>(),
                g_tcache.sortable_tile_indices.data<uint32_t>());
    msplat::validateTileIntersectionWorkLimit(intersectionLayout);
    const bool needsRadixScratch =
        msplat::tileIntersectionLayoutNeedsRadixScratch(intersectionLayout);
    g_tcache.ensure_intersection_arena(
        intersectionLayout.totalCount, needsRadixScratch,
        !ctx->gather_intersection_attributes, ctx->device);
    if (needsRadixScratch && !g_tcache.intersection_keys_b.defined()) {
        throw std::logic_error("Exact radix sort requires its scratch arena");
    }

    MTensor &packed_xy_opac = g_tcache.packed_xy_opac;
    MTensor &packed_conic = g_tcache.packed_conic;
    MTensor &packed_rgb = g_tcache.packed_rgb;
    MTensor &raster_xy_attributes = ctx->gather_intersection_attributes
        ? xys : packed_xy_opac;
    MTensor &raster_conic_attributes = ctx->gather_intersection_attributes
        ? conics : packed_conic;
    MTensor &raster_rgb_attributes = ctx->gather_intersection_attributes
        ? colors : packed_rgb;
    const uint32_t intersection_attribute_layout =
        ctx->gather_intersection_attributes ? 1u : 0u;
    MTensor &radix_sort_scratch_keys =
        g_tcache.intersection_keys_b.defined()
            ? g_tcache.intersection_keys_b
            : g_tcache.intersection_keys_a;
    const uint32_t capacity_u32 = static_cast<uint32_t>(g_tcache.capacity);
    const uint32_t num_tiles_u32 = static_cast<uint32_t>(num_tiles);
    const uint32_t sortable_tile_count =
        intersectionLayout.sortableTileCount;
    const uint32_t small_sort_tile_count =
        intersectionLayout.smallTileCount;
    const uint32_t general_sort_tile_count =
        intersectionLayout.mediumTileCount + intersectionLayout.largeTileCount;
    const uint32_t general_sort_tile_offset = small_sort_tile_count;
    const uint32_t total_intersections = intersectionLayout.totalCount;

    auto encode_sort_pack = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger scatterTpg = MIN(
            ctx->scatter_to_exact_bins_kernel_cpso.maxTotalThreadsPerThreadgroup,
            static_cast<NSUInteger>(num_points));
        [enc setComputePipelineState:ctx->scatter_to_exact_bins_kernel_cpso];
        ENC_SCALAR(enc, num_points_u32, 0); ENC_BUF(enc, xys, 1);
        ENC_BUF(enc, depths, 2); ENC_BUF(enc, radii_out, 3);
        ENC_BUF(enc, aabb, 4);
        [enc setBytes:tile_bounds_arr->data()
               length:sizeof(*tile_bounds_arr) atIndex:5];
        ENC_BUF(enc, g_tcache.tile_offsets, 6);
        ENC_BUF(enc, g_tcache.tile_scatter_counters, 7);
        ENC_BUF(enc, g_tcache.intersection_keys_a, 8);
        ENC_SCALAR(enc, capacity_u32, 9);
        ENC_BUF(enc, g_tcache.overflow_flag, 10);
        ENC_BUF(enc, opacities, 11);
        ENC_BUF(enc, projected_opacities, 12);
        ENC_BUF(enc, g_tcache.tile_scatter_counters, 13);
        const uint32_t coverageRenderTileStrideDisabled = 0u;
        ENC_SCALAR(enc, coverageRenderTileStrideDisabled, 14);
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(scatterTpg, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        if (small_sort_tile_count > 0) {
            [enc setComputePipelineState:ctx->small_sort_per_tile_kernel_cpso];
            ENC_BUF(enc, g_tcache.tile_offsets, 0);
            ENC_BUF(enc, g_tcache.intersection_keys_a, 1);
            ENC_SCALAR(enc, num_tiles_u32, 2);
            ENC_BUF(enc, tile_bins, 3);
            ENC_SCALAR(enc, capacity_u32, 4);
            ENC_BUF(enc, g_tcache.overflow_flag, 5);
            ENC_BUF(enc, g_tcache.sortable_tile_indices, 6);
            ENC_SCALAR(enc, small_sort_tile_count, 7);
            [enc dispatchThreadgroups:MTLSizeMake(small_sort_tile_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(
                    msplat::kExactSmallTileMaximum, 1, 1)];
        }
        if (general_sort_tile_count > 0) {
            [enc setComputePipelineState:ctx->radix_sort_per_tile_kernel_cpso];
            ENC_BUF(enc, g_tcache.tile_offsets, 0);
            ENC_BUF(enc, g_tcache.intersection_keys_a, 1);
            ENC_BUF(enc, radix_sort_scratch_keys, 2);
            ENC_SCALAR(enc, num_tiles_u32, 3);
            ENC_BUF(enc, tile_bins, 4);
            ENC_SCALAR(enc, capacity_u32, 5);
            ENC_BUF(enc, g_tcache.overflow_flag, 6);
            ENC_BUF(enc, g_tcache.sortable_tile_indices, 7);
            ENC_SCALAR(enc, general_sort_tile_count, 8);
            ENC_SCALAR(enc, general_sort_tile_offset, 9);
            [enc dispatchThreadgroups:MTLSizeMake(general_sort_tile_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        if (total_intersections == 0) return;
        if (sortable_tile_count > 0) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        if (ctx->gather_intersection_attributes) return;
        NSUInteger packTpg = MIN(
            ctx->pack_sorted_gaussians_kernel_cpso.maxTotalThreadsPerThreadgroup,
            static_cast<NSUInteger>(total_intersections));
        [enc setComputePipelineState:ctx->pack_sorted_gaussians_kernel_cpso];
        ENC_BUF(enc, g_tcache.intersection_keys_a, 0);
        ENC_BUF(enc, xys, 1); ENC_BUF(enc, conics, 2);
        ENC_BUF(enc, colors, 3); ENC_BUF(enc, projected_opacities, 4);
        ENC_BUF(enc, packed_xy_opac, 5); ENC_BUF(enc, packed_conic, 6);
        ENC_BUF(enc, packed_rgb, 7);
        ENC_SCALAR(enc, total_intersections, 8);
        [enc dispatchThreads:MTLSizeMake(total_intersections, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(packTpg, 1, 1)];
    };

    static int diag_count = 0;
    diag_count++;
    if (std::getenv("BENCHMARK") &&
        (diag_count == 100 || diag_count == 500 || diag_count == 1500)) {
        fprintf(stderr, "\n=== Roofline Dimensions (iter %d) ===\n", diag_count);
        fprintf(stderr, "  num_points:     %d\n", num_points);
        fprintf(stderr, "  intersections:  %u exact, %u arena capacity\n",
                total_intersections, capacity_u32);
        fprintf(stderr, "  max tile:       %u intersections (tile %zu)\n",
                intersectionLayout.maximumTileCount,
                intersectionLayout.maximumTileIndex);
        fprintf(stderr, "  img:            %u x %u = %u pixels\n",
                img_width, img_height, img_width * img_height);
        fprintf(stderr, "  tiles:          %d x %d = %d\n",
                tile_bounds_x, tile_bounds_y, num_tiles);
        fprintf(stderr, "  SH degree:      %u (bases: %u)\n",
                degree, (degree + 1) * (degree + 1));
        fprintf(stderr,
                "  sort tiles:     %u small32, %u bitonic, %u radix (%u / %u)\n",
                intersectionLayout.smallTileCount,
                intersectionLayout.mediumTileCount,
                intersectionLayout.largeTileCount,
                sortable_tile_count, num_tiles_u32);
        fprintf(stderr, "  sort arenas:    %.1f MB\n",
                static_cast<double>(
                    g_tcache.intersection_keys_a.nbytes() +
                    (g_tcache.intersection_keys_b.defined()
                        ? g_tcache.intersection_keys_b.nbytes()
                        : 0)) / 1e6);
        fprintf(stderr, "===========================\n\n");
    }

    // K_max for chunked rasterization — set after GPU readback
    uint32_t K_max = 1;
    constexpr uint32_t CHUNK_SIZE = 512;

    auto encode_rast_fwd_monolithic = [&](id<MTLComputeCommandEncoder> enc) {
        const uint32_t blockX = ctx->monolithic_raster_block_x;
        const uint32_t blockY = ctx->monolithic_raster_block_y;
        MTLSize num_tg = MTLSizeMake(
            (img_width + blockX - 1) / blockX,
            (img_height + blockY - 1) / blockY, 1);
        MTLSize tg_size = MTLSizeMake(blockX, blockY, 1);
        [enc setComputePipelineState:ctx->nd_rasterize_forward_kernel_cpso];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
        [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
        ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
        ENC_BUF(enc, raster_xy_attributes, 4);
        ENC_BUF(enc, raster_conic_attributes, 5);
        ENC_BUF(enc, raster_rgb_attributes, 6);
        ENC_BUF(enc, final_Ts, 7); ENC_BUF(enc, final_idx, 8); ENC_BUF(enc, out_img, 9);
        ENC_BUF(enc, background, 10);
        [enc setBytes:monolithic_block_size_dim2->data()
               length:sizeof(*monolithic_block_size_dim2) atIndex:11];
        ENC_BUF(enc, g_tcache.intersection_keys_a, 12);
        ENC_BUF(enc, projected_opacities, 13);
        ENC_SCALAR(enc, intersection_attribute_layout, 14);
        [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:tg_size];
    };

    auto encode_rast_fwd_chunked = [&](id<MTLComputeCommandEncoder> enc) {
        // Phase 1: dispatch forward chunked kernel — grid (tile_x, tile_y, K_max)
        uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
        uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
        uint32_t num_pix = img_width * img_height;
        auto img_sz_2 = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
        MTLSize chunked_tg = MTLSizeMake(tile_x, tile_y, K_max);
        MTLSize tg_size = MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1);
        [enc setComputePipelineState:ctx->rasterize_forward_chunked_kernel_cpso];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
        [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
        ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
        ENC_BUF(enc, raster_xy_attributes, 4);
        ENC_BUF(enc, raster_conic_attributes, 5);
        ENC_BUF(enc, raster_rgb_attributes, 6);
        ENC_BUF(enc, g_tcache.chunk_T, 7); ENC_BUF(enc, g_tcache.chunk_C, 8); ENC_BUF(enc, g_tcache.chunk_final_idx, 9);
        ENC_SCALAR(enc, CHUNK_SIZE, 10); ENC_SCALAR(enc, K_max, 11);
        [enc setBytes:chunked_block_size_dim2->data()
               length:sizeof(*chunked_block_size_dim2) atIndex:12];
        ENC_BUF(enc, g_tcache.intersection_keys_a, 13);
        ENC_BUF(enc, projected_opacities, 14);
        ENC_SCALAR(enc, intersection_attribute_layout, 15);
        [enc dispatchThreadgroups:chunked_tg threadsPerThreadgroup:tg_size];

        // Phase 2: merge kernel — one thread per pixel
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        NSUInteger merge_tpg = 256;
        [enc setComputePipelineState:ctx->rasterize_forward_merge_kernel_cpso];
        ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, K_max, 1);
        ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3); ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
        ENC_BUF(enc, final_Ts, 5); ENC_BUF(enc, final_idx, 6); ENC_BUF(enc, out_img, 7);
        ENC_BUF(enc, background, 8);
        [enc setBytes:img_sz_2->data() length:sizeof(*img_sz_2) atIndex:9];
        [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    };

    auto encode_rast_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (K_max <= 1) {
            encode_rast_fwd_monolithic(enc);
        } else {
            encode_rast_fwd_chunked(enc);
        }
    };

    // Zero cached buffers will be done inside the command encoder via blit.
    // (CPU memset would race with previous CB's GPU reads.)

    // The exact densest-tile count removes the previous heuristic chunk bound,
    // so the chunked path cannot silently skip a heavy-tailed tile.
    K_max = msplat::tileRasterChunkCount(
        static_cast<uint32_t>(num_tiles),
        intersectionLayout.maximumTileCount, CHUNK_SIZE);
    g_tcache.ensure_forward_chunks(K_max, img_height, img_width, ctx->device);

    {
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        __block const char* encodingFailure = nullptr;

        dispatch_sync(ctx->d_queue, ^(){
            // Reset the scatter cursors separately from the exact counts retained
            // for layout and telemetry.
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            if (!blit) {
                encodingFailure = "msplat: failed to create a Metal blit encoder";
                return;
            }
            [blit fillBuffer:g_tcache.overflow_flag.buffer()
                       range:NSMakeRange(0, sizeof(uint32_t)) value:0];
            [blit fillBuffer:g_tcache.tile_scatter_counters.buffer()
                       range:NSMakeRange(
                           0, g_tcache.tile_scatter_counters.nbytes())
                       value:0];
            [blit endEncoding];

            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (!encoder) {
                encodingFailure = "msplat: failed to create a Metal compute encoder";
                return;
            }

            encode_sort_pack(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_rast_fwd(encoder);

            [encoder endEncoding];
        });
        if (encodingFailure) {
            ctx->discardCB();
            throw std::runtime_error(encodingFailure);
        }
    }

    // All outputs are in g_tcache — no return needed
}

MTensor msplat_render(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background
) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    render_pipeline(num_points, means3d, scales, glob_scale,
        quats, viewmat, projmat, fx, fy, cx, cy,
        img_height, img_width, tile_bounds, clip_thresh,
        degree, degrees_to_use, cam_pos, features_dc, features_rest,
        opacities, background);
    return g_tcache.out_img;
}

static MTensor msplat_train_step_locked(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &gt, const MTensor* coverage_mask,
    const MTensor* coverage_render_tiles,
    uint64_t loss_coverage_units, float ssim_weight,
    float loss_inv_n, bool transparent_mask,
    float alpha_loss_weight,
    int num_adam_groups,
    MTensor adam_params[], MTensor adam_exp_avg[], MTensor adam_exp_avg_sq[],
    float adam_step_sizes[], float adam_bc2_sqrts[],
    float adam_beta1, float adam_beta2, float adam_eps,
    const MsplatPhotometricRefinementStep& photometric,
    const MsplatPoseRefinementStep& pose,
    bool collect_densification_stats,
    MTensor &vis_counts, MTensor &xys_grad_norm, MTensor &max_2d_size,
    float inv_max_dim
) {
    MetalContext* ctx = get_global_context();
    auto logicalStep = ctx->activeTrainingStep;
    constexpr int kExpectedAdamGroups = 6;
    if (num_adam_groups != kExpectedAdamGroups) {
        throw std::invalid_argument(
            "Training requires exactly six Adam parameter groups");
    }
    int tile_bounds_x = std::get<0>(tile_bounds);
    int tile_bounds_y = std::get<1>(tile_bounds);
    int num_tiles = tile_bounds_x * tile_bounds_y;

    if (!g_profile_stages_checked) {
        g_profile_stages = std::getenv("PROFILE_STAGES") != nullptr;
        g_profile_stages_checked = true;
    }
    std::shared_ptr<ScopedObjCRelease> stageCounterOwner;
    id<MTLCounterSampleBuffer> stageCounterSampleBuffer = nil;
    if (g_profile_stages && ctx->counterSamplingAvailable) {
        auto owner = std::make_shared<ScopedObjCRelease>();
        owner->object = ctx->newTrainingCounterSampleBuffer();
        if (owner->object) {
            stageCounterSampleBuffer =
                static_cast<id<MTLCounterSampleBuffer>>(owner->object);
            stageCounterOwner = std::move(owner);
        }
    }
    const bool profileThisStep = stageCounterSampleBuffer != nil;
    uint32_t channels = 3;
    const uint64_t pixelCount = static_cast<uint64_t>(img_height) *
        static_cast<uint64_t>(img_width);
    if (pixelCount == 0 ||
        pixelCount > std::numeric_limits<uint64_t>::max() / 255u) {
        throw std::invalid_argument("Training image dimensions are invalid");
    }
    const bool targetShapeMatches = gt.defined() && gt.isGpu() &&
        gt.ndim() == 3 &&
        gt.size(0) == static_cast<int64_t>(img_height) &&
        gt.size(1) == static_cast<int64_t>(img_width);
    const bool targetIsRGBA8 = targetShapeMatches &&
        gt.dtype() == DType::UInt8 && gt.size(2) == 4;
    const bool targetIsFloatRGB = targetShapeMatches &&
        gt.dtype() == DType::Float32 && gt.size(2) == 3;
    if (!targetIsRGBA8 && !targetIsFloatRGB) {
        throw std::invalid_argument(
            "Training target must be a GPU uint8 RGBA or float32 RGB image");
    }
    const uint32_t target_pixel_stride_bytes = targetIsRGBA8
        ? 4u
        : 3u * static_cast<uint32_t>(sizeof(float));
    const uint64_t fullCoverageUnits = pixelCount * 255u;
    if (loss_coverage_units == 0 ||
        loss_coverage_units > fullCoverageUnits ||
        !std::isfinite(loss_inv_n) || loss_inv_n <= 0.0f) {
        throw std::invalid_argument("Training loss normalization is invalid");
    }
    if (!std::isfinite(alpha_loss_weight) || alpha_loss_weight < 0.0f) {
        throw std::invalid_argument(
            "Training alpha loss weight must be finite and non-negative");
    }
    const bool coverageIsPackedAlpha = coverage_mask == &gt;
    if (coverage_mask) {
        const bool standaloneCoverageValid =
            coverage_mask->defined() && coverage_mask->isGpu() &&
            coverage_mask->dtype() == DType::UInt8 &&
            coverage_mask->ndim() == 2 &&
            coverage_mask->size(0) == static_cast<int64_t>(img_height) &&
            coverage_mask->size(1) == static_cast<int64_t>(img_width);
        if ((!coverageIsPackedAlpha && !standaloneCoverageValid) ||
            (coverageIsPackedAlpha && !targetIsRGBA8)) {
            throw std::invalid_argument(
                "Training coverage must be packed RGBA alpha or a GPU uint8 image");
        }
    } else if (loss_coverage_units != fullCoverageUnits) {
        throw std::invalid_argument(
            "Unmasked training coverage denominator is inconsistent");
    }
    if (transparent_mask && !coverage_mask) {
        throw std::invalid_argument(
            "Transparent training requires a per-frame mask");
    }
    if (transparent_mask && loss_coverage_units != fullCoverageUnits) {
        throw std::invalid_argument(
            "Transparent training must use the full-frame loss denominator");
    }
    if (coverage_render_tiles) {
        const int64_t expectedTileHeight =
            (static_cast<int64_t>(img_height) + BLOCK_Y - 1) / BLOCK_Y;
        const int64_t expectedTileWidth =
            (static_cast<int64_t>(img_width) + BLOCK_X - 1) / BLOCK_X;
        if (tile_bounds_x != expectedTileWidth ||
            tile_bounds_y != expectedTileHeight ||
            !coverage_mask || !coverage_render_tiles->defined() ||
            !coverage_render_tiles->isGpu() ||
            coverage_render_tiles->dtype() != DType::UInt8 ||
            coverage_render_tiles->ndim() != 2 ||
            coverage_render_tiles->size(0) != expectedTileHeight ||
            coverage_render_tiles->size(1) != expectedTileWidth) {
            throw std::invalid_argument(
                "Coverage render tiles must be a GPU uint8 tile map matching "
                "the training image");
        }
    }
    const MTensor& loss_coverage_buffer = coverage_mask ? *coverage_mask : gt;
    // uint2(x, y) is byte stride per pixel plus byte offset within the pixel.
    // Camera cache targets use RGBA alpha (4,3); legacy standalone masks use
    // one byte per pixel (1,0). A zero stride disables that objective.
    const std::array<uint32_t, 2> mask_layout = coverageIsPackedAlpha
        ? std::array<uint32_t, 2>{4u, 3u}
        : std::array<uint32_t, 2>{1u, 0u};
    const std::array<uint32_t, 2> coverage_layout =
        coverage_mask && !transparent_mask
            ? mask_layout
            : std::array<uint32_t, 2>{0u, 0u};
    const std::array<uint32_t, 2> alpha_layout =
        coverage_mask && transparent_mask
            ? mask_layout
            : std::array<uint32_t, 2>{0u, 0u};
    const bool useCoverageRenderTiles = coverage_mask &&
        !transparent_mask && coverage_render_tiles;
    const MTensor &coverageRenderTileBuffer = useCoverageRenderTiles
        ? *coverage_render_tiles
        : loss_coverage_buffer;
    const uint32_t coverageRenderTileStride = useCoverageRenderTiles
        ? static_cast<uint32_t>(tile_bounds_x)
        : 0u;
    const float alpha_gradient_scale = transparent_mask
        ? alpha_loss_weight / static_cast<float>(pixelCount)
        : 0.0f;

    auto validatePhotometricTensor = [](const MTensor* tensor,
                                        const char* name) {
        if (!tensor || !tensor->defined() || !tensor->isGpu() ||
            tensor->dtype() != DType::Float32 || tensor->ndim() != 2 ||
            tensor->size(0) <= 0 || tensor->size(1) != 3) {
            throw std::invalid_argument(
                std::string(name) + " must be a non-empty GPU Float32 [N,3] tensor");
        }
    };
    validatePhotometricTensor(photometric.logRgbGains,
                              "Photometric log-RGB gains");
    if (photometric.cameraIndex >
            std::numeric_limits<uint32_t>::max() / 3u ||
        static_cast<int64_t>(photometric.cameraIndex) >=
            photometric.logRgbGains->size(0)) {
        throw std::invalid_argument("Photometric camera index is out of range");
    }
    if (!photometric.enabled && photometric.cameraIndex != 0u) {
        throw std::invalid_argument(
            "Disabled photometric refinement must use the identity gain row");
    }
    if (photometric.enabled) {
        validatePhotometricTensor(photometric.expAvg,
                                  "Photometric Adam first moment");
        validatePhotometricTensor(photometric.expAvgSq,
                                  "Photometric Adam second moment");
        if (photometric.expAvg->size(0) !=
                photometric.logRgbGains->size(0) ||
            photometric.expAvgSq->size(0) !=
                photometric.logRgbGains->size(0)) {
            throw std::invalid_argument(
                "Photometric Adam tensors must match the gain tensor shape");
        }
        constexpr float expectedMaxAbsLogGain = 1.38629436112f; // log(4)
        if (!std::isfinite(photometric.adamStepSize) ||
            photometric.adamStepSize <= 0.0f ||
            !std::isfinite(photometric.adamBiasCorrection2Sqrt) ||
            photometric.adamBiasCorrection2Sqrt <= 0.0f ||
            !std::isfinite(photometric.regularization) ||
            photometric.regularization < 0.0f ||
            !std::isfinite(photometric.maxAbsLogGain) ||
            std::abs(photometric.maxAbsLogGain - expectedMaxAbsLogGain) >
                1.0e-6f) {
            throw std::invalid_argument(
                "Photometric optimizer parameters are invalid");
        }
    }
    MTensor& photometricLogGains = *photometric.logRgbGains;
    MTensor& photometricExpAvg = photometric.enabled
        ? *photometric.expAvg
        : photometricLogGains;
    MTensor& photometricExpAvgSq = photometric.enabled
        ? *photometric.expAvgSq
        : photometricLogGains;
    const uint32_t cameraGainOffset = photometric.cameraIndex * 3u;
    const uint32_t photometricEnabled = photometric.enabled ? 1u : 0u;

    auto validatePoseTensor = [](const MTensor* tensor, const char* name) {
        if (!tensor || !tensor->defined() || !tensor->isGpu() ||
            tensor->dtype() != DType::Float32 || tensor->ndim() != 2 ||
            tensor->size(0) <= 0 || tensor->size(1) != 6) {
            throw std::invalid_argument(
                std::string(name) +
                " must be a non-empty GPU Float32 [N,6] tensor");
        }
    };
    validatePoseTensor(pose.deltas, "Camera pose deltas");
    if (pose.cameraIndex > std::numeric_limits<uint32_t>::max() / 6u ||
        static_cast<int64_t>(pose.cameraIndex) >= pose.deltas->size(0)) {
        throw std::invalid_argument("Pose-refinement camera index is out of range");
    }
    if (!pose.enabled && pose.cameraIndex != 0u) {
        throw std::invalid_argument(
            "Disabled pose refinement must use the identity pose row");
    }
    if (pose.enabled) {
        validatePoseTensor(pose.expAvg, "Camera pose Adam first moment");
        validatePoseTensor(pose.expAvgSq, "Camera pose Adam second moment");
        if (pose.expAvg->size(0) != pose.deltas->size(0) ||
            pose.expAvgSq->size(0) != pose.deltas->size(0)) {
            throw std::invalid_argument(
                "Camera pose Adam tensors must match the delta tensor shape");
        }
        constexpr float expectedMaxTranslation = 0.05f;
        constexpr float expectedMaxRotation = 0.05235987756f;
        if (!std::isfinite(pose.adamStepSize) ||
            pose.adamStepSize <= 0.0f ||
            !std::isfinite(pose.adamBiasCorrection2Sqrt) ||
            pose.adamBiasCorrection2Sqrt <= 0.0f ||
            !std::isfinite(pose.regularization) ||
            pose.regularization < 0.0f ||
            !std::isfinite(pose.maxTranslation) ||
            std::abs(pose.maxTranslation - expectedMaxTranslation) > 1.0e-6f ||
            !std::isfinite(pose.maxRotation) ||
            std::abs(pose.maxRotation - expectedMaxRotation) > 1.0e-6f) {
            throw std::invalid_argument(
                "Camera pose optimizer parameters are invalid");
        }
    }
    MTensor& poseDeltas = *pose.deltas;
    MTensor& poseExpAvg = pose.enabled ? *pose.expAvg : poseDeltas;
    MTensor& poseExpAvgSq = pose.enabled ? *pose.expAvgSq : poseDeltas;
    const uint32_t cameraPoseOffset = pose.cameraIndex * 6u;
    const uint32_t poseEnabled = pose.enabled ? 1u : 0u;

    if (collect_densification_stats) {
        const auto validStatsTensor = [num_points](const MTensor& tensor) {
            return tensor.defined() && tensor.isGpu() &&
                tensor.dtype() == DType::Float32 &&
                tensor.numel() >= num_points;
        };
        if (!validStatsTensor(vis_counts) ||
            !validStatsTensor(xys_grad_norm) ||
            !validStatsTensor(max_2d_size)) {
            throw std::invalid_argument(
                "Densification statistics must be GPU Float32 tensors with one value per Gaussian");
        }
    }

    // --- Cached buffer pool ---
    const bool intersectionResolutionChanged =
        static_cast<int>(img_height) != g_tcache.shared_img_height ||
        static_cast<int>(img_width) != g_tcache.shared_img_width ||
        num_tiles != g_tcache.num_tiles;
    g_tcache.ensure_shared_forward(
        num_points, img_height, img_width, num_tiles, ctx->device);
    g_tcache.ensure_tile_count_diff(
        tile_bounds_x, tile_bounds_y, ctx->difference_tile_counting,
        ctx->device);
    g_tcache.ensure_tile_layout_metadata(ctx->gpu_tile_layout, ctx->device);
    g_tcache.ensure_tile_attempt_dispatch_control(
        ctx->retry_intersection_attempts, ctx->device);
    g_tcache.ensure_training_image(
        img_height, img_width, !ctx->fused_ssim_backward, ctx->device);
    g_tcache.ensure_backward(num_points, ctx->device);
    if (pose.enabled) g_tcache.ensure_pose_refinement(ctx->device);

    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &conics = g_tcache.conics;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
    MTensor &projected_opacities = g_tcache.projected_opacities;
    MTensor &tileCountStorage = ctx->difference_tile_counting
        ? g_tcache.tile_count_diff
        : g_tcache.tile_scatter_counters;
    const uint32_t tileCountMode =
        ctx->difference_tile_counting ? 1u : 0u;
    MTensor &tile_bins = g_tcache.tile_bins;
    MTensor &loss_sum = g_tcache.loss_sum;
    MTensor &overflow_flag = logicalStep
        ? logicalStep->readbackBuffer()
        : g_tcache.overflow_flag;
    MTensor &out_img = g_tcache.out_img;
    MTensor &final_Ts = g_tcache.final_Ts;
    MTensor &final_idx = g_tcache.final_idx;
    MTensor &photometric_gradient = g_tcache.photometric_gradient;
    MTensor &activeViewmat = pose.enabled ? g_tcache.pose_viewmat : viewmat;
    MTensor &poseGradient = pose.enabled
        ? g_tcache.pose_gradient
        : photometric_gradient;

    // SSIM V-backward reads and writes only the same pixel. Training no longer
    // needs the rendered image afterward, so reuse it in place for dL/dRGB.
    MTensor &rendered_gradient = out_img;
    MTensor &v_xy = g_tcache.v_xy;
    MTensor &v_conic = g_tcache.v_conic;
    MTensor &v_colors_rast = g_tcache.v_colors_rast;
    MTensor &v_opacity = g_tcache.v_opacity;
    // Metal requires every declared pointer binding to reference a buffer even
    // when statistics collection is disabled. v_opacity is N-sized and the
    // shader does not touch these aliases while collectStats is zero.
    MTensor &visCountsBuffer = collect_densification_stats
        ? vis_counts : v_opacity;
    MTensor &xysGradNormBuffer = collect_densification_stats
        ? xys_grad_norm : v_opacity;
    MTensor &max2DSizeBuffer = collect_densification_stats
        ? max_2d_size : v_opacity;
    const uint32_t collectStats = collect_densification_stats ? 1u : 0u;
    const uint32_t attemptGatingEnabled =
        ctx->retry_intersection_attempts ? 1u : 0u;

    // A resolution change deliberately resets the arena high-water mark. That
    // frame uses the established synchronized bootstrap; subsequent retry-mode
    // frames keep layout, capacity, sorting, and raster selection on the GPU.
    const bool gpuResidentIntersectionAttempt =
        ctx->retry_intersection_attempts &&
        !intersectionResolutionChanged && g_tcache.capacity > 0;
    uint32_t plannedAttemptChunkCount = 1;
    if (gpuResidentIntersectionAttempt && num_tiles < 400 &&
        g_tcache.chunk_T.defined() && g_tcache.chunk_C.defined() &&
        g_tcache.chunk_final_idx.defined() &&
        g_tcache.prefix_T.defined() && g_tcache.after_C.defined() &&
        g_tcache.forward_chunk_height == static_cast<int>(img_height) &&
        g_tcache.forward_chunk_width == static_cast<int>(img_width) &&
        g_tcache.backward_chunk_height == static_cast<int>(img_height) &&
        g_tcache.backward_chunk_width == static_cast<int>(img_width)) {
        plannedAttemptChunkCount = static_cast<uint32_t>(std::max(
            1, std::min(g_tcache.forward_chunk_K_max,
                        g_tcache.backward_chunk_K_max)));
    }

    // --- Constants (heap-allocated for Obj-C block capture) ---
    auto loss_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto proj_intrins = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto tile_bounds_arr = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (uint32_t)tile_bounds_x, (uint32_t)tile_bounds_y,
        (uint32_t)std::get<2>(tile_bounds), 0xDEAD
    });
    // Metal constant float3 arguments occupy 16 bytes, despite carrying only
    // three values.
    auto cam_pos_arr = std::make_shared<std::array<float, 4>>(
        std::array<float, 4>{cam_pos[0], cam_pos[1], cam_pos[2], 0.0f});
    uint32_t num_points_u32 = (uint32_t)num_points;
    auto img_size_dim3 = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{img_width, img_height, 1, 0xDEAD});
    auto monolithic_block_size_dim2 =
        std::make_shared<std::array<int32_t, 2>>(
            std::array<int32_t, 2>{
                static_cast<int32_t>(ctx->monolithic_raster_block_x),
                static_cast<int32_t>(ctx->monolithic_raster_block_y)});
    auto chunked_block_size_dim2 =
        std::make_shared<std::array<int32_t, 2>>(
            std::array<int32_t, 2>{RAST_BLOCK_X, RAST_BLOCK_Y});
    // tile_bounds for rasterize kernels must be 16x16 tile counts (tile_bins granularity)
    auto rast_tb = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (img_width + 15u) / 16u,
        (img_height + 15u) / 16u, 1, 0xDEAD});
    auto rast_isz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto proj_bwd_intr = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_bwd_isz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});

    // ========================== FORWARD ENCODE LAMBDAS ==========================

    auto encode_pose_prepare = [&](id<MTLComputeCommandEncoder> enc) {
        if (!pose.enabled) return;
        [enc setComputePipelineState:ctx->prepare_camera_pose_kernel_cpso];
        ENC_BUF(enc, viewmat, 0);
        ENC_BUF(enc, poseDeltas, 1);
        ENC_SCALAR(enc, cameraPoseOffset, 2);
        ENC_BUF(enc, g_tcache.pose_viewmat, 3);
        ENC_BUF(enc, g_tcache.pose_cam_pos, 4);
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };

    auto encode_proj_sh = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_forward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_forward_kernel_cpso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, activeViewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_intrins->data() length:sizeof(*proj_intrins) atIndex:7];
        [enc setBytes:proj_img_size->data() length:sizeof(*proj_img_size) atIndex:8];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:9];
        ENC_SCALAR(enc, clip_thresh, 10);
        ENC_BUF(enc, xys, 11); ENC_BUF(enc, depths, 12);
        ENC_BUF(enc, radii_out, 13); ENC_BUF(enc, conics, 14);
        ENC_BUF(enc, tileCountStorage, 15);
        ENC_SCALAR(enc, degree, 16); ENC_SCALAR(enc, degrees_to_use, 17);
        if (pose.enabled) {
            ENC_BUF(enc, g_tcache.pose_cam_pos, 18);
        } else {
            [enc setBytes:cam_pos_arr->data()
                   length:sizeof(*cam_pos_arr) atIndex:18];
        }
        ENC_BUF(enc, features_dc, 19); ENC_BUF(enc, features_rest, 20);
        ENC_BUF(enc, colors, 21); ENC_BUF(enc, aabb, 22);
        ENC_SCALAR(enc, poseEnabled, 23);
        ENC_BUF(enc, coverageRenderTileBuffer, 24);
        ENC_SCALAR(enc, coverageRenderTileStride, 25);
        ENC_SCALAR(enc, tileCountMode, 26);

        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    // Pass 1 completes before any arena or optimizer work. This makes the
    // packed allocation exact and turns allocation failure into a recoverable
    // step error rather than an incomplete training frame.
    {
        id<MTLCommandBuffer> commandBuffer = ctx->getCommandBuffer();
        __block const char* encodingFailure = nullptr;
        dispatch_sync(ctx->d_queue, ^{
            id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
            if (!blit) {
                encodingFailure = "msplat: failed to create a Metal blit encoder";
                return;
            }
            [blit fillBuffer:tileCountStorage.buffer()
                       range:NSMakeRange(
                           0, tileCountStorage.nbytes())
                       value:0];
            // Clear once, before layout publishes the capacity decision. The
            // post-layout work must preserve this status through every
            // persistent update and completion-side retry decision.
            [blit fillBuffer:overflow_flag.buffer()
                       range:NSMakeRange(0, sizeof(uint32_t)) value:0];
            [blit endEncoding];

            id<MTLComputeCommandEncoder> encoder = nil;
            if (profileThisStep) {
                MTLComputePassDescriptor *passDescriptor =
                    [MTLComputePassDescriptor computePassDescriptor];
                passDescriptor.sampleBufferAttachments[0].sampleBuffer =
                    stageCounterSampleBuffer;
                passDescriptor.sampleBufferAttachments[0]
                    .startOfEncoderSampleIndex = 0;
                passDescriptor.sampleBufferAttachments[0]
                    .endOfEncoderSampleIndex = 1;
                encoder = [commandBuffer
                    computeCommandEncoderWithDescriptor:passDescriptor];
            } else {
                encoder = [commandBuffer computeCommandEncoder];
            }
            if (!encoder) {
                encodingFailure =
                    "msplat: failed to create a Metal compute encoder";
                return;
            }
            encode_pose_prepare(encoder);
            encode_proj_sh(encoder);
            encode_tile_count_difference_scan(
                ctx, encoder, g_tcache.tile_count_diff,
                g_tcache.tile_scatter_counters, tile_bounds_arr->data(),
                coverageRenderTileBuffer, coverageRenderTileStride);
            encode_gpu_tile_intersection_layout(
                ctx, encoder, g_tcache.tile_scatter_counters,
                g_tcache.tile_offsets, g_tcache.tile_bins,
                g_tcache.sortable_tile_indices,
                g_tcache.tile_layout_metadata,
                static_cast<uint32_t>(num_tiles));
            if (gpuResidentIntersectionAttempt) {
                const uint32_t arenaCapacity =
                    static_cast<uint32_t>(g_tcache.capacity);
                const uint32_t radixScratchAvailable =
                    g_tcache.intersection_keys_b.defined() ? 1u : 0u;
                const uint32_t packThreadsPerGroup = static_cast<uint32_t>(
                    ctx->pack_sorted_gaussians_kernel_cpso
                        .maxTotalThreadsPerThreadgroup);
                encode_validate_tile_intersection_attempt(
                    ctx, encoder, g_tcache.tile_bins,
                    static_cast<uint32_t>(num_tiles), arenaCapacity,
                    radixScratchAvailable, plannedAttemptChunkCount,
                    packThreadsPerGroup, overflow_flag,
                    g_tcache.tile_layout_metadata,
                    g_tcache.tile_attempt_dispatch_control,
                    g_tcache.tile_offsets);
            }
            [encoder endEncoding];
        });
        if (encodingFailure) {
            ctx->discardCB();
            throw std::runtime_error(encodingFailure);
        }
        if (!gpuResidentIntersectionAttempt) {
            const SynchronousGpuMetrics countPassMetrics = ctx->syncCB();
            if (logicalStep) {
                logicalStep->recordExactCountPass(countPassMetrics);
            }
        }
    }

    msplat::TileIntersectionLayout intersectionLayout;
    bool intersectionArenaGrew = false;
    if (!gpuResidentIntersectionAttempt) {
        intersectionLayout = ctx->gpu_tile_layout
            ? completed_gpu_tile_intersection_layout(num_tiles)
            : msplat::buildTileIntersectionLayout(
                g_tcache.tile_scatter_counters.data<uint32_t>(),
                g_tcache.tile_offsets.data<int32_t>(),
                static_cast<size_t>(num_tiles),
                g_tcache.tile_bins.data<int32_t>(),
                g_tcache.sortable_tile_indices.data<uint32_t>());
        msplat::validateTileIntersectionWorkLimit(intersectionLayout);
        const bool needsRadixScratch =
            msplat::tileIntersectionLayoutNeedsRadixScratch(intersectionLayout);
        const auto arenaGrowStart = TelemetryClock::now();
        intersectionArenaGrew = g_tcache.ensure_intersection_arena(
            intersectionLayout.totalCount, needsRadixScratch,
            !ctx->gather_intersection_attributes, ctx->device);
        const auto arenaGrowEnd = TelemetryClock::now();
        if (needsRadixScratch && !g_tcache.intersection_keys_b.defined()) {
            throw std::logic_error("Exact radix sort requires its scratch arena");
        }
        if (logicalStep) {
            logicalStep->recordIntersectionLayout(
                intersectionLayout,
                intersectionArenaGrew
                    ? elapsedMilliseconds(arenaGrowStart, arenaGrowEnd)
                    : 0.0);
        }
    }

    MTensor &packed_xy_opac = g_tcache.packed_xy_opac;
    MTensor &packed_conic = g_tcache.packed_conic;
    MTensor &packed_rgb = g_tcache.packed_rgb;
    MTensor &raster_xy_attributes = ctx->gather_intersection_attributes
        ? xys : packed_xy_opac;
    MTensor &raster_conic_attributes = ctx->gather_intersection_attributes
        ? conics : packed_conic;
    MTensor &raster_rgb_attributes = ctx->gather_intersection_attributes
        ? colors : packed_rgb;
    const uint32_t intersection_attribute_layout =
        ctx->gather_intersection_attributes ? 1u : 0u;
    MTensor &radix_sort_scratch_keys =
        g_tcache.intersection_keys_b.defined()
            ? g_tcache.intersection_keys_b
            : g_tcache.intersection_keys_a;
    const uint32_t capacity_u32 = static_cast<uint32_t>(g_tcache.capacity);
    const uint32_t num_tiles_u32 = static_cast<uint32_t>(num_tiles);
    const uint32_t sortable_tile_count = gpuResidentIntersectionAttempt
        ? 0u : intersectionLayout.sortableTileCount;
    const uint32_t small_sort_tile_count = gpuResidentIntersectionAttempt
        ? 0u : intersectionLayout.smallTileCount;
    const uint32_t general_sort_tile_count = gpuResidentIntersectionAttempt
        ? 0u
        : intersectionLayout.mediumTileCount + intersectionLayout.largeTileCount;
    const uint32_t general_sort_tile_offset = small_sort_tile_count;
    const uint32_t total_intersections = gpuResidentIntersectionAttempt
        ? 0u : intersectionLayout.totalCount;

    constexpr uint32_t CHUNK_SIZE = 512;
    const uint32_t K_max = gpuResidentIntersectionAttempt
        ? plannedAttemptChunkCount
        : msplat::tileRasterChunkCount(
            static_cast<uint32_t>(num_tiles),
            intersectionLayout.maximumTileCount, CHUNK_SIZE);
    if (!gpuResidentIntersectionAttempt) {
        g_tcache.ensure_forward_chunks(
            K_max, img_height, img_width, ctx->device);
        g_tcache.ensure_backward_chunks(
            K_max, img_height, img_width, ctx->device);
    }

    uint32_t bwd_K_max = K_max;
    constexpr uint32_t BWD_CHUNK_SIZE = 512;

    auto encode_scatter_sort_finalize =
        [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger scatterTpg = MIN(
            ctx->scatter_to_exact_bins_kernel_cpso.maxTotalThreadsPerThreadgroup,
            static_cast<NSUInteger>(num_points));
        [enc setComputePipelineState:ctx->scatter_to_exact_bins_kernel_cpso];
        ENC_SCALAR(enc, num_points_u32, 0); ENC_BUF(enc, xys, 1);
        ENC_BUF(enc, depths, 2); ENC_BUF(enc, radii_out, 3);
        ENC_BUF(enc, aabb, 4);
        [enc setBytes:tile_bounds_arr->data()
               length:sizeof(*tile_bounds_arr) atIndex:5];
        ENC_BUF(enc, g_tcache.tile_offsets, 6);
        ENC_BUF(enc, g_tcache.tile_scatter_counters, 7);
        ENC_BUF(enc, g_tcache.intersection_keys_a, 8);
        ENC_SCALAR(enc, capacity_u32, 9);
        ENC_BUF(enc, overflow_flag, 10);
        ENC_BUF(enc, opacities, 11);
        ENC_BUF(enc, projected_opacities, 12);
        ENC_BUF(enc, coverageRenderTileBuffer, 13);
        ENC_SCALAR(enc, coverageRenderTileStride, 14);
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(scatterTpg, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        if (gpuResidentIntersectionAttempt || small_sort_tile_count > 0) {
            [enc setComputePipelineState:ctx->small_sort_per_tile_kernel_cpso];
            ENC_BUF(enc, g_tcache.tile_offsets, 0);
            ENC_BUF(enc, g_tcache.intersection_keys_a, 1);
            ENC_SCALAR(enc, num_tiles_u32, 2);
            ENC_BUF(enc, tile_bins, 3);
            ENC_SCALAR(enc, capacity_u32, 4);
            ENC_BUF(enc, overflow_flag, 5);
            ENC_BUF(enc, g_tcache.sortable_tile_indices, 6);
            if (gpuResidentIntersectionAttempt) {
                [enc setBuffer:g_tcache.tile_attempt_dispatch_control.buffer()
                        offset:9u * sizeof(uint32_t) atIndex:7];
                [enc dispatchThreadgroupsWithIndirectBuffer:
                        g_tcache.tile_attempt_dispatch_control.buffer()
                    indirectBufferOffset:0
                    threadsPerThreadgroup:MTLSizeMake(
                        msplat::kExactSmallTileMaximum, 1, 1)];
            } else {
                ENC_SCALAR(enc, small_sort_tile_count, 7);
                [enc dispatchThreadgroups:
                        MTLSizeMake(small_sort_tile_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(
                        msplat::kExactSmallTileMaximum, 1, 1)];
            }
        }
        if (gpuResidentIntersectionAttempt || general_sort_tile_count > 0) {
            [enc setComputePipelineState:ctx->radix_sort_per_tile_kernel_cpso];
            ENC_BUF(enc, g_tcache.tile_offsets, 0);
            ENC_BUF(enc, g_tcache.intersection_keys_a, 1);
            ENC_BUF(enc, radix_sort_scratch_keys, 2);
            ENC_SCALAR(enc, num_tiles_u32, 3);
            ENC_BUF(enc, tile_bins, 4);
            ENC_SCALAR(enc, capacity_u32, 5);
            ENC_BUF(enc, overflow_flag, 6);
            ENC_BUF(enc, g_tcache.sortable_tile_indices, 7);
            if (gpuResidentIntersectionAttempt) {
                [enc setBuffer:g_tcache.tile_attempt_dispatch_control.buffer()
                        offset:10u * sizeof(uint32_t) atIndex:8];
                [enc setBuffer:g_tcache.tile_attempt_dispatch_control.buffer()
                        offset:11u * sizeof(uint32_t) atIndex:9];
                [enc dispatchThreadgroupsWithIndirectBuffer:
                        g_tcache.tile_attempt_dispatch_control.buffer()
                    indirectBufferOffset:3u * sizeof(uint32_t)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                ENC_SCALAR(enc, general_sort_tile_count, 8);
                ENC_SCALAR(enc, general_sort_tile_offset, 9);
                [enc dispatchThreadgroups:
                        MTLSizeMake(general_sort_tile_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        if (gpuResidentIntersectionAttempt) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:
                ctx->finalize_tile_intersection_attempt_kernel_cpso];
            ENC_BUF(enc, tile_bins, 0);
            ENC_BUF(enc, overflow_flag, 1);
            ENC_SCALAR(enc, num_tiles_u32, 2);
            ENC_BUF(enc, g_tcache.tile_attempt_dispatch_control, 3);
            const NSUInteger finalizeTpg = MIN(
                ctx->finalize_tile_intersection_attempt_kernel_cpso
                    .maxTotalThreadsPerThreadgroup,
                static_cast<NSUInteger>(num_tiles));
            [enc dispatchThreads:MTLSizeMake(num_tiles, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(finalizeTpg, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else {
            if (total_intersections == 0) return;
            if (sortable_tile_count > 0) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        }

    };

    auto encode_pack = [&](id<MTLComputeCommandEncoder> enc) {
        if (ctx->gather_intersection_attributes ||
            (!gpuResidentIntersectionAttempt && total_intersections == 0u)) {
            return;
        }
        const NSUInteger packTpg = gpuResidentIntersectionAttempt
            ? ctx->pack_sorted_gaussians_kernel_cpso
                .maxTotalThreadsPerThreadgroup
            : MIN(
                ctx->pack_sorted_gaussians_kernel_cpso
                    .maxTotalThreadsPerThreadgroup,
                static_cast<NSUInteger>(total_intersections));
        [enc setComputePipelineState:ctx->pack_sorted_gaussians_kernel_cpso];
        ENC_BUF(enc, g_tcache.intersection_keys_a, 0);
        ENC_BUF(enc, xys, 1); ENC_BUF(enc, conics, 2);
        ENC_BUF(enc, colors, 3); ENC_BUF(enc, projected_opacities, 4);
        ENC_BUF(enc, packed_xy_opac, 5); ENC_BUF(enc, packed_conic, 6);
        ENC_BUF(enc, packed_rgb, 7);
        if (gpuResidentIntersectionAttempt) {
            [enc setBuffer:g_tcache.tile_layout_metadata.buffer()
                    offset:0 atIndex:8];
        } else {
            ENC_SCALAR(enc, total_intersections, 8);
        }
        if (gpuResidentIntersectionAttempt) {
            [enc dispatchThreadgroupsWithIndirectBuffer:
                    g_tcache.tile_attempt_dispatch_control.buffer()
                indirectBufferOffset:6u * sizeof(uint32_t)
                threadsPerThreadgroup:MTLSizeMake(packTpg, 1, 1)];
        } else {
            [enc dispatchThreads:MTLSizeMake(total_intersections, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(packTpg, 1, 1)];
        }
    };

    auto encode_rast_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (K_max <= 1) {
            // Monolithic
            const uint32_t blockX = ctx->monolithic_raster_block_x;
            const uint32_t blockY = ctx->monolithic_raster_block_y;
            MTLSize num_tg = MTLSizeMake(
                (img_width + blockX - 1) / blockX,
                (img_height + blockY - 1) / blockY, 1);
            [enc setComputePipelineState:ctx->nd_rasterize_forward_kernel_cpso];
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
            [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
            ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, raster_xy_attributes, 4);
            ENC_BUF(enc, raster_conic_attributes, 5);
            ENC_BUF(enc, raster_rgb_attributes, 6);
            ENC_BUF(enc, final_Ts, 7); ENC_BUF(enc, final_idx, 8); ENC_BUF(enc, out_img, 9);
            ENC_BUF(enc, background, 10);
            [enc setBytes:monolithic_block_size_dim2->data()
                   length:sizeof(*monolithic_block_size_dim2) atIndex:11];
            ENC_BUF(enc, g_tcache.intersection_keys_a, 12);
            ENC_BUF(enc, projected_opacities, 13);
            ENC_SCALAR(enc, intersection_attribute_layout, 14);
            [enc dispatchThreadgroups:num_tg
                threadsPerThreadgroup:MTLSizeMake(blockX, blockY, 1)];
        } else {
            // Chunked
            uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
            uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
            uint32_t num_pix = img_width * img_height;
            auto img_sz_2 = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
            [enc setComputePipelineState:ctx->rasterize_forward_chunked_kernel_cpso];
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
            [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
            ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, raster_xy_attributes, 4);
            ENC_BUF(enc, raster_conic_attributes, 5);
            ENC_BUF(enc, raster_rgb_attributes, 6);
            ENC_BUF(enc, g_tcache.chunk_T, 7); ENC_BUF(enc, g_tcache.chunk_C, 8); ENC_BUF(enc, g_tcache.chunk_final_idx, 9);
            ENC_SCALAR(enc, CHUNK_SIZE, 10); ENC_SCALAR(enc, K_max, 11);
            [enc setBytes:chunked_block_size_dim2->data()
                   length:sizeof(*chunked_block_size_dim2) atIndex:12];
            ENC_BUF(enc, g_tcache.intersection_keys_a, 13);
            ENC_BUF(enc, projected_opacities, 14);
            ENC_SCALAR(enc, intersection_attribute_layout, 15);
            [enc dispatchThreadgroups:MTLSizeMake(tile_x, tile_y, K_max) threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Merge
            [enc setComputePipelineState:ctx->rasterize_forward_merge_kernel_cpso];
            ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, K_max, 1);
            ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3); ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
            ENC_BUF(enc, final_Ts, 5); ENC_BUF(enc, final_idx, 6); ENC_BUF(enc, out_img, 7);
            ENC_BUF(enc, background, 8);
            [enc setBytes:img_sz_2->data() length:sizeof(*img_sz_2) atIndex:9];
            [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
    };

    // Fused loss: SSIM always begins with a horizontal image convolution.
    // The fused default keeps both derivative convolutions in one 16x8
    // threadgroup and writes only the final gradient. The staged fallback
    // materializes compact horizontal derivatives before its vertical pass.
    auto encode_loss_fwd_bwd = [&](id<MTLComputeCommandEncoder> enc) {
        MTLSize threadgroups = MTLSizeMake(
            (img_width + 15) / 16, (img_height + 15) / 16, 1);
        MTLSize tg = MTLSizeMake(16, 16, 1);
        // Pass 1: H conv on images → ssim_h_buf
        [enc setComputePipelineState:ctx->ssim_h_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:2];
        ENC_BUF(enc, g_tcache.ssim_h_buf, 3);
        ENC_BUF(enc, photometricLogGains, 4);
        ENC_SCALAR(enc, cameraGainOffset, 5);
        ENC_SCALAR(enc, photometricEnabled, 6);
        ENC_BUF(enc, loss_coverage_buffer, 7);
        ENC_SCALAR(enc, alpha_layout, 8);
        ENC_BUF(enc, background, 9);
        ENC_SCALAR(enc, target_pixel_stride_bytes, 10);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        if (ctx->fused_ssim_backward) {
            MTLSize fusedThreadgroups = MTLSizeMake(
                (img_width + 15) / 16, (img_height + 7) / 8, 1);
            MTLSize fusedTg = MTLSizeMake(16, 8, 1);
            [enc setComputePipelineState:
                ctx->ssim_fused_v_fwd_bwd_kernel_cpso];
            ENC_BUF(enc, rendered_gradient, 0); ENC_BUF(enc, gt, 1);
            ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
            [enc setBytes:loss_img_size->data()
                   length:sizeof(*loss_img_size) atIndex:3];
            ENC_SCALAR(enc, ssim_weight, 4); ENC_SCALAR(enc, loss_inv_n, 5);
            ENC_BUF(enc, loss_sum, 6);
            ENC_BUF(enc, loss_coverage_buffer, 7);
            ENC_SCALAR(enc, coverage_layout, 8);
            ENC_BUF(enc, photometricLogGains, 9);
            ENC_SCALAR(enc, cameraGainOffset, 10);
            ENC_BUF(enc, photometric_gradient, 11);
            ENC_SCALAR(enc, photometricEnabled, 12);
            ENC_SCALAR(enc, alpha_layout, 13);
            ENC_BUF(enc, background, 14);
            ENC_BUF(enc, final_Ts, 15);
            ENC_SCALAR(enc, alpha_loss_weight, 16);
            ENC_SCALAR(enc, target_pixel_stride_bytes, 17);
            [enc dispatchThreadgroups:fusedThreadgroups
                threadsPerThreadgroup:fusedTg];
        } else {
            // Pass 2: Fused V fwd + H bwd
            [enc setComputePipelineState:
                ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso];
            ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
            ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
            [enc setBytes:loss_img_size->data()
                   length:sizeof(*loss_img_size) atIndex:3];
            ENC_SCALAR(enc, ssim_weight, 4); ENC_SCALAR(enc, loss_inv_n, 5);
            ENC_BUF(enc, g_tcache.ssim_deriv_h_buf, 6);
            ENC_BUF(enc, loss_sum, 7);
            ENC_BUF(enc, loss_coverage_buffer, 8);
            ENC_SCALAR(enc, coverage_layout, 9);
            ENC_BUF(enc, photometricLogGains, 10);
            ENC_SCALAR(enc, cameraGainOffset, 11);
            ENC_SCALAR(enc, photometricEnabled, 12);
            ENC_SCALAR(enc, alpha_layout, 13);
            ENC_BUF(enc, background, 14);
            ENC_BUF(enc, final_Ts, 15);
            ENC_SCALAR(enc, alpha_loss_weight, 16);
            ENC_SCALAR(enc, target_pixel_stride_bytes, 17);
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:tg];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Pass 3: V bwd
            [enc setComputePipelineState:ctx->ssim_v_bwd_kernel_cpso];
            ENC_BUF(enc, rendered_gradient, 0); ENC_BUF(enc, gt, 1);
            ENC_BUF(enc, g_tcache.ssim_deriv_h_buf, 2);
            [enc setBytes:loss_img_size->data()
                   length:sizeof(*loss_img_size) atIndex:3];
            ENC_SCALAR(enc, ssim_weight, 4); ENC_SCALAR(enc, loss_inv_n, 5);
            ENC_BUF(enc, loss_coverage_buffer, 6);
            ENC_SCALAR(enc, coverage_layout, 7);
            ENC_BUF(enc, photometricLogGains, 8);
            ENC_SCALAR(enc, cameraGainOffset, 9);
            ENC_BUF(enc, photometric_gradient, 10);
            ENC_SCALAR(enc, photometricEnabled, 11);
            ENC_SCALAR(enc, alpha_layout, 12);
            ENC_BUF(enc, background, 13);
            ENC_SCALAR(enc, target_pixel_stride_bytes, 14);
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:tg];
        }

        if (photometric.enabled) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:ctx->photometric_adam_kernel_cpso];
            ENC_BUF(enc, photometricLogGains, 0);
            ENC_BUF(enc, photometric_gradient, 1);
            ENC_BUF(enc, photometricExpAvg, 2);
            ENC_BUF(enc, photometricExpAvgSq, 3);
            ENC_SCALAR(enc, cameraGainOffset, 4);
            ENC_SCALAR(enc, photometric.adamStepSize, 5);
            ENC_SCALAR(enc, adam_beta1, 6);
            ENC_SCALAR(enc, adam_beta2, 7);
            ENC_SCALAR(enc, photometric.adamBiasCorrection2Sqrt, 8);
            ENC_SCALAR(enc, adam_eps, 9);
            ENC_SCALAR(enc, photometric.regularization, 10);
            ENC_SCALAR(enc, photometric.maxAbsLogGain, 11);
            ENC_BUF(enc, overflow_flag, 12);
            ENC_SCALAR(enc, attemptGatingEnabled, 13);
            [enc dispatchThreads:MTLSizeMake(3, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(3, 1, 1)];
        }
    };

    auto encode_rast_bwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (bwd_K_max <= 1) {
            // Monolithic
            MTLSize num_tg = MTLSizeMake((img_width+RAST_BLOCK_X-1)/RAST_BLOCK_X, (img_height+RAST_BLOCK_Y-1)/RAST_BLOCK_Y, 1);
            [enc setComputePipelineState:ctx->rasterize_backward_kernel_cpso];
            [enc setBytes:rast_tb->data() length:sizeof(*rast_tb) atIndex:0];
            [enc setBytes:rast_isz->data() length:sizeof(*rast_isz) atIndex:1];
            ENC_BUF(enc, g_tcache.intersection_keys_a, 2);
            ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, raster_xy_attributes, 4);
            ENC_BUF(enc, raster_conic_attributes, 5);
            ENC_BUF(enc, raster_rgb_attributes, 6);
            ENC_BUF(enc, background, 7); ENC_BUF(enc, final_Ts, 8);
            ENC_BUF(enc, final_idx, 9); ENC_BUF(enc, rendered_gradient, 10);
            ENC_BUF(enc, v_xy, 11); ENC_BUF(enc, v_conic, 12);
            ENC_BUF(enc, v_colors_rast, 13); ENC_BUF(enc, v_opacity, 14);
            ENC_BUF(enc, loss_coverage_buffer, 15);
            ENC_SCALAR(enc, alpha_layout, 16);
            ENC_SCALAR(enc, alpha_gradient_scale, 17);
            ENC_BUF(enc, projected_opacities, 18);
            ENC_SCALAR(enc, intersection_attribute_layout, 19);
            [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        } else {
            // Chunked backward
            uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
            uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
            uint32_t num_pix = img_width * img_height;
            auto bwd_img_sz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
            // Phase 1: prefix_T and after_C
            [enc setComputePipelineState:ctx->compute_chunk_prefix_suffix_kernel_cpso];
            ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, bwd_K_max, 1);
            ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3);
            ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
            ENC_BUF(enc, g_tcache.prefix_T, 5); ENC_BUF(enc, g_tcache.after_C, 6);
            [enc setBytes:bwd_img_sz->data() length:sizeof(*bwd_img_sz) atIndex:7];
            [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Phase 2: backward chunked
            [enc setComputePipelineState:ctx->rasterize_backward_chunked_kernel_cpso];
            [enc setBytes:rast_tb->data() length:sizeof(*rast_tb) atIndex:0];
            [enc setBytes:rast_isz->data() length:sizeof(*rast_isz) atIndex:1];
            ENC_BUF(enc, g_tcache.intersection_keys_a, 2);
            ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, raster_xy_attributes, 4);
            ENC_BUF(enc, raster_conic_attributes, 5);
            ENC_BUF(enc, raster_rgb_attributes, 6);
            ENC_BUF(enc, background, 7); ENC_BUF(enc, final_Ts, 8);
            ENC_BUF(enc, g_tcache.chunk_final_idx, 9);
            ENC_BUF(enc, g_tcache.prefix_T, 10); ENC_BUF(enc, g_tcache.chunk_T, 11);
            ENC_BUF(enc, g_tcache.after_C, 12);
            ENC_BUF(enc, rendered_gradient, 13);
            ENC_BUF(enc, v_xy, 14); ENC_BUF(enc, v_conic, 15);
            ENC_BUF(enc, v_colors_rast, 16); ENC_BUF(enc, v_opacity, 17);
            ENC_SCALAR(enc, BWD_CHUNK_SIZE, 18); ENC_SCALAR(enc, bwd_K_max, 19);
            ENC_BUF(enc, loss_coverage_buffer, 20);
            ENC_SCALAR(enc, alpha_layout, 21);
            ENC_SCALAR(enc, alpha_gradient_scale, 22);
            ENC_BUF(enc, projected_opacities, 23);
            ENC_SCALAR(enc, intersection_attribute_layout, 24);
            [enc dispatchThreadgroups:MTLSizeMake(tile_x, tile_y, bwd_K_max) threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        }
    };

    // Packed optimizer hyperparameters. Layouts must match the Metal structs.
    struct SHOpacityAdamParams {
        float dc_step_size, dc_bc2_sqrt;
        float rest_step_size, rest_bc2_sqrt;
        float opacity_step_size, opacity_bc2_sqrt;
        float beta1, beta2, eps;
    };
    static_assert(sizeof(SHOpacityAdamParams) == 36);
    auto sh_adam_hp = std::make_shared<SHOpacityAdamParams>();
    sh_adam_hp->dc_step_size = adam_step_sizes[3];
    sh_adam_hp->dc_bc2_sqrt = adam_bc2_sqrts[3];
    sh_adam_hp->rest_step_size = adam_step_sizes[4];
    sh_adam_hp->rest_bc2_sqrt = adam_bc2_sqrts[4];
    sh_adam_hp->opacity_step_size = adam_step_sizes[5];
    sh_adam_hp->opacity_bc2_sqrt = adam_bc2_sqrts[5];
    sh_adam_hp->beta1 = adam_beta1;
    sh_adam_hp->beta2 = adam_beta2;
    sh_adam_hp->eps = adam_eps;

    struct GeometryAdamParams {
        float mean_step_size, mean_bc2_sqrt;
        float scale_step_size, scale_bc2_sqrt;
        float quat_step_size, quat_bc2_sqrt;
        float beta1, beta2, eps;
    };
    static_assert(sizeof(GeometryAdamParams) == 36);
    auto geometry_adam_hp = std::make_shared<GeometryAdamParams>();
    geometry_adam_hp->mean_step_size = adam_step_sizes[0];
    geometry_adam_hp->mean_bc2_sqrt = adam_bc2_sqrts[0];
    geometry_adam_hp->scale_step_size = adam_step_sizes[1];
    geometry_adam_hp->scale_bc2_sqrt = adam_bc2_sqrts[1];
    geometry_adam_hp->quat_step_size = adam_step_sizes[2];
    geometry_adam_hp->quat_bc2_sqrt = adam_bc2_sqrts[2];
    geometry_adam_hp->beta1 = adam_beta1;
    geometry_adam_hp->beta2 = adam_beta2;
    geometry_adam_hp->eps = adam_eps;

    struct PoseAdamParams {
        uint32_t pose_offset;
        float step_size;
        float beta1;
        float beta2;
        float bias_correction2_sqrt;
        float eps;
        float regularization;
        float max_translation;
        float max_rotation;
    };
    static_assert(sizeof(PoseAdamParams) == 36);
    auto pose_adam_hp = std::make_shared<PoseAdamParams>();
    if (pose.enabled) {
        pose_adam_hp->pose_offset = cameraPoseOffset;
        pose_adam_hp->step_size = pose.adamStepSize;
        pose_adam_hp->beta1 = adam_beta1;
        pose_adam_hp->beta2 = adam_beta2;
        pose_adam_hp->bias_correction2_sqrt =
            pose.adamBiasCorrection2Sqrt;
        pose_adam_hp->eps = adam_eps;
        pose_adam_hp->regularization = pose.regularization;
        pose_adam_hp->max_translation = pose.maxTranslation;
        pose_adam_hp->max_rotation = pose.maxRotation;
    }

    auto encode_proj_sh_bwd_adam = [&](id<MTLComputeCommandEncoder> enc) {
        // The attempt status is immutable after sort finalization. Publish it
        // before any kernel that can mutate model or optimizer state.
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        NSUInteger shTpg = MIN(
            ctx->sh_opacity_backward_adam_kernel_cpso.maxTotalThreadsPerThreadgroup,
            (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->sh_opacity_backward_adam_kernel_cpso];
        ENC_SCALAR(enc, num_points, 0);
        ENC_BUF(enc, means3d, 1); ENC_BUF(enc, radii_out, 2);
        ENC_SCALAR(enc, degree, 3); ENC_SCALAR(enc, degrees_to_use, 4);
        if (pose.enabled) {
            ENC_BUF(enc, g_tcache.pose_cam_pos, 5);
        } else {
            [enc setBytes:cam_pos_arr->data()
                   length:sizeof(*cam_pos_arr) atIndex:5];
        }
        ENC_BUF(enc, v_colors_rast, 6);
        ENC_BUF(enc, adam_params[3], 7); ENC_BUF(enc, adam_params[4], 8);
        ENC_BUF(enc, adam_exp_avg[3], 9); ENC_BUF(enc, adam_exp_avg_sq[3], 10);
        ENC_BUF(enc, adam_exp_avg[4], 11); ENC_BUF(enc, adam_exp_avg_sq[4], 12);
        ENC_BUF(enc, v_opacity, 13); ENC_BUF(enc, adam_params[5], 14);
        ENC_BUF(enc, adam_exp_avg[5], 15); ENC_BUF(enc, adam_exp_avg_sq[5], 16);
        [enc setBytes:sh_adam_hp.get()
               length:sizeof(SHOpacityAdamParams) atIndex:17];
        ENC_BUF(enc, overflow_flag, 18);
        ENC_SCALAR(enc, attemptGatingEnabled, 19);
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(shTpg, 1, 1)];

        // SH reads means from the pre-update model. Complete those reads before
        // the terminal geometry dispatch updates means in place.
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        NSUInteger geometryTpg = MIN(
            ctx->project_backward_adam_kernel_cpso.maxTotalThreadsPerThreadgroup,
            (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_backward_adam_kernel_cpso];
        ENC_SCALAR(enc, num_points, 0);
        ENC_BUF(enc, adam_params[0], 1); ENC_BUF(enc, adam_params[1], 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, adam_params[2], 4);
        ENC_BUF(enc, activeViewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_bwd_intr->data() length:sizeof(*proj_bwd_intr) atIndex:7];
        [enc setBytes:proj_bwd_isz->data() length:sizeof(*proj_bwd_isz) atIndex:8];
        ENC_BUF(enc, radii_out, 9); ENC_BUF(enc, conics, 10);
        ENC_BUF(enc, v_xy, 11); ENC_BUF(enc, v_conic, 12);
        ENC_BUF(enc, adam_exp_avg[0], 13); ENC_BUF(enc, adam_exp_avg_sq[0], 14);
        ENC_BUF(enc, adam_exp_avg[1], 15); ENC_BUF(enc, adam_exp_avg_sq[1], 16);
        ENC_BUF(enc, adam_exp_avg[2], 17); ENC_BUF(enc, adam_exp_avg_sq[2], 18);
        [enc setBytes:geometry_adam_hp.get()
               length:sizeof(GeometryAdamParams) atIndex:19];
        ENC_SCALAR(enc, poseEnabled, 20); ENC_BUF(enc, poseGradient, 21);
        ENC_SCALAR(enc, collectStats, 22);
        ENC_BUF(enc, visCountsBuffer, 23); ENC_BUF(enc, xysGradNormBuffer, 24);
        ENC_BUF(enc, max2DSizeBuffer, 25); ENC_SCALAR(enc, inv_max_dim, 26);
        ENC_BUF(enc, overflow_flag, 27);
        ENC_SCALAR(enc, attemptGatingEnabled, 28);
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(geometryTpg, 1, 1)];

        if (pose.enabled) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:ctx->camera_pose_adam_kernel_cpso];
            ENC_BUF(enc, poseDeltas, 0);
            ENC_BUF(enc, poseGradient, 1);
            ENC_BUF(enc, poseExpAvg, 2);
            ENC_BUF(enc, poseExpAvgSq, 3);
            [enc setBytes:pose_adam_hp.get()
                   length:sizeof(PoseAdamParams) atIndex:4];
            ENC_BUF(enc, overflow_flag, 5);
            ENC_SCALAR(enc, attemptGatingEnabled, 6);
            [enc dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        }
    };

    // ========================== DISPATCH ==========================

    // Blit-zero helper (shared by both paths)
    auto do_blit_zero = [&](id<MTLCommandBuffer> cb) -> bool {
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (!blit) return false;
        [blit fillBuffer:loss_sum.buffer() range:NSMakeRange(0, loss_sum.nbytes()) value:0];
        [blit fillBuffer:photometric_gradient.buffer()
                   range:NSMakeRange(0, photometric_gradient.nbytes()) value:0];
        if (pose.enabled) {
            [blit fillBuffer:poseGradient.buffer()
                       range:NSMakeRange(0, poseGradient.nbytes()) value:0];
        }
        [blit fillBuffer:g_tcache.tile_scatter_counters.buffer()
                   range:NSMakeRange(
                       0, g_tcache.tile_scatter_counters.nbytes()) value:0];
        [blit fillBuffer:v_xy.buffer() range:NSMakeRange(0, v_xy.nbytes()) value:0];
        [blit fillBuffer:v_conic.buffer() range:NSMakeRange(0, v_conic.nbytes()) value:0];
        [blit fillBuffer:v_colors_rast.buffer() range:NSMakeRange(0, v_colors_rast.nbytes()) value:0];
        [blit fillBuffer:v_opacity.buffer() range:NSMakeRange(0, v_opacity.nbytes()) value:0];
        [blit endEncoding];
        return true;
    };

    auto encode_retry_attempt_snapshot = [&](id<MTLCommandBuffer> cb) -> bool {
        if (!logicalStep) return true;
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (!blit) return false;
        [blit copyFromBuffer:g_tcache.tile_layout_metadata.buffer()
                sourceOffset:0
                    toBuffer:logicalStep->readbackBuffer().buffer()
           destinationOffset:kTrainingReadbackAttemptMetadataOffset
                        size:msplat::kTileIntersectionLayoutMetadataWordCount *
                             sizeof(uint32_t)];
        [blit copyFromBuffer:g_tcache.tile_offsets.buffer()
                sourceOffset:static_cast<NSUInteger>(num_tiles - 1) *
                             sizeof(int32_t)
                    toBuffer:logicalStep->readbackBuffer().buffer()
           destinationOffset:kTrainingReadbackIntersectionOffset
                        size:sizeof(int32_t)];
        [blit endEncoding];
        logicalStep->markRetryAttemptSnapshotEncoded(
            static_cast<size_t>(num_tiles));
        return true;
    };

    auto readCompletedRetryAttemptSnapshot = [&]() {
        if (logicalStep) {
            return logicalStep->completedRetryAttemptSnapshot();
        }

        // Direct native callers without a logical telemetry step still run
        // under the public engine lock. This fallback executes only after the
        // retry preflight's syncCB(), so copy the reusable globals by value
        // before deciding whether to replay the same call.
        RetryAttemptSnapshot snapshot;
        snapshot.failureReasons = overflow_flag.data<uint32_t>()[0];
        std::copy_n(
            g_tcache.tile_layout_metadata.data<uint32_t>(),
            snapshot.layoutMetadata.size(), snapshot.layoutMetadata.begin());
        snapshot.finalInclusiveOffset = num_tiles > 0
            ? g_tcache.tile_offsets.data<int32_t>()[num_tiles - 1]
            : 0;
        snapshot.tileCount = static_cast<size_t>(num_tiles);
        return snapshot;
    };

    auto inspectCompletedRetryAttempt = [&]() {
        constexpr uint32_t knownAttemptFailures =
            MSPLAT_TRAINING_OVERFLOW_TILE_CAP |
            MSPLAT_TRAINING_OVERFLOW_PACKED_CAPACITY;
        const RetryAttemptSnapshot snapshot =
            readCompletedRetryAttemptSnapshot();
        const msplat::TileIntersectionLayout completedLayout =
            msplat::tileIntersectionLayoutFromGpuMetadata(
                snapshot.layoutMetadata.data(),
                snapshot.layoutMetadata.size(), snapshot.tileCount,
                snapshot.finalInclusiveOffset);
        msplat::validateTileIntersectionWorkLimit(completedLayout);
        const uint32_t attemptFailures = snapshot.failureReasons;
        if ((attemptFailures & ~knownAttemptFailures) != 0u) {
            throw std::runtime_error(
                "GPU intersection attempt reported an unknown failure");
        }
        if ((attemptFailures & MSPLAT_TRAINING_OVERFLOW_TILE_CAP) != 0u) {
            // Layout errors and the bounded per-tile work limit are not fixed
            // by retaining a larger arena.
            throw std::length_error(
                "GPU intersection attempt exceeds the exact-sort work limit");
        }

        if ((attemptFailures &
             MSPLAT_TRAINING_OVERFLOW_PACKED_CAPACITY) != 0u) {
            const auto retryGrowStart = TelemetryClock::now();
            const bool needsRadixScratch =
                msplat::tileIntersectionLayoutNeedsRadixScratch(
                    completedLayout);
            bool grew = g_tcache.ensure_intersection_arena(
                completedLayout.totalCount, needsRadixScratch,
                !ctx->gather_intersection_attributes, ctx->device);

            const uint32_t requiredChunkCount =
                msplat::tileRasterChunkCount(
                    static_cast<uint32_t>(snapshot.tileCount),
                    completedLayout.maximumTileCount, CHUNK_SIZE);
            const int previousForwardChunkCount =
                g_tcache.forward_chunk_K_max;
            const int previousBackwardChunkCount =
                g_tcache.backward_chunk_K_max;
            g_tcache.ensure_forward_chunks(
                requiredChunkCount, img_height, img_width, ctx->device);
            g_tcache.ensure_backward_chunks(
                requiredChunkCount, img_height, img_width, ctx->device);
            grew = grew ||
                g_tcache.forward_chunk_K_max > previousForwardChunkCount ||
                g_tcache.backward_chunk_K_max > previousBackwardChunkCount;

            if (!grew) {
                throw std::runtime_error(
                    "GPU intersection attempt failed without a growable capacity shortfall");
            }
            const auto retryGrowEnd = TelemetryClock::now();
            if (logicalStep) {
                logicalStep->recordRecoveredIntersectionRetry(
                    attemptFailures,
                    elapsedMilliseconds(retryGrowStart, retryGrowEnd));
            }
            return true;
        }

        return false;
    };

    auto replaySameLogicalStep = [&]() -> MTensor {
        // The recursive call remains under the public operation's engine lock
        // and within this logical training step, so neither shared cache state
        // nor candidate Adam counters can advance between attempts.
        return msplat_train_step_locked(
            num_points, means3d, scales, glob_scale, quats, viewmat,
            projmat, fx, fy, cx, cy, img_height, img_width, tile_bounds,
            clip_thresh, degree, degrees_to_use, cam_pos, features_dc,
            features_rest, opacities, background, gt, coverage_mask,
            coverage_render_tiles, loss_coverage_units, ssim_weight,
            loss_inv_n, transparent_mask, alpha_loss_weight,
            num_adam_groups, adam_params, adam_exp_avg, adam_exp_avg_sq,
            adam_step_sizes, adam_bc2_sqrts, adam_beta1, adam_beta2,
            adam_eps, photometric, pose, collect_densification_stats,
            vis_counts, xys_grad_norm, max_2d_size, inv_max_dim);
    };

    if (ctx->retry_intersection_attempts) {
        id<MTLCommandBuffer> commandBuffer = ctx->getCommandBuffer();
        __block const char* encodingFailure = nullptr;
        dispatch_sync(ctx->d_queue, ^{
            if (!do_blit_zero(commandBuffer)) {
                encodingFailure =
                    "msplat: failed to create a Metal blit encoder";
                return;
            }

            id<MTLComputeCommandEncoder> enc = nil;
            if (profileThisStep) {
                MTLComputePassDescriptor *passDescriptor =
                    [MTLComputePassDescriptor computePassDescriptor];
                passDescriptor.sampleBufferAttachments[0].sampleBuffer =
                    stageCounterSampleBuffer;
                passDescriptor.sampleBufferAttachments[0]
                    .startOfEncoderSampleIndex = 2;
                passDescriptor.sampleBufferAttachments[0]
                    .endOfEncoderSampleIndex = 3;
                enc = [commandBuffer
                    computeCommandEncoderWithDescriptor:passDescriptor];
            } else {
                enc = [commandBuffer computeCommandEncoder];
            }
            if (!enc) {
                encodingFailure =
                    "msplat: failed to create a Metal compute encoder";
                return;
            }
            // Scatter and both sort paths are the final kernels allowed to
            // poison an attempt. Retire them before any model/Adam mutation;
            // packing and the expensive training tail can then stay queued.
            encode_scatter_sort_finalize(enc);
            [enc endEncoding];
            if (!encode_retry_attempt_snapshot(commandBuffer)) {
                encodingFailure =
                    "msplat: failed to create a retry snapshot blit encoder";
                return;
            }
        });
        if (encodingFailure) {
            ctx->discardCB();
            throw std::runtime_error(encodingFailure);
        }

        const SynchronousGpuMetrics preflightMetrics = ctx->syncCB();
        if (logicalStep) {
            // In retry mode this field represents the synchronized
            // count/layout/scatter/sort preflight as one GPU phase.
            logicalStep->recordExactCountPass(preflightMetrics);
        }
        if (inspectCompletedRetryAttempt())
            return replaySameLogicalStep();
    }

    auto encode_step_readback = [&](id<MTLCommandBuffer> cb) -> bool {
        if (!logicalStep) return true;
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (!blit) return false;
        [blit copyFromBuffer:loss_sum.buffer()
                sourceOffset:0
                    toBuffer:logicalStep->readbackBuffer().buffer()
           destinationOffset:kTrainingReadbackLossOffset
                        size:sizeof(float)];
        [blit copyFromBuffer:g_tcache.tile_offsets.buffer()
                sourceOffset:static_cast<NSUInteger>(num_tiles - 1) *
                             sizeof(int32_t)
                    toBuffer:logicalStep->readbackBuffer().buffer()
           destinationOffset:kTrainingReadbackIntersectionOffset
                        size:sizeof(int32_t)];
        [blit endEncoding];
        logicalStep->markReadbackEncoded(
            static_cast<uint64_t>(capacity_u32),
            loss_coverage_units);
        return true;
    };

    const auto postCountEncodeStart = TelemetryClock::now();
    if (profileThisStep) {
        // Projection was sampled in the exact-count command buffer. The
        // remaining stages use separate encoders on the final command buffer.
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        __block const char* encodingFailure = nullptr;

        id<MTLCounterSampleBuffer> csb = stageCounterSampleBuffer;
        auto csbOwner = stageCounterOwner;
        double ticksToMs = ctx->ticksToMs;

        dispatch_sync(ctx->d_queue, ^(){
            // Retry mode already cleared transients in its synchronized
            // preflight. Exact mode retains the established post-count clear.
            if (!ctx->retry_intersection_attempts &&
                !do_blit_zero(command_buffer)) {
                encodingFailure = "msplat: failed to create a Metal blit encoder";
                return;
            }

            // Projection already owns indices 0-1 from the count pass.
            auto make_profiled_encoder = [&](int stage_idx) -> id<MTLComputeCommandEncoder> {
                MTLComputePassDescriptor *passDesc = [MTLComputePassDescriptor computePassDescriptor];
                passDesc.sampleBufferAttachments[0].sampleBuffer = csb;
                passDesc.sampleBufferAttachments[0].startOfEncoderSampleIndex = stage_idx * 2;
                passDesc.sampleBufferAttachments[0].endOfEncoderSampleIndex = stage_idx * 2 + 1;
                return [command_buffer computeCommandEncoderWithDescriptor:passDesc];
            };

            id<MTLComputeCommandEncoder> enc;

            // Stage 2: exact scatter/sort/finalize. Retry mode sampled this
            // status-writing work in its synchronized preflight.
            if (!ctx->retry_intersection_attempts) {
                enc = make_profiled_encoder(1);
                if (!enc) {
                    encodingFailure =
                        "msplat: failed to create a profiled Metal compute encoder";
                    return;
                }
                encode_scatter_sort_finalize(enc);
                [enc endEncoding];
            }

            // Stage 3: attribute pack.
            enc = make_profiled_encoder(2);
            if (!enc) {
                encodingFailure = "msplat: failed to create a profiled Metal compute encoder";
                return;
            }
            encode_pack(enc);
            [enc endEncoding];

            // Stage 4: rast_fwd
            enc = make_profiled_encoder(3);
            if (!enc) {
                encodingFailure = "msplat: failed to create a profiled Metal compute encoder";
                return;
            }
            encode_rast_fwd(enc);
            [enc endEncoding];

            // Stage 5: loss_fwd_bwd
            enc = make_profiled_encoder(4);
            if (!enc) {
                encodingFailure = "msplat: failed to create a profiled Metal compute encoder";
                return;
            }
            encode_loss_fwd_bwd(enc);
            [enc endEncoding];

            // Stage 6: rast_bwd
            enc = make_profiled_encoder(5);
            if (!enc) {
                encodingFailure = "msplat: failed to create a profiled Metal compute encoder";
                return;
            }
            encode_rast_bwd(enc);
            [enc endEncoding];

            // Stage 7: proj_sh_bwd + Adam
            enc = make_profiled_encoder(6);
            if (!enc) {
                encodingFailure = "msplat: failed to create a profiled Metal compute encoder";
                return;
            }
            encode_proj_sh_bwd_adam(enc);
            [enc endEncoding];

            if (!encode_step_readback(command_buffer)) {
                encodingFailure =
                    "msplat: failed to create a training telemetry blit encoder";
                return;
            }
        });

        if (encodingFailure) {
            ctx->discardCB();
            throw std::runtime_error(encodingFailure);
        }

        // Add completion handler to read timestamps after GPU finishes
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            @autoreleasepool {
                (void)cb;
                (void)csbOwner;
                NSData *data = [csb resolveCounterRange:NSMakeRange(0, (N_TRAIN_STAGES - 1) * 2)];
                if (!data) return;
                const MTLCounterResultTimestamp *samples =
                    (const MTLCounterResultTimestamp *)[data bytes];

                std::lock_guard<std::mutex> lock(g_stage_timing_mutex);
                for (int i = 0; i < N_TRAIN_STAGES - 1; i++) {
                    uint64_t start = samples[i * 2].timestamp;
                    uint64_t end = samples[i * 2 + 1].timestamp;
                    if (start == MTLCounterErrorValue ||
                        end == MTLCounterErrorValue || end < start) {
                        continue;
                    }
                    // Counter stage i maps to name i+1 because blit_zero has no
                    // direct timestamp sample.
                    g_stage_times[i + 1].push_back((double)(end - start) * ticksToMs);
                }
                g_stage_report_count++;

                if (g_stage_report_count % 500 == 0) {
                    fprintf(stderr, "\n  === GPU Stage Profile (n=%d) ===\n", g_stage_report_count);
                    double total_median = 0;
                    for (int i = 1; i < N_TRAIN_STAGES; i++) {
                        auto &v = g_stage_times[i];
                        if (v.empty()) continue;
                        auto sorted = v;
                        std::sort(sorted.begin(), sorted.end());
                        double med = sorted[sorted.size() / 2];
                        double sum = 0;
                        for (auto x : sorted) sum += x;
                        total_median += med;
                        fprintf(stderr, "  %-20s median=%.3fms  mean=%.3fms\n",
                                trainingStageName(i), med, sum / sorted.size());
                    }
                    fprintf(stderr, "  %-20s %.3fms\n", "TOTAL (sum medians)", total_median);
                }
            }
        }];

    } else {
        // Production: the post-count work stays in one encoder.
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        __block const char* encodingFailure = nullptr;

        dispatch_sync(ctx->d_queue, ^(){
            if (!ctx->retry_intersection_attempts &&
                !do_blit_zero(command_buffer)) {
                encodingFailure = "msplat: failed to create a Metal blit encoder";
                return;
            }

            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            if (!enc) {
                encodingFailure = "msplat: failed to create a Metal compute encoder";
                return;
            }

            // --- Forward: exact sort/pack → raster → loss ---
            if (!ctx->retry_intersection_attempts)
                encode_scatter_sort_finalize(enc);
            encode_pack(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_rast_fwd(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // --- Fused loss forward + backward ---
            encode_loss_fwd_bwd(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_rast_bwd(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_proj_sh_bwd_adam(enc);

            [enc endEncoding];
            if (!encode_step_readback(command_buffer)) {
                encodingFailure =
                    "msplat: failed to create a training telemetry blit encoder";
                return;
            }
        });
        if (encodingFailure) {
            ctx->discardCB();
            throw std::runtime_error(encodingFailure);
        }
    }

    if (logicalStep) {
        logicalStep->recordPostCountEncode(
            postCountEncodeStart, TelemetryClock::now());
    }

    // Loss is copied into the step's unique readback and published only after
    // GPU completion; the synchronous return remains the densification radii.
    return radii_out;
}

MTensor msplat_train_step(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &gt, const MTensor* coverage_mask,
    const MTensor* coverage_render_tiles,
    uint64_t loss_coverage_units, float ssim_weight,
    float loss_inv_n, bool transparent_mask,
    float alpha_loss_weight,
    int num_adam_groups,
    MTensor adam_params[], MTensor adam_exp_avg[], MTensor adam_exp_avg_sq[],
    float adam_step_sizes[], float adam_bc2_sqrts[],
    float adam_beta1, float adam_beta2, float adam_eps,
    const MsplatPhotometricRefinementStep& photometric,
    const MsplatPoseRefinementStep& pose,
    bool collect_densification_stats,
    MTensor &vis_counts, MTensor &xys_grad_norm, MTensor &max_2d_size,
    float inv_max_dim
) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    return msplat_train_step_locked(
        num_points, means3d, scales, glob_scale, quats, viewmat, projmat,
        fx, fy, cx, cy, img_height, img_width, tile_bounds, clip_thresh,
        degree, degrees_to_use, cam_pos, features_dc, features_rest,
        opacities, background, gt, coverage_mask, coverage_render_tiles,
        loss_coverage_units, ssim_weight, loss_inv_n, transparent_mask,
        alpha_loss_weight, num_adam_groups, adam_params, adam_exp_avg,
        adam_exp_avg_sq, adam_step_sizes, adam_bc2_sqrts, adam_beta1,
        adam_beta2, adam_eps, photometric, pose, collect_densification_stats,
        vis_counts, xys_grad_norm, max_2d_size, inv_max_dim);
}

void msplat_reset_opacity_state(
    MTensor &opacities, MTensor &exp_avg, MTensor &exp_avg_sq,
    float max_logit
) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (!opacities.defined() || !exp_avg.defined() ||
        !exp_avg_sq.defined() || !opacities.isGpu() ||
        !exp_avg.isGpu() || !exp_avg_sq.isGpu() ||
        opacities.dtype() != DType::Float32 ||
        exp_avg.dtype() != DType::Float32 ||
        exp_avg_sq.dtype() != DType::Float32 ||
        opacities.numel() <= 0 ||
        exp_avg.numel() != opacities.numel() ||
        exp_avg_sq.numel() != opacities.numel()) {
        throw std::invalid_argument(
            "Opacity reset requires equally sized non-empty GPU Float32 tensors");
    }
    if (!std::isfinite(max_logit)) {
        throw std::invalid_argument("Opacity reset ceiling must be finite");
    }
    if (opacities.numel() > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Opacity reset exceeds the kernel index range");
    }

    MetalContext* ctx = get_global_context();
    const uint32_t count = static_cast<uint32_t>(opacities.numel());
    id<MTLCommandBuffer> commandBuffer = ctx->getCommandBuffer();
    __block bool encoderCreationFailed = false;
    dispatch_sync(ctx->d_queue, ^{
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (!encoder) {
            encoderCreationFailed = true;
            return;
        }
        [encoder setComputePipelineState:ctx->reset_opacity_state_kernel_cpso];
        ENC_BUF(encoder, opacities, 0);
        ENC_BUF(encoder, exp_avg, 1);
        ENC_BUF(encoder, exp_avg_sq, 2);
        ENC_SCALAR(encoder, count, 3);
        ENC_SCALAR(encoder, max_logit, 4);
        const NSUInteger threadsPerThreadgroup = MIN(
            ctx->reset_opacity_state_kernel_cpso.maxTotalThreadsPerThreadgroup,
            static_cast<NSUInteger>(count));
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:
                MTLSizeMake(threadsPerThreadgroup, 1, 1)];
        [encoder endEncoding];
    });

    if (encoderCreationFailed) {
        // The completed training update may already have advanced host-side
        // optimizer counters. Preserve that transaction: finish its GPU work,
        // then fall back to the established shared-memory maintenance path.
        ctx->syncCB();
        float* opacityValues = opacities.data<float>();
        for (uint32_t index = 0; index < count; ++index) {
            if (opacityValues[index] > max_logit)
                opacityValues[index] = max_logit;
        }
        std::memset(exp_avg.data_ptr(), 0, exp_avg.nbytes());
        std::memset(exp_avg_sq.data_ptr(), 0, exp_avg_sq.nbytes());
        fprintf(stderr,
                "msplat: opacity reset used the synchronous CPU fallback\n");
    }
}

// ============================================================================
// GPU-native densification (v34 Phase 3)
// Entire classify → grow → cull → compact pipeline in one compute encoder.
// Returns new num_active after densification.
// ============================================================================
// Classification and its prefix sums, split out so the caller learns how many
// gaussians will actually be written before it allocates room for them.
// Growing to the theoretical 3*N worst case instead costs three times the
// parameter and Adam buffers at every refine step, which is the difference
// between fitting on a phone and not.
void msplat_prepare_densify(
    int N, int max_population,
    float grad_thresh, float size_thresh, float screen_thresh, int check_screen,
    MTensor &xys_grad_norm, MTensor &vis_counts, MTensor &max_2d_size,
    float half_max_dim,
    MTensor &scales_buf,
    MTensor &split_flag, MTensor &dup_flag,
    MTensor &split_prefix, MTensor &dup_prefix,
    MTensor &block_totals,
    int &num_splits, int &num_dups
) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (N <= 0)
        throw std::invalid_argument("msplat_prepare_densify requires a positive population");
    if (max_population > 0 && max_population < N)
        throw std::invalid_argument("Densification population already exceeds its limit");

    auto requireElements = [](const MTensor &tensor, int64_t required,
                              const char *name) {
        if (!tensor.defined() || required < 0 || tensor.numel() < required)
            throw std::runtime_error(std::string("Densification buffer is too small: ") + name);
    };
    requireElements(xys_grad_norm, N, "xys_grad_norm");
    requireElements(vis_counts, N, "vis_counts");
    requireElements(max_2d_size, N, "max_2d_size");
    requireElements(scales_buf, 3LL * N, "scales");
    requireElements(split_flag, N, "split_flag");
    requireElements(dup_flag, N, "dup_flag");
    requireElements(split_prefix, N, "split_prefix");
    requireElements(dup_prefix, N, "dup_prefix");
    requireElements(block_totals,
                    (static_cast<int64_t>(N) + 1023) / 1024,
                    "block_totals");

    MetalContext* ctx = get_global_context();

    uint32_t N_u32 = (uint32_t)N;
    uint32_t K = static_cast<uint32_t>(
        (static_cast<int64_t>(N) + 1023) / 1024);
    int check_screen_int = check_screen;

    id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
    __block bool encoderCreationFailed = false;

    dispatch_sync(ctx->d_queue, ^(){
        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
        if (!enc) {
            encoderCreationFailed = true;
            return;
        }

        // ---- Stage 1: Classify (split/dup) ----
        {
            NSUInteger tpg = MIN(ctx->densify_classify_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
            [enc setComputePipelineState:ctx->densify_classify_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, xys_grad_norm, 1);
            ENC_BUF(enc, vis_counts, 2);
            ENC_BUF(enc, scales_buf, 3);
            ENC_BUF(enc, max_2d_size, 4);
            ENC_SCALAR(enc, half_max_dim, 5);
            ENC_SCALAR(enc, grad_thresh, 6);
            ENC_SCALAR(enc, size_thresh, 7);
            ENC_SCALAR(enc, screen_thresh, 8);
            ENC_SCALAR(enc, check_screen_int, 9);
            ENC_BUF(enc, split_flag, 10);
            ENC_BUF(enc, dup_flag, 11);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 2: Prefix sum on split_flag → split_prefix ----
        {
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, split_flag, 1);
            ENC_BUF(enc, block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, split_flag, 1);
            ENC_BUF(enc, split_prefix, 2); ENC_BUF(enc, block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 3: Prefix sum on dup_flag → dup_prefix ----
        {
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, dup_flag, 1);
            ENC_BUF(enc, block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, dup_flag, 1);
            ENC_BUF(enc, dup_prefix, 2); ENC_BUF(enc, block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [enc endEncoding];
    });

    if (encoderCreationFailed) {
        ctx->discardCB();
        throw std::runtime_error("msplat: failed to create a Metal compute encoder");
    }

    // The one readback: totals live in the last prefix entry.
    ctx->syncCB();
    num_splits = N > 0 ? split_prefix.data<int32_t>()[N - 1] : 0;
    num_dups = N > 0 ? dup_prefix.data<int32_t>()[N - 1] : 0;
    if (num_splits < 0 || num_splits > N || num_dups < 0 || num_dups > N ||
        num_splits > N - num_dups) {
        throw std::runtime_error("Densification classification produced invalid totals");
    }

    // A split writes two children before its parent is culled; a duplicate
    // writes one. Trim the classification itself so neither the temporary
    // population nor any backing allocation can cross the hard limit. Metal
    // buffers use shared storage, and syncCB completed the GPU writes above,
    // so these prefix arrays can be safely rewritten before the next command
    // buffer consumes them.
    const int64_t population = static_cast<int64_t>(N) +
        2LL * num_splits + num_dups;
    if (max_population > 0 && population > max_population) {
        struct Candidate {
            int index;
            int cost;
            float score;
        };

        int remaining = max_population - N;
        int32_t *split_flags = split_flag.data<int32_t>();
        int32_t *dup_flags = dup_flag.data<int32_t>();
        int32_t *split_prefixes = split_prefix.data<int32_t>();
        int32_t *dup_prefixes = dup_prefix.data<int32_t>();
        const float *gradients = xys_grad_norm.data<float>();
        const float *visibility = vis_counts.data<float>();
        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<size_t>(num_splits + num_dups));

        for (int index = 0; index < N; ++index) {
            const bool is_split = split_flags[index] != 0;
            const bool is_dup = dup_flags[index] != 0;
            if (is_split && is_dup)
                throw std::runtime_error("Densification classified one Gaussian twice");
            if (!is_split && !is_dup) continue;

            float score = gradients[index] / visibility[index];
            if (std::isnan(score)) score = -std::numeric_limits<float>::infinity();
            candidates.push_back({index, is_split ? 2 : 1, score});
            split_flags[index] = 0;
            dup_flags[index] = 0;
        }
        if (candidates.size() != static_cast<size_t>(num_splits + num_dups))
            throw std::runtime_error("Densification flags do not match their prefix totals");

        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate &lhs, const Candidate &rhs) {
            if (lhs.score > rhs.score) return true;
            if (lhs.score < rhs.score) return false;
            return lhs.index < rhs.index;
        });

        for (const Candidate &candidate : candidates) {
            if (candidate.cost > remaining) continue;
            if (candidate.cost == 2) {
                split_flags[candidate.index] = 1;
            } else {
                dup_flags[candidate.index] = 1;
            }
            remaining -= candidate.cost;
            if (remaining == 0) break;
        }

        int accepted_splits = 0;
        int accepted_dups = 0;
        for (int index = 0; index < N; ++index) {
            accepted_splits += split_flags[index];
            accepted_dups += dup_flags[index];
            split_prefixes[index] = accepted_splits;
            dup_prefixes[index] = accepted_dups;
        }

        num_splits = accepted_splits;
        num_dups = accepted_dups;
    }
}

// Runs the prepared grow -> cull -> compact pipeline. `population` is
// N + 2*splits + dups as reported by msplat_prepare_densify; the buffers must
// already be that large. Returns the active count after culling.
int msplat_densify(
    int N, int population,
    float cull_alpha_thresh, float cull_scale_thresh, float cull_screen_size,
    int check_screen, int check_huge,
    MTensor &max_2d_size,
    MTensor &means_buf, MTensor &scales_buf, MTensor &quats_buf,
    MTensor &featuresDc_buf, MTensor &featuresRest_buf, MTensor &opacities_buf,
    int fr_stride,
    MTensor adam_exp_avg_buf[], MTensor adam_exp_avg_sq_buf[],
    MTensor &split_flag, MTensor &dup_flag,
    MTensor &split_prefix, MTensor &dup_prefix,
    MTensor &keep_flag, MTensor &keep_prefix,
    MTensor &block_totals, MTensor &compact_scratch,
    MTensor &random_samples, uint32_t random_seed
) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (N <= 0 || population < N || fr_stride < 0)
        throw std::invalid_argument("Invalid densification dimensions");

    // Strides for each of the 18 buffers (6 params + 12 optimizer states)
    // Order: means(3), scales(3), quats(4), featuresDc(3), featuresRest(fr_stride), opacities(1)
    int max_stride = std::max(fr_stride, 4);

    // Collect all 18 buffers in order for compact loops (std::array for block capture)
    std::array<MTensor*, 18> all_bufs = {{
        &means_buf, &scales_buf, &quats_buf, &featuresDc_buf, &featuresRest_buf, &opacities_buf,
        &adam_exp_avg_buf[0], &adam_exp_avg_buf[1], &adam_exp_avg_buf[2],
        &adam_exp_avg_buf[3], &adam_exp_avg_buf[4], &adam_exp_avg_buf[5],
        &adam_exp_avg_sq_buf[0], &adam_exp_avg_sq_buf[1], &adam_exp_avg_sq_buf[2],
        &adam_exp_avg_sq_buf[3], &adam_exp_avg_sq_buf[4], &adam_exp_avg_sq_buf[5]
    }};
    std::array<int, 18> all_strides = {{
        3, 3, 4, 3, fr_stride, 1,
        3, 3, 4, 3, fr_stride, 1,
        3, 3, 4, 3, fr_stride, 1
    }};

    auto requireElements = [](const MTensor &tensor, int64_t required,
                              const char *name) {
        if (!tensor.defined() || required < 0 || tensor.numel() < required)
            throw std::runtime_error(std::string("Densification buffer is too small: ") + name);
    };

    requireElements(split_flag, N, "split_flag");
    requireElements(dup_flag, N, "dup_flag");
    requireElements(split_prefix, N, "split_prefix");
    requireElements(dup_prefix, N, "dup_prefix");
    requireElements(max_2d_size, N, "max_2d_size");
    requireElements(keep_flag, population, "keep_flag");
    requireElements(keep_prefix, population, "keep_prefix");
    requireElements(block_totals,
                    (static_cast<int64_t>(population) + 1023) / 1024,
                    "block_totals");

    const int num_splits = split_prefix.data<int32_t>()[N - 1];
    const int num_dups = dup_prefix.data<int32_t>()[N - 1];
    if (num_splits < 0 || num_splits > N || num_dups < 0 || num_dups > N ||
        num_splits > N - num_dups ||
        static_cast<int64_t>(N) + 2LL * num_splits + num_dups != population) {
        throw std::runtime_error("Densification population does not match its prefixes");
    }

    const int64_t compact_elements = static_cast<int64_t>(population) * max_stride;
    requireElements(compact_scratch, compact_elements, "compact_scratch");
    if (compact_elements > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("Densification compaction exceeds the kernel index range");

    for (size_t index = 0; index < all_bufs.size(); ++index) {
        if (all_bufs[index]->stride0() != all_strides[index])
            throw std::runtime_error("Densification buffer has an unexpected row stride");
        requireElements(*all_bufs[index],
                        static_cast<int64_t>(population) * all_strides[index],
                        "parameter or optimizer state");
    }

    MetalContext* ctx = get_global_context();
    const uint32_t generate_random_on_gpu =
        ctx->gpu_densify_random ? 1u : 0u;
    requireElements(
        random_samples,
        generate_random_on_gpu != 0u ? 1LL : 6LL * num_splits,
        "random_samples");
    if (generate_random_on_gpu == 0u) {
        // Preserve the legacy libc++ stream exactly in the default mode. The
        // classification pass has already synchronized shared storage.
        std::mt19937 generator(random_seed);
        std::normal_distribution<float> distribution(0.0f, 1.0f);
        float* samples = random_samples.data<float>();
        for (int64_t index = 0; index < 6LL * num_splits; ++index)
            samples[index] = distribution(generator);
    }
    int worst_case = population;
    float log_size_fac = std::log(1.6f);

    uint32_t N_u32 = (uint32_t)N;
    int check_screen_int = check_screen;
    int check_huge_int = check_huge;

    id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
    __block bool encoderCreationFailed = false;

    dispatch_sync(ctx->d_queue, ^(){
        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
        if (!enc) {
            encoderCreationFailed = true;
            return;
        }

        // ---- Stage 4: Append split children ----
        {
            NSUInteger tpg = MIN(ctx->densify_append_split_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
            [enc setComputePipelineState:ctx->densify_append_split_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, split_flag, 1);
            ENC_BUF(enc, split_prefix, 2);
            ENC_BUF(enc, random_samples, 3);
            ENC_SCALAR(enc, log_size_fac, 4);
            ENC_BUF(enc, means_buf, 5);
            ENC_BUF(enc, scales_buf, 6);
            ENC_BUF(enc, quats_buf, 7);
            ENC_BUF(enc, featuresDc_buf, 8);
            ENC_BUF(enc, featuresRest_buf, 9);
            ENC_BUF(enc, opacities_buf, 10);
            int fr_stride_val = fr_stride;
            ENC_SCALAR(enc, fr_stride_val, 11);
            ENC_BUF(enc, adam_exp_avg_buf[0], 12);
            ENC_BUF(enc, adam_exp_avg_buf[1], 13);
            ENC_BUF(enc, adam_exp_avg_buf[2], 14);
            ENC_BUF(enc, adam_exp_avg_buf[3], 15);
            ENC_BUF(enc, adam_exp_avg_buf[4], 16);
            ENC_BUF(enc, adam_exp_avg_buf[5], 17);
            ENC_BUF(enc, adam_exp_avg_sq_buf[0], 18);
            ENC_BUF(enc, adam_exp_avg_sq_buf[1], 19);
            ENC_BUF(enc, adam_exp_avg_sq_buf[2], 20);
            ENC_BUF(enc, adam_exp_avg_sq_buf[3], 21);
            ENC_BUF(enc, adam_exp_avg_sq_buf[4], 22);
            ENC_BUF(enc, adam_exp_avg_sq_buf[5], 23);
            ENC_SCALAR(enc, random_seed, 24);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 5: Append duplicates ----
        {
            NSUInteger tpg = MIN(ctx->densify_append_dup_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
            [enc setComputePipelineState:ctx->densify_append_dup_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, dup_flag, 1);
            ENC_BUF(enc, dup_prefix, 2);
            ENC_BUF(enc, split_prefix, 3);
            ENC_BUF(enc, means_buf, 4);
            ENC_BUF(enc, scales_buf, 5);
            ENC_BUF(enc, quats_buf, 6);
            ENC_BUF(enc, featuresDc_buf, 7);
            ENC_BUF(enc, featuresRest_buf, 8);
            ENC_BUF(enc, opacities_buf, 9);
            int fr_stride_val = fr_stride;
            ENC_SCALAR(enc, fr_stride_val, 10);
            ENC_BUF(enc, adam_exp_avg_buf[0], 11);
            ENC_BUF(enc, adam_exp_avg_buf[1], 12);
            ENC_BUF(enc, adam_exp_avg_buf[2], 13);
            ENC_BUF(enc, adam_exp_avg_buf[3], 14);
            ENC_BUF(enc, adam_exp_avg_buf[4], 15);
            ENC_BUF(enc, adam_exp_avg_buf[5], 16);
            ENC_BUF(enc, adam_exp_avg_sq_buf[0], 17);
            ENC_BUF(enc, adam_exp_avg_sq_buf[1], 18);
            ENC_BUF(enc, adam_exp_avg_sq_buf[2], 19);
            ENC_BUF(enc, adam_exp_avg_sq_buf[3], 20);
            ENC_BUF(enc, adam_exp_avg_sq_buf[4], 21);
            ENC_BUF(enc, adam_exp_avg_sq_buf[5], 22);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 6: Cull classify (on post-growth population) ----
        // Dispatch worst_case threads; kernel reads N_new from prefix sums
        {
            uint32_t wc = (uint32_t)worst_case;
            NSUInteger tpg = MIN(ctx->densify_cull_classify_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)worst_case);
            [enc setComputePipelineState:ctx->densify_cull_classify_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, split_prefix, 1);
            ENC_BUF(enc, dup_prefix, 2);
            ENC_BUF(enc, split_flag, 3);
            ENC_BUF(enc, opacities_buf, 4);
            ENC_BUF(enc, scales_buf, 5);
            ENC_BUF(enc, max_2d_size, 6);
            ENC_SCALAR(enc, cull_alpha_thresh, 7);
            ENC_SCALAR(enc, cull_scale_thresh, 8);
            ENC_SCALAR(enc, cull_screen_size, 9);
            ENC_SCALAR(enc, check_huge_int, 10);
            ENC_SCALAR(enc, check_screen_int, 11);
            ENC_BUF(enc, keep_flag, 12);
            [enc dispatchThreads:MTLSizeMake(worst_case, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 7: Prefix sum on keep_flag → keep_prefix ----
        // Over worst_case elements (includes padding zeros for unused slots)
        {
            uint32_t wc = (uint32_t)worst_case;
            uint32_t K2 = static_cast<uint32_t>(
                (static_cast<int64_t>(worst_case) + 1023) / 1024);
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, wc, 0); ENC_BUF(enc, keep_flag, 1);
            ENC_BUF(enc, block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K2, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            uint32_t wc = (uint32_t)worst_case;
            uint32_t K2 = static_cast<uint32_t>(
                (static_cast<int64_t>(worst_case) + 1023) / 1024);
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, wc, 0); ENC_BUF(enc, keep_flag, 1);
            ENC_BUF(enc, keep_prefix, 2); ENC_BUF(enc, block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K2, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 8: Compact scatter (18 buffers → scratch) ----
        // For each buffer: scatter kept elements into compact_scratch
        // Then copy back. We reuse compact_scratch at different offsets per stride.
        for (int b = 0; b < 18; b++) {
            // SH degree 0 has no rest coefficients, so this logical buffer has
            // a zero row stride and requires no compaction dispatch.
            if (all_strides[b] == 0) continue;
            uint32_t wc = (uint32_t)worst_case;
            uint32_t stride_u32 = (uint32_t)all_strides[b];
            uint32_t total_threads = wc * stride_u32;
            NSUInteger tpg = MIN(ctx->compact_scatter_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)total_threads);
            [enc setComputePipelineState:ctx->compact_scatter_kernel_cpso];
            [enc setBuffer:all_bufs[b]->buffer() offset:0 atIndex:0];
            ENC_BUF(enc, compact_scratch, 1);
            ENC_BUF(enc, keep_prefix, 2);
            ENC_BUF(enc, keep_flag, 3);
            ENC_SCALAR(enc, wc, 4);
            ENC_SCALAR(enc, stride_u32, 5);
            [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Copy back from scratch to buffer
            uint32_t last_idx = wc - 1;
            [enc setComputePipelineState:ctx->compact_copy_back_kernel_cpso];
            ENC_BUF(enc, compact_scratch, 0);
            [enc setBuffer:all_bufs[b]->buffer() offset:0 atIndex:1];
            ENC_BUF(enc, keep_prefix, 2);
            ENC_SCALAR(enc, last_idx, 3);
            ENC_SCALAR(enc, stride_u32, 4);
            [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        [enc endEncoding];
    });

    if (encoderCreationFailed) {
        ctx->discardCB();
        throw std::runtime_error("msplat: failed to create a Metal compute encoder");
    }

    // Single GPU→CPU sync: read new_count from keep_prefix[worst_case - 1]
    ctx->syncCB();
    int new_count = keep_prefix.data<int32_t>()[worst_case - 1];
    return new_count;
}
