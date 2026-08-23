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
#import <stdexcept>
#import <string>
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
    "blit_zero", "proj_sh_fwd", "exact_sort_pack", "rast_fwd",
    "loss_fwd_bwd", "rast_bwd", "proj_sh_bwd_adam", "grad_stats"
};
static constexpr int N_TRAIN_STAGES = 8;

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
static constexpr size_t kTrainingReadbackWordCount = 4;
static constexpr size_t kTrainingReadbackBytes =
    kTrainingReadbackWordCount * sizeof(uint32_t);
static constexpr NSUInteger kTrainingReadbackLossOffset = sizeof(uint32_t);
static constexpr NSUInteger kTrainingReadbackIntersectionOffset =
    2 * sizeof(uint32_t);

double elapsedMilliseconds(TelemetryClock::time_point start,
                           TelemetryClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

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
                          bool intersectionCountValid) noexcept {
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
                MSPLAT_TRAINING_TELEMETRY_INTERSECTION_COUNT_VALID);
            if (gpuTimingValid)
                snapshot.flags |= MSPLAT_TRAINING_TELEMETRY_GPU_TIMING_VALID;
            if (lossValid)
                snapshot.flags |= MSPLAT_TRAINING_TELEMETRY_LOSS_VALID;
            if (intersectionCountValid) {
                snapshot.flags |=
                    MSPLAT_TRAINING_TELEMETRY_INTERSECTION_COUNT_VALID;
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
                              MTensor stepReadback)
        : telemetry(std::move(telemetryState)),
          generation(stepGeneration), iteration(stepIteration),
          wallStart(stepWallStart),
          readback(std::move(stepReadback)) {}

    MTensor& readbackBuffer() { return readback; }

    void markCpuStart() {
        std::lock_guard<std::mutex> lock(mutex);
        if (sealed || aborted)
            throw std::logic_error("Training telemetry step is no longer active");
        cpuStart = TelemetryClock::now();
        cpuStarted = true;
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
            }
            if (pendingCommandBuffers > 0) --pendingCommandBuffers;
            finishIfReadyLocked();
        } catch (...) {
            // A C++ exception must never escape a Metal completion handler.
        }
    }

    void markReadbackEncoded(uint64_t capacity, uint64_t pixelCount) {
        std::lock_guard<std::mutex> lock(mutex);
        packedIntersectionCapacity = capacity;
        lossPixelCount = pixelCount;
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

            bool lossValid = false;
            bool intersectionCountValid = false;
            if (readbackEncoded && readback.defined()) {
                const auto* words = readback.data<uint32_t>();
                completed.overflowReasons = words[0] & (
                    MSPLAT_TRAINING_OVERFLOW_TILE_CAP |
                    MSPLAT_TRAINING_OVERFLOW_PACKED_CAPACITY);

                float rawLoss = 0.0f;
                std::memcpy(&rawLoss, &words[1], sizeof(rawLoss));
                if (lossPixelCount > 0 && std::isfinite(rawLoss)) {
                    completed.loss =
                        static_cast<double>(rawLoss) /
                        static_cast<double>(lossPixelCount);
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
            telemetry->publishCompleted(
                generation, completed, completedGpuTimingValid, lossValid,
                intersectionCountValid);
        }

        telemetry->releaseReadback(std::move(readback));
    }

    MsplatTrainingTelemetryHandle telemetry;
    uint64_t generation = 0;
    int64_t iteration = 0;
    TelemetryClock::time_point wallStart;
    TelemetryClock::time_point cpuStart;
    MTensor readback;
    mutable std::mutex mutex;
    MsplatTrainingStepDescriptor descriptor;
    size_t pendingCommandBuffers = 0;
    uint32_t commandBufferCount = 0;
    uint64_t packedIntersectionCapacity = 0;
    uint64_t lossPixelCount = 0;
    double cpuSubmitMs = 0.0;
    double synchronousGpuWaitMs = 0.0;
    double gpuExecutionMs = 0.0;
    bool cpuStarted = false;
    bool gpuTimingValid = true;
    bool readbackEncoded = false;
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
    void syncCB() {
        if (!g_gpu_timing_checked) {
            g_gpu_timing_enabled = std::getenv("PROFILE_GPU") != nullptr;
            g_gpu_timing_checked = true;
        }
        const bool collectTiming = g_gpu_timing_enabled;
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
    }

    // Per-stage GPU timestamp profiling (Metal counter sample buffer)
    id<MTLCounterSampleBuffer> counterSampleBuffer = nil;
    bool counterSamplingAvailable = false;
    double ticksToMs = 0.0;  // conversion factor from GPU ticks to milliseconds

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

        MTLCounterSampleBufferDescriptor *desc = [MTLCounterSampleBufferDescriptor new];
        ScopedObjCRelease descOwner{desc};
        desc.counterSet = timestampSet;
        desc.sampleCount = sampleCount;
        desc.storageMode = MTLStorageModeShared;
        desc.label = @"msplat stage profiling";

        NSError *error = nil;
        counterSampleBuffer = [device newCounterSampleBufferWithDescriptor:desc error:&error];
        if (!counterSampleBuffer) {
            fprintf(stderr, "PROFILE_STAGES: Failed to create counter sample buffer: %s\n",
                    error.localizedDescription.UTF8String);
            return;
        }

        // Compute ticks-to-ms conversion (Apple Silicon: mach_absolute_time units)
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        ticksToMs = (double)tb.numer / (double)tb.denom / 1e6;

        counterSamplingAvailable = true;
        fprintf(stderr, "PROFILE_STAGES: GPU timestamp profiling enabled (%lu sample slots)\n",
                (unsigned long)sampleCount);
    }

    // Forward pipeline kernels
    id<MTLComputePipelineState> project_and_sh_forward_kernel_cpso = nil;
    id<MTLComputePipelineState> nd_rasterize_forward_kernel_cpso = nil;
    // Exact tile-intersection compaction and sorting
    id<MTLComputePipelineState> scatter_to_exact_bins_kernel_cpso = nil;
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
    id<MTLComputePipelineState> ssim_v_bwd_kernel_cpso = nil;
    // Backward pipeline kernels
    id<MTLComputePipelineState> project_and_sh_backward_kernel_cpso = nil;
    id<MTLComputePipelineState> fused_adam_kernel_cpso = nil;
    id<MTLComputePipelineState> accumulate_grad_stats_kernel_cpso = nil;
    // GPU densification kernels
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
            counterSampleBuffer,
            project_and_sh_forward_kernel_cpso,
            nd_rasterize_forward_kernel_cpso,
            scatter_to_exact_bins_kernel_cpso,
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
            ssim_v_bwd_kernel_cpso,
            project_and_sh_backward_kernel_cpso,
            fused_adam_kernel_cpso,
            accumulate_grad_stats_kernel_cpso,
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

    // Forward pipeline
    ctx->project_and_sh_forward_kernel_cpso       = load(@"project_and_sh_forward_kernel");
    ctx->nd_rasterize_forward_kernel_cpso         = load(@"nd_rasterize_forward_kernel");
    // Exact tile-intersection compaction and sorting
    ctx->scatter_to_exact_bins_kernel_cpso        = load(@"scatter_to_exact_bins_kernel");
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
    ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso       = load(@"ssim_fused_v_fwd_h_bwd_kernel");
    ctx->ssim_v_bwd_kernel_cpso                   = load(@"ssim_v_bwd_kernel");
    // Backward pipeline
    ctx->project_and_sh_backward_kernel_cpso      = load(@"project_and_sh_backward_kernel");
    ctx->fused_adam_kernel_cpso                    = load(@"fused_adam_kernel");
    ctx->accumulate_grad_stats_kernel_cpso        = load(@"accumulate_grad_stats_kernel");
    // GPU densification
    ctx->densify_classify_kernel_cpso             = load(@"densify_classify_kernel");
    ctx->densify_append_split_kernel_cpso         = load(@"densify_append_split_kernel");
    ctx->densify_append_dup_kernel_cpso           = load(@"densify_append_dup_kernel");
    ctx->densify_cull_classify_kernel_cpso        = load(@"densify_cull_classify_kernel");
    ctx->compact_scatter_kernel_cpso              = load(@"compact_scatter_kernel");
    ctx->compact_copy_back_kernel_cpso            = load(@"compact_copy_back_kernel");

    if (pipelineLoadFailed) {
        throw std::runtime_error(pipelineFailure);
    }
    if (ctx->radix_sort_per_tile_kernel_cpso.maxTotalThreadsPerThreadgroup <
        256) {
        throw std::runtime_error(
            "msplat: exact radix sort requires 256 threads per threadgroup");
    }

    // Initialize counter sampling if PROFILE_STAGES is set
    ctx->counterSampleBuffer = nil;
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

id<MTLDevice> msplat_device() {
    return get_global_context()->device;
}

MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype) {
    return mtensor_zeros(get_global_context()->device, std::move(shape), dtype);
}

MTensor gpu_empty(std::vector<int64_t> shape, DType dtype) {
    return mtensor_empty(get_global_context()->device, std::move(shape), dtype);
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
    auto step = std::make_shared<MsplatLogicalTrainingStep>(
        telemetry, telemetry->currentGeneration(), iteration, wallStart,
        std::move(readback));
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
        stage_names[i] = g_train_stage_names[i];
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
    int training_img_height = -1, training_img_width = -1;
    int bwd_num_points = -1;
    int64_t capacity = -1;

    // Forward intermediates
    MTensor xys, depths, radii_out, conics, colors, aabb;
    MTensor gaussian_ids;
    MTensor packed_xy_opac, packed_conic, packed_rgb;
    MTensor out_img, final_Ts, final_idx;
    MTensor tile_bins;

    // Exact tile-intersection layout and sorting buffers
    MTensor tile_offsets, tile_scatter_counters;
    MTensor intersection_keys_a, intersection_keys_b;

    // Defensive invariant check: exact sizing should make this remain zero.
    MTensor overflow_flag;

    // Shared forward depth-chunked rasterization buffers
    int forward_chunk_K_max = -1, forward_chunk_height = -1, forward_chunk_width = -1;
    MTensor chunk_T, chunk_C, chunk_final_idx;

    // Training-only image and backward depth-chunked buffers
    MTensor ssim_deriv_h_buf, ssim_h_buf, loss_sum;
    int backward_chunk_K_max = -1, backward_chunk_height = -1, backward_chunk_width = -1;
    MTensor prefix_T, after_C;

    // Training-only backward gradient accumulators
    MTensor v_xy, v_conic, v_colors_rast, v_opacity, v_depth;
    MTensor v_mean3d, v_scale, v_quat;

    size_t sharedEstimatedBytes() const {
        const MTensor* tensors[] = {
            &xys, &depths, &radii_out, &conics, &colors, &aabb,
            &gaussian_ids, &packed_xy_opac, &packed_conic, &packed_rgb,
            &out_img, &final_Ts, &final_idx,
            &tile_bins, &tile_offsets, &tile_scatter_counters,
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
            &prefix_T, &after_C,
            &v_xy, &v_conic, &v_colors_rast, &v_opacity, &v_depth,
            &v_mean3d, &v_scale, &v_quat
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
        gaussian_ids.reset(); packed_xy_opac.reset();
        packed_conic.reset(); packed_rgb.reset();
    }

    void ensure_shared_forward(int np, int ih, int iw, int nt,
                               id<MTLDevice> dev) {
        const bool resolutionChanged =
            ih != shared_img_height || iw != shared_img_width || nt != num_tiles;
        if (resolutionChanged) resetIntersectionArena();

        if (np != fwd_num_points) {
            fwd_num_points = -1;
            xys.reset(); depths.reset(); radii_out.reset(); conics.reset();
            colors.reset(); aabb.reset();

            try {
                xys = mtensor_empty(dev, {np, 2}, DType::Float32);
                depths = mtensor_empty(dev, {np}, DType::Float32);
                radii_out = mtensor_empty(dev, {np}, DType::Int32);
                conics = mtensor_empty(dev, {np, 3}, DType::Float32);
                colors = mtensor_empty(dev, {np, 3}, DType::Float32);
                aabb = mtensor_empty(dev, {np, 2}, DType::Float32);
            } catch (...) {
                xys.reset(); depths.reset(); radii_out.reset(); conics.reset();
                colors.reset(); aabb.reset();
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

            try {
                tile_bins = mtensor_empty(dev, {nt, 2}, DType::Int32);
                tile_offsets = mtensor_empty(dev, {nt}, DType::Int32);
                tile_scatter_counters =
                    mtensor_empty(dev, {nt}, DType::Int32);
            } catch (...) {
                tile_bins.reset(); tile_offsets.reset();
                tile_scatter_counters.reset();
                throw;
            }
            num_tiles = nt;
        }
        if (!overflow_flag.defined()) {
            overflow_flag = mtensor_empty(dev, {1}, DType::Int32);
        }
    }

    void ensure_intersection_arena(uint32_t requiredCount,
                                   id<MTLDevice> dev) {
        const uint32_t currentCapacity = capacity > 0
            ? static_cast<uint32_t>(capacity)
            : 0;
        const uint32_t requestedCapacity =
            msplat::tileIntersectionArenaCapacity(
                requiredCount, currentCapacity, false);
        if (requestedCapacity == currentCapacity &&
            intersection_keys_a.defined() && intersection_keys_b.defined() &&
            gaussian_ids.defined() && packed_xy_opac.defined() &&
            packed_conic.defined() && packed_rgb.defined()) {
            return;
        }

        const uint64_t largestBufferBytes =
            static_cast<uint64_t>(requestedCapacity) * 3 * sizeof(float);
        if (largestBufferBytes > static_cast<uint64_t>(dev.maxBufferLength)) {
            throw std::length_error(
                "Exact tile-intersection arena exceeds Metal's maximum buffer length");
        }

        resetIntersectionArena();
        const int64_t cap = requestedCapacity;
        try {
            intersection_keys_a = mtensor_empty(dev, {cap}, DType::Int64);
            intersection_keys_b = mtensor_empty(dev, {cap}, DType::Int64);
            gaussian_ids = mtensor_empty(dev, {cap}, DType::Int32);
            packed_xy_opac = mtensor_empty(dev, {cap, 3}, DType::Float32);
            packed_conic = mtensor_empty(dev, {cap, 3}, DType::Float32);
            packed_rgb = mtensor_empty(dev, {cap, 3}, DType::Float32);
        } catch (...) {
            resetIntersectionArena();
            throw;
        }
        capacity = cap;
    }

    void ensure_training_image(int ih, int iw, id<MTLDevice> dev) {
        if (ih == training_img_height && iw == training_img_width &&
            ssim_deriv_h_buf.defined() && ssim_h_buf.defined() &&
            loss_sum.defined()) {
            return;
        }

        training_img_height = -1; training_img_width = -1;
        ssim_deriv_h_buf.reset(); ssim_h_buf.reset();
        loss_sum.reset();

        ssim_deriv_h_buf = mtensor_empty(
            dev, {(int64_t)ih, (int64_t)iw, 9}, DType::Float32);
        ssim_h_buf = mtensor_empty(
            dev, {(int64_t)ih, (int64_t)iw, 15}, DType::Float32);
        loss_sum = mtensor_empty(dev, {1}, DType::Float32);
        training_img_height = ih; training_img_width = iw;
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
            !v_colors_rast.defined() || !v_opacity.defined() ||
            !v_depth.defined() || !v_mean3d.defined() ||
            !v_scale.defined() || !v_quat.defined()) {
            bwd_num_points = -1;
            v_xy.reset(); v_conic.reset(); v_colors_rast.reset(); v_opacity.reset();
            v_depth.reset(); v_mean3d.reset(); v_scale.reset(); v_quat.reset();

            v_xy = mtensor_empty(dev, {np, 2}, DType::Float32);
            v_conic = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_colors_rast = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_opacity = mtensor_empty(dev, {np, 1}, DType::Float32);
            v_depth = mtensor_empty(dev, {np}, DType::Float32);
            v_mean3d = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_scale = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_quat = mtensor_empty(dev, {np, 4}, DType::Float32);
            bwd_num_points = np;
        }
    }
};
static FusedTensorCache g_tcache;

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
    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &conics = g_tcache.conics;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
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
    auto cam_pos_arr = std::make_shared<std::array<float, 3>>(std::array<float, 3>{cam_pos[0], cam_pos[1], cam_pos[2]});
    uint32_t num_points_u32 = (uint32_t)num_points;
    auto img_size_dim3 = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{img_width, img_height, 1, 0xDEAD});
    auto block_size_dim2 = std::make_shared<std::array<int32_t, 2>>(std::array<int32_t, 2>{RAST_BLOCK_X, RAST_BLOCK_Y});

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
        ENC_BUF(enc, g_tcache.tile_scatter_counters, 15);
        ENC_SCALAR(enc, degree, 16); ENC_SCALAR(enc, degrees_to_use, 17);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:18];
        ENC_BUF(enc, features_dc, 19); ENC_BUF(enc, features_rest, 20);
        ENC_BUF(enc, colors, 21); ENC_BUF(enc, aabb, 22);
        // buffer 23 removed (was opacity-aware AABB, reverted)

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
            [blit fillBuffer:g_tcache.tile_scatter_counters.buffer()
                       range:NSMakeRange(
                           0, g_tcache.tile_scatter_counters.nbytes())
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
            [encoder endEncoding];
        });
        if (encodingFailure) {
            ctx->discardCB();
            throw std::runtime_error(encodingFailure);
        }
        ctx->syncCB();
    }

    const msplat::TileIntersectionLayout intersectionLayout =
        msplat::buildTileIntersectionLayout(
            g_tcache.tile_scatter_counters.data<uint32_t>(),
            g_tcache.tile_offsets.data<int32_t>(),
            static_cast<size_t>(num_tiles));
    msplat::validateTileIntersectionWorkLimit(intersectionLayout);
    g_tcache.ensure_intersection_arena(
        intersectionLayout.totalCount, ctx->device);

    MTensor &gaussian_ids = g_tcache.gaussian_ids;
    MTensor &packed_xy_opac = g_tcache.packed_xy_opac;
    MTensor &packed_conic = g_tcache.packed_conic;
    MTensor &packed_rgb = g_tcache.packed_rgb;
    const uint32_t capacity_u32 = static_cast<uint32_t>(g_tcache.capacity);
    const uint32_t num_tiles_u32 = static_cast<uint32_t>(num_tiles);
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
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(scatterTpg, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:ctx->radix_sort_per_tile_kernel_cpso];
        ENC_BUF(enc, g_tcache.tile_offsets, 0);
        ENC_BUF(enc, g_tcache.intersection_keys_a, 1);
        ENC_BUF(enc, g_tcache.intersection_keys_b, 2);
        ENC_SCALAR(enc, num_tiles_u32, 3);
        ENC_BUF(enc, tile_bins, 4);
        ENC_SCALAR(enc, capacity_u32, 5);
        ENC_BUF(enc, g_tcache.overflow_flag, 6);
        [enc dispatchThreadgroups:MTLSizeMake(num_tiles, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (total_intersections == 0) return;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        NSUInteger packTpg = MIN(
            ctx->pack_sorted_gaussians_kernel_cpso.maxTotalThreadsPerThreadgroup,
            static_cast<NSUInteger>(total_intersections));
        [enc setComputePipelineState:ctx->pack_sorted_gaussians_kernel_cpso];
        ENC_BUF(enc, g_tcache.intersection_keys_a, 0);
        ENC_BUF(enc, xys, 1); ENC_BUF(enc, conics, 2);
        ENC_BUF(enc, colors, 3); ENC_BUF(enc, opacities, 4);
        ENC_BUF(enc, gaussian_ids, 5); ENC_BUF(enc, packed_xy_opac, 6);
        ENC_BUF(enc, packed_conic, 7); ENC_BUF(enc, packed_rgb, 8);
        ENC_SCALAR(enc, total_intersections, 9);
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
        fprintf(stderr, "  sort:           tile-local exact radix\n");
        fprintf(stderr, "  sort arenas:    %.1f MB\n",
                static_cast<double>(capacity_u32) * 16.0 / 1e6);
        fprintf(stderr, "===========================\n\n");
    }

    // K_max for chunked rasterization — set after GPU readback
    uint32_t K_max = 1;
    constexpr uint32_t CHUNK_SIZE = 512;

    auto encode_rast_fwd_monolithic = [&](id<MTLComputeCommandEncoder> enc) {
        MTLSize num_tg = MTLSizeMake((img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X, (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y, 1);
        MTLSize tg_size = MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1);
        [enc setComputePipelineState:ctx->nd_rasterize_forward_kernel_cpso];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
        [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
        ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
        ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
        ENC_BUF(enc, final_Ts, 7); ENC_BUF(enc, final_idx, 8); ENC_BUF(enc, out_img, 9);
        ENC_BUF(enc, background, 10);
        [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:11];
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
        ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
        ENC_BUF(enc, g_tcache.chunk_T, 7); ENC_BUF(enc, g_tcache.chunk_C, 8); ENC_BUF(enc, g_tcache.chunk_final_idx, 9);
        ENC_SCALAR(enc, CHUNK_SIZE, 10); ENC_SCALAR(enc, K_max, 11);
        [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:12];
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

MTensor msplat_train_step(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &gt, float ssim_weight,
    float loss_inv_n,
    int num_adam_groups,
    MTensor adam_params[], MTensor adam_exp_avg[], MTensor adam_exp_avg_sq[],
    float adam_step_sizes[], float adam_bc2_sqrts[],
    float adam_beta1, float adam_beta2, float adam_eps,
    bool collect_densification_stats,
    MTensor &vis_counts, MTensor &xys_grad_norm, MTensor &max_2d_size,
    float inv_max_dim
) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    MetalContext* ctx = get_global_context();
    auto logicalStep = ctx->activeTrainingStep;
    int tile_bounds_x = std::get<0>(tile_bounds);
    int tile_bounds_y = std::get<1>(tile_bounds);
    int num_tiles = tile_bounds_x * tile_bounds_y;

    if (!g_profile_stages_checked) {
        g_profile_stages = std::getenv("PROFILE_STAGES") != nullptr;
        g_profile_stages_checked = true;
    }
    uint32_t channels = 3;

    // --- Cached buffer pool ---
    g_tcache.ensure_shared_forward(
        num_points, img_height, img_width, num_tiles, ctx->device);
    g_tcache.ensure_training_image(img_height, img_width, ctx->device);
    g_tcache.ensure_backward(num_points, ctx->device);

    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &conics = g_tcache.conics;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
    MTensor &tile_bins = g_tcache.tile_bins;
    MTensor &loss_sum = g_tcache.loss_sum;
    MTensor &overflow_flag = logicalStep
        ? logicalStep->readbackBuffer()
        : g_tcache.overflow_flag;
    MTensor &out_img = g_tcache.out_img;
    MTensor &final_Ts = g_tcache.final_Ts;
    MTensor &final_idx = g_tcache.final_idx;
    MTensor &ssim_deriv_h_buf = g_tcache.ssim_deriv_h_buf;

    // SSIM V-backward reads and writes only the same pixel. Training no longer
    // needs the rendered image afterward, so reuse it in place for dL/dRGB.
    MTensor &rendered_gradient = out_img;
    MTensor &v_xy = g_tcache.v_xy;
    MTensor &v_conic = g_tcache.v_conic;
    MTensor &v_colors_rast = g_tcache.v_colors_rast;
    MTensor &v_opacity = g_tcache.v_opacity;
    MTensor &v_depth = g_tcache.v_depth;
    MTensor &v_mean3d = g_tcache.v_mean3d;
    MTensor &v_scale = g_tcache.v_scale;
    MTensor &v_quat = g_tcache.v_quat;
    // Wire backward outputs as Adam grads (MTensor references for gradient buffers)
    auto adam_grads = std::make_shared<std::array<MTensor, 6>>(
        std::array<MTensor, 6>{
            v_mean3d, v_scale, v_quat, MTensor{}, MTensor{}, v_opacity});

    // --- Constants (heap-allocated for Obj-C block capture) ---
    auto loss_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto proj_intrins = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto tile_bounds_arr = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (uint32_t)tile_bounds_x, (uint32_t)tile_bounds_y,
        (uint32_t)std::get<2>(tile_bounds), 0xDEAD
    });
    auto cam_pos_arr = std::make_shared<std::array<float, 3>>(std::array<float, 3>{cam_pos[0], cam_pos[1], cam_pos[2]});
    uint32_t num_points_u32 = (uint32_t)num_points;
    auto img_size_dim3 = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{img_width, img_height, 1, 0xDEAD});
    auto block_size_dim2 = std::make_shared<std::array<int32_t, 2>>(std::array<int32_t, 2>{RAST_BLOCK_X, RAST_BLOCK_Y});
    // tile_bounds for rasterize kernels must be 16x16 tile counts (tile_bins granularity)
    auto rast_tb = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (img_width + 15u) / 16u,
        (img_height + 15u) / 16u, 1, 0xDEAD});
    auto rast_isz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto proj_bwd_intr = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_bwd_isz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});

    // ========================== FORWARD ENCODE LAMBDAS ==========================

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
        ENC_BUF(enc, g_tcache.tile_scatter_counters, 15);
        ENC_SCALAR(enc, degree, 16); ENC_SCALAR(enc, degrees_to_use, 17);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:18];
        ENC_BUF(enc, features_dc, 19); ENC_BUF(enc, features_rest, 20);
        ENC_BUF(enc, colors, 21); ENC_BUF(enc, aabb, 22);
        // buffer 23 removed (was opacity-aware AABB, reverted)

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
            [blit fillBuffer:g_tcache.tile_scatter_counters.buffer()
                       range:NSMakeRange(
                           0, g_tcache.tile_scatter_counters.nbytes())
                       value:0];
            [blit endEncoding];

            id<MTLComputeCommandEncoder> encoder = nil;
            if (g_profile_stages && ctx->counterSamplingAvailable) {
                MTLComputePassDescriptor *passDescriptor =
                    [MTLComputePassDescriptor computePassDescriptor];
                passDescriptor.sampleBufferAttachments[0].sampleBuffer =
                    ctx->counterSampleBuffer;
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
            encode_proj_sh(encoder);
            [encoder endEncoding];
        });
        if (encodingFailure) {
            ctx->discardCB();
            throw std::runtime_error(encodingFailure);
        }
        ctx->syncCB();
    }

    const msplat::TileIntersectionLayout intersectionLayout =
        msplat::buildTileIntersectionLayout(
            g_tcache.tile_scatter_counters.data<uint32_t>(),
            g_tcache.tile_offsets.data<int32_t>(),
            static_cast<size_t>(num_tiles));
    msplat::validateTileIntersectionWorkLimit(intersectionLayout);
    g_tcache.ensure_intersection_arena(
        intersectionLayout.totalCount, ctx->device);

    MTensor &gaussian_ids = g_tcache.gaussian_ids;
    MTensor &packed_xy_opac = g_tcache.packed_xy_opac;
    MTensor &packed_conic = g_tcache.packed_conic;
    MTensor &packed_rgb = g_tcache.packed_rgb;
    const uint32_t capacity_u32 = static_cast<uint32_t>(g_tcache.capacity);
    const uint32_t num_tiles_u32 = static_cast<uint32_t>(num_tiles);
    const uint32_t total_intersections = intersectionLayout.totalCount;

    uint32_t K_max = 1;
    constexpr uint32_t CHUNK_SIZE = 512;
    K_max = msplat::tileRasterChunkCount(
        static_cast<uint32_t>(num_tiles),
        intersectionLayout.maximumTileCount, CHUNK_SIZE);
    g_tcache.ensure_forward_chunks(K_max, img_height, img_width, ctx->device);
    g_tcache.ensure_backward_chunks(K_max, img_height, img_width, ctx->device);

    uint32_t bwd_K_max = K_max;
    constexpr uint32_t BWD_CHUNK_SIZE = 512;

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
        ENC_BUF(enc, overflow_flag, 10);
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(scatterTpg, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:ctx->radix_sort_per_tile_kernel_cpso];
        ENC_BUF(enc, g_tcache.tile_offsets, 0);
        ENC_BUF(enc, g_tcache.intersection_keys_a, 1);
        ENC_BUF(enc, g_tcache.intersection_keys_b, 2);
        ENC_SCALAR(enc, num_tiles_u32, 3);
        ENC_BUF(enc, tile_bins, 4);
        ENC_SCALAR(enc, capacity_u32, 5);
        ENC_BUF(enc, overflow_flag, 6);
        [enc dispatchThreadgroups:MTLSizeMake(num_tiles, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (total_intersections == 0) return;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        NSUInteger packTpg = MIN(
            ctx->pack_sorted_gaussians_kernel_cpso.maxTotalThreadsPerThreadgroup,
            static_cast<NSUInteger>(total_intersections));
        [enc setComputePipelineState:ctx->pack_sorted_gaussians_kernel_cpso];
        ENC_BUF(enc, g_tcache.intersection_keys_a, 0);
        ENC_BUF(enc, xys, 1); ENC_BUF(enc, conics, 2);
        ENC_BUF(enc, colors, 3); ENC_BUF(enc, opacities, 4);
        ENC_BUF(enc, gaussian_ids, 5); ENC_BUF(enc, packed_xy_opac, 6);
        ENC_BUF(enc, packed_conic, 7); ENC_BUF(enc, packed_rgb, 8);
        ENC_SCALAR(enc, total_intersections, 9);
        [enc dispatchThreads:MTLSizeMake(total_intersections, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(packTpg, 1, 1)];
    };

    auto encode_rast_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (K_max <= 1) {
            // Monolithic
            MTLSize num_tg = MTLSizeMake((img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X, (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y, 1);
            [enc setComputePipelineState:ctx->nd_rasterize_forward_kernel_cpso];
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
            [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
            ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, final_Ts, 7); ENC_BUF(enc, final_idx, 8); ENC_BUF(enc, out_img, 9);
            ENC_BUF(enc, background, 10);
            [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:11];
            [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
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
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, g_tcache.chunk_T, 7); ENC_BUF(enc, g_tcache.chunk_C, 8); ENC_BUF(enc, g_tcache.chunk_final_idx, 9);
            ENC_SCALAR(enc, CHUNK_SIZE, 10); ENC_SCALAR(enc, K_max, 11);
            [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:12];
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

    // Fused loss: ssim_h_fwd → fused_v_fwd_h_bwd → ssim_v_bwd.
    // The fused middle pass writes only its compact horizontal derivatives.
    auto encode_loss_fwd_bwd = [&](id<MTLComputeCommandEncoder> enc) {
        MTLSize threadgroups = MTLSizeMake(
            (img_width + 15) / 16, (img_height + 15) / 16, 1);
        MTLSize tg = MTLSizeMake(16, 16, 1);
        // Pass 1: H conv on images → ssim_h_buf
        [enc setComputePipelineState:ctx->ssim_h_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:2];
        ENC_BUF(enc, g_tcache.ssim_h_buf, 3);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // Pass 2: Fused V fwd + H bwd
        [enc setComputePipelineState:ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4); ENC_SCALAR(enc, loss_inv_n, 5);
        ENC_BUF(enc, ssim_deriv_h_buf, 6); ENC_BUF(enc, loss_sum, 7);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // Pass 3: V bwd
        [enc setComputePipelineState:ctx->ssim_v_bwd_kernel_cpso];
        ENC_BUF(enc, rendered_gradient, 0); ENC_BUF(enc, gt, 1);
        ENC_BUF(enc, ssim_deriv_h_buf, 2);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4); ENC_SCALAR(enc, loss_inv_n, 5);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:tg];
    };

    auto encode_rast_bwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (bwd_K_max <= 1) {
            // Monolithic
            MTLSize num_tg = MTLSizeMake((img_width+RAST_BLOCK_X-1)/RAST_BLOCK_X, (img_height+RAST_BLOCK_Y-1)/RAST_BLOCK_Y, 1);
            [enc setComputePipelineState:ctx->rasterize_backward_kernel_cpso];
            [enc setBytes:rast_tb->data() length:sizeof(*rast_tb) atIndex:0];
            [enc setBytes:rast_isz->data() length:sizeof(*rast_isz) atIndex:1];
            ENC_BUF(enc, gaussian_ids, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5);
            ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, background, 7); ENC_BUF(enc, final_Ts, 8);
            ENC_BUF(enc, final_idx, 9); ENC_BUF(enc, rendered_gradient, 10);
            ENC_BUF(enc, v_xy, 11); ENC_BUF(enc, v_conic, 12);
            ENC_BUF(enc, v_colors_rast, 13); ENC_BUF(enc, v_opacity, 14);
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
            ENC_BUF(enc, gaussian_ids, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5);
            ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, background, 7); ENC_BUF(enc, final_Ts, 8);
            ENC_BUF(enc, g_tcache.chunk_final_idx, 9);
            ENC_BUF(enc, g_tcache.prefix_T, 10); ENC_BUF(enc, g_tcache.chunk_T, 11);
            ENC_BUF(enc, g_tcache.after_C, 12);
            ENC_BUF(enc, rendered_gradient, 13);
            ENC_BUF(enc, v_xy, 14); ENC_BUF(enc, v_conic, 15);
            ENC_BUF(enc, v_colors_rast, 16); ENC_BUF(enc, v_opacity, 17);
            ENC_SCALAR(enc, BWD_CHUNK_SIZE, 18); ENC_SCALAR(enc, bwd_K_max, 19);
            [enc dispatchThreadgroups:MTLSizeMake(tile_x, tile_y, bwd_K_max) threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        }
    };

    // Packed SH Adam hyperparameters (must match SHAdamParams in .metal)
    struct SHAdamParams {
        float dc_step_size, dc_bc2_sqrt;
        float rest_step_size, rest_bc2_sqrt;
        float beta1, beta2, eps;
    };
    auto sh_adam_hp = std::make_shared<SHAdamParams>();
    if (num_adam_groups >= 5) {
        sh_adam_hp->dc_step_size = adam_step_sizes[3];
        sh_adam_hp->dc_bc2_sqrt = adam_bc2_sqrts[3];
        sh_adam_hp->rest_step_size = adam_step_sizes[4];
        sh_adam_hp->rest_bc2_sqrt = adam_bc2_sqrts[4];
        sh_adam_hp->beta1 = adam_beta1;
        sh_adam_hp->beta2 = adam_beta2;
        sh_adam_hp->eps = adam_eps;
    }

    auto encode_proj_sh_bwd_adam = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_backward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_backward_kernel_cpso];
        ENC_SCALAR(enc, num_points, 0); ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_bwd_intr->data() length:sizeof(*proj_bwd_intr) atIndex:7];
        [enc setBytes:proj_bwd_isz->data() length:sizeof(*proj_bwd_isz) atIndex:8];
        ENC_BUF(enc, radii_out, 9); ENC_BUF(enc, conics, 10);
        ENC_BUF(enc, v_xy, 11); ENC_BUF(enc, v_depth, 12); ENC_BUF(enc, v_conic, 13);
        ENC_BUF(enc, v_mean3d, 14); ENC_BUF(enc, v_scale, 15); ENC_BUF(enc, v_quat, 16);
        ENC_SCALAR(enc, degree, 17); ENC_SCALAR(enc, degrees_to_use, 18);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:19];
        ENC_BUF(enc, v_colors_rast, 20);
        // Fused SH backward + Adam: pass params + optimizer state instead of gradient buffers
        [enc setBuffer:adam_params[3].buffer() offset:0 atIndex:21];  // features_dc params
        [enc setBuffer:adam_params[4].buffer() offset:0 atIndex:22];  // features_rest params
        [enc setBuffer:adam_exp_avg[3].buffer() offset:0 atIndex:23]; // dc exp_avg
        [enc setBuffer:adam_exp_avg_sq[3].buffer() offset:0 atIndex:24]; // dc exp_avg_sq
        [enc setBuffer:adam_exp_avg[4].buffer() offset:0 atIndex:25]; // rest exp_avg
        [enc setBuffer:adam_exp_avg_sq[4].buffer() offset:0 atIndex:26]; // rest exp_avg_sq
        [enc setBytes:sh_adam_hp.get() length:sizeof(SHAdamParams) atIndex:27];
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        // Adam for remaining groups (skip 3=featuresDc, 4=featuresRest — fused above)
        if (num_adam_groups > 0) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            for (int g = 0; g < num_adam_groups; ++g) {
                if (g == 3 || g == 4) continue;  // fused into backward kernel
                uint32_t n = adam_params[g].numel();
                if (n == 0) continue;
                NSUInteger atpg = MIN(ctx->fused_adam_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
                [enc setComputePipelineState:ctx->fused_adam_kernel_cpso];
                [enc setBuffer:adam_params[g].buffer() offset:0 atIndex:0];
                [enc setBuffer:(*adam_grads)[g].buffer() offset:0 atIndex:1];
                [enc setBuffer:adam_exp_avg[g].buffer() offset:0 atIndex:2];
                [enc setBuffer:adam_exp_avg_sq[g].buffer() offset:0 atIndex:3];
                ENC_SCALAR(enc, adam_step_sizes[g], 4);
                ENC_SCALAR(enc, adam_beta1, 5);
                ENC_SCALAR(enc, adam_beta2, 6);
                ENC_SCALAR(enc, adam_bc2_sqrts[g], 7);
                ENC_SCALAR(enc, adam_eps, 8);
                ENC_SCALAR(enc, n, 9);
                [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(atpg, 1, 1)];
            }
        }
    };

    // ========================== DISPATCH ==========================

    // Encode accumulate_grad_stats as a lambda (shared by both paths)
    auto encode_grad_stats = [&](id<MTLComputeCommandEncoder> enc) {
        if (!collect_densification_stats) return;
        NSUInteger tpg = MIN(ctx->accumulate_grad_stats_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->accumulate_grad_stats_kernel_cpso];
        ENC_SCALAR(enc, num_points, 0);
        ENC_BUF(enc, radii_out, 1);
        ENC_BUF(enc, v_xy, 2);
        ENC_BUF(enc, vis_counts, 3);
        ENC_BUF(enc, xys_grad_norm, 4);
        ENC_BUF(enc, max_2d_size, 5);
        ENC_SCALAR(enc, inv_max_dim, 6);
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    // Blit-zero helper (shared by both paths)
    auto do_blit_zero = [&](id<MTLCommandBuffer> cb) -> bool {
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (!blit) return false;
        [blit fillBuffer:loss_sum.buffer() range:NSMakeRange(0, loss_sum.nbytes()) value:0];
        [blit fillBuffer:overflow_flag.buffer()
                   range:NSMakeRange(0, sizeof(uint32_t)) value:0];
        [blit fillBuffer:g_tcache.tile_scatter_counters.buffer()
                   range:NSMakeRange(
                       0, g_tcache.tile_scatter_counters.nbytes()) value:0];
        [blit fillBuffer:v_xy.buffer() range:NSMakeRange(0, v_xy.nbytes()) value:0];
        [blit fillBuffer:v_conic.buffer() range:NSMakeRange(0, v_conic.nbytes()) value:0];
        [blit fillBuffer:v_colors_rast.buffer() range:NSMakeRange(0, v_colors_rast.nbytes()) value:0];
        [blit fillBuffer:v_opacity.buffer() range:NSMakeRange(0, v_opacity.nbytes()) value:0];
        [blit fillBuffer:v_depth.buffer() range:NSMakeRange(0, v_depth.nbytes()) value:0];
        [blit fillBuffer:v_mean3d.buffer() range:NSMakeRange(0, v_mean3d.nbytes()) value:0];
        [blit fillBuffer:v_scale.buffer() range:NSMakeRange(0, v_scale.nbytes()) value:0];
        [blit fillBuffer:v_quat.buffer() range:NSMakeRange(0, v_quat.nbytes()) value:0];
        [blit endEncoding];
        return true;
    };

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
            static_cast<uint64_t>(img_height) *
                static_cast<uint64_t>(img_width));
        return true;
    };

    if (g_profile_stages && ctx->counterSamplingAvailable) {
        // Projection was sampled in the exact-count command buffer. The
        // remaining stages use separate encoders on the final command buffer.
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        __block const char* encodingFailure = nullptr;

        id<MTLCounterSampleBuffer> csb = ctx->counterSampleBuffer;
        double ticksToMs = ctx->ticksToMs;

        dispatch_sync(ctx->d_queue, ^(){
            // Blit zero has no direct timestamp sample.
            if (!do_blit_zero(command_buffer)) {
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

            // Stage 2: exact scatter, radix sort, and pack
            enc = make_profiled_encoder(1);
            if (!enc) {
                encodingFailure = "msplat: failed to create a profiled Metal compute encoder";
                return;
            }
            encode_sort_pack(enc);
            [enc endEncoding];

            // Stage 3: rast_fwd
            enc = make_profiled_encoder(2);
            if (!enc) {
                encodingFailure = "msplat: failed to create a profiled Metal compute encoder";
                return;
            }
            encode_rast_fwd(enc);
            [enc endEncoding];

            // Stage 4+5: loss_fwd_bwd (fused)
            enc = make_profiled_encoder(3);
            if (!enc) {
                encodingFailure = "msplat: failed to create a profiled Metal compute encoder";
                return;
            }
            encode_loss_fwd_bwd(enc);
            [enc endEncoding];

            // Stage 5: rast_bwd
            enc = make_profiled_encoder(4);
            if (!enc) {
                encodingFailure = "msplat: failed to create a profiled Metal compute encoder";
                return;
            }
            encode_rast_bwd(enc);
            [enc endEncoding];

            // Stage 6: proj_sh_bwd + Adam
            enc = make_profiled_encoder(5);
            if (!enc) {
                encodingFailure = "msplat: failed to create a profiled Metal compute encoder";
                return;
            }
            encode_proj_sh_bwd_adam(enc);
            [enc endEncoding];

            // Stage 7: grad_stats. Keep an empty encoder after the cutoff so
            // the fixed counter-sample indices remain stable.
            enc = make_profiled_encoder(6);
            if (!enc) {
                encodingFailure = "msplat: failed to create a profiled Metal compute encoder";
                return;
            }
            encode_grad_stats(enc);
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
                NSData *data = [csb resolveCounterRange:NSMakeRange(0, (N_TRAIN_STAGES - 1) * 2)];
                if (!data) return;
                const MTLCounterResultTimestamp *samples =
                    (const MTLCounterResultTimestamp *)[data bytes];

                std::lock_guard<std::mutex> lock(g_stage_timing_mutex);
                for (int i = 0; i < N_TRAIN_STAGES - 1; i++) {
                    uint64_t start = samples[i * 2].timestamp;
                    uint64_t end = samples[i * 2 + 1].timestamp;
                    if (start == MTLCounterErrorValue || end == MTLCounterErrorValue) continue;
                    // stage_idx 0-7 maps to g_train_stage_names[1-8] (skip blit_zero)
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
                                g_train_stage_names[i], med, sum / sorted.size());
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
            if (!do_blit_zero(command_buffer)) {
                encodingFailure = "msplat: failed to create a Metal blit encoder";
                return;
            }

            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            if (!enc) {
                encodingFailure = "msplat: failed to create a Metal compute encoder";
                return;
            }

            // --- Forward: exact sort/pack → raster → loss ---
            encode_sort_pack(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_rast_fwd(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // --- Fused loss forward + backward ---
            encode_loss_fwd_bwd(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_rast_bwd(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_proj_sh_bwd_adam(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Accumulate grad stats ---
            encode_grad_stats(enc);

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

    // Loss is copied into the step's unique readback and published only after
    // GPU completion; the synchronous return remains the densification radii.
    return radii_out;
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
    MTensor &random_samples
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

    requireElements(random_samples, 6LL * num_splits, "random_samples");
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
