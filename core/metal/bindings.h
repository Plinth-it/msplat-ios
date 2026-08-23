#ifndef MSPLAT_BINDINGS_H
#define MSPLAT_BINDINGS_H

#include <cstdint>
#include <memory>
#include <tuple>
#include "metal_tensor.hpp"

// Configure the precompiled shader library before the first Metal operation.
// Throws when the path cannot be copied or initialization has already started.
void msplat_set_metallib_path_checked(const char* path);

// Release all cached GPU tensors (call before exit to prevent GPU memory leak)
void cleanup_msplat_metal();

// Returns the Metal device used by the msplat context (void* in C++, id<MTLDevice> in ObjC++)
#ifdef __OBJC__
id<MTLDevice> msplat_device();
#else
void* msplat_device();
#endif

// GPU tensor allocation (callable from C++ — delegates to Metal device)
MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype);
MTensor gpu_empty(std::vector<int64_t> shape, DType dtype);

// Commit current command buffer (non-blocking)
void msplat_commit();

// Synchronize (commit + wait for completion)
void msplat_gpu_sync();

// Completion-only training telemetry. A trainer owns one shared state. Each
// submitted logical step owns a distinct readback buffer until every command
// buffer associated with that step has completed, so pipelined steps cannot
// overwrite each other's loss or overflow result.
enum MsplatTrainingOverflowReason : uint32_t {
    MSPLAT_TRAINING_OVERFLOW_NONE = 0,
    MSPLAT_TRAINING_OVERFLOW_TILE_CAP = 1u << 0,
    MSPLAT_TRAINING_OVERFLOW_PACKED_CAPACITY = 1u << 1,
};

enum MsplatTrainingTelemetryFlag : uint32_t {
    MSPLAT_TRAINING_TELEMETRY_HAS_SUBMITTED = 1u << 0,
    MSPLAT_TRAINING_TELEMETRY_HAS_COMPLETED = 1u << 1,
    MSPLAT_TRAINING_TELEMETRY_GPU_TIMING_VALID = 1u << 2,
    MSPLAT_TRAINING_TELEMETRY_LOSS_VALID = 1u << 3,
    MSPLAT_TRAINING_TELEMETRY_INTERSECTION_COUNT_VALID = 1u << 4,
    MSPLAT_TRAINING_TELEMETRY_HAS_FAILED = 1u << 5,
};

struct MsplatTrainingStepDescriptor {
    int64_t iteration = 0;
    int64_t splatCount = 0;
    int64_t modelCapacity = 0;
    int32_t effectiveWidth = 0;
    int32_t effectiveHeight = 0;
    int32_t activeShDegree = 0;
    int32_t reserved = 0;
};

struct MsplatCompletedTrainingStepMetrics {
    MsplatTrainingStepDescriptor step;
    double cpuSubmitMs = 0.0;
    double gpuExecutionMs = 0.0;
    double endToEndMs = 0.0;
    double loss = 0.0;
    uint64_t retainedPackedIntersections = 0;
    uint64_t packedIntersectionCapacity = 0;
    uint32_t overflowReasons = MSPLAT_TRAINING_OVERFLOW_NONE;
    uint32_t commandBufferCount = 0;
};

struct MsplatTrainingTelemetrySnapshot {
    uint32_t flags = 0;
    uint32_t reserved = 0;
    uint64_t generation = 0;
    MsplatTrainingStepDescriptor submittedStep;
    double submittedCpuSubmitMs = 0.0;
    MsplatCompletedTrainingStepMetrics completedStep;
    uint64_t overflowedStepCount = 0;
    uint64_t tileCapOverflowedStepCount = 0;
    uint64_t packedCapacityOverflowedStepCount = 0;
    int64_t lastOverflowIteration = 0;
    uint64_t failedStepCount = 0;
    int64_t lastFailedIteration = 0;
};

struct MsplatTrainingTelemetryState;
struct MsplatLogicalTrainingStep;
using MsplatTrainingTelemetryHandle =
    std::shared_ptr<MsplatTrainingTelemetryState>;
using MsplatLogicalTrainingStepHandle =
    std::shared_ptr<MsplatLogicalTrainingStep>;

MsplatTrainingTelemetryHandle msplat_training_telemetry_create();
void msplat_training_telemetry_reset(
    const MsplatTrainingTelemetryHandle& telemetry);
MsplatTrainingTelemetrySnapshot msplat_training_telemetry_snapshot(
    const MsplatTrainingTelemetryHandle& telemetry);
size_t msplat_training_telemetry_readback_bytes(
    const MsplatTrainingTelemetryHandle& telemetry);

// Begin at Trainer::step entry so endToEndMs includes image preparation. Mark
// CPU start immediately before native GPU encoding. Submit commits the final
// command buffer non-blockingly, seals the descriptor, and returns cpuSubmitMs.
// The caller must abort the token if any work between begin and submit throws.
MsplatLogicalTrainingStepHandle msplat_training_step_begin(
    const MsplatTrainingTelemetryHandle& telemetry, int64_t iteration);
void msplat_training_step_mark_cpu_start(
    const MsplatLogicalTrainingStepHandle& step);
double msplat_training_step_submit(
    const MsplatLogicalTrainingStepHandle& step,
    const MsplatTrainingStepDescriptor& descriptor);
void msplat_training_step_abort(
    const MsplatLogicalTrainingStepHandle& step) noexcept;

// Bytes held by the cached per-iteration intermediates (the "temp" line in
// MSPLAT_MEM_LOG_EVERY). These dwarf the model at full resolution.
size_t msplat_cached_tensor_bytes();
// Internal accounting split for native tests and diagnostics. The shared side
// is sufficient for a cold render; the training side is allocated lazily.
size_t msplat_shared_cached_tensor_bytes();
size_t msplat_training_cached_tensor_bytes();

// GPU timing — non-invasive, uses completion handlers on committed CBs
void msplat_enable_gpu_timing(bool enable);
// Drains accumulated GPU times (ms per CB) into the provided vector. Thread-safe.
void msplat_drain_gpu_times(std::vector<double>& out);
// Drains per-stage GPU times. stage_times must be an array of N_STAGES vectors.
void msplat_drain_stage_times(std::vector<double> stage_times[], int max_stages, int& n_stages,
                              const char** stage_names);

// Render-only forward pass (no loss computation)
// Returns: out_img (H, W, 3) as MTensor
MTensor msplat_render(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background
);

struct MsplatPhotometricRefinementStep {
    bool enabled = false;
    uint32_t cameraIndex = 0;
    MTensor* logRgbGains = nullptr;
    MTensor* expAvg = nullptr;
    MTensor* expAvgSq = nullptr;
    float adamStepSize = 0.0f;
    float adamBiasCorrection2Sqrt = 1.0f;
    float regularization = 0.0f;
    float maxAbsLogGain = 0.0f;
};

// Fused forward + backward + Adam, with optional densification grad stats.
// Normalized data loss is available from the completed logical-step telemetry
// snapshot. Optimizer-only regularization terms are intentionally excluded.
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
    uint64_t loss_coverage_units, float ssim_weight,
    float loss_inv_n,
    int num_adam_groups,
    MTensor adam_params[], MTensor adam_exp_avg[], MTensor adam_exp_avg_sq[],
    float adam_step_sizes[], float adam_bc2_sqrts[],
    float adam_beta1, float adam_beta2, float adam_eps,
    const MsplatPhotometricRefinementStep& photometric,
    bool collect_densification_stats,
    MTensor &vis_counts, MTensor &xys_grad_norm, MTensor &max_2d_size,
    float inv_max_dim
);

// Classification and its prefix sums, split out so the caller learns how many
// gaussians will actually be written before it allocates room for them.
// A positive max_population keeps the highest-gradient candidates whose
// temporary append population fits; non-positive preserves unlimited behavior.
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
);

// Runs the prepared grow -> cull -> compact pipeline. `population` is
// N + 2*splits + dups as reported by msplat_prepare_densify.
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
);

#endif
