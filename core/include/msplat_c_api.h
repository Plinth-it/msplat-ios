// C API for Swift interop. Thin wrapper around msplat C++ types.
// Opaque handles + free functions — works with any Swift version.

#ifndef MSPLAT_C_API_H
#define MSPLAT_C_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// The exported header remains valid C/C++. Objective-C and Swift clients get
// the typed Metal protocol object used by ABI v13 preview frames.
#ifdef __OBJC__
#import <Metal/Metal.h>
typedef id<MTLTexture> MsplatMTLTextureRef;
#else
typedef void* MsplatMTLTextureRef;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ABI v2 added checked, error-returning entry points. ABI v3 adds an optional
// hard training-limit contract without changing MsplatConfig's v2 layout.
// ABI v4 adds query-only completed-step and live-memory telemetry.
// ABI v5 adds a checked, caller-owned canonical dataset descriptor boundary.
// ABI v6 adds optional per-frame training masks without changing the v5 frame
// or descriptor layouts.
// ABI v7 adds CPU-only capture diagnostics over canonical observations.
// ABI v8 adds versioned refinement options without changing MsplatConfig's
// locked layout.
// ABI v9 adds the camera-pose refinement capability bit to those unchanged
// v8 options. The global bump lets new clients reject older binaries that
// would interpret that bit as unknown.
// ABI v10 adds opt-in Brush-compatible training-mask discovery for path-based
// COLMAP loading without changing any existing structure or symbol.
// ABI v11 adds versioned training-mask treatment options without changing
// MsplatConfig or the existing coverage-mask behavior.
// ABI v12 adds detailed count-barrier and tile-distribution telemetry through
// a new query structure without changing the ABI v4 telemetry layout.
// ABI v13 adds separately owned, asynchronously completed Metal preview
// frames while retaining every CPU render entry point.
// All earlier symbols remain available for existing clients.
#define MSPLAT_ABI_VERSION 13u
#define MSPLAT_ERROR_MESSAGE_CAPACITY 512u

// Checked descriptor input limits. Wrappers should reject larger values before
// constructing pointer/count views so they match the native allocation guard.
#define MSPLAT_DATASET_V5_MAX_STRING_BYTES 1048576u
#define MSPLAT_DATASET_V5_MAX_FRAMES 1000000u
#define MSPLAT_DATASET_V5_MAX_POINTS 100000000u
#define MSPLAT_DATASET_V5_MAX_OBSERVATIONS 100000000u

typedef enum {
    MSPLAT_STATUS_OK = 0,
    MSPLAT_STATUS_INVALID_ARGUMENT = 1,
    MSPLAT_STATUS_INVALID_DATASET = 2,
    MSPLAT_STATUS_OUT_OF_MEMORY = 3,
    MSPLAT_STATUS_GPU_ERROR = 4,
    MSPLAT_STATUS_IO_ERROR = 5,
    MSPLAT_STATUS_CANCELLED = 6,
    MSPLAT_STATUS_INTERNAL_ERROR = 7
} MsplatStatus;

/// Structured error populated by checked entry points. The message is always
/// NUL-terminated. Passing NULL for an error output is allowed.
typedef struct {
    MsplatStatus status;
    char message[MSPLAT_ERROR_MESSAGE_CAPACITY];
} MsplatErrorInfo;

uint32_t msplat_abi_version(void);
MsplatStatus msplat_last_status(void);
const char* msplat_last_error_message(void);

// ── Config ──────────────────────────────────────────────────────────────────

typedef struct {
    int iterations;
    int shDegree;
    int shDegreeInterval;
    float ssimWeight;
    int numDownscales;
    int resolutionSchedule;
    int refineEvery;
    int warmupLength;
    int resetAlphaEvery;
    float densifyGradThresh;
    float densifySizeThresh;
    int stopScreenSizeAt;
    int stopDensifyAt;
    float splitScreenSize;
    bool keepCrs;
    float downscaleFactor; // Legacy ABI field; training resolution does not use it.
    float bgColor[3];
} MsplatConfig;

static inline MsplatConfig msplat_default_config(void) {
    MsplatConfig c;
    c.iterations = 30000;
    c.shDegree = 3;
    c.shDegreeInterval = 1000;
    c.ssimWeight = 0.2f;
    c.numDownscales = 2;
    c.resolutionSchedule = 3000;
    c.refineEvery = 100;
    c.warmupLength = 500;
    c.resetAlphaEvery = 30;
    c.densifyGradThresh = 0.0002f;
    c.densifySizeThresh = 0.01f;
    c.stopScreenSizeAt = 4000;
    c.stopDensifyAt = -1;
    c.splitScreenSize = 0.05f;
    c.keepCrs = false;
    c.downscaleFactor = 1.0f;
    c.bgColor[0] = 0.6130f; c.bgColor[1] = 0.0101f; c.bgColor[2] = 0.3984f;
    return c;
}

/// Limits introduced in ABI v3. `maxGaussians` is a hard population and
/// backing-buffer ceiling. Use -1 for the legacy unlimited behavior.
typedef struct {
    int maxGaussians;
} MsplatTrainingLimits;

static inline MsplatTrainingLimits msplat_default_training_limits(void) {
    MsplatTrainingLimits limits;
    limits.maxGaussians = -1;
    return limits;
}

/// Optional training refinements introduced in ABI v8 and extended in ABI v9.
/// These flags are kept outside MsplatConfig so its established binary layout
/// remains unchanged.
/// Photometric RGB gains affect only the training objective; rendering and
/// evaluation continue to use the canonical model colors.
#define MSPLAT_REFINEMENT_PHOTOMETRIC_RGB_GAINS (1u << 0)
/// Learn bounded camera-space pose corrections only for training. Imported
/// camera geometry and canonical render/evaluation/export remain unchanged.
#define MSPLAT_REFINEMENT_CAMERA_POSE_DELTAS     (1u << 1)

typedef struct {
    uint32_t flags;
    uint32_t reserved[3];
} MsplatRefinementOptionsV8;

static inline MsplatRefinementOptionsV8
msplat_default_refinement_options_v8(void) {
    MsplatRefinementOptionsV8 options;
    options.flags = 0u;
    options.reserved[0] = 0u;
    options.reserved[1] = 0u;
    options.reserved[2] = 0u;
    return options;
}

/// How a decoded per-frame training mask participates in the objective.
/// Coverage mode preserves the ABI v6 behavior: mask values weight RGB loss.
/// Transparent mode treats the mask as target alpha, composites source RGB
/// over the configured background, and adds a full-frame alpha loss.
typedef enum {
    MSPLAT_TRAINING_MASK_MODE_COVERAGE = 0,
    MSPLAT_TRAINING_MASK_MODE_TRANSPARENT = 1
} MsplatTrainingMaskModeV11;

typedef struct {
    uint32_t mode;
    float alphaLossWeight;
    uint32_t reserved[2];
} MsplatTrainingMaskOptionsV11;

static inline MsplatTrainingMaskOptionsV11
msplat_default_training_mask_options_v11(void) {
    MsplatTrainingMaskOptionsV11 options;
    options.mode = MSPLAT_TRAINING_MASK_MODE_COVERAGE;
    options.alphaLossWeight = 0.1f;
    options.reserved[0] = 0u;
    options.reserved[1] = 0u;
    return options;
}

// ── Stats ───────────────────────────────────────────────────────────────────

typedef struct {
    int iteration;
    int splatCount;
    float msPerStep; // CPU encode + command submission time; not completed GPU time.
} MsplatStats;

// Flags shared by the v4 and v12 training-metrics snapshots. The count-GPU
// timing flag is emitted only by the v12 query.
#define MSPLAT_TRAINING_METRICS_HAS_SUBMITTED_STEP (1u << 0)
#define MSPLAT_TRAINING_METRICS_HAS_COMPLETED_STEP (1u << 1)
#define MSPLAT_TRAINING_METRICS_GPU_TIME_VALID     (1u << 2)
#define MSPLAT_TRAINING_METRICS_LOSS_VALID         (1u << 3)
#define MSPLAT_TRAINING_METRICS_INTERSECTIONS_VALID (1u << 4)
#define MSPLAT_TRAINING_METRICS_HAS_FAILED_STEP    (1u << 5)
#define MSPLAT_TRAINING_METRICS_COUNT_GPU_TIME_VALID (1u << 6)

// Bit values for MsplatCompletedTrainingStep.overflowKinds.
#define MSPLAT_RASTER_OVERFLOW_TILE_CAP        (1u << 0)
#define MSPLAT_RASTER_OVERFLOW_PACKED_CAPACITY (1u << 1)

/// Identity and CPU submission cost of one submitted training step.
typedef struct {
    int32_t iteration;
    int32_t splatCount;
    int32_t modelCapacity;
    int32_t effectiveWidth;
    int32_t effectiveHeight;
    int32_t activeSHDegree;
    float cpuSubmitMs;
    uint32_t reserved;
} MsplatSubmittedTrainingStep;

/// Measurements published only after every command buffer belonging to this
/// logical training step has completed successfully.
typedef struct {
    int32_t iteration;
    int32_t splatCount;
    int32_t modelCapacity;
    int32_t effectiveWidth;
    int32_t effectiveHeight;
    int32_t activeSHDegree;
    float cpuSubmitMs;
    float gpuExecutionMs;
    float endToEndMs;
    float loss;
    uint32_t overflowKinds;
    uint32_t reserved;
    uint64_t retainedPackedIntersectionCount;
    uint64_t packedIntersectionCapacity;
} MsplatCompletedTrainingStep;

/// Query-only snapshot. Submitted and completed descriptors are intentionally
/// separate because GPU completion may lag CPU submission by several steps.
typedef struct {
    uint32_t flags;
    uint32_t reserved;
    MsplatSubmittedTrainingStep submitted;
    MsplatCompletedTrainingStep completed;
    uint64_t overflowedCompletedSteps;
    uint64_t tileCapOverflowedSteps;
    uint64_t packedCapacityOverflowedSteps;
    int32_t lastOverflowIteration;
    int32_t lastFailedIteration;
} MsplatTrainingMetrics;

/// ABI v12 completed-step telemetry. The first 64 bytes intentionally match
/// MsplatCompletedTrainingStep exactly; new fields are appended only here.
typedef struct {
    int32_t iteration;
    int32_t splatCount;
    int32_t modelCapacity;
    int32_t effectiveWidth;
    int32_t effectiveHeight;
    int32_t activeSHDegree;
    float cpuSubmitMs;
    float gpuExecutionMs;
    float endToEndMs;
    float loss;
    uint32_t overflowKinds;
    uint32_t reserved;
    uint64_t retainedPackedIntersectionCount;
    uint64_t packedIntersectionCapacity;
    float imagePrepareMs;
    float countGpuMs;
    float countWaitWallMs;
    float postCountEncodeMs;
    float intersectionArenaGrowMs;
    uint32_t maximumTileCount;
    uint32_t activeTileCount;
    uint32_t trivialTileCount;
    uint32_t smallTileCount;
    uint32_t mediumTileCount;
    uint32_t largeTileCount;
    uint32_t reservedV12;
} MsplatCompletedTrainingStepV12;

/// ABI v12 query snapshot. The v4 query remains available for existing clients.
typedef struct {
    uint32_t flags;
    uint32_t reserved;
    MsplatSubmittedTrainingStep submitted;
    MsplatCompletedTrainingStepV12 completed;
    uint64_t overflowedCompletedSteps;
    uint64_t tileCapOverflowedSteps;
    uint64_t packedCapacityOverflowedSteps;
    int32_t lastOverflowIteration;
    int32_t lastFailedIteration;
} MsplatTrainingMetricsV12;

// Flags for MsplatTrainingMemoryMetrics.flags.
#define MSPLAT_MEMORY_METRICS_PHYS_FOOTPRINT_VALID (1u << 0)
#define MSPLAT_MEMORY_METRICS_AVAILABLE_VALID      (1u << 1)

/// Live memory/accounting snapshot. Buffer categories are logical owned bytes;
/// process footprint additionally includes driver, framework, and allocator
/// overhead. Available bytes is an iOS jetsam-headroom measurement.
typedef struct {
    uint32_t flags;
    uint32_t reserved;
    uint64_t trainerModelBufferBytes;
    uint64_t engineSharedTransientBufferBytes;
    uint64_t engineTrainingTransientBufferBytes;
    uint64_t trainerTelemetryReadbackBytes;
    uint64_t trainerImageCacheCpuBytes;
    uint64_t trainerImageCacheGpuBytes;
    uint64_t trainerImageCacheBudgetBytes;
    uint64_t processPhysFootprintBytes;
    uint64_t processAvailableBytes;
    uint64_t trainingGpuImageCacheHits;
    uint64_t trainingGpuImageCacheMisses;
} MsplatTrainingMemoryMetrics;

typedef struct {
    float psnr;
    float ssim;
    float l1;
    int numTest;
    int numGaussians;
} MsplatEvalMetrics;

// ── Pixel buffer ────────────────────────────────────────────────────────────

typedef struct {
    float* data;   // RGB float32, HWC layout. Caller must free() this.
    int width;
    int height;
} MsplatPixelBuffer;

/// Opaque ABI v13 preview-frame handle. The frame owns a private BGRA8Unorm
/// Metal texture until destroyed. Its texture is immutable once ready and may
/// outlive the trainer/session that submitted it.
typedef void* MsplatPreviewFrame;

// ── Dataset ─────────────────────────────────────────────────────────────────

typedef void* MsplatDataset;

/// A length-delimited UTF-8 string. `data` must be NULL exactly when `length`
/// is zero. Embedded NUL bytes are not accepted.
typedef struct {
    const char* data;
    size_t length;
} MsplatStringViewV5;

#define MSPLAT_RASTER_ORIENTATION_ENCODED_PIXELS 0u
/// Pixels are EXIF-normalized only after the caller has transformed
/// calibration, observations, and pose into that frame. Mirrored EXIF tags
/// are rejected because the camera model is right-handed.
#define MSPLAT_RASTER_ORIENTATION_EXIF_NORMALIZED 1u

#define MSPLAT_MASK_COVERAGE_LUMINANCE 0u
#define MSPLAT_MASK_COVERAGE_ALPHA 1u

typedef struct {
    int32_t width;
    int32_t height;
    float fx;
    float fy;
    float cx;
    float cy;
    float k1;
    float k2;
    float k3;
    float p1;
    float p2;
} MsplatCameraCalibrationV5;

typedef struct {
    MsplatStringViewV5 id;
    MsplatStringViewV5 calibrationId;
    MsplatStringViewV5 imagePath;
    uint32_t rasterOrientation;
    uint32_t reserved;
    MsplatCameraCalibrationV5 calibration;
    /// Rigid row-major OpenGL camera-to-world transform (Y-up, Z-back).
    float cameraToWorld[16];
} MsplatDatasetFrameV5;

typedef struct {
    uint32_t frameIndex;
    uint32_t frameObservationIndex;
    /// Index into the point arrays, or -1 for an untriangulated feature.
    int32_t pointIndex;
    uint32_t reserved;
    /// Image-edge coordinates in the owning frame's declared source raster;
    /// the upper-left pixel center is (0.5, 0.5). Native decode scaling,
    /// rectification, and cropping do not mutate these values.
    float x;
    float y;
} MsplatSparseObservationV5;

/// Canonical, caller-owned dataset input introduced in ABI v5. Array counts
/// are element counts: XYZ and RGB contain three elements per point, while
/// source IDs and reprojection errors contain one. Optional arrays use a NULL
/// pointer and zero count. All reserved fields must be zero.
typedef struct {
    const MsplatDatasetFrameV5* frames;
    size_t frameCount;

    const float* pointXYZ;
    size_t pointXYZCount;
    const uint8_t* pointRGB;
    size_t pointRGBCount;

    const uint64_t* pointSourceIds;
    size_t pointSourceIdCount;
    const float* pointReprojectionErrors;
    size_t pointReprojectionErrorCount;

    const MsplatSparseObservationV5* observations;
    size_t observationCount;

    MsplatStringViewV5 provenanceAdapter;
    MsplatStringViewV5 provenanceSource;
    uint64_t reserved[2];
} MsplatDatasetDescriptorV5;

/// Per-frame training-mask sidecar introduced in ABI v6. There must be one
/// element for every v5 frame. A NULL/zero-length path means that frame is
/// unmasked and requires every other field to be zero. Non-empty mask paths
/// are copied synchronously and use one of the MSPLAT_MASK_COVERAGE_* modes.
typedef struct {
    MsplatStringViewV5 maskPath;
    uint32_t coverageChannel;
    uint32_t reserved;
    uint64_t reserved2[2];
} MsplatFrameMaskV6;

/// One-pass residual statistics. A zero sample count represents unavailable
/// statistics; all numeric fields are then zero.
typedef struct {
    uint64_t sampleCount;
    double meanPixels;
    double rootMeanSquarePixels;
    double maximumPixels;
} MsplatReprojectionErrorStatisticsV7;

/// Capture diagnostics for one canonical frame. Observed coordinates and
/// predicted coordinates use the descriptor's source-pixel frame directly;
/// the upper-left pixel center is (0.5, 0.5). Outside-frame counts use the
/// half-open bounds [0, width) and [0, height).
typedef struct {
    uint32_t frameIndex;
    uint32_t reserved32;
    uint64_t observationCount;
    uint64_t linkedObservationCount;
    uint64_t observedOutsideFrameCount;
    uint64_t reprojectedObservationCount;
    uint64_t behindCameraObservationCount;
    uint64_t nonFiniteProjectionCount;
    uint64_t projectedOutsideFrameCount;
    MsplatReprojectionErrorStatisticsV7 reprojectionError;
    uint64_t reserved[2];
} MsplatFrameCaptureDiagnosticsV7;

/// Threshold-free capture diagnostics over a canonical descriptor. Source
/// point errors are summarized separately from residuals recomputed from the
/// declared camera geometry and linked observations. Track lengths count the
/// distinct observing frames for each point and exclude unobserved points from
/// their mean.
typedef struct {
    uint64_t frameCount;
    uint64_t pointCount;
    uint64_t observationCount;
    uint64_t linkedObservationCount;
    uint64_t observedOutsideFrameCount;
    uint64_t reprojectedObservationCount;
    uint64_t behindCameraObservationCount;
    uint64_t nonFiniteProjectionCount;
    uint64_t projectedOutsideFrameCount;
    uint64_t observedPointCount;
    uint64_t multiViewPointCount;
    uint32_t maximumTrackLength;
    uint32_t reserved32;
    double meanTrackLength;
    MsplatReprojectionErrorStatisticsV7 reprojectionError;
    MsplatReprojectionErrorStatisticsV7 sourcePointReprojectionError;
    uint64_t reserved[2];
} MsplatDatasetCaptureDiagnosticsV7;

// Checked dataset API (ABI v2).
MsplatStatus msplat_dataset_create_v2(const char* path, float downscaleFactor,
                                      bool evalMode, int testEvery,
                                      MsplatDataset* outDataset,
                                      MsplatErrorInfo* error);
/// ABI v10 path creation. When enabled for a COLMAP dataset, regular files
/// below a case-insensitive `masks` directory are matched to image names using
/// Brush-compatible stem and nested-directory rules. Missing matches leave
/// individual frames unmasked. ABI v2 retains its discovery-disabled behavior.
MsplatStatus msplat_dataset_create_v10(
    const char* path, float downscaleFactor, bool evalMode, int32_t testEvery,
    bool discoverTrainingMasks, MsplatDataset* outDataset,
    MsplatErrorInfo* error);
MsplatStatus msplat_dataset_destroy_v2(MsplatDataset ds, MsplatErrorInfo* error);
MsplatStatus msplat_dataset_num_train_v2(MsplatDataset ds, int* outCount,
                                         MsplatErrorInfo* error);
MsplatStatus msplat_dataset_num_test_v2(MsplatDataset ds, int* outCount,
                                        MsplatErrorInfo* error);
MsplatStatus msplat_dataset_camera_pose_v2(MsplatDataset ds, int cameraIndex,
                                           float camToWorld[16],
                                           MsplatErrorInfo* error);

/// Checked canonical-descriptor API (ABI v5). The complete descriptor,
/// including every string and array, is copied synchronously. No caller-owned
/// pointer is retained after this function returns. `descriptorSize` must be
/// exactly sizeof(MsplatDatasetDescriptorV5).
MsplatStatus msplat_dataset_create_from_descriptor_v5(
    const MsplatDatasetDescriptorV5* descriptor,
    size_t descriptorSize,
    float downscaleFactor,
    bool evalMode,
    int32_t testEvery,
    MsplatDataset* outDataset,
    MsplatErrorInfo* error);

/// Checked canonical-descriptor API with optional per-frame masks (ABI v6).
/// The v5 descriptor and complete mask sidecar are copied synchronously. No
/// caller-owned pointer is retained. `descriptorSize` and
/// `frameMaskElementSize` must exactly match their corresponding C structs;
/// `frameMaskCount` must equal descriptor->frameCount.
MsplatStatus msplat_dataset_create_from_descriptor_v6(
    const MsplatDatasetDescriptorV5* descriptor,
    size_t descriptorSize,
    const MsplatFrameMaskV6* frameMasks,
    size_t frameMaskCount,
    size_t frameMaskElementSize,
    float downscaleFactor,
    bool evalMode,
    int32_t testEvery,
    MsplatDataset* outDataset,
    MsplatErrorInfo* error);

/// CPU-only descriptor preflight introduced in ABI v7. The complete v5
/// descriptor is borrowed only for this synchronous call. No image or mask is
/// decoded and Metal is not initialized. Output structure sizes and the frame
/// count/element size must match this ABI exactly. Valid output storage is
/// cleared before descriptor copying so failures never expose partial results.
MsplatStatus msplat_dataset_capture_diagnostics_v7(
    const MsplatDatasetDescriptorV5* descriptor,
    size_t descriptorSize,
    MsplatDatasetCaptureDiagnosticsV7* outDiagnostics,
    size_t diagnosticsSize,
    MsplatFrameCaptureDiagnosticsV7* outFrames,
    size_t frameCount,
    size_t frameElementSize,
    MsplatErrorInfo* error);

MsplatDataset msplat_dataset_create(const char* path, float downscaleFactor,
                                     bool evalMode, int testEvery);
void msplat_dataset_destroy(MsplatDataset ds);
int msplat_dataset_num_train(MsplatDataset ds);
int msplat_dataset_num_test(MsplatDataset ds);

// ── Trainer ─────────────────────────────────────────────────────────────────

typedef void* MsplatTrainer;

// Checked trainer API (ABI v2). A trainer retains its dataset internally, so
// the public dataset handle may be destroyed after trainer creation.
// configSize must be sizeof(MsplatConfig), which
// prevents a wrapper and native binary built from different headers from
// silently interpreting different layouts.
MsplatStatus msplat_config_validate_v2(const MsplatConfig* config,
                                       size_t configSize,
                                       MsplatErrorInfo* error);
MsplatStatus msplat_trainer_create_v2(MsplatDataset ds,
                                      const MsplatConfig* config,
                                      size_t configSize,
                                      MsplatTrainer* outTrainer,
                                      MsplatErrorInfo* error);
// ABI v3 trainer creation. Both structure sizes must match the headers used by
// the caller. ABI v2 creation remains available and uses no Gaussian limit.
MsplatStatus msplat_training_limits_validate_v3(
    const MsplatTrainingLimits* limits, size_t limitsSize,
    MsplatErrorInfo* error);
MsplatStatus msplat_trainer_create_v3(MsplatDataset ds,
                                      const MsplatConfig* config,
                                      size_t configSize,
                                      const MsplatTrainingLimits* limits,
                                      size_t limitsSize,
                                      MsplatTrainer* outTrainer,
                                      MsplatErrorInfo* error);
// ABI v8 trainer creation, extended by ABI v9's camera-pose flag. Refinement
// options are versioned separately from MsplatConfig. Unknown flags and
// non-zero reserved fields are rejected.
MsplatStatus msplat_refinement_options_validate_v8(
    const MsplatRefinementOptionsV8* options, size_t optionsSize,
    MsplatErrorInfo* error);
MsplatStatus msplat_trainer_create_v8(
    MsplatDataset ds,
    const MsplatConfig* config, size_t configSize,
    const MsplatTrainingLimits* limits, size_t limitsSize,
    const MsplatRefinementOptionsV8* refinementOptions,
    size_t refinementOptionsSize,
    MsplatTrainer* outTrainer,
    MsplatErrorInfo* error);
// ABI v11 trainer creation. Mask treatment is versioned separately from the
// locked MsplatConfig layout. Unknown modes, non-finite/negative weights, and
// non-zero reserved fields are rejected. Frames without a mask retain opaque
// RGB training even when transparent mode is selected.
MsplatStatus msplat_training_mask_options_validate_v11(
    const MsplatTrainingMaskOptionsV11* options, size_t optionsSize,
    MsplatErrorInfo* error);
MsplatStatus msplat_trainer_create_v11(
    MsplatDataset ds,
    const MsplatConfig* config, size_t configSize,
    const MsplatTrainingLimits* limits, size_t limitsSize,
    const MsplatRefinementOptionsV8* refinementOptions,
    size_t refinementOptionsSize,
    const MsplatTrainingMaskOptionsV11* maskOptions,
    size_t maskOptionsSize,
    MsplatTrainer* outTrainer,
    MsplatErrorInfo* error);
MsplatStatus msplat_trainer_destroy_v2(MsplatTrainer t, MsplatErrorInfo* error);
MsplatStatus msplat_trainer_step_v2(MsplatTrainer t, MsplatStats* outStats,
                                    MsplatErrorInfo* error);
MsplatStatus msplat_trainer_train_v2(MsplatTrainer t, MsplatErrorInfo* error);
MsplatStatus msplat_trainer_evaluate_v2(MsplatTrainer t,
                                        MsplatEvalMetrics* outMetrics,
                                        MsplatErrorInfo* error);
MsplatStatus msplat_trainer_render_v2(MsplatTrainer t, int cameraIndex,
                                      bool useTest, MsplatPixelBuffer* outBuffer,
                                      MsplatErrorInfo* error);
MsplatStatus msplat_trainer_render_pose_v2(MsplatTrainer t,
                                           const float camToWorld[16],
                                           int refCameraIndex,
                                           MsplatPixelBuffer* outBuffer,
                                           MsplatErrorInfo* error);
/// Query dimensions with outRGBA=NULL and outCapacity=0. When outRGBA is not
/// NULL, outCapacity must be at least width*height*4 bytes.
MsplatStatus msplat_trainer_render_pose_to_buffer_v2(
    MsplatTrainer t, const float camToWorld[16], int refCameraIndex,
    uint8_t* outRGBA, size_t outCapacity, int* outWidth, int* outHeight,
    MsplatErrorInfo* error);
/// Submit a fixed-pose preview into a separately owned BGRA8Unorm texture.
/// The returned frame begins pending. Poll until ready before sampling its
/// texture from another Metal queue. Submission still inherits the renderer's
/// exact-count synchronization; completion of the final raster/conversion is
/// asynchronous. Destroying a pending frame is safe.
MsplatStatus msplat_trainer_render_pose_preview_v13(
    MsplatTrainer t, const float camToWorld[16], int refCameraIndex,
    MsplatPreviewFrame* outFrame, MsplatErrorInfo* error);
/// Returns OK with outReady=false while pending, OK/true when complete, or a
/// structured GPU error when the submitted command buffer failed.
MsplatStatus msplat_preview_frame_poll_v13(
    MsplatPreviewFrame frame, bool* outReady, MsplatErrorInfo* error);
/// Returns the borrowed texture and dimensions only after successful
/// completion. Swift/Objective-C clients retain the returned object normally.
MsplatStatus msplat_preview_frame_texture_v13(
    MsplatPreviewFrame frame, MsplatMTLTextureRef* outTexture,
    int* outWidth, int* outHeight, MsplatErrorInfo* error);
MsplatStatus msplat_preview_frame_destroy_v13(
    MsplatPreviewFrame frame, MsplatErrorInfo* error);
MsplatStatus msplat_trainer_export_ply_v2(MsplatTrainer t, const char* path,
                                          MsplatErrorInfo* error);
MsplatStatus msplat_trainer_export_splat_v2(MsplatTrainer t, const char* path,
                                            MsplatErrorInfo* error);
MsplatStatus msplat_trainer_export_spz_v2(MsplatTrainer t, const char* path,
                                          MsplatErrorInfo* error);
MsplatStatus msplat_trainer_save_checkpoint_v2(MsplatTrainer t,
                                               const char* path,
                                               MsplatErrorInfo* error);
MsplatStatus msplat_trainer_load_checkpoint_v2(MsplatTrainer t,
                                               const char* path,
                                               int* outIteration,
                                               MsplatErrorInfo* error);
MsplatStatus msplat_trainer_splat_count_v2(MsplatTrainer t, int* outCount,
                                           MsplatErrorInfo* error);
MsplatStatus msplat_trainer_iteration_v2(MsplatTrainer t, int* outIteration,
                                         MsplatErrorInfo* error);
// Versioned query APIs. outputSize must exactly match the corresponding struct.
// A size mismatch leaves the output untouched. With a correct size, failures
// zero the output before returning an error status.
MsplatStatus msplat_trainer_metrics_v4(
    MsplatTrainer t, MsplatTrainingMetrics* outMetrics, size_t outputSize,
    MsplatErrorInfo* error);
MsplatStatus msplat_trainer_metrics_v12(
    MsplatTrainer t, MsplatTrainingMetricsV12* outMetrics, size_t outputSize,
    MsplatErrorInfo* error);
MsplatStatus msplat_trainer_memory_metrics_v4(
    MsplatTrainer t, MsplatTrainingMemoryMetrics* outMetrics,
    size_t outputSize, MsplatErrorInfo* error);
void msplat_pixel_buffer_free(MsplatPixelBuffer* buffer);

MsplatTrainer msplat_trainer_create(MsplatDataset ds, MsplatConfig config);
void msplat_trainer_destroy(MsplatTrainer t);

MsplatStats msplat_trainer_step(MsplatTrainer t);
void msplat_trainer_train(MsplatTrainer t);
MsplatEvalMetrics msplat_trainer_evaluate(MsplatTrainer t);
MsplatPixelBuffer msplat_trainer_render(MsplatTrainer t, int cameraIndex, bool useTest);
MsplatPixelBuffer msplat_trainer_render_pose(MsplatTrainer t, const float camToWorld[16], int refCameraIndex);

/// Render into a caller-provided RGBA uint8 buffer (no allocation, no float copy).
/// outRGBA must be at least width*height*4 bytes. Returns dimensions via outWidth/outHeight.
/// Call once with outRGBA=NULL to get dimensions, then allocate and call again.
void msplat_trainer_render_pose_to_buffer(MsplatTrainer t, const float camToWorld[16],
                                      int refCameraIndex, uint8_t* outRGBA,
                                      int* outWidth, int* outHeight);
void msplat_trainer_export_ply(MsplatTrainer t, const char* path);
void msplat_trainer_export_splat(MsplatTrainer t, const char* path);
void msplat_trainer_export_spz(MsplatTrainer t, const char* path);
void msplat_trainer_save_checkpoint(MsplatTrainer t, const char* path);
int msplat_trainer_load_checkpoint(MsplatTrainer t, const char* path);
int msplat_trainer_splat_count(MsplatTrainer t);
int msplat_trainer_iteration(MsplatTrainer t);
void msplat_dataset_camera_pose(MsplatDataset ds, int cameraIndex, float camToWorld[16]);

// ── Lifecycle ───────────────────────────────────────────────────────────────

void msplat_set_metallib_path(const char* path);
void msplat_sync(void);
void msplat_cleanup(void);

MsplatStatus msplat_set_metallib_path_v2(const char* path,
                                         MsplatErrorInfo* error);
MsplatStatus msplat_sync_v2(MsplatErrorInfo* error);
MsplatStatus msplat_cleanup_v2(MsplatErrorInfo* error);

#ifdef __cplusplus
}
#endif

#endif // MSPLAT_C_API_H
