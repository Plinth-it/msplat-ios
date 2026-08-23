// C API for Swift interop. Thin wrapper around msplat C++ types.
// Opaque handles + free functions — works with any Swift version.

#ifndef MSPLAT_C_API_H
#define MSPLAT_C_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ABI v2 added checked, error-returning entry points. ABI v3 adds an optional
// hard training-limit contract without changing MsplatConfig's v2 layout.
// ABI v4 adds query-only completed-step and live-memory telemetry.
// ABI v5 adds a checked, caller-owned canonical dataset descriptor boundary.
// ABI v6 adds optional per-frame training masks without changing the v5 frame
// or descriptor layouts.
// All earlier symbols remain available for existing clients.
#define MSPLAT_ABI_VERSION 6u
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

// ── Stats ───────────────────────────────────────────────────────────────────

typedef struct {
    int iteration;
    int splatCount;
    float msPerStep; // CPU encode + command submission time; not completed GPU time.
} MsplatStats;

// Flags for MsplatTrainingMetrics.flags.
#define MSPLAT_TRAINING_METRICS_HAS_SUBMITTED_STEP (1u << 0)
#define MSPLAT_TRAINING_METRICS_HAS_COMPLETED_STEP (1u << 1)
#define MSPLAT_TRAINING_METRICS_GPU_TIME_VALID     (1u << 2)
#define MSPLAT_TRAINING_METRICS_LOSS_VALID         (1u << 3)
#define MSPLAT_TRAINING_METRICS_INTERSECTIONS_VALID (1u << 4)
#define MSPLAT_TRAINING_METRICS_HAS_FAILED_STEP    (1u << 5)

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

// ── Dataset ─────────────────────────────────────────────────────────────────

typedef void* MsplatDataset;

/// A length-delimited UTF-8 string. `data` must be NULL exactly when `length`
/// is zero. Embedded NUL bytes are not accepted.
typedef struct {
    const char* data;
    size_t length;
} MsplatStringViewV5;

#define MSPLAT_RASTER_ORIENTATION_ENCODED_PIXELS 0u
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

// Checked dataset API (ABI v2).
MsplatStatus msplat_dataset_create_v2(const char* path, float downscaleFactor,
                                      bool evalMode, int testEvery,
                                      MsplatDataset* outDataset,
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
// ABI v4 query APIs. outputSize must exactly match the corresponding struct.
// A size mismatch leaves the output untouched. With a correct size, failures
// zero the output before returning an error status.
MsplatStatus msplat_trainer_metrics_v4(
    MsplatTrainer t, MsplatTrainingMetrics* outMetrics, size_t outputSize,
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
