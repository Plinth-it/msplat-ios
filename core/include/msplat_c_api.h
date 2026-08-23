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

// ABI v2 adds checked, error-returning entry points while retaining the v1
// symbols below for existing clients. New integrations should use the v2 API.
#define MSPLAT_ABI_VERSION 2u
#define MSPLAT_ERROR_MESSAGE_CAPACITY 512u

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

// ── Stats ───────────────────────────────────────────────────────────────────

typedef struct {
    int iteration;
    int splatCount;
    float msPerStep; // CPU encode + command submission time; not completed GPU time.
} MsplatStats;

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
