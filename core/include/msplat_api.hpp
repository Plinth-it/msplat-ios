#pragma once

// Swift-compatible C++ API for msplat.
// Designed for Swift 5.9+ C++ interop — no std::tuple,
// no std::unordered_map, no templates in the public interface.
// Internal types hidden via PIMPL.

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>

struct DatasetDescriptor;

namespace msplat {

class PreviewFrame;

// ── Config ──────────────────────────────────────────────────────────────────

enum class TrainingMaskMode : uint32_t {
    Coverage = 0,
    Transparent = 1,
};

struct Config {
    int iterations = 30000;
    int shDegree = 3;
    int shDegreeInterval = 1000;
    float ssimWeight = 0.2f;
    int numDownscales = 2;
    int resolutionSchedule = 3000;
    int refineEvery = 100;
    int warmupLength = 500;
    int resetAlphaEvery = 30;
    float densifyGradThresh = 0.0002f;
    float densifySizeThresh = 0.01f;
    int stopScreenSizeAt = 4000;
    // Step after which the topology stops growing. -1 means half the budget.
    int stopDensifyAt = -1;
    // Hard population and backing-buffer ceiling. -1 means unlimited.
    int maxGaussians = -1;
    // Learn bounded per-camera log-RGB gains during training. Their mean is
    // exposure-like and their zero-mean residual is channel balance; source
    // pixels remain sRGB encoded, so this is not a physical exposure model.
    bool refinePhotometricGains = false;
    // Learn small regularized camera-space SE(3) corrections after warm-up.
    // Imported poses and canonical render/evaluation/export remain unchanged.
    bool refineCameraPoses = false;
    // Transparent mode applies only to frames that actually carry a mask.
    TrainingMaskMode trainingMaskMode = TrainingMaskMode::Coverage;
    float transparentAlphaLossWeight = 0.1f;
    float splitScreenSize = 0.05f;
    bool keepCrs = false;
    float downscaleFactor = 1.0f; // Legacy field retained for ABI compatibility; unused.
    float bgColor[3] = {0.6130f, 0.0101f, 0.3984f};  // magenta — high contrast for debugging
};

// ── Stats ───────────────────────────────────────────────────────────────────

struct Stats {
    int iteration = 0;
    int splatCount = 0;
    float msPerStep = 0.0f; // CPU encode + submission; not completed GPU time.
};

struct SubmittedTrainingStep {
    int iteration = 0;
    int splatCount = 0;
    int modelCapacity = 0;
    int effectiveWidth = 0;
    int effectiveHeight = 0;
    int activeSHDegree = 0;
    float cpuSubmitMs = 0.0f;
};

struct CompletedTrainingStep : SubmittedTrainingStep {
    float gpuExecutionMs = 0.0f;
    float endToEndMs = 0.0f;
    float loss = 0.0f;
    uint32_t overflowKinds = 0;
    uint64_t retainedPackedIntersectionCount = 0;
    uint64_t packedIntersectionCapacity = 0;
    float imagePrepareMs = 0.0f;
    float countGpuMs = 0.0f;
    float countWaitWallMs = 0.0f;
    float queueIdleMs = 0.0f;
    float postCountEncodeMs = 0.0f;
    float intersectionArenaGrowMs = 0.0f;
    uint32_t maximumTileCount = 0;
    uint32_t activeTileCount = 0;
    uint32_t trivialTileCount = 0;
    uint32_t smallTileCount = 0;
    uint32_t mediumTileCount = 0;
    uint32_t largeTileCount = 0;
};

struct TrainingMetrics {
    bool hasSubmittedStep = false;
    bool hasCompletedStep = false;
    bool gpuTimeValid = false;
    bool lossValid = false;
    bool intersectionsValid = false;
    bool hasFailedStep = false;
    bool countGpuTimeValid = false;
    bool queueIdleTimeValid = false;
    SubmittedTrainingStep submitted;
    CompletedTrainingStep completed;
    uint64_t overflowedCompletedSteps = 0;
    uint64_t tileCapOverflowedSteps = 0;
    uint64_t packedCapacityOverflowedSteps = 0;
    int lastOverflowIteration = 0;
    int lastFailedIteration = 0;
};

struct TrainingMemoryMetrics {
    size_t trainerModelBufferBytes = 0;
    size_t engineSharedTransientBufferBytes = 0;
    size_t engineTrainingTransientBufferBytes = 0;
    size_t trainerTelemetryReadbackBytes = 0;
    size_t trainerImageCacheCpuBytes = 0;
    size_t trainerImageCacheGpuBytes = 0;
    size_t trainerImageCacheBudgetBytes = 0;
    size_t processPhysFootprintBytes = 0;
    size_t processAvailableBytes = 0;
    uint64_t trainingGpuImageCacheHits = 0;
    uint64_t trainingGpuImageCacheMisses = 0;
    bool hasProcessPhysFootprint = false;
    bool hasProcessAvailableBytes = false;
};

/// Read-only snapshot of one canonical camera's opt-in pose refinement.
/// Translation deltas and the corrected matrix translation use the dataset's
/// original pre-normalization length units; rotation deltas use radians.
/// `frameId` is borrowed and remains valid for the owning Trainer's lifetime.
struct PoseRefinementState {
    bool enabled = false;
    bool anchor = false;
    uint32_t canonicalCameraIndex = 0;
    uint32_t optimizerStepCount = 0;
    float poseDelta[6] = {};
    float translationNorm = 0.0f;
    float rotationNorm = 0.0f;
    float correctedCameraToWorld[16] = {};
    const char* frameId = nullptr;
    size_t frameIdLength = 0;
};

struct EvalMetrics {
    float psnr = 0.0f;
    float ssim = 0.0f;
    float l1 = 0.0f;
    int numTest = 0;
    int numGaussians = 0;
};

// ── PixelBuffer ─────────────────────────────────────────────────────────────

/// Rendered image data. Owns its pixel buffer.
struct PixelBuffer {
    float* data = nullptr;   // RGB float32, HWC layout
    int width = 0;
    int height = 0;

    PixelBuffer() = default;
    PixelBuffer(float* d, int w, int h) : data(d), width(w), height(h) {}
    PixelBuffer(const PixelBuffer&) = delete;
    PixelBuffer& operator=(const PixelBuffer&) = delete;
    PixelBuffer(PixelBuffer&& o) : data(o.data), width(o.width), height(o.height) {
        o.data = nullptr;
    }
    PixelBuffer& operator=(PixelBuffer&& o) {
        if (this != &o) {
            free(data);
            data = o.data; width = o.width; height = o.height;
            o.data = nullptr;
        }
        return *this;
    }
    ~PixelBuffer() { free(data); }
};

// ── Dataset ─────────────────────────────────────────────────────────────────

class Dataset {
public:
    Dataset(const std::string& path, float downscaleFactor,
            bool evalMode, int testEvery);
    Dataset(const std::string& path, float downscaleFactor,
            bool evalMode, int testEvery, bool discoverTrainingMasks);
    Dataset(::DatasetDescriptor descriptor, float downscaleFactor,
            bool evalMode, int testEvery);
    ~Dataset();

    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;
    Dataset(Dataset&&) noexcept;
    Dataset& operator=(Dataset&&) noexcept;

    int numTrain() const;
    int numTest() const;
    void cameraPose(int index, float camToWorld[16]) const;
    /// Enables depth-one target preparation for trainers subsequently created
    /// from this dataset. Call before constructing a Trainer.
    void enableTrainingTargetPrefetch() noexcept;

    // Opaque handle for Trainer
    void* _handle() const;

    struct Impl;
private:
    std::unique_ptr<Impl> impl;
};

// ── Trainer ─────────────────────────────────────────────────────────────────

class Trainer {
public:
    // The current Metal engine is process-global. Trainer operations are
    // serialized across instances, and the referenced Dataset must outlive
    // this object. The C ABI owns that relationship automatically.
    Trainer(Dataset& dataset, const Config& config);
    ~Trainer();

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    /// Run one training step. Returns stats.
    Stats step();

    /// Run N steps (from current iteration to config.iterations).
    /// Calls callback every callbackEvery steps with current Stats.
    /// Use callbackEvery=0 to disable callbacks.
    void train(int callbackEvery);

    /// Evaluate on held-out test cameras.
    EvalMetrics evaluate();

    /// Render a camera view. Caller owns the returned PixelBuffer.
    PixelBuffer render(int cameraIndex, bool useTest);

    /// Render from an arbitrary camera-to-world pose (4x4 row-major, OpenGL convention).
    /// Uses intrinsics from the given reference camera.
    PixelBuffer renderFromPose(const float camToWorld[16], int refCameraIndex);

    /// Render from pose directly into a caller-provided RGBA uint8 buffer.
    /// outRGBA must hold width*height*4 bytes. Avoids intermediate float allocation.
    /// Call with outRGBA=nullptr to query dimensions only.
    void renderFromPoseToBuffer(const float camToWorld[16], int refCameraIndex,
                            uint8_t* outRGBA, int* outWidth, int* outHeight);

    /// Capacity-checked overload used by ABI v2.
    void renderFromPoseToBuffer(const float camToWorld[16], int refCameraIndex,
                            uint8_t* outRGBA, size_t outCapacity,
                            int* outWidth, int* outHeight);

    /// Submit a separately owned BGRA8 Metal preview texture. The returned
    /// frame may outlive this trainer; poll it before sampling the texture.
    std::unique_ptr<PreviewFrame> renderFromPosePreview(
        const float camToWorld[16], int refCameraIndex);

    /// Export scene to PLY format.
    void exportPly(const std::string& path);

    /// Export scene to .splat format.
    void exportSplat(const std::string& path);

    /// Export scene to .spz format.
    void exportSpz(const std::string& path);

    /// Save full training state (params + optimizer) for resume.
    void saveCheckpoint(const std::string& path);

    /// Load checkpoint and resume training. Returns the saved iteration.
    int loadCheckpoint(const std::string& path);

    int splatCount() const;
    int iteration() const;
    /// Latest submitted and successfully completed logical-step snapshots.
    TrainingMetrics metrics() const;
    /// Live buffer ownership, cache, and process-memory measurements.
    TrainingMemoryMetrics memoryMetrics() const;
    /// Returns zero when camera-pose refinement was not enabled.
    uint32_t poseRefinementStateCount() const;
    /// Synchronizes pending Metal work before reading the selected pose row.
    PoseRefinementState poseRefinementState(
        uint32_t canonicalCameraIndex) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

/// Ownership-safe asynchronous preview result. The borrowed texture pointer is
/// an id<MTLTexture> in Objective-C++ and remains valid for this object's
/// lifetime. Pixel contents are immutable after poll() returns true.
class PreviewFrame {
public:
    ~PreviewFrame();

    PreviewFrame(const PreviewFrame&) = delete;
    PreviewFrame& operator=(const PreviewFrame&) = delete;
    PreviewFrame(PreviewFrame&&) noexcept = default;
    PreviewFrame& operator=(PreviewFrame&&) noexcept = default;

    /// False while pending, true when complete, and throws on GPU failure.
    bool poll() const;
    void* texture() const;
    int width() const;
    int height() const;

    struct Impl;
private:
    explicit PreviewFrame(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl;
    friend class Trainer;
};

// ── Lifecycle ───────────────────────────────────────────────────────────────

void sync();
void cleanup();

} // namespace msplat
