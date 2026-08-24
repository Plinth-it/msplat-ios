#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <cstddef>
#include <cstdint>
#include <memory>
#include "dataset_descriptor.hpp"
#include "metal_tensor.hpp"

// Simple float32 RGB image — replaces cv::Mat
struct Image {
    std::vector<float> data;  // width * height * 3 floats, RGB, [0,1]
    int width = 0, height = 0;

    bool empty() const { return data.empty(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

/// Compact training raster. Bytes are tightly packed RGBA in the same
/// sRGB-encoded numerical space used by the loss. Loader-produced alpha is 255;
/// the loss ignores it because training transparency uses a separate mask.
struct RGBA8Image {
    std::vector<uint8_t> data;  // width * height * 4 bytes, RGBA
    int width = 0, height = 0;

    bool empty() const { return data.empty(); }
    uint8_t* ptr() { return data.data(); }
    const uint8_t* ptr() const { return data.data(); }
};

struct CoverageMask {
    std::vector<uint8_t> data;  // width * height bytes, soft coverage [0,255]
    int width = 0, height = 0;

    bool empty() const { return data.empty(); }
    uint8_t* ptr() { return data.data(); }
    const uint8_t* ptr() const { return data.data(); }
};

struct CameraTrainingTarget {
    MTensor *image = nullptr;
    MTensor *coverageMask = nullptr;  // null means uniform full coverage
    uint64_t coverageUnits = 0;       // sum(mask), or width * height * 255
};

struct Camera {
    // Effective training-raster calibration. Image loading may scale,
    // rectify, and crop these values; `declared` remains the source geometry.
    int width = 0, height = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
    float k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
    float camToWorld[16] = {};  // 4x4 row-major, camera-to-world (OpenGL: Y-up, Z-back)
    std::string filePath;
    RasterOrientation rasterOrientation = RasterOrientation::EncodedPixels;
    std::optional<TrainingMaskDescriptor> trainingMask;

    RGBA8Image image;
    CoverageMask coverageMask;
    std::unordered_map<int, RGBA8Image> imagePyramids;
    std::unordered_map<int, CoverageMask> coverageMaskPyramids;
    std::unordered_map<int, MTensor> mtensorImageCache;
    std::unordered_map<int, MTensor> mtensorCoverageMaskCache;
    std::unordered_map<int, uint64_t> coverageUnitsByDownscale;
    float loadedImageDownscaleFactor = 0.0f;
    MTensor cachedViewMat, cachedProjViewMat;
    float cachedCamPos[3] = {};
    float cachedFovX = 0, cachedFovY = 0;
    std::array<float, 16> cachedCameraToWorld = {};
    bool cachedPoseValid = false;

    /// The immutable intrinsics and distortion as the dataset declared them.
    /// loadImage re-derives the effective
    /// values from these on every call, so a camera whose image was evicted and
    /// reloaded lands on the same numbers rather than compounding the
    /// correction — and keeps its distortion coefficients, which the undistort
    /// step would otherwise zero out permanently, leaving a reloaded image
    /// distorted while its intrinsics claimed it was not.
    struct DeclaredIntrinsics {
        bool captured = false;
        float fx = 0, fy = 0, cx = 0, cy = 0;
        float k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
        int width = 0, height = 0;
    };
    DeclaredIntrinsics declared;

    void loadImage(float downscaleFactor);
    const RGBA8Image& getImage(int downscaleFactor);
    const CoverageMask& getCoverageMask(int downscaleFactor);
    uint64_t getCoverageUnits(int downscaleFactor);
    MTensor& getGPUImage(int downscaleFactor);
    CameraTrainingTarget getGPUTrainingTarget(int downscaleFactor);
    /// Changes the pose and invalidates every derived render matrix.
    void setCameraToWorld(const float pose[16]);
    /// Direct pose writes remain detectable so legacy/internal callers cannot
    /// accidentally reuse a view matrix derived from an earlier pose.
    bool projectionCacheMatchesPose() const noexcept;
    void recordProjectionCachePose() noexcept;
    void invalidateProjectionCache();
    /// Releases decoded/pyramid pixels after their compact GPU target has
    /// been published, while retaining corrected geometry and GPU caches.
    void releaseCpuImageMemory();
    void releaseImageMemory();
    size_t cachedCpuImageBytes() const;
    size_t cachedGpuImageBytes() const;
    size_t cachedImageBytes() const;
    bool hasDistortion() const { return k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0; }
};

struct Points {
    std::vector<float> xyz;     // N*3 flattened
    std::vector<uint8_t> rgb;   // N*3 flattened
    int64_t count = 0;
};

/// Source identity and reconstruction evidence retained alongside the mutable
/// camera/image runtime without duplicating its large RGB and XYZ arrays.
struct DatasetMetadata {
    std::vector<std::string> frameIds;
    std::vector<std::string> calibrationIds;
    std::vector<uint64_t> pointSourceIds;
    std::vector<float> pointReprojectionErrors;
    // Immutable descriptor/source-raster coordinates. Camera::loadImage may
    // later scale, rectify, and crop its mutable training raster, so these
    // observations must not be combined with effective Camera intrinsics
    // without applying the same source-to-training transform.
    std::vector<SparseObservation> observations;
    DatasetProvenance provenance;
};

/// Holds compact GPU training targets under a byte budget, evicting the
/// least-recently-used camera when the budget is exceeded. Decode and pyramid
/// pixels are released after a target is uploaded.
///
/// Every entry point used to decode all images up front and then copy the
/// camera array, so a dataset occupied twice its decoded size before the first
/// step ran. Three entry points each carried their own copy of that policy —
/// the C/Swift API, the CLI, and the Python bindings — so this lives here and
/// they all share it.
class CameraImageCache {
public:
    /// MSPLAT_IMAGE_CACHE_MB overrides it; otherwise 512MB on iOS, 2GB elsewhere.
    static size_t defaultBudgetBytes();
    /// Experimental depth-one CPU prefetch is opt-in with the exact value
    /// MSPLAT_CAMERA_PREFETCH=1.
    static bool defaultPrefetchEnabled() noexcept;

    CameraImageCache();
    CameraImageCache(float downscaleFactor, size_t budgetBytes,
                     bool prefetchEnabled = defaultPrefetchEnabled());
    ~CameraImageCache();
    CameraImageCache(CameraImageCache&&) noexcept;
    CameraImageCache& operator=(CameraImageCache&&) noexcept;
    CameraImageCache(const CameraImageCache&) = delete;
    CameraImageCache& operator=(const CameraImageCache&) = delete;

    /// Decodes cameras[index] if it is not resident and returns its GPU image at
    /// `downscaleFactor`, evicting other cameras to stay under budget. The
    /// camera being asked for is never the eviction victim.
    MTensor& gpuImage(std::vector<Camera> &cameras, size_t index, int downscaleFactor);

    /// Returns the RGB tensor together with optional soft coverage. A cache hit
    /// requires every tensor needed by the target to already be resident.
    CameraTrainingTarget gpuTrainingTarget(
        std::vector<Camera> &cameras, size_t index, int downscaleFactor);

    /// Best-effort CPU preparation for one exact future target. The worker
    /// never references or mutates `cameras`; a matching foreground target
    /// request performs the Metal upload and cache publication.
    void prefetchTrainingTarget(
        const std::vector<Camera> &cameras, size_t index,
        int downscaleFactor) noexcept;

    /// Waits for and discards any staged target. Decode failures are suppressed
    /// because they only become user-visible when the matching target is used.
    void discardPrefetch() noexcept;

    /// Establishes and accounts for corrected camera geometry without an
    /// upload. It decodes only when that geometry has not yet been established
    /// at the cache's input scale; a prior compact-target upload can satisfy
    /// render paths without recreating released CPU pixels.
    Camera& ensureLoaded(std::vector<Camera> &cameras, size_t index);

    size_t cachedBytes() const;
    size_t cachedCpuBytes() const;
    size_t cachedGpuBytes() const;
    size_t budgetBytes() const { return _budgetBytes; }
    uint64_t hitCount() const { return _hitCount; }
    uint64_t missCount() const { return _missCount; }
    bool prefetchEnabled() const { return _prefetchEnabled; }
    uint64_t prefetchScheduledCount() const { return _prefetchScheduledCount; }
    uint64_t prefetchUsedCount() const { return _prefetchUsedCount; }
    uint64_t prefetchWaitCount() const { return _prefetchWaitCount; }
    uint64_t prefetchDiscardedCount() const { return _prefetchDiscardedCount; }

private:
    void evict(std::vector<Camera> &cameras, size_t protectedIndex);
    struct PrefetchTask;

    struct Entry {
        uint64_t lastUse = 0;
        size_t cpuBytes = 0;
        size_t gpuBytes = 0;

        size_t bytes() const { return cpuBytes + gpuBytes; }
    };
    std::unordered_map<size_t, Entry> _entries;
    float _downscaleFactor = 1.0f;
    size_t _budgetBytes = 0;
    uint64_t _clock = 0;
    uint64_t _hitCount = 0;
    uint64_t _missCount = 0;
    bool _prefetchEnabled = false;
    std::unique_ptr<PrefetchTask> _prefetch;
    uint64_t _prefetchScheduledCount = 0;
    uint64_t _prefetchUsedCount = 0;
    uint64_t _prefetchWaitCount = 0;
    uint64_t _prefetchDiscardedCount = 0;
};

struct InputData {
    std::vector<Camera> cameras;
    float scale = 1.0f;
    float translation[3] = {};
    Points points;
    DatasetMetadata metadata;

    /// Indices into `cameras`. These replaced helpers that returned camera
    /// copies; a Camera owns its decoded Image, so those copies doubled the
    /// resident image memory for as long as training ran.
    std::tuple<std::vector<size_t>, int> trainIndices(bool validate, const std::string &valImage = "random") const;
    std::tuple<std::vector<size_t>, std::vector<size_t>> splitTrainTestIndices(int testEvery) const;
    void saveCameras(const std::string &filename, bool keepCrs) const;
};

// Auto-detect a source adapter, validate its canonical descriptor, and
// materialize the mutable camera/image runtime used by the current trainer.
DatasetDescriptor datasetDescriptorFromX(
    const std::string &path, const std::string &colmapImagePath = "");
DatasetDescriptor datasetDescriptorFromX(
    const std::string &path, const std::string &colmapImagePath,
    bool discoverTrainingMasks);
InputData inputDataFromDescriptor(DatasetDescriptor descriptor);
InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath = "");
InputData inputDataFromX(const std::string &path,
                         const std::string &colmapImagePath,
                         bool discoverTrainingMasks);

#endif
