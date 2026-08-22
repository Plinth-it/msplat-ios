#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <cstddef>
#include <cstdint>
#include "metal_tensor.hpp"

// Simple float32 RGB image — replaces cv::Mat
struct Image {
    std::vector<float> data;  // width * height * 3 floats, RGB, [0,1]
    int width = 0, height = 0;

    bool empty() const { return data.empty(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

struct Camera {
    int width = 0, height = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
    float k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
    float camToWorld[16] = {};  // 4x4 row-major, camera-to-world (OpenGL: Y-up, Z-back)
    std::string filePath;

    Image image;
    std::unordered_map<int, Image> imagePyramids;
    std::unordered_map<int, MTensor> mtensorImageCache;
    float loadedImageDownscaleFactor = 0.0f;
    MTensor cachedViewMat, cachedProjViewMat;
    float cachedCamPos[3] = {};
    float cachedFovX = 0, cachedFovY = 0;

    void loadImage(float downscaleFactor);
    const Image& getImage(int downscaleFactor);
    MTensor& getGPUImage(int downscaleFactor);
    void releaseImageMemory();
    size_t cachedImageBytes() const;
    bool hasDistortion() const { return k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0; }
};

struct Points {
    std::vector<float> xyz;     // N*3 flattened
    std::vector<uint8_t> rgb;   // N*3 flattened
    int64_t count = 0;
};

/// Holds decoded training images under a byte budget, evicting the
/// least-recently-used camera when the budget is exceeded.
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

    CameraImageCache() = default;
    CameraImageCache(float downscaleFactor, size_t budgetBytes)
        : _downscaleFactor(downscaleFactor), _budgetBytes(budgetBytes) {}

    /// Decodes cameras[index] if it is not resident and returns its GPU image at
    /// `downscaleFactor`, evicting other cameras to stay under budget. The
    /// camera being asked for is never the eviction victim.
    MTensor& gpuImage(std::vector<Camera> &cameras, size_t index, int downscaleFactor);

    size_t cachedBytes() const;
    size_t budgetBytes() const { return _budgetBytes; }

private:
    void evict(std::vector<Camera> &cameras, size_t protectedIndex);

    struct Entry {
        uint64_t lastUse = 0;
        size_t bytes = 0;
    };
    std::unordered_map<size_t, Entry> _entries;
    float _downscaleFactor = 1.0f;
    size_t _budgetBytes = 0;
    uint64_t _clock = 0;
};

struct InputData {
    std::vector<Camera> cameras;
    float scale = 1.0f;
    float translation[3] = {};
    Points points;

    /// Indices into `cameras`. These replaced helpers that returned camera
    /// copies; a Camera owns its decoded Image, so those copies doubled the
    /// resident image memory for as long as training ran.
    std::tuple<std::vector<size_t>, int> trainIndices(bool validate, const std::string &valImage = "random") const;
    std::tuple<std::vector<size_t>, std::vector<size_t>> splitTrainTestIndices(int testEvery) const;
    void saveCameras(const std::string &filename, bool keepCrs) const;
};

// Auto-detect format and load dataset
InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath = "");

#endif
