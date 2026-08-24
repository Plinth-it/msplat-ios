#include "input_data.hpp"
#include "dataset_errors.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <TargetConditionals.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int targetDimension(int sourceDimension, float downscaleFactor,
                    const char *axis, const std::string &path) {
    // Match Swift's Float(...).rounded(.towardZero) planning contract exactly.
    const float scaled = static_cast<float>(sourceDimension) / downscaleFactor;
    if (!std::isfinite(scaled) || scaled < 1.0f ||
        static_cast<double>(scaled) >
            static_cast<double>(std::numeric_limits<int>::max())) {
        throw msplat::InvalidDatasetError(
            "Image downscale produces an invalid " + std::string(axis) +
            " for " + path);
    }
    return static_cast<int>(scaled);
}

bool dimensionsShareAspect(int lhsWidth, int lhsHeight,
                           int rhsWidth, int rhsHeight) {
    if (lhsWidth <= 0 || lhsHeight <= 0 || rhsWidth <= 0 || rhsHeight <= 0)
        return false;

    const long double lhs =
        static_cast<long double>(lhsWidth) * rhsHeight;
    const long double rhs =
        static_cast<long double>(rhsWidth) * lhsHeight;
    const long double denominator = std::max(lhs, rhs);
    const long double smallestDimension = std::min({
        static_cast<long double>(lhsWidth),
        static_cast<long double>(lhsHeight),
        static_cast<long double>(rhsWidth),
        static_cast<long double>(rhsHeight),
    });
    // Permit at most one-pixel integer rounding for ordinary image sizes while
    // keeping tiny or malformed calibration dimensions from creating a broad
    // aspect-ratio tolerance.
    const long double relativeTolerance =
        std::min(0.002L, 1.0L / smallestDimension);
    return std::abs(lhs - rhs) / denominator <= relativeTolerance;
}

uint64_t fullCoverageUnits(int width, int height, const std::string &path) {
    if (width <= 0 || height <= 0) {
        throw msplat::InvalidDatasetError(
            "Training target has invalid dimensions for " + path);
    }
    const uint64_t pixelCount =
        static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    if (pixelCount > std::numeric_limits<uint64_t>::max() / 255u) {
        throw msplat::InvalidDatasetError(
            "Training target coverage exceeds the supported range for " + path);
    }
    return pixelCount * 255u;
}

uint64_t maskCoverageUnits(
    const CoverageMask &mask, const std::string &path) {
    const uint64_t expectedSize =
        static_cast<uint64_t>(mask.width) *
        static_cast<uint64_t>(mask.height);
    if (mask.width <= 0 || mask.height <= 0 ||
        expectedSize != mask.data.size()) {
        throw msplat::InvalidDatasetError(
            "Training mask storage does not match its dimensions for " + path);
    }

    uint64_t units = 0;
    for (uint8_t value : mask.data) {
        if (units > std::numeric_limits<uint64_t>::max() - value) {
            throw msplat::InvalidDatasetError(
                "Training mask coverage exceeds the supported range for " +
                path);
        }
        units += value;
    }
    if (units == 0) {
        throw msplat::InvalidDatasetError(
            "Training mask has zero coverage at the generated resolution: " +
            path);
    }
    return units;
}

bool isMirroredExifOrientation(int orientation) {
    return orientation == 2 || orientation == 4 ||
           orientation == 5 || orientation == 7;
}

} // namespace

// ── Image loading ───────────────────────────────────────────────────────────

void Camera::setCameraToWorld(const float pose[16]) {
    if (!pose)
        throw std::invalid_argument("Camera pose must not be null");
    for (int index = 0; index < 16; ++index) {
        if (!std::isfinite(pose[index])) {
            throw std::invalid_argument(
                "Camera pose must contain only finite values");
        }
    }
    std::copy_n(pose, 16, camToWorld);
    invalidateProjectionCache();
}

bool Camera::projectionCacheMatchesPose() const noexcept {
    return cachedPoseValid &&
        std::equal(std::begin(camToWorld), std::end(camToWorld),
                   cachedCameraToWorld.begin());
}

void Camera::recordProjectionCachePose() noexcept {
    std::copy(std::begin(camToWorld), std::end(camToWorld),
              cachedCameraToWorld.begin());
    cachedPoseValid = true;
}

void Camera::invalidateProjectionCache() {
    cachedViewMat.reset();
    cachedProjViewMat.reset();
    cachedPoseValid = false;
}

void Camera::loadImage(float downscaleFactor) {
    if (!image.empty() && loadedImageDownscaleFactor == downscaleFactor &&
        (!trainingMask || !coverageMask.empty())) {
        return;
    }
    releaseImageMemory();

    if (!std::isfinite(downscaleFactor) || downscaleFactor < 1.0f) {
        throw std::invalid_argument(
            "Image downscale factor must be finite and at least 1");
    }

    // Capture the dataset's own numbers once, then always derive from them.
    if (!declared.captured) {
        declared = {true, fx, fy, cx, cy, k1, k2, k3, p1, p2, width, height};
    }
    fx = declared.fx; fy = declared.fy; cx = declared.cx; cy = declared.cy;
    k1 = declared.k1; k2 = declared.k2; k3 = declared.k3;
    p1 = declared.p1; p2 = declared.p2;
    width = declared.width; height = declared.height;

    if (width < 0 || height < 0 || (width == 0) != (height == 0)) {
        throw msplat::InvalidDatasetError(
            "Camera dimensions must both be positive or both be omitted for " +
            filePath);
    }

    const ImageSourceInfo sourceInfo = inspectImageSource(filePath);
    const bool normalizeExif =
        rasterOrientation == RasterOrientation::ExifNormalized;
    if (normalizeExif &&
        isMirroredExifOrientation(sourceInfo.exifOrientation)) {
        throw msplat::InvalidDatasetError(
            "EXIF-normalized camera geometry cannot represent mirrored EXIF "
            "orientation " + std::to_string(sourceInfo.exifOrientation) +
            " for " + filePath +
            "; preserve encoded pixels or provide an unmirrored source");
    }
    const int sourceWidth = normalizeExif
        ? sourceInfo.orientedWidth : sourceInfo.rawWidth;
    const int sourceHeight = normalizeExif
        ? sourceInfo.orientedHeight : sourceInfo.rawHeight;

    ImageSourceInfo maskSourceInfo;
    if (trainingMask) {
        maskSourceInfo = inspectImageSource(trainingMask->path);
        const int maskWidth = normalizeExif
            ? maskSourceInfo.orientedWidth : maskSourceInfo.rawWidth;
        const int maskHeight = normalizeExif
            ? maskSourceInfo.orientedHeight : maskSourceInfo.rawHeight;
        if (maskWidth != sourceWidth || maskHeight != sourceHeight) {
            throw msplat::InvalidDatasetError(
                "Training mask dimensions " + std::to_string(maskWidth) +
                "x" + std::to_string(maskHeight) +
                " do not match the selected " +
                (normalizeExif ? "EXIF-oriented" : "encoded") +
                " image dimensions " + std::to_string(sourceWidth) + "x" +
                std::to_string(sourceHeight) + " for " + filePath +
                ": " + trainingMask->path);
        }
    }

    // Each descriptor explicitly names the pixel frame its calibration uses.
    // Existing adapters preserve encoded pixels; a native adapter may request
    // ImageIO's EXIF-normalized frame after transforming its calibration and
    // camera pose to match.
    if (width > 0 &&
        !dimensionsShareAspect(width, height,
                               sourceWidth, sourceHeight)) {
        std::string detail;
        if (!normalizeExif && sourceInfo.exifOrientation >= 5 &&
            dimensionsShareAspect(width, height,
                                  sourceInfo.orientedWidth,
                                  sourceInfo.orientedHeight)) {
            detail = " The declared dimensions instead match the EXIF-oriented "
                     "raster, but this camera path preserves encoded pixel "
                     "coordinates; normalize both pixels and calibration before "
                     "training.";
        } else if (normalizeExif && sourceInfo.exifOrientation >= 5 &&
                   dimensionsShareAspect(width, height,
                                         sourceInfo.rawWidth,
                                         sourceInfo.rawHeight)) {
            detail = " The declared dimensions instead match the encoded "
                     "raster, but this camera requests EXIF-normalized pixels; "
                     "provide calibration and pose in the normalized frame.";
        }
        throw msplat::InvalidDatasetError(
            "Camera dimensions " + std::to_string(width) + "x" +
            std::to_string(height) + " do not match " +
            (normalizeExif ? "EXIF-oriented" : "encoded") +
            " image dimensions " + std::to_string(sourceWidth) + "x" +
            std::to_string(sourceHeight) + " for " + filePath + "." +
            detail);
    }

    const int targetWidth = targetDimension(
        sourceWidth, downscaleFactor, "width", filePath);
    const int targetHeight = targetDimension(
        sourceHeight, downscaleFactor, "height", filePath);
    RGBA8Image raw = imreadRGBA8(
        filePath, sourceInfo, targetWidth, targetHeight, normalizeExif);
    CoverageMask rawMask;
    if (trainingMask) {
        rawMask = imreadCoverageMask(
            trainingMask->path, maskSourceInfo, targetWidth, targetHeight,
            normalizeExif, trainingMask->channel);
        (void)maskCoverageUnits(rawMask, trainingMask->path);
    }

    // Scale from the calibration's declared pixel frame directly to the exact
    // target canvas. COLMAP uses image-edge coordinates (the upper-left pixel
    // center is 0.5, 0.5), so focal lengths and principal points both scale
    // directly with their axis.
    const int calibrationWidth = width > 0 ? width : sourceWidth;
    const int calibrationHeight = height > 0 ? height : sourceHeight;
    const float sx = static_cast<float>(targetWidth) /
                     static_cast<float>(calibrationWidth);
    const float sy = static_cast<float>(targetHeight) /
                     static_cast<float>(calibrationHeight);
    fx *= sx;
    fy *= sy;
    cx *= sx;
    cy *= sy;
    width = targetWidth;
    height = targetHeight;

    // Undistort if needed
    if (hasDistortion()) {
        if (trainingMask) {
            auto result = undistortRGBA8ImageAndCoverageMask(
                raw, rawMask, fx, fy, cx, cy, k1, k2, p1, p2, k3);
            raw = std::move(result.image);
            rawMask = std::move(result.coverageMask);
            fx = result.fx; fy = result.fy;
            cx = result.cx; cy = result.cy;
            width = result.width; height = result.height;
            (void)maskCoverageUnits(rawMask, trainingMask->path);
        } else {
            auto result = undistortRGBA8Image(
                raw, fx, fy, cx, cy, k1, k2, p1, p2, k3);
            raw = std::move(result.image);
            fx = result.fx; fy = result.fy;
            cx = result.cx; cy = result.cy;
            width = result.width; height = result.height;
        }
        k1 = k2 = k3 = p1 = p2 = 0;
    }

    image = std::move(raw);
    coverageMask = std::move(rawMask);
    if (trainingMask) {
        coverageUnitsByDownscale.emplace(
            1, maskCoverageUnits(coverageMask, trainingMask->path));
    }
    loadedImageDownscaleFactor = downscaleFactor;
}

const RGBA8Image& Camera::getImage(int downscaleFactor) {
    if (image.empty()) loadImage(1.0f);
    if (downscaleFactor <= 1) return image;

    auto it = imagePyramids.find(downscaleFactor);
    if (it != imagePyramids.end()) return it->second;

    int newW = image.width / downscaleFactor;
    int newH = image.height / downscaleFactor;
    RGBA8Image scaled = resizeRGBA8Area(image, newW, newH);
    auto inserted = imagePyramids.emplace(downscaleFactor, std::move(scaled));
    return inserted.first->second;
}

const CoverageMask& Camera::getCoverageMask(int downscaleFactor) {
    if (!trainingMask) {
        throw std::logic_error(
            "Coverage was requested for a camera without a training mask");
    }
    if (coverageMask.empty()) loadImage(1.0f);
    if (downscaleFactor <= 1) return coverageMask;

    auto it = coverageMaskPyramids.find(downscaleFactor);
    if (it != coverageMaskPyramids.end()) return it->second;

    const int newWidth = coverageMask.width / downscaleFactor;
    const int newHeight = coverageMask.height / downscaleFactor;
    CoverageMask scaled = resizeCoverageArea(
        coverageMask, newWidth, newHeight);
    const uint64_t units = maskCoverageUnits(scaled, trainingMask->path);
    auto inserted = coverageMaskPyramids.emplace(
        downscaleFactor, std::move(scaled));
    coverageUnitsByDownscale.emplace(downscaleFactor, units);
    return inserted.first->second;
}

uint64_t Camera::getCoverageUnits(int downscaleFactor) {
    if (!trainingMask) {
        throw std::logic_error(
            "Coverage was requested for a camera without a training mask");
    }
    const int cacheKey = std::max(downscaleFactor, 1);
    const auto cached = coverageUnitsByDownscale.find(cacheKey);
    if (cached != coverageUnitsByDownscale.end()) return cached->second;

    const CoverageMask &mask = getCoverageMask(cacheKey);
    const uint64_t units = maskCoverageUnits(mask, trainingMask->path);
    coverageUnitsByDownscale.emplace(cacheKey, units);
    return units;
}

CameraTrainingTarget Camera::getGPUTrainingTarget(int downscaleFactor) {
    const auto imageIt = mtensorImageCache.find(downscaleFactor);
    const auto maskIt = mtensorCoverageMaskCache.find(downscaleFactor);
    const bool imageIsResident = imageIt != mtensorImageCache.end();
    const bool maskIsResident = !trainingMask ||
        maskIt != mtensorCoverageMaskCache.end();
    if (imageIsResident && maskIsResident) {
        const uint64_t units = trainingMask
            ? getCoverageUnits(downscaleFactor)
            : fullCoverageUnits(
                  static_cast<int>(imageIt->second.size(1)),
                  static_cast<int>(imageIt->second.size(0)), filePath);
        return {
            &imageIt->second,
            trainingMask ? &maskIt->second : nullptr,
            units,
        };
    }

    // The training resolution only moves from coarse to fine. Keeping GPU and
    // CPU pyramid copies for earlier stages makes an uninterrupted run retain
    // substantially more memory than a checkpoint resume at the same step.
    for (auto& item : mtensorImageCache) item.second.reset();
    mtensorImageCache.clear();
    for (auto& item : mtensorCoverageMaskCache) item.second.reset();
    mtensorCoverageMaskCache.clear();
    imagePyramids.clear();
    coverageMaskPyramids.clear();

    const CoverageMask *mask = nullptr;
    uint64_t units = 0;
    if (trainingMask) {
        // Validate coverage before materializing the matching RGB pyramid so a
        // zero-coverage stage cannot leave half of a CPU cache pair resident.
        mask = &getCoverageMask(downscaleFactor);
        units = getCoverageUnits(downscaleFactor);
    }

    const RGBA8Image& img = getImage(downscaleFactor);
    if (img.empty())
        throw msplat::DatasetIOError("Failed to decode image: " + filePath);
    const uint64_t fullUnits =
        fullCoverageUnits(img.width, img.height, filePath);
    const uint64_t pixelCount = fullUnits / 255u;
    const uint64_t expectedImageBytes = pixelCount * 4u;
    if (expectedImageBytes != img.data.size()) {
        throw msplat::InvalidDatasetError(
            "Training image storage does not match its dimensions for " +
            filePath);
    }
    if (mask) {
        if (mask->width != img.width || mask->height != img.height) {
            throw msplat::InvalidDatasetError(
                "Generated training mask dimensions do not match the image: " +
                trainingMask->path);
        }
        if (pixelCount != mask->data.size()) {
            throw msplat::InvalidDatasetError(
                "Training mask storage does not match its dimensions for " +
                trainingMask->path);
        }
    } else {
        units = fullUnits;
    }

    // Allocate and populate the complete pair before publishing either cache
    // entry. A mask validation/allocation failure must not strand an RGB tensor
    // outside CameraImageCache accounting.
    MTensor imageStorage = gpu_empty(
        {img.height, img.width, 4}, DType::UInt8);
    memcpy(imageStorage.data_ptr(), img.ptr(),
           static_cast<size_t>(expectedImageBytes));
    std::optional<MTensor> maskStorage;
    if (mask) {
        maskStorage.emplace(gpu_empty(
            {mask->height, mask->width}, DType::UInt8));
        memcpy(maskStorage->data_ptr(), mask->ptr(),
               static_cast<size_t>(pixelCount));
    }

    try {
        auto imageInserted = mtensorImageCache.emplace(
            downscaleFactor, std::move(imageStorage));
        if (!imageInserted.second) {
            throw std::logic_error("Training image cache insertion failed");
        }

        MTensor *maskTensor = nullptr;
        if (maskStorage) {
            auto maskInserted = mtensorCoverageMaskCache.emplace(
                downscaleFactor, std::move(*maskStorage));
            if (!maskInserted.second) {
                throw std::logic_error(
                    "Training mask cache insertion failed");
            }
            maskTensor = &maskInserted.first->second;
        }
        return {&imageInserted.first->second, maskTensor, units};
    } catch (...) {
        if (auto it = mtensorImageCache.find(downscaleFactor);
            it != mtensorImageCache.end()) {
            it->second.reset();
            mtensorImageCache.erase(it);
        }
        if (auto it = mtensorCoverageMaskCache.find(downscaleFactor);
            it != mtensorCoverageMaskCache.end()) {
            it->second.reset();
            mtensorCoverageMaskCache.erase(it);
        }
        throw;
    }
}

MTensor& Camera::getGPUImage(int downscaleFactor) {
    return *getGPUTrainingTarget(downscaleFactor).image;
}

void Camera::releaseImageMemory() {
    releaseCpuImageMemory();
    for (auto& item : mtensorImageCache) {
        item.second.reset();
    }
    mtensorImageCache.clear();
    for (auto& item : mtensorCoverageMaskCache) {
        item.second.reset();
    }
    mtensorCoverageMaskCache.clear();
    coverageUnitsByDownscale.clear();
    loadedImageDownscaleFactor = 0.0f;
}

void Camera::releaseCpuImageMemory() {
    image = RGBA8Image();
    coverageMask = CoverageMask();
    imagePyramids.clear();
    coverageMaskPyramids.clear();
}

size_t Camera::cachedCpuImageBytes() const {
    size_t bytes = image.data.size();
    for (const auto& item : imagePyramids) {
        bytes += item.second.data.size();
    }
    bytes += coverageMask.data.size();
    for (const auto& item : coverageMaskPyramids) {
        bytes += item.second.data.size();
    }
    return bytes;
}

size_t Camera::cachedGpuImageBytes() const {
    size_t bytes = 0;
    for (const auto& item : mtensorImageCache) {
        bytes += item.second.nbytes();
    }
    for (const auto& item : mtensorCoverageMaskCache) {
        bytes += item.second.nbytes();
    }
    return bytes;
}

size_t Camera::cachedImageBytes() const {
    return cachedCpuImageBytes() + cachedGpuImageBytes();
}

// ── Image cache ─────────────────────────────────────────────────────────────

size_t CameraImageCache::defaultBudgetBytes() {
    if (const char *env = std::getenv("MSPLAT_IMAGE_CACHE_MB")) {
        int mb = std::atoi(env);
        if (mb > 0) return (size_t)mb * 1024 * 1024;
    }
#if TARGET_OS_IPHONE
    return (size_t)512 * 1024 * 1024;
#else
    return (size_t)2048 * 1024 * 1024;
#endif
}

MTensor& CameraImageCache::gpuImage(std::vector<Camera> &cameras, size_t index,
                                    int downscaleFactor) {
    return *gpuTrainingTarget(cameras, index, downscaleFactor).image;
}

CameraTrainingTarget CameraImageCache::gpuTrainingTarget(
    std::vector<Camera> &cameras, size_t index, int downscaleFactor) {
    Camera &cam = cameras[index];
    const bool imageIsResident =
        cam.mtensorImageCache.find(downscaleFactor) !=
        cam.mtensorImageCache.end();
    const bool maskIsResident = !cam.trainingMask ||
        cam.mtensorCoverageMaskCache.find(downscaleFactor) !=
            cam.mtensorCoverageMaskCache.end();
    try {
        CameraTrainingTarget target;
        if (imageIsResident && maskIsResident) {
            ++_hitCount;
            target = cam.getGPUTrainingTarget(downscaleFactor);
        } else {
            ++_missCount;
            cam.loadImage(_downscaleFactor);
            target = cam.getGPUTrainingTarget(downscaleFactor);
        }

        // Once the compact GPU pair is published, retaining decoded pixels or
        // a CPU pyramid only duplicates cache state. A later resolution-stage
        // miss deliberately re-decodes that camera once.
        cam.releaseCpuImageMemory();

        Entry &entry = _entries[index];
        entry.lastUse = ++_clock;
        entry.cpuBytes = cam.cachedCpuImageBytes();
        entry.gpuBytes = cam.cachedGpuImageBytes();
        evict(cameras, index);
        return target;
    } catch (...) {
        // A failed lazy decode, validation, or paired upload must not leave
        // resident bytes that the LRU has never accounted for.
        cam.releaseImageMemory();
        _entries.erase(index);
        throw;
    }
}

Camera& CameraImageCache::ensureLoaded(std::vector<Camera> &cameras, size_t index) {
    Camera &cam = cameras[index];
    // Render paths need corrected calibration, not decoded pixels. Preserve
    // the geometry established by an earlier compact-target upload.
    if (cam.loadedImageDownscaleFactor != _downscaleFactor)
        cam.loadImage(_downscaleFactor);

    Entry &entry = _entries[index];
    entry.lastUse = ++_clock;
    entry.cpuBytes = cam.cachedCpuImageBytes();
    entry.gpuBytes = cam.cachedGpuImageBytes();
    evict(cameras, index);
    return cam;
}

void CameraImageCache::evict(std::vector<Camera> &cameras, size_t protectedIndex) {
    while (cachedBytes() > _budgetBytes && _entries.size() > 1) {
        auto victim = _entries.end();
        for (auto it = _entries.begin(); it != _entries.end(); ++it) {
            if (it->first == protectedIndex) continue;
            if (victim == _entries.end() || it->second.lastUse < victim->second.lastUse) {
                victim = it;
            }
        }
        if (victim == _entries.end()) break;

        cameras[victim->first].releaseImageMemory();
        _entries.erase(victim);
    }
}

size_t CameraImageCache::cachedBytes() const {
    size_t bytes = 0;
    for (const auto &item : _entries) bytes += item.second.bytes();
    return bytes;
}

size_t CameraImageCache::cachedCpuBytes() const {
    size_t bytes = 0;
    for (const auto &item : _entries) bytes += item.second.cpuBytes;
    return bytes;
}

size_t CameraImageCache::cachedGpuBytes() const {
    size_t bytes = 0;
    for (const auto &item : _entries) bytes += item.second.gpuBytes;
    return bytes;
}

// ── Scale & center ──────────────────────────────────────────────────────────

void autoScaleAndCenter(InputData &data) {
    if (data.cameras.empty()) return;

    // Compute mean camera position
    double mean[3] = {};
    for (auto &cam : data.cameras) {
        mean[0] += static_cast<double>(cam.camToWorld[3]);
        mean[1] += static_cast<double>(cam.camToWorld[7]);
        mean[2] += static_cast<double>(cam.camToWorld[11]);
    }
    const double n = static_cast<double>(data.cameras.size());
    mean[0] /= n; mean[1] /= n; mean[2] /= n;

    auto checkedFloat = [](double value, const char *label) {
        if (!std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<float>::max()) {
            throw std::invalid_argument(
                std::string("Dataset ") + label + " exceeds float32 range");
        }
        return static_cast<float>(value);
    };

    data.translation[0] = checkedFloat(mean[0], "translation");
    data.translation[1] = checkedFloat(mean[1], "translation");
    data.translation[2] = checkedFloat(mean[2], "translation");

    // Center camera poses
    for (auto &cam : data.cameras) {
        cam.camToWorld[3] = checkedFloat(
            static_cast<double>(cam.camToWorld[3]) - mean[0],
            "centered camera position");
        cam.camToWorld[7] = checkedFloat(
            static_cast<double>(cam.camToWorld[7]) - mean[1],
            "centered camera position");
        cam.camToWorld[11] = checkedFloat(
            static_cast<double>(cam.camToWorld[11]) - mean[2],
            "centered camera position");
    }

    // Compute scale from max absolute camera position
    double maxAbs = 0.0;
    for (auto &cam : data.cameras) {
        maxAbs = std::max(maxAbs, std::abs(static_cast<double>(cam.camToWorld[3])));
        maxAbs = std::max(maxAbs, std::abs(static_cast<double>(cam.camToWorld[7])));
        maxAbs = std::max(maxAbs, std::abs(static_cast<double>(cam.camToWorld[11])));
    }
    data.scale = checkedFloat(maxAbs > 0.0 ? 1.0 / maxAbs : 1.0,
                              "normalization scale");

    // Apply scale to camera positions
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  *= data.scale;
        cam.camToWorld[7]  *= data.scale;
        cam.camToWorld[11] *= data.scale;
        cam.invalidateProjectionCache();
    }

    // Apply to point cloud
    for (int64_t i = 0; i < data.points.count; i++) {
        data.points.xyz[i*3+0] = checkedFloat(
            (static_cast<double>(data.points.xyz[i*3+0]) - mean[0]) * data.scale,
            "normalized point coordinate");
        data.points.xyz[i*3+1] = checkedFloat(
            (static_cast<double>(data.points.xyz[i*3+1]) - mean[1]) * data.scale,
            "normalized point coordinate");
        data.points.xyz[i*3+2] = checkedFloat(
            (static_cast<double>(data.points.xyz[i*3+2]) - mean[2]) * data.scale,
            "normalized point coordinate");
    }
}

// ── Train/test split ────────────────────────────────────────────────────────

std::tuple<std::vector<size_t>, int> InputData::trainIndices(bool validate, const std::string &valImage) const {
    std::vector<size_t> train;
    if (!validate) {
        train.reserve(cameras.size());
        for (size_t i = 0; i < cameras.size(); i++) train.push_back(i);
        return {train, -1};
    }

    // Find validation camera
    int valIdx = -1;
    if (valImage == "random") {
        std::mt19937 rng(42);
        valIdx = rng() % cameras.size();
    } else {
        for (int i = 0; i < (int)cameras.size(); i++) {
            if (cameras[i].filePath.find(valImage) != std::string::npos) { valIdx = i; break; }
        }
    }
    if (valIdx < 0) valIdx = 0;

    for (int i = 0; i < (int)cameras.size(); i++)
        if (i != valIdx) train.push_back((size_t)i);

    return {train, valIdx};
}

std::tuple<std::vector<size_t>, std::vector<size_t>> InputData::splitTrainTestIndices(int testEvery) const {
    std::vector<size_t> train, test;
    for (int i = 0; i < (int)cameras.size(); i++) {
        if (i % testEvery == 0)
            test.push_back((size_t)i);
        else
            train.push_back((size_t)i);
    }
    return {train, test};
}

// ── Save cameras ────────────────────────────────────────────────────────────

void InputData::saveCameras(const std::string &filename, bool keepCrs) const {
    json arr = json::array();
    for (auto &cam : cameras) {
        json c;
        c["file_path"] = fs::path(cam.filePath).filename().string();
        c["width"] = cam.width;
        c["height"] = cam.height;
        c["fx"] = cam.fx; c["fy"] = cam.fy;
        c["cx"] = cam.cx; c["cy"] = cam.cy;

        // Extract rotation and translation from camToWorld
        float R[9], T[3];
        // Undo OpenGL flip (negate columns 1,2 back to OpenCV convention)
        R[0] =  cam.camToWorld[0]; R[1] = -cam.camToWorld[1]; R[2] = -cam.camToWorld[2];
        R[3] =  cam.camToWorld[4]; R[4] = -cam.camToWorld[5]; R[5] = -cam.camToWorld[6];
        R[6] =  cam.camToWorld[8]; R[7] = -cam.camToWorld[9]; R[8] = -cam.camToWorld[10];
        T[0] =  cam.camToWorld[3]; T[1] =  cam.camToWorld[7]; T[2] =  cam.camToWorld[11];

        if (keepCrs) {
            T[0] = T[0] / scale + translation[0];
            T[1] = T[1] / scale + translation[1];
            T[2] = T[2] / scale + translation[2];
        }

        c["rotation"] = {{R[0],R[1],R[2]},{R[3],R[4],R[5]},{R[6],R[7],R[8]}};
        c["translation"] = {T[0], T[1], T[2]};
        arr.push_back(c);
    }

    std::ofstream f(filename);
    f << arr.dump(2);
}

// ── Canonical descriptor materialization and format dispatch ────────────────

InputData inputDataFromDescriptor(DatasetDescriptor descriptor) {
    validateDatasetDescriptor(descriptor);

    InputData data;
    data.cameras.reserve(descriptor.frames.size());
    data.metadata.frameIds.reserve(descriptor.frames.size());
    data.metadata.calibrationIds.reserve(descriptor.frames.size());
    for (auto &frame : descriptor.frames) {
        Camera camera;
        camera.width = frame.calibration.width;
        camera.height = frame.calibration.height;
        camera.fx = frame.calibration.fx;
        camera.fy = frame.calibration.fy;
        camera.cx = frame.calibration.cx;
        camera.cy = frame.calibration.cy;
        camera.k1 = frame.calibration.k1;
        camera.k2 = frame.calibration.k2;
        camera.k3 = frame.calibration.k3;
        camera.p1 = frame.calibration.p1;
        camera.p2 = frame.calibration.p2;
        camera.declared = {
            true,
            camera.fx, camera.fy, camera.cx, camera.cy,
            camera.k1, camera.k2, camera.k3, camera.p1, camera.p2,
            camera.width, camera.height,
        };
        camera.setCameraToWorld(frame.cameraToWorld.data());
        camera.filePath = std::move(frame.imagePath);
        camera.rasterOrientation = frame.rasterOrientation;
        camera.trainingMask = std::move(frame.trainingMask);
        data.cameras.push_back(std::move(camera));
        data.metadata.frameIds.push_back(std::move(frame.id));
        data.metadata.calibrationIds.push_back(std::move(frame.calibrationId));
    }

    data.points.count = static_cast<int64_t>(descriptor.points.count());
    data.points.xyz = std::move(descriptor.points.xyz);
    data.points.rgb = std::move(descriptor.points.rgb);
    data.metadata.pointSourceIds = std::move(descriptor.points.sourceIds);
    data.metadata.pointReprojectionErrors =
        std::move(descriptor.points.reprojectionErrors);
    data.metadata.observations = std::move(descriptor.observations);
    data.metadata.provenance = std::move(descriptor.provenance);

    autoScaleAndCenter(data);
    return data;
}

DatasetDescriptor datasetDescriptorFromX(
    const std::string &path, const std::string &colmapImagePath) {
    return datasetDescriptorFromX(path, colmapImagePath, false);
}

DatasetDescriptor datasetDescriptorFromX(
    const std::string &path, const std::string &colmapImagePath,
    bool discoverTrainingMasks) {
    fs::path root(path);
    auto validated = [](DatasetDescriptor descriptor) {
        validateDatasetDescriptor(descriptor);
        return descriptor;
    };

    // Nerfstudio: transforms.json
    if (fs::exists(root / "transforms.json"))
        return validated(loaders::loadNerfstudio(path));

    // COLMAP: cameras.bin or cameras.txt (direct or in sparse/0/)
    for (const auto &name : {"cameras.bin", "cameras.txt"})
        if (fs::exists(root / name) || fs::exists(root / "sparse" / "0" / name))
            return validated(loaders::loadColmap(
                path, colmapImagePath, discoverTrainingMasks));

    // Polycam: keyframes/ directory or cameras.json
    if (fs::exists(root / "keyframes" / "corrected_cameras") ||
        fs::exists(root / "keyframes" / "cameras") ||
        fs::exists(root / "cameras.json"))
        return validated(loaders::loadPolycam(path));

    throw std::runtime_error("Unrecognized dataset format in: " + path +
        "\nSupported: COLMAP (cameras.bin or cameras.txt), Nerfstudio (transforms.json), Polycam (keyframes/)");
}

InputData inputDataFromX(
    const std::string &path, const std::string &colmapImagePath) {
    return inputDataFromX(path, colmapImagePath, false);
}

InputData inputDataFromX(
    const std::string &path, const std::string &colmapImagePath,
    bool discoverTrainingMasks) {
    return inputDataFromDescriptor(
        datasetDescriptorFromX(
            path, colmapImagePath, discoverTrainingMasks));
}
