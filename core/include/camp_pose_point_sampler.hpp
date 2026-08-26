#pragma once

#include "camp_pose_preconditioner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace msplat::detail {

constexpr size_t kCampMinimumRepresentativePointCount = 256;
constexpr size_t kCampDefaultRepresentativePointCount = 512;
constexpr size_t kCampMaximumRepresentativePointCount = 1024;
constexpr size_t kCampMaximumWorldPointPoolCount = 65536;
constexpr double kCampRepresentativeNearPlane = 0.01;

/// Effective, rectified training-raster geometry. `cameraToWorld` is row-major
/// OpenGL camera-to-world (Y up, Z back), matching msplat's Camera contract.
struct CampPoseCameraGeometry {
    std::array<double, 16> cameraToWorld{};
    double focalX = 0.0;
    double focalY = 0.0;
    double principalX = 0.0;
    double principalY = 0.0;
    int width = 0;
    int height = 0;
    /// Stable per-camera key used only to decorrelate deterministic sampling.
    uint64_t deterministicKey = 0;
};

/// A caller-bounded pool of normalized world-space XYZ values. When supplied,
/// stable IDs make the result independent of point storage order. Otherwise,
/// the canonical pool index is the stable identity and ordering is significant.
struct CampPoseWorldPointPool {
    const float* xyz = nullptr;
    const uint64_t* stableIds = nullptr;
    size_t count = 0;
};

struct CampPosePointSample {
    std::vector<std::array<double, 3>> viewPoints;
    /// `max(size_t)` identifies a synthetic frustum point.
    std::vector<size_t> sourcePointIndices;
    size_t visiblePointCount = 0;
    size_t selectedVisiblePointCount = 0;
    size_t fallbackPointCount = 0;
};

struct CampPosePreconditionerBuildResult {
    CampPoseMatrix hessian{};
    CampPoseMatrix preconditioner{};
    size_t visiblePointCount = 0;
    size_t selectedVisiblePointCount = 0;
    size_t fallbackPointCount = 0;
};

inline uint64_t campRepresentativeMix64(uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

/// Selects a bounded deterministic subset before per-camera visibility work.
/// Stable IDs make the selected set independent of point storage order. When
/// IDs are unavailable, canonical storage index is the fallback identity.
inline std::vector<size_t> selectCampPoseWorldPointPoolIndices(
    size_t pointCount, const uint64_t* stableIds,
    size_t maximumPointCount = kCampMaximumWorldPointPoolCount) {
    if (maximumPointCount == 0 ||
        maximumPointCount > kCampMaximumWorldPointPoolCount) {
        throw std::invalid_argument(
            "CamP maximum world point count must be in 1...65536");
    }
    const size_t selectedCount = std::min(pointCount, maximumPointCount);
    std::vector<size_t> result;
    result.reserve(selectedCount);
    if (selectedCount == pointCount) {
        for (size_t point = 0; point < pointCount; ++point)
            result.push_back(point);
        return result;
    }
    if (stableIds == nullptr) {
        for (size_t sample = 0; sample < selectedCount; ++sample) {
            result.push_back(static_cast<size_t>(
                (static_cast<uint64_t>(sample) * pointCount) /
                selectedCount));
        }
        return result;
    }

    using RankedPoint = std::tuple<uint64_t, uint64_t, size_t>;
    std::priority_queue<RankedPoint> selected;
    for (size_t point = 0; point < pointCount; ++point) {
        const uint64_t stableId = stableIds[point];
        const RankedPoint candidate{
            campRepresentativeMix64(stableId), stableId, point};
        if (selected.size() < selectedCount) {
            selected.push(candidate);
        } else if (candidate < selected.top()) {
            selected.pop();
            selected.push(candidate);
        }
    }

    std::vector<RankedPoint> rankedPoints;
    rankedPoints.reserve(selected.size());
    while (!selected.empty()) {
        rankedPoints.push_back(selected.top());
        selected.pop();
    }
    std::sort(rankedPoints.begin(), rankedPoints.end());
    for (const RankedPoint& point : rankedPoints)
        result.push_back(std::get<2>(point));
    return result;
}

inline double campRepresentativeUnitValue(uint64_t value) noexcept {
    return static_cast<double>(value >> 11) *
        (1.0 / 9007199254740992.0);
}

inline double campRepresentativeRadicalInverse(
    uint64_t index, uint32_t base) noexcept {
    const double inverseBase = 1.0 / static_cast<double>(base);
    double inversePower = inverseBase;
    double value = 0.0;
    while (index != 0) {
        value += static_cast<double>(index % base) * inversePower;
        index /= base;
        inversePower *= inverseBase;
    }
    return value;
}

inline void validateCampPoseCameraGeometry(
    const CampPoseCameraGeometry& camera) {
    if (camera.width <= 0 || camera.height <= 0) {
        throw std::invalid_argument(
            "CamP camera dimensions must be positive");
    }
    if (!std::isfinite(camera.focalX) ||
        !std::isfinite(camera.focalY) ||
        camera.focalX <= 0.0 || camera.focalY <= 0.0 ||
        !std::isfinite(camera.principalX) ||
        !std::isfinite(camera.principalY)) {
        throw std::invalid_argument(
            "CamP camera intrinsics must be finite with positive focal lengths");
    }
    for (double value : camera.cameraToWorld) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "CamP camera-to-world transform must be finite");
        }
    }
}

inline std::array<double, 3> campRepresentativeWorldToView(
    const CampPoseCameraGeometry& camera,
    const std::array<double, 3>& worldPoint) noexcept {
    const auto& matrix = camera.cameraToWorld;
    const double dx = worldPoint[0] - matrix[3];
    const double dy = worldPoint[1] - matrix[7];
    const double dz = worldPoint[2] - matrix[11];

    // Transpose the OpenGL C2W rotation, then flip Y and Z into the renderer's
    // x-right/y-down/z-forward view convention.
    return {
        matrix[0] * dx + matrix[4] * dy + matrix[8] * dz,
        -(matrix[1] * dx + matrix[5] * dy + matrix[9] * dz),
        -(matrix[2] * dx + matrix[6] * dy + matrix[10] * dz),
    };
}

namespace camp_pose_sampler_internal {

constexpr size_t kScreenGridSide = 16;
constexpr size_t kScreenCellCount =
    kScreenGridSide * kScreenGridSide;

struct Candidate {
    std::array<double, 3> viewPoint{};
    size_t sourceIndex = 0;
    uint64_t stableId = 0;
    uint64_t rank = 0;
};

inline bool candidateLess(const Candidate& lhs, const Candidate& rhs) noexcept {
    if (lhs.rank != rhs.rank) return lhs.rank < rhs.rank;
    if (lhs.stableId != rhs.stableId) return lhs.stableId < rhs.stableId;
    return lhs.sourceIndex < rhs.sourceIndex;
}

inline std::pair<double, double> fallbackDepthRange(
    std::vector<double> depths,
    bool hasWorldBounds,
    const std::array<double, 3>& minimumWorld,
    const std::array<double, 3>& maximumWorld,
    const CampPoseCameraGeometry& camera) {
    const double minimumFallbackDepth = std::nextafter(
        kCampRepresentativeNearPlane,
        std::numeric_limits<double>::infinity());
    if (!depths.empty()) {
        std::sort(depths.begin(), depths.end());
        const size_t last = depths.size() - 1;
        const double lower = depths[static_cast<size_t>(
            std::floor(0.05 * static_cast<double>(last)))];
        const double upper = depths[static_cast<size_t>(
            std::floor(0.95 * static_cast<double>(last)))];
        const double nearDepth = std::max(minimumFallbackDepth, lower);
        const double farDepth = upper > nearDepth * 1.01
            ? upper
            : nearDepth * 2.0;
        return {nearDepth, farDepth};
    }

    if (hasWorldBounds) {
        std::array<double, 3> center{};
        double radiusSquared = 0.0;
        double distanceSquared = 0.0;
        for (size_t axis = 0; axis < 3; ++axis) {
            center[axis] = 0.5 * (minimumWorld[axis] + maximumWorld[axis]);
            const double halfExtent =
                0.5 * (maximumWorld[axis] - minimumWorld[axis]);
            radiusSquared += halfExtent * halfExtent;
            const double cameraOffset =
                center[axis] - camera.cameraToWorld[axis * 4 + 3];
            distanceSquared += cameraOffset * cameraOffset;
        }
        const double radius = std::sqrt(radiusSquared);
        const double distance = std::sqrt(distanceSquared);
        const double nearDepth = std::max(
            minimumFallbackDepth, distance - radius);
        const double farDepth = std::max(
            nearDepth * 2.0, distance + radius);
        return {nearDepth, farDepth};
    }

    // InputData normalization keeps ordinary camera baselines near unit scale.
    // This path is only for an empty or wholly non-finite seed pool.
    return {0.1, 1.0};
}

} // namespace camp_pose_sampler_internal

/// Selects frustum-visible seed points with deterministic screen-space
/// stratification. If fewer than `targetPointCount` are visible, deterministic
/// Halton points fill the camera frustum so the Hessian always has a fixed
/// sample count. Visibility is geometric (not an occlusion or mask test).
inline CampPosePointSample sampleCampPoseRepresentativePoints(
    const CampPoseCameraGeometry& camera,
    const CampPoseWorldPointPool& pool,
    size_t targetPointCount = kCampDefaultRepresentativePointCount) {
    using namespace camp_pose_sampler_internal;

    validateCampPoseCameraGeometry(camera);
    if (targetPointCount < kCampMinimumRepresentativePointCount ||
        targetPointCount > kCampMaximumRepresentativePointCount) {
        throw std::invalid_argument(
            "CamP representative point count must be in 256...1024");
    }
    if (pool.count > kCampMaximumWorldPointPoolCount) {
        throw std::invalid_argument(
            "CamP world point pool exceeds its deterministic bound");
    }
    if (pool.count != 0 && pool.xyz == nullptr) {
        throw std::invalid_argument(
            "CamP world point pool requires XYZ storage");
    }

    std::array<std::vector<Candidate>, kScreenCellCount> cells;
    std::vector<double> positiveDepths;
    std::vector<double> visibleDepths;
    positiveDepths.reserve(pool.count);
    visibleDepths.reserve(std::min(pool.count, targetPointCount * 2));

    std::array<double, 3> minimumWorld = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    std::array<double, 3> maximumWorld = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    bool hasWorldBounds = false;
    size_t visiblePointCount = 0;
    const uint64_t cameraRankKey =
        campRepresentativeMix64(camera.deterministicKey);

    for (size_t pointIndex = 0; pointIndex < pool.count; ++pointIndex) {
        const size_t offset = pointIndex * 3;
        const std::array<double, 3> worldPoint = {
            static_cast<double>(pool.xyz[offset]),
            static_cast<double>(pool.xyz[offset + 1]),
            static_cast<double>(pool.xyz[offset + 2]),
        };
        if (!std::isfinite(worldPoint[0]) ||
            !std::isfinite(worldPoint[1]) ||
            !std::isfinite(worldPoint[2])) {
            continue;
        }
        hasWorldBounds = true;
        for (size_t axis = 0; axis < 3; ++axis) {
            minimumWorld[axis] = std::min(minimumWorld[axis], worldPoint[axis]);
            maximumWorld[axis] = std::max(maximumWorld[axis], worldPoint[axis]);
        }

        const std::array<double, 3> viewPoint =
            campRepresentativeWorldToView(camera, worldPoint);
        if (!std::isfinite(viewPoint[0]) ||
            !std::isfinite(viewPoint[1]) ||
            !std::isfinite(viewPoint[2]) ||
            !(viewPoint[2] > kCampRepresentativeNearPlane)) {
            continue;
        }
        positiveDepths.push_back(viewPoint[2]);

        const double pixelX =
            camera.focalX * viewPoint[0] / viewPoint[2] +
            camera.principalX;
        const double pixelY =
            camera.focalY * viewPoint[1] / viewPoint[2] +
            camera.principalY;
        if (!std::isfinite(pixelX) || !std::isfinite(pixelY) ||
            pixelX < 0.0 || pixelY < 0.0 ||
            pixelX >= static_cast<double>(camera.width) ||
            pixelY >= static_cast<double>(camera.height)) {
            continue;
        }

        ++visiblePointCount;
        visibleDepths.push_back(viewPoint[2]);
        const size_t cellX = std::min(
            kScreenGridSide - 1,
            static_cast<size_t>(pixelX * kScreenGridSide /
                                static_cast<double>(camera.width)));
        const size_t cellY = std::min(
            kScreenGridSide - 1,
            static_cast<size_t>(pixelY * kScreenGridSide /
                                static_cast<double>(camera.height)));
        const uint64_t stableId = pool.stableIds != nullptr
            ? pool.stableIds[pointIndex]
            : static_cast<uint64_t>(pointIndex);
        const uint64_t rank = campRepresentativeMix64(
            stableId ^ cameraRankKey);
        cells[cellY * kScreenGridSide + cellX].push_back(
            {viewPoint, pointIndex, stableId, rank});
    }

    for (auto& cell : cells)
        std::sort(cell.begin(), cell.end(), candidateLess);

    std::array<size_t, kScreenCellCount> cellOrder{};
    std::iota(cellOrder.begin(), cellOrder.end(), size_t{0});
    std::sort(cellOrder.begin(), cellOrder.end(),
        [&](size_t lhs, size_t rhs) {
            const uint64_t leftRank = campRepresentativeMix64(
                cameraRankKey ^ static_cast<uint64_t>(lhs));
            const uint64_t rightRank = campRepresentativeMix64(
                cameraRankKey ^ static_cast<uint64_t>(rhs));
            return leftRank != rightRank ? leftRank < rightRank : lhs < rhs;
        });

    CampPosePointSample result;
    result.visiblePointCount = visiblePointCount;
    result.viewPoints.reserve(targetPointCount);
    result.sourcePointIndices.reserve(targetPointCount);
    for (size_t rankInCell = 0;
         result.viewPoints.size() < targetPointCount;
         ++rankInCell) {
        bool selectedAny = false;
        for (size_t cellIndex : cellOrder) {
            const auto& cell = cells[cellIndex];
            if (rankInCell >= cell.size()) continue;
            result.viewPoints.push_back(cell[rankInCell].viewPoint);
            result.sourcePointIndices.push_back(cell[rankInCell].sourceIndex);
            selectedAny = true;
            if (result.viewPoints.size() == targetPointCount) break;
        }
        if (!selectedAny) break;
    }
    result.selectedVisiblePointCount = result.viewPoints.size();

    if (result.viewPoints.size() < targetPointCount) {
        std::vector<double> depthCandidates = !visibleDepths.empty()
            ? std::move(visibleDepths)
            : std::move(positiveDepths);
        const auto depthRange = fallbackDepthRange(
            std::move(depthCandidates), hasWorldBounds,
            minimumWorld, maximumWorld, camera);
        const double logNear = std::log(depthRange.first);
        const double logFar = std::log(depthRange.second);
        const uint64_t seed = campRepresentativeMix64(
            camera.deterministicKey ^ 0xd1b54a32d192ed03ULL);
        const double pixelOffsetX = campRepresentativeUnitValue(seed);
        const double pixelOffsetY = campRepresentativeUnitValue(
            campRepresentativeMix64(seed));
        const double depthOffset = campRepresentativeUnitValue(
            campRepresentativeMix64(campRepresentativeMix64(seed)));
        const size_t fallbackCount =
            targetPointCount - result.viewPoints.size();

        for (size_t fallbackIndex = 0;
             fallbackIndex < fallbackCount;
             ++fallbackIndex) {
            const uint64_t sequenceIndex =
                static_cast<uint64_t>(fallbackIndex) + 1;
            const double unitX = std::fmod(
                campRepresentativeRadicalInverse(sequenceIndex, 2) +
                    pixelOffsetX,
                1.0);
            const double unitY = std::fmod(
                campRepresentativeRadicalInverse(sequenceIndex, 3) +
                    pixelOffsetY,
                1.0);
            const double unitDepth = std::fmod(
                campRepresentativeRadicalInverse(sequenceIndex, 5) +
                    depthOffset,
                1.0);
            const double pixelX = 0.5 + unitX *
                static_cast<double>(std::max(0, camera.width - 1));
            const double pixelY = 0.5 + unitY *
                static_cast<double>(std::max(0, camera.height - 1));
            const double depth = std::exp(
                logNear + unitDepth * (logFar - logNear));
            result.viewPoints.push_back({
                (pixelX - camera.principalX) * depth / camera.focalX,
                (pixelY - camera.principalY) * depth / camera.focalY,
                depth,
            });
            result.sourcePointIndices.push_back(
                std::numeric_limits<size_t>::max());
        }
        result.fallbackPointCount = fallbackCount;
    }
    return result;
}

inline CampPosePreconditionerBuildResult buildCampPosePreconditioner(
    const CampPoseCameraGeometry& camera,
    const CampPoseWorldPointPool& pool,
    size_t targetPointCount = kCampDefaultRepresentativePointCount,
    double absoluteDamping = kCampAbsoluteDamping,
    double relativeDamping = kCampRelativeDamping) {
    const CampPosePointSample sample = sampleCampPoseRepresentativePoints(
        camera, pool, targetPointCount);
    CampPoseMatrix hessianSum{};
    for (const auto& viewPoint : sample.viewPoints) {
        accumulateCampPoseApproximateHessian(
            hessianSum,
            campPoseProjectionJacobian(
                viewPoint, camera.focalX, camera.focalY,
                static_cast<double>(camera.width),
                static_cast<double>(camera.height)));
    }

    CampPosePreconditionerBuildResult result;
    result.hessian = finishCampPoseApproximateHessian(
        hessianSum, sample.viewPoints.size());
    result.preconditioner = campPosePreconditionerFromHessian(
        result.hessian, absoluteDamping, relativeDamping);
    result.visiblePointCount = sample.visiblePointCount;
    result.selectedVisiblePointCount = sample.selectedVisiblePointCount;
    result.fallbackPointCount = sample.fallbackPointCount;
    return result;
}

} // namespace msplat::detail
