#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace msplat::detail {

struct PoseRefinementGeometry {
    std::array<float, 6> poseDelta{};
    float translationNorm = 0.0f;
    float rotationNorm = 0.0f;
    std::array<float, 16> correctedCameraToWorld{};
};

inline std::array<float, 9> poseSo3Exp(const float rotation[3]) {
    const float theta2 = rotation[0] * rotation[0] +
        rotation[1] * rotation[1] + rotation[2] * rotation[2];
    float a = 0.0f;
    float b = 0.0f;
    if (theta2 < 1.0e-8f) {
        const float theta4 = theta2 * theta2;
        a = 1.0f - theta2 / 6.0f + theta4 / 120.0f;
        b = 0.5f - theta2 / 24.0f + theta4 / 720.0f;
    } else {
        const float theta = std::sqrt(theta2);
        a = std::sin(theta) / theta;
        b = (1.0f - std::cos(theta)) / theta2;
    }

    const float x = rotation[0];
    const float y = rotation[1];
    const float z = rotation[2];
    const std::array<float, 9> skew = {
        0.0f, -z, y,
        z, 0.0f, -x,
        -y, x, 0.0f,
    };
    std::array<float, 9> skew2{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            for (int inner = 0; inner < 3; ++inner) {
                skew2[row * 3 + column] +=
                    skew[row * 3 + inner] * skew[inner * 3 + column];
            }
        }
    }

    std::array<float, 9> result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            const int index = row * 3 + column;
            result[index] = (row == column ? 1.0f : 0.0f) +
                a * skew[index] + b * skew2[index];
        }
    }
    return result;
}

/// Reproduces Metal's left correction of the OpenCV world-to-camera matrix,
/// then returns an OpenGL camera-to-world matrix in the dataset's original
/// pre-normalization coordinate system. The first three exposed delta values
/// are converted from normalized view-space lengths to original dataset units;
/// the final three remain axis-angle radians.
inline PoseRefinementGeometry makePoseRefinementGeometry(
    const float baseCameraToWorld[16], const float normalizedPoseDelta[6],
    float normalizationScale, const float normalizationTranslation[3]) {
    if (!baseCameraToWorld || !normalizedPoseDelta ||
        !normalizationTranslation) {
        throw std::invalid_argument(
            "Pose-refinement geometry inputs must not be null");
    }
    if (!std::isfinite(normalizationScale) || normalizationScale <= 0.0f) {
        throw std::invalid_argument(
            "Pose-refinement normalization scale must be finite and positive");
    }
    for (int index = 0; index < 16; ++index) {
        if (!std::isfinite(baseCameraToWorld[index])) {
            throw std::invalid_argument(
                "Pose-refinement base pose must contain only finite values");
        }
    }
    for (int index = 0; index < 6; ++index) {
        if (!std::isfinite(normalizedPoseDelta[index])) {
            throw std::invalid_argument(
                "Pose-refinement delta must contain only finite values");
        }
    }
    for (int index = 0; index < 3; ++index) {
        if (!std::isfinite(normalizationTranslation[index])) {
            throw std::invalid_argument(
                "Pose-refinement normalization translation must be finite");
        }
    }

    PoseRefinementGeometry result;
    float translationNorm2 = 0.0f;
    float rotationNorm2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        result.poseDelta[component] =
            normalizedPoseDelta[component] / normalizationScale;
        result.poseDelta[component + 3] = normalizedPoseDelta[component + 3];
        translationNorm2 +=
            result.poseDelta[component] * result.poseDelta[component];
        rotationNorm2 += normalizedPoseDelta[component + 3] *
            normalizedPoseDelta[component + 3];
    }
    result.translationNorm = std::sqrt(translationNorm2);
    result.rotationNorm = std::sqrt(rotationNorm2);

    // Match Model::prepareCam: convert OpenGL C2W to OpenCV C2W by
    // negating the Y/Z camera axes, then invert it into the declared view.
    constexpr float axisFlip[3] = {1.0f, -1.0f, -1.0f};
    float declaredViewRotation[9] = {};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            declaredViewRotation[row * 3 + column] =
                baseCameraToWorld[column * 4 + row] * axisFlip[row];
        }
    }
    float declaredViewTranslation[3] = {};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            declaredViewTranslation[row] -=
                declaredViewRotation[row * 3 + column] *
                baseCameraToWorld[column * 4 + 3];
        }
    }

    const std::array<float, 9> correctionRotation =
        poseSo3Exp(normalizedPoseDelta + 3);
    float correctedViewRotation[9] = {};
    float correctedViewTranslation[3] = {};
    for (int row = 0; row < 3; ++row) {
        for (int inner = 0; inner < 3; ++inner) {
            correctedViewTranslation[row] +=
                correctionRotation[row * 3 + inner] *
                declaredViewTranslation[inner];
            for (int column = 0; column < 3; ++column) {
                correctedViewRotation[row * 3 + column] +=
                    correctionRotation[row * 3 + inner] *
                    declaredViewRotation[inner * 3 + column];
            }
        }
        correctedViewTranslation[row] += normalizedPoseDelta[row];
    }

    // Invert the corrected OpenCV view and flip its camera axes back to the
    // public OpenGL convention.
    float correctedNormalizedPosition[3] = {};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            const float inverseRotation =
                correctedViewRotation[column * 3 + row];
            result.correctedCameraToWorld[row * 4 + column] =
                inverseRotation * axisFlip[column];
            correctedNormalizedPosition[row] -=
                inverseRotation * correctedViewTranslation[column];
        }
        result.correctedCameraToWorld[row * 4 + 3] =
            correctedNormalizedPosition[row] / normalizationScale +
            normalizationTranslation[row];
    }
    result.correctedCameraToWorld[12] = 0.0f;
    result.correctedCameraToWorld[13] = 0.0f;
    result.correctedCameraToWorld[14] = 0.0f;
    result.correctedCameraToWorld[15] = 1.0f;
    return result;
}

inline uint32_t poseRefinementStateCount(
    bool enabled, int datasetCameraCount) {
    if (!enabled) return 0u;
    if (datasetCameraCount <= 0 ||
        static_cast<uint64_t>(datasetCameraCount) >
            std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(
            "Pose-refinement camera count is outside the supported range");
    }
    return static_cast<uint32_t>(datasetCameraCount);
}

} // namespace msplat::detail
