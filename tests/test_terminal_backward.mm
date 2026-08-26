#include "bindings.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

[[noreturn]] void fail(const char *expression, int line) {
    throw std::runtime_error(
        "line " + std::to_string(line) + ": " + expression);
}

#define CHECK(condition) \
    do { if (!(condition)) fail(#condition, __LINE__); } while (false)

constexpr int kWidth = 32;
constexpr int kHeight = 32;
constexpr int kGaussianCount = 2;
constexpr int kAdamGroups = 6;
constexpr float kBeta1 = 0.9f;
constexpr float kBeta2 = 0.999f;
constexpr float kEpsilon = 1.0e-8f;

MTensor gpuFloats(std::initializer_list<int64_t> shape,
                  std::initializer_list<float> values) {
    MTensor tensor = gpu_empty(std::vector<int64_t>(shape), DType::Float32);
    CHECK(static_cast<size_t>(tensor.numel()) == values.size());
    std::copy(values.begin(), values.end(), tensor.data<float>());
    return tensor;
}

MTensor gpuFloats(std::initializer_list<int64_t> shape,
                  const std::vector<float>& values) {
    MTensor tensor = gpu_empty(std::vector<int64_t>(shape), DType::Float32);
    CHECK(static_cast<size_t>(tensor.numel()) == values.size());
    std::copy(values.begin(), values.end(), tensor.data<float>());
    return tensor;
}

bool nearlyEqual(float lhs, float rhs, float tolerance = 2.0e-6f) {
    return std::abs(lhs - rhs) <= tolerance *
        std::max({1.0f, std::abs(lhs), std::abs(rhs)});
}

struct GroupState {
    std::vector<float> parameters;
    std::vector<float> firstMoments;
    std::vector<float> secondMoments;
};

std::array<float, 24> degreeFourBasis(float x, float y, float z) {
    constexpr float c1 = 0.4886025119029199f;
    constexpr std::array<float, 5> c2 = {
        1.0925484305920792f,
        -1.0925484305920792f,
        0.31539156525252005f,
        -1.0925484305920792f,
        0.5462742152960396f,
    };
    constexpr std::array<float, 7> c3 = {
        -0.5900435899266435f,
        2.890611442640554f,
        -0.4570457994644658f,
        0.3731763325901154f,
        -0.4570457994644658f,
        1.445305721320277f,
        -0.5900435899266435f,
    };
    constexpr std::array<float, 9> c4 = {
        2.5033429417967046f,
        -1.7701307697799304f,
        0.9461746957575601f,
        -0.6690465435572892f,
        0.10578554691520431f,
        -0.6690465435572892f,
        0.47308734787878004f,
        -1.7701307697799304f,
        0.6258357354491761f,
    };
    const float xx = x * x;
    const float xy = x * y;
    const float xz = x * z;
    const float yy = y * y;
    const float yz = y * z;
    const float zz = z * z;
    return {
        -c1 * y,
        c1 * z,
        -c1 * x,
        c2[0] * xy,
        c2[1] * yz,
        c2[2] * (2.0f * zz - xx - yy),
        c2[3] * xz,
        c2[4] * (xx - yy),
        c3[0] * y * (3.0f * xx - yy),
        c3[1] * xy * z,
        c3[2] * y * (4.0f * zz - xx - yy),
        c3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy),
        c3[4] * x * (4.0f * zz - xx - yy),
        c3[5] * z * (xx - yy),
        c3[6] * x * (xx - 3.0f * yy),
        c4[0] * xy * (xx - yy),
        c4[1] * yz * (3.0f * xx - yy),
        c4[2] * xy * (7.0f * zz - 1.0f),
        c4[3] * yz * (7.0f * zz - 3.0f),
        c4[4] * (zz * (35.0f * zz - 30.0f) + 3.0f),
        c4[5] * xz * (7.0f * zz - 3.0f),
        c4[6] * (xx - yy) * (7.0f * zz - 1.0f),
        c4[7] * xz * (xx - 3.0f * yy),
        c4[8] * (xx * (xx - 3.0f * yy) -
            yy * (3.0f * xx - yy)),
    };
}

struct TerminalStepGradients {
    std::array<float, 3> logScale;
    std::array<float, 4> quaternion;
};

TerminalStepGradients checkTerminalStep(
    bool collectStats,
    int degreesToUse,
    const std::array<float, 4>& visibleQuat = {1.0f, 0.0f, 0.0f, 0.0f},
    bool verifyCapturedGoldens = true,
    float globalScale = 1.0f) {
    CHECK(globalScale > 0.0f);
    const float logScale = std::log(0.03f / globalScale);
    constexpr float shC0 = 0.28209479177387814f;
    constexpr int shDegree = 4;
    constexpr int restBasisCount = 24;
    constexpr int restRowWidth = restBasisCount * 3;
    CHECK(degreesToUse >= 1 && degreesToUse <= shDegree);
    const int activeRestBasisCount =
        (degreesToUse + 1) * (degreesToUse + 1) - 1;
    const int activeRestWidth = activeRestBasisCount * 3;
    const float initialOpacity = std::log(0.1f / 0.9f);

    // Gaussian zero is visible. Gaussian one lies behind the camera: geometry
    // and opacity retain full-tensor Adam decay, while SH keeps its established
    // visibility gate.
    MTensor means = gpuFloats({kGaussianCount, 3}, {
        0.25f, -0.25f, -1.0f,
        0.1f, 0.2f, 1.0f,
    });
    MTensor scales = gpuFloats({kGaussianCount, 3}, {
        logScale, logScale - 0.1f, logScale + 0.1f,
        logScale + 0.2f, logScale - 0.2f, logScale,
    });
    MTensor quats = gpuFloats({kGaussianCount, 4}, {
        visibleQuat[0], visibleQuat[1], visibleQuat[2], visibleQuat[3],
        0.9f, 0.1f, -0.2f, 0.3f,
    });
    MTensor featuresDc = gpuFloats({kGaussianCount, 3}, {
        -0.2f / shC0, 0.1f / shC0, 0.3f / shC0,
        0.2f, -0.3f, 0.4f,
    });
    MTensor featuresRest = gpuFloats(
        {kGaussianCount, restBasisCount, 3},
        std::vector<float>(kGaussianCount * restRowWidth, 0.0f));
    MTensor opacities = gpuFloats({kGaussianCount, 1}, {
        initialOpacity, initialOpacity + 0.25f,
    });

    MTensor viewmat = gpuFloats({4, 4}, {
        1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1,
    });
    constexpr float nearPlane = 0.001f;
    constexpr float farPlane = 1000.0f;
    const float depthScale = (farPlane + nearPlane) / (farPlane - nearPlane);
    const float depthOffset = -farPlane * nearPlane / (farPlane - nearPlane);
    MTensor projmat = gpuFloats({4, 4}, {
        2, 0, 0, 0,
        0, -2, 0, 0,
        0, 0, -depthScale, depthOffset,
        0, 0, -1, 0,
    });

    MTensor background = gpu_zeros({3}, DType::Float32);
    std::vector<float> targetValues(
        static_cast<size_t>(kWidth) * kHeight * 3);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const size_t offset = static_cast<size_t>(y * kWidth + x) * 3;
            targetValues[offset] = 0.1f + 0.7f * x / (kWidth - 1);
            targetValues[offset + 1] = 0.15f + 0.6f * y / (kHeight - 1);
            targetValues[offset + 2] = ((x / 4 + y / 4) % 2 == 0)
                ? 0.25f
                : 0.75f;
        }
    }
    MTensor target = gpuFloats({kHeight, kWidth, 3}, targetValues);

    std::array<MTensor, kAdamGroups> params = {
        means, scales, quats, featuresDc, featuresRest, opacities,
    };
    std::array<MTensor, kAdamGroups> expAvg;
    std::array<MTensor, kAdamGroups> expAvgSq;
    const std::array<int, kAdamGroups> rowWidths = {
        3, 3, 4, 3, restRowWidth, 1,
    };
    std::array<GroupState, kAdamGroups> visibleBefore;
    std::array<GroupState, kAdamGroups> invisibleBefore;
    for (int group = 0; group < kAdamGroups; ++group) {
        expAvg[group] = gpu_zeros(params[group].shape(), DType::Float32);
        expAvgSq[group] = gpu_zeros(params[group].shape(), DType::Float32);
        const int offset = rowWidths[group];
        for (int element = 0; element < rowWidths[group]; ++element) {
            if (group == 4 && element >= activeRestWidth) {
                expAvg[group].data<float>()[element] =
                    0.03f + 0.0001f * element;
                expAvgSq[group].data<float>()[element] =
                    0.05f + 0.0002f * element;
            }
            visibleBefore[group].parameters.push_back(
                params[group].data<float>()[element]);
            visibleBefore[group].firstMoments.push_back(
                expAvg[group].data<float>()[element]);
            visibleBefore[group].secondMoments.push_back(
                expAvgSq[group].data<float>()[element]);
            const int index = offset + element;
            expAvg[group].data<float>()[index] =
                0.10f + 0.01f * group + 0.001f * element;
            expAvgSq[group].data<float>()[index] =
                0.20f + 0.02f * group + 0.002f * element;
            invisibleBefore[group].parameters.push_back(
                params[group].data<float>()[index]);
            invisibleBefore[group].firstMoments.push_back(
                expAvg[group].data<float>()[index]);
            invisibleBefore[group].secondMoments.push_back(
                expAvgSq[group].data<float>()[index]);
        }
    }

    float stepSizes[kAdamGroups];
    float biasCorrection2Sqrts[kAdamGroups];
    for (int group = 0; group < kAdamGroups; ++group) {
        stepSizes[group] = 0.01f + 0.002f * group;
        biasCorrection2Sqrts[group] = std::sqrt(0.001f);
    }

    MTensor logRgbGains = gpu_zeros({1, 3}, DType::Float32);
    MTensor logRgbGainsExpAvg = gpu_zeros({1, 3}, DType::Float32);
    MTensor logRgbGainsExpAvgSq = gpu_zeros({1, 3}, DType::Float32);
    MsplatPhotometricRefinementStep photometric;
    photometric.enabled = true;
    photometric.logRgbGains = &logRgbGains;
    photometric.expAvg = &logRgbGainsExpAvg;
    photometric.expAvgSq = &logRgbGainsExpAvgSq;
    photometric.adamStepSize = 0.005f;
    photometric.adamBiasCorrection2Sqrt = std::sqrt(0.001f);
    photometric.maxAbsLogGain = 1.38629436112f; // log(4)
    MTensor poseDeltas = gpu_zeros({1, 6}, DType::Float32);
    MsplatPoseRefinementStep pose;
    pose.deltas = &poseDeltas;

    constexpr float initialVisibleCount = 2.0f;
    constexpr float initialInvisibleCount = 7.0f;
    constexpr float initialVisibleGradientNorm = 0.5f;
    constexpr float initialInvisibleGradientNorm = 3.0f;
    constexpr float initialVisibleMaxSize = 0.01f;
    constexpr float initialInvisibleMaxSize = 0.9f;
    MTensor visibility = gpuFloats({kGaussianCount}, {
        initialVisibleCount, initialInvisibleCount,
    });
    MTensor xyGradientNorm = gpuFloats({kGaussianCount}, {
        initialVisibleGradientNorm, initialInvisibleGradientNorm,
    });
    MTensor max2DSize = gpuFloats({kGaussianCount}, {
        initialVisibleMaxSize, initialInvisibleMaxSize,
    });

    float cameraPosition[3] = {0, 0, 0};
    const uint64_t coverageUnits =
        static_cast<uint64_t>(kWidth) * kHeight * 255u;
    const float lossInvN = static_cast<float>(
        255.0 / (static_cast<double>(coverageUnits) * 3.0));

    MTensor radii = msplat_train_step(
        kGaussianCount, means, scales, globalScale,
        quats, viewmat, projmat,
        32.0f, 32.0f, 16.0f, 16.0f,
        kHeight, kWidth, std::make_tuple(2, 2, 1), 0.01f,
        shDegree, degreesToUse, cameraPosition,
        featuresDc, featuresRest, opacities, background,
        target, nullptr, nullptr, coverageUnits, 0.2f, lossInvN,
        false, 0.0f,
        kAdamGroups, params.data(), expAvg.data(), expAvgSq.data(),
        stepSizes, biasCorrection2Sqrts,
        kBeta1, kBeta2, kEpsilon,
        photometric, pose, collectStats,
        visibility, xyGradientNorm, max2DSize, 1.0f / kWidth);
    msplat_gpu_sync();

    constexpr std::array<float, 3> expectedPhotometricFirstMoments = {
        -7.47689e-6f,
        -1.47715e-5f,
        -1.52062e-5f,
    };
    for (int channel = 0; channel < 3; ++channel) {
        const float parameter = logRgbGains.data<float>()[channel];
        const float firstMoment =
            logRgbGainsExpAvg.data<float>()[channel];
        const float secondMoment =
            logRgbGainsExpAvgSq.data<float>()[channel];
        CHECK(std::isfinite(parameter));
        CHECK(std::isfinite(firstMoment));
        CHECK(std::isfinite(secondMoment));
        if (verifyCapturedGoldens) {
            CHECK(std::abs(firstMoment -
                expectedPhotometricFirstMoments[channel]) <= 1.0e-9f);
        }
        const float gradient = firstMoment / (1.0f - kBeta1);
        const float expectedSecondMoment =
            (1.0f - kBeta2) * gradient * gradient;
        const float expectedParameter = -photometric.adamStepSize *
            firstMoment /
            (std::sqrt(secondMoment) /
                photometric.adamBiasCorrection2Sqrt + kEpsilon);
        CHECK(std::abs(secondMoment - expectedSecondMoment) <= 1.0e-14f);
        CHECK(std::abs(parameter - expectedParameter) <= 1.0e-8f);
    }

    CHECK(radii.data<int32_t>()[0] > 0);
    CHECK(radii.data<int32_t>()[1] == 0);

    // Captured from the committed pre-fusion implementation with this exact
    // fixture. These values independently guard geometry/opacity VJP signs,
    // scales, and host bindings; the checks below separately validate Adam.
    auto checkVisibleGolden = [&params, &expAvg, &expAvgSq](
        int group,
        std::initializer_list<std::array<float, 3>> expected) {
        int element = 0;
        for (const auto& golden : expected) {
            const float parameter = params[group].data<float>()[element];
            const float firstMoment = expAvg[group].data<float>()[element];
            const float secondMoment = expAvgSq[group].data<float>()[element];
            CHECK(std::abs(parameter - golden[0]) <=
                2.0e-6f + 2.0e-6f * std::abs(golden[0]));
            CHECK(std::abs(firstMoment - golden[1]) <=
                2.0e-9f + 2.0e-4f * std::abs(golden[1]));
            CHECK(std::abs(secondMoment - golden[2]) <=
                2.0e-15f + 5.0e-4f * std::abs(golden[2]));
            ++element;
        }
    };
    if (verifyCapturedGoldens) {
        checkVisibleGolden(0, {
            {0.250999361f, -1.54147745e-6f, 2.37612122e-13f},
            {-0.250999331f, 1.49219454e-6f, 2.22661462e-13f},
            {-0.999000013f, -4.97649307e-5f, 2.47651538e-10f},
        });
        checkVisibleGolden(1, {
            {-3.50535798f, -2.32320835e-5f, 5.39722537e-11f},
            {-3.60535789f, -2.23245897e-5f, 4.98380608e-11f},
            {-3.40535831f, -3.44983891e-6f, 1.19012298e-12f},
        });
        checkVisibleGolden(2, {
            {1.0f, 0.0f, 0.0f},
            {0.00139972696f, -4.95558697e-6f, 2.45575153e-12f},
            {0.00139939517f, -2.27804594e-6f, 5.18942419e-13f},
            {0.00139855593f, -9.62098056e-7f, 9.2562019e-14f},
        });
        checkVisibleGolden(5, {
            {-2.19522476f, -3.3709206e-5f, 1.13629536e-10f},
        });
    }

    for (int group = 0; group < kAdamGroups; ++group) {
        bool visibleGradientIsNonzero = false;
        for (int element = 0; element < rowWidths[group]; ++element) {
            const float oldParameter =
                visibleBefore[group].parameters[element];
            const float oldFirstMoment =
                visibleBefore[group].firstMoments[element];
            const float oldSecondMoment =
                visibleBefore[group].secondMoments[element];
            const float firstMoment =
                expAvg[group].data<float>()[element];
            const float secondMoment =
                expAvgSq[group].data<float>()[element];
            CHECK(std::isfinite(params[group].data<float>()[element]));
            CHECK(std::isfinite(firstMoment));
            CHECK(std::isfinite(secondMoment));
            const bool dormantRest =
                group == 4 && element >= activeRestWidth;
            if (dormantRest) {
                CHECK(params[group].data<float>()[element] == oldParameter);
                CHECK(firstMoment == oldFirstMoment);
                CHECK(secondMoment == oldSecondMoment);
                continue;
            }
            const float gradient =
                (firstMoment - kBeta1 * oldFirstMoment) /
                (1.0f - kBeta1);
            const float expectedSecondMoment =
                kBeta2 * oldSecondMoment +
                (1.0f - kBeta2) * gradient * gradient;
            const float expectedParameter = oldParameter -
                stepSizes[group] * firstMoment /
                (std::sqrt(secondMoment) /
                    biasCorrection2Sqrts[group] + kEpsilon);
            CHECK(nearlyEqual(secondMoment, expectedSecondMoment, 1.0e-5f));
            CHECK(nearlyEqual(
                params[group].data<float>()[element], expectedParameter));
            visibleGradientIsNonzero |= std::abs(firstMoment) > 1.0e-8f;
        }
        if (!visibleGradientIsNonzero) {
            throw std::runtime_error(
                "visible gradient remained zero for Adam group " +
                std::to_string(group));
        }

        const int offset = rowWidths[group];
        const bool decaysOutsideView = group <= 2 || group == 5;
        for (int element = 0; element < rowWidths[group]; ++element) {
            const float oldParameter =
                invisibleBefore[group].parameters[element];
            const float oldFirstMoment =
                invisibleBefore[group].firstMoments[element];
            const float oldSecondMoment =
                invisibleBefore[group].secondMoments[element];
            const float expectedFirstMoment = decaysOutsideView
                ? kBeta1 * oldFirstMoment
                : oldFirstMoment;
            const float expectedSecondMoment = decaysOutsideView
                ? kBeta2 * oldSecondMoment
                : oldSecondMoment;
            const float expectedParameter = decaysOutsideView
                ? oldParameter - stepSizes[group] * expectedFirstMoment /
                    (std::sqrt(expectedSecondMoment) /
                        biasCorrection2Sqrts[group] + kEpsilon)
                : oldParameter;
            const int index = offset + element;
            CHECK(nearlyEqual(
                expAvg[group].data<float>()[index], expectedFirstMoment));
            CHECK(nearlyEqual(
                expAvgSq[group].data<float>()[index], expectedSecondMoment));
            CHECK(nearlyEqual(
                params[group].data<float>()[index], expectedParameter));
        }
    }

    // DC reveals the raster color cotangent, which lets the fixture check all
    // active basis gradients without exposing production intermediates. The
    // visible-state loop above checks dormant coefficients byte-for-byte.
    constexpr std::array<float, 3> visibleMean = {0.25f, -0.25f, -1.0f};
    const float meanNorm = std::sqrt(
        visibleMean[0] * visibleMean[0] +
        visibleMean[1] * visibleMean[1] +
        visibleMean[2] * visibleMean[2]);
    const auto basis = degreeFourBasis(
        visibleMean[0] / meanNorm,
        visibleMean[1] / meanNorm,
        visibleMean[2] / meanNorm);
    for (int channel = 0; channel < 3; ++channel) {
        const float dcFirstMoment = expAvg[3].data<float>()[channel];
        const float colorGradient =
            dcFirstMoment / ((1.0f - kBeta1) * shC0);
        for (int basisIndex = 0; basisIndex < restBasisCount; ++basisIndex) {
            const int index = basisIndex * 3 + channel;
            const float expectedFirstMoment = basisIndex < activeRestBasisCount
                ? (1.0f - kBeta1) * basis[basisIndex] * colorGradient
                : visibleBefore[4].firstMoments[index];
            CHECK(nearlyEqual(
                expAvg[4].data<float>()[index],
                expectedFirstMoment,
                1.0e-5f));
        }
    }

    if (collectStats) {
        CHECK(visibility.data<float>()[0] == initialVisibleCount + 1.0f);
        CHECK(visibility.data<float>()[1] == initialInvisibleCount);
        CHECK(std::isfinite(xyGradientNorm.data<float>()[0]));
        CHECK(xyGradientNorm.data<float>()[0] > initialVisibleGradientNorm);
        CHECK(xyGradientNorm.data<float>()[1] == initialInvisibleGradientNorm);
        const float expectedMaxSize = std::max(
            initialVisibleMaxSize,
            static_cast<float>(radii.data<int32_t>()[0]) / kWidth);
        CHECK(nearlyEqual(max2DSize.data<float>()[0], expectedMaxSize));
        CHECK(max2DSize.data<float>()[1] == initialInvisibleMaxSize);
    } else {
        CHECK(visibility.data<float>()[0] == initialVisibleCount);
        CHECK(visibility.data<float>()[1] == initialInvisibleCount);
        CHECK(xyGradientNorm.data<float>()[0] == initialVisibleGradientNorm);
        CHECK(xyGradientNorm.data<float>()[1] == initialInvisibleGradientNorm);
        CHECK(max2DSize.data<float>()[0] == initialVisibleMaxSize);
        CHECK(max2DSize.data<float>()[1] == initialInvisibleMaxSize);
    }

    TerminalStepGradients gradients;
    for (int element = 0; element < 3; ++element) {
        gradients.logScale[element] =
            expAvg[1].data<float>()[element] / (1.0f - kBeta1);
    }
    for (int element = 0; element < 4; ++element) {
        gradients.quaternion[element] =
            expAvg[2].data<float>()[element] / (1.0f - kBeta1);
    }
    return gradients;
}

void checkMinimumFootprintAndScaleGradient() {
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr float focalLength = 32.0f;
    constexpr float principalPoint = 16.5f;
    constexpr float physicalScale = 0.01f;
    constexpr float finiteDifferenceStep = 0.01f;
    constexpr int adamGroups = 6;
    const float baseLogScale = std::log(physicalScale);

    MTensor means = gpuFloats({1, 3}, {0.0f, 0.0f, -1.0f});
    MTensor scales = gpuFloats({1, 3}, {
        baseLogScale, baseLogScale, baseLogScale,
    });
    MTensor quats = gpuFloats({1, 4}, {1.0f, 0.0f, 0.0f, 0.0f});
    MTensor featuresDc = gpuFloats({1, 3}, {0.0f, 0.0f, 0.0f});
    MTensor featuresRest = gpu_zeros({1, 3, 3}, DType::Float32);
    MTensor opacities = gpuFloats({1, 1}, {0.0f});
    MTensor background = gpu_zeros({3}, DType::Float32);
    MTensor viewmat = gpuFloats({4, 4}, {
        1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1,
    });
    constexpr float nearPlane = 0.001f;
    constexpr float farPlane = 1000.0f;
    const float depthScale = (farPlane + nearPlane) / (farPlane - nearPlane);
    const float depthOffset = -farPlane * nearPlane / (farPlane - nearPlane);
    MTensor projmat = gpuFloats({4, 4}, {
        2, 0, 0, 0,
        0, -2, 0, 0,
        0, 0, -depthScale, depthOffset,
        0, 0, -1, 0,
    });
    float cameraPosition[3] = {0.0f, 0.0f, 0.0f};

    struct RenderEvaluation {
        double objective;
        int support;
        float center;
        float neighbor;
    };
    auto evaluate = [&](float xLogScale) {
        scales.data<float>()[0] = xLogScale;
        MTensor rendered = msplat_render(
            1, means, scales, 1.0f, quats, viewmat, projmat,
            focalLength, focalLength, principalPoint, principalPoint,
            height, width, std::make_tuple(2, 2, 1), 0.01f,
            1, 1, cameraPosition, featuresDc, featuresRest,
            opacities, background);
        msplat_gpu_sync();

        RenderEvaluation result = {0.0, 0, 0.0f, 0.0f};
        const float* pixels = rendered.data<float>();
        for (int pixel = 0; pixel < width * height; ++pixel) {
            result.support += pixels[pixel * 3] > 0.0f ? 1 : 0;
            for (int channel = 0; channel < 3; ++channel) {
                result.objective += std::abs(
                    static_cast<double>(pixels[pixel * 3 + channel]));
            }
        }
        result.objective /= static_cast<double>(width * height * 3);
        result.center = pixels[(16 * width + 16) * 3];
        result.neighbor = pixels[(16 * width + 17) * 3];
        return result;
    };

    const RenderEvaluation base = evaluate(baseLogScale);
    CHECK(std::abs(base.center - 0.25f) <= 2.0e-6f);
    CHECK(base.neighbor > 0.0f && base.neighbor < base.center);
    const double observedVariance =
        -0.5 / std::log(static_cast<double>(base.neighbor / base.center));
    const double expectedVariance =
        focalLength * focalLength * physicalScale * physicalScale + 0.3;
    CHECK(std::abs(observedVariance - expectedVariance) <= 5.0e-4);

    const RenderEvaluation before =
        evaluate(baseLogScale - finiteDifferenceStep);
    const RenderEvaluation after =
        evaluate(baseLogScale + finiteDifferenceStep);
    CHECK(before.support == after.support);
    CHECK(before.support > 0);
    const double finiteDifference =
        (after.objective - before.objective) /
        (2.0 * finiteDifferenceStep);

    scales.data<float>()[0] = baseLogScale;
    std::array<MTensor, adamGroups> params = {
        means, scales, quats, featuresDc, featuresRest, opacities,
    };
    std::array<MTensor, adamGroups> expAvg;
    std::array<MTensor, adamGroups> expAvgSq;
    for (int group = 0; group < adamGroups; ++group) {
        expAvg[group] = gpu_zeros(params[group].shape(), DType::Float32);
        expAvgSq[group] = gpu_zeros(params[group].shape(), DType::Float32);
    }
    float stepSizes[adamGroups] = {};
    float biasCorrection2Sqrts[adamGroups];
    std::fill(
        std::begin(biasCorrection2Sqrts),
        std::end(biasCorrection2Sqrts), std::sqrt(1.0f - kBeta2));

    MTensor target = gpu_zeros({height, width, 3}, DType::Float32);
    MTensor logRgbGains = gpu_zeros({1, 3}, DType::Float32);
    MsplatPhotometricRefinementStep photometric;
    photometric.logRgbGains = &logRgbGains;
    MTensor poseDeltas = gpu_zeros({1, 6}, DType::Float32);
    MsplatPoseRefinementStep pose;
    pose.deltas = &poseDeltas;
    MTensor visibility = gpu_zeros({1}, DType::Float32);
    MTensor xyGradientNorm = gpu_zeros({1}, DType::Float32);
    MTensor max2DSize = gpu_zeros({1}, DType::Float32);
    const uint64_t coverageUnits =
        static_cast<uint64_t>(width) * height * 255u;
    const float lossInvN = static_cast<float>(
        255.0 / (static_cast<double>(coverageUnits) * 3.0));

    MTensor radii = msplat_train_step(
        1, means, scales, 1.0f, quats, viewmat, projmat,
        focalLength, focalLength, principalPoint, principalPoint,
        height, width, std::make_tuple(2, 2, 1), 0.01f,
        1, 1, cameraPosition, featuresDc, featuresRest,
        opacities, background, target, nullptr, nullptr, coverageUnits,
        0.0f, lossInvN, false, 0.0f,
        adamGroups, params.data(), expAvg.data(), expAvgSq.data(),
        stepSizes, biasCorrection2Sqrts, kBeta1, kBeta2, kEpsilon,
        photometric, pose, false,
        visibility, xyGradientNorm, max2DSize, 1.0f / width);
    msplat_gpu_sync();

    CHECK(radii.data<int32_t>()[0] > 0);
    const double analytic =
        expAvg[1].data<float>()[0] / (1.0f - kBeta1);
    CHECK(std::abs(analytic) > 1.0e-7);
    CHECK(std::abs(analytic - finiteDifference) <=
        std::max(3.0e-6, 0.03 * std::abs(finiteDifference)));
}

}  // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc != 2)
                throw std::invalid_argument("Expected the metallib path");
            msplat_set_metallib_path_checked(argv[1]);
            checkMinimumFootprintAndScaleGradient();
            checkTerminalStep(false, 1);
            checkTerminalStep(true, 4);

            // Rotation is invariant under positive quaternion scaling. Its raw
            // gradient must be tangent and transform inversely with that scale.
            constexpr std::array<float, 4> quaternion = {
                0.8f, 0.2f, -0.4f, 0.4f,
            };
            constexpr float scale = 3.0f;
            std::array<float, 4> scaledQuaternion;
            for (int element = 0; element < 4; ++element)
                scaledQuaternion[element] = scale * quaternion[element];
            const auto unitQuaternionStep =
                checkTerminalStep(false, 1, quaternion, false);
            const auto gradient = unitQuaternionStep.quaternion;
            const auto scaledGradient =
                checkTerminalStep(false, 1, scaledQuaternion, false).quaternion;

            float gradientNormSquared = 0.0f;
            float scaleErrorSquared = 0.0f;
            float radialGradient = 0.0f;
            float scaledRadialGradient = 0.0f;
            for (int element = 0; element < 4; ++element) {
                gradientNormSquared += gradient[element] * gradient[element];
                const float scaleError =
                    gradient[element] - scale * scaledGradient[element];
                scaleErrorSquared += scaleError * scaleError;
                radialGradient += quaternion[element] * gradient[element];
                scaledRadialGradient +=
                    scaledQuaternion[element] * scaledGradient[element];
            }
            const float gradientNorm = std::sqrt(gradientNormSquared);
            CHECK(gradientNorm > 1.0e-8f);
            CHECK(std::sqrt(scaleErrorSquared) <= 5.0e-3f * gradientNorm);
            CHECK(std::abs(radialGradient) <= 5.0e-3f * gradientNorm);
            CHECK(std::abs(scaledRadialGradient) <=
                5.0e-3f * scale * gradientNorm);

            // Compensating log-scales for a global scale preserves the exact
            // forward covariance, so its log-scale gradient must also remain
            // unchanged. This exercises the active fused Metal backward path.
            constexpr float compensatedGlobalScale = 2.0f;
            const auto compensatedScale = checkTerminalStep(
                false, 1, quaternion, false, compensatedGlobalScale).logScale;
            for (int element = 0; element < 3; ++element) {
                CHECK(std::abs(unitQuaternionStep.logScale[element]) > 1.0e-6f);
                CHECK(std::abs(
                    unitQuaternionStep.logScale[element] -
                    compensatedScale[element]) <=
                    5.0e-3f * std::abs(unitQuaternionStep.logScale[element]));
            }
            cleanup_msplat_metal();
            return 0;
        } catch (const std::exception &error) {
            if (std::string(error.what()) ==
                "msplat: no Metal device is available") {
                std::cerr << "SKIP: " << error.what() << '\n';
                return 77;
            }
            std::cerr << error.what() << '\n';
            cleanup_msplat_metal();
            return 1;
        }
    }
}
