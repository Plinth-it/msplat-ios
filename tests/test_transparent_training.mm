#include "bindings.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
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
constexpr int kAdamGroups = 6;

struct StepResult {
    std::array<double, kAdamGroups> firstMomentL1 = {};
    std::array<int, kAdamGroups> nonfiniteFirstMomentCount = {};
    std::vector<double> colorFirstMomentL1;
    std::vector<std::array<float, 3>> colorFirstMoments;
    std::vector<float> opacityFirstMoments;
    float opacity = 0.0f;
    float opacityFirstMoment = 0.0f;
    float lastOpacity = 0.0f;
    float lastOpacityFirstMoment = 0.0f;
    float minimumOpacity = 0.0f;
    float maximumOpacity = 0.0f;
    int positiveOpacityMomentCount = 0;
    int nonfiniteOpacityCount = 0;
    int nonfiniteOpacityMomentCount = 0;
    int radius = 0;
    float visibilityCount = 0.0f;
    float lastVisibilityCount = 0.0f;
    uint32_t overflowReasons = MSPLAT_TRAINING_OVERFLOW_NONE;
    uint64_t retainedIntersectionCount = 0;
    uint64_t intersectionCapacity = 0;
    double intersectionArenaGrowMs = 0.0;
    double countGpuMs = 0.0;
    double countWaitWallMs = 0.0;
    double queueIdleMs = 0.0;
    bool queueIdleTimeValid = false;
    uint32_t commandBufferCount = 0;
    uint32_t maximumTileCount = 0;
    uint32_t activeTileCount = 0;
    uint32_t trivialTileCount = 0;
    uint32_t smallTileCount = 0;
    uint32_t mediumTileCount = 0;
    uint32_t largeTileCount = 0;
    uint64_t overflowedStepCount = 0;
    uint64_t tileCapOverflowedStepCount = 0;
    uint64_t packedCapacityOverflowedStepCount = 0;
    int64_t lastOverflowIteration = 0;
};

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

StepResult runStep(bool transparent, float alphaLossWeight,
                   int gaussianCount = 1,
                   float initialAlpha = 0.1f,
                   bool packedMask = false,
                   bool useCoverageRenderTiles = false,
                   bool allRenderTilesActive = false,
                   bool collectStats = false,
                   bool collectTelemetry = false,
                   int width = kWidth,
                   int height = kHeight,
                   bool exerciseAppearance = false,
                   MsplatTrainingTelemetryHandle telemetry = {},
                   int64_t iteration = 1,
                   float gaussianScale = 0.03f,
                   const std::vector<float>& alphaOverrides = {},
                   const std::vector<float>& depthOverrides = {},
                   std::array<float, 3> appearanceColors = {
                       0.4f, 0.6f, 0.8f}) {
    CHECK(gaussianCount > 0);
    CHECK(gaussianScale > 0.0f);
    CHECK(alphaOverrides.empty() ||
          alphaOverrides.size() == static_cast<size_t>(gaussianCount));
    CHECK(depthOverrides.empty() ||
          depthOverrides.size() == static_cast<size_t>(gaussianCount));
    std::vector<float> meanValues(static_cast<size_t>(gaussianCount) * 3);
    std::vector<float> scaleValues(static_cast<size_t>(gaussianCount) * 3);
    std::vector<float> quatValues(static_cast<size_t>(gaussianCount) * 4);
    std::vector<float> dcValues(static_cast<size_t>(gaussianCount) * 3);
    std::vector<float> opacityValues(static_cast<size_t>(gaussianCount));

    const float logScale = std::log(gaussianScale);
    constexpr float shC0 = 0.28209479177387814f;
    const float blackDc = -0.5f / shC0;
    const std::array<float, 3> appearanceDc = {
        (appearanceColors[0] - 0.5f) / shC0,
        (appearanceColors[1] - 0.5f) / shC0,
        (appearanceColors[2] - 0.5f) / shC0,
    };
    for (int index = 0; index < gaussianCount; ++index) {
        const size_t meanOffset = static_cast<size_t>(index) * 3;
        meanValues[meanOffset + 0] = 0.25f;
        meanValues[meanOffset + 1] = -0.25f;
        meanValues[meanOffset + 2] = depthOverrides.empty()
            ? -1.0f
            : depthOverrides[static_cast<size_t>(index)];
        scaleValues[meanOffset + 0] = logScale;
        scaleValues[meanOffset + 1] = logScale;
        scaleValues[meanOffset + 2] = logScale;
        for (size_t channel = 0; channel < appearanceDc.size(); ++channel) {
            dcValues[meanOffset + channel] = exerciseAppearance
                ? appearanceDc[channel]
                : blackDc;
        }
        quatValues[static_cast<size_t>(index) * 4] = 1.0f;
        const float alpha = alphaOverrides.empty()
            ? initialAlpha
            : alphaOverrides[static_cast<size_t>(index)];
        CHECK(alpha > 0.0f && alpha < 1.0f);
        opacityValues[static_cast<size_t>(index)] =
            std::log(alpha / (1.0f - alpha));
    }

    MTensor means = gpuFloats({gaussianCount, 3}, meanValues);
    MTensor scales = gpuFloats({gaussianCount, 3}, scaleValues);
    MTensor quats = gpuFloats({gaussianCount, 4}, quatValues);

    // Identity camera-to-world becomes diag(1,-1,-1,1) in the renderer's
    // OpenGL camera convention. P*V below projects the Gaussian to (23.5,23.5).
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

    MTensor featuresDc = gpuFloats({gaussianCount, 3}, dcValues);
    MTensor featuresRest = gpu_zeros(
        {gaussianCount, 0, 3}, DType::Float32);
    MTensor opacities = gpuFloats({gaussianCount, 1}, opacityValues);
    MTensor background = gpu_zeros({3}, DType::Float32);
    MTensor gt = gpu_empty({height, width, 4}, DType::UInt8);
    auto *targetBytes = gt.data<uint8_t>();
    for (int pixel = 0; pixel < width * height; ++pixel) {
        const int offset = pixel * 4;
        // Deliberately asymmetric RGB exercises byte normalization in every
        // loss pass. The standalone-mask case gives alpha unrelated padding;
        // the packed case below replaces it with coverage.
        targetBytes[offset + 0] = 23;
        targetBytes[offset + 1] = 91;
        targetBytes[offset + 2] = 207;
        targetBytes[offset + 3] = 37;
    }
    MTensor mask = gpu_zeros({height, width}, DType::UInt8);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x)
            mask.data<uint8_t>()[y * width + x] = 255;
    }
    if (packedMask) {
        for (int pixel = 0; pixel < width * height; ++pixel) {
            targetBytes[pixel * 4 + 3] = mask.data<uint8_t>()[pixel];
        }
    }
    MTensor coverageRenderTiles;
    const MTensor *coverageRenderTilePointer = nullptr;
    if (useCoverageRenderTiles) {
        coverageRenderTiles = gpu_empty({2, 2}, DType::UInt8);
        std::fill_n(coverageRenderTiles.data<uint8_t>(), 4,
                    allRenderTilesActive ? uint8_t{1} : uint8_t{0});
        if (!allRenderTilesActive)
            coverageRenderTiles.data<uint8_t>()[0] = 1;
        coverageRenderTilePointer = &coverageRenderTiles;
    }

    std::array<MTensor, kAdamGroups> params = {
        means, scales, quats, featuresDc, featuresRest, opacities,
    };
    std::array<MTensor, kAdamGroups> expAvg;
    std::array<MTensor, kAdamGroups> expAvgSq;
    for (int group = 0; group < kAdamGroups; ++group) {
        expAvg[group] = gpu_zeros(params[group].shape(), DType::Float32);
        expAvgSq[group] = gpu_zeros(params[group].shape(), DType::Float32);
    }
    float stepSizes[kAdamGroups] = {};
    stepSizes[3] = 1.0e-4f;
    stepSizes[5] = 0.5f;
    float biasCorrection2Sqrts[kAdamGroups];
    std::fill_n(biasCorrection2Sqrts, kAdamGroups, std::sqrt(0.001f));

    MTensor logRgbGains = gpu_zeros({1, 3}, DType::Float32);
    MsplatPhotometricRefinementStep photometric;
    photometric.logRgbGains = &logRgbGains;
    MsplatPpispRefinementStep ppisp;
    MTensor poseDeltas = gpu_zeros({1, 6}, DType::Float32);
    MsplatPoseRefinementStep pose;
    pose.deltas = &poseDeltas;

    MTensor visibility = gpu_zeros({gaussianCount}, DType::Float32);
    MTensor xyGradientNorm = gpu_zeros({gaussianCount}, DType::Float32);
    MTensor max2DSize = gpu_zeros({gaussianCount}, DType::Float32);
    float cameraPosition[3] = {0, 0, 0};
    const uint64_t coverageUnits = transparent
        ? static_cast<uint64_t>(width) * height * 255u
        : 4u * 4u * 255u;
    const float lossInvN = static_cast<float>(
        255.0 / (static_cast<double>(coverageUnits) * 3.0));

    MsplatLogicalTrainingStepHandle logicalStep;
    if (collectTelemetry) {
        if (!telemetry)
            telemetry = msplat_training_telemetry_create();
        logicalStep = msplat_training_step_begin(telemetry, iteration);
        msplat_training_step_mark_cpu_start(logicalStep);
    }

    MTensor radii;
    try {
        radii = msplat_train_step(
            gaussianCount, means, scales, 1.0f,
            quats, viewmat, projmat,
            float(width), float(height),
            0.5f * float(width), 0.5f * float(height),
            height, width,
            std::make_tuple((width + 15) / 16, (height + 15) / 16, 1),
            0.01f,
            0, 0, cameraPosition,
            featuresDc, featuresRest, opacities, background,
            gt, packedMask ? &gt : &mask, coverageRenderTilePointer,
            coverageUnits, 0.2f, lossInvN,
            transparent, alphaLossWeight,
            kAdamGroups, params.data(), expAvg.data(), expAvgSq.data(),
            stepSizes, biasCorrection2Sqrts,
            0.9f, 0.999f, 1.0e-8f,
            photometric, ppisp, pose, collectStats,
            visibility, xyGradientNorm, max2DSize, 1.0f / float(width));
        if (logicalStep) {
            MsplatTrainingStepDescriptor descriptor;
            descriptor.iteration = iteration;
            descriptor.splatCount = gaussianCount;
            descriptor.modelCapacity = gaussianCount;
            descriptor.effectiveWidth = width;
            descriptor.effectiveHeight = height;
            descriptor.activeShDegree = 0;
            msplat_training_step_submit(logicalStep, descriptor);
        }
        msplat_gpu_sync();
    } catch (...) {
        if (logicalStep) msplat_training_step_abort(logicalStep);
        throw;
    }

    StepResult result;
    result.colorFirstMomentL1.resize(static_cast<size_t>(gaussianCount));
    result.colorFirstMoments.resize(static_cast<size_t>(gaussianCount));
    result.opacityFirstMoments.resize(static_cast<size_t>(gaussianCount));
    const float* colorMoments = expAvg[3].data<float>();
    const float* opacityMoments = expAvg[5].data<float>();
    for (int index = 0; index < gaussianCount; ++index) {
        const size_t colorOffset = static_cast<size_t>(index) * 3;
        for (size_t channel = 0; channel < 3; ++channel) {
            result.colorFirstMoments[static_cast<size_t>(index)][channel] =
                colorMoments[colorOffset + channel];
            result.colorFirstMomentL1[static_cast<size_t>(index)] +=
                std::abs(colorMoments[colorOffset + channel]);
        }
        result.opacityFirstMoments[static_cast<size_t>(index)] =
            opacityMoments[index];
    }
    result.opacity = opacities.data<float>()[0];
    result.opacityFirstMoment = expAvg[5].data<float>()[0];
    result.lastOpacity = opacities.data<float>()[gaussianCount - 1];
    result.lastOpacityFirstMoment =
        expAvg[5].data<float>()[gaussianCount - 1];
    result.minimumOpacity = result.opacity;
    result.maximumOpacity = result.opacity;
    for (int index = 0; index < gaussianCount; ++index) {
        const float opacity = opacities.data<float>()[index];
        const float moment = expAvg[5].data<float>()[index];
        if (!std::isfinite(opacity)) ++result.nonfiniteOpacityCount;
        if (!std::isfinite(moment)) ++result.nonfiniteOpacityMomentCount;
        result.minimumOpacity = std::min(result.minimumOpacity, opacity);
        result.maximumOpacity = std::max(result.maximumOpacity, opacity);
        if (moment > 0.0f)
            ++result.positiveOpacityMomentCount;
    }
    result.radius = radii.data<int32_t>()[0];
    result.visibilityCount = visibility.data<float>()[0];
    result.lastVisibilityCount =
        visibility.data<float>()[gaussianCount - 1];
    for (int group = 0; group < kAdamGroups; ++group) {
        const float* moments = expAvg[group].data<float>();
        for (int64_t index = 0; index < expAvg[group].numel(); ++index) {
            if (std::isfinite(moments[index])) {
                result.firstMomentL1[group] += std::abs(moments[index]);
            } else {
                ++result.nonfiniteFirstMomentCount[group];
            }
        }
    }
    if (telemetry) {
        const MsplatTrainingTelemetrySnapshot snapshot =
            msplat_training_telemetry_snapshot(telemetry);
        CHECK((snapshot.flags &
               MSPLAT_TRAINING_TELEMETRY_HAS_COMPLETED) != 0);
        result.overflowReasons = snapshot.completedStep.overflowReasons;
        result.retainedIntersectionCount =
            snapshot.completedStep.retainedPackedIntersections;
        result.intersectionCapacity =
            snapshot.completedStep.packedIntersectionCapacity;
        result.intersectionArenaGrowMs =
            snapshot.completedStep.intersectionArenaGrowMs;
        result.countGpuMs = snapshot.completedStep.countGpuMs;
        result.countWaitWallMs = snapshot.completedStep.countWaitWallMs;
        result.queueIdleMs = snapshot.completedStep.queueIdleMs;
        result.queueIdleTimeValid =
            (snapshot.flags &
             MSPLAT_TRAINING_TELEMETRY_QUEUE_IDLE_TIMING_VALID) != 0;
        result.commandBufferCount =
            snapshot.completedStep.commandBufferCount;
        result.maximumTileCount = snapshot.completedStep.maximumTileCount;
        result.activeTileCount = snapshot.completedStep.activeTileCount;
        result.trivialTileCount = snapshot.completedStep.trivialTileCount;
        result.smallTileCount = snapshot.completedStep.smallTileCount;
        result.mediumTileCount = snapshot.completedStep.mediumTileCount;
        result.largeTileCount = snapshot.completedStep.largeTileCount;
        result.overflowedStepCount = snapshot.overflowedStepCount;
        result.tileCapOverflowedStepCount =
            snapshot.tileCapOverflowedStepCount;
        result.packedCapacityOverflowedStepCount =
            snapshot.packedCapacityOverflowedStepCount;
        result.lastOverflowIteration = snapshot.lastOverflowIteration;
    }
    return result;
}

void checkTransparentAlphaSupervision() {
    const float initialOpacity = std::log(0.1f / 0.9f);

    const StepResult coverage = runStep(false, 0.1f);
    CHECK(coverage.radius > 0);
    CHECK(coverage.opacity == initialOpacity);
    CHECK(coverage.opacityFirstMoment == 0.0f);

    const StepResult disabledAlpha = runStep(true, 0.0f);
    CHECK(disabledAlpha.radius > 0);
    CHECK(disabledAlpha.opacity == initialOpacity);
    CHECK(disabledAlpha.opacityFirstMoment == 0.0f);

    const StepResult transparent = runStep(true, 0.1f);
    CHECK(transparent.radius > 0);
    CHECK(transparent.opacity < initialOpacity);
    CHECK(transparent.opacityFirstMoment > 0.0f);

    const StepResult packedCoverage =
        runStep(false, 0.1f, 1, 0.1f, true);
    CHECK(packedCoverage.opacity == coverage.opacity);
    CHECK(packedCoverage.opacityFirstMoment == coverage.opacityFirstMoment);

    // The Gaussian projects wholly outside the mask's five-pixel SSIM halo.
    // Intersection pruning must retain its projection radius while producing
    // the same zero-gradient coverage update as the dense path.
    const StepResult prunedCoverage =
        runStep(false, 0.1f, 1, 0.1f, false, true);
    CHECK(prunedCoverage.radius == coverage.radius);
    CHECK(prunedCoverage.opacity == coverage.opacity);
    CHECK(prunedCoverage.opacityFirstMoment ==
          coverage.opacityFirstMoment);
    const StepResult packedPrunedCoverage =
        runStep(false, 0.1f, 1, 0.1f, true, true);
    CHECK(packedPrunedCoverage.radius == packedCoverage.radius);
    CHECK(packedPrunedCoverage.opacity == packedCoverage.opacity);
    CHECK(packedPrunedCoverage.opacityFirstMoment ==
          packedCoverage.opacityFirstMoment);
    const StepResult fullTileCoverage =
        runStep(false, 0.1f, 1, 0.1f, false, true, true);
    CHECK(fullTileCoverage.opacity == coverage.opacity);
    CHECK(fullTileCoverage.opacityFirstMoment ==
          coverage.opacityFirstMoment);
    const StepResult packedFullTileCoverage =
        runStep(false, 0.1f, 1, 0.1f, true, true, true);
    CHECK(packedFullTileCoverage.opacity == packedCoverage.opacity);
    CHECK(packedFullTileCoverage.opacityFirstMoment ==
          packedCoverage.opacityFirstMoment);

    const StepResult packedTransparent =
        runStep(true, 0.1f, 1, 0.1f, true);
    CHECK(packedTransparent.opacity == transparent.opacity);
    CHECK(packedTransparent.opacityFirstMoment ==
          transparent.opacityFirstMoment);
    const StepResult transparentWithCoverageTiles =
        runStep(true, 0.1f, 1, 0.1f, true, true);
    CHECK(transparentWithCoverageTiles.opacity == transparent.opacity);
    CHECK(transparentWithCoverageTiles.opacityFirstMoment ==
          transparent.opacityFirstMoment);
}

void checkChunkedTransparentAlphaSupervision() {
    // More than 512 identical tile intersections forces both chunked raster
    // passes. Keep per-Gaussian opacity just above the 1/255 raster cutoff and
    // aggregate transmittance high enough that every chunk contributes.
    constexpr int gaussianCount = 513;
    constexpr float initialAlpha = 0.005f;
    const float initialOpacity =
        std::log(initialAlpha / (1.0f - initialAlpha));

    const StepResult transparent =
        runStep(true, 0.1f, gaussianCount, initialAlpha,
                false, true, false, false, false,
                kWidth, kHeight, true);
    if (!(transparent.opacity < initialOpacity &&
          transparent.lastOpacity < initialOpacity &&
          transparent.opacityFirstMoment > 0.0f &&
          transparent.lastOpacityFirstMoment > 0.0f &&
          transparent.positiveOpacityMomentCount == gaussianCount &&
          transparent.nonfiniteOpacityCount == 0 &&
          transparent.nonfiniteOpacityMomentCount == 0)) {
        std::cerr << "chunked initial=" << initialOpacity
                  << " first=" << transparent.opacity
                  << " last=" << transparent.lastOpacity
                  << " firstMoment=" << transparent.opacityFirstMoment
                  << " lastMoment=" << transparent.lastOpacityFirstMoment
                  << " min=" << transparent.minimumOpacity
                  << " max=" << transparent.maximumOpacity
                  << " positiveMoments="
                  << transparent.positiveOpacityMomentCount
                  << " nonfiniteOpacity="
                  << transparent.nonfiniteOpacityCount
                  << " nonfiniteMoments="
                  << transparent.nonfiniteOpacityMomentCount
                  << '\n';
    }
    CHECK(transparent.radius > 0);
    CHECK(transparent.opacity < initialOpacity);
    CHECK(transparent.lastOpacity < initialOpacity);
    CHECK(transparent.opacityFirstMoment > 0.0f);
    CHECK(transparent.lastOpacityFirstMoment > 0.0f);
    CHECK(transparent.positiveOpacityMomentCount == gaussianCount);
    CHECK(transparent.nonfiniteOpacityCount == 0);
    CHECK(transparent.nonfiniteOpacityMomentCount == 0);
    CHECK(transparent.nonfiniteFirstMomentCount[0] == 0);
    CHECK(transparent.nonfiniteFirstMomentCount[1] == 0);
    CHECK(transparent.nonfiniteFirstMomentCount[2] == 0);
    CHECK(transparent.nonfiniteFirstMomentCount[3] == 0);
    CHECK(transparent.firstMomentL1[0] > 0.0);
    CHECK(transparent.firstMomentL1[1] > 0.0);
    CHECK(transparent.firstMomentL1[3] > 0.0);

    const char* attributes =
        std::getenv("MSPLAT_INTERSECTION_ATTRIBUTES");
    const bool gather = !attributes || std::string(attributes) != "packed";
    const size_t packedAttributeBytes =
        msplat_packed_intersection_attribute_bytes();
    CHECK(gather ? packedAttributeBytes == 0 : packedAttributeBytes > 0);
}

void checkCappedAlphaBackward() {
    // A very large Gaussian keeps sigma close enough to zero across the image
    // that opacity * exp(-sigma) always reaches the forward 0.999 alpha cap.
    // The capped contributor still owns an RGB gradient, while the clamp makes
    // its conic, position, scale, quaternion, and opacity gradients zero.
    constexpr float cappedAlpha = 0.9999f;
    constexpr float fullFrameScale = 100.0f;

    const auto runCapped = [](int gaussianCount) {
        return runStep(
            true, 0.1f, gaussianCount, cappedAlpha,
            false, false, false, false, false,
            kWidth, kHeight, true, {}, 1, fullFrameScale);
    };

    const StepResult monolithic = runCapped(1);
    CHECK(monolithic.radius > 0);
    CHECK(monolithic.firstMomentL1[3] > 0.0);
    CHECK(monolithic.firstMomentL1[0] == 0.0);
    CHECK(monolithic.firstMomentL1[1] == 0.0);
    CHECK(monolithic.firstMomentL1[2] == 0.0);
    CHECK(monolithic.firstMomentL1[5] == 0.0);

    // 513 intersections in each covered tile require two 512-entry depth
    // chunks. This exercises the same cap behavior in the chunked backward
    // kernel without relying on a source-only assertion.
    const StepResult chunked = runCapped(513);
    CHECK(chunked.radius > 0);
    CHECK(chunked.firstMomentL1[3] > 0.0);
    CHECK(chunked.firstMomentL1[0] == 0.0);
    CHECK(chunked.firstMomentL1[1] == 0.0);
    CHECK(chunked.firstMomentL1[2] == 0.0);
    CHECK(chunked.firstMomentL1[5] == 0.0);

    // Put an uncapped near Gaussian in front of a capped far Gaussian. The
    // backward walk encounters the far Gaussian first and must divide T by
    // (1 - 0.999) before it reaches the near one. Compare against a far alpha
    // just below the cap; the near RGB gradient should remain the same order
    // of magnitude rather than being suppressed by roughly 1000x.
    const auto runLayered = [](int gaussianCount, float farAlpha) {
        std::vector<float> alphas(
            static_cast<size_t>(gaussianCount), 0.005f);
        std::vector<float> depths(
            static_cast<size_t>(gaussianCount), -2.0f);
        alphas[0] = 0.5f;
        alphas[1] = farAlpha;
        depths[0] = -0.8f;
        depths[1] = -1.2f;
        return runStep(
            true, 0.1f, gaussianCount, 0.1f,
            false, false, false, false, false,
            kWidth, kHeight, true, {}, 1, fullFrameScale,
            alphas, depths);
    };
    const auto checkLayered = [&](int gaussianCount) {
        const StepResult capped = runLayered(gaussianCount, cappedAlpha);
        const StepResult uncapped = runLayered(gaussianCount, 0.998f);
        const double nearCapped = capped.colorFirstMomentL1[0];
        const double nearUncapped = uncapped.colorFirstMomentL1[0];
        CHECK(nearUncapped > 0.0);
        CHECK(nearCapped > 0.5 * nearUncapped);
        CHECK(nearCapped < 2.0 * nearUncapped);
        CHECK(capped.colorFirstMomentL1[1] > 0.0);
        CHECK(capped.opacityFirstMoments[1] == 0.0f);
    };

    checkLayered(2);
    checkLayered(513);
}

void checkFinalColorSaturationBackward() {
    // The renderer clamps the final composite to one. A saturated channel is
    // therefore locally constant and must not reach the raster/SH backward,
    // while neighboring channels below the clamp must retain their gradients.
    constexpr std::array<float, 3> appearance = {1.5f, 0.8f, 0.25f};
    constexpr float fullFrameScale = 100.0f;

    const StepResult monolithic = runStep(
        true, 0.0f, 1, 0.9f,
        false, false, false, false, false,
        kWidth, kHeight, true, {}, 1, fullFrameScale,
        {}, {}, appearance);
    CHECK(monolithic.radius > 0);
    CHECK(monolithic.nonfiniteFirstMomentCount[3] == 0);
    CHECK(monolithic.colorFirstMoments[0][0] == 0.0f);
    CHECK(monolithic.colorFirstMoments[0][1] != 0.0f);
    CHECK(monolithic.colorFirstMoments[0][2] != 0.0f);

    // 513 contributors force the chunked forward merge and backward path.
    // Saturate a different channel so the fixture also rejects a channel-
    // specific gate rather than only proving that some gradient was removed.
    constexpr int gaussianCount = 513;
    constexpr std::array<float, 3> chunkedAppearance = {
        0.8f, 1.5f, 0.25f,
    };
    const StepResult chunked = runStep(
        true, 0.0f, gaussianCount, 0.005f,
        false, false, false, false, false,
        kWidth, kHeight, true, {}, 1, fullFrameScale,
        {}, {}, chunkedAppearance);
    CHECK(chunked.nonfiniteFirstMomentCount[3] == 0);
    for (int index : {0, gaussianCount - 1}) {
        CHECK(chunked.colorFirstMoments[index][0] != 0.0f);
        CHECK(chunked.colorFirstMoments[index][1] == 0.0f);
        CHECK(chunked.colorFirstMoments[index][2] != 0.0f);
    }
}

void checkProjectedColorClampBackward() {
    // The raster backward path receives raw SH output and applies a +0.5 bias
    // before clamping. Exercise both sides and the exact clamp boundary so the
    // optimized cooperative load cannot change the established derivative.
    constexpr std::array<float, 3> appearance = {-0.1f, 0.0f, 0.1f};
    constexpr float fullFrameScale = 100.0f;

    const auto check = [](const StepResult& result, int index) {
        CHECK(result.colorFirstMoments[index][0] == 0.0f);
        CHECK(result.colorFirstMoments[index][1] != 0.0f);
        CHECK(result.colorFirstMoments[index][2] != 0.0f);
    };

    const StepResult monolithic = runStep(
        true, 0.0f, 1, 0.9f,
        false, false, false, false, false,
        kWidth, kHeight, true, {}, 1, fullFrameScale,
        {}, {}, appearance);
    CHECK(monolithic.radius > 0);
    check(monolithic, 0);

    constexpr int gaussianCount = 513;
    const StepResult chunked = runStep(
        true, 0.0f, gaussianCount, 0.005f,
        false, false, false, false, false,
        kWidth, kHeight, true, {}, 1, fullFrameScale,
        {}, {}, appearance);
    check(chunked, 0);
    check(chunked, gaussianCount - 1);
}

void checkPartialSsimThreadgroups() {
    const float initialOpacity = std::log(0.1f / 0.9f);
    const StepResult partial = runStep(
        true, 0.1f, 1, 0.1f,
        false, false, false, false, false,
        31, 23);
    CHECK(partial.radius > 0);
    CHECK(std::isfinite(partial.opacity));
    CHECK(std::isfinite(partial.opacityFirstMoment));
    CHECK(partial.opacity < initialOpacity);
    CHECK(partial.opacityFirstMoment > 0.0f);
}

void checkArenaRetryTransaction() {
    const char *mode = std::getenv("MSPLAT_TRAINING_ARENA_MODE");
    if (!mode || std::string(mode) != "retry") return;

    // Earlier single-Gaussian steps establish a 4,097-entry arena without
    // radix scratch. First force the synchronous, engine-lock-held fallback
    // used by direct native callers that do not collect telemetry. Its rejected
    // attempt must not collect statistics; only its replay may mutate state.
    constexpr int fallbackGaussianCount = 5'000;
    const StepResult fallback = runStep(
        false, 0.0f, fallbackGaussianCount, 0.001f,
        false, false, false, true, false);
    CHECK(fallback.radius > 0);
    CHECK(fallback.visibilityCount == 1.0f);
    CHECK(fallback.lastVisibilityCount == 1.0f);

    // The fallback grows to 9,096 entries. A larger logical step then forces
    // the step-owned readback path through the same retry transaction.
    constexpr int gaussianCount = 10'000;
    const StepResult retried = runStep(
        false, 0.0f, gaussianCount, 0.001f,
        false, false, false, true, true);

    CHECK(retried.radius > 0);
    CHECK(retried.visibilityCount == 1.0f);
    CHECK(retried.lastVisibilityCount == 1.0f);
    CHECK(retried.overflowReasons ==
          MSPLAT_TRAINING_OVERFLOW_PACKED_CAPACITY);
    CHECK(retried.retainedIntersectionCount == gaussianCount);
    CHECK(retried.intersectionCapacity >=
          retried.retainedIntersectionCount);
    CHECK(std::isfinite(retried.intersectionArenaGrowMs));
    CHECK(retried.intersectionArenaGrowMs >= 0.0);
    CHECK(std::isfinite(retried.countGpuMs));
    CHECK(retried.countGpuMs >= 0.0);
    CHECK(std::isfinite(retried.countWaitWallMs));
    CHECK(retried.countWaitWallMs >= 0.0);
    CHECK(retried.queueIdleTimeValid);
    CHECK(std::isfinite(retried.queueIdleMs));
    CHECK(retried.queueIdleMs >= 0.0);
    // Failed preflight + successful preflight + asynchronous post-count work.
    CHECK(retried.commandBufferCount == 3u);
    CHECK(retried.maximumTileCount == gaussianCount);
    CHECK(retried.activeTileCount == 1u);
    CHECK(retried.trivialTileCount == 3u);
    CHECK(retried.smallTileCount == 0u);
    CHECK(retried.mediumTileCount == 0u);
    CHECK(retried.largeTileCount == 1u);
    CHECK(retried.trivialTileCount + retried.smallTileCount +
          retried.mediumTileCount + retried.largeTileCount == 4u);
    CHECK(retried.overflowedStepCount == 1u);
    CHECK(retried.tileCapOverflowedStepCount == 0u);
    CHECK(retried.packedCapacityOverflowedStepCount == 1u);
    CHECK(retried.lastOverflowIteration == 1);

    const char* attributes =
        std::getenv("MSPLAT_INTERSECTION_ATTRIBUTES");
    const bool gather = !attributes || std::string(attributes) != "packed";
    const size_t packedAttributeBytes =
        msplat_packed_intersection_attribute_bytes();
    CHECK(gather ? packedAttributeBytes == 0 : packedAttributeBytes > 0);
}

void checkRetryReadbackPoolReuse() {
    const char *mode = std::getenv("MSPLAT_TRAINING_ARENA_MODE");
    if (!mode || std::string(mode) != "retry") return;

    const MsplatTrainingTelemetryHandle telemetry =
        msplat_training_telemetry_create();
    const StepResult dense = runStep(
        false, 0.0f, 64, 0.01f,
        false, false, false, false, true,
        kWidth, kHeight, false, telemetry, 11);
    CHECK(dense.retainedIntersectionCount == 64u);
    CHECK(dense.maximumTileCount == 64u);
    CHECK(dense.mediumTileCount == 1u);
    const size_t firstReadbackBytes =
        msplat_training_telemetry_readback_bytes(telemetry);
    CHECK(firstReadbackBytes > 0);

    const StepResult sparse = runStep(
        false, 0.0f, 1, 0.01f,
        false, false, false, false, true,
        kWidth, kHeight, false, telemetry, 12);
    CHECK(msplat_training_telemetry_readback_bytes(telemetry) ==
          firstReadbackBytes);
    CHECK(sparse.retainedIntersectionCount == 1u);
    CHECK(sparse.maximumTileCount == 1u);
    CHECK(sparse.activeTileCount == 1u);
    CHECK(sparse.trivialTileCount == 4u);
    CHECK(sparse.smallTileCount == 0u);
    CHECK(sparse.mediumTileCount == 0u);
    CHECK(sparse.largeTileCount == 0u);
}

void checkStageProfiling() {
    if (!std::getenv("PROFILE_STAGES")) return;

    msplat_gpu_sync();
    constexpr int maxStages = 16;
    // Discard timings from the functional checks above. Profile one known
    // step so adding a new regression cannot silently change this test's
    // expected sample count.
    {
        std::vector<double> discarded[maxStages];
        const char* discardedNames[maxStages] = {};
        int discardedCount = 0;
        msplat_drain_stage_times(
            discarded, maxStages, discardedCount, discardedNames);
    }
    const StepResult probe = runStep(true, 0.1f);
    CHECK(probe.radius > 0);
    msplat_gpu_sync();

    std::vector<double> stageTimes[maxStages];
    const char* stageNames[maxStages] = {};
    int stageCount = 0;
    msplat_drain_stage_times(
        stageTimes, maxStages, stageCount, stageNames);

    const std::array<const char*, 8> expectedNames = {
        "blit_zero", "proj_layout_validate", "scatter_sort_finalize", "pack",
        "rast_fwd", "loss_fwd_bwd", "rast_bwd", "proj_sh_bwd_adam",
    };
    CHECK(stageCount == static_cast<int>(expectedNames.size()));
    size_t profiledStepCount = 0;
    for (int index = 0; index < stageCount; ++index) {
        CHECK(stageNames[index] != nullptr);
        CHECK(std::string(stageNames[index]) == expectedNames[index]);
        if (index > 0 && !stageTimes[index].empty() && profiledStepCount == 0)
            profiledStepCount = stageTimes[index].size();
    }
    // Counter sampling is not available on every macOS Metal device. When it
    // is available, every active compute stage must produce a measurement.
    if (profiledStepCount == 0) return;
    CHECK(profiledStepCount == 1u);
    for (int index = 1; index < stageCount; ++index) {
        CHECK(stageTimes[index].size() == profiledStepCount);
        for (double sampleMs : stageTimes[index]) {
            CHECK(std::isfinite(sampleMs));
            CHECK(sampleMs >= 0.0);
            CHECK(sampleMs < 60'000.0);
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc != 2)
                throw std::invalid_argument("Expected the metallib path");
            msplat_set_metallib_path_checked(argv[1]);
            checkTransparentAlphaSupervision();
            checkChunkedTransparentAlphaSupervision();
            checkCappedAlphaBackward();
            checkFinalColorSaturationBackward();
            checkProjectedColorClampBackward();
            checkRetryReadbackPoolReuse();
            checkArenaRetryTransaction();
            checkPartialSsimThreadgroups();
            checkStageProfiling();
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
