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
    uint32_t maximumTileCount = 0;
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
                   bool exerciseAppearance = false) {
    CHECK(gaussianCount > 0);
    std::vector<float> meanValues(static_cast<size_t>(gaussianCount) * 3);
    std::vector<float> scaleValues(static_cast<size_t>(gaussianCount) * 3);
    std::vector<float> quatValues(static_cast<size_t>(gaussianCount) * 4);
    std::vector<float> dcValues(static_cast<size_t>(gaussianCount) * 3);
    std::vector<float> opacityValues(static_cast<size_t>(gaussianCount));

    const float logScale = std::log(0.03f);
    constexpr float shC0 = 0.28209479177387814f;
    const float blackDc = -0.5f / shC0;
    const std::array<float, 3> appearanceDc = {
        -0.1f / shC0,
         0.1f / shC0,
         0.3f / shC0,
    };
    const float initialOpacity = std::log(initialAlpha / (1.0f - initialAlpha));
    for (int index = 0; index < gaussianCount; ++index) {
        const size_t meanOffset = static_cast<size_t>(index) * 3;
        meanValues[meanOffset + 0] = 0.25f;
        meanValues[meanOffset + 1] = -0.25f;
        meanValues[meanOffset + 2] = -1.0f;
        scaleValues[meanOffset + 0] = logScale;
        scaleValues[meanOffset + 1] = logScale;
        scaleValues[meanOffset + 2] = logScale;
        for (size_t channel = 0; channel < appearanceDc.size(); ++channel) {
            dcValues[meanOffset + channel] = exerciseAppearance
                ? appearanceDc[channel]
                : blackDc;
        }
        quatValues[static_cast<size_t>(index) * 4] = 1.0f;
        opacityValues[static_cast<size_t>(index)] = initialOpacity;
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

    MsplatTrainingTelemetryHandle telemetry;
    MsplatLogicalTrainingStepHandle logicalStep;
    if (collectTelemetry) {
        telemetry = msplat_training_telemetry_create();
        logicalStep = msplat_training_step_begin(telemetry, 1);
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
            photometric, pose, collectStats,
            visibility, xyGradientNorm, max2DSize, 1.0f / float(width));
        if (logicalStep) {
            MsplatTrainingStepDescriptor descriptor;
            descriptor.iteration = 1;
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
        result.maximumTileCount = snapshot.completedStep.maximumTileCount;
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
    const bool gather = attributes && std::string(attributes) == "gather";
    const size_t packedAttributeBytes =
        msplat_packed_intersection_attribute_bytes();
    CHECK(gather ? packedAttributeBytes == 0 : packedAttributeBytes > 0);
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
    // radix scratch. One identical 5,000-Gaussian tile then requires a larger
    // arena, radix scratch, and more raster chunks in the GPU-resident path.
    // The rejected attempt must not collect statistics; only its replay may
    // mutate persistent state.
    constexpr int gaussianCount = 5'000;
    const StepResult retried = runStep(
        false, 0.0f, gaussianCount, 0.001f,
        false, false, false, true, true);

    CHECK(retried.radius > 0);
    CHECK(retried.visibilityCount == 1.0f);
    CHECK(retried.lastVisibilityCount == 1.0f);
    CHECK(retried.overflowReasons ==
          MSPLAT_TRAINING_OVERFLOW_PACKED_CAPACITY);
    CHECK(retried.retainedIntersectionCount > 4'097u);
    CHECK(retried.intersectionCapacity >=
          retried.retainedIntersectionCount);
    CHECK(std::isfinite(retried.intersectionArenaGrowMs));
    CHECK(retried.intersectionArenaGrowMs >= 0.0);
    CHECK(retried.maximumTileCount == gaussianCount);
    CHECK(retried.overflowedStepCount == 1u);
    CHECK(retried.tileCapOverflowedStepCount == 0u);
    CHECK(retried.packedCapacityOverflowedStepCount == 1u);
    CHECK(retried.lastOverflowIteration == 1);

    const char* attributes =
        std::getenv("MSPLAT_INTERSECTION_ATTRIBUTES");
    const bool gather = attributes && std::string(attributes) == "gather";
    const size_t packedAttributeBytes =
        msplat_packed_intersection_attribute_bytes();
    CHECK(gather ? packedAttributeBytes == 0 : packedAttributeBytes > 0);
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
            checkArenaRetryTransaction();
            checkPartialSsimThreadgroups();
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
