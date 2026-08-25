#include "bindings.h"

#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const char *expression, int line) {
    throw std::runtime_error(
        "line " + std::to_string(line) + ": " + expression);
}

#define CHECK(condition) \
    do { if (!(condition)) fail(#condition, __LINE__); } while (false)

float floatFromBits(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t floatBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint32_t densifyMixBitsReference(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

std::array<float, 2> densifyNormalPairReference(
    uint32_t seed, uint32_t pairIndex) {
    const uint32_t stream = pairIndex + 1u;
    const uint32_t base = seed ^ (stream * 0x9e3779b9u);
    const float u1 = static_cast<float>(
        (densifyMixBitsReference(base ^ 0xa511e9b3u) >> 8) | 1u) *
        (1.0f / 16777216.0f);
    const float u2 = static_cast<float>(
        (densifyMixBitsReference(base ^ 0x63d83595u) >> 8) | 1u) *
        (1.0f / 16777216.0f);
    const float radius = std::sqrt(-2.0f * std::log(u1));
    const float angle = 6.28318530717958647692f * u2;
    return {radius * std::cos(angle), radius * std::sin(angle)};
}

std::array<float, 6> expectedGpuDensifySamples(
    uint32_t seed, uint32_t splitOrdinal) {
    const uint32_t firstPair = splitOrdinal * 3u;
    const auto pair0 = densifyNormalPairReference(seed, firstPair);
    const auto pair1 = densifyNormalPairReference(seed, firstPair + 1u);
    const auto pair2 = densifyNormalPairReference(seed, firstPair + 2u);
    return {pair0[0], pair0[1], pair1[0],
            pair1[1], pair2[0], pair2[1]};
}

void checkOpacityReset() {
    constexpr int activeCount = 7;
    constexpr int capacity = 11;
    constexpr float resetLogit = -1.3862943611198906f;
    MTensor opacityBacking = gpu_empty({capacity, 1}, DType::Float32);
    MTensor firstMomentBacking = gpu_empty({capacity, 1}, DType::Float32);
    MTensor secondMomentBacking = gpu_empty({capacity, 1}, DType::Float32);

    constexpr uint32_t payloadNaNBits = 0x7fc12345u;
    const float opacityValues[capacity] = {
        std::nextafter(resetLogit,
                       -std::numeric_limits<float>::infinity()),
        resetLogit,
        std::nextafter(resetLogit,
                       std::numeric_limits<float>::infinity()),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        floatFromBits(payloadNaNBits),
        -0.0f,
        17.0f,
        floatFromBits(payloadNaNBits),
        -42.0f,
        resetLogit,
    };
    uint32_t originalOpacityBits[capacity] = {};
    uint32_t originalFirstMomentBits[capacity] = {};
    uint32_t originalSecondMomentBits[capacity] = {};
    for (int index = 0; index < capacity; ++index) {
        opacityBacking.data<float>()[index] = opacityValues[index];
        firstMomentBacking.data<float>()[index] = index == 2
            ? std::numeric_limits<float>::infinity()
            : (index == 4 ? floatFromBits(payloadNaNBits) : 10.0f + index);
        secondMomentBacking.data<float>()[index] = index == 1
            ? -0.0f
            : (index == 5 ? -std::numeric_limits<float>::infinity()
                          : 20.0f + index);
        originalOpacityBits[index] =
            floatBits(opacityBacking.data<float>()[index]);
        originalFirstMomentBits[index] =
            floatBits(firstMomentBacking.data<float>()[index]);
        originalSecondMomentBits[index] =
            floatBits(secondMomentBacking.data<float>()[index]);
    }

    MTensor opacities = opacityBacking.view(activeCount);
    MTensor firstMoment = firstMomentBacking.view(activeCount);
    MTensor secondMoment = secondMomentBacking.view(activeCount);
    msplat_reset_opacity_state(
        opacities, firstMoment, secondMoment, resetLogit);
    msplat_gpu_sync();

    CHECK(floatBits(opacityBacking.data<float>()[0]) ==
          originalOpacityBits[0]);
    CHECK(floatBits(opacityBacking.data<float>()[1]) ==
          originalOpacityBits[1]);
    CHECK(floatBits(opacityBacking.data<float>()[2]) ==
          floatBits(resetLogit));
    CHECK(floatBits(opacityBacking.data<float>()[3]) ==
          floatBits(resetLogit));
    CHECK(floatBits(opacityBacking.data<float>()[4]) ==
          originalOpacityBits[4]);
    CHECK(floatBits(opacityBacking.data<float>()[5]) == payloadNaNBits);
    CHECK(floatBits(opacityBacking.data<float>()[6]) ==
          floatBits(resetLogit));
    for (int index = 0; index < activeCount; ++index) {
        CHECK(floatBits(firstMomentBacking.data<float>()[index]) == 0u);
        CHECK(floatBits(secondMomentBacking.data<float>()[index]) == 0u);
    }

    // Views limit maintenance to the active prefix; capacity slack must remain
    // untouched for later topology growth.
    for (int index = activeCount; index < capacity; ++index) {
        CHECK(floatBits(opacityBacking.data<float>()[index]) ==
              originalOpacityBits[index]);
        CHECK(floatBits(firstMomentBacking.data<float>()[index]) ==
              originalFirstMomentBits[index]);
        CHECK(floatBits(secondMomentBacking.data<float>()[index]) ==
              originalSecondMomentBits[index]);
    }

    uint32_t resetOpacityBits[capacity] = {};
    uint32_t resetFirstMomentBits[capacity] = {};
    uint32_t resetSecondMomentBits[capacity] = {};
    for (int index = 0; index < capacity; ++index) {
        resetOpacityBits[index] =
            floatBits(opacityBacking.data<float>()[index]);
        resetFirstMomentBits[index] =
            floatBits(firstMomentBacking.data<float>()[index]);
        resetSecondMomentBits[index] =
            floatBits(secondMomentBacking.data<float>()[index]);
    }

    msplat_reset_opacity_state(
        opacities, firstMoment, secondMoment, resetLogit);
    msplat_gpu_sync();
    for (int index = 0; index < capacity; ++index) {
        CHECK(floatBits(opacityBacking.data<float>()[index]) ==
              resetOpacityBits[index]);
        CHECK(floatBits(firstMomentBacking.data<float>()[index]) ==
              resetFirstMomentBits[index]);
        CHECK(floatBits(secondMomentBacking.data<float>()[index]) ==
              resetSecondMomentBits[index]);
    }

    MTensor mismatchedMoment =
        gpu_zeros({activeCount - 1, 1}, DType::Float32);
    bool rejectedMismatch = false;
    try {
        msplat_reset_opacity_state(
            opacities, mismatchedMoment, secondMoment, resetLogit);
    } catch (const std::invalid_argument&) {
        rejectedMismatch = true;
    }
    CHECK(rejectedMismatch);
}

constexpr int kRandomFixtureParents = 512;
constexpr int kRandomFixturePopulation = 3 * kRandomFixtureParents;
constexpr int kRandomFixtureCapacity = kRandomFixturePopulation + 5;
constexpr uint32_t kRandomScratchSentinelBits = 0x7f7fffffu;

struct DensifyRandomResult {
    std::vector<uint32_t> childMeanBits;
    std::vector<uint32_t> randomScratchBits;
    std::vector<uint32_t> capacityTailBits;
};

DensifyRandomResult runDensifyRandomFixture(
    uint32_t randomSeed, bool gpuMode,
    bool forceUndersizedRandomScratch = false) {
    constexpr int featuresRestStride = 1;
    constexpr int parameterStrides[] = {3, 3, 4, 3, 1, 1};

    MTensor max2DSize = gpu_zeros(
        {kRandomFixtureParents}, DType::Float32);
    MTensor means = gpu_zeros(
        {kRandomFixtureCapacity, 3}, DType::Float32);
    MTensor scales = gpu_zeros(
        {kRandomFixtureCapacity, 3}, DType::Float32);
    MTensor quaternions = gpu_zeros(
        {kRandomFixtureCapacity, 4}, DType::Float32);
    MTensor featuresDc = gpu_zeros(
        {kRandomFixtureCapacity, 3}, DType::Float32);
    MTensor featuresRest = gpu_zeros(
        {kRandomFixtureCapacity, featuresRestStride}, DType::Float32);
    MTensor opacities = gpu_zeros(
        {kRandomFixtureCapacity, 1}, DType::Float32);

    MTensor firstMoments[6];
    MTensor secondMoments[6];
    for (int group = 0; group < 6; ++group) {
        firstMoments[group] = gpu_zeros(
            {kRandomFixtureCapacity, parameterStrides[group]},
            DType::Float32);
        secondMoments[group] = gpu_zeros(
            {kRandomFixtureCapacity, parameterStrides[group]},
            DType::Float32);
    }

    MTensor splitFlag = gpu_zeros(
        {kRandomFixtureParents}, DType::Int32);
    MTensor duplicateFlag = gpu_zeros(
        {kRandomFixtureParents}, DType::Int32);
    MTensor splitPrefix = gpu_zeros(
        {kRandomFixtureParents}, DType::Int32);
    MTensor duplicatePrefix = gpu_zeros(
        {kRandomFixtureParents}, DType::Int32);
    MTensor keepFlag = gpu_zeros(
        {kRandomFixtureCapacity}, DType::Int32);
    MTensor keepPrefix = gpu_zeros(
        {kRandomFixtureCapacity}, DType::Int32);
    MTensor blockTotals = gpu_zeros(
        {(kRandomFixturePopulation + 1023) / 1024}, DType::Int32);
    MTensor compactScratch = gpu_zeros(
        {4LL * kRandomFixturePopulation}, DType::Float32);
    MTensor randomSamples = gpu_empty(
        gpuMode || forceUndersizedRandomScratch
            ? std::vector<int64_t>{1}
            : std::vector<int64_t>{kRandomFixtureCapacity, 3},
        DType::Float32);

    const float sentinel = floatFromBits(kRandomScratchSentinelBits);
    for (int index = 0; index < randomSamples.numel(); ++index)
        randomSamples.data<float>()[index] = sentinel;
    for (int index = 3 * kRandomFixturePopulation;
         index < means.numel(); ++index) {
        means.data<float>()[index] = sentinel;
    }
    for (int index = 0; index < kRandomFixtureParents; ++index) {
        splitFlag.data<int32_t>()[index] = 1;
        splitPrefix.data<int32_t>()[index] = index + 1;
        quaternions.data<float>()[index * 4] = 1.0f;
    }

    const int densifiedCount = msplat_densify(
        kRandomFixtureParents, kRandomFixturePopulation,
        0.1f, 0.5f, 0.15f, 0, 0,
        max2DSize,
        means, scales, quaternions,
        featuresDc, featuresRest, opacities, featuresRestStride,
        firstMoments, secondMoments,
        splitFlag, duplicateFlag,
        splitPrefix, duplicatePrefix,
        keepFlag, keepPrefix,
        blockTotals, compactScratch,
        randomSamples, randomSeed);
    CHECK(densifiedCount == 2 * kRandomFixtureParents);

    DensifyRandomResult result;
    result.childMeanBits.reserve(3 * densifiedCount);
    for (int index = 0; index < 3 * densifiedCount; ++index)
        result.childMeanBits.push_back(floatBits(means.data<float>()[index]));
    result.randomScratchBits.reserve(randomSamples.numel());
    for (int index = 0; index < randomSamples.numel(); ++index) {
        result.randomScratchBits.push_back(
            floatBits(randomSamples.data<float>()[index]));
    }
    for (int index = 3 * kRandomFixturePopulation;
         index < means.numel(); ++index) {
        result.capacityTailBits.push_back(
            floatBits(means.data<float>()[index]));
    }
    return result;
}

void checkDensificationRandomMode(bool gpuMode) {
    CHECK(msplat_densify_uses_gpu_random() == gpuMode);
    DensifyRandomResult first = runDensifyRandomFixture(600u, gpuMode);
    DensifyRandomResult repeated = runDensifyRandomFixture(600u, gpuMode);
    DensifyRandomResult different = runDensifyRandomFixture(
        std::numeric_limits<uint32_t>::max(), gpuMode);

    CHECK(first.childMeanBits == repeated.childMeanBits);
    CHECK(first.childMeanBits != different.childMeanBits);
    for (uint32_t bits : first.capacityTailBits)
        CHECK(bits == kRandomScratchSentinelBits);

    const int64_t generatedSampleCount =
        6LL * kRandomFixtureParents;
    if (gpuMode) {
        CHECK(first.randomScratchBits.size() == 1u);
        for (uint32_t bits : first.randomScratchBits)
            CHECK(bits == kRandomScratchSentinelBits);
        constexpr uint32_t representativeOrdinals[] = {
            0u, 1u, 17u, kRandomFixtureParents - 1u};
        for (uint32_t ordinal : representativeOrdinals) {
            const auto expected = expectedGpuDensifySamples(600u, ordinal);
            for (int component = 0; component < 6; ++component) {
                const float actual = floatFromBits(
                    first.childMeanBits[ordinal * 6u + component]);
                const float tolerance =
                    2.0e-4f * (1.0f + std::abs(expected[component]));
                CHECK(std::abs(actual - expected[component]) <= tolerance);
            }
        }
    } else {
        bool rejectedUndersizedScratch = false;
        try {
            (void)runDensifyRandomFixture(600u, false, true);
        } catch (const std::runtime_error& error) {
            rejectedUndersizedScratch = std::string(error.what()) ==
                "Densification buffer is too small: random_samples";
        }
        CHECK(rejectedUndersizedScratch);

        std::mt19937 expectedGenerator(600u);
        std::normal_distribution<float> expectedDistribution(0.0f, 1.0f);
        for (int64_t index = 0; index < generatedSampleCount; ++index) {
            CHECK(first.randomScratchBits[index] ==
                  floatBits(expectedDistribution(expectedGenerator)));
        }
        for (size_t index = static_cast<size_t>(generatedSampleCount);
             index < first.randomScratchBits.size(); ++index) {
            CHECK(first.randomScratchBits[index] ==
                  kRandomScratchSentinelBits);
        }
    }

    double sum = 0.0;
    double sumSquares = 0.0;
    for (uint32_t bits : first.childMeanBits) {
        const float sample = floatFromBits(bits);
        CHECK(std::isfinite(sample));
        sum += sample;
        sumSquares += static_cast<double>(sample) * sample;
    }
    const double count = static_cast<double>(first.childMeanBits.size());
    const double mean = sum / count;
    const double variance = sumSquares / count - mean * mean;
    CHECK(std::abs(mean) < 0.08);
    CHECK(variance > 0.80);
    CHECK(variance < 1.20);
}

void checkMutuallyExclusiveClassification() {
    constexpr int count = 4;
    MTensor gradients = gpu_zeros({count}, DType::Float32);
    MTensor visibility = gpu_zeros({count}, DType::Float32);
    MTensor screenSize = gpu_zeros({count}, DType::Float32);
    MTensor scales = gpu_zeros({count, 3}, DType::Float32);
    MTensor splitFlag = gpu_zeros({count}, DType::Int32);
    MTensor duplicateFlag = gpu_zeros({count}, DType::Int32);
    MTensor splitPrefix = gpu_zeros({count}, DType::Int32);
    MTensor duplicatePrefix = gpu_zeros({count}, DType::Int32);
    MTensor blockTotals = gpu_zeros({1}, DType::Int32);

    const float smallScale = std::log(0.005f);
    const float largeScale = std::log(0.02f);
    for (int index = 0; index < count; ++index) {
        gradients.data<float>()[index] = index == 3 ? 0.0001f : 0.001f;
        visibility.data<float>()[index] = 1.0f;
        screenSize.data<float>()[index] = index == 0 || index == 3
            ? 0.06f
            : 0.01f;
        const float scale = index == 1 ? largeScale : smallScale;
        for (int axis = 0; axis < 3; ++axis)
            scales.data<float>()[index * 3 + axis] = scale;
    }

    int splits = -1;
    int duplicates = -1;
    msplat_prepare_densify(
        count, -1,
        0.0002f, 0.01f, 0.05f, 1,
        gradients, visibility, screenSize, 1.0f,
        scales,
        splitFlag, duplicateFlag,
        splitPrefix, duplicatePrefix,
        blockTotals,
        splits, duplicates);

    CHECK(splits == 2);
    CHECK(duplicates == 1);
    const int32_t expectedSplitFlag[] = {1, 1, 0, 0};
    const int32_t expectedDuplicateFlag[] = {0, 0, 1, 0};
    const int32_t expectedSplitPrefix[] = {1, 2, 2, 2};
    const int32_t expectedDuplicatePrefix[] = {0, 0, 1, 1};
    for (int index = 0; index < count; ++index) {
        CHECK(splitFlag.data<int32_t>()[index] == expectedSplitFlag[index]);
        CHECK(duplicateFlag.data<int32_t>()[index] ==
              expectedDuplicateFlag[index]);
        CHECK(splitPrefix.data<int32_t>()[index] ==
              expectedSplitPrefix[index]);
        CHECK(duplicatePrefix.data<int32_t>()[index] ==
              expectedDuplicatePrefix[index]);
    }

    // Egg Preview starts at its Gaussian ceiling. The same classification must
    // be trimmed to zero growth without tripping the overlap invariant.
    msplat_prepare_densify(
        count, count,
        0.0002f, 0.01f, 0.05f, 1,
        gradients, visibility, screenSize, 1.0f,
        scales,
        splitFlag, duplicateFlag,
        splitPrefix, duplicatePrefix,
        blockTotals,
        splits, duplicates);

    CHECK(splits == 0);
    CHECK(duplicates == 0);
    for (int index = 0; index < count; ++index) {
        CHECK(splitFlag.data<int32_t>()[index] == 0);
        CHECK(duplicateFlag.data<int32_t>()[index] == 0);
        CHECK(splitPrefix.data<int32_t>()[index] == 0);
        CHECK(duplicatePrefix.data<int32_t>()[index] == 0);
    }
}

}  // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc != 3)
                throw std::invalid_argument(
                    "Expected the metallib path and random mode");
            const std::string expectedMode = argv[2];
            const char* configuredMode =
                std::getenv("MSPLAT_DENSIFY_RANDOM_MODE");
            if (expectedMode == "default") {
                CHECK(configuredMode == nullptr);
            } else {
                CHECK(configuredMode != nullptr);
                CHECK(expectedMode == configuredMode);
            }
            if (expectedMode == "invalid") {
                bool rejected = false;
                msplat_set_metallib_path_checked(argv[1]);
                try {
                    (void)gpu_zeros({1}, DType::Float32);
                } catch (const std::invalid_argument& error) {
                    rejected = std::string(error.what()) ==
                        "msplat: MSPLAT_DENSIFY_RANDOM_MODE must be cpu or gpu";
                }
                CHECK(rejected);
                return 0;
            }

            msplat_set_metallib_path_checked(argv[1]);
            checkOpacityReset();
            checkDensificationRandomMode(expectedMode == "gpu");
            checkMutuallyExclusiveClassification();
            cleanup_msplat_metal();
            return 0;
        } catch (const std::exception &error) {
            if (std::string(error.what()) ==
                "msplat: no Metal device is available") {
                std::cerr << "SKIP: " << error.what() << '\n';
                return 77;
            }
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
