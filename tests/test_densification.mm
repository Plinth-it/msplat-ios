#include "bindings.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>

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
            if (argc != 2)
                throw std::invalid_argument("Expected the metallib path");
            msplat_set_metallib_path_checked(argv[1]);
            checkOpacityReset();
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
