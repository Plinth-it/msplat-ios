#include "bindings.h"

#include <cmath>
#include <cstdint>
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
