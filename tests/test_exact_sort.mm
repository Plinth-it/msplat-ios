#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "intersection_layout.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const char* expression, int line) {
    throw std::runtime_error(
        "line " + std::to_string(line) + ": " + expression);
}

#define CHECK(condition) \
    do { if (!(condition)) fail(#condition, __LINE__); } while (false)

struct ScopedObjCRelease {
    id object = nil;

    ~ScopedObjCRelease() {
        [object release];
    }
};

std::runtime_error metalError(const char* operation, NSError* error) {
    std::string message = operation;
    const char* description =
        error ? error.localizedDescription.UTF8String : nullptr;
    if (description) {
        message += ": ";
        message += description;
    }
    return std::runtime_error(message);
}

uint64_t mixedKey(uint32_t tile, uint32_t localIndex) {
    uint64_t value =
        (static_cast<uint64_t>(tile) << 32) | localIndex;
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, const void* bytes,
                         size_t byteCount) {
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:bytes
                            length:byteCount
                           options:MTLResourceStorageModeShared];
    if (!buffer) {
        throw std::runtime_error("Failed to allocate a Metal test buffer");
    }
    return buffer;
}

void checkExactSort(id<MTLDevice> device, const char* metallibPath) {
    constexpr std::array<uint32_t, 9> counts = {
        33, 2, 2'049, 31, 1, 2'048, 3, 32, 0,
    };
    constexpr std::array<uint32_t, 7> expectedSortableTiles = {
        1, 3, 6, 7, 0, 5, 2,
    };
    constexpr int32_t untouchedBin = -77;
    constexpr uint64_t scratchPoison = 0xCDCDCDCDCDCDCDCDULL;

    std::array<int32_t, counts.size()> offsets = {};
    std::array<uint32_t, counts.size()> sortableTiles;
    sortableTiles.fill(std::numeric_limits<uint32_t>::max());
    const msplat::TileIntersectionLayout layout =
        msplat::buildTileIntersectionLayout(
            counts.data(), offsets.data(), counts.size(), nullptr,
            sortableTiles.data());

    CHECK(layout.totalCount == 4'199);
    CHECK(layout.smallTileCount == 4);
    CHECK(layout.mediumTileCount == 2);
    CHECK(layout.largeTileCount == 1);
    CHECK(layout.sortableTileCount == expectedSortableTiles.size());
    CHECK(layout.smallTileCount + layout.mediumTileCount +
              layout.largeTileCount == layout.sortableTileCount);
    CHECK(std::equal(expectedSortableTiles.begin(),
                     expectedSortableTiles.end(), sortableTiles.begin()));
    CHECK(sortableTiles[7] == std::numeric_limits<uint32_t>::max());
    CHECK(sortableTiles[8] == std::numeric_limits<uint32_t>::max());

    std::vector<uint64_t> keys(layout.totalCount);
    size_t start = 0;
    for (uint32_t tile = 0; tile < counts.size(); ++tile) {
        const uint32_t count = counts[tile];
        for (uint32_t local = 0; local < count; ++local) {
            // Reverse the local index before mixing so no range begins sorted.
            keys[start + local] = mixedKey(tile, count - local - 1);
        }
        start += count;
    }
    std::vector<uint64_t> expectedKeys = keys;
    start = 0;
    for (uint32_t count : counts) {
        std::sort(expectedKeys.begin() + static_cast<ptrdiff_t>(start),
                  expectedKeys.begin() +
                      static_cast<ptrdiff_t>(start + count));
        start += count;
    }

    std::vector<uint64_t> scratch(layout.totalCount, scratchPoison);
    std::array<int32_t, counts.size() * 2> bins;
    bins.fill(untouchedBin);
    uint32_t overflow = 0;

    NSString* path = [NSString stringWithUTF8String:metallibPath];
    CHECK(path != nil);
    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if (!library) throw metalError("Failed to load the test metallib", error);
    ScopedObjCRelease libraryOwner{library};

    id<MTLFunction> smallFunction =
        [library newFunctionWithName:@"small_sort_per_tile_kernel"];
    CHECK(smallFunction != nil);
    ScopedObjCRelease smallFunctionOwner{smallFunction};
    id<MTLFunction> generalFunction =
        [library newFunctionWithName:@"radix_sort_per_tile_kernel"];
    CHECK(generalFunction != nil);
    ScopedObjCRelease generalFunctionOwner{generalFunction};

    error = nil;
    id<MTLComputePipelineState> smallPipeline =
        [device newComputePipelineStateWithFunction:smallFunction error:&error];
    if (!smallPipeline) {
        throw metalError("Failed to create the small-sort pipeline", error);
    }
    ScopedObjCRelease smallPipelineOwner{smallPipeline};
    error = nil;
    id<MTLComputePipelineState> generalPipeline =
        [device newComputePipelineStateWithFunction:generalFunction error:&error];
    if (!generalPipeline) {
        throw metalError("Failed to create the general-sort pipeline", error);
    }
    ScopedObjCRelease generalPipelineOwner{generalPipeline};
    CHECK(smallPipeline.maxTotalThreadsPerThreadgroup >= 32);
    CHECK(generalPipeline.maxTotalThreadsPerThreadgroup >= 256);

    id<MTLBuffer> offsetsBuffer = makeBuffer(
        device, offsets.data(), offsets.size() * sizeof(offsets[0]));
    ScopedObjCRelease offsetsBufferOwner{offsetsBuffer};
    id<MTLBuffer> keysBuffer = makeBuffer(
        device, keys.data(), keys.size() * sizeof(keys[0]));
    ScopedObjCRelease keysBufferOwner{keysBuffer};
    id<MTLBuffer> scratchBuffer = makeBuffer(
        device, scratch.data(), scratch.size() * sizeof(scratch[0]));
    ScopedObjCRelease scratchBufferOwner{scratchBuffer};
    id<MTLBuffer> binsBuffer = makeBuffer(
        device, bins.data(), bins.size() * sizeof(bins[0]));
    ScopedObjCRelease binsBufferOwner{binsBuffer};
    id<MTLBuffer> overflowBuffer =
        makeBuffer(device, &overflow, sizeof(overflow));
    ScopedObjCRelease overflowBufferOwner{overflowBuffer};
    id<MTLBuffer> sortableTilesBuffer = makeBuffer(
        device, sortableTiles.data(),
        sortableTiles.size() * sizeof(sortableTiles[0]));
    ScopedObjCRelease sortableTilesBufferOwner{sortableTilesBuffer};

    id<MTLCommandQueue> queue = [device newCommandQueue];
    CHECK(queue != nil);
    ScopedObjCRelease queueOwner{queue};
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    CHECK(commandBuffer != nil);
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    CHECK(encoder != nil);

    const uint32_t numTiles = static_cast<uint32_t>(counts.size());
    const uint32_t capacity = layout.totalCount;
    const uint32_t smallTileCount = layout.smallTileCount;
    [encoder setComputePipelineState:smallPipeline];
    [encoder setBuffer:offsetsBuffer offset:0 atIndex:0];
    [encoder setBuffer:keysBuffer offset:0 atIndex:1];
    [encoder setBytes:&numTiles length:sizeof(numTiles) atIndex:2];
    [encoder setBuffer:binsBuffer offset:0 atIndex:3];
    [encoder setBytes:&capacity length:sizeof(capacity) atIndex:4];
    [encoder setBuffer:overflowBuffer offset:0 atIndex:5];
    [encoder setBuffer:sortableTilesBuffer offset:0 atIndex:6];
    [encoder setBytes:&smallTileCount
                length:sizeof(smallTileCount) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(smallTileCount, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    const uint32_t generalTileCount =
        layout.mediumTileCount + layout.largeTileCount;
    const uint32_t generalTileOffset = layout.smallTileCount;
    [encoder setComputePipelineState:generalPipeline];
    [encoder setBuffer:offsetsBuffer offset:0 atIndex:0];
    [encoder setBuffer:keysBuffer offset:0 atIndex:1];
    [encoder setBuffer:scratchBuffer offset:0 atIndex:2];
    [encoder setBytes:&numTiles length:sizeof(numTiles) atIndex:3];
    [encoder setBuffer:binsBuffer offset:0 atIndex:4];
    [encoder setBytes:&capacity length:sizeof(capacity) atIndex:5];
    [encoder setBuffer:overflowBuffer offset:0 atIndex:6];
    [encoder setBuffer:sortableTilesBuffer offset:0 atIndex:7];
    [encoder setBytes:&generalTileCount
                length:sizeof(generalTileCount) atIndex:8];
    [encoder setBytes:&generalTileOffset
                length:sizeof(generalTileOffset) atIndex:9];
    [encoder dispatchThreadgroups:MTLSizeMake(generalTileCount, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) {
        throw metalError("Exact-sort command buffer failed", commandBuffer.error);
    }
    CHECK(commandBuffer.status == MTLCommandBufferStatusCompleted);

    const auto* sortedKeys =
        static_cast<const uint64_t*>(keysBuffer.contents);
    CHECK(std::equal(expectedKeys.begin(), expectedKeys.end(), sortedKeys));
    const auto* resultBins =
        static_cast<const int32_t*>(binsBuffer.contents);
    int32_t expectedStart = 0;
    for (uint32_t tile = 0; tile < counts.size(); ++tile) {
        const int32_t expectedEnd =
            expectedStart + static_cast<int32_t>(counts[tile]);
        if (counts[tile] > 1) {
            CHECK(resultBins[2 * tile] == expectedStart);
            CHECK(resultBins[2 * tile + 1] == expectedEnd);
        } else {
            CHECK(resultBins[2 * tile] == untouchedBin);
            CHECK(resultBins[2 * tile + 1] == untouchedBin);
        }
        expectedStart = expectedEnd;
    }
    CHECK(*static_cast<const uint32_t*>(overflowBuffer.contents) == 0);

    // Only the radix tile needs scratch; every other range must retain poison.
    const uint32_t radixTile = expectedSortableTiles.back();
    const size_t radixStart = radixTile == 0
        ? 0
        : static_cast<size_t>(offsets[radixTile - 1]);
    const size_t radixEnd = static_cast<size_t>(offsets[radixTile]);
    const auto* resultScratch =
        static_cast<const uint64_t*>(scratchBuffer.contents);
    for (size_t index = 0; index < scratch.size(); ++index) {
        if (index < radixStart || index >= radixEnd) {
            CHECK(resultScratch[index] == scratchPoison);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 2) {
                throw std::invalid_argument("Expected the metallib path");
            }
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                std::cerr << "SKIP: msplat: no Metal device is available\n";
                return 77;
            }
            checkExactSort(device, argv[1]);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
