#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "intersection_layout.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
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

constexpr uint32_t kPoisonUint = 0xCDCDCDCDu;
constexpr int32_t kPoisonInt = static_cast<int32_t>(0xCDCDCDCDu);
constexpr uint32_t kSignedIndexOverflow =
    msplat::kTileIntersectionLayoutSignedIndexOverflow;
constexpr size_t kMetadataWordCount =
    msplat::kTileIntersectionLayoutMetadataWordCount;

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

id<MTLBuffer> makeBuffer(id<MTLDevice> device, const void* bytes,
                         size_t byteCount) {
    CHECK(bytes != nullptr);
    CHECK(byteCount > 0);
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:bytes
                            length:byteCount
                           options:MTLResourceStorageModeShared];
    if (!buffer) {
        throw std::runtime_error("Failed to allocate a Metal test buffer");
    }
    return buffer;
}

template <typename T>
std::vector<T> guarded(size_t liveCount, T value) {
    return std::vector<T>(liveCount + 2, value);
}

std::array<uint32_t, kMetadataWordCount> expectedMetadata(
    const msplat::TileIntersectionLayout& layout) {
    return {
        layout.totalCount,
        layout.maximumTileCount,
        static_cast<uint32_t>(layout.maximumTileIndex),
        layout.activeTileCount,
        layout.sortableTileCount,
        layout.trivialTileCount,
        layout.smallTileCount,
        layout.mediumTileCount,
        layout.largeTileCount,
        0u,
    };
}

void runValidCase(
    id<MTLDevice> device, id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    const std::vector<uint32_t>& counts) {
    CHECK(!counts.empty());
    CHECK(counts.size() <= std::numeric_limits<uint32_t>::max());

    std::vector<int32_t> expectedOffsets(counts.size(), kPoisonInt);
    std::vector<int32_t> expectedBins(counts.size() * 2, kPoisonInt);
    std::vector<uint32_t> expectedSortable(counts.size(), kPoisonUint);
    const msplat::TileIntersectionLayout expectedLayout =
        msplat::buildTileIntersectionLayout(
            counts.data(), expectedOffsets.data(), counts.size(),
            expectedBins.data(), expectedSortable.data());
    const auto expectedMeta = expectedMetadata(expectedLayout);

    std::vector<int32_t> offsets = guarded(counts.size(), kPoisonInt);
    std::vector<int32_t> bins = guarded(counts.size() * 2, kPoisonInt);
    std::vector<uint32_t> sortable = guarded(counts.size(), kPoisonUint);
    std::vector<uint32_t> metadata =
        guarded(kMetadataWordCount, kPoisonUint);

    id<MTLBuffer> countsBuffer = makeBuffer(
        device, counts.data(), counts.size() * sizeof(counts[0]));
    ScopedObjCRelease countsOwner{countsBuffer};
    id<MTLBuffer> offsetsBuffer = makeBuffer(
        device, offsets.data(), offsets.size() * sizeof(offsets[0]));
    ScopedObjCRelease offsetsOwner{offsetsBuffer};
    id<MTLBuffer> binsBuffer = makeBuffer(
        device, bins.data(), bins.size() * sizeof(bins[0]));
    ScopedObjCRelease binsOwner{binsBuffer};
    id<MTLBuffer> sortableBuffer = makeBuffer(
        device, sortable.data(), sortable.size() * sizeof(sortable[0]));
    ScopedObjCRelease sortableOwner{sortableBuffer};
    id<MTLBuffer> metadataBuffer = makeBuffer(
        device, metadata.data(), metadata.size() * sizeof(metadata[0]));
    ScopedObjCRelease metadataOwner{metadataBuffer};

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    CHECK(commandBuffer != nil);
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    CHECK(encoder != nil);
    const uint32_t numTiles = static_cast<uint32_t>(counts.size());
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:countsBuffer offset:0 atIndex:0];
    [encoder setBuffer:offsetsBuffer offset:sizeof(int32_t) atIndex:1];
    [encoder setBuffer:binsBuffer offset:sizeof(int32_t) atIndex:2];
    [encoder setBuffer:sortableBuffer offset:sizeof(uint32_t) atIndex:3];
    [encoder setBuffer:metadataBuffer offset:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&numTiles length:sizeof(numTiles) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) {
        throw metalError(
            "GPU tile-layout command buffer failed", commandBuffer.error);
    }
    CHECK(commandBuffer.status == MTLCommandBufferStatusCompleted);

    const auto* actualOffsets =
        static_cast<const int32_t*>(offsetsBuffer.contents) + 1;
    const auto* actualBins =
        static_cast<const int32_t*>(binsBuffer.contents) + 1;
    const auto* actualSortable =
        static_cast<const uint32_t*>(sortableBuffer.contents) + 1;
    const auto* actualMetadata =
        static_cast<const uint32_t*>(metadataBuffer.contents) + 1;
    CHECK(std::equal(expectedOffsets.begin(), expectedOffsets.end(),
                     actualOffsets));
    CHECK(std::equal(expectedBins.begin(), expectedBins.end(), actualBins));
    CHECK(std::equal(expectedSortable.begin(), expectedSortable.end(),
                     actualSortable));
    CHECK(std::equal(expectedMeta.begin(), expectedMeta.end(),
                     actualMetadata));

    const auto* offsetStorage =
        static_cast<const int32_t*>(offsetsBuffer.contents);
    const auto* binStorage =
        static_cast<const int32_t*>(binsBuffer.contents);
    const auto* sortableStorage =
        static_cast<const uint32_t*>(sortableBuffer.contents);
    const auto* metadataStorage =
        static_cast<const uint32_t*>(metadataBuffer.contents);
    CHECK(offsetStorage[0] == kPoisonInt);
    CHECK(offsetStorage[offsets.size() - 1] == kPoisonInt);
    CHECK(binStorage[0] == kPoisonInt);
    CHECK(binStorage[bins.size() - 1] == kPoisonInt);
    CHECK(sortableStorage[0] == kPoisonUint);
    CHECK(sortableStorage[sortable.size() - 1] == kPoisonUint);
    CHECK(metadataStorage[0] == kPoisonUint);
    CHECK(metadataStorage[metadata.size() - 1] == kPoisonUint);
}

void runOverflowCase(
    id<MTLDevice> device, id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline) {
    const std::array<uint32_t, 2> counts = {
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max()), 1u,
    };
    std::vector<int32_t> offsets = guarded(counts.size(), kPoisonInt);
    std::vector<int32_t> bins = guarded(counts.size() * 2, kPoisonInt);
    std::vector<uint32_t> sortable = guarded(counts.size(), kPoisonUint);
    std::vector<uint32_t> metadata =
        guarded(kMetadataWordCount, kPoisonUint);

    id<MTLBuffer> countsBuffer = makeBuffer(
        device, counts.data(), counts.size() * sizeof(counts[0]));
    ScopedObjCRelease countsOwner{countsBuffer};
    id<MTLBuffer> offsetsBuffer = makeBuffer(
        device, offsets.data(), offsets.size() * sizeof(offsets[0]));
    ScopedObjCRelease offsetsOwner{offsetsBuffer};
    id<MTLBuffer> binsBuffer = makeBuffer(
        device, bins.data(), bins.size() * sizeof(bins[0]));
    ScopedObjCRelease binsOwner{binsBuffer};
    id<MTLBuffer> sortableBuffer = makeBuffer(
        device, sortable.data(), sortable.size() * sizeof(sortable[0]));
    ScopedObjCRelease sortableOwner{sortableBuffer};
    id<MTLBuffer> metadataBuffer = makeBuffer(
        device, metadata.data(), metadata.size() * sizeof(metadata[0]));
    ScopedObjCRelease metadataOwner{metadataBuffer};

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    CHECK(commandBuffer != nil);
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    CHECK(encoder != nil);
    const uint32_t numTiles = static_cast<uint32_t>(counts.size());
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:countsBuffer offset:0 atIndex:0];
    [encoder setBuffer:offsetsBuffer offset:sizeof(int32_t) atIndex:1];
    [encoder setBuffer:binsBuffer offset:sizeof(int32_t) atIndex:2];
    [encoder setBuffer:sortableBuffer offset:sizeof(uint32_t) atIndex:3];
    [encoder setBuffer:metadataBuffer offset:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&numTiles length:sizeof(numTiles) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) {
        throw metalError(
            "Overflowing GPU tile-layout command buffer failed",
            commandBuffer.error);
    }
    CHECK(commandBuffer.status == MTLCommandBufferStatusCompleted);

    const auto* actualOffsets =
        static_cast<const int32_t*>(offsetsBuffer.contents) + 1;
    const auto* actualBins =
        static_cast<const int32_t*>(binsBuffer.contents) + 1;
    const auto* actualSortable =
        static_cast<const uint32_t*>(sortableBuffer.contents) + 1;
    const auto* actualMetadata =
        static_cast<const uint32_t*>(metadataBuffer.contents) + 1;

    CHECK(std::all_of(actualOffsets, actualOffsets + counts.size(),
                      [](int32_t value) { return value == 0; }));
    CHECK(std::all_of(actualBins, actualBins + counts.size() * 2,
                      [](int32_t value) { return value == 0; }));
    CHECK(std::all_of(actualSortable, actualSortable + counts.size(),
                      [](uint32_t value) { return value == 0u; }));
    CHECK(actualMetadata[0] == 0u);
    CHECK(actualMetadata[1] ==
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
    CHECK(actualMetadata[2] == 0u);
    CHECK(actualMetadata[3] == 2u);
    CHECK(actualMetadata[4] == 1u);
    CHECK(actualMetadata[5] == 1u);
    CHECK(actualMetadata[6] == 0u);
    CHECK(actualMetadata[7] == 0u);
    CHECK(actualMetadata[8] == 1u);
    CHECK(actualMetadata[9] == kSignedIndexOverflow);

    const auto* offsetStorage =
        static_cast<const int32_t*>(offsetsBuffer.contents);
    const auto* binStorage =
        static_cast<const int32_t*>(binsBuffer.contents);
    const auto* sortableStorage =
        static_cast<const uint32_t*>(sortableBuffer.contents);
    const auto* metadataStorage =
        static_cast<const uint32_t*>(metadataBuffer.contents);
    CHECK(offsetStorage[0] == kPoisonInt);
    CHECK(offsetStorage[offsets.size() - 1] == kPoisonInt);
    CHECK(binStorage[0] == kPoisonInt);
    CHECK(binStorage[bins.size() - 1] == kPoisonInt);
    CHECK(sortableStorage[0] == kPoisonUint);
    CHECK(sortableStorage[sortable.size() - 1] == kPoisonUint);
    CHECK(metadataStorage[0] == kPoisonUint);
    CHECK(metadataStorage[metadata.size() - 1] == kPoisonUint);

    bool cpuRejected = false;
    try {
        std::array<int32_t, counts.size()> cpuOffsets = {};
        (void)msplat::buildTileIntersectionLayout(
            counts.data(), cpuOffsets.data(), counts.size());
    } catch (const std::overflow_error&) {
        cpuRejected = true;
    }
    CHECK(cpuRejected);
}

std::vector<uint32_t> randomizedCounts(size_t count) {
    CHECK(count >= 12);
    std::mt19937 generator(0x4D53504Cu);
    std::uniform_int_distribution<uint32_t> distribution(0, 4'096);
    std::vector<uint32_t> result(count);
    std::generate(result.begin(), result.end(), [&] {
        return distribution(generator);
    });
    const std::array<uint32_t, 12> boundaries = {
        0u, 1u, 2u, 31u, 32u, 33u,
        2'047u, 2'048u, 2'049u, 65'535u, 65'536u, 65'537u,
    };
    std::copy(boundaries.begin(), boundaries.end(), result.begin());
    return result;
}

void checkGpuTileLayout(id<MTLDevice> device, const char* metallibPath) {
    NSString* path = [NSString stringWithUTF8String:metallibPath];
    CHECK(path != nil);
    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if (!library) throw metalError("Failed to load the test metallib", error);
    ScopedObjCRelease libraryOwner{library};

    id<MTLFunction> function =
        [library newFunctionWithName:@"build_tile_intersection_layout_kernel"];
    CHECK(function != nil);
    ScopedObjCRelease functionOwner{function};
    error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        throw metalError("Failed to create the GPU tile-layout pipeline", error);
    }
    ScopedObjCRelease pipelineOwner{pipeline};
    CHECK(pipeline.maxTotalThreadsPerThreadgroup >= 1);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    CHECK(queue != nil);
    ScopedObjCRelease queueOwner{queue};

    runValidCase(device, queue, pipeline, {0u});
    runValidCase(device, queue, pipeline, {0u, 3u, 1u, 0u, 4u});
    runValidCase(
        device, queue, pipeline,
        {33u, 2u, 2'049u, 31u, 1u, 2'048u, 3u, 32u, 0u});
    runValidCase(device, queue, pipeline, {5u, 5u, 4u});
    runValidCase(
        device, queue, pipeline,
        {static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) - 1u,
         1u});
    runValidCase(device, queue, pipeline, randomizedCounts(4'097));
    runOverflowCase(device, queue, pipeline);
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
            checkGpuTileLayout(device, argv[1]);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
