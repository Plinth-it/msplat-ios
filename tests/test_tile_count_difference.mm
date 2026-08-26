#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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

struct ScopedObjCRelease {
    id object = nil;

    ~ScopedObjCRelease() {
        [object release];
    }
};

struct TileRectangle {
    uint32_t minX;
    uint32_t minY;
    uint32_t maxX;
    uint32_t maxY;
};

struct CountCase {
    uint32_t width;
    uint32_t height;
    std::vector<TileRectangle> rectangles;
    std::vector<uint8_t> coverage;
    uint32_t coverageStride;
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

id<MTLComputePipelineState> makePipeline(
    id<MTLDevice> device, id<MTLLibrary> library, NSString* functionName) {
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    CHECK(function != nil);
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (!pipeline) {
        throw metalError("Failed to create a tile-count scan pipeline", error);
    }
    return pipeline;
}

std::vector<int32_t> makeDifferenceGrid(const CountCase& testCase) {
    CHECK(testCase.width > 0);
    CHECK(testCase.height > 0);
    const size_t rowStride = static_cast<size_t>(testCase.width) + 1;
    std::vector<int32_t> difference(
        rowStride * (static_cast<size_t>(testCase.height) + 1), 0);

    for (const TileRectangle& rectangle : testCase.rectangles) {
        CHECK(rectangle.minX < rectangle.maxX);
        CHECK(rectangle.minY < rectangle.maxY);
        CHECK(rectangle.maxX <= testCase.width);
        CHECK(rectangle.maxY <= testCase.height);
        ++difference[rectangle.minY * rowStride + rectangle.minX];
        --difference[rectangle.minY * rowStride + rectangle.maxX];
        --difference[rectangle.maxY * rowStride + rectangle.minX];
        ++difference[rectangle.maxY * rowStride + rectangle.maxX];
    }
    return difference;
}

std::vector<int32_t> horizontalReference(
    const CountCase& testCase, std::vector<int32_t> difference) {
    const size_t rowStride = static_cast<size_t>(testCase.width) + 1;
    for (uint32_t y = 0; y <= testCase.height; ++y) {
        int32_t running = 0;
        for (uint32_t x = 0; x <= testCase.width; ++x) {
            running += difference[static_cast<size_t>(y) * rowStride + x];
            difference[static_cast<size_t>(y) * rowStride + x] = running;
        }
    }
    return difference;
}

bool tileIsCovered(const CountCase& testCase, uint32_t x, uint32_t y) {
    if (testCase.coverageStride == 0) return true;
    CHECK(testCase.coverageStride >= testCase.width);
    CHECK(testCase.coverage.size() >=
          static_cast<size_t>(testCase.coverageStride) * testCase.height);
    return testCase.coverage[
        static_cast<size_t>(y) * testCase.coverageStride + x] != 0;
}

std::vector<uint32_t> enumeratedReference(const CountCase& testCase) {
    std::vector<uint32_t> counts(
        static_cast<size_t>(testCase.width) * testCase.height, 0);
    for (const TileRectangle& rectangle : testCase.rectangles) {
        for (uint32_t y = rectangle.minY; y < rectangle.maxY; ++y) {
            for (uint32_t x = rectangle.minX; x < rectangle.maxX; ++x) {
                if (tileIsCovered(testCase, x, y)) {
                    ++counts[static_cast<size_t>(y) * testCase.width + x];
                }
            }
        }
    }
    return counts;
}

void runCase(
    id<MTLDevice> device, id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> horizontalPipeline,
    id<MTLComputePipelineState> verticalPipeline,
    const CountCase& testCase) {
    std::vector<int32_t> difference = makeDifferenceGrid(testCase);
    const std::vector<int32_t> expectedHorizontal =
        horizontalReference(testCase, difference);
    const std::vector<uint32_t> expectedCounts =
        enumeratedReference(testCase);
    std::vector<uint32_t> output(expectedCounts.size(),
                                 std::numeric_limits<uint32_t>::max());
    const std::array<uint32_t, 4> tileBounds = {
        testCase.width, testCase.height, 1, 0,
    };
    const uint8_t disabledCoverage = 1;
    const void* coverageBytes = testCase.coverageStride == 0
        ? static_cast<const void*>(&disabledCoverage)
        : static_cast<const void*>(testCase.coverage.data());
    const size_t coverageByteCount = testCase.coverageStride == 0
        ? sizeof(disabledCoverage)
        : testCase.coverage.size();

    id<MTLBuffer> differenceBuffer = makeBuffer(
        device, difference.data(), difference.size() * sizeof(difference[0]));
    ScopedObjCRelease differenceOwner{differenceBuffer};
    id<MTLBuffer> outputBuffer = makeBuffer(
        device, output.data(), output.size() * sizeof(output[0]));
    ScopedObjCRelease outputOwner{outputBuffer};
    id<MTLBuffer> coverageBuffer = makeBuffer(
        device, coverageBytes, coverageByteCount);
    ScopedObjCRelease coverageOwner{coverageBuffer};

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    CHECK(commandBuffer != nil);
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    CHECK(encoder != nil);

    [encoder setComputePipelineState:horizontalPipeline];
    [encoder setBuffer:differenceBuffer offset:0 atIndex:0];
    [encoder setBytes:tileBounds.data()
                length:sizeof(tileBounds)
               atIndex:1];
    const NSUInteger horizontalThreads = testCase.height + 1;
    [encoder dispatchThreads:MTLSizeMake(horizontalThreads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(
             std::min(horizontalThreads,
                      horizontalPipeline.maxTotalThreadsPerThreadgroup),
             1, 1)];

    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:verticalPipeline];
    [encoder setBuffer:differenceBuffer offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    [encoder setBytes:tileBounds.data()
                length:sizeof(tileBounds)
               atIndex:2];
    [encoder setBuffer:coverageBuffer offset:0 atIndex:3];
    [encoder setBytes:&testCase.coverageStride
                length:sizeof(testCase.coverageStride)
               atIndex:4];
    const NSUInteger verticalThreads = testCase.width;
    [encoder dispatchThreads:MTLSizeMake(verticalThreads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(
             std::min(verticalThreads,
                      verticalPipeline.maxTotalThreadsPerThreadgroup),
             1, 1)];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) {
        throw metalError(
            "Tile-count scan command buffer failed", commandBuffer.error);
    }
    CHECK(commandBuffer.status == MTLCommandBufferStatusCompleted);

    const auto* horizontalResult =
        static_cast<const int32_t*>(differenceBuffer.contents);
    CHECK(std::equal(expectedHorizontal.begin(), expectedHorizontal.end(),
                     horizontalResult));
    const auto* countResult =
        static_cast<const uint32_t*>(outputBuffer.contents);
    CHECK(std::equal(expectedCounts.begin(), expectedCounts.end(), countResult));
}

std::vector<TileRectangle> randomizedRectangles(
    uint32_t width, uint32_t height, size_t count) {
    std::mt19937 generator(0x4D53504Cu);
    std::uniform_int_distribution<uint32_t> xDistribution(0, width - 1);
    std::uniform_int_distribution<uint32_t> yDistribution(0, height - 1);
    std::uniform_int_distribution<uint32_t> xEndDistribution(1, width);
    std::uniform_int_distribution<uint32_t> yEndDistribution(1, height);
    std::vector<TileRectangle> rectangles;
    rectangles.reserve(count + 2);
    rectangles.push_back({0, 0, width, height});
    rectangles.push_back({0, 0, width, height});
    while (rectangles.size() < count + 2) {
        const uint32_t x0 = xDistribution(generator);
        const uint32_t y0 = yDistribution(generator);
        const uint32_t x1 = xEndDistribution(generator);
        const uint32_t y1 = yEndDistribution(generator);
        if (x0 < x1 && y0 < y1) {
            rectangles.push_back({x0, y0, x1, y1});
        }
    }
    return rectangles;
}

void checkTileCountDifference(id<MTLDevice> device, const char* metallibPath) {
    NSString* path = [NSString stringWithUTF8String:metallibPath];
    CHECK(path != nil);
    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if (!library) throw metalError("Failed to load the test metallib", error);
    ScopedObjCRelease libraryOwner{library};

    id<MTLComputePipelineState> horizontalPipeline = makePipeline(
        device, library, @"tile_count_diff_horizontal_kernel");
    ScopedObjCRelease horizontalOwner{horizontalPipeline};
    id<MTLComputePipelineState> verticalPipeline = makePipeline(
        device, library, @"tile_count_diff_vertical_kernel");
    ScopedObjCRelease verticalOwner{verticalPipeline};
    id<MTLCommandQueue> queue = [device newCommandQueue];
    CHECK(queue != nil);
    ScopedObjCRelease queueOwner{queue};

    runCase(device, queue, horizontalPipeline, verticalPipeline,
            CountCase{1, 1, {{0, 0, 1, 1}, {0, 0, 1, 1}}, {}, 0});
    runCase(device, queue, horizontalPipeline, verticalPipeline,
            CountCase{5, 3, {}, {}, 0});

    const std::vector<TileRectangle> edgeAndOverlap = {
        {0, 0, 7, 5}, {0, 0, 1, 1}, {6, 4, 7, 5},
        {0, 2, 7, 3}, {3, 0, 4, 5}, {1, 1, 6, 4},
        {1, 1, 6, 4}, {0, 0, 7, 1}, {0, 4, 7, 5},
    };
    std::vector<uint8_t> checkerCoverage(7 * 5);
    for (uint32_t y = 0; y < 5; ++y) {
        for (uint32_t x = 0; x < 7; ++x) {
            checkerCoverage[y * 7 + x] = ((x + 2 * y) % 3) == 0 ? 0 : 1;
        }
    }
    runCase(device, queue, horizontalPipeline, verticalPipeline,
            CountCase{7, 5, edgeAndOverlap, checkerCoverage, 7});

    constexpr uint32_t randomWidth = 37;
    constexpr uint32_t randomHeight = 23;
    std::vector<uint8_t> randomCoverage(randomWidth * randomHeight);
    for (uint32_t y = 0; y < randomHeight; ++y) {
        for (uint32_t x = 0; x < randomWidth; ++x) {
            randomCoverage[y * randomWidth + x] =
                ((x * 17 + y * 29 + 3) % 11) < 8 ? 1 : 0;
        }
    }
    runCase(
        device, queue, horizontalPipeline, verticalPipeline,
        CountCase{
            randomWidth, randomHeight,
            randomizedRectangles(randomWidth, randomHeight, 512),
            randomCoverage, randomWidth});

    std::vector<uint8_t> noCoverage(randomWidth * randomHeight, 0);
    runCase(
        device, queue, horizontalPipeline, verticalPipeline,
        CountCase{
            randomWidth, randomHeight,
            randomizedRectangles(randomWidth, randomHeight, 64),
            noCoverage, randomWidth});
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
            checkTileCountDifference(device, argv[1]);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
