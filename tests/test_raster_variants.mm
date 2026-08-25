#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
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

struct RasterOutput {
    std::vector<float> finalT;
    std::vector<int32_t> finalIndex;
    std::vector<float> rgb;
};

struct RasterVariant {
    const char* functionName;
    uint32_t width;
    uint32_t height;
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
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:bytes
                            length:byteCount
                           options:MTLResourceStorageModeShared];
    if (!buffer) {
        throw std::runtime_error("Failed to allocate a Metal test buffer");
    }
    return buffer;
}

bool nearlyEqual(float lhs, float rhs) {
    constexpr float absoluteTolerance = 2e-6f;
    constexpr float relativeTolerance = 2e-6f;
    return std::abs(lhs - rhs) <=
        absoluteTolerance + relativeTolerance * std::abs(lhs);
}

RasterOutput runVariant(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    id<MTLCommandQueue> queue,
    const RasterVariant& variant,
    const std::array<uint32_t, 4>& tileBounds,
    const std::array<uint32_t, 4>& imageSize,
    id<MTLBuffer> tileBins,
    id<MTLBuffer> packedXYOpacity,
    id<MTLBuffer> packedConic,
    id<MTLBuffer> packedRGB,
    id<MTLBuffer> background) {
    NSString* functionName =
        [NSString stringWithUTF8String:variant.functionName];
    CHECK(functionName != nil);
    id<MTLFunction> function =
        [library newFunctionWithName:functionName];
    CHECK(function != nil);
    ScopedObjCRelease functionOwner{function};

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        throw metalError("Failed to create a raster pipeline", error);
    }
    ScopedObjCRelease pipelineOwner{pipeline};

    const NSUInteger threadCount =
        static_cast<NSUInteger>(variant.width) * variant.height;
    CHECK(pipeline.maxTotalThreadsPerThreadgroup >= threadCount);

    const size_t pixelCount =
        static_cast<size_t>(imageSize[0]) * imageSize[1];
    constexpr float floatSentinel = -123.0f;
    constexpr int32_t indexSentinel = std::numeric_limits<int32_t>::min();
    std::vector<float> initialFinalT(pixelCount, floatSentinel);
    std::vector<int32_t> initialFinalIndex(pixelCount, indexSentinel);
    std::vector<float> initialRGB(pixelCount * 3, floatSentinel);

    id<MTLBuffer> finalT = makeBuffer(
        device, initialFinalT.data(), initialFinalT.size() * sizeof(float));
    ScopedObjCRelease finalTOwner{finalT};
    id<MTLBuffer> finalIndex = makeBuffer(
        device, initialFinalIndex.data(),
        initialFinalIndex.size() * sizeof(int32_t));
    ScopedObjCRelease finalIndexOwner{finalIndex};
    id<MTLBuffer> rgb = makeBuffer(
        device, initialRGB.data(), initialRGB.size() * sizeof(float));
    ScopedObjCRelease rgbOwner{rgb};

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    CHECK(commandBuffer != nil);
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    CHECK(encoder != nil);

    const uint32_t channels = 3;
    const std::array<uint32_t, 2> blockDim = {
        variant.width, variant.height,
    };
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:tileBounds.data()
                length:sizeof(tileBounds)
               atIndex:0];
    [encoder setBytes:imageSize.data()
                length:sizeof(imageSize)
               atIndex:1];
    [encoder setBytes:&channels length:sizeof(channels) atIndex:2];
    [encoder setBuffer:tileBins offset:0 atIndex:3];
    [encoder setBuffer:packedXYOpacity offset:0 atIndex:4];
    [encoder setBuffer:packedConic offset:0 atIndex:5];
    [encoder setBuffer:packedRGB offset:0 atIndex:6];
    [encoder setBuffer:finalT offset:0 atIndex:7];
    [encoder setBuffer:finalIndex offset:0 atIndex:8];
    [encoder setBuffer:rgb offset:0 atIndex:9];
    [encoder setBuffer:background offset:0 atIndex:10];
    [encoder setBytes:blockDim.data()
                length:sizeof(blockDim)
               atIndex:11];

    const MTLSize threadgroups = MTLSizeMake(
        (imageSize[0] + variant.width - 1) / variant.width,
        (imageSize[1] + variant.height - 1) / variant.height,
        1);
    const MTLSize threadsPerThreadgroup =
        MTLSizeMake(variant.width, variant.height, 1);
    [encoder dispatchThreadgroups:threadgroups
             threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) {
        throw metalError("Raster command buffer failed", commandBuffer.error);
    }
    CHECK(commandBuffer.status == MTLCommandBufferStatusCompleted);

    RasterOutput output;
    const auto* finalTValues =
        static_cast<const float*>(finalT.contents);
    output.finalT.assign(finalTValues, finalTValues + pixelCount);
    const auto* finalIndexValues =
        static_cast<const int32_t*>(finalIndex.contents);
    output.finalIndex.assign(
        finalIndexValues, finalIndexValues + pixelCount);
    const auto* rgbValues = static_cast<const float*>(rgb.contents);
    output.rgb.assign(rgbValues, rgbValues + pixelCount * 3);

    CHECK(std::find(output.finalT.begin(), output.finalT.end(),
                    floatSentinel) == output.finalT.end());
    CHECK(std::find(output.finalIndex.begin(), output.finalIndex.end(),
                    indexSentinel) == output.finalIndex.end());
    CHECK(std::find(output.rgb.begin(), output.rgb.end(),
                    floatSentinel) == output.rgb.end());
    return output;
}

void checkRasterVariants(id<MTLDevice> device, const char* metallibPath) {
    constexpr uint32_t imageWidth = 35;
    constexpr uint32_t imageHeight = 35;
    constexpr uint32_t parentTileWidth = 16;
    constexpr uint32_t parentTileHeight = 16;
    constexpr uint32_t tileColumns =
        (imageWidth + parentTileWidth - 1) / parentTileWidth;
    constexpr uint32_t tileRows =
        (imageHeight + parentTileHeight - 1) / parentTileHeight;
    constexpr std::array<uint32_t, tileColumns * tileRows> counts = {
        0, 1, 64,
        65, 128, 129,
        256, 257, 3,
    };
    constexpr std::array<RasterVariant, 3> variants = {{
        {"nd_rasterize_forward_kernel", 8, 8},
        {"nd_rasterize_forward_16x8_kernel", 16, 8},
        {"nd_rasterize_forward_16x16_kernel", 16, 16},
    }};
    constexpr std::array<float, 3> backgroundValues = {
        0.07f, 0.11f, 0.19f,
    };

    std::array<int32_t, counts.size() * 2> bins = {};
    size_t totalCount = 0;
    for (size_t tile = 0; tile < counts.size(); ++tile) {
        bins[2 * tile] = static_cast<int32_t>(totalCount);
        totalCount += counts[tile];
        bins[2 * tile + 1] = static_cast<int32_t>(totalCount);
    }
    CHECK(totalCount == 903);

    std::vector<float> packedXYOpacity(totalCount * 3);
    std::vector<float> packedConic(totalCount * 3);
    std::vector<float> packedRGB(totalCount * 3);
    size_t packedIndex = 0;
    for (uint32_t tile = 0; tile < counts.size(); ++tile) {
        const uint32_t tileX = tile % tileColumns;
        const uint32_t tileY = tile / tileColumns;
        const uint32_t firstX = tileX * parentTileWidth;
        const uint32_t firstY = tileY * parentTileHeight;
        const uint32_t pastLastX =
            std::min(firstX + parentTileWidth, imageWidth);
        const uint32_t pastLastY =
            std::min(firstY + parentTileHeight, imageHeight);
        const float centerX =
            0.5f * static_cast<float>(firstX + pastLastX - 1);
        const float centerY =
            0.5f * static_cast<float>(firstY + pastLastY - 1);

        for (uint32_t local = 0; local < counts[tile]; ++local) {
            const float offsetX =
                0.06f * static_cast<float>(static_cast<int>(local % 7) - 3);
            const float offsetY =
                0.05f * static_cast<float>(static_cast<int>(local % 5) - 2);
            packedXYOpacity[3 * packedIndex] = centerX + offsetX;
            packedXYOpacity[3 * packedIndex + 1] = centerY + offsetY;
            packedXYOpacity[3 * packedIndex + 2] =
                0.0060f + 0.0001f * static_cast<float>(local % 5);

            packedConic[3 * packedIndex] =
                0.0018f + 0.00005f * static_cast<float>(local % 3);
            packedConic[3 * packedIndex + 1] =
                0.00004f * static_cast<float>(static_cast<int>(local % 3) - 1);
            packedConic[3 * packedIndex + 2] =
                0.0020f + 0.00005f * static_cast<float>(local % 4);

            packedRGB[3 * packedIndex] =
                -0.32f + 0.025f * static_cast<float>(local % 9);
            packedRGB[3 * packedIndex + 1] =
                -0.18f + 0.020f * static_cast<float>(local % 7);
            packedRGB[3 * packedIndex + 2] =
                0.08f + 0.018f * static_cast<float>(local % 6);
            ++packedIndex;
        }
    }
    CHECK(packedIndex == totalCount);

    NSString* path = [NSString stringWithUTF8String:metallibPath];
    CHECK(path != nil);
    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if (!library) {
        throw metalError("Failed to load the test metallib", error);
    }
    ScopedObjCRelease libraryOwner{library};

    id<MTLBuffer> binsBuffer = makeBuffer(
        device, bins.data(), bins.size() * sizeof(bins[0]));
    ScopedObjCRelease binsBufferOwner{binsBuffer};
    id<MTLBuffer> xyOpacityBuffer = makeBuffer(
        device, packedXYOpacity.data(),
        packedXYOpacity.size() * sizeof(packedXYOpacity[0]));
    ScopedObjCRelease xyOpacityBufferOwner{xyOpacityBuffer};
    id<MTLBuffer> conicBuffer = makeBuffer(
        device, packedConic.data(),
        packedConic.size() * sizeof(packedConic[0]));
    ScopedObjCRelease conicBufferOwner{conicBuffer};
    id<MTLBuffer> rgbBuffer = makeBuffer(
        device, packedRGB.data(), packedRGB.size() * sizeof(packedRGB[0]));
    ScopedObjCRelease rgbBufferOwner{rgbBuffer};
    id<MTLBuffer> backgroundBuffer = makeBuffer(
        device, backgroundValues.data(),
        backgroundValues.size() * sizeof(backgroundValues[0]));
    ScopedObjCRelease backgroundBufferOwner{backgroundBuffer};

    id<MTLCommandQueue> queue = [device newCommandQueue];
    CHECK(queue != nil);
    ScopedObjCRelease queueOwner{queue};

    const std::array<uint32_t, 4> tileBounds = {
        tileColumns, tileRows, 1, 0,
    };
    const std::array<uint32_t, 4> imageSize = {
        imageWidth, imageHeight, 1, 0,
    };
    std::array<RasterOutput, variants.size()> outputs;
    for (size_t index = 0; index < variants.size(); ++index) {
        outputs[index] = runVariant(
            device, library, queue, variants[index], tileBounds, imageSize,
            binsBuffer, xyOpacityBuffer, conicBuffer, rgbBuffer,
            backgroundBuffer);
    }

    const RasterOutput& baseline = outputs[0];
    for (size_t pixel = 0; pixel < baseline.finalT.size(); ++pixel) {
        const uint32_t x = static_cast<uint32_t>(pixel % imageWidth);
        const uint32_t y = static_cast<uint32_t>(pixel / imageWidth);
        const uint32_t tile =
            (y / parentTileHeight) * tileColumns + (x / parentTileWidth);
        CHECK(baseline.finalIndex[pixel] == bins[2 * tile + 1] - 1);
        CHECK(std::isfinite(baseline.finalT[pixel]));
        CHECK(baseline.finalT[pixel] > 0.0f);
        CHECK(baseline.finalT[pixel] <= 1.0f);
        for (size_t channel = 0; channel < 3; ++channel) {
            const float value = baseline.rgb[3 * pixel + channel];
            CHECK(std::isfinite(value));
            CHECK(value >= 0.0f);
            CHECK(value <= 1.0f);
        }
    }

    for (size_t variant = 1; variant < outputs.size(); ++variant) {
        CHECK(outputs[variant].finalIndex == baseline.finalIndex);
        CHECK(outputs[variant].finalT.size() == baseline.finalT.size());
        CHECK(outputs[variant].rgb.size() == baseline.rgb.size());
        for (size_t index = 0; index < baseline.finalT.size(); ++index) {
            CHECK(nearlyEqual(outputs[variant].finalT[index],
                              baseline.finalT[index]));
        }
        for (size_t index = 0; index < baseline.rgb.size(); ++index) {
            CHECK(nearlyEqual(outputs[variant].rgb[index],
                              baseline.rgb[index]));
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
            checkRasterVariants(device, argv[1]);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
