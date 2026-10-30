#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Checks the exact raster footprint shared by tile counting and scatter: every
// pixel the rasterizers can blend must lie in an emitted tile. The probe kernel
// is compiled together with the production shader source so it calls the same
// footprint helpers that both intersection passes use.

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

constexpr uint32_t kTileSize = 16;
constexpr uint32_t kWidth = 160;
constexpr uint32_t kHeight = 96;
constexpr uint32_t kTilesX = kWidth / kTileSize;
constexpr uint32_t kTilesY = kHeight / kTileSize;
constexpr uint32_t kMaskWords = (kTilesX * kTilesY + 31) / 32;

constexpr const char* kProbeSource = R"(
kernel void raster_footprint_probe_kernel(
    constant float* centers   [[buffer(0)]],
    constant float* conics    [[buffer(1)]],
    constant float* opacities [[buffer(2)]],
    constant uint2& tiles     [[buffer(3)]],
    device uint* masks        [[buffer(4)]],
    constant uint& count      [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= count) return;
    const float2 center = float2(centers[2 * idx], centers[2 * idx + 1]);
    const float3 conic = float3(
        conics[3 * idx], conics[3 * idx + 1], conics[3 * idx + 2]);
    const float2 extents = raster_footprint_extents(conic, opacities[idx]);
    const uint words = (tiles.x * tiles.y + 31u) / 32u;
    device uint* mask = masks + idx * words;
    uint row_begin, row_end;
    if (!raster_footprint_rows(center, extents, tiles.y, row_begin, row_end)) {
        return;
    }
    for (uint tile_y = row_begin; tile_y < row_end; ++tile_y) {
        uint column_begin, column_end;
        if (!raster_footprint_row_span(
                center, conic, extents, tile_y, tiles.x,
                column_begin, column_end)) {
            continue;
        }
        for (uint tile_x = column_begin; tile_x < column_end; ++tile_x) {
            const uint bit = tile_y * tiles.x + tile_x;
            mask[bit / 32u] |= 1u << (bit % 32u);
        }
    }
}
)";

struct Footprint {
    float centerX;
    float centerY;
    float conicA;
    float conicB;
    float conicC;
    float opacity;
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

Footprint fromCovariance(float centerX, float centerY, double sigmaMajor,
                         double sigmaMinor, double angle, float opacity) {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    // Projection adds the same 0.3 px^2 low-pass dilation.
    const double xx = c * c * sigmaMajor * sigmaMajor +
        s * s * sigmaMinor * sigmaMinor + 0.3;
    const double xy = c * s * (sigmaMajor * sigmaMajor -
        sigmaMinor * sigmaMinor);
    const double yy = s * s * sigmaMajor * sigmaMajor +
        c * c * sigmaMinor * sigmaMinor + 0.3;
    const double det = xx * yy - xy * xy;
    return Footprint{
        centerX, centerY,
        static_cast<float>(yy / det), static_cast<float>(-xy / det),
        static_cast<float>(xx / det), opacity};
}

std::vector<Footprint> makeFootprints() {
    std::mt19937 random(20261030u);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<Footprint> footprints;
    const auto randomCenter = [&](double extent) {
        return static_cast<float>(-48.0 + unit(random) * (extent + 96.0));
    };
    for (int index = 0; index < 20'000; ++index) {
        const double sigmaMajor = 0.3 * std::pow(200.0, unit(random));
        const double sigmaMinor = 0.3 * std::pow(200.0, unit(random));
        const double choice = unit(random);
        float opacity;
        if (choice < 0.25) {
            // Straddle the 1/255 blending threshold.
            opacity = static_cast<float>((0.5 + 1.5 * unit(random)) / 255.0);
        } else if (choice < 0.5) {
            // Opaque Gaussians reach the 5.55 sigma cutoff past 3 sigma.
            opacity = static_cast<float>(0.9 + 0.0999 * unit(random));
        } else {
            opacity = static_cast<float>(std::max(1e-4, unit(random)));
        }
        footprints.push_back(fromCovariance(
            randomCenter(kWidth), randomCenter(kHeight),
            sigmaMajor, sigmaMinor, 6.283185307179586 * unit(random),
            opacity));
    }
    // Centers on and beside tile and pixel boundaries, extreme anisotropy,
    // a screen-filling Gaussian, and the exact threshold opacity.
    for (float center : {15.5f, 16.0f, 16.5f, 31.999f, 48.0f}) {
        footprints.push_back(fromCovariance(
            center, center, 6.0, 0.05, 0.3, 0.95f));
        footprints.push_back(fromCovariance(
            center, center + 0.5f, 0.05, 6.0, 1.1, 0.5f));
    }
    footprints.push_back(fromCovariance(80.0f, 48.0f, 400.0, 300.0, 0.7, 0.99f));
    footprints.push_back(fromCovariance(40.0f, 40.0f, 3.0, 2.0, 0.0, 1.0f / 255.0f));
    return footprints;
}

// The rasterizers' acceptance test, with a small tolerance so transcendental
// rounding differences between CPU and GPU cannot flag a boundary pixel.
bool pixelBlends(const Footprint& footprint, float pixelX, float pixelY,
                 float tolerance) {
    const float dx = footprint.centerX - pixelX;
    const float dy = footprint.centerY - pixelY;
    const float sigma = std::fma(
        0.5f,
        std::fma(footprint.conicA, dx * dx, footprint.conicC * dy * dy),
        footprint.conicB * dx * dy);
    if (sigma < 0.0f || sigma >= 5.55f * (1.0f - tolerance)) return false;
    const float alpha = std::min(
        0.999f, footprint.opacity * std::exp(-sigma));
    return alpha >= (1.0f / 255.0f) * (1.0f + tolerance);
}

bool maskHas(const uint32_t* mask, uint32_t tileX, uint32_t tileY) {
    const uint32_t bit = tileY * kTilesX + tileX;
    return (mask[bit / 32] >> (bit % 32)) & 1u;
}

void checkRasterFootprint(id<MTLDevice> device, const char* sourcePath) {
    std::ifstream sourceFile(sourcePath);
    CHECK(sourceFile.good());
    std::string source(
        (std::istreambuf_iterator<char>(sourceFile)),
        std::istreambuf_iterator<char>());
    source += kProbeSource;

    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    ScopedObjCRelease optionsOwner{options};
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                     options:options
                       error:&error];
    if (!library) throw metalError("Failed to compile the probe shader", error);
    ScopedObjCRelease libraryOwner{library};
    id<MTLFunction> function =
        [library newFunctionWithName:@"raster_footprint_probe_kernel"];
    CHECK(function != nil);
    ScopedObjCRelease functionOwner{function};
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) throw metalError("Failed to create the probe pipeline", error);
    ScopedObjCRelease pipelineOwner{pipeline};
    id<MTLCommandQueue> queue = [device newCommandQueue];
    CHECK(queue != nil);
    ScopedObjCRelease queueOwner{queue};

    const std::vector<Footprint> footprints = makeFootprints();
    const uint32_t count = static_cast<uint32_t>(footprints.size());
    std::vector<float> centers, conics, opacities;
    for (const Footprint& footprint : footprints) {
        centers.insert(centers.end(), {footprint.centerX, footprint.centerY});
        conics.insert(conics.end(),
                      {footprint.conicA, footprint.conicB, footprint.conicC});
        opacities.push_back(footprint.opacity);
    }
    const auto makeBuffer = [&](const void* bytes, size_t length) {
        id<MTLBuffer> buffer = [device
            newBufferWithBytes:bytes length:length
                       options:MTLResourceStorageModeShared];
        CHECK(buffer != nil);
        return buffer;
    };
    id<MTLBuffer> centerBuffer =
        makeBuffer(centers.data(), centers.size() * sizeof(float));
    ScopedObjCRelease centerOwner{centerBuffer};
    id<MTLBuffer> conicBuffer =
        makeBuffer(conics.data(), conics.size() * sizeof(float));
    ScopedObjCRelease conicOwner{conicBuffer};
    id<MTLBuffer> opacityBuffer =
        makeBuffer(opacities.data(), opacities.size() * sizeof(float));
    ScopedObjCRelease opacityOwner{opacityBuffer};
    const std::vector<uint32_t> zeroMasks(
        static_cast<size_t>(count) * kMaskWords, 0u);
    id<MTLBuffer> maskBuffer =
        makeBuffer(zeroMasks.data(), zeroMasks.size() * sizeof(uint32_t));
    ScopedObjCRelease maskOwner{maskBuffer};

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:centerBuffer offset:0 atIndex:0];
    [encoder setBuffer:conicBuffer offset:0 atIndex:1];
    [encoder setBuffer:opacityBuffer offset:0 atIndex:2];
    const uint32_t tiles[2] = {kTilesX, kTilesY};
    [encoder setBytes:tiles length:sizeof(tiles) atIndex:3];
    [encoder setBuffer:maskBuffer offset:0 atIndex:4];
    [encoder setBytes:&count length:sizeof(count) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted);

    const uint32_t* masks =
        static_cast<const uint32_t*>(maskBuffer.contents);
    uint64_t emittedTiles = 0;
    uint64_t blendingTiles = 0;
    uint64_t missedTiles = 0;
    uint64_t threeSigmaMissedTiles = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const Footprint& footprint = footprints[index];
        const uint32_t* mask = masks + static_cast<size_t>(index) * kMaskWords;
        const double det = double(footprint.conicA) * footprint.conicC -
            double(footprint.conicB) * footprint.conicB;
        const double covXX = footprint.conicC / det;
        const double covYY = footprint.conicA / det;
        const double covXY = -footprint.conicB / det;
        const double mid = 0.5 * (covXX + covYY);
        const double lambda = mid + std::sqrt(
            std::max(0.0, mid * mid - (covXX * covYY - covXY * covXY)));
        // Past sqrt(11.1 * lambda_max) every pixel has sigma >= 5.55.
        const double reach = std::sqrt(11.1 * lambda) + 2.0;
        // Previous footprint: per-axis 3-sigma box mapped to tiles.
        const double boxX = std::ceil(3.0 * std::sqrt(covXX));
        const double boxY = std::ceil(3.0 * std::sqrt(covYY));
        const int boxMinX = std::max(0, int(footprint.centerX / 16.0 - boxX / 16.0));
        const int boxMaxX = std::min(int(kTilesX), int(footprint.centerX / 16.0 + boxX / 16.0 + 1.0));
        const int boxMinY = std::max(0, int(footprint.centerY / 16.0 - boxY / 16.0));
        const int boxMaxY = std::min(int(kTilesY), int(footprint.centerY / 16.0 + boxY / 16.0 + 1.0));

        for (uint32_t tileY = 0; tileY < kTilesY; ++tileY) {
            for (uint32_t tileX = 0; tileX < kTilesX; ++tileX) {
                const bool emitted = maskHas(mask, tileX, tileY);
                emittedTiles += emitted;
                bool blends = false;
                for (uint32_t y = tileY * kTileSize;
                     !blends && y < (tileY + 1) * kTileSize; ++y) {
                    if (std::abs(double(y) - footprint.centerY) > reach) continue;
                    for (uint32_t x = tileX * kTileSize;
                         x < (tileX + 1) * kTileSize; ++x) {
                        if (std::abs(double(x) - footprint.centerX) > reach) continue;
                        if (pixelBlends(footprint, float(x), float(y), 1e-4f)) {
                            blends = true;
                            break;
                        }
                    }
                }
                blendingTiles += blends;
                if (blends && !emitted) {
                    ++missedTiles;
                    std::cerr << "missed tile (" << tileX << ", " << tileY
                              << ") for Gaussian " << index << '\n';
                }
                const bool inBox = int(tileX) >= boxMinX &&
                    int(tileX) < boxMaxX && int(tileY) >= boxMinY &&
                    int(tileY) < boxMaxY;
                if (blends && !inBox) ++threeSigmaMissedTiles;
            }
        }
    }
    std::cout << "exact footprint: " << emittedTiles << " tiles emitted, "
              << blendingTiles << " contain a blending pixel; the 3-sigma box "
              << "missed " << threeSigmaMissedTiles << '\n';
    CHECK(missedTiles == 0);
    CHECK(blendingTiles > 0);
    // Slack comes only from ellipses that pass between pixel centers.
    CHECK(emittedTiles <= blendingTiles + blendingTiles / 50);
    // The threshold-opacity Gaussian reaches alpha 1/255 only at its center.
    const uint32_t* thresholdMask =
        masks + static_cast<size_t>(count - 1) * kMaskWords;
    uint32_t thresholdTiles = 0;
    for (uint32_t word = 0; word < kMaskWords; ++word) {
        thresholdTiles += __builtin_popcount(thresholdMask[word]);
    }
    CHECK(thresholdTiles <= 1);
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 2) {
                throw std::invalid_argument("Expected the Metal source path");
            }
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                std::cerr << "SKIP: msplat: no Metal device is available\n";
                return 77;
            }
            checkRasterFootprint(device, argv[1]);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
