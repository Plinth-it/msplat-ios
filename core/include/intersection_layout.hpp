#ifndef MSPLAT_INTERSECTION_LAYOUT_H
#define MSPLAT_INTERSECTION_LAYOUT_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace msplat {

inline constexpr uint32_t kExactBitonicFastPath = 2'048;
inline constexpr uint32_t kExactRadixScratchBytesPerEntry = sizeof(uint64_t);
inline constexpr uint32_t kExactIntersectionBytesPerEntry =
    2 * sizeof(uint64_t) + 3 * 3 * sizeof(float);
inline constexpr uint32_t kExactBitonicOnlyIntersectionBytesPerEntry =
    kExactIntersectionBytesPerEntry - kExactRadixScratchBytesPerEntry;
inline constexpr uint32_t kExactTileMetadataBytes =
    sizeof(uint32_t) + sizeof(int32_t) + 2 * sizeof(int32_t);
// The current large-tile sorter assigns one threadgroup to a tile. Keep that
// work bounded so a pathological screen-filling population fails before arena
// allocation or optimizer submission instead of risking the GPU watchdog.
inline constexpr uint32_t kMaximumExactIntersectionsPerTile = 65'536;

struct TileIntersectionLayout {
    uint32_t totalCount = 0;
    uint32_t maximumTileCount = 0;
    size_t maximumTileIndex = 0;
    uint32_t activeTileCount = 0;
    uint32_t trivialTileCount = 0;
    uint32_t smallTileCount = 0;
    uint32_t mediumTileCount = 0;
    uint32_t largeTileCount = 0;
};

inline bool tileIntersectionLayoutNeedsRadixScratch(
    const TileIntersectionLayout& layout) {
    return layout.maximumTileCount > kExactBitonicFastPath;
}

/// Converts exact per-tile counts into inclusive offsets for the packed arena.
/// The native rasterizer uses signed 32-bit ranges, so a larger scene is
/// rejected before any packed buffers or optimizer work are submitted.
inline TileIntersectionLayout buildTileIntersectionLayout(
    const uint32_t* counts, int32_t* inclusiveOffsets, size_t tileCount) {
    if (tileCount > 0 && (!counts || !inclusiveOffsets)) {
        throw std::invalid_argument(
            "Tile-intersection counts and offsets must not be null");
    }

    TileIntersectionLayout layout;
    uint64_t running = 0;
    for (size_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t count = counts[tile];
        running += count;
        if (running > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::overflow_error(
                "Exact tile-intersection count exceeds the native index range");
        }
        inclusiveOffsets[tile] = static_cast<int32_t>(running);
        if (count > 0) ++layout.activeTileCount;
        if (count <= 1) {
            ++layout.trivialTileCount;
        } else if (count <= 32) {
            ++layout.smallTileCount;
        } else if (count <= kExactBitonicFastPath) {
            ++layout.mediumTileCount;
        } else {
            ++layout.largeTileCount;
        }
        if (count > layout.maximumTileCount) {
            layout.maximumTileCount = count;
            layout.maximumTileIndex = tile;
        }
    }
    layout.totalCount = static_cast<uint32_t>(running);
    return layout;
}

inline void validateTileIntersectionWorkLimit(
    const TileIntersectionLayout& layout,
    uint32_t maximumPerTile = kMaximumExactIntersectionsPerTile) {
    if (maximumPerTile == 0) {
        throw std::invalid_argument(
            "Maximum exact intersections per tile must be greater than zero");
    }
    if (layout.maximumTileCount > maximumPerTile) {
        throw std::length_error(
            "A tile exceeds the exact-sort work limit; reduce the training "
            "resolution or Gaussian budget");
    }
}

/// Returns a grow-only arena capacity for one resolution. A resolution change
/// deliberately discards the previous high-water mark so a coarse or unrelated
/// render does not pin an oversized arena indefinitely.
inline uint32_t tileIntersectionArenaCapacity(
    uint32_t requiredCount, uint32_t currentCapacity,
    bool resolutionChanged) {
    if (requiredCount >
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        throw std::overflow_error(
            "Exact tile-intersection arena exceeds the native index range");
    }
    if (!resolutionChanged && currentCapacity >= requiredCount &&
        currentCapacity > 0) {
        return currentCapacity;
    }
    if (requiredCount == 0) return 1;

    const uint64_t slack = std::max<uint64_t>(requiredCount / 4, 4'096);
    return static_cast<uint32_t>(std::min<uint64_t>(
        static_cast<uint64_t>(requiredCount) + slack,
        static_cast<uint64_t>(std::numeric_limits<int32_t>::max())));
}

inline uint32_t tileRasterChunkCount(
    uint32_t tileCount, uint32_t maximumTileCount,
    uint32_t chunkSize = 512, uint32_t monolithicTileThreshold = 400) {
    if (chunkSize == 0) {
        throw std::invalid_argument("Raster chunk size must be greater than zero");
    }
    if (tileCount >= monolithicTileThreshold) return 1;
    return static_cast<uint32_t>(std::max<uint64_t>(
        1, (static_cast<uint64_t>(maximumTileCount) + chunkSize - 1) /
               chunkSize));
}

} // namespace msplat

#endif // MSPLAT_INTERSECTION_LAYOUT_H
