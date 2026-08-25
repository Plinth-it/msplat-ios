#ifndef MSPLAT_INTERSECTION_LAYOUT_H
#define MSPLAT_INTERSECTION_LAYOUT_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace msplat {

inline constexpr uint32_t kExactBitonicFastPath = 2'048;
inline constexpr uint32_t kExactSmallTileMaximum = 32;
inline constexpr uint32_t kExactRadixScratchBytesPerEntry = sizeof(uint64_t);
inline constexpr uint32_t kExactIntersectionBytesPerEntry =
    2 * sizeof(uint64_t) + 3 * 3 * sizeof(float);
inline constexpr uint32_t kExactBitonicOnlyIntersectionBytesPerEntry =
    kExactIntersectionBytesPerEntry - kExactRadixScratchBytesPerEntry;
inline constexpr uint32_t kExactTileMetadataBytes =
    2 * sizeof(uint32_t) + sizeof(int32_t) + 2 * sizeof(int32_t);
// The current large-tile sorter assigns one threadgroup to a tile. Keep that
// work bounded so a pathological screen-filling population fails before arena
// allocation or optimizer submission instead of risking the GPU watchdog.
inline constexpr uint32_t kMaximumExactIntersectionsPerTile = 65'536;

// GPU layout metadata mirrors TileIntersectionLayout as a compact uint32 array.
// Keep these word offsets in sync with build_tile_intersection_layout_kernel.
inline constexpr size_t kTileIntersectionLayoutMetadataWordCount = 10;
inline constexpr size_t kTileIntersectionLayoutTotalCountWord = 0;
inline constexpr size_t kTileIntersectionLayoutMaximumTileCountWord = 1;
inline constexpr size_t kTileIntersectionLayoutMaximumTileIndexWord = 2;
inline constexpr size_t kTileIntersectionLayoutActiveTileCountWord = 3;
inline constexpr size_t kTileIntersectionLayoutSortableTileCountWord = 4;
inline constexpr size_t kTileIntersectionLayoutTrivialTileCountWord = 5;
inline constexpr size_t kTileIntersectionLayoutSmallTileCountWord = 6;
inline constexpr size_t kTileIntersectionLayoutMediumTileCountWord = 7;
inline constexpr size_t kTileIntersectionLayoutLargeTileCountWord = 8;
inline constexpr size_t kTileIntersectionLayoutErrorFlagsWord = 9;
inline constexpr uint32_t kTileIntersectionLayoutSignedIndexOverflow = 1u << 0;

struct TileIntersectionLayout {
    uint32_t totalCount = 0;
    uint32_t maximumTileCount = 0;
    size_t maximumTileIndex = 0;
    uint32_t activeTileCount = 0;
    uint32_t sortableTileCount = 0;
    uint32_t trivialTileCount = 0;
    uint32_t smallTileCount = 0;
    uint32_t mediumTileCount = 0;
    uint32_t largeTileCount = 0;
};

/// Converts completed GPU layout metadata back into the checked host layout.
/// The final inclusive offset is validated independently so corrupt or stale
/// metadata cannot determine an arena allocation or dispatch size.
inline TileIntersectionLayout tileIntersectionLayoutFromGpuMetadata(
    const uint32_t* metadata, size_t metadataWordCount, size_t tileCount,
    const int32_t* inclusiveOffsets) {
    if (!metadata || metadataWordCount < kTileIntersectionLayoutMetadataWordCount) {
        throw std::invalid_argument(
            "GPU tile-intersection layout metadata is missing or truncated");
    }
    if (tileCount > 0 && !inclusiveOffsets) {
        throw std::invalid_argument(
            "GPU tile-intersection offsets must not be null");
    }
    if (metadata[kTileIntersectionLayoutErrorFlagsWord] &
        kTileIntersectionLayoutSignedIndexOverflow) {
        throw std::overflow_error(
            "Exact tile-intersection count exceeds the native index range");
    }
    if (metadata[kTileIntersectionLayoutErrorFlagsWord] != 0) {
        throw std::runtime_error(
            "GPU tile-intersection layout reported an unknown error");
    }

    TileIntersectionLayout layout;
    layout.totalCount = metadata[kTileIntersectionLayoutTotalCountWord];
    layout.maximumTileCount =
        metadata[kTileIntersectionLayoutMaximumTileCountWord];
    layout.maximumTileIndex =
        metadata[kTileIntersectionLayoutMaximumTileIndexWord];
    layout.activeTileCount =
        metadata[kTileIntersectionLayoutActiveTileCountWord];
    layout.sortableTileCount =
        metadata[kTileIntersectionLayoutSortableTileCountWord];
    layout.trivialTileCount =
        metadata[kTileIntersectionLayoutTrivialTileCountWord];
    layout.smallTileCount =
        metadata[kTileIntersectionLayoutSmallTileCountWord];
    layout.mediumTileCount =
        metadata[kTileIntersectionLayoutMediumTileCountWord];
    layout.largeTileCount =
        metadata[kTileIntersectionLayoutLargeTileCountWord];

    if (layout.totalCount >
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        throw std::overflow_error(
            "Exact tile-intersection count exceeds the native index range");
    }
    const uint64_t classifiedTiles =
        static_cast<uint64_t>(layout.trivialTileCount) +
        layout.sortableTileCount;
    const uint64_t sortableTiles =
        static_cast<uint64_t>(layout.smallTileCount) +
        layout.mediumTileCount + layout.largeTileCount;
    if (classifiedTiles != tileCount ||
        sortableTiles != layout.sortableTileCount ||
        layout.activeTileCount > tileCount ||
        layout.activeTileCount < layout.sortableTileCount ||
        layout.activeTileCount > layout.totalCount ||
        (layout.maximumTileCount > 0 &&
         layout.maximumTileIndex >= tileCount) ||
        (layout.maximumTileCount == 0 && layout.totalCount != 0) ||
        (layout.maximumTileCount > layout.totalCount) ||
        (layout.largeTileCount == 0 &&
         layout.maximumTileCount > kExactBitonicFastPath) ||
        (layout.largeTileCount > 0 &&
         layout.maximumTileCount <= kExactBitonicFastPath)) {
        throw std::runtime_error(
            "GPU tile-intersection layout metadata is inconsistent");
    }

    const int32_t finalOffset = tileCount > 0
        ? inclusiveOffsets[tileCount - 1]
        : 0;
    if (finalOffset < 0 ||
        static_cast<uint32_t>(finalOffset) != layout.totalCount) {
        throw std::runtime_error(
            "GPU tile-intersection layout offset does not match its total");
    }
    return layout;
}

inline bool tileIntersectionLayoutNeedsRadixScratch(
    const TileIntersectionLayout& layout) {
    return layout.maximumTileCount > kExactBitonicFastPath;
}

/// Converts exact per-tile counts into inclusive offsets for the packed arena.
/// The native rasterizer uses signed 32-bit ranges, so a larger scene is
/// rejected before any packed buffers or optimizer work are submitted.
inline TileIntersectionLayout buildTileIntersectionLayout(
    const uint32_t* counts, int32_t* inclusiveOffsets, size_t tileCount,
    int32_t* tileBins = nullptr,
    uint32_t* sortableTileIndices = nullptr) {
    if (tileCount > 0 && (!counts || !inclusiveOffsets)) {
        throw std::invalid_argument(
            "Tile-intersection counts and offsets must not be null");
    }

    TileIntersectionLayout layout;
    uint64_t running = 0;
    for (size_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t count = counts[tile];
        const int32_t start = static_cast<int32_t>(running);
        running += count;
        if (running > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::overflow_error(
                "Exact tile-intersection count exceeds the native index range");
        }
        const int32_t end = static_cast<int32_t>(running);
        inclusiveOffsets[tile] = end;
        if (tileBins) {
            tileBins[2 * tile] = start;
            tileBins[2 * tile + 1] = end;
        }
        if (count > 0) ++layout.activeTileCount;
        if (count <= 1) {
            ++layout.trivialTileCount;
        } else {
            ++layout.sortableTileCount;
            if (count <= kExactSmallTileMaximum) {
                ++layout.smallTileCount;
            } else if (count <= kExactBitonicFastPath) {
                ++layout.mediumTileCount;
            } else {
                ++layout.largeTileCount;
            }
        }
        if (count > layout.maximumTileCount) {
            layout.maximumTileCount = count;
            layout.maximumTileIndex = tile;
        }
    }
    layout.totalCount = static_cast<uint32_t>(running);

    // Stable bucket ordering lets the host dispatch the 2-32 entry prefix with
    // one 32-thread group per tile and the remaining tiles with 256 threads,
    // without allocating another compact-index buffer.
    if (sortableTileIndices) {
        uint32_t nextSmall = 0;
        uint32_t nextMedium = layout.smallTileCount;
        uint32_t nextLarge =
            layout.smallTileCount + layout.mediumTileCount;
        for (size_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t count = counts[tile];
            uint32_t* next = nullptr;
            if (count > 1 && count <= kExactSmallTileMaximum) {
                next = &nextSmall;
            } else if (count <= kExactBitonicFastPath &&
                       count > kExactSmallTileMaximum) {
                next = &nextMedium;
            } else if (count > kExactBitonicFastPath) {
                next = &nextLarge;
            }
            if (next) {
                sortableTileIndices[*next] = static_cast<uint32_t>(tile);
                ++(*next);
            }
        }
    }
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
