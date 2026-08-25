#include "intersection_layout.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    const auto noTiles = msplat::buildTileIntersectionLayout(
        nullptr, nullptr, 0);
    CHECK(noTiles.totalCount == 0);
    CHECK(noTiles.maximumTileCount == 0);
    CHECK(noTiles.activeTileCount == 0);
    CHECK(noTiles.sortableTileCount == 0);
    CHECK(noTiles.trivialTileCount == 0);

    const uint32_t zeroCount[] = {0};
    int32_t zeroOffset[] = {-1};
    const auto zero = msplat::buildTileIntersectionLayout(
        zeroCount, zeroOffset, 1);
    CHECK(zero.totalCount == 0);
    CHECK(zero.maximumTileCount == 0);
    CHECK(zero.activeTileCount == 0);
    CHECK(zero.sortableTileCount == 0);
    CHECK(zero.trivialTileCount == 1);
    CHECK(zeroOffset[0] == 0);

    const uint32_t counts[] = {0, 3, 1, 0, 4};
    int32_t offsets[5] = {};
    int32_t bins[10] = {};
    uint32_t sortableTileIndices[5] = {};
    const auto layout = msplat::buildTileIntersectionLayout(
        counts, offsets, 5, bins, sortableTileIndices);
    CHECK(layout.totalCount == 8);
    CHECK(layout.maximumTileCount == 4);
    CHECK(layout.maximumTileIndex == 4);
    CHECK(layout.activeTileCount == 3);
    CHECK(layout.sortableTileCount == 2);
    CHECK(layout.trivialTileCount == 3);
    CHECK(layout.smallTileCount == 2);
    CHECK(layout.mediumTileCount == 0);
    CHECK(layout.largeTileCount == 0);
    CHECK(!msplat::tileIntersectionLayoutNeedsRadixScratch(layout));
    CHECK(offsets[0] == 0);
    CHECK(offsets[1] == 3);
    CHECK(offsets[2] == 4);
    CHECK(offsets[3] == 4);
    CHECK(offsets[4] == 8);
    CHECK(bins[0] == 0 && bins[1] == 0);
    CHECK(bins[2] == 0 && bins[3] == 3);
    CHECK(bins[4] == 3 && bins[5] == 4);
    CHECK(bins[6] == 4 && bins[7] == 4);
    CHECK(bins[8] == 4 && bins[9] == 8);
    CHECK(sortableTileIndices[0] == 1);
    CHECK(sortableTileIndices[1] == 4);

    uint32_t gpuMetadata[msplat::kTileIntersectionLayoutMetadataWordCount] = {
        8, 4, 4, 3, 2, 3, 2, 0, 0, 0
    };
    const auto gpuLayout = msplat::tileIntersectionLayoutFromGpuMetadata(
        gpuMetadata, msplat::kTileIntersectionLayoutMetadataWordCount,
        5, offsets);
    CHECK(gpuLayout.totalCount == layout.totalCount);
    CHECK(gpuLayout.maximumTileCount == layout.maximumTileCount);
    CHECK(gpuLayout.maximumTileIndex == layout.maximumTileIndex);
    CHECK(gpuLayout.activeTileCount == layout.activeTileCount);
    CHECK(gpuLayout.sortableTileCount == layout.sortableTileCount);
    CHECK(gpuLayout.trivialTileCount == layout.trivialTileCount);
    CHECK(gpuLayout.smallTileCount == layout.smallTileCount);
    CHECK(gpuLayout.mediumTileCount == layout.mediumTileCount);
    CHECK(gpuLayout.largeTileCount == layout.largeTileCount);

    struct AttemptSnapshot {
        uint32_t failureReasons;
        std::array<uint32_t, msplat::kTileIntersectionLayoutMetadataWordCount>
            metadata;
        int32_t finalInclusiveOffset;
    };
    uint32_t sourceFailureReasons = 1u << 1;
    AttemptSnapshot attemptSnapshot{
        sourceFailureReasons, {}, offsets[4]
    };
    std::copy_n(
        gpuMetadata, attemptSnapshot.metadata.size(),
        attemptSnapshot.metadata.begin());

    // The decoder must consume copied values rather than reusable source
    // buffers that a later GPU attempt can overwrite.
    sourceFailureReasons = 0;
    gpuMetadata[msplat::kTileIntersectionLayoutTotalCountWord] = 0;
    offsets[4] = 0;
    const auto snapshottedLayout =
        msplat::tileIntersectionLayoutFromGpuMetadata(
            attemptSnapshot.metadata.data(), attemptSnapshot.metadata.size(),
            5, attemptSnapshot.finalInclusiveOffset);
    CHECK(attemptSnapshot.failureReasons == (1u << 1));
    CHECK(snapshottedLayout.totalCount == layout.totalCount);
    CHECK(snapshottedLayout.maximumTileCount == layout.maximumTileCount);
    CHECK(snapshottedLayout.maximumTileIndex == layout.maximumTileIndex);
    CHECK(snapshottedLayout.activeTileCount == layout.activeTileCount);
    CHECK(snapshottedLayout.sortableTileCount == layout.sortableTileCount);
    CHECK(snapshottedLayout.trivialTileCount == layout.trivialTileCount);
    CHECK(snapshottedLayout.smallTileCount == layout.smallTileCount);
    CHECK(snapshottedLayout.mediumTileCount == layout.mediumTileCount);
    CHECK(snapshottedLayout.largeTileCount == layout.largeTileCount);

    std::copy_n(
        attemptSnapshot.metadata.begin(), attemptSnapshot.metadata.size(),
        gpuMetadata);
    offsets[4] = attemptSnapshot.finalInclusiveOffset;

    bool truncatedGpuMetadataRejected = false;
    try {
        (void)msplat::tileIntersectionLayoutFromGpuMetadata(
            gpuMetadata,
            msplat::kTileIntersectionLayoutMetadataWordCount - 1,
            5, offsets);
    } catch (const std::invalid_argument&) {
        truncatedGpuMetadataRejected = true;
    }
    CHECK(truncatedGpuMetadataRejected);

    bool gpuOverflowRejected = false;
    gpuMetadata[msplat::kTileIntersectionLayoutErrorFlagsWord] =
        msplat::kTileIntersectionLayoutSignedIndexOverflow;
    try {
        (void)msplat::tileIntersectionLayoutFromGpuMetadata(
            gpuMetadata, msplat::kTileIntersectionLayoutMetadataWordCount,
            5, offsets);
    } catch (const std::overflow_error&) {
        gpuOverflowRejected = true;
    }
    CHECK(gpuOverflowRejected);

    bool unknownGpuErrorRejected = false;
    gpuMetadata[msplat::kTileIntersectionLayoutErrorFlagsWord] = 1u << 1;
    try {
        (void)msplat::tileIntersectionLayoutFromGpuMetadata(
            gpuMetadata, msplat::kTileIntersectionLayoutMetadataWordCount,
            5, offsets);
    } catch (const std::runtime_error&) {
        unknownGpuErrorRejected = true;
    }
    CHECK(unknownGpuErrorRejected);

    gpuMetadata[msplat::kTileIntersectionLayoutErrorFlagsWord] = 0;
    bool inconsistentGpuBucketsRejected = false;
    ++gpuMetadata[msplat::kTileIntersectionLayoutSmallTileCountWord];
    try {
        (void)msplat::tileIntersectionLayoutFromGpuMetadata(
            gpuMetadata, msplat::kTileIntersectionLayoutMetadataWordCount,
            5, offsets);
    } catch (const std::runtime_error&) {
        inconsistentGpuBucketsRejected = true;
    }
    CHECK(inconsistentGpuBucketsRejected);
    --gpuMetadata[msplat::kTileIntersectionLayoutSmallTileCountWord];

    bool mismatchedGpuOffsetRejected = false;
    int32_t mismatchedOffsets[5] = {0, 3, 4, 4, 7};
    try {
        (void)msplat::tileIntersectionLayoutFromGpuMetadata(
            gpuMetadata, msplat::kTileIntersectionLayoutMetadataWordCount,
            5, mismatchedOffsets);
    } catch (const std::runtime_error&) {
        mismatchedGpuOffsetRejected = true;
    }
    CHECK(mismatchedGpuOffsetRejected);

    const uint32_t emptyCounts[] = {0, 0};
    int32_t emptyOffsets[2] = {-1, -1};
    const auto empty = msplat::buildTileIntersectionLayout(
        emptyCounts, emptyOffsets, 2);
    CHECK(empty.totalCount == 0);
    CHECK(empty.maximumTileCount == 0);
    CHECK(empty.activeTileCount == 0);
    CHECK(empty.sortableTileCount == 0);
    CHECK(empty.trivialTileCount == 2);
    CHECK(emptyOffsets[0] == 0);
    CHECK(emptyOffsets[1] == 0);

    const uint32_t denseSingleTileCount[] = {4'097};
    int32_t denseSingleTileOffset[] = {};
    const auto denseSingleTile = msplat::buildTileIntersectionLayout(
        denseSingleTileCount, denseSingleTileOffset, 1);
    CHECK(denseSingleTile.totalCount == 4'097);
    CHECK(denseSingleTile.maximumTileCount == 4'097);
    CHECK(denseSingleTileOffset[0] == 4'097);
    msplat::validateTileIntersectionWorkLimit(denseSingleTile);
    CHECK(denseSingleTile.largeTileCount == 1);

    const uint32_t bucketBoundaryCounts[] = {
        33, 2, 2'049, 31, 1, 2'048, 3, 32, 0
    };
    int32_t bucketBoundaryOffsets[9] = {};
    uint32_t bucketBoundarySortableTiles[9] = {
        UINT32_MAX, UINT32_MAX, UINT32_MAX,
        UINT32_MAX, UINT32_MAX, UINT32_MAX,
        UINT32_MAX, UINT32_MAX, UINT32_MAX
    };
    const auto bucketBoundaries = msplat::buildTileIntersectionLayout(
        bucketBoundaryCounts, bucketBoundaryOffsets, 9, nullptr,
        bucketBoundarySortableTiles);
    CHECK(bucketBoundaries.activeTileCount == 8);
    CHECK(bucketBoundaries.sortableTileCount == 7);
    CHECK(bucketBoundaries.trivialTileCount == 2);
    CHECK(bucketBoundaries.smallTileCount == 4);
    CHECK(bucketBoundaries.mediumTileCount == 2);
    CHECK(bucketBoundaries.largeTileCount == 1);
    CHECK(bucketBoundaries.sortableTileCount ==
          bucketBoundaries.smallTileCount +
              bucketBoundaries.mediumTileCount +
              bucketBoundaries.largeTileCount);
    CHECK(msplat::tileIntersectionLayoutNeedsRadixScratch(bucketBoundaries));
    const uint32_t expectedBucketOrder[] = {1, 3, 6, 7, 0, 5, 2};
    for (uint32_t index = 0; index < 7; ++index) {
        CHECK(bucketBoundarySortableTiles[index] == expectedBucketOrder[index]);
    }
    CHECK(bucketBoundarySortableTiles[7] == UINT32_MAX);
    CHECK(bucketBoundarySortableTiles[8] == UINT32_MAX);

    const uint32_t bitonicBoundaryCounts[] = {2'048};
    int32_t bitonicBoundaryOffsets[1] = {};
    const auto bitonicBoundary = msplat::buildTileIntersectionLayout(
        bitonicBoundaryCounts, bitonicBoundaryOffsets, 1);
    CHECK(bitonicBoundary.maximumTileCount == 2'048);
    CHECK(!msplat::tileIntersectionLayoutNeedsRadixScratch(bitonicBoundary));

    const uint32_t radixBoundaryCounts[] = {2'049};
    int32_t radixBoundaryOffsets[1] = {};
    const auto radixBoundary = msplat::buildTileIntersectionLayout(
        radixBoundaryCounts, radixBoundaryOffsets, 1);
    CHECK(radixBoundary.maximumTileCount == 2'049);
    CHECK(msplat::tileIntersectionLayoutNeedsRadixScratch(radixBoundary));

    const msplat::TileIntersectionLayout workLimitBoundary{
        65'536, 65'536, 0};
    msplat::validateTileIntersectionWorkLimit(workLimitBoundary);
    bool workLimitRejected = false;
    try {
        msplat::validateTileIntersectionWorkLimit(
            msplat::TileIntersectionLayout{65'537, 65'537, 0});
    } catch (const std::length_error&) {
        workLimitRejected = true;
    }
    CHECK(workLimitRejected);

    bool zeroWorkLimitRejected = false;
    try {
        msplat::validateTileIntersectionWorkLimit(denseSingleTile, 0);
    } catch (const std::invalid_argument&) {
        zeroWorkLimitRejected = true;
    }
    CHECK(zeroWorkLimitRejected);

    bool overflowRejected = false;
    const uint32_t overflowingCounts[] = {
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max()), 1
    };
    try {
        int32_t overflowingOffsets[2] = {};
        (void)msplat::buildTileIntersectionLayout(
            overflowingCounts, overflowingOffsets, 2);
    } catch (const std::overflow_error&) {
        overflowRejected = true;
    }
    CHECK(overflowRejected);

    const uint32_t acceptedBoundaryCounts[] = {
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max() - 1), 1
    };
    int32_t acceptedBoundaryOffsets[2] = {};
    const auto acceptedBoundary = msplat::buildTileIntersectionLayout(
        acceptedBoundaryCounts, acceptedBoundaryOffsets, 2);
    CHECK(acceptedBoundary.totalCount ==
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
    CHECK(acceptedBoundaryOffsets[1] == std::numeric_limits<int32_t>::max());

    CHECK(msplat::tileIntersectionArenaCapacity(0, 0, false) == 1);
    CHECK(msplat::tileIntersectionArenaCapacity(1, 0, false) == 4'097);
    CHECK(msplat::tileIntersectionArenaCapacity(5'000, 0, false) == 9'096);
    CHECK(msplat::tileIntersectionArenaCapacity(5'000, 8'000, false) == 8'000);
    CHECK(msplat::tileIntersectionArenaCapacity(5'000, 8'000, true) == 9'096);
    CHECK(msplat::tileIntersectionArenaCapacity(1'000'000, 0, false) ==
          1'250'000);
    CHECK(msplat::tileIntersectionArenaCapacity(
              static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
              0, false) ==
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));

    bool arenaOverflowRejected = false;
    try {
        (void)msplat::tileIntersectionArenaCapacity(
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1u,
            0, false);
    } catch (const std::overflow_error&) {
        arenaOverflowRejected = true;
    }
    CHECK(arenaOverflowRejected);

    CHECK(msplat::kExactBitonicFastPath == 2'048);
    CHECK(msplat::kExactSmallTileMaximum == 32);
    CHECK(msplat::kExactRadixScratchBytesPerEntry == 8);
    CHECK(msplat::kExactBitonicOnlyIntersectionBytesPerEntry == 44);
    CHECK(msplat::kExactIntersectionBytesPerEntry == 52);
    CHECK(msplat::kExactTileMetadataBytes == 20);
    CHECK(msplat::tileRasterChunkCount(1, 0) == 1);
    CHECK(msplat::tileRasterChunkCount(1, 1) == 1);
    CHECK(msplat::tileRasterChunkCount(1, 512) == 1);
    CHECK(msplat::tileRasterChunkCount(1, 513) == 2);
    CHECK(msplat::tileRasterChunkCount(400, 513) == 1);

    bool zeroChunkRejected = false;
    try {
        (void)msplat::tileRasterChunkCount(1, 1, 0);
    } catch (const std::invalid_argument&) {
        zeroChunkRejected = true;
    }
    CHECK(zeroChunkRejected);

    return 0;
}
