#include "intersection_layout.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    const auto noTiles = msplat::buildTileIntersectionLayout(
        nullptr, nullptr, 0);
    CHECK(noTiles.totalCount == 0);
    CHECK(noTiles.maximumTileCount == 0);

    const uint32_t zeroCount[] = {0};
    int32_t zeroOffset[] = {-1};
    const auto zero = msplat::buildTileIntersectionLayout(
        zeroCount, zeroOffset, 1);
    CHECK(zero.totalCount == 0);
    CHECK(zero.maximumTileCount == 0);
    CHECK(zeroOffset[0] == 0);

    const uint32_t counts[] = {0, 3, 1, 0, 4};
    int32_t offsets[5] = {};
    const auto layout = msplat::buildTileIntersectionLayout(counts, offsets, 5);
    CHECK(layout.totalCount == 8);
    CHECK(layout.maximumTileCount == 4);
    CHECK(layout.maximumTileIndex == 4);
    CHECK(offsets[0] == 0);
    CHECK(offsets[1] == 3);
    CHECK(offsets[2] == 4);
    CHECK(offsets[3] == 4);
    CHECK(offsets[4] == 8);

    const uint32_t emptyCounts[] = {0, 0};
    int32_t emptyOffsets[2] = {-1, -1};
    const auto empty = msplat::buildTileIntersectionLayout(
        emptyCounts, emptyOffsets, 2);
    CHECK(empty.totalCount == 0);
    CHECK(empty.maximumTileCount == 0);
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

    CHECK(msplat::kExactIntersectionBytesPerEntry == 56);
    CHECK(msplat::kExactTileMetadataBytes == 16);
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
