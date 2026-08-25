if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/include/intersection_layout.hpp" layout_header)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "GPU tile-layout contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "GPU tile-layout contract rejected: ${label}")
    endif()
endfunction()

function(require_count contents regex expected label)
    string(REGEX MATCHALL "${regex}" matches "${contents}")
    list(LENGTH matches actual)
    if(NOT actual EQUAL expected)
        message(FATAL_ERROR
            "Expected ${expected} ${label}, found ${actual}")
    endif()
endfunction()

function(extract_section contents start_marker end_marker output)
    string(FIND "${contents}" "${start_marker}" start_position)
    if(start_position EQUAL -1)
        message(FATAL_ERROR
            "GPU tile-layout contract cannot find ${start_marker}")
    endif()
    string(FIND "${contents}" "${end_marker}" end_position)
    if(end_position EQUAL -1 OR end_position LESS_EQUAL start_position)
        message(FATAL_ERROR
            "GPU tile-layout contract cannot find ${end_marker} after ${start_marker}")
    endif()
    math(EXPR section_length "${end_position} - ${start_position}")
    string(SUBSTRING "${contents}" ${start_position} ${section_length} section)
    set(${output} "${section}" PARENT_SCOPE)
endfunction()

function(require_ordered contents first second label)
    string(FIND "${contents}" "${first}" first_position)
    string(FIND "${contents}" "${second}" second_position)
    if(first_position EQUAL -1 OR second_position EQUAL -1 OR
       second_position LESS_EQUAL first_position)
        message(FATAL_ERROR
            "GPU tile-layout contract ordering failed: ${label}")
    endif()
endfunction()

extract_section("${metal_source}"
    "kernel void build_tile_intersection_layout_kernel("
    "kernel void scatter_to_exact_bins_kernel(" layout_kernel)

# The opt-in GPU kernel mirrors the checked CPU layout exactly. Its one-thread
# implementation is deliberate groundwork: it removes no synchronization yet,
# but establishes GPU-resident offsets, bins, stable sort lists, and metadata.
require_count("${metal_source}"
    "kernel void build_tile_intersection_layout_kernel\\(" 1
    "GPU tile-layout kernel definitions")
require_contains("${layout_kernel}"
    "device const uint* tile_counts         [[buffer(0)]]"
    "count input ABI")
require_contains("${layout_kernel}"
    "device int* inclusive_offsets          [[buffer(1)]]"
    "inclusive-offset output ABI")
require_contains("${layout_kernel}"
    "device int* tile_bins                  [[buffer(2)]]"
    "tile-bin output ABI")
require_contains("${layout_kernel}"
    "device uint* sortable_tile_indices     [[buffer(3)]]"
    "sortable-tile output ABI")
require_contains("${layout_kernel}"
    "device uint* metadata                  [[buffer(4)]]"
    "metadata output ABI")
require_contains("${layout_kernel}"
    "constant uint& num_tiles               [[buffer(5)]]"
    "tile-count scalar ABI")
require_contains("${layout_kernel}" "if (index != 0) return;"
    "single-thread ownership")
require_contains("${layout_kernel}"
    "count <= 0x7FFFFFFFu - running"
    "pre-add signed-index overflow check")
require_contains("${layout_kernel}"
    "error_flags |= EXACT_LAYOUT_ERROR_SIGNED_INDEX_OVERFLOW;"
    "overflow status")
require_contains("${layout_kernel}"
    "if (count > maximum_tile_count)"
    "first-maximum tie behavior")
require_contains("${layout_kernel}"
    "uint next_small = 0;"
    "stable small bucket cursor")
require_contains("${layout_kernel}"
    "uint next_medium = small_tile_count;"
    "stable medium bucket cursor")
require_contains("${layout_kernel}"
    "uint next_large = small_tile_count + medium_tile_count;"
    "stable large bucket cursor")
require_contains("${layout_kernel}"
    "sortable_tile_indices[next_small++] = tile;"
    "stable small list write")
require_contains("${layout_kernel}"
    "sortable_tile_indices[next_medium++] = tile;"
    "stable medium list write")
require_contains("${layout_kernel}"
    "sortable_tile_indices[next_large++] = tile;"
    "stable large list write")
require_contains("${layout_kernel}"
    "inclusive_offsets[tile] = 0;"
    "neutralized overflowing offsets")
require_contains("${layout_kernel}"
    "tile_bins[tile * 2u + 1u] = 0;"
    "neutralized overflowing bins")
require_contains("${layout_kernel}"
    "sortable_tile_indices[tile] = 0;"
    "neutralized overflowing sort list")
require_contains("${layout_kernel}"
    "metadata[EXACT_LAYOUT_ERROR_FLAGS] = error_flags;"
    "metadata error publication")

# Host and shader share a fixed ten-word metadata contract. Host conversion
# rejects overflow and independently verifies classification and final offset.
require_contains("${layout_header}"
    "kTileIntersectionLayoutMetadataWordCount = 10"
    "ten-word host metadata ABI")
require_contains("${layout_header}"
    "kTileIntersectionLayoutErrorFlagsWord = 9"
    "host error word")
require_contains("${layout_header}"
    "kTileIntersectionLayoutSignedIndexOverflow = 1u << 0"
    "host overflow bit")
require_contains("${layout_header}"
    "tileIntersectionLayoutFromGpuMetadata("
    "checked metadata conversion")
require_contains("${layout_header}"
    "metadata[kTileIntersectionLayoutErrorFlagsWord] &\n        kTileIntersectionLayoutSignedIndexOverflow"
    "signed-index overflow rejection")
require_contains("${layout_header}"
    "classifiedTiles != tileCount"
    "tile classification validation")
require_contains("${layout_header}"
    "sortableTiles != layout.sortableTileCount"
    "sort bucket validation")
require_contains("${layout_header}"
    "static_cast<uint32_t>(finalOffset) != layout.totalCount"
    "final-offset validation")

# Configuration is strict and defaults to the established CPU builder. The
# optional PSO and metadata allocation stay absent unless the process opts in.
require_contains("${host_source}" "bool gpu_tile_layout = false;"
    "CPU default state")
require_contains("${host_source}"
    "std::getenv(\"MSPLAT_TILE_LAYOUT_MODE\")"
    "tile-layout environment lookup")
require_contains("${host_source}"
    "std::strcmp(tileLayoutModeOverride, \"cpu\") == 0"
    "strict CPU value")
require_contains("${host_source}"
    "std::strcmp(tileLayoutModeOverride, \"gpu\") == 0"
    "strict GPU value")
require_contains("${host_source}"
    "msplat: MSPLAT_TILE_LAYOUT_MODE must be cpu or gpu"
    "invalid-mode diagnostic")
require_absent("${host_source}"
    "strcasecmp(tileLayoutModeOverride"
    "case-insensitive tile-layout parsing")
require_contains("${host_source}"
    "id<MTLComputePipelineState> build_tile_intersection_layout_kernel_cpso = nil;"
    "optional layout PSO")
require_contains("${host_source}"
    "if (ctx->gpu_tile_layout) {\n        ctx->build_tile_intersection_layout_kernel_cpso =\n            load(@\"build_tile_intersection_layout_kernel\");\n    }"
    "lazy layout PSO load")
require_contains("${host_source}" "MTensor tile_layout_metadata;"
    "optional metadata tensor")
require_contains("${host_source}"
    "void ensure_tile_layout_metadata(bool enabled"
    "metadata allocation helper")
require_contains("${host_source}"
    "if (!enabled) {\n            tile_layout_metadata.reset();\n            return;\n        }"
    "default-path metadata release")
require_contains("${host_source}" "&tile_layout_metadata,"
    "metadata memory accounting")

extract_section("${host_source}"
    "static void encode_gpu_tile_intersection_layout("
    "MTensor gpu_zeros(" encode_helper)
require_contains("${encode_helper}"
    "if (!ctx->gpu_tile_layout) return;"
    "default-path encode bypass")
require_contains("${encode_helper}"
    "memoryBarrierWithScope:MTLBarrierScopeBuffers"
    "count-to-layout buffer barrier")
require_contains("${encode_helper}"
    "ctx->build_tile_intersection_layout_kernel_cpso"
    "GPU layout pipeline selection")
require_contains("${encode_helper}" "ENC_BUF(encoder, tileCounts, 0);"
    "host count binding")
require_contains("${encode_helper}" "ENC_BUF(encoder, tileOffsets, 1);"
    "host offset binding")
require_contains("${encode_helper}" "ENC_BUF(encoder, tileBins, 2);"
    "host bin binding")
require_contains("${encode_helper}"
    "ENC_BUF(encoder, sortableTileIndices, 3);"
    "host sortable-list binding")
require_contains("${encode_helper}" "ENC_BUF(encoder, metadata, 4);"
    "host metadata binding")
require_contains("${encode_helper}" "ENC_SCALAR(encoder, numTiles, 5);"
    "host tile-count binding")
require_contains("${encode_helper}"
    "dispatchThreads:MTLSizeMake(1, 1, 1)"
    "single-thread dispatch")

require_count("${host_source}"
    "encode_gpu_tile_intersection_layout\\(" 3
    "shared GPU-layout definition plus render and training calls")
extract_section("${host_source}" "static void render_pipeline("
    "MTensor msplat_render(" render_pipeline)
extract_section("${host_source}" "static MTensor msplat_train_step_locked("
    "MTensor msplat_train_step(" training_pipeline)
foreach(pipeline IN ITEMS render_pipeline training_pipeline)
    require_contains("${${pipeline}}"
        "g_tcache.ensure_tile_layout_metadata(ctx->gpu_tile_layout, ctx->device);"
        "${pipeline} lazy metadata setup")
    require_contains("${${pipeline}}"
        "encode_gpu_tile_intersection_layout("
        "${pipeline} GPU layout encode")
    require_ordered("${${pipeline}}"
        "encode_tile_count_difference_scan("
        "encode_gpu_tile_intersection_layout("
        "${pipeline} exact count precedes GPU layout")
    require_contains("${${pipeline}}" "ctx->syncCB()"
        "${pipeline} retained host synchronization")
    require_ordered("${${pipeline}}"
        "encode_gpu_tile_intersection_layout("
        "ctx->syncCB()"
        "${pipeline} completes GPU layout before host readback")
    require_contains("${${pipeline}}"
        "? completed_gpu_tile_intersection_layout(num_tiles)"
        "${pipeline} checked completed GPU layout")
    require_contains("${${pipeline}}"
        "msplat::buildTileIntersectionLayout("
        "${pipeline} established CPU default")
    require_contains("${${pipeline}}"
        "g_tcache.ensure_intersection_arena("
        "${pipeline} retained host arena sizing")
endforeach()

extract_section("${host_source}"
    "static msplat::TileIntersectionLayout completed_gpu_tile_intersection_layout("
    "size_t msplat_cached_tensor_bytes(" completed_helper)
require_contains("${completed_helper}"
    "msplat::tileIntersectionLayoutFromGpuMetadata("
    "checked GPU metadata branch")
