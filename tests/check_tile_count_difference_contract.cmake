if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Tile-count difference contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Tile-count difference contract rejected: ${label}")
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
            "Tile-count difference contract cannot find ${start_marker}")
    endif()
    string(FIND "${contents}" "${end_marker}" end_position)
    if(end_position EQUAL -1 OR end_position LESS_EQUAL start_position)
        message(FATAL_ERROR
            "Tile-count difference contract cannot find ${end_marker} after ${start_marker}")
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
            "Tile-count difference contract ordering failed: ${label}")
    endif()
endfunction()

extract_section("${metal_source}"
    "kernel void project_and_sh_forward_kernel("
    "kernel void tile_count_diff_horizontal_kernel(" projection_kernel)
extract_section("${metal_source}"
    "kernel void tile_count_diff_horizontal_kernel("
    "kernel void tile_count_diff_vertical_kernel(" horizontal_kernel)
extract_section("${metal_source}"
    "kernel void tile_count_diff_vertical_kernel("
    "// Adam update helper" vertical_kernel)

# Projection retains the established enumerated path and adds one explicit mode
# input plus the four half-open difference-grid corners.
require_contains("${projection_kernel}"
    "device atomic_uint* tile_count_storage"
    "projected count-storage binding")
require_contains("${projection_kernel}"
    "constant uint& tile_count_mode"
    "projected count-mode binding")
require_contains("${projection_kernel}" "if (tile_count_mode == 0u)"
    "enumerated default branch")
require_contains("${projection_kernel}"
    "training_render_tile_active("
    "enumerated coverage predicate")
require_contains("${projection_kernel}"
    "const uint diff_width = tile_bounds.x + 1u;"
    "difference-grid border column")
require_contains("${projection_kernel}"
    "const uint top_left = tile_min.y * diff_width + tile_min.x;"
    "top-left corner")
require_contains("${projection_kernel}"
    "const uint top_right = tile_min.y * diff_width + tile_max.x;"
    "top-right corner")
require_contains("${projection_kernel}"
    "const uint bottom_left = tile_max.y * diff_width + tile_min.x;"
    "bottom-left corner")
require_contains("${projection_kernel}"
    "const uint bottom_right = tile_max.y * diff_width + tile_max.x;"
    "bottom-right corner")
require_contains("${projection_kernel}"
    "&tile_count_storage[top_left], 1u"
    "positive top-left update")
require_contains("${projection_kernel}"
    "&tile_count_storage[top_right], uint(-1)"
    "negative top-right update")
require_contains("${projection_kernel}"
    "&tile_count_storage[bottom_left], uint(-1)"
    "negative bottom-left update")
require_contains("${projection_kernel}"
    "&tile_count_storage[bottom_right], 1u"
    "positive bottom-right update")
require_count("${projection_kernel}"
    "atomic_fetch_add_explicit\\(" 5
    "projection count atomics: one enumerated plus four corners")

# The row pass scans the complete signed border grid in place. The column pass
# emits only live tile rows and applies coverage after exact reconstruction.
require_contains("${horizontal_kernel}"
    "device int* diff                         [[buffer(0)]]"
    "horizontal signed-grid ABI")
require_contains("${horizontal_kernel}"
    "constant uint3& tile_bounds              [[buffer(1)]]"
    "horizontal bounds ABI")
require_contains("${horizontal_kernel}"
    "if (row > tile_bounds.y) return;"
    "horizontal border-row guard")
require_contains("${horizontal_kernel}"
    "for (uint tile_x = 0; tile_x <= tile_bounds.x; ++tile_x)"
    "horizontal inclusive border scan")
require_contains("${horizontal_kernel}"
    "running += diff[index];\n        diff[index] = running;"
    "horizontal in-place signed prefix")

require_contains("${vertical_kernel}"
    "constant int* horizontal_diff            [[buffer(0)]]"
    "vertical signed-grid ABI")
require_contains("${vertical_kernel}"
    "device uint* tile_counts                  [[buffer(1)]]"
    "vertical count-output ABI")
require_contains("${vertical_kernel}"
    "constant uint3& tile_bounds               [[buffer(2)]]"
    "vertical bounds ABI")
require_contains("${vertical_kernel}"
    "constant uchar* coverage_render_tiles    [[buffer(3)]]"
    "vertical coverage ABI")
require_contains("${vertical_kernel}"
    "constant uint& coverage_render_tile_stride [[buffer(4)]]"
    "vertical coverage-stride ABI")
require_contains("${vertical_kernel}"
    "if (tile_x >= tile_bounds.x) return;"
    "vertical column guard")
require_contains("${vertical_kernel}"
    "for (uint tile_y = 0; tile_y < tile_bounds.y; ++tile_y)"
    "vertical tile-row scan")
require_contains("${vertical_kernel}"
    "running += horizontal_diff[tile_y * diff_width + tile_x];"
    "vertical signed prefix")
require_contains("${vertical_kernel}"
    "tile_counts[tile_id] = training_render_tile_active("
    "post-scan coverage mask")
require_contains("${vertical_kernel}"
    "? uint(running)\n            : 0u;"
    "covered exact count or zero")

# Host configuration is strict, process-wide, and frozen before pipeline use.
# Enumerated remains the zero-initialized default; difference owns its PSOs and
# scratch only when explicitly selected.
require_contains("${host_source}"
    "bool difference_tile_counting = false;"
    "enumerated default state")
require_contains("${host_source}"
    "std::getenv(\"MSPLAT_TILE_COUNT_MODE\")"
    "tile-count environment override")
require_contains("${host_source}"
    "std::strcmp(tileCountModeOverride, \"enumerated\") == 0"
    "strict enumerated value")
require_contains("${host_source}"
    "std::strcmp(tileCountModeOverride, \"difference\") == 0"
    "strict difference value")
require_contains("${host_source}"
    "msplat: MSPLAT_TILE_COUNT_MODE must be enumerated or difference"
    "invalid-mode diagnostic")
require_absent("${host_source}"
    "strcasecmp(tileCountModeOverride"
    "case-insensitive tile-count parsing")

require_contains("${host_source}"
    "id<MTLComputePipelineState> tile_count_diff_horizontal_kernel_cpso = nil;"
    "horizontal optional PSO")
require_contains("${host_source}"
    "id<MTLComputePipelineState> tile_count_diff_vertical_kernel_cpso = nil;"
    "vertical optional PSO")
require_count("${host_source}"
    "load\\(@\"tile_count_diff_horizontal_kernel\"\\)" 1
    "horizontal PSO loads")
require_count("${host_source}"
    "load\\(@\"tile_count_diff_vertical_kernel\"\\)" 1
    "vertical PSO loads")
require_contains("${host_source}"
    "if (ctx->difference_tile_counting) {"
    "lazy difference PSO gate")
require_contains("${host_source}"
    "if (ctx->difference_tile_counting) {\n        ctx->tile_count_diff_horizontal_kernel_cpso =\n            load(@\"tile_count_diff_horizontal_kernel\");\n        ctx->tile_count_diff_vertical_kernel_cpso =\n            load(@\"tile_count_diff_vertical_kernel\");\n    }"
    "difference PSOs loaded only inside the opt-in gate")

require_contains("${host_source}" "MTensor tile_count_diff;"
    "difference-grid cache tensor")
require_contains("${host_source}"
    "int tile_count_diff_width = -1, tile_count_diff_height = -1;"
    "difference-grid cache dimensions")
require_contains("${host_source}" "void ensure_tile_count_diff("
    "difference-grid cache helper")
require_contains("${host_source}"
    "if (!enabled) {\n            resetTileCountDifference();\n            return;\n        }"
    "difference scratch disabled on the default path")
require_contains("${host_source}"
    "{static_cast<int64_t>(tileHeight) + 1,\n                 static_cast<int64_t>(tileWidth) + 1},\n                DType::Int32"
    "signed bordered difference-grid allocation")
require_contains("${host_source}" "&tile_count_diff,"
    "difference-grid memory accounting")

# A shared helper encodes horizontal then vertical exactly once, with a buffer
# barrier before each scan. Render and training both call it immediately after
# projection, may then build the equivalent layout on the GPU, and retain the
# existing GPU wait and checked host-side arena decision.
require_count("${host_source}"
    "ENC_SCALAR\\(enc, tileCountMode, 26\\)" 2
    "projection mode bindings")
require_count("${host_source}"
    "ENC_BUF\\(enc, tileCountStorage, 15\\)" 2
    "selected projection count-storage bindings")
require_count("${host_source}"
    "encode_tile_count_difference_scan\\(" 3
    "shared scan definition plus two call sites")

extract_section("${host_source}"
    "static void encode_tile_count_difference_scan("
    "id<MTLDevice> msplat_device(" scan_helper)
require_contains("${scan_helper}"
    "if (!ctx->difference_tile_counting) return;"
    "default-path scan bypass")
require_contains("${scan_helper}"
    "static_cast<NSUInteger>(tileBounds[1]) + 1"
    "horizontal border-row count")
require_contains("${scan_helper}"
    "static_cast<NSUInteger>(tileBounds[0])"
    "vertical tile-column count")
require_count("${scan_helper}"
    "setComputePipelineState:" 2
    "shared scan pipeline selections")
require_contains("${scan_helper}"
    "[encoder setComputePipelineState:\n        ctx->tile_count_diff_horizontal_kernel_cpso];"
    "shared horizontal scan dispatch")
require_contains("${scan_helper}"
    "[encoder setComputePipelineState:\n        ctx->tile_count_diff_vertical_kernel_cpso];"
    "shared vertical scan dispatch")
require_count("${scan_helper}"
    "memoryBarrierWithScope:MTLBarrierScopeBuffers" 2
    "scan buffer barriers")
require_contains("${scan_helper}"
    "dispatchThreads:MTLSizeMake(horizontalThreadCount, 1, 1)"
    "horizontal row dispatch")
require_contains("${scan_helper}"
    "dispatchThreads:MTLSizeMake(verticalThreadCount, 1, 1)"
    "vertical column dispatch")
require_ordered("${scan_helper}"
    "[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];"
    "[encoder setComputePipelineState:\n        ctx->tile_count_diff_horizontal_kernel_cpso];"
    "projection barrier precedes horizontal scan")
require_ordered("${scan_helper}"
    "[encoder setComputePipelineState:\n        ctx->tile_count_diff_horizontal_kernel_cpso];"
    "[encoder setComputePipelineState:\n        ctx->tile_count_diff_vertical_kernel_cpso];"
    "horizontal scan precedes vertical scan")

# The second barrier must sit between the two dispatches, not merely exist
# somewhere in the helper.
string(FIND "${scan_helper}"
    "[encoder setComputePipelineState:\n        ctx->tile_count_diff_horizontal_kernel_cpso];"
    horizontal_pso_position)
string(FIND "${scan_helper}"
    "[encoder setComputePipelineState:\n        ctx->tile_count_diff_vertical_kernel_cpso];"
    vertical_pso_position)
math(EXPR between_scans_length
    "${vertical_pso_position} - ${horizontal_pso_position}")
string(SUBSTRING "${scan_helper}" ${horizontal_pso_position}
    ${between_scans_length} between_scans)
require_contains("${between_scans}"
    "[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];"
    "horizontal-to-vertical buffer barrier")

extract_section("${host_source}" "static void render_pipeline("
    "MTensor msplat_render(" render_pipeline)
extract_section("${host_source}" "static MTensor msplat_train_step_locked("
    "MTensor msplat_train_step(" training_pipeline)

foreach(pipeline IN ITEMS render_pipeline training_pipeline)
    require_contains("${${pipeline}}" "encode_proj_sh(encoder);"
        "${pipeline} projection dispatch")
    require_contains("${${pipeline}}"
        "encode_tile_count_difference_scan("
        "${pipeline} shared scan call")
    require_count("${${pipeline}}"
        "encode_tile_count_difference_scan\\(" 1
        "${pipeline} shared scan calls")
    require_ordered("${${pipeline}}"
        "encode_proj_sh(encoder);"
        "encode_tile_count_difference_scan("
        "${pipeline} projection precedes scan helper")
    require_contains("${${pipeline}}"
        "encode_gpu_tile_intersection_layout("
        "${pipeline} optional GPU layout call")
    require_ordered("${${pipeline}}"
        "encode_tile_count_difference_scan("
        "encode_gpu_tile_intersection_layout("
        "${pipeline} reconstructed counts precede GPU layout")
    require_contains("${${pipeline}}" "ctx->syncCB()"
        "${pipeline} retained GPU-to-CPU wait")
    require_contains("${${pipeline}}"
        "? completed_gpu_tile_intersection_layout(num_tiles)"
        "${pipeline} opt-in checked GPU layout")
    require_contains("${${pipeline}}"
        "msplat::buildTileIntersectionLayout("
        "${pipeline} retained checked CPU layout")
    require_contains("${${pipeline}}"
        "g_tcache.tile_scatter_counters.data<uint32_t>()"
        "${pipeline} CPU layout reads reconstructed counts")
    require_contains("${${pipeline}}"
        "g_tcache.ensure_intersection_arena("
        "${pipeline} retained exact arena sizing")
endforeach()

extract_section("${host_source}"
    "static msplat::TileIntersectionLayout completed_gpu_tile_intersection_layout("
    "size_t msplat_cached_tensor_bytes(" completed_layout_helper)
require_contains("${completed_layout_helper}"
    "msplat::tileIntersectionLayoutFromGpuMetadata("
    "opt-in checked GPU metadata conversion")
