if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/include/intersection_layout.hpp" layout_header)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Exact-intersection contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Legacy fixed-bin contract remains: ${label}")
    endif()
endfunction()

foreach(kernel_name IN ITEMS
        scatter_to_exact_bins_kernel
        small_sort_per_tile_kernel
        radix_sort_per_tile_kernel
        pack_sorted_gaussians_kernel)
    string(REGEX MATCHALL "kernel void ${kernel_name}\\(" matches "${metal_source}")
    list(LENGTH matches match_count)
    if(NOT match_count EQUAL 1)
        message(FATAL_ERROR
            "Expected exactly one ${kernel_name} definition, found ${match_count}")
    endif()
    require_contains("${host_source}" "load(@\"${kernel_name}\")"
        "host pipeline load for ${kernel_name}")
endforeach()

require_contains("${metal_source}" "device atomic_uint* tile_count_storage"
    "selected atomic projection-count storage")
require_contains("${metal_source}"
    "inline bool training_render_tile_active("
    "shared coverage render-tile predicate")
string(REGEX MATCHALL
    "training_render_tile_active" active_tile_checks "${metal_source}")
list(LENGTH active_tile_checks active_tile_check_count)
if(NOT active_tile_check_count EQUAL 4)
    message(FATAL_ERROR
        "Expected one active-tile helper and three gates, found ${active_tile_check_count}")
endif()
require_contains("${metal_source}"
    "constant uchar* coverage_render_tiles   [[buffer(13)]]"
    "scatter coverage render-tile binding")
require_contains("${metal_source}"
    "device float* projected_opacities       [[buffer(12)]]"
    "per-Gaussian projected opacity output")
require_contains("${metal_source}"
    "projected_opacities[idx] = 1.f / (1.f + exp(-opacities[idx]));"
    "one sigmoid evaluation per visible Gaussian")
require_contains("${metal_source}" "EXACT_BITONIC_FAST_PATH 2048"
    "exact-range bitonic fast path")
require_contains("${layout_header}" "kExactBitonicFastPath = 2'048"
    "matching host bitonic fast-path boundary")
require_contains("${metal_source}" "EXACT_SMALL_SORT_TG_SIZE 32"
    "32-thread small-sort group")
require_contains("${metal_source}" "EXACT_SMALL_TILE_MAX 32"
    "small-sort shader threshold")
require_contains("${layout_header}" "kExactSmallTileMaximum = 32"
    "matching host small-tile threshold")
require_contains("${layout_header}"
    "tileIntersectionLayoutNeedsRadixScratch("
    "host radix-scratch decision")
require_contains("${layout_header}"
    "int32_t* tileBins = nullptr,\n    uint32_t* sortableTileIndices = nullptr"
    "optional compact-sort layout outputs")
require_contains("${layout_header}"
    "tileBins[2 * tile] = start;\n            tileBins[2 * tile + 1] = end;"
    "all-tile range initialization")
require_contains("${layout_header}"
    "uint32_t nextSmall = 0;\n        uint32_t nextMedium = layout.smallTileCount;"
    "small-first compact-sort bucket cursors")
require_contains("${layout_header}"
    "uint32_t nextLarge =\n            layout.smallTileCount + layout.mediumTileCount;"
    "large compact-sort bucket cursor")
require_contains("${layout_header}"
    "count > 1 && count <= kExactSmallTileMaximum"
    "small compact-sort threshold")
require_contains("${layout_header}"
    "count <= kExactBitonicFastPath &&\n                       count > kExactSmallTileMaximum"
    "medium compact-sort threshold")
require_contains("${layout_header}"
    "count > kExactBitonicFastPath"
    "large compact-sort threshold")
require_contains("${layout_header}"
    "sortableTileIndices[*next] = static_cast<uint32_t>(tile);"
    "bucket-ordered compact sortable-tile write")
require_contains("${metal_source}" "for (uint pass = 0; pass < 8; ++pass)"
    "eight full-key radix passes")
require_contains("${metal_source}" "const uint shift = pass * 8u"
    "full-key deterministic ordering")
require_contains("${metal_source}" "(uint64_t)end_i > (uint64_t)capacity"
    "exact-sort arena bounds check")
require_contains("${metal_source}"
    "constant uint& small_tile_count  [[buffer(7)]]"
    "small-sort compact-count binding")
require_contains("${metal_source}"
    "if (sort_index >= small_tile_count) return;"
    "small-sort dispatch bounds check")
string(FIND "${metal_source}"
    "kernel void small_sort_per_tile_kernel(" small_sort_kernel_position)
string(FIND "${metal_source}"
    "if (sort_index >= num_tiles) {" small_sort_list_bound_position)
string(FIND "${metal_source}"
    "const uint tile_id = sortable_tile_indices[sort_index];"
    small_sort_list_read_position)
if(small_sort_kernel_position EQUAL -1 OR
   small_sort_list_bound_position EQUAL -1 OR
   small_sort_list_read_position EQUAL -1 OR
   small_sort_list_bound_position LESS small_sort_kernel_position OR
   small_sort_list_bound_position GREATER small_sort_list_read_position)
    message(FATAL_ERROR
        "Small-sort compact-list bounds check must precede its tile-index read")
endif()
require_contains("${metal_source}"
    "if (count < 2 || count > EXACT_SMALL_TILE_MAX)"
    "small-sort bucket invariant")
require_contains("${metal_source}"
    "threadgroup uint64_t bitonic_data[EXACT_SMALL_SORT_TG_SIZE]"
    "bounded small-sort threadgroup storage")
require_contains("${metal_source}"
    "constant uint& sortable_tile_offset [[buffer(9)]]"
    "general-sort compact offset binding")
require_contains("${metal_source}"
    "if (sortable_tile_offset > num_tiles ||\n        sort_index >= num_tiles - sortable_tile_offset)"
    "overflow-safe general-sort list bounds check")
require_contains("${metal_source}"
    "sortable_tile_indices[sortable_tile_offset + sort_index]"
    "offset general-sort work-index mapping")
require_contains("${metal_source}" "constant uint& exact_count"
    "packing uses the exact live count")
require_contains("${metal_source}"
    "float3(xy.x, xy.y, projected_opacities[gaussian_id])"
    "packing reuses projected opacity")
string(REGEX MATCHALL
    "constant uint64_t\\* sorted_keys" backward_key_inputs "${metal_source}")
list(LENGTH backward_key_inputs backward_key_input_count)
if(NOT backward_key_input_count EQUAL 3)
    message(FATAL_ERROR
        "Expected sorted-key inputs for packing and both backward paths, found ${backward_key_input_count}")
endif()
string(REGEX MATCHALL
    "id_batch\\[tr\\] = \\(int32_t\\)\\(sorted_keys\\[idx\\] & 0xFFFFFFFFu\\)"
    backward_id_extracts "${metal_source}")
list(LENGTH backward_id_extracts backward_id_extract_count)
if(NOT backward_id_extract_count EQUAL 2)
    message(FATAL_ERROR
        "Expected two backward Gaussian-ID key extracts, found ${backward_id_extract_count}")
endif()
require_contains("${host_source}"
    "ENC_BUF(enc, tileCountStorage, 15)"
    "selected projection count binding")
require_contains("${host_source}"
    "ENC_BUF(enc, coverageRenderTileBuffer, 24);\n        ENC_SCALAR(enc, coverageRenderTileStride, 25);"
    "training projection coverage tile bindings")
require_contains("${host_source}"
    "ENC_BUF(enc, coverageRenderTileBuffer, 13);\n        ENC_SCALAR(enc, coverageRenderTileStride, 14);"
    "training scatter coverage tile bindings")
require_contains("${host_source}"
    "const uint32_t coverageRenderTileStrideDisabled = 0u;"
    "render pipeline disables coverage pruning")
require_contains("${host_source}" "ENC_BUF(enc, projected_opacities, 12)"
    "projected opacity binding")
require_contains("${host_source}"
    "ENC_BUF(enc, g_tcache.intersection_keys_a, 2);"
    "backward sorted-key binding")
require_contains("${host_source}" "ENC_SCALAR(enc, capacity_u32, 5)"
    "radix capacity binding")
require_contains("${host_source}" "bool needsRadixScratch,"
    "lazy radix-scratch arena input")
require_contains("${host_source}"
    "if (!needsRadixScratch || intersection_keys_b.defined())"
    "lazy radix-scratch allocation gate")
require_contains("${host_source}"
    "if (needsRadixScratch) {\n                intersection_keys_b ="
    "conditional radix-scratch allocation after arena growth")
require_contains("${host_source}"
    "sort tiles:     %u small32, %u bitonic, %u radix (%u / %u)"
    "bucket-specific exact-sort benchmark label")
require_contains("${host_source}"
    "intersectionLayout.smallTileCount,\n                intersectionLayout.mediumTileCount,\n                intersectionLayout.largeTileCount,"
    "bucket-specific exact-sort benchmark values")
string(REGEX MATCHALL
    "tileIntersectionLayoutNeedsRadixScratch\\(intersectionLayout\\)"
    radix_scratch_decisions "${host_source}")
list(LENGTH radix_scratch_decisions radix_scratch_decision_count)
if(NOT radix_scratch_decision_count EQUAL 2)
    message(FATAL_ERROR
        "Expected render and training radix-scratch decisions, found ${radix_scratch_decision_count}")
endif()
string(REGEX MATCHALL
    "ENC_BUF\\(enc, radix_sort_scratch_keys, 2\\)"
    radix_scratch_bindings "${host_source}")
list(LENGTH radix_scratch_bindings radix_scratch_binding_count)
if(NOT radix_scratch_binding_count EQUAL 2)
    message(FATAL_ERROR
        "Expected two safe radix-scratch bindings, found ${radix_scratch_binding_count}")
endif()
string(REGEX MATCHALL
    "radix_sort_scratch_keys ="
    radix_scratch_fallbacks "${host_source}")
list(LENGTH radix_scratch_fallbacks radix_scratch_fallback_count)
if(NOT radix_scratch_fallback_count EQUAL 2)
    message(FATAL_ERROR
        "Expected two radix-scratch fallback initializers, found ${radix_scratch_fallback_count}")
endif()
require_contains("${host_source}"
    "? g_tcache.intersection_keys_b\n            : g_tcache.intersection_keys_a;"
    "primary-key fallback when radix scratch is absent")
string(REGEX MATCHALL
    "if \\(needsRadixScratch && !g_tcache\\.intersection_keys_b\\.defined\\(\\)\\)"
    radix_scratch_invariants "${host_source}")
list(LENGTH radix_scratch_invariants radix_scratch_invariant_count)
if(NOT radix_scratch_invariant_count EQUAL 2)
    message(FATAL_ERROR
        "Expected render and training radix-scratch invariants, found ${radix_scratch_invariant_count}")
endif()
require_contains("${metal_source}"
    "        return;\n    }\n\n    threadgroup atomic_uint digit_histogram"
    "bitonic return before radix scratch use")
require_contains("${metal_source}"
    "device uint64_t* source = (pass & 1u) == 0u ? keys_a : keys_b;"
    "radix scratch runtime use")
require_contains("${metal_source}"
    "constant uint* sortable_tile_indices [[buffer(7)]]"
    "general-sort compact-tile shader input")
require_contains("${metal_source}"
    "const uint tile_id = sortable_tile_indices[sort_index];"
    "small-sort compact work-index mapping")
require_contains("${host_source}"
    "id<MTLComputePipelineState> small_sort_per_tile_kernel_cpso = nil;"
    "small-sort pipeline declaration")
require_contains("${host_source}"
    "scatter_to_exact_bins_kernel_cpso,\n            small_sort_per_tile_kernel_cpso,\n            radix_sort_per_tile_kernel_cpso,"
    "small-sort pipeline release")
require_contains("${host_source}"
    "ctx->small_sort_per_tile_kernel_cpso          = load(@\"small_sort_per_tile_kernel\");"
    "small-sort pipeline assignment")
require_contains("${host_source}"
    "small_sort_per_tile_kernel_cpso.maxTotalThreadsPerThreadgroup <\n        msplat::kExactSmallTileMaximum"
    "32-thread small-sort support check")
string(REGEX MATCHALL
    "const uint32_t small_sort_tile_count ="
    small_sort_count_initializers "${host_source}")
list(LENGTH small_sort_count_initializers small_sort_count_initializer_count)
if(NOT small_sort_count_initializer_count EQUAL 2)
    message(FATAL_ERROR
        "Expected render and training small-sort counts, found ${small_sort_count_initializer_count}")
endif()
string(REGEX MATCHALL
    "const uint32_t general_sort_tile_count ="
    general_sort_count_initializers "${host_source}")
list(LENGTH general_sort_count_initializers general_sort_count_initializer_count)
if(NOT general_sort_count_initializer_count EQUAL 2)
    message(FATAL_ERROR
        "Expected render and training general-sort counts, found ${general_sort_count_initializer_count}")
endif()
require_contains("${host_source}"
    "intersectionLayout.mediumTileCount + intersectionLayout.largeTileCount;"
    "medium-plus-large general-sort range")
string(REGEX MATCHALL
    "const uint32_t general_sort_tile_offset = small_sort_tile_count"
    general_sort_offset_initializers "${host_source}")
list(LENGTH general_sort_offset_initializers general_sort_offset_initializer_count)
if(NOT general_sort_offset_initializer_count EQUAL 2)
    message(FATAL_ERROR
        "Expected render and training general-sort offsets, found ${general_sort_offset_initializer_count}")
endif()
string(REGEX MATCHALL
    "if \\(small_sort_tile_count > 0\\)"
    small_sort_guards "${host_source}")
list(LENGTH small_sort_guards small_sort_guard_count)
if(NOT small_sort_guard_count EQUAL 1)
    message(FATAL_ERROR
        "Expected the render non-empty small-sort guard, found ${small_sort_guard_count}")
endif()
require_contains("${host_source}"
    "if (gpuResidentIntersectionAttempt || small_sort_tile_count > 0)"
    "training small-sort exact/retry guard")
string(REGEX MATCHALL
    "if \\(general_sort_tile_count > 0\\)"
    general_sort_guards "${host_source}")
list(LENGTH general_sort_guards general_sort_guard_count)
if(NOT general_sort_guard_count EQUAL 1)
    message(FATAL_ERROR
        "Expected the render non-empty general-sort guard, found ${general_sort_guard_count}")
endif()
require_contains("${host_source}"
    "if (gpuResidentIntersectionAttempt || general_sort_tile_count > 0)"
    "training general-sort exact/retry guard")
string(REGEX MATCHALL
    "ENC_BUF\\(enc, g_tcache\\.sortable_tile_indices, 6\\)"
    small_sort_tile_bindings "${host_source}")
list(LENGTH small_sort_tile_bindings small_sort_tile_binding_count)
if(NOT small_sort_tile_binding_count EQUAL 2)
    message(FATAL_ERROR
        "Expected two small compact-tile bindings, found ${small_sort_tile_binding_count}")
endif()
string(REGEX MATCHALL
    "ENC_BUF\\(enc, g_tcache\\.sortable_tile_indices, 7\\)"
    general_sort_tile_bindings "${host_source}")
list(LENGTH general_sort_tile_bindings general_sort_tile_binding_count)
if(NOT general_sort_tile_binding_count EQUAL 2)
    message(FATAL_ERROR
        "Expected two general compact-tile bindings, found ${general_sort_tile_binding_count}")
endif()
string(REGEX MATCHALL
    "ENC_SCALAR\\(enc, small_sort_tile_count, 7\\)"
    small_sort_count_bindings "${host_source}")
list(LENGTH small_sort_count_bindings small_sort_count_binding_count)
if(NOT small_sort_count_binding_count EQUAL 2)
    message(FATAL_ERROR
        "Expected two small-sort count bindings, found ${small_sort_count_binding_count}")
endif()
string(REGEX MATCHALL
    "ENC_SCALAR\\(enc, general_sort_tile_count, 8\\)"
    general_sort_count_bindings "${host_source}")
list(LENGTH general_sort_count_bindings general_sort_count_binding_count)
if(NOT general_sort_count_binding_count EQUAL 2)
    message(FATAL_ERROR
        "Expected two general-sort count bindings, found ${general_sort_count_binding_count}")
endif()
string(REGEX MATCHALL
    "ENC_SCALAR\\(enc, general_sort_tile_offset, 9\\)"
    general_sort_offset_bindings "${host_source}")
list(LENGTH general_sort_offset_bindings general_sort_offset_binding_count)
if(NOT general_sort_offset_binding_count EQUAL 2)
    message(FATAL_ERROR
        "Expected two general-sort offset bindings, found ${general_sort_offset_binding_count}")
endif()
string(REGEX MATCHALL
    "dispatchThreadgroups:MTLSizeMake\\(small_sort_tile_count, 1, 1\\)"
    small_sort_dispatches "${host_source}")
list(LENGTH small_sort_dispatches small_sort_dispatch_count)
if(NOT small_sort_dispatch_count EQUAL 1)
    message(FATAL_ERROR
        "Expected the render compact small-sort dispatch, found ${small_sort_dispatch_count}")
endif()
require_contains("${host_source}"
    "[enc dispatchThreadgroups:\n                        MTLSizeMake(small_sort_tile_count, 1, 1)"
    "training compact small-sort dispatch")
require_contains("${host_source}"
    "threadsPerThreadgroup:MTLSizeMake(\n                    msplat::kExactSmallTileMaximum, 1, 1)"
    "32-thread small-sort dispatch size")
string(REGEX MATCHALL
    "dispatchThreadgroups:MTLSizeMake\\(general_sort_tile_count, 1, 1\\)"
    general_sort_dispatches "${host_source}")
list(LENGTH general_sort_dispatches general_sort_dispatch_count)
if(NOT general_sort_dispatch_count EQUAL 1)
    message(FATAL_ERROR
        "Expected the render compact general-sort dispatch, found ${general_sort_dispatch_count}")
endif()
require_contains("${host_source}"
    "[enc dispatchThreadgroups:\n                        MTLSizeMake(general_sort_tile_count, 1, 1)"
    "training compact general-sort dispatch")
require_contains("${host_source}"
    "threadsPerThreadgroup:MTLSizeMake(256, 1, 1)"
    "256-thread general-sort dispatch size")
require_contains("${host_source}"
    "[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];\n        if (small_sort_tile_count > 0) {"
    "scatter-to-split-sort barrier")
require_contains("${host_source}"
    "msplat::kExactSmallTileMaximum, 1, 1)];\n        }\n        if (general_sort_tile_count > 0) {"
    "barrier-free small-to-general sort transition")
string(REGEX MATCHALL
    "if \\(total_intersections == 0\\) return"
    zero_intersection_returns "${host_source}")
list(LENGTH zero_intersection_returns zero_intersection_return_count)
if(NOT zero_intersection_return_count EQUAL 2)
    message(FATAL_ERROR
        "Expected two zero-intersection pack skips, found ${zero_intersection_return_count}")
endif()
string(REGEX MATCHALL
    "if \\(sortable_tile_count > 0\\)"
    aggregate_sort_barrier_guards "${host_source}")
list(LENGTH aggregate_sort_barrier_guards aggregate_sort_barrier_guard_count)
if(NOT aggregate_sort_barrier_guard_count EQUAL 2)
    message(FATAL_ERROR
        "Expected two aggregate sort-to-pack barrier guards, found ${aggregate_sort_barrier_guard_count}")
endif()
require_contains("${host_source}"
    "if (total_intersections == 0) return;\n        if (sortable_tile_count > 0) {\n            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];"
    "aggregate sort-to-pack barrier")
require_contains("${host_source}"
    "radix_sort_per_tile_kernel_cpso.maxTotalThreadsPerThreadgroup <"
    "256-thread radix support check")
require_contains("${host_source}"
    "validateTileIntersectionWorkLimit(intersectionLayout)"
    "host work-limit validation")

foreach(legacy_token IN ITEMS
        MAX_TILE_ELEMS
        scatter_to_prealloc_bins_kernel
        bitonic_sort_per_tile_kernel
        prealloc_bins)
    require_absent("${metal_source}" "${legacy_token}" "${legacy_token} in Metal")
    require_absent("${host_source}" "${legacy_token}" "${legacy_token} in host")
endforeach()

require_absent("${host_source}" "MTensor gaussian_ids;"
    "redundant Gaussian-ID arena")
require_absent("${host_source}" "ENC_BUF(enc, gaussian_ids"
    "redundant Gaussian-ID binding")
require_absent("${host_source}"
    "ENC_BUF(enc, g_tcache.intersection_keys_b, 2);"
    "unconditional radix-scratch binding")
require_absent("${host_source}" "retainRadixScratch"
    "radix scratch retained across a bitonic-only arena rebuild")
require_absent("${host_source}"
    "dispatchThreadgroups:MTLSizeMake(num_tiles, 1, 1)"
    "full-grid exact-sort dispatch")
require_absent("${host_source}"
    "dispatchThreadgroups:MTLSizeMake(sortable_tile_count, 1, 1)"
    "single-width compact exact-sort dispatch")
require_absent("${host_source}"
    "needsRadixScratch ? \"radix\" : \"bitonic\""
    "pre-split exact-sort benchmark label")
require_absent("${layout_header}"
    "sortableTileIndices[layout.sortableTileCount]"
    "input-ordered compact sortable-tile append")
require_absent("${metal_source}"
    "const float opacity = 1.f / (1.f + exp(-opacities[gaussian_id]));"
    "per-intersection opacity sigmoid")
