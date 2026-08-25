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

require_contains("${metal_source}" "device atomic_uint* tile_counts"
    "atomic projection counts")
require_contains("${metal_source}"
    "inline bool training_render_tile_active("
    "shared coverage render-tile predicate")
string(REGEX MATCHALL
    "training_render_tile_active" active_tile_checks "${metal_source}")
list(LENGTH active_tile_checks active_tile_check_count)
if(NOT active_tile_check_count EQUAL 3)
    message(FATAL_ERROR
        "Expected one active-tile helper and two gates, found ${active_tile_check_count}")
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
require_contains("${layout_header}"
    "tileIntersectionLayoutNeedsRadixScratch("
    "host radix-scratch decision")
require_contains("${metal_source}" "for (uint pass = 0; pass < 8; ++pass)"
    "eight full-key radix passes")
require_contains("${metal_source}" "const uint shift = pass * 8u"
    "full-key deterministic ordering")
require_contains("${metal_source}" "(uint64_t)end_i > (uint64_t)capacity"
    "radix arena bounds check")
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
    "ENC_BUF(enc, g_tcache.tile_scatter_counters, 15)"
    "projection count binding")
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
    "needsRadixScratch ? \"radix\" : \"bitonic\""
    "accurate exact-sort benchmark label")
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
require_absent("${metal_source}"
    "const float opacity = 1.f / (1.f + exp(-opacities[gaussian_id]));"
    "per-intersection opacity sigmoid")
