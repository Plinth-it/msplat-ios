if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)

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
require_contains("${metal_source}" "EXACT_BITONIC_FAST_PATH 2048"
    "exact-range bitonic fast path")
require_contains("${metal_source}" "for (uint pass = 0; pass < 8; ++pass)"
    "eight full-key radix passes")
require_contains("${metal_source}" "const uint shift = pass * 8u"
    "full-key deterministic ordering")
require_contains("${metal_source}" "(uint64_t)end_i > (uint64_t)capacity"
    "radix arena bounds check")
require_contains("${metal_source}" "constant uint& exact_count"
    "packing uses the exact live count")
require_contains("${host_source}"
    "ENC_BUF(enc, g_tcache.tile_scatter_counters, 15)"
    "projection count binding")
require_contains("${host_source}" "ENC_SCALAR(enc, capacity_u32, 5)"
    "radix capacity binding")
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
