if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/CMakeLists.txt" cmake_source)
file(READ "${MSPLAT_SOURCE_DIR}/README.md" readme_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR
            "Intersection-attribute gather contract missing: ${label}")
    endif()
endfunction()

function(require_not_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR
            "Intersection-attribute gather contract retained: ${label}")
    endif()
endfunction()

function(require_substring_count contents needle expected label)
    set(remainder "${contents}")
    string(LENGTH "${needle}" needle_length)
    set(match_count 0)
    while(TRUE)
        string(FIND "${remainder}" "${needle}" position)
        if(position EQUAL -1)
            break()
        endif()
        math(EXPR next_position "${position} + ${needle_length}")
        string(SUBSTRING "${remainder}" ${next_position} -1 remainder)
        math(EXPR match_count "${match_count} + 1")
    endwhile()
    if(NOT match_count EQUAL expected)
        message(FATAL_ERROR
            "Intersection-attribute gather contract expected ${expected} ${label}, found ${match_count}")
    endif()
endfunction()

function(extract_test_properties test_name output_name)
    set(start_marker "set_tests_properties(${test_name} PROPERTIES")
    string(FIND "${cmake_source}" "${start_marker}" section_start)
    if(section_start EQUAL -1)
        message(FATAL_ERROR
            "Intersection-attribute gather test properties missing: ${test_name}")
    endif()
    string(SUBSTRING "${cmake_source}" ${section_start} -1 section_tail)
    set(end_marker "SKIP_RETURN_CODE 77)")
    string(FIND "${section_tail}" "${end_marker}" section_end)
    if(section_end EQUAL -1)
        message(FATAL_ERROR
            "Intersection-attribute gather test properties incomplete: ${test_name}")
    endif()
    string(LENGTH "${end_marker}" end_length)
    math(EXPR section_length "${section_end} + ${end_length}")
    string(SUBSTRING "${section_tail}" 0 ${section_length} section_contents)
    set(${output_name} "${section_contents}" PARENT_SCOPE)
endfunction()

# Selection is immutable with the Metal context and defaults to key-driven
# gathers. Packed attributes remain available as a strict fallback override.
require_contains("${host_source}"
    "bool gather_intersection_attributes = true;"
    "default gather context state")
require_contains("${host_source}"
    "std::getenv(\"MSPLAT_INTERSECTION_ATTRIBUTES\")"
    "mode environment lookup")
require_contains("${host_source}"
    "intersectionAttributesOverride, \"packed\") == 0) {\n            ctx->gather_intersection_attributes = false;"
    "packed fallback selection")
foreach(mode IN ITEMS packed gather)
    require_contains("${host_source}"
        "intersectionAttributesOverride, \"${mode}\") == 0"
        "${mode} mode selection")
endforeach()
require_contains("${host_source}"
    "MSPLAT_INTERSECTION_ATTRIBUTES must be packed or gather"
    "invalid-mode rejection")

# Gather mode retains only the canonical sorted key arena (and optional radix
# scratch); the three float3 arrays are allocated only for packed mode.
require_contains("${host_source}"
    "bool needsPackedAttributes,"
    "mode-aware arena input")
require_contains("${host_source}"
    "if (!needsPackedAttributes) {\n            packed_xy_opac.reset();\n            packed_conic.reset();\n            packed_rgb.reset();"
    "gather packed-buffer release")
require_contains("${host_source}"
    "const bool packedAttributesReady = !needsPackedAttributes ||"
    "mode-aware arena readiness")
require_contains("${host_source}"
    "const uint64_t bytesPerLargestBuffer = needsPackedAttributes\n            ? 3u * sizeof(float)\n            : sizeof(uint64_t);"
    "mode-aware Metal buffer limit")
require_contains("${host_source}"
    "if (needsPackedAttributes) {\n                packed_xy_opac ="
    "conditional packed allocation")
require_substring_count("${host_source}"
    "!ctx->gather_intersection_attributes, ctx->device)" 3
    "render, training, and retry arena mode bindings")
require_substring_count("${host_source}"
    "if (ctx->gather_intersection_attributes) return;" 1
    "render pack-dispatch skip")
require_contains("${host_source}"
    "if (ctx->gather_intersection_attributes ||\n            (!gpuResidentIntersectionAttempt && total_intersections == 0u))"
    "training pack-dispatch skip")
require_contains("${host_source}"
    "size_t msplat_packed_intersection_attribute_bytes()"
    "runtime packed-attribute accounting")

# The shared cooperative loader preserves opacity/color semantics while using
# the low key word only as the per-Gaussian source index in gather mode.
require_contains("${metal_source}"
    "inline RasterIntersectionAttributes load_raster_intersection_attributes("
    "shared raster attribute loader")
require_contains("${metal_source}"
    "sorted_keys[intersection_index] & 0xffffffffu"
    "Gaussian-ID extraction")
require_contains("${metal_source}"
    "read_packed_float2(xy_attributes, source_index)"
    "per-Gaussian XY gather")
require_contains("${metal_source}"
    "xy.x, xy.y, projected_opacities[source_index]"
    "projected sigmoid opacity gather")
require_contains("${metal_source}"
    "attributes.conic = read_packed_float3(conic_attributes, source_index);"
    "per-Gaussian conic gather")
require_contains("${metal_source}"
    "attributes.rgb = read_packed_float3(rgb_attributes, source_index);"
    "raw projected color gather")
require_contains("${metal_source}"
    "max(attributes.rgb + 0.5f, 0.0f)"
    "forward raw-color clamp preservation")

# All active forward/backward ABIs receive the key/layout inputs. Forward needs
# a new key binding; backward reuses its existing key and appends opacity/mode.
foreach(binding IN ITEMS
        "constant uint64_t* sorted_keys [[buffer(12)]]"
        "constant float* projected_opacities [[buffer(13)]]"
        "constant uint& attribute_layout [[buffer(14)]]")
    require_contains("${metal_source}" "${binding}"
        "monolithic forward ${binding}")
endforeach()
require_contains("${metal_source}"
    "constant uint2& blockDim,\n    constant uint64_t* sorted_keys,\n    constant float* projected_opacities,\n    constant uint& attribute_layout,"
    "chunked forward gather ABI")
require_contains("${metal_source}"
    "constant float& alpha_gradient_scale,\n    constant float* projected_opacities,\n    constant uint& attribute_layout,"
    "active backward gather ABI")
require_substring_count("${metal_source}"
    "load_raster_intersection_attributes(" 5
    "loader definition plus four active raster consumers")

foreach(binding IN ITEMS
        "ENC_BUF(enc, g_tcache.intersection_keys_a, 12);"
        "ENC_BUF(enc, projected_opacities, 13);"
        "ENC_SCALAR(enc, intersection_attribute_layout, 14);"
        "ENC_BUF(enc, g_tcache.intersection_keys_a, 13);"
        "ENC_BUF(enc, projected_opacities, 14);"
        "ENC_SCALAR(enc, intersection_attribute_layout, 15);"
        "ENC_BUF(enc, projected_opacities, 18);"
        "ENC_SCALAR(enc, intersection_attribute_layout, 19);"
        "ENC_BUF(enc, projected_opacities, 23);"
        "ENC_SCALAR(enc, intersection_attribute_layout, 24);")
    require_contains("${host_source}" "${binding}" "host ${binding}")
endforeach()
require_substring_count("${host_source}"
    "MTensor &raster_xy_attributes = ctx->gather_intersection_attributes" 2
    "render and training source aliases")

# The primary runtime case exercises the unset gather default. Separate
# processes cover the explicit packed fallback, gather chunking, and retry.
require_contains("${cmake_source}"
    "MSPLAT_INTERSECTION_ATTRIBUTES=packed"
    "packed fallback runtime case")
extract_test_properties("msplat_transparent_training" default_properties)
require_not_contains("${default_properties}"
    "MSPLAT_INTERSECTION_ATTRIBUTES="
    "default runtime attribute override")
foreach(test_name IN ITEMS
        msplat_transparent_training_gather
        msplat_transparent_training_gather_arena_retry
        msplat_terminal_backward_gather)
    require_contains("${cmake_source}" "${test_name}"
        "${test_name} runtime case")
    extract_test_properties("${test_name}" test_properties)
    require_contains("${test_properties}"
        "MSPLAT_INTERSECTION_ATTRIBUTES=gather"
        "${test_name} gather environment")
endforeach()
extract_test_properties(
    "msplat_transparent_training_gather_arena_retry" retry_properties)
require_contains("${retry_properties}"
    "MSPLAT_TRAINING_ARENA_MODE=retry"
    "gather retry environment")
require_contains("${readme_source}"
    "`MSPLAT_INTERSECTION_ATTRIBUTES=packed`"
    "documented packed fallback")
