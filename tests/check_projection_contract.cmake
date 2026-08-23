if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Projection contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Stale projection contract remains: ${label}")
    endif()
endfunction()

string(REGEX MATCHALL "float rw = 1\\.f / p_hom\\.w" exact_reciprocals
    "${metal_source}")
list(LENGTH exact_reciprocals exact_reciprocal_count)
if(NOT exact_reciprocal_count EQUAL 2)
    message(FATAL_ERROR
        "Expected exact homogeneous reciprocal in forward and VJP, found ${exact_reciprocal_count}")
endif()

require_contains("${metal_source}"
    "return 0.5f * W * x + cx - 0.5;"
    "edge-coordinate to raster-index conversion")
require_contains("${metal_source}"
    "-(v_ndc.x * p_hom.x + v_ndc.y * p_hom.y) * rw * rw"
    "homogeneous-w cotangent")
require_contains("${metal_source}"
    "mat[0] * v_hom.x + mat[4] * v_hom.y + mat[12] * v_hom.w"
    "projection row-three x contribution")
require_contains("${metal_source}"
    "mat[1] * v_hom.x + mat[5] * v_hom.y + mat[13] * v_hom.w"
    "projection row-three y contribution")
require_contains("${metal_source}"
    "mat[2] * v_hom.x + mat[6] * v_hom.y + mat[14] * v_hom.w"
    "projection row-three z contribution")

require_contains("${metal_source}"
    "const float2 ratio = p_view.xy / p_view.z;"
    "unclamped EWA ratio")
require_contains("${metal_source}"
    "ratio.x >= -limit.x && ratio.x <= limit.x ? 1.f : 0.f"
    "EWA x clamp derivative")
require_contains("${metal_source}"
    "ratio.y >= -limit.y && ratio.y <= limit.y ? 1.f : 0.f"
    "EWA y clamp derivative")

string(REGEX MATCHALL
    "clamp_ewa_position\\(p_view, tan_fovx, tan_fovy\\)"
    clamp_uses "${metal_source}")
list(LENGTH clamp_uses clamp_use_count)
if(NOT clamp_use_count EQUAL 5)
    message(FATAL_ERROR
        "Expected two forward and three backward EWA clamp uses, found ${clamp_use_count}")
endif()

foreach(axis IN ITEMS x y)
    if(axis STREQUAL "x")
        set(focal "fx")
        set(j_index "0")
        set(position "x")
    else()
        set(focal "fy")
        set(j_index "1")
        set(position "y")
    endif()

    string(REGEX MATCHALL
        "-clamped\\.ratio_gradient\\.${axis} \\* ${focal}_rz2 \\* v_J\\[2\\]\\[${j_index}\\]"
        axis_cotangents "${metal_source}")
    list(LENGTH axis_cotangents axis_cotangent_count)
    if(NOT axis_cotangent_count EQUAL 3)
        message(FATAL_ERROR
            "Expected all three EWA ${axis} cotangents, found ${axis_cotangent_count}")
    endif()

    string(REGEX MATCHALL
        "\\(1\\.f \\+ clamped\\.ratio_gradient\\.${axis}\\) \\* ${focal} \\* p_view\\.${position} \\* rz3"
        depth_cotangents "${metal_source}")
    list(LENGTH depth_cotangents depth_cotangent_count)
    if(NOT depth_cotangent_count EQUAL 3)
        message(FATAL_ERROR
            "Expected all three EWA ${axis} depth cotangents, found ${depth_cotangent_count}")
    endif()
endforeach()

require_absent("${metal_source}" "p_hom.w + 1e-6f"
    "biased homogeneous reciprocal")
require_absent("${metal_source}" "-(v_ndc.x + v_ndc.y)"
    "homogeneous cotangent without projected coordinates")
