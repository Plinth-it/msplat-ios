if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/swift/Sources/Msplat/TrainingPlan.swift"
    training_plan_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Terminal backward contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Terminal backward contract violation: ${label}")
    endif()
endfunction()

function(extract_section source_name start_marker end_marker output_name)
    string(FIND "${${source_name}}" "${start_marker}" section_start)
    if(section_start EQUAL -1)
        message(FATAL_ERROR
            "Terminal backward section start missing: ${start_marker}")
    endif()
    string(SUBSTRING "${${source_name}}" ${section_start} -1 section_tail)
    string(FIND "${section_tail}" "${end_marker}" section_length)
    if(section_length EQUAL -1 OR section_length EQUAL 0)
        message(FATAL_ERROR
            "Terminal backward section end missing: ${end_marker}")
    endif()
    string(SUBSTRING "${section_tail}" 0 ${section_length}
        section_contents)
    set(${output_name} "${section_contents}" PARENT_SCOPE)
endfunction()

extract_section(host_source "struct FusedTensorCache {"
    "static FusedTensorCache g_tcache;" tensor_cache)
extract_section(host_source "auto encode_proj_sh_bwd_adam ="
    "// ========================== DISPATCH" terminal_host)
extract_section(host_source "auto do_blit_zero ="
    "auto encode_step_readback =" blit_zero)
extract_section(metal_source "kernel void sh_opacity_backward_adam_kernel("
    "struct GeometryAdamParams" appearance_kernel)
extract_section(metal_source "kernel void project_backward_adam_kernel("
    "// ===== Exact Tile Intersection Pipeline" geometry_kernel)

# The three geometry gradients are register-local and never enter the cache.
foreach(symbol IN ITEMS v_mean3d v_scale v_quat)
    require_absent("${tensor_cache}" "MTensor ${symbol}"
        "cached ${symbol} tensor")
    require_absent("${blit_zero}" "fillBuffer:${symbol}.buffer()"
        "${symbol} clear")
endforeach()
foreach(symbol IN ITEMS
        fused_adam_kernel_cpso
        accumulate_grad_stats_kernel_cpso
        encode_grad_stats
        adam_grads)
    require_absent("${host_source}" "${symbol}" "obsolete host ${symbol}")
endforeach()
require_absent("${metal_source}" "kernel void fused_adam_kernel("
    "standalone Adam kernel")
require_absent("${metal_source}" "kernel void accumulate_grad_stats_kernel("
    "standalone statistics kernel")
require_absent("${host_source}" "\"grad_stats\""
    "standalone statistics profiling stage")

# Appearance runs before in-place mean updates. SH retains its visibility and
# active-degree gates; opacity preserves full-population zero-gradient decay.
require_contains("${terminal_host}"
    "setComputePipelineState:ctx->sh_opacity_backward_adam_kernel_cpso"
    "appearance terminal dispatch")
require_contains("${terminal_host}"
    "SH reads means from the pre-update model"
    "appearance-before-geometry barrier rationale")
require_contains("${terminal_host}"
    "setComputePipelineState:ctx->project_backward_adam_kernel_cpso"
    "geometry terminal dispatch")
require_absent("${appearance_kernel}" "if (!active) return;"
    "appearance early return before Adam")
require_contains("${appearance_kernel}"
    "const float opacity_gradient = active ? v_opacity[idx] : 0.0f;"
    "zero-gradient opacity decay")
require_contains("${appearance_kernel}"
    "if (active && degree >= 1 && degrees_to_use >= 1)"
    "degree-one SH activation gate")
require_contains("${appearance_kernel}"
    "if (active && degree >= 4 && degrees_to_use >= 4)"
    "degree-four SH backward support")

# Geometry VJPs stay local, but Adam still runs for every in-bounds Gaussian.
require_contains("${geometry_kernel}"
    "float local_v_mean3d[3] = {0.0f, 0.0f, 0.0f};"
    "zero-initialized mean gradient")
require_contains("${geometry_kernel}"
    "if (!in_bounds) return;"
    "post-pose bounds return")
require_absent("${geometry_kernel}" "if (!active) return;"
    "geometry early return before Adam")
require_contains("${geometry_kernel}"
    "means3d[parameter_index], mean_exp_avg[parameter_index]"
    "inline mean Adam")
require_contains("${geometry_kernel}"
    "scales[parameter_index], scale_exp_avg[parameter_index]"
    "inline scale Adam")
require_contains("${geometry_kernel}"
    "quats[parameter_index], quat_exp_avg[parameter_index]"
    "inline quaternion Adam")

# Densification accumulation is visibility-gated and has one writer per point.
require_contains("${geometry_kernel}"
    "if (collect_densification_stats != 0 && active)"
    "statistics gate")
require_contains("${geometry_kernel}" "vis_counts[idx] += 1.0f;"
    "visibility accumulation")
require_contains("${geometry_kernel}"
    "xys_grad_norm[idx] += sqrt(gx * gx + gy * gy);"
    "screen-gradient norm accumulation")
require_contains("${geometry_kernel}"
    "max_2d_size[idx], (float)radii[idx] * inv_max_dim"
    "maximum screen-radius accumulation")
require_contains("${host_source}"
    "? vis_counts : v_opacity;"
    "disabled-statistics dummy binding")

require_contains("${training_plan_source}" "[88, gaussianCount]"
    "reduced per-Gaussian training-cache estimate")
