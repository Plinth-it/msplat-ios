if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/swift/Sources/Msplat/TrainingPlan.swift" planner_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Compact SSIM contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Stale SSIM storage contract remains: ${label}")
    endif()
endfunction()

function(require_barrier_between contents before after label)
    string(FIND "${contents}" "${before}" before_position)
    string(FIND "${contents}" "${after}" after_position)
    if(before_position EQUAL -1 OR after_position EQUAL -1 OR
       after_position LESS_EQUAL before_position)
        message(FATAL_ERROR "Could not order SSIM scratch phases: ${label}")
    endif()
    math(EXPR phase_length "${after_position} - ${before_position}")
    string(SUBSTRING "${contents}" ${before_position}
        ${phase_length} phase_source)
    string(FIND "${phase_source}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);" barrier_position)
    if(barrier_position EQUAL -1)
        message(FATAL_ERROR "SSIM scratch phase is missing a barrier: ${label}")
    endif()
endfunction()

require_contains("${host_source}" "ssim_deriv_h_buf = mtensor_empty("
    "compact derivative allocation")
require_contains("${host_source}"
    "dev, {(int64_t)ih, (int64_t)iw, 9}, DType::Float32);"
    "nine-float derivative shape")
require_contains("${host_source}"
    "ssim_h_buf = mtensor_empty(\n            dev, {(int64_t)ih, (int64_t)iw, 15}, DType::Float32);"
    "fifteen-float horizontal workspace")
require_contains("${host_source}" "MTensor &rendered_gradient = out_img;"
    "in-place rendered-gradient alias")
require_contains("${host_source}" "ENC_BUF(enc, rendered_gradient, 0);"
    "in-place SSIM V-backward input and output")
require_contains("${host_source}" "ENC_BUF(enc, rendered_gradient, 10);"
    "monolithic raster backward gradient input")
require_contains("${host_source}" "ENC_BUF(enc, rendered_gradient, 13);"
    "chunked raster backward gradient input")
require_contains("${metal_source}" "device float* rendered_gradient"
    "single in-place rendered-gradient buffer")
require_contains("${metal_source}" "uint out = (py*W+px)*9 + c*3;"
    "compact derivative producer stride")
require_contains("${metal_source}" "uint hp = (gy * W + gx) * 9 + c * 3;"
    "compact derivative consumer stride")
require_contains("${host_source}"
    "const MTensor& loss_coverage_buffer = coverage_mask ? *coverage_mask : gt;"
    "allocation-free unmasked coverage buffer")
require_contains("${host_source}"
    "const std::array<uint32_t, 2> coverage_layout ="
    "coverage byte-layout contract")
require_contains("${host_source}"
    "std::array<uint32_t, 2>{0u, 0u}"
    "zero-stride unmasked or transparent coverage contract")
require_contains("${host_source}"
    "ENC_BUF(enc, loss_coverage_buffer, 8);"
    "fused forward coverage binding")
require_contains("${host_source}"
    "ENC_SCALAR(enc, coverage_layout, 9);"
    "fused forward coverage layout binding")
require_contains("${host_source}"
    "ENC_BUF(enc, loss_coverage_buffer, 6);"
    "backward coverage binding")
require_contains("${host_source}"
    "ENC_SCALAR(enc, coverage_layout, 7);"
    "backward coverage layout binding")
require_contains("${metal_source}"
    "training_mask_coverage("
    "shared packed and standalone mask reader")
require_contains("${metal_source}"
    "ssim_weight * (coverage_sum - ssim_sum) / 3.0f"
    "coverage-weighted SSIM center loss")
require_contains("${metal_source}"
    "derivative_values[slot] = coverage *"
    "coverage-weighted SSIM center derivative")
require_contains("${metal_source}"
    "float v_l1 = coverage *"
    "coverage-weighted direct L1 derivative")
string(FIND "${metal_source}"
    "kernel void ssim_fused_v_fwd_h_bwd_kernel(" fused_kernel_start)
string(FIND "${metal_source}"
    "kernel void ssim_h_bwd_kernel(" fused_kernel_end)
if(fused_kernel_start EQUAL -1 OR fused_kernel_end EQUAL -1 OR
   fused_kernel_end LESS_EQUAL fused_kernel_start)
    message(FATAL_ERROR "Could not isolate fused SSIM middle pass")
endif()
math(EXPR fused_kernel_length "${fused_kernel_end} - ${fused_kernel_start}")
string(SUBSTRING "${metal_source}" ${fused_kernel_start}
    ${fused_kernel_length} fused_kernel_source)
string(FIND "${fused_kernel_source}"
    "threadgroup float tg_scratch[TILE_PIXELS * 5];" fused_scratch_position)
string(FIND "${fused_kernel_source}"
    "for (uint c = 0; c < 3; c++) {" fused_channel_loop_position)
if(fused_scratch_position EQUAL -1 OR fused_channel_loop_position EQUAL -1 OR
   fused_scratch_position GREATER fused_channel_loop_position)
    message(FATAL_ERROR
        "Fused SSIM scratch must be allocated once outside the channel loop")
endif()
string(FIND "${fused_kernel_source}"
    "threadgroup float tg_f1" separate_derivative_scratch_position)
string(FIND "${fused_kernel_source}"
    "threadgroup float tg_sum" separate_loss_scratch_position)
if(NOT separate_derivative_scratch_position EQUAL -1 OR
   NOT separate_loss_scratch_position EQUAL -1)
    message(FATAL_ERROR
        "Fused SSIM derivatives and loss must reuse the statistics scratch")
endif()
require_barrier_between("${fused_kernel_source}"
    "cross_xy += w * tg_scratch[hp + 4];"
    "tg_scratch[i] = derivative_values[slot].x;"
    "statistics read before derivative overwrite")
require_barrier_between("${fused_kernel_source}"
    "tg_scratch[2 * DERIV_PIXELS + i] = derivative_values[slot].z;"
    "h1 += w * tg_scratch[hp];"
    "derivative write before horizontal read")
require_contains("${host_source}"
    "static_cast<double>(rawLoss) * 255.0"
    "coverage-unit telemetry normalization")
require_contains("${planner_source}" "[116, pixelCount]"
    "116-byte per-pixel planner coefficient")

string(REGEX MATCHALL
    "dispatchThreadgroups:threadgroups threadsPerThreadgroup:tg"
    full_group_dispatches "${host_source}")
list(LENGTH full_group_dispatches full_group_dispatch_count)
if(NOT full_group_dispatch_count EQUAL 3)
    message(FATAL_ERROR
        "Expected three full-threadgroup SSIM dispatches, found ${full_group_dispatch_count}")
endif()

require_absent("${host_source}" "dispatchThreads:grid threadsPerThreadgroup:tg"
    "partial SSIM dispatch")
require_absent("${host_source}" "loss_intermediates"
    "15-float derivative workspace name")
require_absent("${host_source}" "g_tcache.v_rendered"
    "separate rendered-gradient allocation")
require_absent("${host_source}" "ENC_BUF(enc, rendered_gradient, 6)"
    "redundant aliased output binding")
require_absent("${host_source}" "lossPixelCount"
    "pixel-count-only loss telemetry denominator")
require_absent("${metal_source}"
    "float pixel_loss = (px < W && py < H)\n        ? ssim_weight * (coverage_sum"
    "partial-threadgroup loss gate")
