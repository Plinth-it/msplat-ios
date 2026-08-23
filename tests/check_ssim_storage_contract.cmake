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
