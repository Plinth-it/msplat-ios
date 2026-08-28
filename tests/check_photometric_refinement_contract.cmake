if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/model.cpp" model_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/msplat_api.mm" api_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Photometric refinement contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Photometric refinement leaked into ${label}")
    endif()
endfunction()

function(extract_section source_name start_marker end_marker output_name)
    string(FIND "${${source_name}}" "${start_marker}" section_start)
    if(section_start EQUAL -1)
        message(FATAL_ERROR "Photometric contract section start missing: ${start_marker}")
    endif()
    string(FIND "${${source_name}}" "${end_marker}" section_end)
    if(section_end EQUAL -1 OR section_end LESS_EQUAL section_start)
        message(FATAL_ERROR "Photometric contract section end missing: ${end_marker}")
    endif()
    math(EXPR section_length "${section_end} - ${section_start}")
    string(SUBSTRING "${${source_name}}" ${section_start} ${section_length}
        section_contents)
    set(${output_name} "${section_contents}" PARENT_SCOPE)
endfunction()

extract_section(metal_source "kernel void ssim_h_fwd_kernel("
    "kernel void ssim_v_fwd_kernel(" h_forward)
extract_section(metal_source "kernel void ssim_fused_v_fwd_h_bwd_kernel("
    "kernel void ssim_fused_v_fwd_bwd_kernel(" staged_middle)
extract_section(metal_source "kernel void ssim_fused_v_fwd_bwd_kernel("
    "kernel void ssim_h_bwd_kernel(" fused_terminal)
extract_section(metal_source "kernel void ssim_v_bwd_kernel("
    "kernel void photometric_adam_kernel(" v_backward)
extract_section(metal_source "kernel void photometric_adam_kernel("
    "// GPU Densification Kernels" photo_adam)

foreach(section IN ITEMS h_forward staged_middle fused_terminal v_backward)
    require_contains("${${section}}" "constant float* log_rgb_gains"
        "${section} log-RGB gain argument")
    require_contains("${${section}}" "constant uint& camera_gain_offset"
        "${section} canonical-camera offset")
    require_contains("${${section}}" "photometric_gain("
        "${section} gain application")
endforeach()
require_contains("${h_forward}" "rv = rendered[idx] * tg_gain[c];"
    "horizontal SSIM input adjustment")
require_contains("${staged_middle}"
    "rendered[(gpy*W+gpx)*3+c] * gain"
    "staged-middle L1 adjustment")
require_contains("${v_backward}"
    "(defer_renderer_clamp != 0u || rendered_value < 1.0f)"
    "staged deferred upper-clamp render gradient chain rule")
require_contains("${v_backward}"
    "local_log_gain_gradient[c] = adjusted_gradient * rend_val;"
    "staged log-gain gradient chain rule")
require_contains("${v_backward}" "tg_h1[c][other_row][other_column]"
    "staged threadgroup log-gain reduction in reused SSIM storage")
require_contains("${v_backward}" "atomic_fetch_add_explicit("
    "staged device log-gain accumulation")
require_contains("${v_backward}" "if (photometric_enabled != 0)"
    "staged disabled-path gain-gradient bypass")

require_contains("${fused_terminal}"
    "const float rendered_value = rendered_gradient[pixel_channel];"
    "fused terminal preserves saturated rendered value")
require_contains("${fused_terminal}"
    "const float rend_val = rendered_value * gain;"
    "fused terminal applies the photometric gain")
require_contains("${fused_terminal}"
    "(defer_renderer_clamp != 0u || rendered_value < 1.0f)"
    "fused terminal deferred upper-clamp render gradient chain rule")
require_contains("${fused_terminal}"
    "local_log_gain_gradient[c] = adjusted_gradient * rend_val;"
    "fused terminal log-gain gradient chain rule")
require_contains("${fused_terminal}"
    "tg_scratch[THREADS_PER_GROUP + tr] = local_log_gain_gradient.x;"
    "fused terminal photometric reduction plane")
require_contains("${fused_terminal}"
    "tg_scratch[2 * THREADS_PER_GROUP + tr] = local_log_gain_gradient.y;"
    "fused terminal green-gain reduction plane")
require_contains("${fused_terminal}"
    "tg_scratch[3 * THREADS_PER_GROUP + tr] = local_log_gain_gradient.z;"
    "fused terminal blue-gain reduction plane")
require_contains("${fused_terminal}"
    "for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1)"
    "fused terminal threadgroup reduction")
require_contains("${fused_terminal}" "log_gain_gradient + c"
    "fused terminal device log-gain accumulation")
require_contains("${fused_terminal}"
    "tg_scratch[(c + 1) * THREADS_PER_GROUP]"
    "fused terminal reduced photometric value")
require_contains("${fused_terminal}" "if (photometric_enabled != 0)"
    "fused terminal disabled-path gain-gradient bypass")

require_contains("${photo_adam}"
    "log_gain_gradient[channel] + regularization * parameter"
    "L2 regularization")
require_contains("${photo_adam}" "const uint index = camera_gain_offset + channel;"
    "per-camera optimizer row")
require_contains("${photo_adam}"
    "-max_abs_log_gain, max_abs_log_gain);"
    "bounded Adam update")

extract_section(host_source "auto encode_loss_fwd_bwd ="
    "auto encode_rast_bwd =" host_loss)
extract_section(host_loss "ctx->ssim_fused_v_fwd_bwd_kernel_cpso"
    "// Pass 2: Fused V fwd + H bwd" host_fused_terminal)
extract_section(host_loss "ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso"
    "// Pass 3: V bwd" host_staged_middle)
extract_section(host_loss "ctx->ssim_v_bwd_kernel_cpso"
    "if (photometric.enabled)" host_staged_backward)
require_contains("${host_loss}" "ENC_BUF(enc, photometricLogGains, 4);"
    "horizontal-pass gain bindings")
require_contains("${host_loss}" "ENC_SCALAR(enc, cameraGainOffset, 5);"
    "horizontal-pass canonical-camera binding")
require_contains("${host_loss}" "ENC_SCALAR(enc, photometricEnabled, 6);"
    "horizontal-pass opt-in binding")
require_contains("${host_fused_terminal}"
    "ENC_BUF(enc, photometricLogGains, 9);"
    "fused-terminal gain binding")
require_contains("${host_fused_terminal}"
    "ENC_SCALAR(enc, cameraGainOffset, 10);"
    "fused-terminal canonical-camera binding")
require_contains("${host_fused_terminal}"
    "ENC_BUF(enc, photometric_gradient, 11);"
    "fused-terminal gradient binding")
require_contains("${host_fused_terminal}"
    "ENC_SCALAR(enc, photometricEnabled, 12);"
    "fused-terminal opt-in binding")
require_contains("${host_staged_middle}"
    "ENC_BUF(enc, photometricLogGains, 10);"
    "staged-middle gain binding")
require_contains("${host_staged_middle}"
    "ENC_SCALAR(enc, cameraGainOffset, 11);"
    "staged-middle canonical-camera binding")
require_contains("${host_staged_middle}"
    "ENC_SCALAR(enc, photometricEnabled, 12);"
    "staged-middle opt-in binding")
require_contains("${host_staged_backward}"
    "ENC_BUF(enc, photometricLogGains, 8);"
    "staged-backward gain binding")
require_contains("${host_staged_backward}"
    "ENC_SCALAR(enc, cameraGainOffset, 9);"
    "staged-backward canonical-camera binding")
require_contains("${host_staged_backward}"
    "ENC_BUF(enc, photometric_gradient, 10);"
    "staged-backward gradient binding")
require_contains("${host_staged_backward}"
    "ENC_SCALAR(enc, photometricEnabled, 11);"
    "staged-backward opt-in binding")
require_contains("${host_loss}"
    "if (photometric.enabled) {\n            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];\n            [enc setComputePipelineState:ctx->photometric_adam_kernel_cpso];"
    "gradient barrier before opt-in Adam")
require_contains("${host_loss}" "ENC_SCALAR(enc, photometric.regularization, 10);"
    "regularization binding")
require_contains("${host_loss}" "ENC_SCALAR(enc, photometric.maxAbsLogGain, 11);"
    "gain-bound binding")

extract_section(host_source "auto do_blit_zero ="
    "auto encode_step_readback =" blit_zero)
require_contains("${blit_zero}" "fillBuffer:photometric_gradient.buffer()"
    "per-step gradient scratch reset")

extract_section(host_source "MTensor msplat_render("
    "static MTensor msplat_train_step_locked(" canonical_host_render)
extract_section(model_source "MTensor Model::render("
    "void Model::fullIteration(" canonical_model_render)
require_absent("${canonical_host_render}" "photometric"
    "canonical Metal render")
require_absent("${canonical_model_render}" "cameraLogGains"
    "canonical model render")
require_absent("${canonical_model_render}" "photometric"
    "canonical model render")

require_contains("${model_source}"
    "? static_cast<uint32_t>(cameraIndex)\n        : 0u;"
    "model canonical-camera optimizer row")
require_contains("${api_source}"
    "cam, impl->ds->trainIndices[camIdx], nextStep, target,"
    "C++ API canonical training-camera index")
