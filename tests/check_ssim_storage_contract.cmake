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

function(extract_section source_name start_marker end_marker output_name)
    string(FIND "${${source_name}}" "${start_marker}" section_start)
    if(section_start EQUAL -1)
        message(FATAL_ERROR "Compact SSIM section start missing: ${start_marker}")
    endif()
    string(SUBSTRING "${${source_name}}" ${section_start} -1 section_tail)
    string(FIND "${section_tail}" "${end_marker}" relative_section_end)
    if(relative_section_end EQUAL -1)
        message(FATAL_ERROR "Compact SSIM section end missing: ${end_marker}")
    endif()
    string(SUBSTRING "${section_tail}" 0 ${relative_section_end}
        section_contents)
    set(${output_name} "${section_contents}" PARENT_SCOPE)
endfunction()

function(require_barrier_between contents before after label)
    string(FIND "${contents}" "${before}" before_position)
    if(before_position EQUAL -1)
        message(FATAL_ERROR "Could not order SSIM scratch phases: ${label}")
    endif()
    string(SUBSTRING "${contents}" ${before_position} -1 phase_tail)
    string(FIND "${phase_tail}" "${after}" after_position)
    if(after_position EQUAL -1)
        message(FATAL_ERROR "Could not order SSIM scratch phases: ${label}")
    endif()
    string(SUBSTRING "${phase_tail}" 0 ${after_position} phase_source)
    string(FIND "${phase_source}" "threadgroup_barrier(" barrier_position)
    if(barrier_position EQUAL -1)
        message(FATAL_ERROR "SSIM scratch phase is missing a barrier: ${label}")
    endif()
endfunction()

require_contains("${host_source}" "bool fused_ssim_backward = true;"
    "fused SSIM default")
require_contains("${host_source}"
    "const char* ssimModeOverride = std::getenv(\"MSPLAT_SSIM_MODE\");"
    "SSIM mode environment override")
require_contains("${host_source}"
    "std::strcmp(ssimModeOverride, \"staged\") == 0"
    "explicit staged mode")
require_contains("${host_source}" "ctx->fused_ssim_backward = false;"
    "staged fallback activation")
require_contains("${host_source}"
    "std::strcmp(ssimModeOverride, \"fused\") == 0"
    "explicit fused mode")
require_contains("${host_source}" "ctx->fused_ssim_backward = true;"
    "fused mode activation")
require_contains("${host_source}"
    "MSPLAT_SSIM_MODE must be staged or fused"
    "invalid SSIM mode rejection")
extract_section(host_source "if (ctx->fused_ssim_backward) {"
    "ctx->photometric_adam_kernel_cpso" ssim_pipeline_selection)
require_contains("${ssim_pipeline_selection}"
    "load(@\"ssim_fused_v_fwd_bwd_kernel\")"
    "fused terminal pipeline")
require_contains("${ssim_pipeline_selection}"
    "load(@\"ssim_fused_v_fwd_h_bwd_kernel\")"
    "staged middle pipeline")
require_contains("${ssim_pipeline_selection}"
    "ctx->ssim_v_bwd_kernel_cpso = load(@\"ssim_v_bwd_kernel\")"
    "staged vertical-backward pipeline")

extract_section(host_source
    "void ensure_training_image(int ih, int iw, bool needsDerivativeBuffer,"
    "void ensure_pose_refinement(" training_image_cache)
require_contains("${training_image_cache}"
    "!needsDerivativeBuffer || ssim_deriv_h_buf.defined()"
    "mode-aware derivative readiness")
require_contains("${training_image_cache}" "if (needsDerivativeBuffer) {"
    "conditional derivative allocation")
extract_section(training_image_cache "if (needsDerivativeBuffer) {"
    "ssim_h_buf = mtensor_empty(" derivative_allocation)
require_contains("${derivative_allocation}" "ssim_deriv_h_buf = mtensor_empty("
    "compact derivative allocation")
require_contains("${derivative_allocation}"
    "dev, {(int64_t)ih, (int64_t)iw, 9}, DType::Float32);"
    "nine-float derivative shape")
require_contains("${training_image_cache}"
    "ssim_h_buf = mtensor_empty(\n            dev, {(int64_t)ih, (int64_t)iw, 15}, DType::Float32);"
    "fifteen-float horizontal workspace")
require_contains("${host_source}"
    "img_height, img_width, !ctx->fused_ssim_backward, ctx->device"
    "fused mode skips derivative storage")
require_contains("${host_source}" "MTensor &rendered_gradient = out_img;"
    "in-place rendered-gradient alias")
require_contains("${host_source}" "ENC_BUF(enc, rendered_gradient, 10);"
    "monolithic raster backward gradient input")
require_contains("${host_source}" "ENC_BUF(enc, rendered_gradient, 13);"
    "chunked raster backward gradient input")
require_contains("${metal_source}" "device float* rendered_gradient"
    "single in-place rendered-gradient buffer")
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
extract_section(metal_source "kernel void ssim_fused_v_fwd_h_bwd_kernel("
    "kernel void ssim_fused_v_fwd_bwd_kernel(" staged_middle)
extract_section(metal_source "kernel void ssim_fused_v_fwd_bwd_kernel("
    "kernel void ssim_h_bwd_kernel(" fused_terminal)

require_contains("${staged_middle}" "uint out = (py*W+px)*9 + c*3;"
    "staged compact derivative producer stride")
require_contains("${metal_source}" "uint hp = (gy * W + gx) * 9 + c * 3;"
    "staged compact derivative consumer stride")
string(FIND "${staged_middle}"
    "threadgroup float tg_scratch[TILE_PIXELS * 5];" fused_scratch_position)
string(FIND "${staged_middle}"
    "for (uint c = 0; c < 3; c++) {" fused_channel_loop_position)
if(fused_scratch_position EQUAL -1 OR fused_channel_loop_position EQUAL -1 OR
   fused_scratch_position GREATER fused_channel_loop_position)
    message(FATAL_ERROR
        "Fused SSIM scratch must be allocated once outside the channel loop")
endif()
string(FIND "${staged_middle}"
    "threadgroup float tg_f1" separate_derivative_scratch_position)
string(FIND "${staged_middle}"
    "threadgroup float tg_sum" separate_loss_scratch_position)
if(NOT separate_derivative_scratch_position EQUAL -1 OR
   NOT separate_loss_scratch_position EQUAL -1)
    message(FATAL_ERROR
        "Fused SSIM derivatives and loss must reuse the statistics scratch")
endif()
require_barrier_between("${staged_middle}"
    "cross_xy += w * tg_scratch[hp + 4];"
    "tg_scratch[i] = derivative_values[slot].x;"
    "statistics read before derivative overwrite")
require_barrier_between("${staged_middle}"
    "tg_scratch[2 * DERIV_PIXELS + i] = derivative_values[slot].z;"
    "h1 += w * tg_scratch[hp];"
    "derivative write before horizontal read")

require_contains("${fused_terminal}" "constexpr uint OUTPUT_W = 16;"
    "fused output width")
require_contains("${fused_terminal}" "constexpr uint OUTPUT_H = 8;"
    "fused output height")
require_contains("${fused_terminal}"
    "threadgroup float tg_scratch[STATS_PIXELS * 5];"
    "single fused scratch tile")
require_absent("${fused_terminal}" "deriv_h_buf"
    "fused derivative device buffer")
require_contains("${fused_terminal}" "const bool owns_loss_center ="
    "fused loss ownership gate")
require_contains("${fused_terminal}" "rx < SSIM_HALF_WIN + OUTPUT_W"
    "fused horizontal loss ownership")
require_contains("${fused_terminal}" "ry < SSIM_HALF_WIN + OUTPUT_H"
    "fused vertical loss ownership")
require_contains("${fused_terminal}" "phase_values[slot] = coverage * float3("
    "fused coverage-weighted SSIM derivative")
require_contains("${fused_terminal}" "training_target_rgb("
    "fused compact-target reads")
require_contains("${fused_terminal}"
    "(1.0f - final_Ts[center_pixel]) - target_alpha"
    "fused transparent alpha loss")
require_contains("${fused_terminal}" "const float v_l1 = coverage * ("
    "fused coverage-weighted L1 derivative")
require_contains("${fused_terminal}" "const float adjusted_gradient = inv_n * ("
    "fused final loss gradient")
require_contains("${fused_terminal}"
    "rendered_gradient[pixel_channel] = rendered_value < 1.0f"
    "fused upper-clamp render gradient")
require_contains("${fused_terminal}"
    "local_log_gain_gradient[c] = adjusted_gradient * rend_val;"
    "fused photometric chain rule")
require_contains("${fused_terminal}" "log_gain_gradient + c"
    "fused photometric device reduction")
require_contains("${fused_terminal}"
    "mem_flags::mem_threadgroup | mem_flags::mem_device"
    "fused device fence before in-place gradient writes")
require_barrier_between("${fused_terminal}"
    "cross_xy += w * tg_scratch[hp + 4];"
    "tg_scratch[i] = phase_values[slot].x;"
    "fused statistics read before raw derivative overwrite")
require_barrier_between("${fused_terminal}"
    "tg_scratch[2 * RAW_PIXELS + i] = phase_values[slot].z;"
    "value.x += w * tg_scratch[raw];"
    "fused raw derivative write before horizontal read")
require_barrier_between("${fused_terminal}"
    "value.z += w * tg_scratch[2 * RAW_PIXELS + raw];"
    "tg_scratch[H_PIXELS + i] = phase_values[slot].y;"
    "fused horizontal read before scratch overwrite")
require_barrier_between("${fused_terminal}"
    "tg_scratch[2 * H_PIXELS + i] = phase_values[slot].z;"
    "convolved.x += w * tg_scratch[hp];"
    "fused horizontal write before vertical read")
require_barrier_between("${fused_terminal}"
    "l1_sum += coverage * fabs("
    "const float rendered_value = rendered_gradient[pixel_channel];"
    "fused loss reads before in-place gradient writes")
require_barrier_between("${fused_terminal}"
    "rendered_gradient[pixel_channel] = rendered_value < 1.0f"
    "const float loss_contribution ="
    "fused in-place output before next-channel input")
require_barrier_between("${fused_terminal}"
    "tg_scratch[3 * THREADS_PER_GROUP + tr] = local_log_gain_gradient.z;"
    "for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1)"
    "fused loss and photometric planes before reduction")

extract_section(host_source "auto encode_loss_fwd_bwd ="
    "auto encode_rast_bwd =" host_loss)
require_contains("${host_loss}" "ENC_BUF(enc, rendered_gradient, 0);"
    "in-place SSIM input and output")
require_contains("${host_loss}"
    "ctx->ssim_fused_v_fwd_bwd_kernel_cpso"
    "fused terminal dispatch")
require_contains("${host_loss}"
    "MTLSize fusedTg = MTLSizeMake(16, 8, 1);"
    "fused 16x8 threadgroup")
require_contains("${host_loss}"
    "(img_width + 15) / 16, (img_height + 7) / 8, 1"
    "fused partial-edge dispatch coverage")
require_contains("${host_loss}" "ENC_BUF(enc, photometric_gradient, 11);"
    "fused photometric-gradient binding")
require_contains("${host_loss}"
    "ENC_SCALAR(enc, target_pixel_stride_bytes, 17);"
    "fused target-stride binding")
require_contains("${host_loss}"
    "ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso"
    "staged middle dispatch remains")
require_contains("${host_loss}" "ENC_BUF(enc, g_tcache.ssim_deriv_h_buf, 6);"
    "staged derivative producer binding remains")
require_contains("${host_loss}" "ctx->ssim_v_bwd_kernel_cpso"
    "staged vertical-backward dispatch remains")
require_contains("${host_loss}" "ENC_BUF(enc, g_tcache.ssim_deriv_h_buf, 2);"
    "staged derivative consumer binding remains")
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
