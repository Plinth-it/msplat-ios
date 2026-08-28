if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/model.cpp" model_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/msplat_api.mm" api_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/include/msplat_c_api.h" c_api_source)
file(READ "${MSPLAT_SOURCE_DIR}/swift/Sources/Msplat/TrainingConfig.swift" swift_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Transparent-mask contract missing: ${label}")
    endif()
endfunction()

function(require_not_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Transparent-mask contract violation: ${label}")
    endif()
endfunction()

function(extract_section source_name start_marker end_marker output_name)
    string(FIND "${${source_name}}" "${start_marker}" section_start)
    if(section_start EQUAL -1)
        message(FATAL_ERROR "Transparent-mask section start missing: ${start_marker}")
    endif()
    string(FIND "${${source_name}}" "${end_marker}" section_end)
    if(section_end EQUAL -1 OR section_end LESS_EQUAL section_start)
        message(FATAL_ERROR "Transparent-mask section end missing: ${end_marker}")
    endif()
    math(EXPR section_length "${section_end} - ${section_start}")
    string(SUBSTRING "${${source_name}}" ${section_start} ${section_length}
        section_contents)
    set(${output_name} "${section_contents}" PARENT_SCOPE)
endfunction()

extract_section(metal_source "kernel void rasterize_backward_kernel("
    "kernel void nd_rasterize_backward_kernel(" raster_backward)
extract_section(metal_source "kernel void rasterize_backward_chunked_kernel("
    "// ============================================================================\n// Separable SSIM loss kernels" chunked_backward)
extract_section(metal_source "kernel void ssim_h_fwd_kernel("
    "kernel void ssim_v_fwd_kernel(" h_forward)
extract_section(metal_source "kernel void ssim_fused_v_fwd_h_bwd_kernel("
    "kernel void ssim_fused_v_fwd_bwd_kernel(" staged_middle)
extract_section(metal_source "kernel void ssim_fused_v_fwd_bwd_kernel("
    "kernel void ssim_h_bwd_kernel(" fused_terminal)
extract_section(metal_source "kernel void ssim_v_bwd_kernel("
    "kernel void photometric_adam_kernel(" v_backward)

foreach(section IN ITEMS raster_backward chunked_backward)
    require_contains("${${section}}" "constant uchar* training_mask"
        "${section} mask input")
    require_contains("${${section}}" "if (valid) {"
        "${section} capped-alpha outer gradient gate")
    require_contains("${${section}}" "T *= ra;"
        "${section} capped-alpha transmittance recovery")
    require_contains("${${section}}" "v_rgb_local = fac * v_out;"
        "${section} capped-alpha color gradient")
    require_contains("${${section}}" "if (alpha < 0.999f) {"
        "${section} capped-alpha sigma-gradient gate")
    require_not_contains("${${section}}" "valid && alpha < 0.999f"
        "${section} whole-gradient cap gate")
    require_contains("${${section}}" "v_alpha_pixel * T_final * ra"
        "${section} alpha opacity VJP")
endforeach()
require_not_contains("${chunked_backward}"
    "bin_final < chunk_start) {\n        return;"
    "chunked pixels must not return before threadgroup barriers")

foreach(section IN ITEMS h_forward staged_middle fused_terminal v_backward)
    require_contains("${${section}}" "alpha_layout"
        "${section} transparent-mode gate")
    require_contains("${${section}}" "training_target_rgb("
        "${section} background-composited RGB target")
endforeach()
foreach(section IN ITEMS staged_middle fused_terminal)
    require_contains("${${section}}"
        "if (c == 0 && alpha_layout.x != 0)"
        "${section} alpha-loss mode gate")
    require_contains("${${section}}"
        "(1.0f - final_Ts[center_pixel]) - target_alpha"
        "${section} full-frame rendered-alpha loss")
    require_contains("${${section}}" "alpha_loss_sum"
        "${section} alpha loss reduction")
endforeach()
require_contains("${fused_terminal}"
    "phase_values[slot] = coverage * float3("
    "fused terminal coverage-weighted SSIM derivative")
require_contains("${fused_terminal}" "const float v_l1 = coverage * ("
    "fused terminal coverage-weighted L1 derivative")
require_contains("${fused_terminal}"
    "ssim_weight * (coverage_sum - ssim_sum) / 3.0f"
    "fused terminal coverage-weighted SSIM loss")
require_contains("${fused_terminal}" "alpha_loss_sum;"
    "fused terminal alpha loss joins the reduced loss")

extract_section(host_source "auto encode_loss_fwd_bwd ="
    "auto encode_rast_bwd =" host_loss)
extract_section(host_source "auto encode_rast_bwd ="
    "// Packed optimizer hyperparameters" host_raster_backward)
extract_section(host_loss "ctx->ssim_fused_v_fwd_bwd_kernel_cpso"
    "// Pass 2: Fused V fwd + H bwd" host_fused_terminal)
extract_section(host_loss "ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso"
    "// Pass 3: V bwd" host_staged_middle)
extract_section(host_loss "ctx->ssim_v_bwd_kernel_cpso"
    "if (photometric.enabled)" host_staged_backward)
require_contains("${host_loss}" "ENC_BUF(enc, loss_coverage_buffer, 7);"
    "horizontal loss bindings")
require_contains("${host_loss}" "ENC_SCALAR(enc, alpha_layout, 8);"
    "horizontal transparent-target layout binding")
require_contains("${host_loss}" "ENC_BUF(enc, background, 9);"
    "horizontal transparent-target background binding")
foreach(section IN ITEMS host_fused_terminal host_staged_middle)
    require_contains("${${section}}" "ENC_SCALAR(enc, alpha_layout, 13);"
        "${section} alpha layout binding")
    require_contains("${${section}}" "ENC_BUF(enc, background, 14);"
        "${section} alpha background binding")
    require_contains("${${section}}" "ENC_BUF(enc, final_Ts, 15);"
        "${section} final-transmittance binding")
    require_contains("${${section}}" "ENC_SCALAR(enc, alpha_loss_weight, 16);"
        "${section} alpha-loss weight binding")
endforeach()
require_contains("${host_staged_backward}"
    "ENC_SCALAR(enc, alpha_layout, 12);"
    "staged loss-backward target layout binding")
require_contains("${host_staged_backward}" "ENC_BUF(enc, background, 13);"
    "staged loss-backward background binding")
require_contains("${host_raster_backward}"
    "ENC_BUF(enc, loss_coverage_buffer, 15);\n            ENC_SCALAR(enc, alpha_layout, 16);\n            ENC_SCALAR(enc, alpha_gradient_scale, 17);"
    "monolithic raster alpha bindings")
require_contains("${host_raster_backward}"
    "ENC_BUF(enc, loss_coverage_buffer, 20);\n            ENC_SCALAR(enc, alpha_layout, 21);\n            ENC_SCALAR(enc, alpha_gradient_scale, 22);"
    "chunked raster alpha bindings")

require_contains("${model_source}"
    "transparentTrainingMasks && target.coverageMask != nullptr"
    "per-frame transparent activation")
require_contains("${model_source}"
    "target.coverageMask && !transparentMask\n            ? target.coverageRenderTiles\n            : nullptr;"
    "transparent mode suppresses coverage render-tile pruning")
require_contains("${host_source}"
    "const bool useCoverageRenderTiles = coverage_mask &&\n        !transparent_mask && coverage_render_tiles;"
    "Metal host enables render-tile pruning only for coverage mode")
require_contains("${model_source}"
    "transparentMask\n        ? fullCoverageUnits\n        : target.coverageUnits"
    "full-frame transparent normalization")
require_contains("${model_source}"
    "transparentTrainingMasks &&\n        appearanceMode != msplat::AppearanceMode::None"
    "C++ transparent and appearance incompatibility")
require_contains("${api_source}" "msplat_default_training_mask_options_v11()"
    "legacy trainer coverage default")
require_contains("${api_source}"
    "config.trainingMaskMode == TrainingMaskMode::Transparent,\n        config.transparentAlphaLossWeight,\n        config.appearanceMode"
    "C++ config to model plumbing")
require_contains("${api_source}"
    "cfg.trainingMaskMode =\n            maskOptions->mode == MSPLAT_TRAINING_MASK_MODE_TRANSPARENT"
    "C ABI mode to C++ config mapping")
require_contains("${api_source}"
    "cfg.transparentAlphaLossWeight = maskOptions->alphaLossWeight;"
    "C ABI alpha weight to C++ config mapping")
require_contains("${api_source}"
    "MSPLAT_TRAINING_MASK_MODE_TRANSPARENT &&\n                  appearanceMode != msplat::AppearanceMode::None"
    "C ABI transparent and appearance incompatibility")
require_contains("${c_api_source}"
    "options.mode = MSPLAT_TRAINING_MASK_MODE_COVERAGE;"
    "C ABI coverage default")
require_contains("${swift_source}"
    "public var trainingMaskMode: TrainingMaskMode = .coverage"
    "Swift API coverage default")
require_contains("${swift_source}"
    "trainingMaskMode != .transparent || resolvedAppearanceMode == .none"
    "Swift transparent and appearance incompatibility")
