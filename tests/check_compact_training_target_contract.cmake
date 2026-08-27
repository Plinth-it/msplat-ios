if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/src/input_data.cpp" input_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/include/input_data.hpp" input_header)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/model.cpp" model_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/swift/Sources/Msplat/TrainingPlan.swift" planner_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Compact-target contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Compact-target contract violation: ${label}")
    endif()
endfunction()

function(extract_section source_name start_marker end_marker output_name)
    string(FIND "${${source_name}}" "${start_marker}" section_start)
    if(section_start EQUAL -1)
        message(FATAL_ERROR "Compact-target section start missing: ${start_marker}")
    endif()
    string(FIND "${${source_name}}" "${end_marker}" section_end)
    if(section_end EQUAL -1 OR section_end LESS_EQUAL section_start)
        message(FATAL_ERROR "Compact-target section end missing: ${end_marker}")
    endif()
    math(EXPR section_length "${section_end} - ${section_start}")
    string(SUBSTRING "${${source_name}}" ${section_start} ${section_length}
        section_contents)
    set(${output_name} "${section_contents}" PARENT_SCOPE)
endfunction()

extract_section(input_source "UploadedTrainingTarget uploadTrainingTarget("
    "} // namespace\n\nbool Camera::hasGPUTrainingTarget("
    target_upload)
require_contains("${target_upload}"
    "{image.height, image.width, 4}, DType::UInt8"
    "Camera uploads tightly packed uint8 RGBA")
require_contains("${target_upload}"
    "expectedImageBytes != image.data.size()"
    "Camera validates compact image storage before upload")
require_contains("${input_source}"
    "pixelCount != mask.data.size()"
    "Camera validates compact mask storage before upload")
require_contains("${target_upload}"
    "static_cast<size_t>(expectedImageBytes)"
    "Camera copies the validated compact raster without float expansion")
require_contains("${target_upload}"
    "rgba[pixel * 4u + 3u] = mask->data[pixel]"
    "Camera packs coverage into RGBA alpha")
require_absent("${target_upload}" "gpu_empty(\n            {mask->height"
    "Camera must not allocate a standalone GPU mask")
extract_section(input_source "CameraTrainingTarget Camera::getGPUTrainingTarget("
    "MTensor& Camera::getGPUImage(" camera_upload)
require_contains("${camera_upload}" "uploadTrainingTarget("
    "Camera publishes through the validated compact upload helper")
require_contains("${camera_upload}"
    "trainingMask ? &imageInserted.first->second : nullptr"
    "Camera marks packed coverage by aliasing the image")
require_contains("${input_header}" "MTensor *coverageMask = nullptr;"
    "CameraTrainingTarget ABI fields remain intact")
require_contains("${input_header}" "MTensor *coverageRenderTiles = nullptr;"
    "CameraTrainingTarget appends optional render tiles")
require_contains("${input_header}" "kTrainingSsimHalo = 5"
    "CPU render-tile halo matches 11x11 SSIM")
require_contains("${metal_source}" "#define SSIM_HALF_WIN 5"
    "Metal SSIM radius matches the CPU render-tile halo")
require_contains("${input_source}" "buildCoverageRenderTileMap("
    "Camera builds conservative coverage render tiles")
require_contains("${input_source}"
    "gpuTrainingMaskSourceByDownscale.emplace("
    "resident targets retain their mask source identity")
require_contains("${input_source}"
    "decodedTrainingMaskSource = trainingMask;"
    "decoded coverage retains its mask source identity")
require_contains("${input_source}"
    "!decodedTrainingMaskMatches(*this)"
    "coverage lookup rejects a stale decoded mask source")
require_contains("${input_source}"
    "includeCoverageRenderTiles"
    "prefetch residency includes the requested coverage-tile capability")
require_contains("${planner_source}" "[4, pixelCount]"
    "planner budgets four bytes for masked and unmasked targets")
require_contains("${planner_source}"
    "includesTrainingMasks ? tileCount : 0"
    "planner budgets one byte per masked render tile")
require_contains("${input_source}" "cam.releaseCpuImageMemory();"
    "decoded CPU pixels are released after compact publication")

require_contains("${model_source}"
    "target.image->dtype() == DType::UInt8 &&\n         target.image->size(2) == 4"
    "model accepts compact uint8 RGBA targets")
require_contains("${host_source}"
    "gt.dtype() == DType::UInt8 && gt.size(2) == 4"
    "Metal entry point validates compact uint8 RGBA targets")
require_contains("${host_source}"
    "const uint32_t target_pixel_stride_bytes = targetIsRGBA8\n        ? 4u\n        : 3u * static_cast<uint32_t>(sizeof(float));"
    "host derives the compact target byte stride")
require_contains("${host_source}"
    "const bool coverageIsPackedAlpha = coverage_mask == &gt;"
    "Metal entry point recognizes packed alpha")
require_contains("${host_source}"
    "std::array<uint32_t, 2>{4u, 3u}"
    "Metal host exposes packed mask stride and offset")

require_contains("${metal_source}"
    "training_mask[pixel * layout.x + layout.y]"
    "Metal kernels address interleaved packed coverage")

extract_section(metal_source "inline float training_target_rgb("
    "// Forward pass 1: horizontal convolution" target_helper)
require_contains("${target_helper}" "target_pixel_stride_bytes == 4"
    "target helper selects the compact representation")
require_contains("${target_helper}"
    "reinterpret_cast<constant uchar*>(gt)"
    "target helper reads compact bytes")
require_contains("${target_helper}" "bytes[pixel * 4 + channel]"
    "target helper reads RGB from a four-byte pixel")
require_contains("${target_helper}" "(1.0f / 255.0f)"
    "target helper normalizes bytes to unit floats")
require_absent("${target_helper}" "bytes[pixel * 4 + 3]"
    "compact target alpha must be ignored")

extract_section(metal_source "kernel void ssim_h_fwd_kernel("
    "kernel void ssim_v_fwd_kernel(" horizontal_forward)
extract_section(metal_source "kernel void ssim_fused_v_fwd_h_bwd_kernel("
    "kernel void ssim_fused_v_fwd_bwd_kernel(" staged_middle)
extract_section(metal_source "kernel void ssim_fused_v_fwd_bwd_kernel("
    "kernel void ssim_h_bwd_kernel(" fused_terminal)
extract_section(metal_source "kernel void ssim_v_bwd_kernel("
    "kernel void photometric_adam_kernel(" vertical_backward)

foreach(section IN ITEMS horizontal_forward staged_middle vertical_backward)
    require_contains("${${section}}" "target_pixel_stride_bytes"
        "${section} receives target byte stride")
    require_contains("${${section}}" "training_target_rgb("
        "${section} reads the target through the compact helper")
    require_contains("${${section}}" "for (uint c = 0; c < 3; c++)"
        "${section} reads RGB only")
endforeach()
require_contains("${fused_terminal}" "constant uint& target_pixel_stride_bytes"
    "fused terminal receives target byte stride")
require_contains("${fused_terminal}" "training_target_rgb("
    "fused terminal reads the target through the compact helper")
require_contains("${fused_terminal}" "for (uint c = 0; c < 3; ++c)"
    "fused terminal reads RGB only")
require_contains("${fused_terminal}"
    "target_pixel_stride_bytes, center_pixel, c"
    "fused terminal loss uses compact-target addressing")
require_contains("${fused_terminal}"
    "target_pixel_stride_bytes, pixel, c"
    "fused terminal backward uses compact-target addressing")

extract_section(host_source "auto encode_loss_fwd_bwd ="
    "auto encode_rast_bwd =" host_loss)
require_contains("${host_loss}"
    "ENC_SCALAR(enc, target_pixel_stride_bytes, 10);"
    "horizontal-forward target-stride binding")
require_contains("${host_loss}"
    "ENC_SCALAR(enc, target_pixel_stride_bytes, 17);"
    "fused and staged-middle target-stride binding")
require_contains("${host_loss}"
    "ENC_SCALAR(enc, target_pixel_stride_bytes, 14);"
    "vertical-backward target-stride binding")
require_contains("${host_loss}"
    "ctx->ssim_fused_v_fwd_bwd_kernel_cpso"
    "compact-target fused terminal dispatch")
require_contains("${host_loss}"
    "ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso"
    "compact-target staged middle dispatch")
require_contains("${host_loss}" "ctx->ssim_v_bwd_kernel_cpso"
    "compact-target staged vertical-backward dispatch")
