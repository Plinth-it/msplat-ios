if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/src/input_data.cpp" input_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/model.cpp" model_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)

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
    "} // namespace\n\nCameraTrainingTarget Camera::getGPUTrainingTarget("
    target_upload)
require_contains("${target_upload}"
    "{image.height, image.width, 4}, DType::UInt8"
    "Camera uploads tightly packed uint8 RGBA")
require_contains("${target_upload}"
    "expectedImageBytes != image.data.size()"
    "Camera validates compact image storage before upload")
require_contains("${target_upload}"
    "pixelCount != mask->data.size()"
    "Camera validates compact mask storage before upload")
require_contains("${target_upload}"
    "static_cast<size_t>(expectedImageBytes)"
    "Camera copies the validated compact raster without float expansion")
extract_section(input_source "CameraTrainingTarget Camera::getGPUTrainingTarget("
    "MTensor& Camera::getGPUImage(" camera_upload)
require_contains("${camera_upload}" "uploadTrainingTarget("
    "Camera publishes through the validated compact upload helper")
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
    "kernel void ssim_h_bwd_kernel(" fused_middle)
extract_section(metal_source "kernel void ssim_v_bwd_kernel("
    "kernel void photometric_adam_kernel(" vertical_backward)

foreach(section IN ITEMS horizontal_forward fused_middle vertical_backward)
    require_contains("${${section}}" "target_pixel_stride_bytes"
        "${section} receives target byte stride")
    require_contains("${${section}}" "training_target_rgb("
        "${section} reads the target through the compact helper")
    require_contains("${${section}}" "for (uint c = 0; c < 3; c++)"
        "${section} reads RGB only")
endforeach()

extract_section(host_source "auto encode_loss_fwd_bwd ="
    "auto encode_rast_bwd =" host_loss)
require_contains("${host_loss}"
    "ENC_SCALAR(enc, target_pixel_stride_bytes, 10);"
    "horizontal-forward target-stride binding")
require_contains("${host_loss}"
    "ENC_SCALAR(enc, target_pixel_stride_bytes, 17);"
    "fused-middle target-stride binding")
require_contains("${host_loss}"
    "ENC_SCALAR(enc, target_pixel_stride_bytes, 14);"
    "vertical-backward target-stride binding")
