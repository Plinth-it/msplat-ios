if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/include/msplat_c_api.h" api_header)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/msplat_api.mm" api_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "GPU-preview contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "GPU-preview contract violation: ${label}")
    endif()
endfunction()

# Extract a C/C++/Metal function whose outer closing brace starts in column 1.
# Nested blocks in this codebase are indented, so the first such brace closes
# the selected function rather than one of its internal branches or lambdas.
function(extract_function source_name start_marker output_name)
    string(FIND "${${source_name}}" "${start_marker}" section_start)
    if(section_start EQUAL -1)
        message(FATAL_ERROR
            "GPU-preview function start missing: ${start_marker}")
    endif()
    string(SUBSTRING "${${source_name}}" ${section_start} -1 section_tail)
    string(FIND "${section_tail}" "\n}\n" section_end)
    if(section_end EQUAL -1)
        message(FATAL_ERROR
            "GPU-preview function end missing: ${start_marker}")
    endif()
    math(EXPR section_length "${section_end} + 3")
    string(SUBSTRING "${section_tail}" 0 ${section_length} section_contents)
    set(${output_name} "${section_contents}" PARENT_SCOPE)
endfunction()

require_contains("${api_header}" "#define MSPLAT_ABI_VERSION 16u"
    "current ABI declaration")
require_contains("${api_header}" "typedef void* MsplatPreviewFrame;"
    "opaque preview-frame handle")
foreach(symbol IN ITEMS
        msplat_trainer_render_pose_preview_v13
        msplat_preview_frame_poll_v13
        msplat_preview_frame_texture_v13
        msplat_preview_frame_destroy_v13)
    string(REGEX MATCHALL "${symbol}\\(" matches "${api_header}")
    list(LENGTH matches match_count)
    if(NOT match_count EQUAL 1)
        message(FATAL_ERROR
            "Expected one public ${symbol} declaration, found ${match_count}")
    endif()
endforeach()

# ABI v13 is additive. Keep every existing CPU render entry point and its
# caller-owned Float32/RGBA contracts available to older clients.
foreach(legacy_symbol IN ITEMS
        msplat_trainer_render_v2
        msplat_trainer_render_pose_v2
        msplat_trainer_render_pose_to_buffer_v2
        msplat_trainer_render
        msplat_trainer_render_pose
        msplat_trainer_render_pose_to_buffer)
    require_contains("${api_header}" "${legacy_symbol}("
        "legacy render symbol ${legacy_symbol}")
endforeach()
require_contains("${api_header}" "float* data;"
    "legacy Float32 pixel-buffer layout")

extract_function(metal_source
    "kernel void float_rgb_to_preview_texture_kernel("
    preview_conversion_kernel)
require_contains("${preview_conversion_kernel}" "texture2d<"
    "conversion destination texture")
require_contains("${preview_conversion_kernel}" "access::write"
    "GPU texture-write access")
require_contains("${preview_conversion_kernel}" ".write("
    "GPU conversion output write")

require_contains("${host_source}"
    "float_rgb_to_preview_texture_kernel_cpso"
    "preview conversion pipeline state")
require_contains("${host_source}"
    "load(@\"float_rgb_to_preview_texture_kernel\")"
    "preview conversion pipeline load")
require_contains("${api_source}" "struct PreviewFrame::Impl"
    "independently owned preview completion state")
require_contains("${api_source}" "MTLPixelFormatBGRA8Unorm"
    "BGRA8Unorm preview texture")
require_contains("${api_source}" "newTextureWithDescriptor:"
    "separate preview texture allocation")
require_contains("${api_source}" "MTLTextureUsageShaderWrite"
    "conversion texture write usage")
require_contains("${api_source}" "MTLTextureUsageShaderRead"
    "display texture read usage")
require_contains("${host_source}" "addCompletedHandler:"
    "asynchronous preview completion")

extract_function(host_source "msplat_submit_preview_texture("
    preview_submit)
require_contains("${preview_submit}"
    "float_rgb_to_preview_texture_kernel_cpso"
    "conversion dispatch")
require_contains("${preview_submit}" "setTexture:"
    "separate destination texture binding")
require_contains("${preview_submit}" "commitCB("
    "non-blocking final preview commit")
require_contains("${preview_submit}" "context->discardCB();"
    "failed preview commit discards contaminated command-buffer state")
require_contains("${preview_submit}"
    "false, \"msplat: preview command buffer failed\""
    "allocation-free completion failure fallback")
foreach(forbidden IN ITEMS
        "syncCB("
        "waitUntilCompleted"
        "msplat_gpu_sync("
        ".cpu("
        "data_ptr("
        "memcpy("
        "fminf("
        "outRGBA")
    require_absent("${preview_submit}" "${forbidden}"
        "CPU readback token ${forbidden} in preview submit")
endforeach()

extract_function(api_source "Trainer::renderFromPosePreview("
    trainer_preview)
require_contains("${trainer_preview}" "impl->model->render("
    "canonical render encoding")
require_contains("${trainer_preview}" "msplat_submit_preview_texture("
    "GPU-native preview submission")
foreach(forbidden IN ITEMS
        "msplat_gpu_sync("
        ".cpu("
        "data_ptr("
        "memcpy("
        "fminf("
        "outRGBA")
    require_absent("${trainer_preview}" "${forbidden}"
        "CPU readback token ${forbidden} in Trainer preview")
endforeach()

# The checked C boundary owns the opaque handle and clears every caller output
# before validation. Runtime tests exercise these failure paths as well.
extract_function(api_source
    "MsplatStatus msplat_trainer_render_pose_preview_v13("
    render_wrapper)
extract_function(api_source "MsplatStatus msplat_preview_frame_poll_v13("
    poll_wrapper)
extract_function(api_source "MsplatStatus msplat_preview_frame_texture_v13("
    texture_wrapper)
require_contains("${render_wrapper}" "*outFrame = nullptr"
    "render output clearing")
require_contains("${poll_wrapper}" "*outReady = false"
    "poll output clearing")
require_contains("${texture_wrapper}" "*outTexture = nil"
    "texture output clearing")
require_contains("${texture_wrapper}" "*outWidth = 0"
    "texture width clearing")
require_contains("${texture_wrapper}" "*outHeight = 0"
    "texture height clearing")
