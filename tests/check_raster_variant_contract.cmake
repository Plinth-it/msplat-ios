if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/include/msplat.hpp" public_header)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/model.cpp" model_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Raster-variant contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Raster-variant contract violation: ${label}")
    endif()
endfunction()

function(require_regex_count contents pattern expected label)
    string(REGEX MATCHALL "${pattern}" matches "${contents}")
    list(LENGTH matches match_count)
    if(NOT match_count EQUAL expected)
        message(FATAL_ERROR
            "Raster-variant contract expected ${expected} ${label}, found ${match_count}")
    endif()
endfunction()

function(require_substring_count contents needle expected label)
    set(remainder "${contents}")
    string(LENGTH "${needle}" needle_length)
    set(match_count 0)
    while(TRUE)
        string(FIND "${remainder}" "${needle}" position)
        if(position EQUAL -1)
            break()
        endif()
        math(EXPR next_position "${position} + ${needle_length}")
        string(SUBSTRING "${remainder}" ${next_position} -1 remainder)
        math(EXPR match_count "${match_count} + 1")
    endwhile()
    if(NOT match_count EQUAL expected)
        message(FATAL_ERROR
            "Raster-variant contract expected ${expected} ${label}, found ${match_count}")
    endif()
endfunction()

function(extract_section source_name start_marker end_marker output_name)
    string(FIND "${${source_name}}" "${start_marker}" section_start)
    if(section_start EQUAL -1)
        message(FATAL_ERROR
            "Raster-variant section start missing: ${start_marker}")
    endif()
    string(SUBSTRING "${${source_name}}" ${section_start} -1 section_tail)
    string(FIND "${section_tail}" "${end_marker}" section_length)
    if(section_length EQUAL -1 OR section_length EQUAL 0)
        message(FATAL_ERROR
            "Raster-variant section end missing: ${end_marker}")
    endif()
    string(SUBSTRING "${section_tail}" 0 ${section_length} section_contents)
    set(${output_name} "${section_contents}" PARENT_SCOPE)
endfunction()

extract_section(metal_source "inline void nd_rasterize_forward_impl("
    "// The exact-intersection bins remain 16x16." raster_helper)
extract_section(metal_source "kernel void nd_rasterize_forward_kernel("
    "kernel void nd_rasterize_forward_16x8_kernel(" raster_8x8)
extract_section(metal_source "kernel void nd_rasterize_forward_16x8_kernel("
    "kernel void nd_rasterize_forward_16x16_kernel(" raster_16x8)
extract_section(metal_source "kernel void nd_rasterize_forward_16x16_kernel("
    "void sh_coeffs_to_color(" raster_16x16)

# The three entrypoints are a shared implementation with an identical Metal
# buffer ABI. Only their caller-owned threadgroup arrays and batch width differ.
require_regex_count("${metal_source}"
    "nd_rasterize_forward_impl\\(" 4
    "helper definition plus three calls")
require_regex_count("${raster_helper}"
    "threadgroup float3\\*" 3 "helper threadgroup-array parameters")
require_regex_count("${raster_helper}"
    "threadgroup_barrier\\(" 2 "helper threadgroup barriers")
require_contains("${raster_helper}" "const uint raster_block_size"
    "explicit helper batch width")
require_contains("${raster_helper}"
    "int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);"
    "fixed parent-bin lookup")

foreach(kernel_name IN ITEMS
        nd_rasterize_forward_kernel
        nd_rasterize_forward_16x8_kernel
        nd_rasterize_forward_16x16_kernel)
    require_regex_count("${metal_source}"
        "kernel void ${kernel_name}\\(" 1 "${kernel_name} definition")
endforeach()

set(raster_abi_bindings
    "constant uint3& tile_bounds [[buffer(0)]]"
    "constant uint3& img_size [[buffer(1)]]"
    "constant uint& channels [[buffer(2)]]"
    "constant int* tile_bins [[buffer(3)]]"
    "constant float* packed_xy_opac [[buffer(4)]]"
    "constant float* packed_conic [[buffer(5)]]"
    "constant float* packed_rgb [[buffer(6)]]"
    "device float* final_Ts [[buffer(7)]]"
    "device int* final_index [[buffer(8)]]"
    "device float* out_img [[buffer(9)]]"
    "constant float* background [[buffer(10)]]"
    "constant uint2& blockDim [[buffer(11)]]"
    "constant uint64_t* sorted_keys [[buffer(12)]]"
    "constant float* projected_opacities [[buffer(13)]]"
    "constant uint& attribute_layout [[buffer(14)]]"
    "uint2 blockIdx [[threadgroup_position_in_grid]]"
    "uint2 threadIdx [[thread_position_in_threadgroup]]"
    "uint tr [[thread_index_in_threadgroup]]")

foreach(section_name IN ITEMS raster_8x8 raster_16x8 raster_16x16)
    foreach(binding IN LISTS raster_abi_bindings)
        require_contains("${${section_name}}" "${binding}"
            "${section_name} ABI binding ${binding}")
    endforeach()
endforeach()

foreach(array_name IN ITEMS xy_opacity_batch conic_batch rgbs_batch)
    require_contains("${raster_8x8}"
        "threadgroup float3 ${array_name}[8 * 8];"
        "8x8 ${array_name} size")
    require_contains("${raster_16x8}"
        "threadgroup float3 ${array_name}[16 * 8];"
        "16x8 ${array_name} size")
    require_contains("${raster_16x16}"
        "threadgroup float3 ${array_name}[16 * 16];"
        "16x16 ${array_name} size")
endforeach()
require_contains("${raster_8x8}"
    "blockDim, sorted_keys, projected_opacities, attribute_layout,\n        blockIdx, threadIdx, tr, 8 * 8,"
    "8x8 helper batch width")
require_contains("${raster_16x8}"
    "blockDim, sorted_keys, projected_opacities, attribute_layout,\n        blockIdx, threadIdx, tr, 16 * 8,"
    "16x8 helper batch width")
require_contains("${raster_16x16}"
    "blockDim, sorted_keys, projected_opacities, attribute_layout,\n        blockIdx, threadIdx, tr, 16 * 16,"
    "16x16 helper batch width")

# The host chooses a Metal function name first, then creates and owns exactly
# one monolithic pipeline. Avoid eagerly creating the two unused variants.
require_contains("${host_source}"
    "id<MTLComputePipelineState> nd_rasterize_forward_kernel_cpso = nil;"
    "single monolithic pipeline owner")
require_absent("${host_source}"
    "nd_rasterize_forward_16x8_kernel_cpso"
    "eager 16x8 pipeline owner")
require_absent("${host_source}"
    "nd_rasterize_forward_16x16_kernel_cpso"
    "eager 16x16 pipeline owner")
require_absent("${host_source}"
    "selected_nd_rasterize_forward_kernel_cpso"
    "second selected-pipeline alias")
extract_section(host_source "id resources[] = {"
    "for (id resource : resources)" released_resources)
require_regex_count("${released_resources}"
    "nd_rasterize_forward_kernel_cpso," 1
    "monolithic pipeline release")
require_regex_count("${host_source}"
    "load\\(monolithicRasterFunctionName\\)" 1
    "conditional monolithic pipeline load")
foreach(kernel_name IN ITEMS
        nd_rasterize_forward_kernel
        nd_rasterize_forward_16x8_kernel
        nd_rasterize_forward_16x16_kernel)
    require_absent("${host_source}" "load(@\"${kernel_name}\")"
        "eager ${kernel_name} pipeline load")
endforeach()

# Selection defaults to 8x8, accepts exactly the three supported spellings,
# rejects invalid overrides, and validates the selected pipeline's limit.
extract_section(host_source "const char* rasterVariantOverride ="
    "// Forward pipeline"
    raster_selection)
extract_section(host_source "// Forward pipeline"
    "if (ctx->small_sort_per_tile_kernel_cpso.maxTotalThreadsPerThreadgroup"
    raster_pipeline_load)
require_contains("${host_source}" "uint32_t monolithic_raster_block_x = 8;"
    "default monolithic width")
require_contains("${host_source}" "uint32_t monolithic_raster_block_y = 8;"
    "default monolithic height")
require_contains("${raster_selection}"
    "std::getenv(\"MSPLAT_RASTER_VARIANT\")" "variant environment lookup")
require_contains("${raster_selection}"
    "NSString* monolithicRasterFunctionName = @\"nd_rasterize_forward_kernel\";"
    "default 8x8 function selection")
foreach(variant IN ITEMS 8x8 16x8 16x16)
    require_contains("${raster_selection}"
        "std::strcmp(rasterVariantOverride, \"${variant}\") == 0"
        "valid ${variant} selection")
endforeach()
require_contains("${raster_selection}"
    "monolithicRasterFunctionName = @\"nd_rasterize_forward_16x8_kernel\";\n            ctx->monolithic_raster_block_x = 16;\n            ctx->monolithic_raster_block_y = 8;"
    "16x8 function and dimensions")
require_contains("${raster_selection}"
    "monolithicRasterFunctionName = @\"nd_rasterize_forward_16x16_kernel\";\n            ctx->monolithic_raster_block_x = 16;\n            ctx->monolithic_raster_block_y = 16;"
    "16x16 function and dimensions")
require_contains("${raster_selection}" "throw std::invalid_argument("
    "invalid override rejection")
require_contains("${raster_selection}"
    "MSPLAT_RASTER_VARIANT must be 8x8, 16x8, or 16x16"
    "invalid override diagnostic")
require_contains("${raster_pipeline_load}"
    "ctx->nd_rasterize_forward_kernel_cpso         = load(monolithicRasterFunctionName);"
    "single selected pipeline creation")
require_contains("${raster_pipeline_load}"
    "ctx->nd_rasterize_forward_kernel_cpso.maxTotalThreadsPerThreadgroup"
    "selected pipeline thread limit")
require_contains("${raster_pipeline_load}"
    "maximumMonolithicRasterThreads < monolithicRasterThreads"
    "selected thread-count validation")

# Only the render and training monolithic paths may use the chosen pipeline.
require_substring_count("${host_source}"
    "setComputePipelineState:ctx->nd_rasterize_forward_kernel_cpso]" 2
    "chosen monolithic pipeline dispatches")
extract_section(host_source "auto encode_rast_fwd_monolithic ="
    "auto encode_rast_fwd_chunked =" render_monolithic)
extract_section(host_source "auto encode_rast_fwd_chunked ="
    "auto encode_rast_fwd =" render_chunked)
extract_section(host_source "static MTensor msplat_train_step_locked("
    "auto encode_rast_bwd =" training_before_backward)
extract_section(training_before_backward "auto encode_rast_fwd ="
    "// Fused loss:" training_forward)
extract_section(training_forward "if (K_max <= 1) {"
    "        } else {\n            // Chunked" training_monolithic)
extract_section(training_forward
    "        } else {\n            // Chunked"
    "\n        }\n    };" training_chunked)
extract_section(host_source "auto encode_rast_bwd ="
    "// Packed optimizer hyperparameters." training_backward)

foreach(section_name IN ITEMS render_monolithic training_monolithic)
    require_contains("${${section_name}}"
        "ctx->nd_rasterize_forward_kernel_cpso]"
        "${section_name} chosen pipeline")
    require_contains("${${section_name}}"
        "const uint32_t blockX = ctx->monolithic_raster_block_x;"
        "${section_name} selected width")
    require_contains("${${section_name}}"
        "const uint32_t blockY = ctx->monolithic_raster_block_y;"
        "${section_name} selected height")
    require_contains("${${section_name}}"
        "(img_width + blockX - 1) / blockX"
        "${section_name} width dispatch")
    require_contains("${${section_name}}"
        "(img_height + blockY - 1) / blockY"
        "${section_name} height dispatch")
    require_contains("${${section_name}}" "monolithic_block_size_dim2->data()"
        "${section_name} block-dimension ABI")
endforeach()

# Chunked forward and both active backward kernels deliberately remain 8x8.
require_contains("${host_source}" "#define RAST_BLOCK_X 8"
    "fixed host raster width")
require_contains("${host_source}" "#define RAST_BLOCK_Y 8"
    "fixed host raster height")
require_regex_count("${host_source}"
    "chunked_block_size_dim2 =" 2 "fixed chunked block constants")
foreach(section_name IN ITEMS render_chunked training_chunked training_backward)
    require_absent("${${section_name}}"
        "nd_rasterize_forward_kernel_cpso"
        "chosen monolithic pipeline in ${section_name}")
endforeach()
foreach(section_name IN ITEMS render_chunked training_chunked)
    require_contains("${${section_name}}"
        "setComputePipelineState:ctx->rasterize_forward_chunked_kernel_cpso"
        "${section_name} chunked pipeline")
    require_contains("${${section_name}}"
        "MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)"
        "${section_name} fixed 8x8 dispatch")
    require_contains("${${section_name}}" "chunked_block_size_dim2->data()"
        "${section_name} fixed block-dimension ABI")
endforeach()
foreach(pipeline IN ITEMS
        rasterize_backward_kernel_cpso
        rasterize_backward_chunked_kernel_cpso)
    require_contains("${training_backward}"
        "setComputePipelineState:ctx->${pipeline}"
        "fixed backward pipeline ${pipeline}")
endforeach()
require_regex_count("${training_backward}"
    "threadsPerThreadgroup:MTLSizeMake\\(RAST_BLOCK_X, RAST_BLOCK_Y, 1\\)"
    2 "fixed 8x8 backward dispatches")

extract_section(metal_source "kernel void rasterize_backward_kernel("
    "kernel void nd_rasterize_backward_kernel(" raster_backward)
extract_section(metal_source "kernel void rasterize_forward_chunked_kernel("
    "kernel void rasterize_forward_merge_kernel(" raster_forward_chunked)
extract_section(metal_source "kernel void rasterize_backward_chunked_kernel("
    "// ============================================================================\n// Separable SSIM loss kernels"
    raster_backward_chunked)
foreach(section_name IN ITEMS
        raster_backward raster_forward_chunked raster_backward_chunked)
    require_contains("${${section_name}}" "RAST_BLOCK_SIZE"
        "${section_name} fixed 8x8 batch size")
    require_absent("${${section_name}}" "raster_block_size"
        "variant batch width leaked into ${section_name}")
endforeach()
require_contains("${raster_forward_chunked}" "tr % RAST_BLOCK_X"
    "chunked forward fixed thread width")
require_contains("${raster_forward_chunked}" "tr / RAST_BLOCK_X"
    "chunked forward fixed thread height")
foreach(section_name IN ITEMS raster_backward raster_backward_chunked)
    require_contains("${${section_name}}"
        "constexpr uint NUM_WARPS = RAST_BLOCK_SIZE / 32;"
        "${section_name} fixed SIMDgroup accounting")
endforeach()

# Sorting and binning stay 16x16 regardless of the monolithic raster variant.
foreach(source_name IN ITEMS metal_source host_source public_header)
    require_contains("${${source_name}}" "#define BLOCK_X 16"
        "${source_name} 16-pixel parent-bin width")
    require_contains("${${source_name}}" "#define BLOCK_Y 16"
        "${source_name} 16-pixel parent-bin height")
endforeach()
require_contains("${model_source}"
    "(s.width + BLOCK_X - 1) / BLOCK_X,\n        (s.height + BLOCK_Y - 1) / BLOCK_Y, 1)"
    "model 16x16 tile-bound calculation")
require_contains("${host_source}"
    "(img_width + 15u) / 16u,\n        (img_height + 15u) / 16u, 1, 0xDEAD"
    "training 16x16 backward tile bounds")
