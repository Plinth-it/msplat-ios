cmake_minimum_required(VERSION 3.21)
project(msplat LANGUAGES CXX OBJCXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Version
file(READ "VERSION" APP_VERSION)
string(STRIP "${APP_VERSION}" APP_VERSION)
set(GIT_REV "")
execute_process(COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_REV ERROR_QUIET)
string(REGEX REPLACE "\n$" "" GIT_REV "${GIT_REV}")
if(NOT "${GIT_REV}" STREQUAL "")
    set(DAPP_VERSION "${APP_VERSION} (git ${GIT_REV})")
    set(DAPP_REVISION "${GIT_REV}")
else()
    set(DAPP_VERSION "${APP_VERSION}")
    set(DAPP_REVISION "dev")
endif()
message(STATUS "msplat ${DAPP_VERSION}")

add_compile_options("-DAPP_VERSION=\"${DAPP_VERSION}\"")
add_compile_options("-DAPP_REVISION=\"${DAPP_REVISION}\"")

# --- Dependencies ---

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
    URL_HASH SHA256=04022b05d806eb5ff73023c280b68697d12b93e1b7267a0b22a1a39ec7578069)
set(NANOFLANN_BUILD_EXAMPLES OFF)
set(NANOFLANN_BUILD_TESTS OFF)
FetchContent_Declare(nanoflann
    URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.5.5.zip
    URL_HASH SHA256=57496cb27e1310a77a367e5a902c8f1c700496d91ac54ccc87fbe9ccc28bc6cc)
FetchContent_Declare(cli11
    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.zip
    URL_HASH SHA256=43e650d5e1a3acaaf419d1e61a81f77b408d0696f472be0599ddf877d40984b0)
FetchContent_MakeAvailable(nlohmann_json nanoflann cli11)

# Metal frameworks
find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

find_library(IMAGEIO_FRAMEWORK ImageIO REQUIRED)
find_library(COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
find_library(ZLIB_LIBRARY z REQUIRED)

# --- Compile Metal shaders ---

# A metallib is per-platform: the device, the simulator and macOS each need
# their own, compiled against their own SDK. CMAKE_SYSTEM_NAME does not
# distinguish device from simulator, so the sysroot decides and the choice
# stays overridable for callers that know which slice they are building.
set(MSPLAT_METAL_LANGUAGE_STANDARD "metal4.0" CACHE STRING "Metal language standard for precompiled shader libraries")
set(MSPLAT_METAL_IOS_DEPLOYMENT_TARGET "18.0" CACHE STRING "Minimum iOS version for precompiled shader libraries")
set(MSPLAT_METAL_MACOS_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS version for precompiled shader libraries")

if(NOT DEFINED MSPLAT_METAL_SDK)
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        if(CMAKE_OSX_SYSROOT MATCHES "[Ss]imulator")
            set(MSPLAT_METAL_SDK "iphonesimulator")
        else()
            set(MSPLAT_METAL_SDK "iphoneos")
        endif()
    else()
        set(MSPLAT_METAL_SDK "macosx")
    endif()
endif()

if(MSPLAT_METAL_SDK STREQUAL "macosx")
    set(MSPLAT_METAL_DEPLOYMENT_FLAG "-mmacosx-version-min=${MSPLAT_METAL_MACOS_DEPLOYMENT_TARGET}")
elseif(MSPLAT_METAL_SDK STREQUAL "iphonesimulator")
    set(MSPLAT_METAL_DEPLOYMENT_FLAG "-mios-simulator-version-min=${MSPLAT_METAL_IOS_DEPLOYMENT_TARGET}")
else()
    set(MSPLAT_METAL_DEPLOYMENT_FLAG "-mios-version-min=${MSPLAT_METAL_IOS_DEPLOYMENT_TARGET}")
endif()
message(STATUS "msplat metal SDK: ${MSPLAT_METAL_SDK} ${MSPLAT_METAL_DEPLOYMENT_FLAG}")

set(METAL_SOURCE ${CMAKE_SOURCE_DIR}/core/metal/msplat_metal.metal)
configure_file(${METAL_SOURCE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.metal COPYONLY)
add_custom_command(
    OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
    COMMAND xcrun -sdk ${MSPLAT_METAL_SDK} metal -O3 -std=${MSPLAT_METAL_LANGUAGE_STANDARD} ${MSPLAT_METAL_DEPLOYMENT_FLAG} -c
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.metal
        -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.air
    COMMAND xcrun -sdk ${MSPLAT_METAL_SDK} metallib
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.air
        -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
    COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.air
    COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.metal
    DEPENDS ${METAL_SOURCE}
    COMMENT "Compiling Metal kernels -> default.metallib"
)
add_custom_target(metallib ALL
    DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib)

# --- Core static library ---

add_library(msplat_core STATIC
    core/metal/msplat_metal.mm
    core/src/msplat_api.mm
    core/src/model.cpp
    core/src/input_data.cpp
    core/src/dataset_descriptor.cpp
    core/src/dataset_diagnostics.cpp
    core/src/memory_report.cpp
    core/src/loaders/load_colmap.cpp
    core/src/loaders/load_nerfstudio.cpp
    core/src/loaders/load_polycam.cpp
    core/src/loaders/load_ply.cpp
    core/src/loaders/image_io.cpp
    core/src/loaders/save_gaussians.cpp
    core/src/ssim.cpp
)
target_include_directories(msplat_core PUBLIC
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/core/metal
)
target_link_libraries(msplat_core PUBLIC
    ${FOUNDATION_LIBRARY}
    ${METAL_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${IMAGEIO_FRAMEWORK}
    ${COREGRAPHICS_FRAMEWORK}
    ${ZLIB_LIBRARY}
    nlohmann_json::nlohmann_json
    nanoflann::nanoflann
)
target_compile_definitions(msplat_core PRIVATE USE_MPS)
add_dependencies(msplat_core metallib)

# --- CLI binary ---

# The CLI is a desktop tool; an iOS build only wants the static library.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    add_executable(msplat cli/msplat.cpp)
    target_link_libraries(msplat PRIVATE
        msplat_core
        CLI11::CLI11
        pthread
    )
    target_compile_definitions(msplat PRIVATE USE_MPS)
    install(TARGETS msplat DESTINATION bin)

    if(BUILD_TESTING)
        add_executable(msplat_c_api_tests tests/test_c_api.cpp)
        target_link_libraries(msplat_c_api_tests PRIVATE msplat_core)
        add_test(NAME msplat_c_api_errors COMMAND msplat_c_api_tests)

        add_executable(msplat_checkpoint_tests tests/test_checkpoint.cpp)
        target_link_libraries(msplat_checkpoint_tests PRIVATE msplat_core)
        add_test(NAME msplat_checkpoint_validation COMMAND msplat_checkpoint_tests)

        add_executable(msplat_image_io_tests tests/test_image_io.cpp)
        target_link_libraries(msplat_image_io_tests PRIVATE msplat_core)
        add_test(NAME msplat_image_io COMMAND msplat_image_io_tests)

        add_executable(msplat_dataset_descriptor_tests
            tests/test_dataset_descriptor.cpp)
        target_link_libraries(msplat_dataset_descriptor_tests PRIVATE msplat_core)
        add_test(NAME msplat_dataset_descriptor
            COMMAND msplat_dataset_descriptor_tests)

        add_executable(msplat_dataset_diagnostics_tests
            tests/test_dataset_diagnostics.cpp)
        target_link_libraries(msplat_dataset_diagnostics_tests PRIVATE msplat_core)
        add_test(NAME msplat_dataset_diagnostics
            COMMAND msplat_dataset_diagnostics_tests)

        add_executable(msplat_colmap_loader_tests
            tests/test_colmap_loader.cpp)
        target_link_libraries(msplat_colmap_loader_tests PRIVATE msplat_core)
        add_test(NAME msplat_colmap_loader
            COMMAND msplat_colmap_loader_tests)

        add_executable(msplat_intersection_layout_tests
            tests/test_intersection_layout.cpp)
        target_include_directories(msplat_intersection_layout_tests PRIVATE
            ${CMAKE_SOURCE_DIR}/core/include)
        add_test(NAME msplat_intersection_layout
            COMMAND msplat_intersection_layout_tests)
        add_executable(msplat_exact_sort_tests
            tests/test_exact_sort.mm)
        target_include_directories(msplat_exact_sort_tests PRIVATE
            ${CMAKE_SOURCE_DIR}/core/include)
        target_link_libraries(msplat_exact_sort_tests PRIVATE
            ${FOUNDATION_LIBRARY}
            ${METAL_FRAMEWORK})
        add_dependencies(msplat_exact_sort_tests metallib)
        add_test(NAME msplat_exact_sort
            COMMAND msplat_exact_sort_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        set_tests_properties(msplat_exact_sort PROPERTIES
            SKIP_RETURN_CODE 77)
        add_executable(msplat_tile_count_difference_tests
            tests/test_tile_count_difference.mm)
        target_link_libraries(msplat_tile_count_difference_tests PRIVATE
            ${FOUNDATION_LIBRARY}
            ${METAL_FRAMEWORK})
        add_dependencies(msplat_tile_count_difference_tests metallib)
        add_test(NAME msplat_tile_count_difference
            COMMAND msplat_tile_count_difference_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        set_tests_properties(msplat_tile_count_difference PROPERTIES
            SKIP_RETURN_CODE 77)
        add_executable(msplat_gpu_tile_layout_tests
            tests/test_gpu_tile_layout.mm)
        target_include_directories(msplat_gpu_tile_layout_tests PRIVATE
            ${CMAKE_SOURCE_DIR}/core/include)
        target_link_libraries(msplat_gpu_tile_layout_tests PRIVATE
            ${FOUNDATION_LIBRARY}
            ${METAL_FRAMEWORK})
        add_dependencies(msplat_gpu_tile_layout_tests metallib)
        add_test(NAME msplat_gpu_tile_layout
            COMMAND msplat_gpu_tile_layout_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        set_tests_properties(msplat_gpu_tile_layout PROPERTIES
            SKIP_RETURN_CODE 77)
        add_executable(msplat_raster_variants_tests
            tests/test_raster_variants.mm)
        target_link_libraries(msplat_raster_variants_tests PRIVATE
            ${FOUNDATION_LIBRARY}
            ${METAL_FRAMEWORK})
        add_dependencies(msplat_raster_variants_tests metallib)
        add_test(NAME msplat_raster_variants
            COMMAND msplat_raster_variants_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        set_tests_properties(msplat_raster_variants PROPERTIES
            SKIP_RETURN_CODE 77)
        add_executable(msplat_projection_math_tests
            tests/test_projection_math.cpp)
        add_test(NAME msplat_projection_math
            COMMAND msplat_projection_math_tests)
        add_executable(msplat_photometric_refinement_tests
            tests/test_photometric_refinement.cpp)
        add_test(NAME msplat_photometric_refinement_math
            COMMAND msplat_photometric_refinement_tests)
        add_executable(msplat_pose_refinement_tests
            tests/test_pose_refinement.cpp)
        add_test(NAME msplat_pose_refinement_math
            COMMAND msplat_pose_refinement_tests)
        add_executable(msplat_masked_metrics_tests
            tests/test_masked_metrics.cpp
            core/src/ssim.cpp)
        target_include_directories(msplat_masked_metrics_tests PRIVATE
            ${CMAKE_SOURCE_DIR}/core/include)
        target_link_libraries(msplat_masked_metrics_tests PRIVATE
            ${FOUNDATION_LIBRARY})
        add_test(NAME msplat_masked_metrics
            COMMAND msplat_masked_metrics_tests)
        add_executable(msplat_densification_tests
            tests/test_densification.mm)
        target_link_libraries(msplat_densification_tests PRIVATE msplat_core)
        add_test(NAME msplat_densification_classification
            COMMAND ${CMAKE_COMMAND} -E env
                --unset=MSPLAT_DENSIFY_RANDOM_MODE --
                $<TARGET_FILE:msplat_densification_tests>
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib"
                default)
        add_test(NAME msplat_densification_cpu_random
            COMMAND ${CMAKE_COMMAND} -E env
                MSPLAT_DENSIFY_RANDOM_MODE=cpu --
                $<TARGET_FILE:msplat_densification_tests>
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib"
                cpu)
        add_test(NAME msplat_densification_gpu_random
            COMMAND ${CMAKE_COMMAND} -E env
                MSPLAT_DENSIFY_RANDOM_MODE=gpu --
                $<TARGET_FILE:msplat_densification_tests>
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib"
                gpu)
        add_test(NAME msplat_densification_invalid_random_mode
            COMMAND ${CMAKE_COMMAND} -E env
                MSPLAT_DENSIFY_RANDOM_MODE=invalid --
                $<TARGET_FILE:msplat_densification_tests>
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib"
                invalid)
        set_tests_properties(
            msplat_densification_classification
            msplat_densification_cpu_random
            msplat_densification_gpu_random
            msplat_densification_invalid_random_mode
            PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(msplat_transparent_training_tests
            tests/test_transparent_training.mm)
        target_link_libraries(msplat_transparent_training_tests PRIVATE msplat_core)
        add_test(NAME msplat_transparent_training
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_transparent_training_profile
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_transparent_training_16x8
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_transparent_training_16x16
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_transparent_training_difference
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_transparent_training_gpu_layout
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_transparent_training_difference_gpu_layout
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_transparent_training_arena_retry
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_transparent_training_arena_retry_profile
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_transparent_training_difference_arena_retry
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_transparent_training_ssim_fused
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_transparent_training_gather
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_transparent_training_gather_arena_retry
            COMMAND msplat_transparent_training_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        set_tests_properties(msplat_transparent_training PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_transparent_training_profile PROPERTIES
            ENVIRONMENT
                "PROFILE_STAGES=1;MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_TRAINING_ARENA_MODE=exact;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_transparent_training_16x8 PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=16x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_transparent_training_16x16 PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=16x16;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_transparent_training_difference PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=difference;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_transparent_training_gpu_layout PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=gpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_transparent_training_difference_gpu_layout PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=difference;MSPLAT_TILE_LAYOUT_MODE=gpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_transparent_training_arena_retry PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TRAINING_ARENA_MODE=retry;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(
            msplat_transparent_training_arena_retry_profile PROPERTIES
            ENVIRONMENT
                "PROFILE_STAGES=1;MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TRAINING_ARENA_MODE=retry;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(
            msplat_transparent_training_difference_arena_retry PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=difference;MSPLAT_TRAINING_ARENA_MODE=retry;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_transparent_training_ssim_fused PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_SSIM_MODE=fused;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_transparent_training_gather PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=16x16;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=gather"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_transparent_training_gather_arena_retry PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TRAINING_ARENA_MODE=retry;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=gather"
            SKIP_RETURN_CODE 77)
        add_executable(msplat_terminal_backward_tests
            tests/test_terminal_backward.mm)
        target_link_libraries(msplat_terminal_backward_tests PRIVATE msplat_core)
        add_test(NAME msplat_terminal_backward
            COMMAND msplat_terminal_backward_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_terminal_backward_16x8
            COMMAND msplat_terminal_backward_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_terminal_backward_16x16
            COMMAND msplat_terminal_backward_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_terminal_backward_difference
            COMMAND msplat_terminal_backward_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_terminal_backward_gpu_layout
            COMMAND msplat_terminal_backward_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_terminal_backward_difference_gpu_layout
            COMMAND msplat_terminal_backward_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_terminal_backward_arena_retry
            COMMAND msplat_terminal_backward_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_terminal_backward_ssim_fused
            COMMAND msplat_terminal_backward_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        add_test(NAME msplat_terminal_backward_gather
            COMMAND msplat_terminal_backward_tests
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib")
        set_tests_properties(msplat_terminal_backward PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_terminal_backward_16x8 PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=16x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_terminal_backward_16x16 PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=16x16;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_terminal_backward_difference PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=difference;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_terminal_backward_gpu_layout PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=gpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_terminal_backward_difference_gpu_layout PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=difference;MSPLAT_TILE_LAYOUT_MODE=gpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_terminal_backward_arena_retry PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TRAINING_ARENA_MODE=retry;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_terminal_backward_ssim_fused PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_SSIM_MODE=fused;MSPLAT_INTERSECTION_ATTRIBUTES=packed"
            SKIP_RETURN_CODE 77)
        set_tests_properties(msplat_terminal_backward_gather PROPERTIES
            ENVIRONMENT
                "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TILE_LAYOUT_MODE=cpu;MSPLAT_SSIM_MODE=staged;MSPLAT_INTERSECTION_ATTRIBUTES=gather"
            SKIP_RETURN_CODE 77)
        add_test(NAME msplat_exact_intersection_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_exact_intersection_contract.cmake)
        add_test(NAME msplat_raster_variant_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_raster_variant_contract.cmake)
        add_test(NAME msplat_intersection_attribute_gather_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_intersection_attribute_gather_contract.cmake)
        add_test(NAME msplat_tile_count_difference_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_tile_count_difference_contract.cmake)
        add_test(NAME msplat_gpu_tile_layout_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_gpu_tile_layout_contract.cmake)
        add_test(NAME msplat_training_arena_retry_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_training_arena_retry_contract.cmake)
        add_test(NAME msplat_ssim_storage_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_ssim_storage_contract.cmake)
        add_test(NAME msplat_transparent_mask_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_transparent_mask_contract.cmake)
        add_test(NAME msplat_projection_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_projection_contract.cmake)
        add_test(NAME msplat_photometric_refinement_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_photometric_refinement_contract.cmake)
        add_test(NAME msplat_pose_refinement_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_pose_refinement_contract.cmake)
        add_test(NAME msplat_terminal_backward_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_terminal_backward_contract.cmake)
        add_test(NAME msplat_compact_training_target_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_compact_training_target_contract.cmake)
        add_test(NAME msplat_gpu_preview_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_gpu_preview_contract.cmake)
        add_test(NAME msplat_densification_contract
            COMMAND ${CMAKE_COMMAND}
                -DMSPLAT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                -P ${CMAKE_SOURCE_DIR}/tests/check_densification_contract.cmake)
    endif()
endif()

# --- Python bindings (optional, enabled when building via scikit-build-core) ---

option(MSPLAT_BUILD_PYTHON "Build nanobind Python bindings" OFF)

if(MSPLAT_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_core python/bindings.cpp)
    target_link_libraries(_core PRIVATE
        msplat_core
        pthread
    )
    target_compile_definitions(_core PRIVATE USE_MPS)

    install(TARGETS _core LIBRARY DESTINATION msplat)
    install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib DESTINATION msplat)
endif()
