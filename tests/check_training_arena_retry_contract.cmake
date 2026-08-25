if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/model.cpp" model_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/msplat_api.mm" api_source)
file(READ "${MSPLAT_SOURCE_DIR}/tests/test_transparent_training.mm"
    retry_fixture_source)
file(READ "${MSPLAT_SOURCE_DIR}/CMakeLists.txt" cmake_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Training-arena retry contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Training-arena retry contract rejected: ${label}")
    endif()
endfunction()

function(require_count contents regex expected label)
    string(REGEX MATCHALL "${regex}" matches "${contents}")
    list(LENGTH matches actual)
    if(NOT actual EQUAL expected)
        message(FATAL_ERROR
            "Training-arena retry contract expected ${expected} ${label}, found ${actual}")
    endif()
endfunction()

function(extract_section contents start_marker end_marker output)
    string(FIND "${contents}" "${start_marker}" start_position)
    if(start_position EQUAL -1)
        message(FATAL_ERROR
            "Training-arena retry contract cannot find ${start_marker}")
    endif()
    string(SUBSTRING "${contents}" ${start_position} -1 section_tail)
    string(FIND "${section_tail}" "${end_marker}" section_length)
    if(section_length EQUAL -1 OR section_length EQUAL 0)
        message(FATAL_ERROR
            "Training-arena retry contract cannot find ${end_marker} after ${start_marker}")
    endif()
    string(SUBSTRING "${section_tail}" 0 ${section_length} section)
    set(${output} "${section}" PARENT_SCOPE)
endfunction()

function(require_ordered contents first second label)
    string(FIND "${contents}" "${first}" first_position)
    string(FIND "${contents}" "${second}" second_position)
    if(first_position EQUAL -1 OR second_position EQUAL -1 OR
       second_position LESS_EQUAL first_position)
        message(FATAL_ERROR
            "Training-arena retry ordering failed: ${label}")
    endif()
endfunction()

extract_section("${host_source}" "static void render_pipeline("
    "MTensor msplat_render(" render_pipeline)
extract_section("${host_source}"
    "const char* trainingArenaModeOverride ="
    "const char* rasterVariantOverride =" arena_mode_config)
extract_section("${host_source}" "static MTensor msplat_train_step_locked("
    "MTensor msplat_train_step(" training_pipeline)
extract_section("${host_source}" "MTensor msplat_train_step("
    "void msplat_prepare_densify(" public_training_wrapper)
extract_section("${host_source}"
    "static void encode_validate_tile_intersection_attempt("
    "MTensor gpu_zeros(" validator_host)
extract_section("${training_pipeline}"
    "// Pass 1 completes before any arena or optimizer work."
    "msplat::TileIntersectionLayout intersectionLayout;" count_phase)
extract_section("${count_phase}" "id<MTLBlitCommandEncoder> blit ="
    "[blit endEncoding];" pre_layout_zero)
extract_section("${training_pipeline}" "auto encode_loss_fwd_bwd ="
    "auto encode_rast_bwd =" loss_and_photometric_host)
extract_section("${training_pipeline}" "auto encode_scatter_sort_finalize ="
    "auto encode_rast_fwd =" sort_pack_host)
extract_section("${sort_pack_host}"
    "ctx->finalize_tile_intersection_attempt_kernel_cpso"
    "ctx->pack_sorted_gaussians_kernel_cpso" finalizer_host)
extract_section("${training_pipeline}" "auto encode_proj_sh_bwd_adam ="
    "// ========================== DISPATCH" terminal_update_host)
extract_section("${training_pipeline}" "auto do_blit_zero ="
    "auto encode_step_readback =" post_layout_zero)
extract_section("${training_pipeline}"
    "auto inspectCompletedRetryAttempt ="
    "auto replaySameLogicalStep =" retry_inspection)
extract_section("${training_pipeline}"
    "auto replaySameLogicalStep ="
    "if (ctx->retry_intersection_attempts) {\n        id<MTLCommandBuffer> commandBuffer ="
    retry_replay)
extract_section("${training_pipeline}"
    "if (ctx->retry_intersection_attempts) {\n        id<MTLCommandBuffer> commandBuffer ="
    "const auto postCountEncodeStart" retry_preflight)
extract_section("${training_pipeline}"
    "const auto postCountEncodeStart"
    "// Loss is copied into the step's unique readback" post_count_dispatch)
extract_section("${host_source}" "struct MsplatLogicalTrainingStep {"
    "struct ScopedObjCRelease" logical_step_telemetry)
extract_section("${logical_step_telemetry}"
    "void recordIntersectionLayout("
    "void recordRecoveredIntersectionRetry(" layout_telemetry)
extract_section("${logical_step_telemetry}"
    "void recordRecoveredIntersectionRetry("
    "void recordPostCountEncode(" recovered_retry_telemetry)
extract_section("${logical_step_telemetry}"
    "void recordPostCountEncode("
    "void validateForSubmit(" post_count_telemetry)
extract_section("${logical_step_telemetry}"
    "void finishIfReadyLocked() noexcept {"
    "MsplatTrainingTelemetryHandle telemetry;" completed_telemetry)
extract_section("${retry_fixture_source}"
    "void checkArenaRetryTransaction() {"
    "}  // namespace" retry_fixture)

extract_section("${metal_source}" "kernel void camera_pose_adam_kernel("
    "kernel void sh_opacity_backward_adam_kernel(" camera_pose_kernel)
extract_section("${metal_source}" "kernel void sh_opacity_backward_adam_kernel("
    "struct GeometryAdamParams" appearance_kernel)
extract_section("${metal_source}" "kernel void project_backward_adam_kernel("
    "// ===== Exact Tile Intersection Pipeline" geometry_kernel)
extract_section("${metal_source}" "kernel void validate_tile_intersection_attempt_kernel("
    "kernel void finalize_tile_intersection_attempt_kernel(" validator_kernel)
extract_section("${metal_source}" "kernel void finalize_tile_intersection_attempt_kernel("
    "kernel void scatter_to_exact_bins_kernel(" finalizer_kernel)
extract_section("${metal_source}" "kernel void photometric_adam_kernel("
    "kernel void densify_classify_kernel(" photometric_kernel)

# Retry is an explicit training-only experiment. With no override, or with the
# exact value, the established synchronized path remains active. Loading its
# two extra pipelines and its indirect-dispatch buffer is also opt-in.
require_contains("${host_source}" "bool retry_intersection_attempts = false;"
    "exact-by-default retry state")
require_contains("${host_source}"
    "std::getenv(\"MSPLAT_TRAINING_ARENA_MODE\")"
    "training-arena environment lookup")
require_contains("${host_source}"
    "std::strcmp(trainingArenaModeOverride, \"exact\") == 0"
    "strict exact mode")
require_contains("${host_source}"
    "std::strcmp(trainingArenaModeOverride, \"retry\") == 0"
    "strict retry mode")
require_contains("${host_source}"
    "ctx->retry_intersection_attempts = true;"
    "retry-mode activation")
require_ordered("${arena_mode_config}"
    "ctx->retry_intersection_attempts = true;"
    "ctx->gpu_tile_layout = true;"
    "retry mode enables GPU layout")
require_contains("${host_source}"
    "msplat: MSPLAT_TRAINING_ARENA_MODE must be exact or retry"
    "invalid-mode diagnostic")
require_absent("${host_source}"
    "strcasecmp(trainingArenaModeOverride"
    "case-insensitive training-arena parsing")
require_contains("${host_source}"
    "if (ctx->retry_intersection_attempts) {\n        ctx->validate_tile_intersection_attempt_kernel_cpso ="
    "lazy validator pipeline load")
require_contains("${host_source}"
    "ctx->finalize_tile_intersection_attempt_kernel_cpso =\n            load(@\"finalize_tile_intersection_attempt_kernel\");"
    "lazy finalizer pipeline load")
require_contains("${host_source}"
    "g_tcache.ensure_tile_attempt_dispatch_control(\n        ctx->retry_intersection_attempts, ctx->device);"
    "retry-gated indirect-dispatch allocation")
require_contains("${host_source}"
    "tile_attempt_dispatch_control =\n            mtensor_empty(dev, {12}, DType::Int32);"
    "three-record indirect-dispatch allocation and scalar controls")
require_contains("${training_pipeline}"
    "const uint32_t attemptGatingEnabled =\n        ctx->retry_intersection_attempts ? 1u : 0u;"
    "default-off persistent-mutation gating")

# Rendering is intentionally outside this experiment: it continues to size the
# exact arena after a synchronized count/layout pass and never consults retry
# status or indirect attempt controls.
require_contains("${render_pipeline}" "ctx->syncCB()"
    "render exact-count synchronization")
require_ordered("${render_pipeline}"
    "encode_gpu_tile_intersection_layout("
    "ctx->syncCB()"
    "render layout completes before its host readback")
require_contains("${render_pipeline}"
    "g_tcache.ensure_intersection_arena("
    "render exact arena sizing")
foreach(retry_symbol IN ITEMS
        retry_intersection_attempts
        gpuResidentIntersectionAttempt
        validate_tile_intersection_attempt
        finalize_tile_intersection_attempt
        tile_attempt_dispatch_control)
    require_absent("${render_pipeline}" "${retry_symbol}"
        "render reference to ${retry_symbol}")
endforeach()

# The validator owns the initial capacity decision and writes three complete
# MTLDispatchThreadgroupsIndirectArguments records (small, general, pack). Any
# failure poisons the attempt and publishes neutral raster bins. The finalizer
# repeats that neutralization after scatter/sort defensive checks.
require_contains("${validator_kernel}"
    "device atomic_uint* attempt_status       [[buffer(6)]]"
    "validator attempt-status ABI")
require_contains("${validator_kernel}"
    "device uint* dispatch_control            [[buffer(7)]]"
    "validator indirect-control ABI")
require_contains("${validator_kernel}"
    "constant int* inclusive_offsets          [[buffer(9)]]"
    "validator inclusive-offset ABI")
require_contains("${validator_host}"
    "MTensor& tileOffsets"
    "validator host inclusive-offset argument")
require_contains("${validator_host}"
    "ENC_BUF(encoder, tileOffsets, 9);"
    "validator host inclusive-offset binding")
require_contains("${count_phase}"
    "g_tcache.tile_attempt_dispatch_control,\n                    g_tcache.tile_offsets);"
    "validator live inclusive-offset argument")
foreach(control_word RANGE 0 11)
    require_contains("${validator_kernel}"
        "dispatch_control[${control_word}]"
        "indirect control word ${control_word}")
endforeach()
require_contains("${validator_kernel}"
    "failure |= MSPLAT_OVERFLOW_PACKED_CAPACITY;"
    "under-capacity failure")
require_contains("${validator_kernel}"
    "failure |= MSPLAT_OVERFLOW_TILE_CAP;"
    "unsupported-tile failure")
require_contains("${validator_kernel}"
    "metadata[EXACT_LAYOUT_ERROR_FLAGS] != 0u"
    "layout error propagation")
require_contains("${validator_kernel}"
    "total_count > 0x7FFFFFFFu"
    "signed-offset range validation")
require_contains("${validator_kernel}"
    "classified_tile_count != (ulong)num_tiles"
    "complete tile classification validation")
require_contains("${validator_kernel}"
    "categorized_sortable_tile_count != (ulong)sortable_tile_count"
    "sortable category validation")
require_contains("${validator_kernel}"
    "active_tile_count > num_tiles"
    "active tile upper-bound validation")
require_contains("${validator_kernel}"
    "active_tile_count < sortable_tile_count"
    "sortable-active coherence validation")
require_contains("${validator_kernel}"
    "active_tile_count > total_count"
    "active-intersection coherence validation")
require_contains("${validator_kernel}"
    "maximum_tile_count > 0u && maximum_tile_index >= num_tiles"
    "maximum tile index validation")
require_contains("${validator_kernel}"
    "maximum_tile_count == 0u && total_count != 0u"
    "empty maximum coherence validation")
require_contains("${validator_kernel}"
    "maximum_tile_count > total_count"
    "maximum population validation")
require_contains("${validator_kernel}"
    "large_tile_count == 0u &&\n         maximum_tile_count > EXACT_BITONIC_FAST_PATH"
    "missing large-tile category validation")
require_contains("${validator_kernel}"
    "large_tile_count > 0u &&\n         maximum_tile_count <= EXACT_BITONIC_FAST_PATH"
    "spurious large-tile category validation")
require_contains("${validator_kernel}"
    "const int final_offset = num_tiles > 0u\n        ? inclusive_offsets[num_tiles - 1u]\n        : 0;"
    "final inclusive-offset readback")
require_contains("${validator_kernel}"
    "final_offset < 0 || (uint)final_offset != total_count"
    "final-offset total coherence validation")
require_contains("${validator_kernel}"
    "maximum_tile_count > 65536u || pack_threads_per_group == 0u"
    "dispatch limit validation")
require_contains("${validator_kernel}"
    "atomic_fetch_or_explicit(\n            attempt_status, failure, memory_order_relaxed);"
    "attempt poisoning")
require_contains("${validator_kernel}"
    "tile_bins[tile * 2u + 1u] = 0;"
    "validator neutral bins")
require_ordered("${validator_host}"
    "memoryBarrierWithScope:MTLBarrierScopeBuffers"
    "ctx->validate_tile_intersection_attempt_kernel_cpso"
    "layout publication precedes validation")
require_contains("${finalizer_kernel}"
    "!training_attempt_failed(attempt_status)"
    "failed-attempt finalizer gate")
require_contains("${finalizer_kernel}"
    "device uint* dispatch_control         [[buffer(3)]]"
    "finalizer indirect-control ABI")
require_contains("${finalizer_kernel}"
    "dispatch_control[6] = 0u;"
    "late-failure pack suppression")
require_contains("${finalizer_kernel}"
    "tile_bins[tile * 2u + 1u] = 0;"
    "finalizer neutral bins")

require_contains("${sort_pack_host}"
    "if (gpuResidentIntersectionAttempt || small_sort_tile_count > 0)"
    "retry small-sort encoder")
require_count("${sort_pack_host}"
    "dispatchThreadgroupsWithIndirectBuffer:" 3
    "retry indirect dispatches")
require_contains("${sort_pack_host}"
    "offset:9u * sizeof(uint32_t) atIndex:7"
    "small-sort count scalar")
require_contains("${sort_pack_host}"
    "offset:10u * sizeof(uint32_t) atIndex:8"
    "general-sort count scalar")
require_contains("${sort_pack_host}"
    "offset:11u * sizeof(uint32_t) atIndex:9"
    "general-sort list offset scalar")
require_contains("${sort_pack_host}" "indirectBufferOffset:0"
    "small-sort indirect record")
require_contains("${sort_pack_host}"
    "indirectBufferOffset:3u * sizeof(uint32_t)"
    "general-sort indirect record")
require_contains("${sort_pack_host}"
    "indirectBufferOffset:6u * sizeof(uint32_t)"
    "pack indirect record")
require_ordered("${sort_pack_host}"
    "indirectBufferOffset:3u * sizeof(uint32_t)"
    "ctx->finalize_tile_intersection_attempt_kernel_cpso"
    "sort completion precedes attempt finalization")
require_ordered("${sort_pack_host}"
    "ctx->finalize_tile_intersection_attempt_kernel_cpso"
    "ctx->pack_sorted_gaussians_kernel_cpso"
    "attempt finalization precedes packing")
require_contains("${sort_pack_host}"
    "ENC_BUF(enc, overflow_flag, 1);"
    "finalizer attempt-status binding")
require_contains("${sort_pack_host}"
    "ENC_BUF(enc, g_tcache.tile_attempt_dispatch_control, 3);"
    "finalizer indirect-control binding")
require_contains("${finalizer_host}"
    "memoryBarrierWithScope:MTLBarrierScopeBuffers"
    "finalized bins published before packing and raster")

# The status starts clean before layout and validation publish a result. The
# later zeroing pass must preserve that result in retry mode until the four
# persistent mutation kernels have observed it.
require_ordered("${count_phase}"
    "[blit fillBuffer:overflow_flag.buffer()"
    "encode_gpu_tile_intersection_layout("
    "retry status cleared before layout")
require_absent("${pre_layout_zero}" "gpuResidentIntersectionAttempt"
    "conditional pre-layout attempt-status clear")
require_ordered("${count_phase}"
    "encode_gpu_tile_intersection_layout("
    "encode_validate_tile_intersection_attempt("
    "layout validated before post-count work")
require_contains("${count_phase}"
    "if (!gpuResidentIntersectionAttempt) {\n            const SynchronousGpuMetrics countPassMetrics = ctx->syncCB();"
    "bootstrap count barrier")
require_absent("${post_layout_zero}"
    "fillBuffer:overflow_flag.buffer()"
    "post-layout attempt-status clear")

# All state that can survive an attempt is guarded by the same immutable status
# word. Geometry must branch before its threadgroup barriers so every lane takes
# the same path, and before Adam/statistics writes.
require_contains("${metal_source}"
    "inline bool training_attempt_failed(device atomic_uint* attempt_status)"
    "shared attempt-status helper")
require_contains("${metal_source}"
    "atomic_load_explicit(\n        attempt_status, memory_order_relaxed) != 0u;"
    "atomic attempt-status read")

require_contains("${camera_pose_kernel}"
    "device atomic_uint* attempt_status"
    "camera-pose status ABI")
require_contains("${camera_pose_kernel}"
    "constant uint& attempt_gating_enabled"
    "camera-pose enable ABI")
require_ordered("${camera_pose_kernel}"
    "attempt_gating_enabled != 0u"
    "training_attempt_failed(attempt_status)"
    "camera-pose opt-in gate")
require_ordered("${camera_pose_kernel}"
    "training_attempt_failed(attempt_status)"
    "pose_deltas[params.pose_offset] ="
    "camera-pose guard precedes mutation")

require_contains("${appearance_kernel}"
    "device atomic_uint* attempt_status"
    "appearance status ABI")
require_contains("${appearance_kernel}"
    "constant uint& attempt_gating_enabled"
    "appearance enable ABI")
require_ordered("${appearance_kernel}"
    "attempt_gating_enabled != 0u"
    "training_attempt_failed(attempt_status)"
    "appearance opt-in gate")
require_ordered("${appearance_kernel}"
    "training_attempt_failed(attempt_status)"
    "adam_update_element(features_dc[dc_idx + c]"
    "appearance guard precedes mutation")

require_contains("${geometry_kernel}"
    "device atomic_uint* attempt_status"
    "geometry status ABI")
require_contains("${geometry_kernel}"
    "constant uint& attempt_gating_enabled"
    "geometry enable ABI")
require_ordered("${geometry_kernel}"
    "attempt_gating_enabled != 0u"
    "training_attempt_failed(attempt_status)"
    "geometry opt-in gate")
require_ordered("${geometry_kernel}"
    "training_attempt_failed(attempt_status)"
    "threadgroup atomic_uint pose_group_gradient[6];"
    "geometry guard precedes threadgroup barriers")
require_ordered("${geometry_kernel}"
    "training_attempt_failed(attempt_status)"
    "vis_counts[idx] += 1.0f;"
    "geometry guard precedes densification statistics")
require_ordered("${geometry_kernel}"
    "training_attempt_failed(attempt_status)"
    "adam_update_element(\n            means3d[parameter_index]"
    "geometry guard precedes Adam mutation")

require_contains("${photometric_kernel}"
    "device atomic_uint* attempt_status"
    "photometric status ABI")
require_contains("${photometric_kernel}"
    "constant uint& attempt_gating_enabled"
    "photometric enable ABI")
require_ordered("${photometric_kernel}"
    "attempt_gating_enabled != 0u"
    "training_attempt_failed(attempt_status)"
    "photometric opt-in gate")
require_ordered("${photometric_kernel}"
    "training_attempt_failed(attempt_status)"
    "exp_avg[index] = first_moment;"
    "photometric guard precedes mutation")

# Keep the host/shader ABIs coupled. These bindings are required even when the
# optional optimizer is disabled because all enabled persistent updates share
# the status written by validation and defensive sorting.
require_contains("${loss_and_photometric_host}"
    "ENC_BUF(enc, overflow_flag, 12);"
    "photometric attempt-status binding")
require_contains("${loss_and_photometric_host}"
    "ENC_SCALAR(enc, attemptGatingEnabled, 13);"
    "photometric attempt-enable binding")
require_contains("${terminal_update_host}"
    "ENC_BUF(enc, overflow_flag, 18);"
    "appearance attempt-status binding")
require_contains("${terminal_update_host}"
    "ENC_SCALAR(enc, attemptGatingEnabled, 19);"
    "appearance attempt-enable binding")
require_contains("${terminal_update_host}"
    "ENC_BUF(enc, overflow_flag, 27);"
    "geometry attempt-status binding")
require_contains("${terminal_update_host}"
    "ENC_SCALAR(enc, attemptGatingEnabled, 28);"
    "geometry attempt-enable binding")
require_contains("${terminal_update_host}"
    "ENC_BUF(enc, overflow_flag, 5);"
    "camera-pose attempt-status binding")
require_contains("${terminal_update_host}"
    "ENC_SCALAR(enc, attemptGatingEnabled, 6);"
    "camera-pose attempt-enable binding")

# The public entry point owns one engine lock across every attempt. GPU-resident
# retry preflight retires validation, scatter, and sort before raster/backward
# can mutate persistent state. A packed capacity failure grows every resource
# derived from the completed layout, then replays the exact arguments through
# the locked helper while the caller's logical-step handle, camera, optimizer
# candidates, and iteration are still unchanged. Successful post-count work
# remains queued so the next step's count validation can pipeline behind it.
require_ordered("${public_training_wrapper}"
    "std::lock_guard<std::mutex> lock(g_engine_mutex);"
    "return msplat_train_step_locked("
    "public wrapper locks before forwarding")
require_count("${public_training_wrapper}"
    "return msplat_train_step_locked\\(" 1
    "public locked-helper forwards")
require_absent("${public_training_wrapper}" "lock.unlock()"
    "public training lock release")
require_absent("${training_pipeline}"
    "std::lock_guard<std::mutex> lock(g_engine_mutex);"
    "nested locked-helper engine lock")
require_ordered("${retry_preflight}"
    "encode_scatter_sort_finalize(enc);"
    "ctx->syncCB();"
    "all status writers complete before retirement")
require_ordered("${retry_preflight}"
    "ctx->syncCB();"
    "const auto completedAttempt = inspectCompletedRetryAttempt();"
    "preflight completion precedes status readback")
require_ordered("${retry_preflight}"
    "const auto completedAttempt = inspectCompletedRetryAttempt();"
    "return replaySameLogicalStep();"
    "preflight retry decision precedes replay")
require_ordered("${retry_inspection}"
    "overflow_flag.data<uint32_t>()[0]"
    "g_tcache.ensure_intersection_arena("
    "attempt status readback precedes arena growth")
require_ordered("${retry_inspection}"
    "MSPLAT_TRAINING_OVERFLOW_TILE_CAP"
    "throw std::length_error("
    "non-growable tile failure is rejected")
require_ordered("${retry_inspection}"
    "MSPLAT_TRAINING_OVERFLOW_PACKED_CAPACITY"
    "g_tcache.ensure_intersection_arena("
    "packed failure grows the intersection arena")
require_contains("${retry_inspection}"
    "g_tcache.ensure_forward_chunks("
    "forward chunk growth")
require_contains("${retry_inspection}"
    "g_tcache.ensure_backward_chunks("
    "backward chunk growth")
require_ordered("${retry_inspection}"
    "if (!grew) {"
    "return std::make_pair(completedLayout, true);"
    "non-progress rejection precedes retry result")
require_absent("${retry_inspection}" "lock.unlock()"
    "inter-attempt engine unlock")
require_count("${retry_replay}"
    "return msplat_train_step_locked\\(" 1
    "same-step replay calls")
require_contains("${retry_replay}"
    "projmat, fx, fy, cx, cy, img_height, img_width, tile_bounds,"
    "same camera and resolution replay")
require_contains("${retry_replay}"
    "features_rest, opacities, background, gt, coverage_mask,"
    "same model and target replay")
require_contains("${retry_replay}"
    "adam_eps, photometric, pose, collect_densification_stats,"
    "same optimizer and refinement replay")
require_contains("${retry_preflight}" "do_blit_zero(commandBuffer)"
    "preflight transient clear")
require_contains("${retry_preflight}" "encode_scatter_sort_finalize(enc);"
    "preflight status writers")
require_absent("${retry_preflight}" "encode_pack("
    "preflight attribute packing")
require_absent("${retry_preflight}" "encode_rast_fwd("
    "preflight raster encoding")
require_absent("${retry_preflight}" "encode_proj_sh_bwd_adam("
    "preflight persistent update encoding")
require_contains("${post_count_dispatch}" "encode_pack(enc);"
    "asynchronous post-count packing")
require_absent("${post_count_dispatch}" "ctx->syncCB()"
    "post-count synchronization")
require_contains("${host_source}"
    "if (index == 2 && g_retry_stage_profile.load(std::memory_order_relaxed))\n        return \"pack_only\";"
    "retry stage profile labels the asynchronous sample as pack-only")
foreach(forbidden_retry_action IN ITEMS
        msplat_training_step_begin
        msplat_training_step_submit
        schedulersStep
        afterTrain
        advanceCamera)
    require_absent("${retry_replay}" "${forbidden_retry_action}"
        "retry-side ${forbidden_retry_action}")
endforeach()

# A recovered attempt remains visible in the completed logical-step telemetry,
# while timing and arena-growth costs accumulate across every attempt. The final
# successful readback contributes its live status in addition to recovered bits.
require_contains("${layout_telemetry}"
    "intersectionArenaGrowMs += arenaGrowDurationMs;"
    "bootstrap arena-growth accumulation")
require_contains("${recovered_retry_telemetry}"
    "if (sealed || aborted) return;"
    "recovered-retry telemetry lifecycle gate")
require_contains("${recovered_retry_telemetry}"
    "recoveredOverflowReasons |= overflowReasons;"
    "recovered overflow accumulation")
require_contains("${recovered_retry_telemetry}"
    "intersectionArenaGrowMs += arenaGrowDurationMs;"
    "retry arena-growth accumulation")
require_contains("${post_count_telemetry}"
    "postCountEncodeMs += elapsedMilliseconds(encodeStart, encodeEnd);"
    "multi-attempt encode-time accumulation")
require_absent("${logical_step_telemetry}"
    "intersectionArenaGrowMs = arenaGrowDurationMs;"
    "overwriting arena-growth telemetry")
require_absent("${logical_step_telemetry}"
    "postCountEncodeMs = elapsedMilliseconds(encodeStart, encodeEnd);"
    "overwriting post-count encode telemetry")
require_contains("${logical_step_telemetry}"
    "uint32_t recoveredOverflowReasons = MSPLAT_TRAINING_OVERFLOW_NONE;"
    "recovered-overflow initial state")
require_contains("${completed_telemetry}"
    "completed.overflowReasons = recoveredOverflowReasons |"
    "recovered overflow publication")
require_ordered("${completed_telemetry}"
    "completed.overflowReasons = recoveredOverflowReasons |"
    "telemetry->publishCompleted("
    "recovered overflow published with the completed step")
require_ordered("${retry_inspection}"
    "const auto retryGrowStart = TelemetryClock::now();"
    "g_tcache.ensure_intersection_arena("
    "retry grow timing begins before allocation")
require_ordered("${retry_inspection}"
    "g_tcache.ensure_backward_chunks("
    "const auto retryGrowEnd = TelemetryClock::now();"
    "retry grow timing ends after all allocations")
require_ordered("${retry_inspection}"
    "logicalStep->recordRecoveredIntersectionRetry("
    "return std::make_pair(completedLayout, true);"
    "recovered telemetry recorded before retry result")
require_contains("${retry_inspection}"
    "attemptFailures,\n                    elapsedMilliseconds(retryGrowStart, retryGrowEnd)"
    "recovered reason and grow-duration recording")
require_ordered("${training_pipeline}"
    "logicalStep->recordIntersectionLayout("
    "const auto postCountEncodeStart"
    "successful preflight telemetry precedes post-count encoding")

# Multiple count validations can belong to one recovered logical step. Their
# timing must accumulate rather than report only the replay's final pass.
require_contains("${logical_step_telemetry}"
    "countWaitWallMs += metrics.waitWallMs;"
    "count-wait accumulation")
require_contains("${logical_step_telemetry}"
    "countGpuMs += metrics.gpuExecutionMs;"
    "count-GPU accumulation")
require_absent("${logical_step_telemetry}"
    "countWaitWallMs = metrics.waitWallMs;"
    "overwriting count-wait telemetry")

# The macOS Metal fixture deterministically warms a small arena, then submits a
# 5,000-Gaussian single-tile step. It proves the rejected attempt is a no-op,
# the replay succeeds once, and recovered overflow reaches aggregate telemetry.
require_contains("${retry_fixture}"
    "std::string(mode) != \"retry\""
    "retry-only fixture gate")
require_contains("${retry_fixture}" "constexpr int gaussianCount = 5'000;"
    "forced under-capacity population")
require_contains("${retry_fixture}"
    "false, false, false, true, true);"
    "fixture statistics and telemetry enablement")
require_contains("${retry_fixture}"
    "CHECK(retried.visibilityCount == 1.0f);"
    "first-Gaussian single successful mutation")
require_contains("${retry_fixture}"
    "CHECK(retried.lastVisibilityCount == 1.0f);"
    "last-Gaussian single successful mutation")
require_contains("${retry_fixture}"
    "MSPLAT_TRAINING_OVERFLOW_PACKED_CAPACITY"
    "recovered packed-capacity reason")
require_contains("${retry_fixture}"
    "CHECK(std::isfinite(retried.intersectionArenaGrowMs));"
    "finite recovered grow telemetry")
require_contains("${retry_fixture}"
    "CHECK(retried.intersectionArenaGrowMs >= 0.0);"
    "nonnegative recovered grow telemetry")
require_contains("${retry_fixture}"
    "CHECK(retried.commandBufferCount == 3u);"
    "failed and successful preflights plus queued post-count work")
require_contains("${retry_fixture}"
    "CHECK(retried.overflowedStepCount == 1u);"
    "aggregate recovered-overflow count")
require_contains("${retry_fixture}"
    "CHECK(retried.tileCapOverflowedStepCount == 0u);"
    "no spurious tile-cap count")
require_contains("${retry_fixture}"
    "CHECK(retried.packedCapacityOverflowedStepCount == 1u);"
    "aggregate packed-capacity count")
require_contains("${retry_fixture}"
    "CHECK(retried.lastOverflowIteration == 1);"
    "recovered-overflow iteration")
require_contains("${retry_fixture_source}"
    "checkArenaRetryTransaction();"
    "forced-retry fixture invocation")
require_contains("${cmake_source}"
    "add_test(NAME msplat_transparent_training_arena_retry"
    "forced-retry fixture registration")
require_contains("${cmake_source}"
    "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=enumerated;MSPLAT_TRAINING_ARENA_MODE=retry"
    "forced-retry fixture environment")
require_contains("${cmake_source}"
    "MSPLAT_RASTER_VARIANT=8x8;MSPLAT_TILE_COUNT_MODE=difference;MSPLAT_TRAINING_ARENA_MODE=retry"
    "device-matching difference/retry fixture environment")

# CPU optimizer counters are candidate values while the Metal attempt is being
# encoded. They become persistent only after msplat_train_step has accepted the
# logical step, so a failed encode or retry cannot consume bias-correction time.
string(FIND "${model_source}"
    "void Model::fullIteration(Camera& cam, size_t cameraIndex, int step,\n                          const CameraTrainingTarget& target,"
    model_iteration_start)
if(model_iteration_start EQUAL -1)
    message(FATAL_ERROR
        "Training-arena retry contract cannot find canonical Model::fullIteration")
endif()
string(SUBSTRING "${model_source}" ${model_iteration_start} -1 model_iteration)
require_ordered("${model_iteration}"
    "MTensor r = msplat_train_step("
    "adam_step_count = nextAdamStep;"
    "global Adam counter commits after accepted Metal step")
require_ordered("${model_iteration}"
    "MTensor r = msplat_train_step("
    "cameraLogGainStepCounts[cameraIndex] = nextPhotometricStep;"
    "photometric counter commits after accepted Metal step")
require_ordered("${model_iteration}"
    "MTensor r = msplat_train_step("
    "cameraPoseStepCounts[cameraIndex] = nextPoseStep;"
    "pose counter commits after accepted Metal step")

# Trainer-visible state advances once, after the retrying native step returns.
# In particular, camera choice and iteration remain stable across attempts, and
# topology/scheduler work cannot observe a failed attempt.
extract_section("${api_source}" "Stats Trainer::step() {"
    "void Trainer::train(" trainer_step)
require_ordered("${trainer_step}"
    "size_t camIdx = impl->currentCamera();"
    "impl->model->fullIteration("
    "camera selected before the retrying model call")
require_ordered("${trainer_step}"
    "impl->model->fullIteration("
    "impl->model->schedulersStep(nextStep);"
    "scheduler waits for accepted step")
require_ordered("${trainer_step}"
    "impl->model->schedulersStep(nextStep);"
    "impl->model->afterTrain(nextStep);"
    "topology maintenance follows the scheduler")
require_ordered("${trainer_step}"
    "cpuSubmitMs = msplat_training_step_submit(logicalStep, descriptor);"
    "impl->advanceCamera();"
    "camera advances only after logical-step submission")
require_ordered("${trainer_step}"
    "impl->advanceCamera();"
    "impl->currentStep = nextStep;"
    "iteration commits after camera advancement")
