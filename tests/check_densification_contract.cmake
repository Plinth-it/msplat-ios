if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/bindings.h" bindings_header)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/model.cpp" model_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/msplat_api.mm" api_source)
file(READ "${MSPLAT_SOURCE_DIR}/python/bindings.cpp" python_source)
file(READ "${MSPLAT_SOURCE_DIR}/tests/test_densification.mm" test_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Densification contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Stale densification contract remains: ${label}")
    endif()
endfunction()

function(extract_section source_name start_marker end_marker output_name)
    string(FIND "${${source_name}}" "${start_marker}" section_start)
    if(section_start EQUAL -1)
        message(FATAL_ERROR
            "Densification contract section start missing: ${start_marker}")
    endif()
    string(SUBSTRING "${${source_name}}" ${section_start} -1 section_tail)
    string(FIND "${section_tail}" "${end_marker}" section_length)
    if(section_length EQUAL -1 OR section_length EQUAL 0)
        message(FATAL_ERROR
            "Densification contract section end missing: ${end_marker}")
    endif()
    string(SUBSTRING "${section_tail}" 0 ${section_length}
        section_contents)
    set(${output_name} "${section_contents}" PARENT_SCOPE)
endfunction()

extract_section(metal_source "kernel void densify_classify_kernel("
    "kernel void densify_append_split_kernel(" classifier)

extract_section(metal_source "kernel void reset_opacity_state_kernel("
    "// ============================================================\n// GPU Densification Kernels"
    opacity_reset_kernel)
extract_section(host_source "void msplat_reset_opacity_state("
    "// ============================================================================\n// GPU-native densification"
    opacity_reset_host)
extract_section(opacity_reset_host "void msplat_reset_opacity_state("
    "    if (encoderCreationFailed) {" opacity_reset_normal)
extract_section(opacity_reset_host "    if (encoderCreationFailed) {"
    "\n    }\n}" opacity_reset_fallback)
extract_section(host_source "id resources[] = {"
    "        };\n        for (id resource : resources)" pipeline_release_list)
extract_section(model_source
    "if (step < stopSplitAt && step % resetInterval == refineEvery){"
    "        xysGradNorm.reset();" opacity_reset_model)
extract_section(api_source "Stats Trainer::step()"
    "void Trainer::train(" trainer_step)
extract_section(python_source "TrainingStats step()"
    "void train(nb::object callback" python_trainer_step)

require_contains("${classifier}"
    "bool do_dup = !do_split && high_grad;"
    "mutually exclusive split and duplicate outcomes")
require_absent("${classifier}"
    "bool do_dup = !is_large && high_grad;"
    "world-space-only duplicate predicate")
require_contains("${host_source}"
    "throw std::runtime_error(\"Densification classified one Gaussian twice\")"
    "host overlap invariant")

# The periodic opacity maintenance stays ordered after Adam in the same command
# buffer while preserving the prior CPU comparison's NaN and active-view rules.
foreach(binding IN ITEMS
        "device float* opacities      [[buffer(0)]]"
        "device float* exp_avg        [[buffer(1)]]"
        "device float* exp_avg_sq     [[buffer(2)]]"
        "constant uint& count         [[buffer(3)]]"
        "constant float& max_logit    [[buffer(4)]]")
    require_contains("${opacity_reset_kernel}" "${binding}"
        "opacity reset ABI ${binding}")
endforeach()
require_contains("${opacity_reset_kernel}" "if (idx >= count) return;"
    "active-prefix bound")
require_contains("${opacity_reset_kernel}"
    "if (opacities[idx] > max_logit) opacities[idx] = max_logit;"
    "conditional opacity ceiling")
require_contains("${opacity_reset_kernel}" "exp_avg[idx] = 0.0f;"
    "first-moment clear")
require_contains("${opacity_reset_kernel}" "exp_avg_sq[idx] = 0.0f;"
    "second-moment clear")

require_contains("${bindings_header}" "void msplat_reset_opacity_state("
    "internal opacity-reset binding")
require_contains("${host_source}"
    "id<MTLComputePipelineState> reset_opacity_state_kernel_cpso = nil;"
    "opacity-reset pipeline ownership")
require_contains("${host_source}"
    "load(@\"reset_opacity_state_kernel\")"
    "opacity-reset pipeline load")
require_contains("${pipeline_release_list}"
    "reset_opacity_state_kernel_cpso,"
    "opacity-reset pipeline release")
foreach(token IN ITEMS
        "opacities.numel() <= 0"
        "exp_avg.numel() != opacities.numel()"
        "std::isfinite(max_logit)"
        "std::numeric_limits<uint32_t>::max()"
        "ctx->getCommandBuffer()"
        "ENC_BUF(encoder, opacities, 0);"
        "ENC_BUF(encoder, exp_avg, 1);"
        "ENC_BUF(encoder, exp_avg_sq, 2);"
        "ENC_SCALAR(encoder, count, 3);"
        "ENC_SCALAR(encoder, max_logit, 4);"
        "reset_opacity_state_kernel_cpso.maxTotalThreadsPerThreadgroup"
        "[encoder dispatchThreads:MTLSizeMake(count, 1, 1)"
        "[encoder endEncoding];")
    require_contains("${opacity_reset_host}" "${token}"
        "opacity reset host ${token}")
endforeach()
require_absent("${opacity_reset_normal}" "ctx->syncCB()"
    "opacity reset synchronizes its normal path")
require_absent("${opacity_reset_normal}" "commitCB()"
    "opacity reset commits its command buffer")
require_absent("${opacity_reset_normal}" "waitUntilCompleted"
    "opacity reset waits on its normal path")
require_contains("${opacity_reset_fallback}" "if (encoderCreationFailed) {"
    "encoder failure fallback")
require_contains("${opacity_reset_fallback}" "ctx->syncCB();"
    "fallback completion before CPU maintenance")
string(FIND "${opacity_reset_fallback}" "ctx->syncCB();" fallback_sync_pos)
string(FIND "${opacity_reset_fallback}" "opacities.data<float>()"
    fallback_cpu_pos)
if(fallback_sync_pos EQUAL -1 OR fallback_cpu_pos EQUAL -1 OR
   NOT fallback_sync_pos LESS fallback_cpu_pos)
    message(FATAL_ERROR
        "Densification contract missing: fallback sync before shared-memory access")
endif()
math(EXPR fallback_after_sync "${fallback_sync_pos} + 14")
string(SUBSTRING "${opacity_reset_fallback}" ${fallback_after_sync} -1
    fallback_after_sync_contents)
string(FIND "${fallback_after_sync_contents}" "ctx->syncCB();"
    second_fallback_sync_pos)
if(NOT second_fallback_sync_pos EQUAL -1)
    message(FATAL_ERROR
        "Densification contract violation: multiple opacity-reset fallback syncs")
endif()
require_contains("${opacity_reset_fallback}"
    "if (opacityValues[index] > max_logit)"
    "fallback NaN-preserving comparison")
require_contains("${opacity_reset_fallback}"
    "std::memset(exp_avg.data_ptr(), 0, exp_avg.nbytes());"
    "fallback first-moment clear")
require_contains("${opacity_reset_fallback}"
    "std::memset(exp_avg_sq.data_ptr(), 0, exp_avg_sq.nbytes());"
    "fallback second-moment clear")

require_contains("${opacity_reset_model}"
    "constexpr float resetLogit = -1.3862943611198906f;"
    "unchanged reset ceiling")
require_contains("${opacity_reset_model}"
    "msplat_reset_opacity_state(\n                opacities, adam_exp_avg[5], adam_exp_avg_sq[5],"
    "scheduled GPU reset")
foreach(stale IN ITEMS
        "msplat_gpu_sync()"
        ".data<float>()"
        ".zero()")
    require_absent("${opacity_reset_model}" "${stale}"
        "host opacity maintenance ${stale}")
endforeach()

string(FIND "${trainer_step}" "impl->model->fullIteration(" iteration_pos)
string(FIND "${trainer_step}" "impl->model->afterTrain(nextStep);" maintenance_pos)
string(FIND "${trainer_step}" "msplat_training_step_submit(" submit_pos)
if(iteration_pos EQUAL -1 OR maintenance_pos EQUAL -1 OR submit_pos EQUAL -1 OR
   NOT iteration_pos LESS maintenance_pos OR
   NOT maintenance_pos LESS submit_pos)
    message(FATAL_ERROR
        "Densification contract missing: opacity reset command-buffer ordering")
endif()

string(FIND "${python_trainer_step}" "model->fullIteration(" python_iteration_pos)
string(FIND "${python_trainer_step}" "model->afterTrain(current_step);"
    python_maintenance_pos)
string(FIND "${python_trainer_step}" "msplat_commit();" python_commit_pos)
if(python_iteration_pos EQUAL -1 OR python_maintenance_pos EQUAL -1 OR
   python_commit_pos EQUAL -1 OR
   NOT python_iteration_pos LESS python_maintenance_pos OR
   NOT python_maintenance_pos LESS python_commit_pos)
    message(FATAL_ERROR
        "Densification contract missing: Python opacity reset submission ordering")
endif()

foreach(token IN ITEMS
        "constexpr uint32_t payloadNaNBits = 0x7fc12345u;"
        "MTensor opacities = opacityBacking.view(activeCount);"
        "originalOpacityBits[index]"
        "originalFirstMomentBits[index]"
        "originalSecondMomentBits[index]"
        "resetOpacityBits[index]"
        "resetFirstMomentBits[index]"
        "resetSecondMomentBits[index]"
        "CHECK(rejectedMismatch);")
    require_contains("${test_source}" "${token}"
        "opacity reset runtime coverage ${token}")
endforeach()
