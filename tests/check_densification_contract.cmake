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
file(READ "${MSPLAT_SOURCE_DIR}/CMakeLists.txt" cmake_source)

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
extract_section(metal_source "inline uint densify_mix_bits("
    "kernel void densify_append_dup_kernel(" densify_split_random)

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
extract_section(cmake_source
    "add_test(NAME msplat_densification_classification"
    "add_executable(msplat_transparent_training_tests"
    densification_ctests)

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

# GPU split randomness is an explicit A/B. CPU mode retains the exact libc++
# stream by default; GPU mode derives a fixed-cost normal stream from step and
# dense split ordinal without touching the shared sample buffer.
foreach(token IN ITEMS
        "value *= 0x7feb352du;"
        "value *= 0x846ca68bu;"
        "constant bool densify_random_on_gpu [[function_constant(0)]];"
        "inline float densify_uniform_open(uint bits)"
        "float((bits >> 8) | 1u) * (1.0f / 16777216.0f)"
        "inline float2 densify_normal_pair(uint seed, uint pair_index)"
        "float radius = sqrt(-2.0f * log(u1));"
        "constant float* random_samples   [[buffer(3)]]"
        "constant uint& random_seed       [[buffer(24)]]"
        "if (densify_random_on_gpu) {"
        "uint first_pair = uint(ord) * 3u;"
        "gpu_pair0 = densify_normal_pair(random_seed, first_pair);"
        "gpu_pair1 = densify_normal_pair(random_seed, first_pair + 1u);"
        "gpu_pair2 = densify_normal_pair(random_seed, first_pair + 2u);"
        "r0 = random_samples[rand_idx*3];")
    require_contains("${densify_split_random}" "${token}"
        "GPU densification random ${token}")
endforeach()
require_contains("${host_source}" "bool gpu_densify_random = false;"
    "CPU densification-random default")
require_contains("${host_source}"
    "std::getenv(\"MSPLAT_DENSIFY_RANDOM_MODE\")"
    "densification-random environment lookup")
require_contains("${host_source}"
    "MSPLAT_DENSIFY_RANDOM_MODE must be cpu or gpu"
    "densification-random mode validation")
foreach(token IN ITEMS
        "auto loadBoolSpecialization ="
        "[constants setConstantValue:&value type:MTLDataTypeBool atIndex:index];"
        "@\"densify_append_split_kernel\", 0, ctx->gpu_densify_random)")
    require_contains("${host_source}" "${token}"
        "densification-random PSO specialization ${token}")
endforeach()
require_absent("${densify_split_random}" "[[buffer(25)]]"
    "runtime densification-random shader branch")
foreach(token IN ITEMS
        "const uint32_t generate_random_on_gpu ="
        "if (generate_random_on_gpu == 0u) {"
        "std::mt19937 generator(random_seed);"
        "std::normal_distribution<float> distribution(0.0f, 1.0f);"
        "for (int64_t index = 0; index < 6LL * num_splits; ++index)"
        "ENC_SCALAR(enc, random_seed, 24);")
    require_contains("${host_source}" "${token}"
        "densification-random host ${token}")
endforeach()
require_contains("${bindings_header}"
    "MTensor &random_samples, uint32_t random_seed"
    "densification-random binding seed")
require_contains("${bindings_header}"
    "bool msplat_densify_uses_gpu_random();"
    "densification-random mode query")
foreach(token IN ITEMS
        "bool msplat_densify_uses_gpu_random()"
        "return get_global_context()->gpu_densify_random;"
        "generate_random_on_gpu != 0u ? 1LL : 6LL * num_splits")
    require_contains("${host_source}" "${token}"
        "densification-random scratch ${token}")
endforeach()
foreach(token IN ITEMS
        "if (msplat_densify_uses_gpu_random()) {"
        "scratch.randomSamples = gpu_zeros({1}, DType::Float32);"
        "if (!msplat_densify_uses_gpu_random()) {"
        "tasks.push_back({&densify_random_samples, {new_cap, 3}, false});")
    require_contains("${model_source}" "${token}"
        "densification-random model scratch ${token}")
endforeach()
require_contains("${model_source}"
    "densify_random_samples, static_cast<uint32_t>(step)"
    "logical-step densification seed")
require_absent("${model_source}" "std::normal_distribution<float>"
    "model-side densification random fill")

foreach(token IN ITEMS
        "add_test(NAME msplat_densification_classification"
        "add_test(NAME msplat_densification_cpu_random"
        "add_test(NAME msplat_densification_gpu_random"
        "add_test(NAME msplat_densification_invalid_random_mode"
        "--unset=MSPLAT_DENSIFY_RANDOM_MODE --"
        "MSPLAT_DENSIFY_RANDOM_MODE=cpu --"
        "MSPLAT_DENSIFY_RANDOM_MODE=gpu --"
        "MSPLAT_DENSIFY_RANDOM_MODE=invalid --"
        "                default)"
        "                cpu)"
        "                gpu)"
        "                invalid)"
        "msplat_densification_cpu_random\n            msplat_densification_gpu_random"
        "PROPERTIES SKIP_RETURN_CODE 77")
    require_contains("${densification_ctests}" "${token}"
        "densification-random CTest ${token}")
endforeach()
require_absent("${densification_ctests}" "WILL_FAIL"
    "invalid-mode false-positive test")
foreach(token IN ITEMS
        "constexpr uint32_t kRandomScratchSentinelBits = 0x7f7fffffu;"
        "DensifyRandomResult runDensifyRandomFixture("
        "uint32_t randomSeed, bool gpuMode,"
        "bool forceUndersizedRandomScratch = false)"
        "CHECK(msplat_densify_uses_gpu_random() == gpuMode);"
        "CHECK(first.randomScratchBits.size() == 1u);"
        "runDensifyRandomFixture(600u, false, true);"
        "Densification buffer is too small: random_samples"
        "first.childMeanBits == repeated.childMeanBits"
        "first.childMeanBits != different.childMeanBits"
        "bits == kRandomScratchSentinelBits"
        "std::mt19937 expectedGenerator(600u);"
        "floatBits(expectedDistribution(expectedGenerator))"
        "expectedGpuDensifySamples(600u, ordinal)"
        "std::abs(actual - expected[component]) <= tolerance"
        "const std::string expectedMode = argv[2];"
        "CHECK(expectedMode == configuredMode);"
        "if (expectedMode == \"invalid\") {"
        "MSPLAT_DENSIFY_RANDOM_MODE must be cpu or gpu"
        "std::abs(mean) < 0.08"
        "variance > 0.80"
        "variance < 1.20")
    require_contains("${test_source}" "${token}"
        "densification-random runtime coverage ${token}")
endforeach()
