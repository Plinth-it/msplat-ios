if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/model.cpp" model_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/src/msplat_api.mm" api_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/include/msplat_c_api.h" c_api_header)
file(READ "${MSPLAT_SOURCE_DIR}/cli/msplat.cpp" cli_source)
file(READ "${MSPLAT_SOURCE_DIR}/python/bindings.cpp" python_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Pose refinement contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Pose refinement leaked into ${label}")
    endif()
endfunction()

function(extract_section source_name start_marker end_marker output_name)
    string(FIND "${${source_name}}" "${start_marker}" section_start)
    if(section_start EQUAL -1)
        message(FATAL_ERROR
            "Pose contract section start missing: ${start_marker}")
    endif()
    string(SUBSTRING "${${source_name}}" ${section_start} -1 section_tail)
    string(FIND "${section_tail}" "${end_marker}" section_length)
    if(section_length EQUAL -1 OR section_length EQUAL 0)
        message(FATAL_ERROR
            "Pose contract section end missing: ${end_marker}")
    endif()
    string(SUBSTRING "${section_tail}" 0 ${section_length}
        section_contents)
    set(${output_name} "${section_contents}" PARENT_SCOPE)
endfunction()

extract_section(metal_source "kernel void prepare_camera_pose_kernel("
    "struct PoseAdamParams" pose_prepare)
extract_section(metal_source "kernel void camera_pose_adam_kernel("
    "// ===== Fused Projection + SH Kernels" pose_adam)
extract_section(metal_source "kernel void project_and_sh_forward_kernel("
    "kernel void project_and_sh_backward_kernel(" fused_forward)
extract_section(metal_source "kernel void project_and_sh_backward_kernel("
    "// ===== Exact Tile Intersection Pipeline" fused_backward)

# D * V0 convention and the camera center derived from the same refined view.
require_contains("${pose_prepare}"
    "const float3x3 rotation = correction_rotation * declared_rotation;"
    "left-multiplied camera-space correction")
require_contains("${pose_prepare}"
    "correction_rotation * declared_translation + translation;"
    "left-composed view translation")
require_contains("${pose_prepare}"
    "-(transpose(rotation) * view_translation);"
    "refined camera center")

# The disabled path must retain the established projection and VJP arithmetic.
require_contains("${fused_forward}"
    "pose_enabled != 0\n        ? project_view_pix(p_view, fx, fy, cx, cy)\n        : project_pix(projmat, p_world, img_size, {cx, cy});"
    "canonical forward projection fallback")
require_contains("${fused_backward}"
    "Preserve the canonical projection VJP arithmetic exactly when pose"
    "canonical backward branch")
require_contains("${fused_backward}"
    "project_pix_vjp(\n                projmat, p_world, img_size"
    "canonical projection VJP")

# The geometric tangent contains both point and anisotropic-covariance terms.
require_contains("${fused_backward}"
    "cross(p_view, pose_v_p_view) +"
    "local left-perturbation point gradient")
require_contains("${fused_backward}"
    "cross(W[0], pose_v_view_rotation[0]) +"
    "covariance rotation column zero")
require_contains("${fused_backward}"
    "cross(W[1], pose_v_view_rotation[1]) +"
    "covariance rotation column one")
require_contains("${fused_backward}"
    "cross(W[2], pose_v_view_rotation[2]);"
    "covariance rotation column two")
require_contains("${fused_backward}"
    "Appearance direction is deliberately detached"
    "geometry-only SH policy")

# Reduce once per threadgroup before touching the shared six-value gradient.
require_contains("${fused_backward}"
    "threadgroup atomic_float pose_group_gradient[6];"
    "threadgroup pose reduction")
require_contains("${fused_backward}"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "threadgroup reduction barrier")
require_contains("${fused_backward}"
    "&pose_gradient[component],\n                    atomic_load_explicit("
    "single global reduction per component")

# Adam applies a negative tangent step by left composition, then norm bounds.
require_contains("${pose_adam}"
    "update[component] = -params.step_size * first_moment /"
    "descent sign")
require_contains("${pose_adam}"
    "float3x3 updated_rotation = increment_rotation * rotation;"
    "left-composed rotation update")
require_contains("${pose_adam}"
    "translation = increment_rotation * translation + increment_translation;"
    "SE3 translation composition")
require_contains("${pose_adam}"
    "if (translation_norm > params.max_translation)"
    "translation norm bound")
require_contains("${pose_adam}"
    "if (rotation_norm > params.max_rotation)"
    "rotation norm bound")

extract_section(host_source "auto encode_pose_prepare ="
    "auto encode_proj_sh =" host_prepare)
require_contains("${host_prepare}"
    "ENC_BUF(enc, viewmat, 0);"
    "declared view binding")
require_contains("${host_prepare}"
    "ENC_BUF(enc, poseDeltas, 1);"
    "pose-delta binding")
require_contains("${host_prepare}"
    "ENC_SCALAR(enc, cameraPoseOffset, 2);"
    "canonical camera-row binding")
require_contains("${host_prepare}"
    "ENC_BUF(enc, g_tcache.pose_cam_pos, 4);"
    "refined camera-center output binding")
require_contains("${host_prepare}"
    "[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];"
    "pose preparation barrier")

extract_section(host_source "static void render_pipeline("
    "MTensor msplat_render(" canonical_render_pipeline)
require_contains("${canonical_render_pipeline}"
    "const uint32_t poseDisabled = 0u;\n        ENC_SCALAR(enc, poseDisabled, 23);"
    "canonical render pose-disabled binding")

extract_section(host_source "auto encode_proj_sh_bwd_adam ="
    "// ========================== DISPATCH" host_backward)
require_contains("${host_backward}"
    "ENC_SCALAR(enc, poseEnabled, 28);\n        ENC_BUF(enc, poseGradient, 29);"
    "backward pose bindings")
require_contains("${host_backward}"
    "if (pose.enabled) {\n            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];\n            [enc setComputePipelineState:ctx->camera_pose_adam_kernel_cpso];"
    "gradient barrier before opt-in pose Adam")
require_contains("${host_backward}"
    "ENC_BUF(enc, poseDeltas, 0);\n            ENC_BUF(enc, poseGradient, 1);\n            ENC_BUF(enc, poseExpAvg, 2);\n            ENC_BUF(enc, poseExpAvgSq, 3);"
    "pose Adam state bindings")

extract_section(host_source "auto do_blit_zero ="
    "auto encode_step_readback =" blit_zero)
require_contains("${blit_zero}"
    "if (pose.enabled) {\n            [blit fillBuffer:poseGradient.buffer()"
    "opt-in pose-gradient zeroing")

# Forward and backward SH both consume the center made from the refined view.
extract_section(host_source
    "MTensor &activeViewmat = pose.enabled ? g_tcache.pose_viewmat : viewmat;"
    "// Pass 1 completes" host_forward)
require_contains("${host_forward}"
    "MTensor &activeViewmat = pose.enabled ? g_tcache.pose_viewmat : viewmat;"
    "disabled canonical view selection")
require_contains("${host_forward}"
    "ENC_BUF(enc, g_tcache.pose_cam_pos, 18);"
    "refined forward camera-center binding")
require_contains("${host_backward}"
    "ENC_BUF(enc, g_tcache.pose_cam_pos, 19);"
    "refined backward camera-center binding")

# Canonical render remains independent of training-only pose corrections.
extract_section(model_source "MTensor Model::render("
    "void Model::fullIteration(" canonical_model_render)
require_absent("${canonical_model_render}" "cameraPose"
    "canonical model render")
require_absent("${canonical_model_render}" "poseDeltas"
    "canonical model render")

# Optimizer rows use the absolute dataset-camera index; the API must pass it.
require_contains("${model_source}"
    "pose.cameraIndex = poseStepEnabled\n        ? static_cast<uint32_t>(cameraIndex)\n        : 0u;"
    "model canonical-camera pose row")
require_contains("${model_source}"
    "cameraIndex != static_cast<size_t>(poseAnchorCameraIndex);"
    "fixed canonical pose anchor")

extract_section(model_source "int Model::loadCheckpoint("
    "Model::CamSetup Model::prepareCam(" checkpoint_load)
require_contains("${checkpoint_load}"
    "checkpoint.poseEnabled && !refineCameraPoses"
    "pose-bearing checkpoint feature mismatch")
require_contains("${checkpoint_load}"
    "checkpoint.poseFrameIds != cameraFrameIds"
    "pose checkpoint frame identity comparison")
require_contains("${checkpoint_load}"
    "checkpoint.poseAnchorCameraIndex !="
    "pose checkpoint anchor comparison")
require_contains("${checkpoint_load}"
    "checkpointBasePoses != cameraBasePoses"
    "pose checkpoint source-geometry comparison")
require_contains("${checkpoint_load}"
    "newCameraPoseDeltas = gpu_zeros("
    "legacy checkpoint identity pose initialization")
require_contains("${checkpoint_load}"
    "newCameraPoseStepCounts.assign("
    "legacy checkpoint zero pose-step initialization")
require_contains("${checkpoint_load}"
    "deltas[anchorOffset + component] != 0.0f"
    "pose checkpoint zero anchor validation")
require_contains("${api_source}"
    "cam, impl->ds->trainIndices[camIdx], nextStep, target,"
    "C++ API canonical training-camera index")
require_contains("${c_api_header}"
    "#define MSPLAT_REFINEMENT_CAMERA_POSE_DELTAS     (1u << 1)"
    "C API camera-pose capability bit")
require_contains("${api_source}"
    "cfg.refineCameraPoses =\n            (refinementOptions->flags &\n             MSPLAT_REFINEMENT_CAMERA_POSE_DELTAS) != 0u;"
    "C API camera-pose option mapping")

# Selecting the fixed anchor must not dereference an empty training split
# before the normal model validation can return a recoverable error.
require_contains("${api_source}"
    "config.refineCameraPoses && !impl->ds->trainIndices.empty()"
    "C++ API empty-split anchor guard")
require_contains("${cli_source}"
    "refineCameraPoses && !cams.empty()"
    "native CLI empty-split anchor guard")
require_contains("${python_source}"
    "cfg.refine_camera_poses && !dataset.train_cams.empty()"
    "Python empty-split anchor guard")
