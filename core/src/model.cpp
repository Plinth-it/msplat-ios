#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <array>
#include <iterator>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include "model.hpp"
#include "atomic_output.hpp"
#include "camp_pose_point_sampler.hpp"
#include "densification_schedule.hpp"
#include "kdtree_tensor.hpp"
#include "msplat.hpp"
#include "loaders.hpp"

namespace fs = std::filesystem;

static const double C0 = 0.28209479177387814;

namespace {

// Densification grows the population in bursts, so a tight fit means a
// reallocation almost every refine step. Doubling instead leaves half the
// largest allocation the process ever makes sitting unused. A quarter is the
// compromise, with a floor so small scenes are not reallocating constantly.
int capacityWithSlack(int required) {
    int64_t slack = std::max<int64_t>(required / 4, 4096);
    int64_t capacity = static_cast<int64_t>(required) + slack;
    return static_cast<int>(std::min<int64_t>(capacity, std::numeric_limits<int>::max()));
}

constexpr bool collectsDensificationStats(int64_t step, int64_t stopStep) {
    return step < stopStep;
}

constexpr bool needsDensificationAfterStep(int64_t completedStep,
                                           int64_t stopStep) {
    return completedStep + 1 < stopStep;
}

constexpr float kPhotometricLearningRate = 1.0e-3f;
constexpr float kPhotometricRegularization = 1.0e-4f;
constexpr float kPhotometricMaxAbsLogGain = 1.38629436112f; // log(4)
constexpr int64_t kPpispParameterCount = 9;
constexpr float kPpispLearningRate = 2.0e-3f;
constexpr float kPpispRegularization = 1.0e-4f;
constexpr float kPpispMaxAbsExposure = 4.0f;
constexpr float kPpispMaxAbsColorParameter = 4.0f;
constexpr size_t kPhotometricMaxCameras = 1'000'000;
constexpr size_t kPhotometricMaxFrameIdBytes = 1U << 20;
constexpr size_t kPhotometricMaxFrameIdentityBytes = 64U << 20;
constexpr float kPoseLearningRate = 1.0e-4f;
constexpr float kPoseRegularization = 1.0e-4f;
constexpr float kPoseMaxTranslation = 0.05f;
constexpr float kPoseMaxRotation = 0.05235987756f; // 3 degrees

msplat::AppearanceMode normalizedAppearanceMode(
    bool refinePhotometricGains, msplat::AppearanceMode appearanceMode) {
    switch (appearanceMode) {
        case msplat::AppearanceMode::None:
            return refinePhotometricGains
                ? msplat::AppearanceMode::RgbGains
                : msplat::AppearanceMode::None;
        case msplat::AppearanceMode::RgbGains:
            return msplat::AppearanceMode::RgbGains;
        case msplat::AppearanceMode::PPISP:
            if (refinePhotometricGains) {
                throw std::invalid_argument(
                    "Legacy photometric gain refinement conflicts with PPISP appearance refinement");
            }
            return msplat::AppearanceMode::PPISP;
    }
    throw std::invalid_argument("Appearance refinement mode is not recognized");
}

uint64_t stablePoseCameraKey(const std::string& frameId) noexcept {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char byte : frameId) {
        hash ^= static_cast<uint64_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

static_assert(!collectsDensificationStats(1, 0));
static_assert(!collectsDensificationStats(1, 1));
static_assert(collectsDensificationStats(1, 2));
static_assert(!needsDensificationAfterStep(0, 1));
static_assert(needsDensificationAfterStep(0, 2));
static_assert(!needsDensificationAfterStep(1, 2));
static_assert(needsDensificationAfterStep(
    static_cast<int64_t>(std::numeric_limits<int>::max()) - 2,
    std::numeric_limits<int>::max()));
static_assert(!needsDensificationAfterStep(
    static_cast<int64_t>(std::numeric_limits<int>::max()) - 1,
    std::numeric_limits<int>::max()));

struct DensificationScratch {
    MTensor splitFlag, dupFlag;
    MTensor splitPrefix, dupPrefix;
    MTensor keepFlag, keepPrefix;
    MTensor blockTotals;
    MTensor compactScratch;
    MTensor randomSamples;
};

DensificationScratch makeDensificationScratch(int capacity,
                                               int64_t compactStride) {
    if (capacity <= 0 || compactStride < 4 ||
        static_cast<int64_t>(capacity) >
            std::numeric_limits<int64_t>::max() / compactStride) {
        throw std::invalid_argument("Invalid densification scratch dimensions");
    }

    DensificationScratch scratch;
    scratch.compactScratch = gpu_zeros(
        {static_cast<int64_t>(capacity) * compactStride}, DType::Float32);
    scratch.splitFlag = gpu_zeros({capacity}, DType::Int32);
    scratch.dupFlag = gpu_zeros({capacity}, DType::Int32);
    scratch.splitPrefix = gpu_zeros({capacity}, DType::Int32);
    scratch.dupPrefix = gpu_zeros({capacity}, DType::Int32);
    scratch.keepFlag = gpu_zeros({capacity}, DType::Int32);
    scratch.keepPrefix = gpu_zeros({capacity}, DType::Int32);
    const int64_t maxBlocks =
        (static_cast<int64_t>(capacity) + 1023) / 1024;
    scratch.blockTotals = gpu_zeros({maxBlocks}, DType::Int32);
    if (msplat_densify_uses_gpu_random()) {
        // The specialized GPU kernel derives samples from its seed and never
        // reads this binding. Metal still requires a valid non-empty buffer.
        scratch.randomSamples = gpu_zeros({1}, DType::Float32);
    } else {
        scratch.randomSamples = gpu_zeros({capacity, 3}, DType::Float32);
    }
    return scratch;
}

} // namespace

int numShBases(int degree){
    switch(degree){
        case 0: return 1;
        case 1: return 4;
        case 2: return 9;
        case 3: return 16;
        default: return 25;
    }
}

// Model constructor
Model::Model(const InputData &inputData, int numCameras,
    int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
    int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
    int maxSteps, bool keepCrs,
    const float* bgColor,
    int stopDensifyAt,
    int maxGaussians,
    bool refinePhotometricGains,
    bool refineCameraPoses,
    int poseAnchorCameraIndex,
    msplat::CameraPoseConditioning cameraPoseConditioning,
    bool transparentTrainingMasks,
    float transparentAlphaLossWeight,
    msplat::AppearanceMode requestedAppearanceMode)
    : numCameras(numCameras),
      datasetCameraCount(0),
      numDownscales(numDownscales), resolutionSchedule(resolutionSchedule),
      shDegree(shDegree), configuredSHDegree(shDegree),
      shDegreeInterval(shDegreeInterval),
      refineEvery(refineEvery), warmupLength(warmupLength), resetAlphaEvery(resetAlphaEvery),
      stopSplitAt(stopDensifyAt >= 0 ? stopDensifyAt : maxSteps / 2), densifyGradThresh(densifyGradThresh), densifySizeThresh(densifySizeThresh),
      stopScreenSizeAt(stopScreenSizeAt), splitScreenSize(splitScreenSize),
      maxSteps(maxSteps), maxGaussians(maxGaussians),
      appearanceMode(normalizedAppearanceMode(
          refinePhotometricGains, requestedAppearanceMode)),
      refineCameraPoses(refineCameraPoses),
      poseAnchorCameraIndex(poseAnchorCameraIndex),
      cameraPoseConditioning(cameraPoseConditioning),
      transparentTrainingMasks(transparentTrainingMasks),
      transparentAlphaLossWeight(transparentAlphaLossWeight),
      keepCrs(keepCrs) {

    if (inputData.points.count <= 0)
        throw std::invalid_argument("Dataset must contain sparse points");
    if (maxGaussians != -1 && maxGaussians <= 0)
        throw std::invalid_argument("maxGaussians must be -1 or greater than zero");
    if (!std::isfinite(transparentAlphaLossWeight) ||
        transparentAlphaLossWeight < 0.0f) {
        throw std::invalid_argument(
            "transparentAlphaLossWeight must be finite and non-negative");
    }
    if (transparentTrainingMasks &&
        appearanceMode != msplat::AppearanceMode::None) {
        throw std::invalid_argument(
            "Transparent training masks cannot be combined with appearance refinement");
    }
    if (inputData.points.count > std::numeric_limits<int>::max())
        throw std::invalid_argument("Dataset contains too many sparse points");
    if (maxGaussians > 0 && inputData.points.count > maxGaussians)
        throw std::invalid_argument("maxGaussians is below the initial Gaussian count");
    if (inputData.cameras.empty() ||
        inputData.cameras.size() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "Dataset camera count is outside the supported range");
    }
    if (numCameras <= 0 ||
        numCameras > static_cast<int>(inputData.cameras.size()))
        throw std::invalid_argument("Dataset must contain training cameras");
    datasetCameraCount = static_cast<int>(inputData.cameras.size());
    if (refineCameraPoses &&
        (poseAnchorCameraIndex < 0 ||
         poseAnchorCameraIndex >= datasetCameraCount)) {
        throw std::invalid_argument(
            "Pose refinement requires a canonical training-camera anchor");
    }
    if (!refineCameraPoses && poseAnchorCameraIndex != -1) {
        throw std::invalid_argument(
            "A pose anchor requires camera pose refinement");
    }
    if (cameraPoseConditioning != msplat::CameraPoseConditioning::Raw &&
        cameraPoseConditioning != msplat::CameraPoseConditioning::CamP) {
        throw std::invalid_argument(
            "Camera pose conditioning mode is not recognized");
    }
    if (!refineCameraPoses &&
        cameraPoseConditioning != msplat::CameraPoseConditioning::Raw) {
        throw std::invalid_argument(
            "CamP conditioning requires camera pose refinement");
    }
    if (appearanceMode != msplat::AppearanceMode::None || refineCameraPoses) {
        if (inputData.cameras.size() > kPhotometricMaxCameras) {
            throw std::invalid_argument(
                "Camera refinement camera count is too large");
        }
        if (inputData.metadata.frameIds.size() != inputData.cameras.size()) {
            throw std::invalid_argument(
                "Camera refinement requires a stable ID for every camera");
        }
        std::unordered_set<std::string> uniqueFrameIds;
        uniqueFrameIds.reserve(inputData.metadata.frameIds.size());
        size_t totalFrameIdBytes = 0;
        for (const std::string& frameId : inputData.metadata.frameIds) {
            if (frameId.empty() ||
                frameId.size() > kPhotometricMaxFrameIdBytes) {
                throw std::invalid_argument(
                    "Camera refinement IDs must be non-empty and bounded");
            }
            if (frameId.size() >
                kPhotometricMaxFrameIdentityBytes - totalFrameIdBytes) {
                throw std::invalid_argument(
                    "Camera refinement IDs exceed the aggregate size limit");
            }
            totalFrameIdBytes += frameId.size();
            if (!uniqueFrameIds.insert(frameId).second) {
                throw std::invalid_argument(
                    "Camera refinement IDs must be unique");
            }
        }
    }
    if (numDownscales < 0 || numDownscales > 30)
        throw std::invalid_argument("numDownscales must be in 0...30");
    if (resolutionSchedule <= 0)
        throw std::invalid_argument("resolutionSchedule must be greater than zero");
    if (shDegree < 0 || shDegree > 4)
        throw std::invalid_argument("shDegree must be in 0...4");
    if (shDegreeInterval <= 0)
        throw std::invalid_argument("shDegreeInterval must be greater than zero");
    if (refineEvery <= 0)
        throw std::invalid_argument("refineEvery must be greater than zero");
    if (warmupLength < 0)
        throw std::invalid_argument("warmupLength must not be negative");
    if (resetAlphaEvery <= 0 ||
        resetAlphaEvery > std::numeric_limits<int>::max() / refineEvery)
        throw std::invalid_argument("resetAlphaEvery * refineEvery is invalid");
    if (!std::isfinite(densifyGradThresh) || densifyGradThresh < 0.0f ||
        !std::isfinite(densifySizeThresh) || densifySizeThresh < 0.0f)
        throw std::invalid_argument("Densification thresholds must be finite and non-negative");
    if (stopScreenSizeAt < 0 || stopDensifyAt < -1)
        throw std::invalid_argument("Densification stop steps are invalid");
    if (!std::isfinite(splitScreenSize) || splitScreenSize < 0.0f)
        throw std::invalid_argument("splitScreenSize must be finite and non-negative");
    if (maxSteps <= 0 || maxSteps > 1000000)
        throw std::invalid_argument("iterations must be in 1...1000000");
    if (stopSplitAt > 1) {
        const auto schedule = msplat::densificationSchedule(
            maxSteps, numCameras, refineEvery, warmupLength,
            resetAlphaEvery, stopSplitAt);
        if (schedule.eventCount == 0) {
            std::cerr << "Warning: no densification opportunities for "
                      << numCameras << " training cameras in " << maxSteps
                      << " iterations. Increase the iteration budget or adjust "
                         "the refinement/reset schedule and densification cutoff; "
                         "set stopDensifyAt=0 for intentional fixed-population training.\n";
        } else {
            std::cout << "Densification schedule: first step " << schedule.firstStep
                      << ", " << schedule.eventCount << " opportunities\n";
        }
    }
    if (bgColor) {
        for (int component = 0; component < 3; ++component) {
            if (!std::isfinite(bgColor[component]) || bgColor[component] < 0.0f ||
                bgColor[component] > 1.0f)
                throw std::invalid_argument("Background components must be finite and in 0...1");
        }
    }

    int64_t numPoints = inputData.points.count;
    if (appearanceMode != msplat::AppearanceMode::None || refineCameraPoses)
        cameraFrameIds = inputData.metadata.frameIds;
    if (refineCameraPoses) {
        cameraBasePoses.reserve(inputData.cameras.size() * 16);
        for (const Camera& camera : inputData.cameras) {
            cameraBasePoses.insert(
                cameraBasePoses.end(), std::begin(camera.camToWorld),
                std::end(camera.camToWorld));
        }
        if (cameraPoseConditioning == msplat::CameraPoseConditioning::CamP) {
            const size_t cameraCount = inputData.cameras.size();
            cameraPosePreconditionerValues.assign(cameraCount * 36, 0.0f);
            cameraPosePreconditionerReady.assign(cameraCount, uint8_t{0});
            for (size_t camera = 0; camera < cameraCount; ++camera) {
                for (size_t axis = 0; axis < 6; ++axis) {
                    cameraPosePreconditionerValues[
                        camera * 36 + axis * 6 + axis] = 1.0f;
                }
            }
            cameraPosePreconditionerReady[
                static_cast<size_t>(poseAnchorCameraIndex)] = 1;
            constexpr size_t kMaximumCampPointPool = 65'536;
            const size_t availablePoints = static_cast<size_t>(numPoints);
            const size_t pooledPoints =
                std::min(availablePoints, kMaximumCampPointPool);
            cameraPosePointPool.reserve(pooledPoints * 3);
            cameraPosePointIds.reserve(pooledPoints);
            const bool hasStablePointIds =
                inputData.metadata.pointSourceIds.size() == availablePoints;
            const std::vector<size_t> selectedPointIndices =
                msplat::detail::selectCampPoseWorldPointPoolIndices(
                    availablePoints,
                    hasStablePointIds
                        ? inputData.metadata.pointSourceIds.data()
                        : nullptr,
                    kMaximumCampPointPool);
            for (size_t pointIndex : selectedPointIndices) {
                const size_t offset = pointIndex * 3;
                cameraPosePointPool.insert(
                    cameraPosePointPool.end(),
                    inputData.points.xyz.begin() + offset,
                    inputData.points.xyz.begin() + offset + 3);
                cameraPosePointIds.push_back(hasStablePointIds
                    ? inputData.metadata.pointSourceIds[pointIndex]
                    : static_cast<uint64_t>(pointIndex));
            }
        }
    }
    scale = inputData.scale;
    memcpy(translation, inputData.translation, sizeof(translation));

    // Means: copy xyz directly to GPU
    means = gpu_empty({numPoints, 3}, DType::Float32);
    memcpy(means.data_ptr(), inputData.points.xyz.data(), numPoints * 3 * sizeof(float));

    // Scales: KD-tree nearest neighbor distances, log'd, repeated 3x
    {
        PointsTensor pt(inputData.points.xyz.data(), numPoints);
        auto sc = pt.scales();  // vector<float> of length numPoints
        scales = gpu_empty({numPoints, 3}, DType::Float32);
        float *sp = scales.data<float>();
        for (int64_t i = 0; i < numPoints; i++) {
            float v = std::log(sc[i]);
            sp[i*3] = sp[i*3+1] = sp[i*3+2] = v;
        }
    }

    // Random quaternions
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        quats = gpu_empty({numPoints, 4}, DType::Float32);
        float *qp = quats.data<float>();
        for (int64_t i = 0; i < numPoints; i++) {
            float u = dist(rng), v = dist(rng), w = dist(rng);
            qp[i*4+0] = std::sqrt(1-u) * std::sin(2*M_PI*v);
            qp[i*4+1] = std::sqrt(1-u) * std::cos(2*M_PI*v);
            qp[i*4+2] = std::sqrt(u) * std::sin(2*M_PI*w);
            qp[i*4+3] = std::sqrt(u) * std::cos(2*M_PI*w);
        }
    }

    // SH features: f_dc = rgb2sh(rgb), f_rest = zeros
    int dimSh = numShBases(shDegree);
    {
        featuresDc = gpu_empty({numPoints, 3}, DType::Float32);
        float *dp = featuresDc.data<float>();
        const uint8_t *rgb = inputData.points.rgb.data();
        for (int64_t i = 0; i < numPoints; i++) {
            for (int c = 0; c < 3; c++)
                dp[i*3+c] = (float)((rgb[i*3+c] / 255.0 - 0.5) / C0);
        }
        featuresRest = gpu_zeros({numPoints, (int64_t)(dimSh - 1), 3}, DType::Float32);
    }

    // Opacities: logit(0.1) = log(0.1/0.9)
    {
        float logit01 = std::log(0.1f / 0.9f);
        opacities = gpu_empty({numPoints, 1}, DType::Float32);
        float *op = opacities.data<float>();
        for (int64_t i = 0; i < numPoints; i++) op[i] = logit01;
    }

    // Background color — default is magenta (high-contrast against typical scenes,
    // makes under-reconstructed regions obvious during training)
    backgroundColor = gpu_empty({3}, DType::Float32);
    static const float defaultBg[3] = {0.6130f, 0.0101f, 0.3984f};
    memcpy(backgroundColor.data_ptr(), bgColor ? bgColor : defaultBg, 3 * sizeof(float));
    setupOptimizers();
}

void Model::setupOptimizers(){
    releaseOptimizers();

    if (means.size(0) <= 0 || means.size(0) > std::numeric_limits<int>::max())
        throw std::runtime_error("Gaussian population is outside the supported range");
    num_active = static_cast<int>(means.size(0));
    buf_capacity = capacityFor(num_active);
    auto allocBuf = [&](MTensor &buf, const MTensor &param) {
        auto shape = param.shape();
        shape[0] = buf_capacity;
        buf = gpu_zeros(shape, DType::Float32);
        memcpy(buf.data_ptr(), param.data_ptr(), param.nbytes());
    };
    allocBuf(means_buf, means);
    allocBuf(scales_buf, scales);
    allocBuf(quats_buf, quats);
    allocBuf(featuresDc_buf, featuresDc);
    allocBuf(featuresRest_buf, featuresRest);
    allocBuf(opacities_buf, opacities);

    static constexpr float lr_init[] = {0.00016f, 0.005f, 0.001f, 0.0025f, 0.000125f, 0.05f};
    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities};
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        auto shape = params[g]->shape();
        shape[0] = buf_capacity;
        adam_exp_avg_buf[g] = gpu_zeros(shape, DType::Float32);
        adam_exp_avg_sq_buf[g] = gpu_zeros(shape, DType::Float32);
        adam_lr[g] = lr_init[g];
    }
    adam_step_count = 0;
    means_lr_init = 0.00016f;
    means_lr_final = 0.0000016f;

    const bool usesRgbGains =
        appearanceMode == msplat::AppearanceMode::RgbGains;
    const int photometricRows = usesRgbGains
        ? datasetCameraCount
        : 1;
    cameraLogGains = gpu_zeros(
        {photometricRows, 3}, DType::Float32);
    if (usesRgbGains) {
        cameraLogGainExpAvg = gpu_zeros(
            {photometricRows, 3}, DType::Float32);
        cameraLogGainExpAvgSq = gpu_zeros(
            {photometricRows, 3}, DType::Float32);
        cameraLogGainStepCounts.assign(
            static_cast<size_t>(datasetCameraCount), 0);
    }

    const bool usesPpisp = appearanceMode == msplat::AppearanceMode::PPISP;
    const int ppispRows = usesPpisp ? datasetCameraCount : 1;
    cameraPpispParameters = gpu_zeros(
        {ppispRows, kPpispParameterCount}, DType::Float32);
    if (usesPpisp) {
        cameraPpispExpAvg = gpu_zeros(
            {ppispRows, kPpispParameterCount}, DType::Float32);
        cameraPpispExpAvgSq = gpu_zeros(
            {ppispRows, kPpispParameterCount}, DType::Float32);
        cameraPpispStepCounts.assign(
            static_cast<size_t>(datasetCameraCount), 0);
    }

    const int poseRows = refineCameraPoses ? datasetCameraCount : 1;
    cameraPoseDeltas = gpu_zeros({poseRows, 6}, DType::Float32);
    if (refineCameraPoses &&
        cameraPoseConditioning == msplat::CameraPoseConditioning::CamP) {
        cameraPosePreconditioners = gpu_empty(
            {poseRows, 36}, DType::Float32);
        memcpy(cameraPosePreconditioners.data_ptr(),
               cameraPosePreconditionerValues.data(),
               cameraPosePreconditionerValues.size() * sizeof(float));
    } else {
        cameraPosePreconditioners.reset();
    }
    if (refineCameraPoses) {
        cameraPoseExpAvg = gpu_zeros({poseRows, 6}, DType::Float32);
        cameraPoseExpAvgSq = gpu_zeros({poseRows, 6}, DType::Float32);
        cameraPoseStepCounts.assign(
            static_cast<size_t>(datasetCameraCount), 0);
    }

    refreshViews();
}

void Model::allocateDensificationScratch() {
    DensificationScratch scratch = makeDensificationScratch(
        buf_capacity, std::max<int64_t>(featuresRest_buf.stride0(), 4));

    densify_split_flag = std::move(scratch.splitFlag);
    densify_dup_flag = std::move(scratch.dupFlag);
    densify_split_prefix = std::move(scratch.splitPrefix);
    densify_dup_prefix = std::move(scratch.dupPrefix);
    densify_keep_flag = std::move(scratch.keepFlag);
    densify_keep_prefix = std::move(scratch.keepPrefix);
    densify_block_totals = std::move(scratch.blockTotals);
    densify_compact_scratch = std::move(scratch.compactScratch);
    densify_random_samples = std::move(scratch.randomSamples);
}

bool Model::hasDensificationScratch() const {
    return densify_split_flag.defined() && densify_dup_flag.defined() &&
        densify_split_prefix.defined() && densify_dup_prefix.defined() &&
        densify_keep_flag.defined() && densify_keep_prefix.defined() &&
        densify_block_totals.defined() && densify_compact_scratch.defined() &&
        densify_random_samples.defined();
}

void Model::resetDensificationScratch() {
    densify_split_flag.reset(); densify_dup_flag.reset();
    densify_split_prefix.reset(); densify_dup_prefix.reset();
    densify_keep_flag.reset(); densify_keep_prefix.reset();
    densify_block_totals.reset(); densify_compact_scratch.reset();
    densify_random_samples.reset();
}

void Model::retireDensificationState() {
    const MTensor* tensors[] = {
        &densify_split_flag, &densify_dup_flag,
        &densify_split_prefix, &densify_dup_prefix,
        &densify_keep_flag, &densify_keep_prefix,
        &densify_block_totals, &densify_compact_scratch,
        &densify_random_samples, &radii, &xysGradNorm, &visCounts, &max2DSize
    };
    if (std::none_of(std::begin(tensors), std::end(tensors),
                     [](const MTensor* tensor) { return tensor->defined(); })) {
        return;
    }

    // The previous step may still reference its stats or scratch through a
    // non-blocking command buffer. Synchronize before releasing those buffers.
    msplat_gpu_sync();
    resetDensificationScratch();
    radii.reset();
    xysGradNorm.reset(); visCounts.reset(); max2DSize.reset();
}

void Model::releaseOptimizers(){
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        adam_exp_avg[g].reset(); adam_exp_avg_sq[g].reset();
        adam_exp_avg_buf[g].reset(); adam_exp_avg_sq_buf[g].reset();
    }
    means_buf.reset(); scales_buf.reset(); quats_buf.reset();
    featuresDc_buf.reset(); featuresRest_buf.reset(); opacities_buf.reset();
    cameraLogGains.reset();
    cameraLogGainExpAvg.reset(); cameraLogGainExpAvgSq.reset();
    cameraLogGainStepCounts.clear();
    cameraPpispParameters.reset();
    cameraPpispExpAvg.reset(); cameraPpispExpAvgSq.reset();
    cameraPpispStepCounts.clear();
    cameraPoseDeltas.reset();
    cameraPoseExpAvg.reset(); cameraPoseExpAvgSq.reset();
    cameraPosePreconditioners.reset();
    cameraPoseStepCounts.clear();
    resetDensificationScratch();
}

void Model::schedulersStep(int step){
    float t = std::clamp((float)step / (float)maxSteps, 0.f, 1.f);
    adam_lr[0] = std::exp(std::log(means_lr_init) * (1.f - t) + std::log(means_lr_final) * t);
}

void Model::refreshViews(){
    means = means_buf.view(num_active);
    scales = scales_buf.view(num_active);
    quats = quats_buf.view(num_active);
    featuresDc = featuresDc_buf.view(num_active);
    featuresRest = featuresRest_buf.view(num_active);
    opacities = opacities_buf.view(num_active);
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        adam_exp_avg[g] = adam_exp_avg_buf[g].view(num_active);
        adam_exp_avg_sq[g] = adam_exp_avg_sq_buf[g].view(num_active);
    }
}

void Model::ensureCameraPosePreconditioner(
    const Camera& camera, size_t canonicalCameraIndex) {
    if (cameraPoseConditioning != msplat::CameraPoseConditioning::CamP)
        return;
    const size_t cameraCount = static_cast<size_t>(datasetCameraCount);
    if (canonicalCameraIndex >= cameraCount ||
        cameraPosePreconditionerReady.size() != cameraCount ||
        cameraPosePreconditionerValues.size() != cameraCount * 36 ||
        cameraFrameIds.size() != cameraCount ||
        cameraPosePointIds.size() * 3 != cameraPosePointPool.size()) {
        throw std::runtime_error(
            "CamP camera-preconditioner state is incomplete");
    }
    if (cameraPosePreconditionerReady[canonicalCameraIndex] != 0)
        return;
    if (!cameraPosePreconditioners.defined() ||
        !cameraPosePreconditioners.isGpu() ||
        cameraPosePreconditioners.dtype() != DType::Float32 ||
        cameraPosePreconditioners.ndim() != 2 ||
        cameraPosePreconditioners.size(0) != datasetCameraCount ||
        cameraPosePreconditioners.size(1) != 36) {
        throw std::runtime_error(
            "CamP GPU preconditioner storage is invalid");
    }

    msplat::detail::CampPoseCameraGeometry geometry;
    for (size_t value = 0; value < 16; ++value)
        geometry.cameraToWorld[value] = camera.camToWorld[value];
    geometry.focalX = camera.fx;
    geometry.focalY = camera.fy;
    geometry.principalX = camera.cx;
    geometry.principalY = camera.cy;
    geometry.width = camera.width;
    geometry.height = camera.height;
    geometry.deterministicKey = stablePoseCameraKey(
        cameraFrameIds[canonicalCameraIndex]);

    const msplat::detail::CampPoseWorldPointPool pointPool{
        cameraPosePointPool.data(), cameraPosePointIds.data(),
        cameraPosePointPool.size() / 3};
    const auto built = msplat::detail::buildCampPosePreconditioner(
        geometry, pointPool);
    std::array<float, 36> values{};
    for (size_t value = 0; value < values.size(); ++value) {
        if (!std::isfinite(built.preconditioner[value]) ||
            std::abs(built.preconditioner[value]) >
                std::numeric_limits<float>::max()) {
            throw std::runtime_error(
                "CamP preconditioner exceeds Float32 range");
        }
        values[value] = static_cast<float>(built.preconditioner[value]);
    }

    // Each row is initialized once. Synchronize before publishing CPU writes
    // into the shared Metal buffer so an in-flight step cannot overlap them.
    msplat_gpu_sync();
    const size_t offset = canonicalCameraIndex * 36;
    std::copy(values.begin(), values.end(),
              cameraPosePreconditionerValues.begin() + offset);
    memcpy(cameraPosePreconditioners.data<float>() + offset,
           values.data(), values.size() * sizeof(float));
    cameraPosePreconditionerReady[canonicalCameraIndex] = 1;
}

size_t Model::estimatedGpuBytes() const {
    size_t bytes = 0;
    const MTensor* tensors[] = {
        &means_buf, &scales_buf, &quats_buf, &featuresDc_buf, &featuresRest_buf, &opacities_buf,
        &densify_split_flag, &densify_dup_flag, &densify_split_prefix, &densify_dup_prefix,
        &densify_keep_flag, &densify_keep_prefix, &densify_block_totals,
        &densify_compact_scratch, &densify_random_samples,
        // `radii` is a non-owning alias of the shared render cache's radii_out
        // buffer, so counting it here would report the same allocation twice.
        &xysGradNorm, &visCounts, &max2DSize, &backgroundColor,
        &cameraLogGains, &cameraLogGainExpAvg, &cameraLogGainExpAvgSq,
        &cameraPpispParameters, &cameraPpispExpAvg, &cameraPpispExpAvgSq,
        &cameraPoseDeltas, &cameraPoseExpAvg, &cameraPoseExpAvgSq,
        &cameraPosePreconditioners
    };
    for (const MTensor* tensor : tensors) {
        if (tensor->defined()) bytes += tensor->nbytes();
    }
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        if (adam_exp_avg_buf[g].defined()) bytes += adam_exp_avg_buf[g].nbytes();
        if (adam_exp_avg_sq_buf[g].defined()) bytes += adam_exp_avg_sq_buf[g].nbytes();
    }
    return bytes;
}

uint32_t Model::poseRefinementStateCount() const {
    return msplat::detail::poseRefinementStateCount(
        refineCameraPoses, datasetCameraCount);
}

ModelPoseRefinementState Model::poseRefinementState(
    uint32_t canonicalCameraIndex) const {
    const uint32_t count = poseRefinementStateCount();
    if (canonicalCameraIndex >= count) {
        throw std::invalid_argument(
            "Pose-refinement camera index is out of range");
    }
    if (!cameraPoseDeltas.defined() || !cameraPoseDeltas.isGpu() ||
        cameraPoseDeltas.dtype() != DType::Float32 ||
        cameraPoseDeltas.ndim() != 2 ||
        cameraPoseDeltas.size(0) != datasetCameraCount ||
        cameraPoseDeltas.size(1) != 6 ||
        cameraPoseStepCounts.size() != static_cast<size_t>(datasetCameraCount) ||
        cameraBasePoses.size() !=
            static_cast<size_t>(datasetCameraCount) * 16 ||
        cameraFrameIds.size() != static_cast<size_t>(datasetCameraCount)) {
        throw std::runtime_error(
            "Pose-refinement query state is inconsistent");
    }

    const size_t camera = static_cast<size_t>(canonicalCameraIndex);
    const size_t deltaOffset = camera * 6;
    const size_t poseOffset = camera * 16;
    const float* deltas = cameraPoseDeltas.data<float>() + deltaOffset;
    float translationNorm2 = 0.0f;
    float rotationNorm2 = 0.0f;
    for (int component = 0; component < 6; ++component) {
        if (!std::isfinite(deltas[component])) {
            throw std::runtime_error(
                "Pose-refinement query contains a non-finite delta");
        }
        if (component < 3)
            translationNorm2 += deltas[component] * deltas[component];
        else
            rotationNorm2 += deltas[component] * deltas[component];
    }
    if (translationNorm2 >
            (kPoseMaxTranslation + 1.0e-6f) *
                (kPoseMaxTranslation + 1.0e-6f) ||
        rotationNorm2 >
            (kPoseMaxRotation + 1.0e-6f) *
                (kPoseMaxRotation + 1.0e-6f)) {
        throw std::runtime_error(
            "Pose-refinement query delta is outside its bounds");
    }

    ModelPoseRefinementState state;
    state.anchor = canonicalCameraIndex ==
        static_cast<uint32_t>(poseAnchorCameraIndex);
    state.optimizerStepCount = cameraPoseStepCounts[camera];
    if (state.anchor) {
        if (state.optimizerStepCount != 0u) {
            throw std::runtime_error(
                "Pose-refinement anchor step count is not zero");
        }
        for (int component = 0; component < 6; ++component) {
            if (deltas[component] != 0.0f) {
                throw std::runtime_error(
                    "Pose-refinement anchor delta is not zero");
            }
        }
    }
    state.geometry = msplat::detail::makePoseRefinementGeometry(
        cameraBasePoses.data() + poseOffset, deltas, scale, translation);
    const std::string& frameId = cameraFrameIds[camera];
    state.frameId = frameId.data();
    state.frameIdLength = frameId.size();
    return state;
}

void Model::ensureCapacity(int needed){
    if (!hasDensificationScratch())
        throw std::logic_error(
            "Densification capacity changed without live scratch buffers");
    if (needed <= buf_capacity) return;
    if (maxGaussians > 0 && needed > maxGaussians)
        throw std::runtime_error("Gaussian population exceeds maxGaussians");
    int new_cap = std::max(needed, capacityWithSlack(buf_capacity));
    if (maxGaussians > 0) new_cap = std::min(new_cap, maxGaussians);

    struct ResizeTask {
        MTensor* tensor;
        MTensor* activeView;
        std::vector<int64_t> shape;
        bool preserveActive;
    };
    std::vector<ResizeTask> tasks;
    auto addLeadingCapacityTask = [&](
        MTensor &tensor, bool preserveActive, MTensor* activeView = nullptr) {
        auto shape = tensor.shape();
        shape[0] = new_cap;
        tasks.push_back(
            {&tensor, activeView, std::move(shape), preserveActive});
    };

    MTensor* parameterBuffers[] = {
        &means_buf, &scales_buf, &quats_buf,
        &featuresDc_buf, &featuresRest_buf, &opacities_buf
    };
    MTensor* parameterViews[] = {
        &means, &scales, &quats,
        &featuresDc, &featuresRest, &opacities
    };
    for (size_t index = 0; index < std::size(parameterBuffers); ++index) {
        addLeadingCapacityTask(
            *parameterBuffers[index], true, parameterViews[index]);
    }
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        addLeadingCapacityTask(
            adam_exp_avg_buf[g], true, &adam_exp_avg[g]);
        addLeadingCapacityTask(
            adam_exp_avg_sq_buf[g], true, &adam_exp_avg_sq[g]);
    }

    // These four already hold this step's classification when the grow happens
    // — the population is only known after the classify pass — so their
    // contents have to survive it. keep_* are written later and do not.
    MTensor* preparedDensifyBuffers[] = {
        &densify_split_flag, &densify_dup_flag,
        &densify_split_prefix, &densify_dup_prefix
    };
    for (MTensor* tensor : preparedDensifyBuffers) addLeadingCapacityTask(*tensor, true);
    addLeadingCapacityTask(densify_keep_flag, false);
    addLeadingCapacityTask(densify_keep_prefix, false);

    int max_blocks = static_cast<int>(
        (static_cast<int64_t>(new_cap) + 1023) / 1024);
    int64_t fr_stride = featuresRest_buf.stride0();
    int64_t compact_stride = std::max<int64_t>(fr_stride, 4);
    tasks.push_back(
        {&densify_block_totals, nullptr, {max_blocks}, false});
    tasks.push_back({&densify_compact_scratch, nullptr,
                     {static_cast<int64_t>(new_cap) * compact_stride}, false});
    if (!msplat_densify_uses_gpu_random()) {
        tasks.push_back(
            {&densify_random_samples, nullptr, {new_cap, 3}, false});
    }

    // Replacing the largest allocations first minimizes the final transient:
    // by the time most new buffers exist, only small old buffers remain.
    std::sort(tasks.begin(), tasks.end(), [](const ResizeTask &lhs, const ResizeTask &rhs) {
        return lhs.tensor->nbytes() > rhs.tensor->nbytes();
    });
    for (const ResizeTask &task : tasks) {
        // Densification writes every appended row and every consumed scratch
        // element before reading it. Preserve only the live prefix instead of
        // clearing capacity slack that may never become active.
        MTensor replacement = gpu_empty(task.shape, task.tensor->dtype());
        if (task.preserveActive) {
            size_t copy_bytes = static_cast<size_t>(num_active) *
                task.tensor->stride0() * task.tensor->elementSize();
            memcpy(replacement.data_ptr(), task.tensor->data_ptr(), copy_bytes);
        }
        // Build the non-owning view before releasing its previous owner. If
        // allocation fails later in the loop, every already-replaced owner
        // still has a valid active view and the model remains destructible.
        MTensor replacementView;
        if (task.activeView)
            replacementView = replacement.view(num_active);
        *task.tensor = std::move(replacement);
        if (task.activeView)
            *task.activeView = std::move(replacementView);
    }

    buf_capacity = new_cap;
}

int Model::capacityFor(int needed) const {
    if (needed <= 0)
        throw std::invalid_argument("Gaussian capacity must be positive");
    if (maxGaussians > 0) {
        if (needed > maxGaussians)
            throw std::runtime_error("Gaussian population exceeds maxGaussians");
        return std::min(capacityWithSlack(needed), maxGaussians);
    }
    return capacityWithSlack(needed);
}

int Model::getDownscaleFactor(int step) {
    int remaining = numDownscales - step / resolutionSchedule;
    return 1 << std::max(remaining, 0);
}

void Model::afterTrain(int step){
    if (!radii.defined()) return;

    if (step % refineEvery == 0 && step > warmupLength){
        int resetInterval = resetAlphaEvery * refineEvery;
        bool doDensification = msplat::isDensificationStep(
            step, numCameras, refineEvery, warmupLength, resetInterval, stopSplitAt);

        if (doDensification){
            int numPointsBefore = num_active;
            float half_max_dim = 0.5f * static_cast<float>((std::max)(lastWidth, lastHeight));
            int check_screen = (step < stopScreenSizeAt) ? 1 : 0;
            bool checkHuge = step > refineEvery * resetAlphaEvery;

            // Classify first, so the grow below asks for the population that
            // will actually be written rather than the 3*N that never is.
            int numSplits = 0;
            int numDups = 0;
            msplat_prepare_densify(
                num_active, maxGaussians,
                densifyGradThresh, densifySizeThresh, splitScreenSize, check_screen,
                xysGradNorm, visCounts, max2DSize, half_max_dim,
                scales_buf,
                densify_split_flag, densify_dup_flag,
                densify_split_prefix, densify_dup_prefix,
                densify_block_totals,
                numSplits, numDups
            );

            int64_t population64 = static_cast<int64_t>(num_active) +
                2LL * numSplits + numDups;
            if (population64 > std::numeric_limits<int>::max()) {
                throw std::runtime_error("Densified population exceeds supported size");
            }
            int population = static_cast<int>(population64);
            ensureCapacity(population);

            int fr_stride = (int)featuresRest_buf.stride0();
            int densifiedCount = msplat_densify(
                num_active, population,
                0.1f, 0.5f, 0.15f, check_screen, checkHuge ? 1 : 0,
                max2DSize,
                means_buf, scales_buf, quats_buf,
                featuresDc_buf, featuresRest_buf, opacities_buf, fr_stride,
                adam_exp_avg_buf, adam_exp_avg_sq_buf,
                densify_split_flag, densify_dup_flag,
                densify_split_prefix, densify_dup_prefix,
                densify_keep_flag, densify_keep_prefix,
                densify_block_totals, densify_compact_scratch,
                densify_random_samples, static_cast<uint32_t>(step)
            );
            if (densifiedCount < 0 || densifiedCount > population ||
                (maxGaussians > 0 && densifiedCount > maxGaussians)) {
                throw std::runtime_error("Densification returned an invalid Gaussian count");
            }
            num_active = densifiedCount;

            refreshViews();
            std::cout << "Densified: " << numPointsBefore << " -> " << num_active << " gaussians" << std::endl;
        }

        if (step < stopSplitAt && step % resetInterval == refineEvery){
            constexpr float resetLogit = -1.3862943611198906f;
            msplat_reset_opacity_state(
                opacities, adam_exp_avg[5], adam_exp_avg_sq[5],
                resetLogit);
            fprintf(stderr, "Opacity reset scheduled at step %d\n", step);
        }

        xysGradNorm.reset();
        visCounts.reset();
        max2DSize.reset();
    }
}

void Model::save(const std::string &filename, int step) {
    std::string ext = fs::path(filename).extension().string();
    if (ext == ".splat")
        saveSplat(filename);
    else if (ext == ".spz")
        saveSpz(filename);
    else
        savePly(filename, step);
    fprintf(stderr, "Saved %s\n", filename.c_str());
}

void Model::savePly(const std::string &filename, int step){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs};
    saveGaussianPly(filename, p, step);
}

void Model::saveSplat(const std::string &filename){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs};
    saveGaussianSplat(filename, p);
}

void Model::saveSpz(const std::string &filename){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs};
    saveGaussianSpz(filename, p);
}

int Model::loadPly(const std::string &filename){
    if (maxGaussians > 0) {
        std::ifstream header(filename, std::ios::binary);
        if (!header.is_open())
            throw std::runtime_error("Cannot open PLY file: " + filename);

        const std::string vertexPrefix = "element vertex ";
        std::string line;
        while (std::getline(header, line)) {
            while (!line.empty() &&
                   (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
            if (line == "end_header") break;
            if (line.rfind(vertexPrefix, 0) != 0) continue;
            size_t consumed = 0;
            const long long declaredCount = std::stoll(
                line.substr(vertexPrefix.size()), &consumed);
            if (consumed != line.size() - vertexPrefix.size() || declaredCount <= 0)
                throw std::runtime_error("PLY has an invalid vertex count");
            if (declaredCount > maxGaussians)
                throw std::runtime_error("PLY Gaussian population exceeds maxGaussians");
            break;
        }
    }

    auto g = loadGaussianPly(filename, scale, translation, keepCrs);
    if (g.means.size(0) <= 0 || g.means.size(0) > std::numeric_limits<int>::max())
        throw std::runtime_error("PLY Gaussian population is outside the supported range");
    if (maxGaussians > 0 && g.means.size(0) > maxGaussians)
        throw std::runtime_error("PLY Gaussian population exceeds maxGaussians");
    means = g.means;
    scales = g.scales;
    quats = g.quats;
    featuresDc = g.featuresDc;
    featuresRest = g.featuresRest;
    opacities = g.opacities;
    setupOptimizers();
    return g.step;
}

// ── Checkpoint save/load ────────────────────────────────────────────────────

static constexpr uint32_t CKPT_MAGIC = 0x4C50534D; // "MSPL"
static constexpr uint32_t CKPT_VERSION = 5;
static constexpr uint32_t CKPT_MIN_VERSION = 1;
static constexpr uint32_t CKPT_MAX_CAMERAS =
    static_cast<uint32_t>(kPhotometricMaxCameras);
static constexpr uint32_t CKPT_MAX_FRAME_ID_BYTES =
    static_cast<uint32_t>(kPhotometricMaxFrameIdBytes);
static constexpr uint64_t CKPT_MAX_FRAME_ID_TOTAL_BYTES =
    kPhotometricMaxFrameIdentityBytes;

static_assert(sizeof(float) == 4, "Checkpoint format requires 32-bit floats");

static void writeTensor(std::ofstream &f, MTensor &t) {
    uint32_t ndim = t.ndim();
    f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (int i = 0; i < (int)ndim; i++) {
        int64_t s = t.size(i);
        f.write(reinterpret_cast<const char*>(&s), sizeof(s));
    }
    uint64_t bytes = t.nbytes();
    f.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    f.write(reinterpret_cast<const char*>(t.data_ptr()), bytes);
}

static void writeFloatTensor(
    std::ofstream &f, const std::vector<float>& values,
    const std::array<int64_t, 2>& shape
) {
    if (shape[0] <= 0 || shape[1] <= 0 ||
        static_cast<uint64_t>(shape[0]) >
            std::numeric_limits<uint64_t>::max() /
                static_cast<uint64_t>(shape[1]) ||
        static_cast<uint64_t>(shape[0]) *
                static_cast<uint64_t>(shape[1]) != values.size()) {
        throw std::runtime_error("Checkpoint Float32 tensor shape is invalid");
    }
    const uint32_t ndim = 2;
    f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    f.write(reinterpret_cast<const char*>(shape.data()), sizeof(shape));
    const uint64_t bytes = values.size() * sizeof(float);
    f.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    if (bytes > 0) {
        f.write(reinterpret_cast<const char*>(values.data()),
                static_cast<std::streamsize>(bytes));
    }
}

static void writeCheckpointString(std::ofstream &f, const std::string &value) {
    if (value.size() > CKPT_MAX_FRAME_ID_BYTES)
        throw std::runtime_error("Checkpoint frame ID is too long");
    const uint32_t length = static_cast<uint32_t>(value.size());
    f.write(reinterpret_cast<const char*>(&length), sizeof(length));
    if (length > 0)
        f.write(value.data(), static_cast<std::streamsize>(length));
}

namespace {

[[noreturn]] void checkpointError(const std::string &message) {
    throw std::runtime_error("Invalid msplat checkpoint: " + message);
}

uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const std::string &field) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
        checkpointError(field + " size overflows");
    return lhs * rhs;
}

uint64_t tensorByteCount(const std::vector<int64_t> &shape,
                         const std::string &name) {
    uint64_t elements = 1;
    for (int64_t dimension : shape) {
        if (dimension < 0)
            checkpointError(name + " has a negative dimension");
        elements = checkedMultiply(elements, static_cast<uint64_t>(dimension), name);
    }
    return checkedMultiply(elements, sizeof(float), name);
}

class CheckpointReader {
public:
    explicit CheckpointReader(const std::string &filename)
        : stream(filename, std::ios::binary | std::ios::ate) {
        if (!stream.is_open())
            throw std::runtime_error("Cannot open checkpoint file: " + filename);

        const std::streampos end = stream.tellg();
        if (end == std::streampos(-1))
            throw std::runtime_error("Cannot determine checkpoint file size: " + filename);
        const std::streamoff endOffset = static_cast<std::streamoff>(end);
        if (endOffset < 0)
            throw std::runtime_error("Invalid checkpoint file size: " + filename);

        totalBytes = static_cast<uint64_t>(endOffset);
        stream.seekg(0, std::ios::beg);
        if (!stream)
            throw std::runtime_error("Cannot seek checkpoint file: " + filename);
    }

    template <typename T>
    T scalar(const std::string &field) {
        T value{};
        bytes(&value, sizeof(value), field);
        return value;
    }

    void bytes(void *destination, uint64_t byteCount, const std::string &field) {
        requireAvailable(byteCount, field);
        readInto(destination, byteCount, field);
        cursor += byteCount;
    }

    void skip(uint64_t byteCount, const std::string &field) {
        requireAvailable(byteCount, field);
        if (byteCount > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()))
            checkpointError(field + " is too large to seek");
        stream.seekg(static_cast<std::streamoff>(byteCount), std::ios::cur);
        if (!stream)
            checkpointError("could not seek past " + field);
        cursor += byteCount;
    }

    void readAt(uint64_t offset, void *destination, uint64_t byteCount,
                const std::string &field) {
        if (offset > totalBytes || byteCount > totalBytes - offset)
            checkpointError(field + " extends past the end of the file");
        if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()))
            checkpointError(field + " offset is too large to seek");

        stream.clear();
        stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!stream)
            checkpointError("could not seek to " + field);
        readInto(destination, byteCount, field);
    }

    uint64_t offset() const { return cursor; }
    uint64_t remaining() const { return totalBytes - cursor; }

    void requireFullyConsumed() const {
        if (cursor != totalBytes)
            checkpointError("unexpected trailing data");
    }

private:
    void requireAvailable(uint64_t byteCount, const std::string &field) const {
        if (cursor > totalBytes || byteCount > totalBytes - cursor)
            checkpointError("truncated while reading " + field);
    }

    void readInto(void *destination, uint64_t byteCount, const std::string &field) {
        auto *output = static_cast<char*>(destination);
        uint64_t remaining = byteCount;
        constexpr uint64_t maxChunk = 1ULL << 30;
        while (remaining > 0) {
            const uint64_t chunk = std::min(remaining, maxChunk);
            stream.read(output, static_cast<std::streamsize>(chunk));
            if (stream.gcount() != static_cast<std::streamsize>(chunk))
                checkpointError("truncated while reading " + field);
            output += static_cast<size_t>(chunk);
            remaining -= chunk;
        }
    }

    std::ifstream stream;
    uint64_t totalBytes = 0;
    uint64_t cursor = 0;
};

struct CheckpointTensorRecord {
    std::vector<int64_t> shape;
    uint64_t dataOffset = 0;
    uint64_t byteCount = 0;
};

CheckpointTensorRecord readTensorRecord(CheckpointReader &reader,
                                        const std::vector<int64_t> &expectedShape,
                                        const std::string &name) {
    const uint32_t ndim = reader.scalar<uint32_t>(name + ".ndim");
    if (ndim != expectedShape.size())
        checkpointError(name + " has the wrong rank");

    for (uint32_t i = 0; i < ndim; ++i) {
        const int64_t dimension = reader.scalar<int64_t>(name + ".shape");
        if (dimension != expectedShape[i])
            checkpointError(name + " has an incompatible shape");
    }

    // Checkpoint tensors are Float32 without a separate dtype tag. An
    // exact shape-derived byte count is therefore also its dtype validation.
    const uint64_t expectedBytes = tensorByteCount(expectedShape, name);
    const uint64_t declaredBytes = reader.scalar<uint64_t>(name + ".bytes");
    if (declaredBytes != expectedBytes)
        checkpointError(name + " byte count does not match its Float32 shape");

    CheckpointTensorRecord record;
    record.shape = expectedShape;
    record.dataOffset = reader.offset();
    record.byteCount = declaredBytes;
    reader.skip(declaredBytes, name + ".data");
    return record;
}

struct ParsedCheckpoint {
    uint32_t version = 0;
    uint32_t step = 0;
    uint32_t numPoints = 0;
    uint32_t shDegree = 0;
    uint32_t adamSteps = 0;
    std::array<float, Model::N_ADAM_GROUPS> adamLearningRates{};
    float meansLearningRateInitial = 0.0f;
    float meansLearningRateFinal = 0.0f;
    std::array<CheckpointTensorRecord, Model::N_ADAM_GROUPS> parameters;
    std::array<CheckpointTensorRecord, Model::N_ADAM_GROUPS> adamExpAvg;
    std::array<CheckpointTensorRecord, Model::N_ADAM_GROUPS> adamExpAvgSq;
    bool photometricEnabled = false;
    uint32_t cameraCount = 0;
    std::vector<std::string> cameraFrameIds;
    std::vector<uint32_t> cameraStepCounts;
    std::array<CheckpointTensorRecord, 3> photometricTensors;
    msplat::AppearanceMode appearanceMode = msplat::AppearanceMode::None;
    uint32_t ppispCameraCount = 0;
    std::vector<std::string> ppispFrameIds;
    std::vector<uint32_t> ppispStepCounts;
    std::array<CheckpointTensorRecord, 3> ppispTensors;
    bool poseEnabled = false;
    uint32_t poseCameraCount = 0;
    uint32_t poseAnchorCameraIndex = 0;
    std::vector<std::string> poseFrameIds;
    std::vector<uint32_t> poseStepCounts;
    CheckpointTensorRecord poseBasePoses;
    msplat::CameraPoseConditioning poseConditioning =
        msplat::CameraPoseConditioning::Raw;
    std::vector<uint8_t> posePreconditionerReady;
    CheckpointTensorRecord posePreconditioners;
    std::array<CheckpointTensorRecord, 3> poseTensors;
};

ParsedCheckpoint parseCheckpointMetadata(CheckpointReader &reader) {
    const uint32_t magic = reader.scalar<uint32_t>("magic");
    const uint32_t version = reader.scalar<uint32_t>("version");
    if (magic != CKPT_MAGIC)
        throw std::runtime_error("Not a valid msplat checkpoint file");
    if (version < CKPT_MIN_VERSION || version > CKPT_VERSION)
        throw std::runtime_error("Unsupported checkpoint version: " +
                                 std::to_string(version));

    ParsedCheckpoint checkpoint;
    checkpoint.version = version;
    checkpoint.step = reader.scalar<uint32_t>("step");
    checkpoint.numPoints = reader.scalar<uint32_t>("num_points");
    checkpoint.shDegree = reader.scalar<uint32_t>("sh_degree");
    checkpoint.adamSteps = reader.scalar<uint32_t>("adam_steps");

    if (checkpoint.step >= static_cast<uint32_t>(std::numeric_limits<int>::max()))
        checkpointError("step exceeds the supported range");
    if (checkpoint.numPoints == 0 ||
        checkpoint.numPoints > static_cast<uint32_t>(std::numeric_limits<int>::max()))
        checkpointError("Gaussian count is outside the supported range");
    if (checkpoint.shDegree > 4)
        checkpointError("SH degree is outside the supported range");
    if (checkpoint.adamSteps >= static_cast<uint32_t>(std::numeric_limits<int>::max()))
        checkpointError("Adam step count exceeds the supported range");

    reader.bytes(checkpoint.adamLearningRates.data(),
                 sizeof(checkpoint.adamLearningRates), "Adam learning rates");
    checkpoint.meansLearningRateInitial = reader.scalar<float>("initial means learning rate");
    checkpoint.meansLearningRateFinal = reader.scalar<float>("final means learning rate");

    for (float rate : checkpoint.adamLearningRates) {
        if (!std::isfinite(rate) || rate < 0.0f)
            checkpointError("Adam learning rates must be finite and non-negative");
    }
    if (!std::isfinite(checkpoint.meansLearningRateInitial) ||
        checkpoint.meansLearningRateInitial <= 0.0f ||
        !std::isfinite(checkpoint.meansLearningRateFinal) ||
        checkpoint.meansLearningRateFinal <= 0.0f)
        checkpointError("means learning rates must be finite and positive");

    const int64_t n = static_cast<int64_t>(checkpoint.numPoints);
    const int64_t restBases = numShBases(static_cast<int>(checkpoint.shDegree)) - 1;
    const std::array<std::vector<int64_t>, Model::N_ADAM_GROUPS> expectedShapes = {{
        {n, 3},
        {n, 3},
        {n, 4},
        {n, 3},
        {n, restBases, 3},
        {n, 1},
    }};
    static constexpr const char *parameterNames[Model::N_ADAM_GROUPS] = {
        "means", "scales", "quats", "features_dc", "features_rest", "opacities"
    };

    for (int group = 0; group < Model::N_ADAM_GROUPS; ++group)
        checkpoint.parameters[group] = readTensorRecord(
            reader, expectedShapes[group], parameterNames[group]);
    for (int group = 0; group < Model::N_ADAM_GROUPS; ++group)
        checkpoint.adamExpAvg[group] = readTensorRecord(
            reader, expectedShapes[group],
            std::string("adam_exp_avg.") + parameterNames[group]);
    for (int group = 0; group < Model::N_ADAM_GROUPS; ++group)
        checkpoint.adamExpAvgSq[group] = readTensorRecord(
            reader, expectedShapes[group],
            std::string("adam_exp_avg_sq.") + parameterNames[group]);

    if (version >= 2) {
        const uint32_t photometricEnabled =
            reader.scalar<uint32_t>("photometric enabled");
        if (photometricEnabled > 1)
            checkpointError("photometric enabled flag is invalid");
        checkpoint.photometricEnabled = photometricEnabled != 0;
        if (checkpoint.photometricEnabled) {
            checkpoint.cameraCount =
                reader.scalar<uint32_t>("photometric camera count");
            if (checkpoint.cameraCount == 0 ||
                checkpoint.cameraCount > CKPT_MAX_CAMERAS ||
                checkpoint.cameraCount >
                    static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                checkpointError(
                    "photometric camera count is outside the supported range");
            }

            // Reject a tiny file with an attacker-sized count before reserving
            // per-camera containers. A complete enabled payload needs at least
            // one ID byte, its length/count fields, three tensor records, and
            // three Float32 RGB tensor payloads per camera.
            constexpr uint64_t minimumTensorRecordBytes =
                sizeof(uint32_t) + 2 * sizeof(int64_t) + sizeof(uint64_t);
            constexpr uint64_t minimumPerCameraBytes =
                sizeof(uint32_t) + 1 + sizeof(uint32_t) +
                3 * 3 * sizeof(float);
            const uint64_t minimumRemainingBytes =
                checkedMultiply(checkpoint.cameraCount,
                                minimumPerCameraBytes,
                                "photometric checkpoint minimum") +
                3 * minimumTensorRecordBytes;
            if (reader.remaining() < minimumRemainingBytes)
                checkpointError("truncated photometric checkpoint payload");

            checkpoint.cameraFrameIds.reserve(checkpoint.cameraCount);
            std::unordered_set<std::string> uniqueFrameIds;
            uniqueFrameIds.reserve(checkpoint.cameraCount);
            uint64_t totalFrameIdBytes = 0;
            for (uint32_t camera = 0; camera < checkpoint.cameraCount; ++camera) {
                const uint32_t length =
                    reader.scalar<uint32_t>("photometric frame ID length");
                if (length == 0 || length > CKPT_MAX_FRAME_ID_BYTES)
                    checkpointError("photometric frame ID length is invalid");
                if (length >
                    CKPT_MAX_FRAME_ID_TOTAL_BYTES - totalFrameIdBytes) {
                    checkpointError(
                        "photometric frame IDs exceed the aggregate size limit");
                }
                totalFrameIdBytes += length;
                std::string frameId(length, '\0');
                reader.bytes(frameId.data(), length, "photometric frame ID");
                if (!uniqueFrameIds.insert(frameId).second)
                    checkpointError("photometric frame IDs are not unique");
                checkpoint.cameraFrameIds.push_back(std::move(frameId));
            }

            checkpoint.cameraStepCounts.resize(checkpoint.cameraCount);
            reader.bytes(
                checkpoint.cameraStepCounts.data(),
                checkedMultiply(checkpoint.cameraCount, sizeof(uint32_t),
                                "photometric step counts"),
                "photometric step counts");
            uint64_t totalPhotometricSteps = 0;
            for (uint32_t count : checkpoint.cameraStepCounts) {
                if (count > checkpoint.adamSteps ||
                    totalPhotometricSteps >
                        static_cast<uint64_t>(checkpoint.adamSteps - count)) {
                    checkpointError(
                        "photometric step counts are inconsistent with Adam state");
                }
                totalPhotometricSteps += count;
            }

            const std::vector<int64_t> photometricShape = {
                static_cast<int64_t>(checkpoint.cameraCount), 3};
            static constexpr const char* photometricNames[3] = {
                "photometric.log_rgb_gains",
                "photometric.adam_exp_avg",
                "photometric.adam_exp_avg_sq",
            };
            for (size_t tensor = 0; tensor < 3; ++tensor) {
                checkpoint.photometricTensors[tensor] = readTensorRecord(
                    reader, photometricShape, photometricNames[tensor]);
            }
        }
        checkpoint.appearanceMode = checkpoint.photometricEnabled
            ? msplat::AppearanceMode::RgbGains
            : msplat::AppearanceMode::None;
    }

    if (version >= 3) {
        const uint32_t poseEnabled =
            reader.scalar<uint32_t>("pose enabled");
        if (poseEnabled > 1)
            checkpointError("pose enabled flag is invalid");
        checkpoint.poseEnabled = poseEnabled != 0;
        if (version == 4 && !checkpoint.poseEnabled) {
            checkpointError(
                "version 4 checkpoints require enabled CamP pose refinement");
        }
        if (checkpoint.poseEnabled) {
            checkpoint.poseCameraCount =
                reader.scalar<uint32_t>("pose camera count");
            if (checkpoint.poseCameraCount == 0 ||
                checkpoint.poseCameraCount > CKPT_MAX_CAMERAS ||
                checkpoint.poseCameraCount >
                    static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                checkpointError(
                    "pose camera count is outside the supported range");
            }
            checkpoint.poseAnchorCameraIndex =
                reader.scalar<uint32_t>("pose anchor index");
            if (checkpoint.poseAnchorCameraIndex >=
                checkpoint.poseCameraCount) {
                checkpointError("pose anchor index is out of range");
            }

            constexpr uint64_t minimumTensorRecordBytes =
                sizeof(uint32_t) + 2 * sizeof(int64_t) + sizeof(uint64_t);
            constexpr uint64_t minimumPerCameraBytes =
                sizeof(uint32_t) + 1 + sizeof(uint32_t) +
                (16 + 3 * 6) * sizeof(float);
            uint64_t minimumRemainingBytes =
                checkedMultiply(checkpoint.poseCameraCount,
                                minimumPerCameraBytes,
                                "pose checkpoint minimum") +
                4 * minimumTensorRecordBytes;
            if (version == 4) {
                minimumRemainingBytes += sizeof(uint32_t) +
                    checkedMultiply(
                        checkpoint.poseCameraCount,
                        sizeof(uint8_t) + 36 * sizeof(float),
                        "pose conditioning checkpoint minimum") +
                    minimumTensorRecordBytes;
            } else if (version >= 5) {
                minimumRemainingBytes += sizeof(uint32_t);
            }
            if (reader.remaining() < minimumRemainingBytes)
                checkpointError("truncated pose checkpoint payload");

            checkpoint.poseFrameIds.reserve(checkpoint.poseCameraCount);
            std::unordered_set<std::string> uniqueFrameIds;
            uniqueFrameIds.reserve(checkpoint.poseCameraCount);
            uint64_t totalFrameIdBytes = 0;
            for (uint32_t camera = 0;
                 camera < checkpoint.poseCameraCount; ++camera) {
                const uint32_t length =
                    reader.scalar<uint32_t>("pose frame ID length");
                if (length == 0 || length > CKPT_MAX_FRAME_ID_BYTES)
                    checkpointError("pose frame ID length is invalid");
                if (length >
                    CKPT_MAX_FRAME_ID_TOTAL_BYTES - totalFrameIdBytes) {
                    checkpointError(
                        "pose frame IDs exceed the aggregate size limit");
                }
                totalFrameIdBytes += length;
                std::string frameId(length, '\0');
                reader.bytes(frameId.data(), length, "pose frame ID");
                if (!uniqueFrameIds.insert(frameId).second)
                    checkpointError("pose frame IDs are not unique");
                checkpoint.poseFrameIds.push_back(std::move(frameId));
            }
            if (checkpoint.photometricEnabled &&
                (checkpoint.cameraCount != checkpoint.poseCameraCount ||
                 checkpoint.cameraFrameIds != checkpoint.poseFrameIds)) {
                checkpointError(
                    "photometric and pose camera identities do not match");
            }

            checkpoint.poseStepCounts.resize(checkpoint.poseCameraCount);
            reader.bytes(
                checkpoint.poseStepCounts.data(),
                checkedMultiply(checkpoint.poseCameraCount,
                                sizeof(uint32_t), "pose step counts"),
                "pose step counts");
            uint64_t totalPoseSteps = 0;
            for (uint32_t camera = 0;
                 camera < checkpoint.poseCameraCount; ++camera) {
                const uint32_t count = checkpoint.poseStepCounts[camera];
                if (count > checkpoint.adamSteps) {
                    checkpointError(
                        "pose step count exceeds the global Adam state");
                }
                if (totalPoseSteps >
                    static_cast<uint64_t>(checkpoint.adamSteps - count)) {
                    checkpointError(
                        "pose step counts are inconsistent with Adam state");
                }
                totalPoseSteps += count;
            }
            if (checkpoint.poseStepCounts[
                    checkpoint.poseAnchorCameraIndex] != 0) {
                checkpointError("pose anchor step count must be zero");
            }

            const int64_t poseCameraCount =
                static_cast<int64_t>(checkpoint.poseCameraCount);
            checkpoint.poseBasePoses = readTensorRecord(
                reader, {poseCameraCount, 16}, "pose.base_camera_to_world");
            if (version >= 4) {
                const uint32_t conditioning =
                    reader.scalar<uint32_t>("pose conditioning mode");
                if (version == 4 && conditioning != static_cast<uint32_t>(
                        msplat::CameraPoseConditioning::CamP)) {
                    checkpointError(
                        "version 4 pose checkpoints require CamP conditioning");
                }
                if (version >= 5 &&
                    conditioning != static_cast<uint32_t>(
                        msplat::CameraPoseConditioning::Raw) &&
                    conditioning != static_cast<uint32_t>(
                        msplat::CameraPoseConditioning::CamP)) {
                    checkpointError("pose conditioning mode is invalid");
                }
                checkpoint.poseConditioning =
                    static_cast<msplat::CameraPoseConditioning>(conditioning);

                if (checkpoint.poseConditioning ==
                    msplat::CameraPoseConditioning::CamP) {
                    checkpoint.posePreconditionerReady.resize(
                        checkpoint.poseCameraCount);
                    reader.bytes(
                        checkpoint.posePreconditionerReady.data(),
                        checkedMultiply(checkpoint.poseCameraCount,
                                        sizeof(uint8_t),
                                        "pose preconditioner readiness"),
                        "pose preconditioner readiness");
                    for (size_t camera = 0;
                         camera < checkpoint.posePreconditionerReady.size();
                         ++camera) {
                        const uint8_t ready =
                            checkpoint.posePreconditionerReady[camera];
                        if (ready > 1) {
                            checkpointError(
                                "pose preconditioner readiness flag is invalid");
                        }
                        if (ready == 0 &&
                            checkpoint.poseStepCounts[camera] != 0) {
                            checkpointError(
                                "pose preconditioner readiness is inconsistent with optimizer steps");
                        }
                    }
                    checkpoint.posePreconditioners = readTensorRecord(
                        reader, {poseCameraCount, 36},
                        "pose.preconditioner");
                }
            }
            static constexpr const char* poseNames[3] = {
                "pose.delta",
                "pose.adam_exp_avg",
                "pose.adam_exp_avg_sq",
            };
            for (size_t tensor = 0; tensor < 3; ++tensor) {
                checkpoint.poseTensors[tensor] = readTensorRecord(
                    reader, {poseCameraCount, 6}, poseNames[tensor]);
            }
        }
    }

    if (version >= 5) {
        const uint32_t rawAppearanceMode =
            reader.scalar<uint32_t>("appearance mode");
        if (rawAppearanceMode >
            static_cast<uint32_t>(msplat::AppearanceMode::PPISP)) {
            checkpointError("appearance mode is invalid");
        }
        checkpoint.appearanceMode =
            static_cast<msplat::AppearanceMode>(rawAppearanceMode);
        if ((checkpoint.appearanceMode == msplat::AppearanceMode::RgbGains) !=
            checkpoint.photometricEnabled) {
            checkpointError(
                "appearance mode is inconsistent with photometric state");
        }

        if (checkpoint.appearanceMode == msplat::AppearanceMode::PPISP) {
            checkpoint.ppispCameraCount =
                reader.scalar<uint32_t>("PPISP camera count");
            if (checkpoint.ppispCameraCount == 0 ||
                checkpoint.ppispCameraCount > CKPT_MAX_CAMERAS ||
                checkpoint.ppispCameraCount > static_cast<uint32_t>(
                    std::numeric_limits<int>::max())) {
                checkpointError(
                    "PPISP camera count is outside the supported range");
            }

            constexpr uint64_t minimumTensorRecordBytes =
                sizeof(uint32_t) + 2 * sizeof(int64_t) + sizeof(uint64_t);
            constexpr uint64_t minimumPerCameraBytes =
                sizeof(uint32_t) + 1 + sizeof(uint32_t) +
                3 * kPpispParameterCount * sizeof(float);
            const uint64_t minimumRemainingBytes =
                checkedMultiply(checkpoint.ppispCameraCount,
                                minimumPerCameraBytes,
                                "PPISP checkpoint minimum") +
                3 * minimumTensorRecordBytes;
            if (reader.remaining() < minimumRemainingBytes)
                checkpointError("truncated PPISP checkpoint payload");

            checkpoint.ppispFrameIds.reserve(checkpoint.ppispCameraCount);
            std::unordered_set<std::string> uniqueFrameIds;
            uniqueFrameIds.reserve(checkpoint.ppispCameraCount);
            uint64_t totalFrameIdBytes = 0;
            for (uint32_t camera = 0;
                 camera < checkpoint.ppispCameraCount; ++camera) {
                const uint32_t length =
                    reader.scalar<uint32_t>("PPISP frame ID length");
                if (length == 0 || length > CKPT_MAX_FRAME_ID_BYTES)
                    checkpointError("PPISP frame ID length is invalid");
                if (length >
                    CKPT_MAX_FRAME_ID_TOTAL_BYTES - totalFrameIdBytes) {
                    checkpointError(
                        "PPISP frame IDs exceed the aggregate size limit");
                }
                totalFrameIdBytes += length;
                std::string frameId(length, '\0');
                reader.bytes(frameId.data(), length, "PPISP frame ID");
                if (!uniqueFrameIds.insert(frameId).second)
                    checkpointError("PPISP frame IDs are not unique");
                checkpoint.ppispFrameIds.push_back(std::move(frameId));
            }
            if (checkpoint.poseEnabled &&
                (checkpoint.ppispCameraCount != checkpoint.poseCameraCount ||
                 checkpoint.ppispFrameIds != checkpoint.poseFrameIds)) {
                checkpointError(
                    "PPISP and pose camera identities do not match");
            }

            checkpoint.ppispStepCounts.resize(checkpoint.ppispCameraCount);
            reader.bytes(
                checkpoint.ppispStepCounts.data(),
                checkedMultiply(checkpoint.ppispCameraCount,
                                sizeof(uint32_t), "PPISP step counts"),
                "PPISP step counts");
            uint64_t totalPpispSteps = 0;
            for (uint32_t count : checkpoint.ppispStepCounts) {
                if (count > checkpoint.adamSteps ||
                    totalPpispSteps > static_cast<uint64_t>(
                        checkpoint.adamSteps - count)) {
                    checkpointError(
                        "PPISP step counts are inconsistent with Adam state");
                }
                totalPpispSteps += count;
            }

            const std::vector<int64_t> ppispShape = {
                static_cast<int64_t>(checkpoint.ppispCameraCount),
                kPpispParameterCount};
            static constexpr const char* ppispNames[3] = {
                "ppisp.parameters",
                "ppisp.adam_exp_avg",
                "ppisp.adam_exp_avg_sq",
            };
            for (size_t tensor = 0; tensor < 3; ++tensor) {
                checkpoint.ppispTensors[tensor] = readTensorRecord(
                    reader, ppispShape, ppispNames[tensor]);
            }
        }
    }

    reader.requireFullyConsumed();
    return checkpoint;
}

MTensor loadCheckpointBuffer(CheckpointReader &reader,
                             const CheckpointTensorRecord &record,
                             int capacity,
                             const std::string &name) {
    std::vector<int64_t> allocationShape = record.shape;
    allocationShape[0] = capacity;
    const uint64_t allocationBytes = tensorByteCount(allocationShape, name + " allocation");
    if (record.byteCount > allocationBytes)
        checkpointError(name + " payload exceeds its destination allocation");
    if (allocationBytes > std::numeric_limits<size_t>::max())
        checkpointError(name + " allocation exceeds the platform size limit");

    MTensor buffer = gpu_zeros(allocationShape, DType::Float32);
    if (record.byteCount > 0)
        reader.readAt(record.dataOffset, buffer.data_ptr(), record.byteCount, name + ".data");
    return buffer;
}

std::vector<float> loadCheckpointFloatVector(
    CheckpointReader& reader, const CheckpointTensorRecord& record,
    const std::string& name
) {
    if (record.byteCount % sizeof(float) != 0)
        checkpointError(name + " payload is not Float32-aligned");
    const uint64_t valueCount = record.byteCount / sizeof(float);
    if (valueCount > std::numeric_limits<size_t>::max())
        checkpointError(name + " payload exceeds the platform size limit");
    std::vector<float> values(static_cast<size_t>(valueCount));
    if (record.byteCount > 0) {
        reader.readAt(record.dataOffset, values.data(), record.byteCount,
                      name + ".data");
    }
    return values;
}

std::vector<float> identityPosePreconditioners(size_t cameraCount) {
    if (cameraCount > std::numeric_limits<size_t>::max() / 36)
        checkpointError("pose preconditioner size overflows");
    std::vector<float> values(cameraCount * 36, 0.0f);
    for (size_t camera = 0; camera < cameraCount; ++camera) {
        for (size_t axis = 0; axis < 6; ++axis)
            values[camera * 36 + axis * 6 + axis] = 1.0f;
    }
    return values;
}

bool posePreconditionersAreValid(const std::vector<float>& values,
                                 size_t cameraCount) {
    if (cameraCount > std::numeric_limits<size_t>::max() / 36 ||
        values.size() != cameraCount * 36) {
        return false;
    }
    for (size_t camera = 0; camera < cameraCount; ++camera) {
        const float* matrix = values.data() + camera * 36;
        float maximumMagnitude = 0.0f;
        for (size_t element = 0; element < 36; ++element) {
            if (!std::isfinite(matrix[element]))
                return false;
            maximumMagnitude = std::max(
                maximumMagnitude, std::abs(matrix[element]));
        }
        const float symmetryTolerance =
            64.0f * std::numeric_limits<float>::epsilon() *
            std::max(1.0f, maximumMagnitude);
        for (size_t row = 0; row < 6; ++row) {
            if (!(matrix[row * 6 + row] > 0.0f))
                return false;
            for (size_t column = row + 1; column < 6; ++column) {
                if (std::abs(matrix[row * 6 + column] -
                             matrix[column * 6 + row]) >
                    symmetryTolerance) {
                    return false;
                }
            }
        }

        // Every persisted transform must be strictly positive definite. A
        // symmetric-but-singular matrix would silently freeze a pose axis;
        // an indefinite matrix would no longer represent CamP's inverse
        // square root. Cholesky is small, deterministic, and sufficient here.
        std::array<double, 36> lower{};
        for (size_t row = 0; row < 6; ++row) {
            for (size_t column = 0; column <= row; ++column) {
                double value = row == column
                    ? static_cast<double>(matrix[row * 6 + column])
                    : 0.5 * static_cast<double>(
                        matrix[row * 6 + column] +
                        matrix[column * 6 + row]);
                for (size_t inner = 0; inner < column; ++inner) {
                    value -= lower[row * 6 + inner] *
                        lower[column * 6 + inner];
                }
                if (row == column) {
                    if (!std::isfinite(value) || !(value > 0.0))
                        return false;
                    lower[row * 6 + column] = std::sqrt(value);
                } else {
                    const double pivot = lower[column * 6 + column];
                    if (!std::isfinite(pivot) || !(pivot > 0.0))
                        return false;
                    lower[row * 6 + column] = value / pivot;
                    if (!std::isfinite(lower[row * 6 + column]))
                        return false;
                }
            }
        }
    }
    return true;
}

} // namespace

void validateCheckpointFile(const std::string &filename) {
    CheckpointReader reader(filename);
    (void)parseCheckpointMetadata(reader);
}

void Model::saveCheckpoint(const std::string &filename, int step) {
    msplat_gpu_sync();

    const bool savesPpisp = appearanceMode == msplat::AppearanceMode::PPISP;
    const bool savesCampPoseConditioning = refineCameraPoses &&
        cameraPoseConditioning == msplat::CameraPoseConditioning::CamP;
    const uint32_t checkpointVersion =
        savesPpisp ? CKPT_VERSION :
        (savesCampPoseConditioning ? 4u : 3u);

    msplat::detail::AtomicOutputFile output(filename);
    std::ofstream f(output.temporary(), std::ios::binary | std::ios::trunc);
    if (!f.is_open()) throw std::runtime_error("Cannot open checkpoint file for writing: " + filename);

    // Header
    f.write(reinterpret_cast<const char*>(&CKPT_MAGIC), sizeof(CKPT_MAGIC));
    f.write(reinterpret_cast<const char*>(&checkpointVersion),
            sizeof(checkpointVersion));

    // Scalar state
    uint32_t u;
    u = (uint32_t)step;            f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)num_active;      f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)shDegree;        f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)adam_step_count;  f.write(reinterpret_cast<const char*>(&u), sizeof(u));

    // Adam learning rates
    f.write(reinterpret_cast<const char*>(adam_lr), sizeof(adam_lr));
    f.write(reinterpret_cast<const char*>(&means_lr_init), sizeof(means_lr_init));
    f.write(reinterpret_cast<const char*>(&means_lr_final), sizeof(means_lr_final));

    // Gaussian parameters (views — only num_active elements)
    writeTensor(f, means);
    writeTensor(f, scales);
    writeTensor(f, quats);
    writeTensor(f, featuresDc);
    writeTensor(f, featuresRest);
    writeTensor(f, opacities);

    // Optimizer state
    for (int g = 0; g < N_ADAM_GROUPS; g++) writeTensor(f, adam_exp_avg[g]);
    for (int g = 0; g < N_ADAM_GROUPS; g++) writeTensor(f, adam_exp_avg_sq[g]);

    // Version 2 extension. Gains model the source images only; stable frame
    // identities prevent optimizer rows from silently attaching to a different
    // camera order after resume.
    const bool savesRgbGains =
        appearanceMode == msplat::AppearanceMode::RgbGains;
    u = savesRgbGains ? 1u : 0u;
    f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    if (savesRgbGains) {
        auto hasPhotometricShape = [&](const MTensor& tensor) {
            return tensor.defined() && tensor.isGpu() &&
                tensor.dtype() == DType::Float32 && tensor.ndim() == 2 &&
                tensor.size(0) == datasetCameraCount && tensor.size(1) == 3;
        };
        if (datasetCameraCount <= 0 || adam_step_count < 0 ||
            static_cast<size_t>(datasetCameraCount) >
                kPhotometricMaxCameras ||
            cameraFrameIds.size() !=
                static_cast<size_t>(datasetCameraCount) ||
            cameraLogGainStepCounts.size() !=
                static_cast<size_t>(datasetCameraCount) ||
            !hasPhotometricShape(cameraLogGains) ||
            !hasPhotometricShape(cameraLogGainExpAvg) ||
            !hasPhotometricShape(cameraLogGainExpAvgSq)) {
            throw std::runtime_error(
                "Photometric checkpoint state is incomplete");
        }

        std::unordered_set<std::string> uniqueFrameIds;
        uniqueFrameIds.reserve(cameraFrameIds.size());
        size_t totalFrameIdBytes = 0;
        for (const std::string& frameId : cameraFrameIds) {
            if (frameId.empty() ||
                frameId.size() > kPhotometricMaxFrameIdBytes ||
                frameId.size() >
                    kPhotometricMaxFrameIdentityBytes - totalFrameIdBytes ||
                !uniqueFrameIds.insert(frameId).second) {
                throw std::runtime_error(
                    "Photometric checkpoint camera identities are invalid");
            }
            totalFrameIdBytes += frameId.size();
        }

        uint64_t totalPhotometricSteps = 0;
        for (uint32_t count : cameraLogGainStepCounts) {
            if (count > static_cast<uint32_t>(adam_step_count) ||
                totalPhotometricSteps >
                    static_cast<uint64_t>(adam_step_count) - count) {
                throw std::runtime_error(
                    "Photometric checkpoint step counts are invalid");
            }
            totalPhotometricSteps += count;
        }

        const int64_t valueCount =
            static_cast<int64_t>(datasetCameraCount) * 3;
        const float* gains = cameraLogGains.data<float>();
        const float* firstMoments = cameraLogGainExpAvg.data<float>();
        const float* secondMoments = cameraLogGainExpAvgSq.data<float>();
        for (int64_t value = 0; value < valueCount; ++value) {
            if (!std::isfinite(gains[value]) ||
                std::abs(gains[value]) >
                    kPhotometricMaxAbsLogGain + 1.0e-6f ||
                !std::isfinite(firstMoments[value]) ||
                !std::isfinite(secondMoments[value]) ||
                secondMoments[value] < 0.0f) {
                throw std::runtime_error(
                    "Photometric checkpoint tensor data is invalid");
            }
        }

        u = static_cast<uint32_t>(datasetCameraCount);
        f.write(reinterpret_cast<const char*>(&u), sizeof(u));
        for (const std::string& frameId : cameraFrameIds)
            writeCheckpointString(f, frameId);
        f.write(
            reinterpret_cast<const char*>(cameraLogGainStepCounts.data()),
            static_cast<std::streamsize>(
                cameraLogGainStepCounts.size() * sizeof(uint32_t)));
        writeTensor(f, cameraLogGains);
        writeTensor(f, cameraLogGainExpAvg);
        writeTensor(f, cameraLogGainExpAvgSq);
    }

    // Version 3 extension. Pose rows remain relative to the exact imported
    // camera geometry recorded here; resume refuses to attach them to changed
    // poses even when frame names are unchanged. Version 4 additionally saves
    // CamP's fixed per-camera conditioning state before the optimizer tensors;
    // version 5 records Raw or CamP explicitly when PPISP requires the v5 tail.
    u = refineCameraPoses ? 1u : 0u;
    f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    if (refineCameraPoses) {
        auto hasPoseShape = [&](const MTensor& tensor) {
            return tensor.defined() && tensor.isGpu() &&
                tensor.dtype() == DType::Float32 && tensor.ndim() == 2 &&
                tensor.size(0) == datasetCameraCount && tensor.size(1) == 6;
        };
        auto hasPosePreconditionerShape = [&](const MTensor& tensor) {
            return tensor.defined() && tensor.isGpu() &&
                tensor.dtype() == DType::Float32 && tensor.ndim() == 2 &&
                tensor.size(0) == datasetCameraCount && tensor.size(1) == 36;
        };
        const size_t cameraCount = static_cast<size_t>(datasetCameraCount);
        if (datasetCameraCount <= 0 || adam_step_count < 0 ||
            cameraCount > kPhotometricMaxCameras ||
            poseAnchorCameraIndex < 0 ||
            poseAnchorCameraIndex >= datasetCameraCount ||
            cameraFrameIds.size() != cameraCount ||
            cameraBasePoses.size() != cameraCount * 16 ||
            cameraPoseStepCounts.size() != cameraCount ||
            !hasPoseShape(cameraPoseDeltas) ||
            !hasPoseShape(cameraPoseExpAvg) ||
            !hasPoseShape(cameraPoseExpAvgSq) ||
            (savesCampPoseConditioning &&
             (!hasPosePreconditionerShape(cameraPosePreconditioners) ||
              cameraPosePreconditionerValues.size() != cameraCount * 36 ||
              cameraPosePreconditionerReady.size() != cameraCount))) {
            throw std::runtime_error("Camera pose checkpoint state is incomplete");
        }

        if (savesCampPoseConditioning) {
            for (size_t camera = 0; camera < cameraCount; ++camera) {
                const uint8_t ready = cameraPosePreconditionerReady[camera];
                if (ready > 1 ||
                    (ready == 0 && cameraPoseStepCounts[camera] != 0)) {
                    throw std::runtime_error(
                        "Camera pose checkpoint preconditioner readiness is invalid");
                }
            }
            if (cameraPosePreconditionerReady[
                    static_cast<size_t>(poseAnchorCameraIndex)] == 0 ||
                !posePreconditionersAreValid(
                    cameraPosePreconditionerValues, cameraCount) ||
                !std::equal(
                    cameraPosePreconditionerValues.begin(),
                    cameraPosePreconditionerValues.end(),
                    cameraPosePreconditioners.data<float>())) {
                throw std::runtime_error(
                    "Camera pose checkpoint preconditioner data is invalid");
            }
        }

        std::unordered_set<std::string> uniqueFrameIds;
        uniqueFrameIds.reserve(cameraFrameIds.size());
        size_t totalFrameIdBytes = 0;
        for (const std::string& frameId : cameraFrameIds) {
            if (frameId.empty() ||
                frameId.size() > kPhotometricMaxFrameIdBytes ||
                frameId.size() >
                    kPhotometricMaxFrameIdentityBytes - totalFrameIdBytes ||
                !uniqueFrameIds.insert(frameId).second) {
                throw std::runtime_error(
                    "Camera pose checkpoint identities are invalid");
            }
            totalFrameIdBytes += frameId.size();
        }

        uint64_t totalPoseSteps = 0;
        for (size_t camera = 0; camera < cameraCount; ++camera) {
            const uint32_t count = cameraPoseStepCounts[camera];
            if (count > static_cast<uint32_t>(adam_step_count) ||
                totalPoseSteps >
                    static_cast<uint64_t>(adam_step_count) - count) {
                throw std::runtime_error(
                    "Camera pose checkpoint step counts are invalid");
            }
            totalPoseSteps += count;
        }
        if (cameraPoseStepCounts[
                static_cast<size_t>(poseAnchorCameraIndex)] != 0) {
            throw std::runtime_error(
                "Camera pose checkpoint anchor state is invalid");
        }

        for (float value : cameraBasePoses) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Camera pose checkpoint base geometry is invalid");
            }
        }
        const float* deltas = cameraPoseDeltas.data<float>();
        const float* firstMoments = cameraPoseExpAvg.data<float>();
        const float* secondMoments = cameraPoseExpAvgSq.data<float>();
        for (size_t camera = 0; camera < cameraCount; ++camera) {
            float translationNorm2 = 0.0f;
            float rotationNorm2 = 0.0f;
            for (size_t component = 0; component < 6; ++component) {
                const size_t index = camera * 6 + component;
                const float delta = deltas[index];
                if (!std::isfinite(delta) ||
                    !std::isfinite(firstMoments[index]) ||
                    !std::isfinite(secondMoments[index]) ||
                    secondMoments[index] < 0.0f) {
                    throw std::runtime_error(
                        "Camera pose checkpoint tensor data is invalid");
                }
                if (component < 3)
                    translationNorm2 += delta * delta;
                else
                    rotationNorm2 += delta * delta;
            }
            if (translationNorm2 >
                    (kPoseMaxTranslation + 1.0e-6f) *
                    (kPoseMaxTranslation + 1.0e-6f) ||
                rotationNorm2 >
                    (kPoseMaxRotation + 1.0e-6f) *
                    (kPoseMaxRotation + 1.0e-6f)) {
                throw std::runtime_error(
                    "Camera pose checkpoint delta is outside its bounds");
            }
        }
        const size_t anchorOffset =
            static_cast<size_t>(poseAnchorCameraIndex) * 6;
        for (size_t component = 0; component < 6; ++component) {
            if (deltas[anchorOffset + component] != 0.0f ||
                firstMoments[anchorOffset + component] != 0.0f ||
                secondMoments[anchorOffset + component] != 0.0f) {
                throw std::runtime_error(
                    "Camera pose checkpoint anchor state is invalid");
            }
        }

        u = static_cast<uint32_t>(datasetCameraCount);
        f.write(reinterpret_cast<const char*>(&u), sizeof(u));
        u = static_cast<uint32_t>(poseAnchorCameraIndex);
        f.write(reinterpret_cast<const char*>(&u), sizeof(u));
        for (const std::string& frameId : cameraFrameIds)
            writeCheckpointString(f, frameId);
        f.write(
            reinterpret_cast<const char*>(cameraPoseStepCounts.data()),
            static_cast<std::streamsize>(
                cameraPoseStepCounts.size() * sizeof(uint32_t)));
        writeFloatTensor(
            f, cameraBasePoses,
            {static_cast<int64_t>(datasetCameraCount), 16});
        if (checkpointVersion >= 5 || savesCampPoseConditioning) {
            u = static_cast<uint32_t>(cameraPoseConditioning);
            f.write(reinterpret_cast<const char*>(&u), sizeof(u));
        }
        if (savesCampPoseConditioning) {
            f.write(
                reinterpret_cast<const char*>(
                    cameraPosePreconditionerReady.data()),
                static_cast<std::streamsize>(
                    cameraPosePreconditionerReady.size() * sizeof(uint8_t)));
            writeFloatTensor(
                f, cameraPosePreconditionerValues,
                {static_cast<int64_t>(datasetCameraCount), 36});
        }
        writeTensor(f, cameraPoseDeltas);
        writeTensor(f, cameraPoseExpAvg);
        writeTensor(f, cameraPoseExpAvgSq);
    }

    // Version 5 extension. PPISP owns an independent sparse Adam row per stable
    // frame identity. None/RGB-only checkpoints keep their exact v3/v4 format.
    if (checkpointVersion >= 5) {
        u = static_cast<uint32_t>(appearanceMode);
        f.write(reinterpret_cast<const char*>(&u), sizeof(u));
        if (savesPpisp) {
            auto hasPpispShape = [&](const MTensor& tensor) {
                return tensor.defined() && tensor.isGpu() &&
                    tensor.dtype() == DType::Float32 && tensor.ndim() == 2 &&
                    tensor.size(0) == datasetCameraCount &&
                    tensor.size(1) == kPpispParameterCount;
            };
            const size_t cameraCount =
                static_cast<size_t>(datasetCameraCount);
            if (datasetCameraCount <= 0 || adam_step_count < 0 ||
                cameraCount > kPhotometricMaxCameras ||
                cameraFrameIds.size() != cameraCount ||
                cameraPpispStepCounts.size() != cameraCount ||
                !hasPpispShape(cameraPpispParameters) ||
                !hasPpispShape(cameraPpispExpAvg) ||
                !hasPpispShape(cameraPpispExpAvgSq)) {
                throw std::runtime_error(
                    "PPISP checkpoint state is incomplete");
            }

            std::unordered_set<std::string> uniqueFrameIds;
            uniqueFrameIds.reserve(cameraFrameIds.size());
            size_t totalFrameIdBytes = 0;
            for (const std::string& frameId : cameraFrameIds) {
                if (frameId.empty() ||
                    frameId.size() > kPhotometricMaxFrameIdBytes ||
                    frameId.size() >
                        kPhotometricMaxFrameIdentityBytes - totalFrameIdBytes ||
                    !uniqueFrameIds.insert(frameId).second) {
                    throw std::runtime_error(
                        "PPISP checkpoint camera identities are invalid");
                }
                totalFrameIdBytes += frameId.size();
            }

            uint64_t totalPpispSteps = 0;
            for (uint32_t count : cameraPpispStepCounts) {
                if (count > static_cast<uint32_t>(adam_step_count) ||
                    totalPpispSteps > static_cast<uint64_t>(
                        adam_step_count) - count) {
                    throw std::runtime_error(
                        "PPISP checkpoint step counts are invalid");
                }
                totalPpispSteps += count;
            }

            const int64_t valueCount =
                static_cast<int64_t>(datasetCameraCount) *
                kPpispParameterCount;
            const float* parameters = cameraPpispParameters.data<float>();
            const float* firstMoments = cameraPpispExpAvg.data<float>();
            const float* secondMoments = cameraPpispExpAvgSq.data<float>();
            for (int64_t value = 0; value < valueCount; ++value) {
                const int64_t component = value % kPpispParameterCount;
                const float bound = component == 0
                    ? kPpispMaxAbsExposure
                    : kPpispMaxAbsColorParameter;
                if (!std::isfinite(parameters[value]) ||
                    std::abs(parameters[value]) > bound + 1.0e-6f ||
                    !std::isfinite(firstMoments[value]) ||
                    !std::isfinite(secondMoments[value]) ||
                    secondMoments[value] < 0.0f) {
                    throw std::runtime_error(
                        "PPISP checkpoint tensor data is invalid");
                }
            }

            u = static_cast<uint32_t>(datasetCameraCount);
            f.write(reinterpret_cast<const char*>(&u), sizeof(u));
            for (const std::string& frameId : cameraFrameIds)
                writeCheckpointString(f, frameId);
            f.write(
                reinterpret_cast<const char*>(cameraPpispStepCounts.data()),
                static_cast<std::streamsize>(
                    cameraPpispStepCounts.size() * sizeof(uint32_t)));
            writeTensor(f, cameraPpispParameters);
            writeTensor(f, cameraPpispExpAvg);
            writeTensor(f, cameraPpispExpAvgSq);
        }
    }

    f.flush();
    if (!f) throw std::runtime_error("Failed while writing checkpoint file: " + filename);
    f.close();
    if (!f) throw std::runtime_error("Failed to close checkpoint file: " + filename);

    output.commit("checkpoint");

    std::error_code sizeError;
    const uintmax_t checkpointBytes = fs::file_size(filename, sizeError);
    std::cout << "Checkpoint saved: " << filename << " (step " << step
              << ", " << num_active << " gaussians";
    if (!sizeError) std::cout << ", " << checkpointBytes / (1024*1024) << " MB";
    std::cout << ")" << std::endl;
}

int Model::loadCheckpoint(const std::string &filename) {
    CheckpointReader reader(filename);
    const ParsedCheckpoint checkpoint = parseCheckpointMetadata(reader);
    const int activeCount = static_cast<int>(checkpoint.numPoints);
    if (checkpoint.shDegree != static_cast<uint32_t>(configuredSHDegree))
        checkpointError("SH degree does not match the configured training degree");
    if (maxGaussians > 0 && activeCount > maxGaussians)
        checkpointError("Gaussian count exceeds maxGaussians");
    if (checkpoint.appearanceMode != msplat::AppearanceMode::None &&
        checkpoint.appearanceMode != appearanceMode) {
        checkpointError(
            "appearance mode does not match the configured mode");
    }
    if (checkpoint.photometricEnabled &&
        (checkpoint.cameraCount != static_cast<uint32_t>(datasetCameraCount) ||
         checkpoint.cameraFrameIds != cameraFrameIds)) {
        checkpointError(
            "photometric camera identities do not match the dataset");
    }
    if (checkpoint.appearanceMode == msplat::AppearanceMode::PPISP &&
        (checkpoint.ppispCameraCount !=
             static_cast<uint32_t>(datasetCameraCount) ||
         checkpoint.ppispFrameIds != cameraFrameIds)) {
        checkpointError("PPISP camera identities do not match the dataset");
    }
    if (checkpoint.poseEnabled && !refineCameraPoses)
        checkpointError(
            "checkpoint requires camera pose refinement to be enabled");
    if (checkpoint.poseEnabled &&
        (checkpoint.poseCameraCount !=
             static_cast<uint32_t>(datasetCameraCount) ||
         checkpoint.poseFrameIds != cameraFrameIds)) {
        checkpointError("pose camera identities do not match the dataset");
    }
    if (checkpoint.poseEnabled &&
        checkpoint.poseAnchorCameraIndex !=
            static_cast<uint32_t>(poseAnchorCameraIndex)) {
        checkpointError("pose anchor does not match the training split");
    }
    if (checkpoint.poseEnabled &&
        checkpoint.poseConditioning != cameraPoseConditioning) {
        checkpointError(
            "pose conditioning mode does not match the configured mode");
    }
    const int capacity = capacityFor(activeCount);
    if (capacity < activeCount)
        checkpointError("Gaussian capacity calculation overflowed");

    static constexpr const char *parameterNames[N_ADAM_GROUPS] = {
        "means", "scales", "quats", "features_dc", "features_rest", "opacities"
    };

    // Load everything into independent buffers before changing the live model.
    // A malformed/truncated file or an allocation failure therefore leaves the
    // existing parameters and optimizer state intact.
    std::array<MTensor, N_ADAM_GROUPS> newParameterBuffers;
    std::array<MTensor, N_ADAM_GROUPS> newAdamExpAvgBuffers;
    std::array<MTensor, N_ADAM_GROUPS> newAdamExpAvgSqBuffers;
    for (int group = 0; group < N_ADAM_GROUPS; ++group) {
        newParameterBuffers[group] = loadCheckpointBuffer(
            reader, checkpoint.parameters[group], capacity, parameterNames[group]);
        newAdamExpAvgBuffers[group] = loadCheckpointBuffer(
            reader, checkpoint.adamExpAvg[group], capacity,
            std::string("adam_exp_avg.") + parameterNames[group]);
        newAdamExpAvgSqBuffers[group] = loadCheckpointBuffer(
            reader, checkpoint.adamExpAvgSq[group], capacity,
            std::string("adam_exp_avg_sq.") + parameterNames[group]);
    }

    const bool usesRgbGains =
        appearanceMode == msplat::AppearanceMode::RgbGains;
    const int photometricRows = usesRgbGains
        ? datasetCameraCount
        : 1;
    MTensor newCameraLogGains;
    MTensor newCameraLogGainExpAvg;
    MTensor newCameraLogGainExpAvgSq;
    std::vector<uint32_t> newCameraLogGainStepCounts;
    if (checkpoint.photometricEnabled) {
        newCameraLogGains = loadCheckpointBuffer(
            reader, checkpoint.photometricTensors[0], datasetCameraCount,
            "photometric.log_rgb_gains");
        newCameraLogGainExpAvg = loadCheckpointBuffer(
            reader, checkpoint.photometricTensors[1], datasetCameraCount,
            "photometric.adam_exp_avg");
        newCameraLogGainExpAvgSq = loadCheckpointBuffer(
            reader, checkpoint.photometricTensors[2], datasetCameraCount,
            "photometric.adam_exp_avg_sq");
        newCameraLogGainStepCounts = checkpoint.cameraStepCounts;

        const int64_t valueCount =
            static_cast<int64_t>(datasetCameraCount) * 3;
        const float* gains = newCameraLogGains.data<float>();
        const float* firstMoments = newCameraLogGainExpAvg.data<float>();
        const float* secondMoments = newCameraLogGainExpAvgSq.data<float>();
        for (int64_t value = 0; value < valueCount; ++value) {
            if (!std::isfinite(gains[value]) ||
                std::abs(gains[value]) > kPhotometricMaxAbsLogGain + 1.0e-6f ||
                !std::isfinite(firstMoments[value]) ||
                !std::isfinite(secondMoments[value]) ||
                secondMoments[value] < 0.0f) {
                checkpointError("photometric tensor data is invalid");
            }
        }
    } else {
        newCameraLogGains = gpu_zeros(
            {photometricRows, 3}, DType::Float32);
        if (usesRgbGains) {
            newCameraLogGainExpAvg = gpu_zeros(
                {photometricRows, 3}, DType::Float32);
            newCameraLogGainExpAvgSq = gpu_zeros(
                {photometricRows, 3}, DType::Float32);
            newCameraLogGainStepCounts.assign(
                static_cast<size_t>(datasetCameraCount), 0);
        }
    }

    const bool usesPpisp = appearanceMode == msplat::AppearanceMode::PPISP;
    const int ppispRows = usesPpisp ? datasetCameraCount : 1;
    MTensor newCameraPpispParameters;
    MTensor newCameraPpispExpAvg;
    MTensor newCameraPpispExpAvgSq;
    std::vector<uint32_t> newCameraPpispStepCounts;
    if (checkpoint.appearanceMode == msplat::AppearanceMode::PPISP) {
        newCameraPpispParameters = loadCheckpointBuffer(
            reader, checkpoint.ppispTensors[0], datasetCameraCount,
            "ppisp.parameters");
        newCameraPpispExpAvg = loadCheckpointBuffer(
            reader, checkpoint.ppispTensors[1], datasetCameraCount,
            "ppisp.adam_exp_avg");
        newCameraPpispExpAvgSq = loadCheckpointBuffer(
            reader, checkpoint.ppispTensors[2], datasetCameraCount,
            "ppisp.adam_exp_avg_sq");
        newCameraPpispStepCounts = checkpoint.ppispStepCounts;

        const int64_t valueCount =
            static_cast<int64_t>(datasetCameraCount) *
            kPpispParameterCount;
        const float* parameters = newCameraPpispParameters.data<float>();
        const float* firstMoments = newCameraPpispExpAvg.data<float>();
        const float* secondMoments = newCameraPpispExpAvgSq.data<float>();
        for (int64_t value = 0; value < valueCount; ++value) {
            const int64_t component = value % kPpispParameterCount;
            const float bound = component == 0
                ? kPpispMaxAbsExposure
                : kPpispMaxAbsColorParameter;
            if (!std::isfinite(parameters[value]) ||
                std::abs(parameters[value]) > bound + 1.0e-6f ||
                !std::isfinite(firstMoments[value]) ||
                !std::isfinite(secondMoments[value]) ||
                secondMoments[value] < 0.0f) {
                checkpointError("PPISP tensor data is invalid");
            }
        }
    } else {
        newCameraPpispParameters = gpu_zeros(
            {ppispRows, kPpispParameterCount}, DType::Float32);
        if (usesPpisp) {
            newCameraPpispExpAvg = gpu_zeros(
                {ppispRows, kPpispParameterCount}, DType::Float32);
            newCameraPpispExpAvgSq = gpu_zeros(
                {ppispRows, kPpispParameterCount}, DType::Float32);
            newCameraPpispStepCounts.assign(
                static_cast<size_t>(datasetCameraCount), 0);
        }
    }

    const int poseRows = refineCameraPoses ? datasetCameraCount : 1;
    MTensor newCameraPoseDeltas;
    MTensor newCameraPoseExpAvg;
    MTensor newCameraPoseExpAvgSq;
    MTensor newCameraPosePreconditioners;
    std::vector<uint32_t> newCameraPoseStepCounts;
    std::vector<float> newCameraPosePreconditionerValues;
    std::vector<uint8_t> newCameraPosePreconditionerReady;
    if (checkpoint.poseEnabled) {
        const std::vector<float> checkpointBasePoses =
            loadCheckpointFloatVector(
                reader, checkpoint.poseBasePoses,
                "pose.base_camera_to_world");
        for (float value : checkpointBasePoses) {
            if (!std::isfinite(value))
                checkpointError("pose base camera geometry is invalid");
        }
        if (checkpointBasePoses != cameraBasePoses)
            checkpointError("pose base camera geometry does not match the dataset");

        if (checkpoint.poseConditioning ==
            msplat::CameraPoseConditioning::CamP) {
            newCameraPosePreconditionerValues = loadCheckpointFloatVector(
                reader, checkpoint.posePreconditioners,
                "pose.preconditioner");
            if (!posePreconditionersAreValid(
                    newCameraPosePreconditionerValues,
                    static_cast<size_t>(datasetCameraCount))) {
                checkpointError("pose preconditioner data is invalid");
            }
            newCameraPosePreconditionerReady =
                checkpoint.posePreconditionerReady;
            if (newCameraPosePreconditionerReady[
                    static_cast<size_t>(poseAnchorCameraIndex)] == 0) {
                checkpointError(
                    "pose anchor preconditioner must be ready");
            }
        }

        newCameraPoseDeltas = loadCheckpointBuffer(
            reader, checkpoint.poseTensors[0], datasetCameraCount,
            "pose.delta");
        newCameraPoseExpAvg = loadCheckpointBuffer(
            reader, checkpoint.poseTensors[1], datasetCameraCount,
            "pose.adam_exp_avg");
        newCameraPoseExpAvgSq = loadCheckpointBuffer(
            reader, checkpoint.poseTensors[2], datasetCameraCount,
            "pose.adam_exp_avg_sq");
        newCameraPoseStepCounts = checkpoint.poseStepCounts;

        const float* deltas = newCameraPoseDeltas.data<float>();
        const float* firstMoments = newCameraPoseExpAvg.data<float>();
        const float* secondMoments = newCameraPoseExpAvgSq.data<float>();
        for (int camera = 0; camera < datasetCameraCount; ++camera) {
            float translationNorm2 = 0.0f;
            float rotationNorm2 = 0.0f;
            for (int component = 0; component < 6; ++component) {
                const int64_t index =
                    static_cast<int64_t>(camera) * 6 + component;
                const float delta = deltas[index];
                if (!std::isfinite(delta) ||
                    !std::isfinite(firstMoments[index]) ||
                    !std::isfinite(secondMoments[index]) ||
                    secondMoments[index] < 0.0f) {
                    checkpointError("pose tensor data is invalid");
                }
                if (component < 3)
                    translationNorm2 += delta * delta;
                else
                    rotationNorm2 += delta * delta;
            }
            if (translationNorm2 >
                    (kPoseMaxTranslation + 1.0e-6f) *
                    (kPoseMaxTranslation + 1.0e-6f) ||
                rotationNorm2 >
                    (kPoseMaxRotation + 1.0e-6f) *
                    (kPoseMaxRotation + 1.0e-6f)) {
                checkpointError("pose delta is outside its bounds");
            }
        }
        const int64_t anchorOffset =
            static_cast<int64_t>(poseAnchorCameraIndex) * 6;
        for (int component = 0; component < 6; ++component) {
            if (deltas[anchorOffset + component] != 0.0f ||
                firstMoments[anchorOffset + component] != 0.0f ||
                secondMoments[anchorOffset + component] != 0.0f) {
                checkpointError("pose anchor optimizer state is not zero");
            }
        }
    } else {
        newCameraPoseDeltas = gpu_zeros(
            {poseRows, 6}, DType::Float32);
        if (refineCameraPoses) {
            newCameraPoseExpAvg = gpu_zeros(
                {poseRows, 6}, DType::Float32);
            newCameraPoseExpAvgSq = gpu_zeros(
                {poseRows, 6}, DType::Float32);
            newCameraPoseStepCounts.assign(
                static_cast<size_t>(datasetCameraCount), 0);
            if (cameraPoseConditioning ==
                msplat::CameraPoseConditioning::CamP) {
                newCameraPosePreconditionerValues =
                    identityPosePreconditioners(
                        static_cast<size_t>(datasetCameraCount));
                newCameraPosePreconditionerReady.assign(
                    static_cast<size_t>(datasetCameraCount), uint8_t{0});
                newCameraPosePreconditionerReady[
                    static_cast<size_t>(poseAnchorCameraIndex)] = 1;
            }
        }
    }

    if (refineCameraPoses &&
        cameraPoseConditioning == msplat::CameraPoseConditioning::CamP) {
        newCameraPosePreconditioners = gpu_empty(
            {poseRows, 36}, DType::Float32);
        memcpy(
            newCameraPosePreconditioners.data_ptr(),
            newCameraPosePreconditionerValues.data(),
            newCameraPosePreconditionerValues.size() * sizeof(float));
    }

    DensificationScratch newDensificationScratch;
    if (needsDensificationAfterStep(checkpoint.step, stopSplitAt)) {
        const auto &featuresRestShape = checkpoint.parameters[4].shape;
        const uint64_t featuresRestStride = checkedMultiply(
            static_cast<uint64_t>(featuresRestShape[1]),
            static_cast<uint64_t>(featuresRestShape[2]),
            "features_rest stride");
        const uint64_t compactStride = std::max<uint64_t>(featuresRestStride, 4);
        const uint64_t compactElements = checkedMultiply(
            static_cast<uint64_t>(capacity), compactStride,
            "densification compact scratch");
        if (compactElements >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            checkpointError(
                "densification compact scratch exceeds the supported range");
        }
        newDensificationScratch = makeDensificationScratch(
            capacity, static_cast<int64_t>(compactStride));
    }

    // Constructing views copies shape metadata and can allocate. Do that before
    // the no-throw state swap as part of the transactional preparation.
    std::array<MTensor, N_ADAM_GROUPS> newParameterViews;
    std::array<MTensor, N_ADAM_GROUPS> newAdamExpAvgViews;
    std::array<MTensor, N_ADAM_GROUPS> newAdamExpAvgSqViews;
    for (int group = 0; group < N_ADAM_GROUPS; ++group) {
        newParameterViews[group] = newParameterBuffers[group].view(activeCount);
        newAdamExpAvgViews[group] = newAdamExpAvgBuffers[group].view(activeCount);
        newAdamExpAvgSqViews[group] = newAdamExpAvgSqBuffers[group].view(activeCount);
    }

    // Do not release buffers that might still be referenced by an outstanding
    // command buffer. Validation and all new allocations remain non-mutating.
    msplat_gpu_sync();
    releaseOptimizers();

    MTensor *parameterBufferDestinations[N_ADAM_GROUPS] = {
        &means_buf, &scales_buf, &quats_buf,
        &featuresDc_buf, &featuresRest_buf, &opacities_buf
    };
    MTensor *parameterViewDestinations[N_ADAM_GROUPS] = {
        &means, &scales, &quats, &featuresDc, &featuresRest, &opacities
    };
    for (int group = 0; group < N_ADAM_GROUPS; ++group) {
        *parameterBufferDestinations[group] = std::move(newParameterBuffers[group]);
        *parameterViewDestinations[group] = std::move(newParameterViews[group]);
        adam_exp_avg_buf[group] = std::move(newAdamExpAvgBuffers[group]);
        adam_exp_avg[group] = std::move(newAdamExpAvgViews[group]);
        adam_exp_avg_sq_buf[group] = std::move(newAdamExpAvgSqBuffers[group]);
        adam_exp_avg_sq[group] = std::move(newAdamExpAvgSqViews[group]);
    }
    cameraLogGains = std::move(newCameraLogGains);
    cameraLogGainExpAvg = std::move(newCameraLogGainExpAvg);
    cameraLogGainExpAvgSq = std::move(newCameraLogGainExpAvgSq);
    cameraLogGainStepCounts = std::move(newCameraLogGainStepCounts);
    cameraPpispParameters = std::move(newCameraPpispParameters);
    cameraPpispExpAvg = std::move(newCameraPpispExpAvg);
    cameraPpispExpAvgSq = std::move(newCameraPpispExpAvgSq);
    cameraPpispStepCounts = std::move(newCameraPpispStepCounts);
    cameraPoseDeltas = std::move(newCameraPoseDeltas);
    cameraPoseExpAvg = std::move(newCameraPoseExpAvg);
    cameraPoseExpAvgSq = std::move(newCameraPoseExpAvgSq);
    cameraPosePreconditioners = std::move(newCameraPosePreconditioners);
    cameraPoseStepCounts = std::move(newCameraPoseStepCounts);
    cameraPosePreconditionerValues =
        std::move(newCameraPosePreconditionerValues);
    cameraPosePreconditionerReady =
        std::move(newCameraPosePreconditionerReady);

    densify_split_flag = std::move(newDensificationScratch.splitFlag);
    densify_dup_flag = std::move(newDensificationScratch.dupFlag);
    densify_split_prefix = std::move(newDensificationScratch.splitPrefix);
    densify_dup_prefix = std::move(newDensificationScratch.dupPrefix);
    densify_keep_flag = std::move(newDensificationScratch.keepFlag);
    densify_keep_prefix = std::move(newDensificationScratch.keepPrefix);
    densify_block_totals = std::move(newDensificationScratch.blockTotals);
    densify_compact_scratch =
        std::move(newDensificationScratch.compactScratch);
    densify_random_samples = std::move(newDensificationScratch.randomSamples);

    num_active = activeCount;
    buf_capacity = capacity;
    shDegree = static_cast<int>(checkpoint.shDegree);
    adam_step_count = static_cast<int>(checkpoint.adamSteps);
    std::copy(checkpoint.adamLearningRates.begin(),
              checkpoint.adamLearningRates.end(), adam_lr);
    means_lr_init = checkpoint.meansLearningRateInitial;
    means_lr_final = checkpoint.meansLearningRateFinal;

    // These are derived from the previous population or image and must be
    // rebuilt on the next training iteration.
    radii.reset();
    xysGradNorm.reset();
    visCounts.reset();
    max2DSize.reset();
    lastHeight = 0;
    lastWidth = 0;

    std::cout << "Checkpoint loaded: " << filename << " (step " << checkpoint.step
              << ", " << num_active << " gaussians)" << std::endl;

    return static_cast<int>(checkpoint.step);
}

Model::CamSetup Model::prepareCam(Camera& cam, int step) {
    const int downscale = getDownscaleFactor(step);
    if (cam.width < downscale || cam.height < downscale)
        throw std::invalid_argument(
            "Training downscale produces a zero-sized image; reduce numDownscales");
    CamSetup s;
    s.height = cam.height / downscale;
    s.width = cam.width / downscale;
    const float sx = static_cast<float>(s.width) /
                     static_cast<float>(cam.width);
    const float sy = static_cast<float>(s.height) /
                     static_cast<float>(cam.height);
    s.fx = cam.fx * sx;
    s.fy = cam.fy * sy;
    s.cx = cam.cx * sx;
    s.cy = cam.cy * sy;

    float fovX = 2.0f * std::atan(s.width / (2.0f * s.fx));
    float fovY = 2.0f * std::atan(s.height / (2.0f * s.fy));

    if (!cam.cachedViewMat.defined() || !cam.cachedProjViewMat.defined() ||
        cam.cachedFovX != fovX ||
        cam.cachedFovY != fovY || !cam.projectionCacheMatchesPose()) {
        const float *d = cam.camToWorld;
        float R[3][3], Rinv[3][3], T[3], Tinv[3];
        for (int i = 0; i < 3; i++) {
            R[i][0] = d[i*4+0]; R[i][1] = -d[i*4+1]; R[i][2] = -d[i*4+2]; T[i] = d[i*4+3];
        }
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Rinv[i][j] = R[j][i];
        for (int i = 0; i < 3; i++) Tinv[i] = -(Rinv[i][0]*T[0] + Rinv[i][1]*T[1] + Rinv[i][2]*T[2]);
        float vm[16] = { Rinv[0][0],Rinv[0][1],Rinv[0][2],Tinv[0], Rinv[1][0],Rinv[1][1],Rinv[1][2],Tinv[1], Rinv[2][0],Rinv[2][1],Rinv[2][2],Tinv[2], 0,0,0,1 };
        float t_p = 0.001f * std::tan(0.5f * fovY), r_p = 0.001f * std::tan(0.5f * fovX);
        float pm[16] = { 0.001f/r_p,0,0,0, 0,0.001f/t_p,0,0, 0,0,(1000.0f+0.001f)/(1000.0f-0.001f),-1000.0f*0.001f/(1000.0f-0.001f), 0,0,1,0 };
        float pvm[16] = {};
        for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<4;k++) pvm[i*4+j] += pm[i*4+k] * vm[k*4+j];

        cam.cachedViewMat = gpu_empty({4, 4}, DType::Float32);
        memcpy(cam.cachedViewMat.data_ptr(), vm, sizeof(vm));
        cam.cachedProjViewMat = gpu_empty({4, 4}, DType::Float32);
        memcpy(cam.cachedProjViewMat.data_ptr(), pvm, sizeof(pvm));
        cam.cachedCamPos[0] = T[0]; cam.cachedCamPos[1] = T[1]; cam.cachedCamPos[2] = T[2];
        cam.cachedFovX = fovX; cam.cachedFovY = fovY;
        cam.recordProjectionCachePose();
    }

    s.degreesToUse = (std::min<int>)(step / shDegreeInterval, shDegree);
    int b = featuresRest.size(-2) + 1;
    s.degree = (b <= 1) ? 0 : (b <= 4) ? 1 : (b <= 9) ? 2 : (b <= 16) ? 3 : 4;
    s.tileBounds = std::make_tuple(
        (s.width + BLOCK_X - 1) / BLOCK_X,
        (s.height + BLOCK_Y - 1) / BLOCK_Y, 1);
    s.cam_pos[0] = cam.cachedCamPos[0];
    s.cam_pos[1] = cam.cachedCamPos[1];
    s.cam_pos[2] = cam.cachedCamPos[2];

    return s;
}

MTensor Model::render(Camera& cam, int step){
    auto s = prepareCam(cam, step);
    return msplat_render(
        means.size(0), means, scales, 1.0f,
        quats, cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, backgroundColor);
}

void Model::fullIteration(Camera& cam, int step, MTensor& gt,
                          float ssimWeight) {
    uint64_t coverageUnits = 0;
    if (gt.ndim() >= 2 && gt.size(0) > 0 && gt.size(1) > 0) {
        const uint64_t pixels = static_cast<uint64_t>(gt.size(0)) *
            static_cast<uint64_t>(gt.size(1));
        if (pixels <= std::numeric_limits<uint64_t>::max() / 255u)
            coverageUnits = pixels * 255u;
    }
    CameraTrainingTarget target{&gt, nullptr, coverageUnits};
    fullIteration(cam, step, target, ssimWeight);
}

void Model::fullIteration(Camera& cam, size_t cameraIndex, int step,
                          MTensor& gt, float ssimWeight) {
    uint64_t coverageUnits = 0;
    if (gt.ndim() >= 2 && gt.size(0) > 0 && gt.size(1) > 0) {
        const uint64_t pixels = static_cast<uint64_t>(gt.size(0)) *
            static_cast<uint64_t>(gt.size(1));
        if (pixels <= std::numeric_limits<uint64_t>::max() / 255u)
            coverageUnits = pixels * 255u;
    }
    CameraTrainingTarget target{&gt, nullptr, coverageUnits};
    fullIteration(cam, cameraIndex, step, target, ssimWeight);
}

void Model::fullIteration(Camera& cam, int step,
                          const CameraTrainingTarget& target,
                          float ssimWeight){
    if (appearanceMode != msplat::AppearanceMode::None || refineCameraPoses) {
        throw std::invalid_argument(
            "Camera refinement requires a canonical camera index");
    }
    fullIteration(cam, 0, step, target, ssimWeight);
}

void Model::fullIteration(Camera& cam, size_t cameraIndex, int step,
                          const CameraTrainingTarget& target,
                          float ssimWeight){
    if ((appearanceMode != msplat::AppearanceMode::None ||
         refineCameraPoses) &&
        cameraIndex >= static_cast<size_t>(datasetCameraCount)) {
        throw std::out_of_range(
            "Camera refinement index is out of range");
    }
    const bool collectDensificationStats =
        collectsDensificationStats(step, stopSplitAt);
    if (!collectDensificationStats) {
        retireDensificationState();
    }

    auto s = prepareCam(cam, step);
    if (refineCameraPoses &&
        cameraPoseConditioning == msplat::CameraPoseConditioning::CamP) {
        ensureCameraPosePreconditioner(cam, cameraIndex);
    }
    lastHeight = s.height; lastWidth = s.width;
    int numPoints = means.size(0);

    const bool targetBaseValid = target.image && target.image->defined() &&
        target.image->isGpu() && target.image->ndim() == 3 &&
        target.image->size(0) == s.height &&
        target.image->size(1) == s.width;
    const bool targetFormatValid = targetBaseValid && (
        (target.image->dtype() == DType::UInt8 &&
         target.image->size(2) == 4) ||
        (target.image->dtype() == DType::Float32 &&
         target.image->size(2) == 3));
    if (!targetFormatValid) {
        throw std::invalid_argument(
            "Training image must be a GPU uint8 RGBA or float32 RGB tensor "
            "matching the camera");
    }
    const uint64_t pixelCount = static_cast<uint64_t>(s.height) *
        static_cast<uint64_t>(s.width);
    if (pixelCount == 0 ||
        pixelCount > std::numeric_limits<uint64_t>::max() / 255u) {
        throw std::invalid_argument("Training image dimensions are invalid");
    }
    const uint64_t fullCoverageUnits = pixelCount * 255u;
    if (target.coverageUnits == 0 ||
        target.coverageUnits > fullCoverageUnits) {
        throw std::invalid_argument("Training coverage denominator is invalid");
    }
    if (target.coverageMask) {
        const bool coverageIsPackedAlpha =
            target.coverageMask == target.image;
        const bool standaloneCoverageValid =
            target.coverageMask->defined() &&
            target.coverageMask->isGpu() &&
            target.coverageMask->dtype() == DType::UInt8 &&
            target.coverageMask->ndim() == 2 &&
            target.coverageMask->size(0) == s.height &&
            target.coverageMask->size(1) == s.width;
        if ((!coverageIsPackedAlpha && !standaloneCoverageValid) ||
            (coverageIsPackedAlpha &&
             target.image->dtype() != DType::UInt8)) {
            throw std::invalid_argument(
                "Training coverage must be packed RGBA alpha or a GPU uint8 "
                "tensor matching the camera");
        }
    } else if (target.coverageUnits != fullCoverageUnits) {
        throw std::invalid_argument(
            "Unmasked training coverage denominator is inconsistent");
    }
    if (target.coverageRenderTiles) {
        const int64_t expectedTileHeight =
            (static_cast<int64_t>(s.height) + kTrainingTileSize - 1) /
            kTrainingTileSize;
        const int64_t expectedTileWidth =
            (static_cast<int64_t>(s.width) + kTrainingTileSize - 1) /
            kTrainingTileSize;
        if (!target.coverageMask ||
            !target.coverageRenderTiles->defined() ||
            !target.coverageRenderTiles->isGpu() ||
            target.coverageRenderTiles->dtype() != DType::UInt8 ||
            target.coverageRenderTiles->ndim() != 2 ||
            target.coverageRenderTiles->size(0) != expectedTileHeight ||
            target.coverageRenderTiles->size(1) != expectedTileWidth) {
            throw std::invalid_argument(
                "Coverage render tiles must be a GPU uint8 tile map matching "
                "the camera");
        }
    }
    MTensor& gt = *target.image;

    if (collectDensificationStats && !hasDensificationScratch()) {
        allocateDensificationScratch();
    }
    const bool validStats = xysGradNorm.defined() && visCounts.defined() &&
        max2DSize.defined() && xysGradNorm.numel() == numPoints &&
        visCounts.numel() == numPoints && max2DSize.numel() == numPoints;
    if (collectDensificationStats && !validStats) {
        MTensor newXysGradNorm = gpu_zeros({numPoints}, DType::Float32);
        MTensor newVisCounts = gpu_zeros({numPoints}, DType::Float32);
        MTensor newMax2DSize = gpu_zeros({numPoints}, DType::Float32);
        xysGradNorm = std::move(newXysGradNorm);
        visCounts = std::move(newVisCounts);
        max2DSize = std::move(newMax2DSize);
    }

    if (adam_step_count == std::numeric_limits<int>::max())
        throw std::overflow_error("Adam step count cannot be incremented further");
    const int previousAdamStep = adam_step_count;
    const int nextAdamStep = previousAdamStep + 1;
    float bc1 = 1.0f - std::pow(adam_beta1, nextAdamStep);
    float bc2 = 1.0f - std::pow(adam_beta2, nextAdamStep);
    MTensor adam_p[N_ADAM_GROUPS];
    MTensor adam_ea[N_ADAM_GROUPS], adam_eas[N_ADAM_GROUPS];
    float adam_ss[N_ADAM_GROUPS], adam_bc2s[N_ADAM_GROUPS];
    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities};
    for (int i = 0; i < N_ADAM_GROUPS; ++i) {
        adam_p[i] = *params[i];
        adam_ea[i] = adam_exp_avg[i];
        adam_eas[i] = adam_exp_avg_sq[i];
        adam_ss[i] = adam_lr[i] / bc1;
        adam_bc2s[i] = std::sqrt(bc2);
    }

    MsplatPhotometricRefinementStep photometric;
    const bool usesRgbGains =
        appearanceMode == msplat::AppearanceMode::RgbGains;
    photometric.enabled = usesRgbGains;
    photometric.cameraIndex = usesRgbGains
        ? static_cast<uint32_t>(cameraIndex)
        : 0u;
    photometric.logRgbGains = &cameraLogGains;
    uint32_t previousPhotometricStep = 0;
    uint32_t nextPhotometricStep = 0;
    if (usesRgbGains) {
        previousPhotometricStep = cameraLogGainStepCounts[cameraIndex];
        if (previousPhotometricStep == std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(
                "Photometric optimizer step count cannot be incremented further");
        }
        nextPhotometricStep = previousPhotometricStep + 1;
        const float photoBc1 = 1.0f -
            std::pow(adam_beta1, static_cast<float>(nextPhotometricStep));
        const float photoBc2 = 1.0f -
            std::pow(adam_beta2, static_cast<float>(nextPhotometricStep));
        photometric.expAvg = &cameraLogGainExpAvg;
        photometric.expAvgSq = &cameraLogGainExpAvgSq;
        photometric.adamStepSize = kPhotometricLearningRate / photoBc1;
        photometric.adamBiasCorrection2Sqrt = std::sqrt(photoBc2);
        photometric.regularization = kPhotometricRegularization;
        photometric.maxAbsLogGain = kPhotometricMaxAbsLogGain;
    }

    MsplatPpispRefinementStep ppisp;
    const bool usesPpisp = appearanceMode == msplat::AppearanceMode::PPISP;
    ppisp.enabled = usesPpisp;
    ppisp.frameIndex = usesPpisp
        ? static_cast<uint32_t>(cameraIndex)
        : 0u;
    ppisp.parameters = &cameraPpispParameters;
    uint32_t nextPpispStep = 0;
    if (usesPpisp) {
        const uint32_t previousPpispStep =
            cameraPpispStepCounts[cameraIndex];
        if (previousPpispStep == std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(
                "PPISP optimizer step count cannot be incremented further");
        }
        nextPpispStep = previousPpispStep + 1;
        const float ppispBc1 = 1.0f -
            std::pow(adam_beta1, static_cast<float>(nextPpispStep));
        const float ppispBc2 = 1.0f -
            std::pow(adam_beta2, static_cast<float>(nextPpispStep));
        ppisp.expAvg = &cameraPpispExpAvg;
        ppisp.expAvgSq = &cameraPpispExpAvgSq;
        ppisp.adamStepSize = kPpispLearningRate / ppispBc1;
        ppisp.adamBiasCorrection2Sqrt = std::sqrt(ppispBc2);
        ppisp.regularization = kPpispRegularization;
        ppisp.maxAbsExposure = kPpispMaxAbsExposure;
        ppisp.maxAbsColorParameter = kPpispMaxAbsColorParameter;
    }

    const bool poseStepEnabled = refineCameraPoses &&
        step > warmupLength &&
        cameraIndex != static_cast<size_t>(poseAnchorCameraIndex);
    MsplatPoseRefinementStep pose;
    pose.enabled = poseStepEnabled;
    pose.conditioned = poseStepEnabled &&
        cameraPoseConditioning == msplat::CameraPoseConditioning::CamP;
    pose.cameraIndex = poseStepEnabled
        ? static_cast<uint32_t>(cameraIndex)
        : 0u;
    pose.deltas = &cameraPoseDeltas;
    pose.preconditioners = pose.conditioned
        ? &cameraPosePreconditioners
        : nullptr;
    uint32_t previousPoseStep = 0;
    uint32_t nextPoseStep = 0;
    if (poseStepEnabled) {
        previousPoseStep = cameraPoseStepCounts[cameraIndex];
        if (previousPoseStep == std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(
                "Camera pose optimizer step count cannot be incremented further");
        }
        nextPoseStep = previousPoseStep + 1;
        const float poseBc1 = 1.0f -
            std::pow(adam_beta1, static_cast<float>(nextPoseStep));
        const float poseBc2 = 1.0f -
            std::pow(adam_beta2, static_cast<float>(nextPoseStep));
        pose.expAvg = &cameraPoseExpAvg;
        pose.expAvgSq = &cameraPoseExpAvgSq;
        pose.adamStepSize = kPoseLearningRate / poseBc1;
        pose.adamBiasCorrection2Sqrt = std::sqrt(poseBc2);
        pose.regularization = kPoseRegularization;
        pose.maxTranslation = kPoseMaxTranslation;
        pose.maxRotation = kPoseMaxRotation;
    }

    const bool transparentMask =
        transparentTrainingMasks && target.coverageMask != nullptr;
    const MTensor *coverageRenderTiles =
        target.coverageMask && !transparentMask
            ? target.coverageRenderTiles
            : nullptr;
    const uint64_t objectiveCoverageUnits = transparentMask
        ? fullCoverageUnits
        : target.coverageUnits;
    float invMaxDim = 1.0f / static_cast<float>((std::max)(lastHeight, lastWidth));
    const float lossInvN = static_cast<float>(
        255.0 / (static_cast<double>(objectiveCoverageUnits) * 3.0));

    MTensor r = msplat_train_step(
        numPoints, means, scales, 1.0f,
        quats, cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, backgroundColor, gt, target.coverageMask,
        coverageRenderTiles,
        objectiveCoverageUnits, ssimWeight,
        lossInvN, transparentMask, transparentAlphaLossWeight,
        N_ADAM_GROUPS,
        adam_p, adam_ea, adam_eas,
        adam_ss, adam_bc2s,
        adam_beta1, adam_beta2, adam_eps,
        photometric,
        ppisp,
        pose,
        collectDensificationStats,
        visCounts, xysGradNorm, max2DSize, invMaxDim);

    // Keep host optimizer state transactional with respect to synchronous
    // encoding failures. The candidate counters above drive this step's bias
    // correction, but become persistent only once the Metal step was accepted.
    adam_step_count = nextAdamStep;
    if (usesRgbGains)
        cameraLogGainStepCounts[cameraIndex] = nextPhotometricStep;
    if (usesPpisp)
        cameraPpispStepCounts[cameraIndex] = nextPpispStep;
    if (poseStepEnabled)
        cameraPoseStepCounts[cameraIndex] = nextPoseStep;

    if (collectDensificationStats) radii = r;
}
