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
#include <vector>
#include "model.hpp"
#include "atomic_output.hpp"
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
    scratch.randomSamples = gpu_zeros({capacity, 3}, DType::Float32);
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

// Metrics on CPU MTensor data
float psnr(const MTensor& rendered, const MTensor& gt) {
    int64_t n = rendered.numel();
    const float *r = rendered.data<float>(), *g = gt.data<float>();
    double mse = 0;
    for (int64_t i = 0; i < n; i++) { double d = r[i] - g[i]; mse += d * d; }
    mse /= n;
    return 10.0f * std::log10(1.0 / mse);
}

float l1_loss(const MTensor& rendered, const MTensor& gt) {
    int64_t n = rendered.numel();
    const float *r = rendered.data<float>(), *g = gt.data<float>();
    double sum = 0;
    for (int64_t i = 0; i < n; i++) sum += std::abs(r[i] - g[i]);
    return (float)(sum / n);
}

// Model constructor
Model::Model(const InputData &inputData, int numCameras,
    int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
    int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
    int maxSteps, bool keepCrs,
    const float* bgColor,
    int stopDensifyAt,
    int maxGaussians)
    : numCameras(numCameras), numDownscales(numDownscales), resolutionSchedule(resolutionSchedule),
      shDegree(shDegree), configuredSHDegree(shDegree),
      shDegreeInterval(shDegreeInterval),
      refineEvery(refineEvery), warmupLength(warmupLength), resetAlphaEvery(resetAlphaEvery),
      stopSplitAt(stopDensifyAt >= 0 ? stopDensifyAt : maxSteps / 2), densifyGradThresh(densifyGradThresh), densifySizeThresh(densifySizeThresh),
      stopScreenSizeAt(stopScreenSizeAt), splitScreenSize(splitScreenSize),
      maxSteps(maxSteps), maxGaussians(maxGaussians), keepCrs(keepCrs) {

    if (inputData.points.count <= 0)
        throw std::invalid_argument("Dataset must contain sparse points");
    if (maxGaussians != -1 && maxGaussians <= 0)
        throw std::invalid_argument("maxGaussians must be -1 or greater than zero");
    if (inputData.points.count > std::numeric_limits<int>::max())
        throw std::invalid_argument("Dataset contains too many sparse points");
    if (maxGaussians > 0 && inputData.points.count > maxGaussians)
        throw std::invalid_argument("maxGaussians is below the initial Gaussian count");
    if (numCameras <= 0)
        throw std::invalid_argument("Dataset must contain training cameras");
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
    if (bgColor) {
        for (int component = 0; component < 3; ++component) {
            if (!std::isfinite(bgColor[component]) || bgColor[component] < 0.0f ||
                bgColor[component] > 1.0f)
                throw std::invalid_argument("Background components must be finite and in 0...1");
        }
    }

    int64_t numPoints = inputData.points.count;
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

size_t Model::estimatedGpuBytes() const {
    size_t bytes = 0;
    const MTensor* tensors[] = {
        &means_buf, &scales_buf, &quats_buf, &featuresDc_buf, &featuresRest_buf, &opacities_buf,
        &densify_split_flag, &densify_dup_flag, &densify_split_prefix, &densify_dup_prefix,
        &densify_keep_flag, &densify_keep_prefix, &densify_block_totals,
        &densify_compact_scratch, &densify_random_samples,
        &radii, &xysGradNorm, &visCounts, &max2DSize, &backgroundColor
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
        std::vector<int64_t> shape;
        bool preserveActive;
    };
    std::vector<ResizeTask> tasks;
    auto addLeadingCapacityTask = [&](MTensor &tensor, bool preserveActive) {
        auto shape = tensor.shape();
        shape[0] = new_cap;
        tasks.push_back({&tensor, std::move(shape), preserveActive});
    };

    MTensor* parameterBuffers[] = {
        &means_buf, &scales_buf, &quats_buf,
        &featuresDc_buf, &featuresRest_buf, &opacities_buf
    };
    for (MTensor* tensor : parameterBuffers) addLeadingCapacityTask(*tensor, true);
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        addLeadingCapacityTask(adam_exp_avg_buf[g], true);
        addLeadingCapacityTask(adam_exp_avg_sq_buf[g], true);
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
    tasks.push_back({&densify_block_totals, {max_blocks}, false});
    tasks.push_back({&densify_compact_scratch,
                     {static_cast<int64_t>(new_cap) * compact_stride}, false});
    tasks.push_back({&densify_random_samples, {new_cap, 3}, false});

    // Replacing the largest allocations first minimizes the final transient:
    // by the time most new buffers exist, only small old buffers remain.
    std::sort(tasks.begin(), tasks.end(), [](const ResizeTask &lhs, const ResizeTask &rhs) {
        return lhs.tensor->nbytes() > rhs.tensor->nbytes();
    });
    for (const ResizeTask &task : tasks) {
        MTensor replacement = gpu_zeros(task.shape, task.tensor->dtype());
        if (task.preserveActive) {
            size_t copy_bytes = static_cast<size_t>(num_active) *
                task.tensor->stride0() * task.tensor->elementSize();
            memcpy(replacement.data_ptr(), task.tensor->data_ptr(), copy_bytes);
        }
        *task.tensor = std::move(replacement);
    }

    buf_capacity = new_cap;
    refreshViews();
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
        bool doDensification = step < stopSplitAt &&
            step % resetInterval > numCameras + refineEvery;

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

            // Fill random samples for splits (CPU randn, shared memory)
            {
                std::mt19937 rng(step);
                std::normal_distribution<float> dist(0.0f, 1.0f);
                float *p = densify_random_samples.data<float>();
                for (int64_t i = 0; i < 2LL * numSplits * 3; i++) p[i] = dist(rng);
            }

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
                densify_random_samples
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
            msplat_gpu_sync();
            constexpr float resetLogit = -1.3862943611198906f;
            float *op = opacities.data<float>();
            for (int64_t i = 0; i < opacities.numel(); i++)
                if (op[i] > resetLogit) op[i] = resetLogit;

            adam_exp_avg[5].zero();
            adam_exp_avg_sq[5].zero();
            fprintf(stderr, "Opacity reset at step %d\n", step);
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
static constexpr uint32_t CKPT_VERSION = 1;

static_assert(sizeof(float) == 4, "Checkpoint v1 requires 32-bit floats");

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

    // Checkpoint v1 stores Float32 tensors without a separate dtype tag. An
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
};

ParsedCheckpoint parseCheckpointMetadata(CheckpointReader &reader) {
    const uint32_t magic = reader.scalar<uint32_t>("magic");
    const uint32_t version = reader.scalar<uint32_t>("version");
    if (magic != CKPT_MAGIC)
        throw std::runtime_error("Not a valid msplat checkpoint file");
    if (version != CKPT_VERSION)
        throw std::runtime_error("Unsupported checkpoint version: " +
                                 std::to_string(version));

    ParsedCheckpoint checkpoint;
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

} // namespace

void validateCheckpointFile(const std::string &filename) {
    CheckpointReader reader(filename);
    (void)parseCheckpointMetadata(reader);
}

void Model::saveCheckpoint(const std::string &filename, int step) {
    msplat_gpu_sync();

    msplat::detail::AtomicOutputFile output(filename);
    std::ofstream f(output.temporary(), std::ios::binary | std::ios::trunc);
    if (!f.is_open()) throw std::runtime_error("Cannot open checkpoint file for writing: " + filename);

    // Header
    f.write(reinterpret_cast<const char*>(&CKPT_MAGIC), sizeof(CKPT_MAGIC));
    f.write(reinterpret_cast<const char*>(&CKPT_VERSION), sizeof(CKPT_VERSION));

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

    if (!cam.cachedViewMat.defined() || cam.cachedFovX != fovX || cam.cachedFovY != fovY) {
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

void Model::fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight){
    const bool collectDensificationStats =
        collectsDensificationStats(step, stopSplitAt);
    if (!collectDensificationStats) {
        retireDensificationState();
    }

    auto s = prepareCam(cam, step);
    lastHeight = s.height; lastWidth = s.width;
    int numPoints = means.size(0);

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
    adam_step_count++;
    float bc1 = 1.0f - std::pow(adam_beta1, adam_step_count);
    float bc2 = 1.0f - std::pow(adam_beta2, adam_step_count);
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

    float invMaxDim = 1.0f / static_cast<float>((std::max)(lastHeight, lastWidth));
    float lossInvN = 1.0f / (float)(s.height * s.width * 3);

    auto [r, loss] = msplat_train_step(
        numPoints, means, scales, 1.0f,
        quats, cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, backgroundColor, gt, ssimWeight,
        lossInvN,
        N_ADAM_GROUPS,
        adam_p, adam_ea, adam_eas,
        adam_ss, adam_bc2s,
        adam_beta1, adam_beta2, adam_eps,
        collectDensificationStats,
        visCounts, xysGradNorm, max2DSize, invMaxDim);

    if (collectDensificationStats) radii = r;
}
