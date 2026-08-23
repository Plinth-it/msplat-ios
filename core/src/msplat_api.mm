// ObjC++ implementation of the Swift-facing C++ API.
// This is the ONLY file that touches internal C++ types (Model, Camera, MTensor).

#include "msplat_api.hpp"

#include "model.hpp"
#include "input_data.hpp"
#include "msplat.hpp"
#include "ssim.hpp"
#include "memory_report.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <numeric>
#include <random>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

#include <TargetConditionals.h>

namespace msplat {

// ── Dataset::Impl ───────────────────────────────────────────────────────────

struct Dataset::Impl {
    InputData data;
    // Indices into data.cameras rather than copies of them. A Camera owns its
    // decoded Image, so the copies upstream made held a second full set.
    std::vector<size_t> trainIndices;
    std::vector<size_t> testIndices;
    CameraImageCache images;

    // Both load before returning: the correction from the dataset's declared
    // image size to the file's real one happens during the decode, so a camera
    // that has never been loaded carries the wrong intrinsics.
    Camera& trainCamera(size_t trainIndex) {
        return images.ensureLoaded(data.cameras, trainIndices[trainIndex]);
    }

    Camera& testCamera(size_t testIndex) {
        return images.ensureLoaded(data.cameras, testIndices[testIndex]);
    }

    MTensor& gpuImageForTrainCamera(size_t trainIndex, int downscaleFactor) {
        return images.gpuImage(data.cameras, trainIndices[trainIndex], downscaleFactor);
    }

    MTensor& gpuImageForTestCamera(size_t testIndex, int downscaleFactor) {
        return images.gpuImage(data.cameras, testIndices[testIndex], downscaleFactor);
    }
};

Dataset::Dataset(const std::string& path, float downscaleFactor,
                 bool evalMode, int testEvery)
    : impl(std::make_unique<Impl>())
{
    impl->data = inputDataFromX(path);
    if (impl->data.cameras.empty())
        throw std::runtime_error("Dataset has no cameras: " + path);
    if (impl->data.points.count <= 0)
        throw std::runtime_error("Dataset has no sparse points: " + path);
    impl->images = CameraImageCache(downscaleFactor, CameraImageCache::defaultBudgetBytes());

    // No image is decoded here. The first step that needs a camera loads it.
    if (evalMode) {
        auto [train, test] = impl->data.splitTrainTestIndices(testEvery);
        impl->trainIndices = train;
        impl->testIndices = test;
    } else {
        auto [train, valIdx] = impl->data.trainIndices(false);
        impl->trainIndices = train;
        (void)valIdx;
    }
}

Dataset::~Dataset() = default;
Dataset::Dataset(Dataset&&) noexcept = default;
Dataset& Dataset::operator=(Dataset&&) noexcept = default;

int Dataset::numTrain() const { return (int)impl->trainIndices.size(); }
int Dataset::numTest() const { return (int)impl->testIndices.size(); }
void Dataset::cameraPose(int index, float camToWorld[16]) const {
    if (index >= 0 && index < (int)impl->trainIndices.size())
        memcpy(camToWorld, impl->data.cameras[impl->trainIndices[index]].camToWorld,
               16 * sizeof(float));
}
void* Dataset::_handle() const { return impl.get(); }

namespace {

// The Metal engine and tensor cache are process-global. Keep each public C++
// trainer operation atomic so a second trainer cannot reuse shared outputs
// between render submission, synchronization, and readback.
std::mutex g_trainerTransactionMutex;

float* allocatePixels(int width, int height) {
    if (width <= 0 || height <= 0)
        throw std::runtime_error("Renderer returned invalid image dimensions");

    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    if (w > std::numeric_limits<size_t>::max() / h / 3 / sizeof(float))
        throw std::bad_alloc();

    auto* pixels = static_cast<float*>(std::malloc(w * h * 3 * sizeof(float)));
    if (!pixels) throw std::bad_alloc();
    return pixels;
}

size_t rgbaCapacityRequired(int width, int height) {
    if (width <= 0 || height <= 0)
        throw std::runtime_error("Renderer returned invalid image dimensions");

    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    if (w > std::numeric_limits<size_t>::max() / h / 4)
        throw std::bad_alloc();
    return w * h * 4;
}

} // namespace

// ── Trainer::Impl ───────────────────────────────────────────────────────────

struct Trainer::Impl {
    std::unique_ptr<Model> model;
    Config config;
    Dataset::Impl* ds = nullptr;
    int currentStep = 0;
    MsplatTrainingTelemetryHandle telemetry = msplat_training_telemetry_create();

    // Camera iteration
    std::vector<size_t> camIndices;
    size_t camIterPos = 0;
    std::mt19937 rng{42};

    void shuffleCameras() {
        std::shuffle(camIndices.begin(), camIndices.end(), rng);
        camIterPos = 0;
    }

    size_t currentCamera() {
        if (camIterPos >= camIndices.size()) shuffleCameras();
        return camIndices[camIterPos];
    }

    void advanceCamera() {
        ++camIterPos;
    }
};

Trainer::Trainer(Dataset& dataset, const Config& config)
    : impl(std::make_unique<Impl>())
{
    std::lock_guard lock(g_trainerTransactionMutex);
    if (config.maxGaussians != -1 && config.maxGaussians <= 0)
        throw std::invalid_argument("maxGaussians must be -1 or greater than zero");
    impl->config = config;
    impl->ds = static_cast<Dataset::Impl*>(dataset._handle());

    impl->model = std::make_unique<Model>(
        impl->ds->data,
        (int)impl->ds->trainIndices.size(),
        config.numDownscales, config.resolutionSchedule,
        config.shDegree, config.shDegreeInterval,
        config.refineEvery, config.warmupLength, config.resetAlphaEvery,
        config.densifyGradThresh, config.densifySizeThresh,
        config.stopScreenSizeAt, config.splitScreenSize,
        config.iterations, config.keepCrs,
        config.bgColor,
        config.stopDensifyAt,
        config.maxGaussians
    );

    impl->camIndices.resize(impl->ds->trainIndices.size());
    std::iota(impl->camIndices.begin(), impl->camIndices.end(), 0);
    impl->shuffleCameras();
}

Trainer::~Trainer() {
    std::lock_guard lock(g_trainerTransactionMutex);
    impl.reset();
}

Stats Trainer::step() {
    std::lock_guard lock(g_trainerTransactionMutex);
    if (impl->currentStep == std::numeric_limits<int>::max())
        throw std::overflow_error("Training iteration cannot be incremented further");
    const int nextStep = impl->currentStep + 1;
    auto logicalStep = msplat_training_step_begin(
        impl->telemetry, nextStep);
    double cpuSubmitMs = 0.0;
    try {
        size_t camIdx = impl->currentCamera();
        Camera& cam = impl->ds->trainCamera(camIdx);

        int ds = impl->model->getDownscaleFactor(nextStep);
        MTensor& gt = impl->ds->gpuImageForTrainCamera(camIdx, ds);

        msplat_training_step_mark_cpu_start(logicalStep);
        impl->model->fullIteration(
            cam, nextStep, gt, impl->config.ssimWeight);
        impl->model->schedulersStep(nextStep);
        impl->model->afterTrain(nextStep);

        MsplatTrainingStepDescriptor descriptor;
        descriptor.iteration = nextStep;
        descriptor.splatCount = impl->model->means.size(0);
        descriptor.modelCapacity = impl->model->buf_capacity;
        descriptor.effectiveWidth = static_cast<int32_t>(gt.size(1));
        descriptor.effectiveHeight = static_cast<int32_t>(gt.size(0));
        descriptor.activeShDegree = std::min(
            nextStep / impl->config.shDegreeInterval,
            impl->config.shDegree);
        cpuSubmitMs = msplat_training_step_submit(logicalStep, descriptor);
    } catch (...) {
        msplat_training_step_abort(logicalStep);
        throw;
    }
    impl->advanceCamera();
    impl->currentStep = nextStep;

    reportMemory(impl->currentStep, (int)impl->model->means.size(0),
                 impl->model->estimatedGpuBytes(),
                 impl->ds->images.cachedBytes(),
                 impl->ds->images.budgetBytes());

    Stats s;
    s.iteration = impl->currentStep;
    s.splatCount = (int)impl->model->means.size(0);
    s.msPerStep = static_cast<float>(cpuSubmitMs);
    return s;
}

void Trainer::train(int callbackEvery) {
    while (impl->currentStep < impl->config.iterations) {
        step();
        // Note: callbacks handled at the Swift level via polling iteration()
        // to keep the C++ API free of function pointer complexity
    }
}

EvalMetrics Trainer::evaluate() {
    std::lock_guard lock(g_trainerTransactionMutex);
    auto& testIndices = impl->ds->testIndices;
    if (testIndices.empty())
        return {};

    double sumPsnr = 0, sumSsim = 0, sumL1 = 0;
    int n = (int)testIndices.size();

    for (int i = 0; i < n; i++) {
        Camera& cam = impl->ds->testCamera(i);
        MTensor rgb = impl->model->render(cam, impl->config.iterations);
        msplat_gpu_sync();
        MTensor rgbCpu = rgb.cpu();
        int dsf = impl->model->getDownscaleFactor(impl->config.iterations);
        MTensor gtCpu = impl->ds->gpuImageForTestCamera(i, dsf).cpu();

        sumPsnr += psnr(rgbCpu, gtCpu);
        sumSsim += ssim_eval(rgbCpu, gtCpu);
        sumL1 += l1_loss(rgbCpu, gtCpu);
    }

    EvalMetrics m;
    m.psnr = (float)(sumPsnr / n);
    m.ssim = (float)(sumSsim / n);
    m.l1 = (float)(sumL1 / n);
    m.numTest = n;
    m.numGaussians = (int)impl->model->means.size(0);
    return m;
}

PixelBuffer Trainer::render(int cameraIndex, bool useTest) {
    std::lock_guard lock(g_trainerTransactionMutex);
    auto& indices = useTest ? impl->ds->testIndices : impl->ds->trainIndices;
    if (cameraIndex < 0 || cameraIndex >= (int)indices.size())
        return {};

    Camera& cam = impl->ds->images.ensureLoaded(impl->ds->data.cameras, indices[cameraIndex]);
    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();
    MTensor rgbCpu = rgb.cpu();

    int h = (int)rgbCpu.size(0);
    int w = (int)rgbCpu.size(1);
    // Use malloc so callers can free() — PixelBuffer destructor handles both
    float* buf = allocatePixels(w, h);
    memcpy(buf, rgbCpu.data_ptr(), static_cast<size_t>(h) * w * 3 * sizeof(float));

    return PixelBuffer(buf, w, h);
}

PixelBuffer Trainer::renderFromPose(const float camToWorld[16], int refCameraIndex) {
    std::lock_guard lock(g_trainerTransactionMutex);
    auto& indices = impl->ds->trainIndices;
    if (refCameraIndex < 0 || refCameraIndex >= (int)indices.size())
        return {};

    Camera cam = impl->ds->images.ensureLoaded(impl->ds->data.cameras, indices[refCameraIndex]);
    memcpy(cam.camToWorld, camToWorld, 16 * sizeof(float));
    // Invalidate cached matrices so prepareCam recomputes from the new pose
    cam.cachedViewMat = MTensor();
    cam.cachedProjViewMat = MTensor();

    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();
    MTensor rgbCpu = rgb.cpu();

    int h = (int)rgbCpu.size(0);
    int w = (int)rgbCpu.size(1);
    float* buf = allocatePixels(w, h);
    memcpy(buf, rgbCpu.data_ptr(), static_cast<size_t>(h) * w * 3 * sizeof(float));
    return PixelBuffer(buf, w, h);
}

void Trainer::renderFromPoseToBuffer(const float camToWorld[16], int refCameraIndex,
                                  uint8_t* outRGBA,
                                  int* outWidth, int* outHeight) {
    renderFromPoseToBuffer(
        camToWorld, refCameraIndex, outRGBA,
        outRGBA ? std::numeric_limits<size_t>::max() : 0,
        outWidth, outHeight);
}

void Trainer::renderFromPoseToBuffer(const float camToWorld[16], int refCameraIndex,
                                  uint8_t* outRGBA, size_t outCapacity,
                                  int* outWidth, int* outHeight) {
    std::lock_guard lock(g_trainerTransactionMutex);
    auto& indices = impl->ds->trainIndices;
    if (refCameraIndex < 0 || refCameraIndex >= (int)indices.size()) {
        *outWidth = 0; *outHeight = 0; return;
    }

    Camera cam = impl->ds->images.ensureLoaded(impl->ds->data.cameras, indices[refCameraIndex]);
    const int downscale = impl->model->getDownscaleFactor(impl->currentStep);
    if (cam.width < downscale || cam.height < downscale)
        throw std::invalid_argument(
            "Training downscale produces a zero-sized image; reduce numDownscales");
    const int expectedWidth = cam.width / downscale;
    const int expectedHeight = cam.height / downscale;
    *outWidth = expectedWidth;
    *outHeight = expectedHeight;
    if (!outRGBA) return;

    const size_t required = rgbaCapacityRequired(expectedWidth, expectedHeight);
    if (outCapacity < required)
        throw std::invalid_argument("RGBA output buffer is too small");

    memcpy(cam.camToWorld, camToWorld, 16 * sizeof(float));
    cam.cachedViewMat = MTensor();
    cam.cachedProjViewMat = MTensor();

    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();

    int h = (int)rgb.size(0), w = (int)rgb.size(1);
    if (w != expectedWidth || h != expectedHeight)
        throw std::runtime_error("Rendered image dimensions changed unexpectedly");

    // Read directly from GPU tensor (unified memory on Apple Silicon)
    const float* src = (const float*)rgb.data_ptr();
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    for (size_t i = 0; i < n; i++) {
        outRGBA[i * 4]     = (uint8_t)(fminf(fmaxf(src[i*3],   0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 1] = (uint8_t)(fminf(fmaxf(src[i*3+1], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 2] = (uint8_t)(fminf(fmaxf(src[i*3+2], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 3] = 255;
    }
}

void Trainer::exportPly(const std::string& path) {
    std::lock_guard lock(g_trainerTransactionMutex);
    impl->model->savePly(path, impl->currentStep);
}

void Trainer::exportSplat(const std::string& path) {
    std::lock_guard lock(g_trainerTransactionMutex);
    impl->model->saveSplat(path);
}

void Trainer::exportSpz(const std::string& path) {
    std::lock_guard lock(g_trainerTransactionMutex);
    impl->model->saveSpz(path);
}

void Trainer::saveCheckpoint(const std::string& path) {
    std::lock_guard lock(g_trainerTransactionMutex);
    impl->model->saveCheckpoint(path, impl->currentStep);
}

int Trainer::loadCheckpoint(const std::string& path) {
    std::lock_guard lock(g_trainerTransactionMutex);
    impl->currentStep = impl->model->loadCheckpoint(path);
    msplat_training_telemetry_reset(impl->telemetry);
    // Re-shuffle cameras for resumed training
    impl->shuffleCameras();
    return impl->currentStep;
}

int Trainer::splatCount() const {
    std::lock_guard lock(g_trainerTransactionMutex);
    return (int)impl->model->means.size(0);
}

int Trainer::iteration() const {
    std::lock_guard lock(g_trainerTransactionMutex);
    return impl->currentStep;
}

TrainingMetrics Trainer::metrics() const {
    std::lock_guard lock(g_trainerTransactionMutex);
    const MsplatTrainingTelemetrySnapshot snapshot =
        msplat_training_telemetry_snapshot(impl->telemetry);
    TrainingMetrics metrics;
    metrics.hasSubmittedStep =
        snapshot.flags & MSPLAT_TRAINING_TELEMETRY_HAS_SUBMITTED;
    metrics.hasCompletedStep =
        snapshot.flags & MSPLAT_TRAINING_TELEMETRY_HAS_COMPLETED;
    metrics.gpuTimeValid =
        snapshot.flags & MSPLAT_TRAINING_TELEMETRY_GPU_TIMING_VALID;
    metrics.lossValid = snapshot.flags & MSPLAT_TRAINING_TELEMETRY_LOSS_VALID;
    metrics.intersectionsValid =
        snapshot.flags & MSPLAT_TRAINING_TELEMETRY_INTERSECTION_COUNT_VALID;
    metrics.hasFailedStep = snapshot.flags & MSPLAT_TRAINING_TELEMETRY_HAS_FAILED;

    auto copyDescriptor = [](const MsplatTrainingStepDescriptor& source,
                             SubmittedTrainingStep& destination) {
        destination.iteration = static_cast<int>(source.iteration);
        destination.splatCount = static_cast<int>(source.splatCount);
        destination.modelCapacity = static_cast<int>(source.modelCapacity);
        destination.effectiveWidth = source.effectiveWidth;
        destination.effectiveHeight = source.effectiveHeight;
        destination.activeSHDegree = source.activeShDegree;
    };
    copyDescriptor(snapshot.submittedStep, metrics.submitted);
    metrics.submitted.cpuSubmitMs =
        static_cast<float>(snapshot.submittedCpuSubmitMs);
    copyDescriptor(snapshot.completedStep.step, metrics.completed);
    metrics.completed.cpuSubmitMs =
        static_cast<float>(snapshot.completedStep.cpuSubmitMs);
    metrics.completed.gpuExecutionMs =
        static_cast<float>(snapshot.completedStep.gpuExecutionMs);
    metrics.completed.endToEndMs =
        static_cast<float>(snapshot.completedStep.endToEndMs);
    metrics.completed.loss = static_cast<float>(snapshot.completedStep.loss);
    metrics.completed.overflowKinds = snapshot.completedStep.overflowReasons;
    metrics.completed.retainedPackedIntersectionCount =
        snapshot.completedStep.retainedPackedIntersections;
    metrics.completed.packedIntersectionCapacity =
        snapshot.completedStep.packedIntersectionCapacity;
    metrics.overflowedCompletedSteps = snapshot.overflowedStepCount;
    metrics.tileCapOverflowedSteps = snapshot.tileCapOverflowedStepCount;
    metrics.packedCapacityOverflowedSteps =
        snapshot.packedCapacityOverflowedStepCount;
    metrics.lastOverflowIteration =
        static_cast<int>(snapshot.lastOverflowIteration);
    metrics.lastFailedIteration =
        static_cast<int>(snapshot.lastFailedIteration);
    return metrics;
}

TrainingMemoryMetrics Trainer::memoryMetrics() const {
    std::lock_guard lock(g_trainerTransactionMutex);
    TrainingMemoryMetrics metrics;
    metrics.trainerModelBufferBytes = impl->model->estimatedGpuBytes();
    metrics.engineSharedTransientBufferBytes =
        msplat_shared_cached_tensor_bytes();
    metrics.engineTrainingTransientBufferBytes =
        msplat_training_cached_tensor_bytes();
    metrics.trainerTelemetryReadbackBytes =
        msplat_training_telemetry_readback_bytes(impl->telemetry);
    metrics.trainerImageCacheCpuBytes = impl->ds->images.cachedCpuBytes();
    metrics.trainerImageCacheGpuBytes = impl->ds->images.cachedGpuBytes();
    metrics.trainerImageCacheBudgetBytes = impl->ds->images.budgetBytes();
    metrics.trainingGpuImageCacheHits = impl->ds->images.hitCount();
    metrics.trainingGpuImageCacheMisses = impl->ds->images.missCount();
    const ProcessMemorySnapshot processMemory = currentProcessMemory();
    metrics.processPhysFootprintBytes = processMemory.physicalFootprintBytes;
    metrics.processAvailableBytes = processMemory.availableBytes;
    metrics.hasProcessPhysFootprint = processMemory.hasPhysicalFootprint;
    metrics.hasProcessAvailableBytes = processMemory.hasAvailableBytes;
    return metrics;
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

void sync() {
    std::lock_guard lock(g_trainerTransactionMutex);
    msplat_gpu_sync();
}
void cleanup() {
    std::lock_guard lock(g_trainerTransactionMutex);
    cleanup_msplat_metal();
}

} // namespace msplat

// ── C API (for Swift interop) ───────────────────────────────────────────────

#include "msplat_c_api.h"

namespace {

thread_local MsplatErrorInfo g_lastError = {MSPLAT_STATUS_OK, {0}};
std::mutex g_cApiMutex;

// C handles own small wrappers rather than exposing the C++ object address.
// A trainer shares its dataset ownership, so destroying the public dataset
// handle cannot invalidate the native pointer retained by Trainer::Impl.
struct CApiDatasetHandle {
    std::shared_ptr<msplat::Dataset> dataset;
};

struct CApiTrainerHandle {
    std::shared_ptr<msplat::Dataset> dataset;
    std::unique_ptr<msplat::Trainer> trainer;
};

CApiDatasetHandle& datasetHandle(MsplatDataset handle) noexcept {
    return *static_cast<CApiDatasetHandle*>(handle);
}

CApiTrainerHandle& trainerHandle(MsplatTrainer handle) noexcept {
    return *static_cast<CApiTrainerHandle*>(handle);
}

void storeError(MsplatStatus status, const char* message,
                MsplatErrorInfo* outError) noexcept {
    g_lastError.status = status;
    std::snprintf(g_lastError.message, sizeof(g_lastError.message), "%s",
                  message ? message : "");
    if (outError) *outError = g_lastError;
}

void clearError(MsplatErrorInfo* outError) noexcept {
    storeError(MSPLAT_STATUS_OK, "", outError);
}

MsplatStatus classifyException(const std::exception& exception,
                               MsplatStatus fallback) noexcept {
    const char* message = exception.what();
    if (message && std::strstr(message, "MTLBuffer allocation failed"))
        return MSPLAT_STATUS_OUT_OF_MEMORY;
    if (message && (std::strstr(message, "Metal") ||
                    std::strstr(message, "metallib") ||
                    std::strstr(message, "GPU")))
        return MSPLAT_STATUS_GPU_ERROR;
    return fallback;
}

template <typename Fn>
MsplatStatus guarded(MsplatErrorInfo* outError, MsplatStatus fallback,
                     Fn&& operation) noexcept {
    @try {
        try {
            // Serialize complete C API transactions.  Acquiring a mutex can
            // itself throw, so keep it inside this exception boundary too.
            std::lock_guard<std::mutex> lock(g_cApiMutex);
            clearError(outError);
            operation();
            return MSPLAT_STATUS_OK;
        } catch (const std::invalid_argument& exception) {
            storeError(MSPLAT_STATUS_INVALID_ARGUMENT, exception.what(), outError);
            return MSPLAT_STATUS_INVALID_ARGUMENT;
        } catch (const std::bad_alloc&) {
            storeError(MSPLAT_STATUS_OUT_OF_MEMORY, "Memory allocation failed", outError);
            return MSPLAT_STATUS_OUT_OF_MEMORY;
        } catch (const std::exception& exception) {
            MsplatStatus status = classifyException(exception, fallback);
            storeError(status, exception.what(), outError);
            return status;
        } catch (...) {
            storeError(MSPLAT_STATUS_INTERNAL_ERROR, "Unknown native exception", outError);
            return MSPLAT_STATUS_INTERNAL_ERROR;
        }
    } @catch (NSException* exception) {
        const char* message = exception.reason.UTF8String;
        storeError(MSPLAT_STATUS_INTERNAL_ERROR,
                   message ? message : "Objective-C exception", outError);
        return MSPLAT_STATUS_INTERNAL_ERROR;
    }
}

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

void requirePath(const char* path) {
    require(path && path[0] != '\0', "Path must not be null or empty");
}

void requirePose(const float* pose) {
    require(pose != nullptr, "Camera pose must not be null");
    for (int i = 0; i < 16; ++i)
        require(std::isfinite(pose[i]), "Camera pose must contain only finite values");
}

void validateConfig(const MsplatConfig& c) {
    require(c.iterations > 0 && c.iterations <= 1000000,
            "iterations must be in 1...1000000");
    require(c.shDegree >= 0 && c.shDegree <= 4,
            "shDegree must be in 0...4");
    require(c.shDegreeInterval > 0, "shDegreeInterval must be greater than zero");
    require(std::isfinite(c.ssimWeight) && c.ssimWeight >= 0.0f && c.ssimWeight <= 1.0f,
            "ssimWeight must be finite and in 0...1");
    require(c.numDownscales >= 0 && c.numDownscales <= 30,
            "numDownscales must be in 0...30");
    require(c.resolutionSchedule > 0, "resolutionSchedule must be greater than zero");
    require(c.refineEvery > 0, "refineEvery must be greater than zero");
    require(c.warmupLength >= 0, "warmupLength must not be negative");
    require(c.resetAlphaEvery > 0, "resetAlphaEvery must be greater than zero");
    require(c.resetAlphaEvery <= std::numeric_limits<int>::max() / c.refineEvery,
            "resetAlphaEvery * refineEvery is too large");
    require(std::isfinite(c.densifyGradThresh) && c.densifyGradThresh >= 0.0f,
            "densifyGradThresh must be finite and non-negative");
    require(std::isfinite(c.densifySizeThresh) && c.densifySizeThresh >= 0.0f,
            "densifySizeThresh must be finite and non-negative");
    require(c.stopScreenSizeAt >= 0, "stopScreenSizeAt must not be negative");
    require(c.stopDensifyAt >= -1, "stopDensifyAt must be -1 or non-negative");
    require(std::isfinite(c.splitScreenSize) && c.splitScreenSize >= 0.0f,
            "splitScreenSize must be finite and non-negative");
    require(std::isfinite(c.downscaleFactor) && c.downscaleFactor >= 1.0f &&
                c.downscaleFactor <= 32.0f,
            "downscaleFactor must be finite and in 1...32");
    for (float component : c.bgColor)
        require(std::isfinite(component) && component >= 0.0f && component <= 1.0f,
                "bgColor components must be finite and in 0...1");
}

void validateTrainingLimits(const MsplatTrainingLimits& limits) {
    require(limits.maxGaussians == -1 || limits.maxGaussians > 0,
            "maxGaussians must be -1 or greater than zero");
}

msplat::Config configFromC(const MsplatConfig& c) {
    msplat::Config cfg;
    cfg.iterations = c.iterations;
    cfg.shDegree = c.shDegree;
    cfg.shDegreeInterval = c.shDegreeInterval;
    cfg.ssimWeight = c.ssimWeight;
    cfg.numDownscales = c.numDownscales;
    cfg.resolutionSchedule = c.resolutionSchedule;
    cfg.refineEvery = c.refineEvery;
    cfg.warmupLength = c.warmupLength;
    cfg.resetAlphaEvery = c.resetAlphaEvery;
    cfg.densifyGradThresh = c.densifyGradThresh;
    cfg.densifySizeThresh = c.densifySizeThresh;
    cfg.stopScreenSizeAt = c.stopScreenSizeAt;
    cfg.stopDensifyAt = c.stopDensifyAt;
    cfg.splitScreenSize = c.splitScreenSize;
    cfg.keepCrs = c.keepCrs;
    cfg.downscaleFactor = c.downscaleFactor;
    memcpy(cfg.bgColor, c.bgColor, sizeof(cfg.bgColor));
    return cfg;
}

} // namespace

uint32_t msplat_abi_version(void) { return MSPLAT_ABI_VERSION; }

MsplatStatus msplat_last_status(void) { return g_lastError.status; }

const char* msplat_last_error_message(void) { return g_lastError.message; }

MsplatStatus msplat_dataset_create_v2(const char* path, float downscaleFactor,
                                      bool evalMode, int testEvery,
                                      MsplatDataset* outDataset,
                                      MsplatErrorInfo* error) {
    if (outDataset) *outDataset = nullptr;
    return guarded(error, MSPLAT_STATUS_INVALID_DATASET, [&] {
        require(outDataset != nullptr, "outDataset must not be null");
        requirePath(path);
        require(std::isfinite(downscaleFactor) && downscaleFactor >= 1.0f &&
                    downscaleFactor <= 32.0f,
                "downscaleFactor must be finite and in 1...32");
        require(testEvery > 0, "testEvery must be greater than zero");
        if (evalMode) require(testEvery >= 2, "testEvery must be at least 2 in eval mode");
        auto handle = std::make_unique<CApiDatasetHandle>();
        handle->dataset = std::make_shared<msplat::Dataset>(
            std::string(path), downscaleFactor, evalMode, testEvery);
        *outDataset = static_cast<MsplatDataset>(handle.release());
    });
}

MsplatStatus msplat_dataset_destroy_v2(MsplatDataset ds, MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(ds != nullptr, "Dataset handle must not be null");
        delete static_cast<CApiDatasetHandle*>(ds);
    });
}

MsplatStatus msplat_dataset_num_train_v2(MsplatDataset ds, int* outCount,
                                         MsplatErrorInfo* error) {
    if (outCount) *outCount = 0;
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(ds != nullptr, "Dataset handle must not be null");
        require(outCount != nullptr, "outCount must not be null");
        *outCount = datasetHandle(ds).dataset->numTrain();
    });
}

MsplatStatus msplat_dataset_num_test_v2(MsplatDataset ds, int* outCount,
                                        MsplatErrorInfo* error) {
    if (outCount) *outCount = 0;
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(ds != nullptr, "Dataset handle must not be null");
        require(outCount != nullptr, "outCount must not be null");
        *outCount = datasetHandle(ds).dataset->numTest();
    });
}

MsplatStatus msplat_dataset_camera_pose_v2(MsplatDataset ds, int cameraIndex,
                                           float camToWorld[16],
                                           MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(ds != nullptr, "Dataset handle must not be null");
        require(camToWorld != nullptr, "camToWorld must not be null");
        auto& dataset = *datasetHandle(ds).dataset;
        require(cameraIndex >= 0 && cameraIndex < dataset.numTrain(),
                "Training camera index is out of range");
        dataset.cameraPose(cameraIndex, camToWorld);
    });
}

MsplatStatus msplat_config_validate_v2(const MsplatConfig* config,
                                       size_t configSize,
                                       MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INVALID_ARGUMENT, [&] {
        require(config != nullptr, "Training config must not be null");
        require(configSize == sizeof(MsplatConfig),
                "Training config size does not match this msplat ABI");
        validateConfig(*config);
    });
}

MsplatStatus msplat_trainer_create_v2(MsplatDataset ds,
                                      const MsplatConfig* config,
                                      size_t configSize,
                                      MsplatTrainer* outTrainer,
                                      MsplatErrorInfo* error) {
    const MsplatTrainingLimits limits = msplat_default_training_limits();
    return msplat_trainer_create_v3(
        ds, config, configSize, &limits, sizeof(limits), outTrainer, error);
}

MsplatStatus msplat_training_limits_validate_v3(
    const MsplatTrainingLimits* limits, size_t limitsSize,
    MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INVALID_ARGUMENT, [&] {
        require(limits != nullptr, "Training limits must not be null");
        require(limitsSize == sizeof(MsplatTrainingLimits),
                "Training limits size does not match this msplat ABI");
        validateTrainingLimits(*limits);
    });
}

MsplatStatus msplat_trainer_create_v3(MsplatDataset ds,
                                      const MsplatConfig* config,
                                      size_t configSize,
                                      const MsplatTrainingLimits* limits,
                                      size_t limitsSize,
                                      MsplatTrainer* outTrainer,
                                      MsplatErrorInfo* error) {
    if (outTrainer) *outTrainer = nullptr;
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(ds != nullptr, "Dataset handle must not be null");
        require(config != nullptr, "Training config must not be null");
        require(configSize == sizeof(MsplatConfig),
                "Training config size does not match this msplat ABI");
        require(limits != nullptr, "Training limits must not be null");
        require(limitsSize == sizeof(MsplatTrainingLimits),
                "Training limits size does not match this msplat ABI");
        require(outTrainer != nullptr, "outTrainer must not be null");
        validateConfig(*config);
        validateTrainingLimits(*limits);
        auto dataset = datasetHandle(ds).dataset;
        require(dataset->numTrain() > 0, "Dataset has no training cameras");
        auto cfg = configFromC(*config);
        cfg.maxGaussians = limits->maxGaussians;
        auto handle = std::make_unique<CApiTrainerHandle>();
        handle->dataset = std::move(dataset);
        handle->trainer = std::make_unique<msplat::Trainer>(*handle->dataset, cfg);
        *outTrainer = static_cast<MsplatTrainer>(handle.release());
    });
}

MsplatStatus msplat_trainer_destroy_v2(MsplatTrainer t, MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        delete static_cast<CApiTrainerHandle*>(t);
    });
}

MsplatStatus msplat_trainer_step_v2(MsplatTrainer t, MsplatStats* outStats,
                                    MsplatErrorInfo* error) {
    if (outStats) *outStats = {};
    return guarded(error, MSPLAT_STATUS_GPU_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        require(outStats != nullptr, "outStats must not be null");
        auto stats = trainerHandle(t).trainer->step();
        *outStats = {stats.iteration, stats.splatCount, stats.msPerStep};
    });
}

MsplatStatus msplat_trainer_train_v2(MsplatTrainer t, MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_GPU_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        trainerHandle(t).trainer->train(0);
    });
}

MsplatStatus msplat_trainer_evaluate_v2(MsplatTrainer t,
                                        MsplatEvalMetrics* outMetrics,
                                        MsplatErrorInfo* error) {
    if (outMetrics) *outMetrics = {};
    return guarded(error, MSPLAT_STATUS_GPU_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        require(outMetrics != nullptr, "outMetrics must not be null");
        auto metrics = trainerHandle(t).trainer->evaluate();
        require(metrics.numTest > 0, "Dataset has no held-out test cameras");
        *outMetrics = {metrics.psnr, metrics.ssim, metrics.l1,
                       metrics.numTest, metrics.numGaussians};
    });
}

MsplatStatus msplat_trainer_render_v2(MsplatTrainer t, int cameraIndex,
                                      bool useTest, MsplatPixelBuffer* outBuffer,
                                      MsplatErrorInfo* error) {
    if (outBuffer) *outBuffer = {};
    return guarded(error, MSPLAT_STATUS_GPU_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        require(outBuffer != nullptr, "outBuffer must not be null");
        auto buffer = trainerHandle(t).trainer->render(cameraIndex, useTest);
        require(buffer.data != nullptr, "Camera index is out of range");
        *outBuffer = {buffer.data, buffer.width, buffer.height};
        buffer.data = nullptr;
    });
}

MsplatStatus msplat_trainer_render_pose_v2(MsplatTrainer t,
                                           const float camToWorld[16],
                                           int refCameraIndex,
                                           MsplatPixelBuffer* outBuffer,
                                           MsplatErrorInfo* error) {
    if (outBuffer) *outBuffer = {};
    return guarded(error, MSPLAT_STATUS_GPU_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        require(outBuffer != nullptr, "outBuffer must not be null");
        requirePose(camToWorld);
        auto buffer = trainerHandle(t).trainer->renderFromPose(
            camToWorld, refCameraIndex);
        require(buffer.data != nullptr, "Reference camera index is out of range");
        *outBuffer = {buffer.data, buffer.width, buffer.height};
        buffer.data = nullptr;
    });
}

MsplatStatus msplat_trainer_render_pose_to_buffer_v2(
    MsplatTrainer t, const float camToWorld[16], int refCameraIndex,
    uint8_t* outRGBA, size_t outCapacity, int* outWidth, int* outHeight,
    MsplatErrorInfo* error) {
    if (outWidth) *outWidth = 0;
    if (outHeight) *outHeight = 0;
    return guarded(error, MSPLAT_STATUS_GPU_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        requirePose(camToWorld);
        require(outWidth != nullptr && outHeight != nullptr,
                "Output dimensions must not be null");
        require(outRGBA != nullptr || outCapacity == 0,
                "outCapacity must be zero when querying dimensions");
        trainerHandle(t).trainer->renderFromPoseToBuffer(
            camToWorld, refCameraIndex, outRGBA, outCapacity, outWidth, outHeight);
        require(*outWidth > 0 && *outHeight > 0,
                "Reference camera index is out of range");
    });
}

namespace {

template <typename Fn>
MsplatStatus withTrainerAndPath(MsplatTrainer t, const char* path,
                                MsplatErrorInfo* error, Fn&& operation) {
    return guarded(error, MSPLAT_STATUS_IO_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        requirePath(path);
        operation(*trainerHandle(t).trainer, std::string(path));
    });
}

} // namespace

MsplatStatus msplat_trainer_export_ply_v2(MsplatTrainer t, const char* path,
                                          MsplatErrorInfo* error) {
    return withTrainerAndPath(t, path, error,
        [](msplat::Trainer& trainer, const std::string& value) { trainer.exportPly(value); });
}

MsplatStatus msplat_trainer_export_splat_v2(MsplatTrainer t, const char* path,
                                            MsplatErrorInfo* error) {
    return withTrainerAndPath(t, path, error,
        [](msplat::Trainer& trainer, const std::string& value) { trainer.exportSplat(value); });
}

MsplatStatus msplat_trainer_export_spz_v2(MsplatTrainer t, const char* path,
                                          MsplatErrorInfo* error) {
    return withTrainerAndPath(t, path, error,
        [](msplat::Trainer& trainer, const std::string& value) { trainer.exportSpz(value); });
}

MsplatStatus msplat_trainer_save_checkpoint_v2(MsplatTrainer t,
                                               const char* path,
                                               MsplatErrorInfo* error) {
    return withTrainerAndPath(t, path, error,
        [](msplat::Trainer& trainer, const std::string& value) { trainer.saveCheckpoint(value); });
}

MsplatStatus msplat_trainer_load_checkpoint_v2(MsplatTrainer t,
                                               const char* path,
                                               int* outIteration,
                                               MsplatErrorInfo* error) {
    if (outIteration) *outIteration = 0;
    return guarded(error, MSPLAT_STATUS_IO_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        requirePath(path);
        require(outIteration != nullptr, "outIteration must not be null");
        *outIteration = trainerHandle(t).trainer->loadCheckpoint(path);
    });
}

MsplatStatus msplat_trainer_splat_count_v2(MsplatTrainer t, int* outCount,
                                           MsplatErrorInfo* error) {
    if (outCount) *outCount = 0;
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        require(outCount != nullptr, "outCount must not be null");
        *outCount = trainerHandle(t).trainer->splatCount();
    });
}

MsplatStatus msplat_trainer_iteration_v2(MsplatTrainer t, int* outIteration,
                                         MsplatErrorInfo* error) {
    if (outIteration) *outIteration = 0;
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        require(outIteration != nullptr, "outIteration must not be null");
        *outIteration = trainerHandle(t).trainer->iteration();
    });
}

MsplatStatus msplat_trainer_metrics_v4(
    MsplatTrainer t, MsplatTrainingMetrics* outMetrics, size_t outputSize,
    MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        // Validate the caller's layout before writing a byte. This keeps an
        // older, smaller client buffer from being zeroed past its boundary.
        require(outputSize == sizeof(MsplatTrainingMetrics),
                "Training metrics size does not match this msplat ABI");
        require(outMetrics != nullptr, "outMetrics must not be null");
        *outMetrics = {};
        require(t != nullptr, "Trainer handle must not be null");

        const msplat::TrainingMetrics metrics =
            trainerHandle(t).trainer->metrics();
        if (metrics.hasSubmittedStep)
            outMetrics->flags |= MSPLAT_TRAINING_METRICS_HAS_SUBMITTED_STEP;
        if (metrics.hasCompletedStep)
            outMetrics->flags |= MSPLAT_TRAINING_METRICS_HAS_COMPLETED_STEP;
        if (metrics.gpuTimeValid)
            outMetrics->flags |= MSPLAT_TRAINING_METRICS_GPU_TIME_VALID;
        if (metrics.lossValid)
            outMetrics->flags |= MSPLAT_TRAINING_METRICS_LOSS_VALID;
        if (metrics.intersectionsValid)
            outMetrics->flags |= MSPLAT_TRAINING_METRICS_INTERSECTIONS_VALID;
        if (metrics.hasFailedStep)
            outMetrics->flags |= MSPLAT_TRAINING_METRICS_HAS_FAILED_STEP;

        auto copySubmitted = [](const msplat::SubmittedTrainingStep& source,
                                MsplatSubmittedTrainingStep& destination) {
            destination.iteration = source.iteration;
            destination.splatCount = source.splatCount;
            destination.modelCapacity = source.modelCapacity;
            destination.effectiveWidth = source.effectiveWidth;
            destination.effectiveHeight = source.effectiveHeight;
            destination.activeSHDegree = source.activeSHDegree;
            destination.cpuSubmitMs = source.cpuSubmitMs;
        };
        copySubmitted(metrics.submitted, outMetrics->submitted);

        const msplat::CompletedTrainingStep& completed = metrics.completed;
        outMetrics->completed.iteration = completed.iteration;
        outMetrics->completed.splatCount = completed.splatCount;
        outMetrics->completed.modelCapacity = completed.modelCapacity;
        outMetrics->completed.effectiveWidth = completed.effectiveWidth;
        outMetrics->completed.effectiveHeight = completed.effectiveHeight;
        outMetrics->completed.activeSHDegree = completed.activeSHDegree;
        outMetrics->completed.cpuSubmitMs = completed.cpuSubmitMs;
        outMetrics->completed.gpuExecutionMs = completed.gpuExecutionMs;
        outMetrics->completed.endToEndMs = completed.endToEndMs;
        outMetrics->completed.loss = completed.loss;
        outMetrics->completed.overflowKinds = completed.overflowKinds;
        outMetrics->completed.retainedPackedIntersectionCount =
            completed.retainedPackedIntersectionCount;
        outMetrics->completed.packedIntersectionCapacity =
            completed.packedIntersectionCapacity;
        outMetrics->overflowedCompletedSteps = metrics.overflowedCompletedSteps;
        outMetrics->tileCapOverflowedSteps = metrics.tileCapOverflowedSteps;
        outMetrics->packedCapacityOverflowedSteps =
            metrics.packedCapacityOverflowedSteps;
        outMetrics->lastOverflowIteration = metrics.lastOverflowIteration;
        outMetrics->lastFailedIteration = metrics.lastFailedIteration;
    });
}

MsplatStatus msplat_trainer_memory_metrics_v4(
    MsplatTrainer t, MsplatTrainingMemoryMetrics* outMetrics,
    size_t outputSize, MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(outputSize == sizeof(MsplatTrainingMemoryMetrics),
                "Training memory metrics size does not match this msplat ABI");
        require(outMetrics != nullptr, "outMetrics must not be null");
        *outMetrics = {};
        require(t != nullptr, "Trainer handle must not be null");

        const msplat::TrainingMemoryMetrics metrics =
            trainerHandle(t).trainer->memoryMetrics();
        if (metrics.hasProcessPhysFootprint)
            outMetrics->flags |= MSPLAT_MEMORY_METRICS_PHYS_FOOTPRINT_VALID;
        if (metrics.hasProcessAvailableBytes)
            outMetrics->flags |= MSPLAT_MEMORY_METRICS_AVAILABLE_VALID;
        outMetrics->trainerModelBufferBytes = metrics.trainerModelBufferBytes;
        outMetrics->engineSharedTransientBufferBytes =
            metrics.engineSharedTransientBufferBytes;
        outMetrics->engineTrainingTransientBufferBytes =
            metrics.engineTrainingTransientBufferBytes;
        outMetrics->trainerTelemetryReadbackBytes =
            metrics.trainerTelemetryReadbackBytes;
        outMetrics->trainerImageCacheCpuBytes =
            metrics.trainerImageCacheCpuBytes;
        outMetrics->trainerImageCacheGpuBytes =
            metrics.trainerImageCacheGpuBytes;
        outMetrics->trainerImageCacheBudgetBytes =
            metrics.trainerImageCacheBudgetBytes;
        outMetrics->processPhysFootprintBytes =
            metrics.processPhysFootprintBytes;
        outMetrics->processAvailableBytes = metrics.processAvailableBytes;
        outMetrics->trainingGpuImageCacheHits =
            metrics.trainingGpuImageCacheHits;
        outMetrics->trainingGpuImageCacheMisses =
            metrics.trainingGpuImageCacheMisses;
    });
}

void msplat_pixel_buffer_free(MsplatPixelBuffer* buffer) {
    if (!buffer) return;
    std::free(buffer->data);
    *buffer = {};
}

MsplatStatus msplat_set_metallib_path_v2(const char* path,
                                         MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_GPU_ERROR, [&] {
        requirePath(path);
        msplat_set_metallib_path_checked(path);
    });
}

MsplatStatus msplat_sync_v2(MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_GPU_ERROR, [] { msplat::sync(); });
}

MsplatStatus msplat_cleanup_v2(MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_GPU_ERROR, [] { msplat::cleanup(); });
}

// ABI v1 compatibility. These wrappers never allow a C++ exception to cross
// the C boundary; callers can inspect msplat_last_status/message on failure.

MsplatDataset msplat_dataset_create(const char* path, float downscaleFactor,
                                     bool evalMode, int testEvery) {
    MsplatDataset dataset = nullptr;
    msplat_dataset_create_v2(path, downscaleFactor, evalMode, testEvery, &dataset, nullptr);
    return dataset;
}

void msplat_dataset_destroy(MsplatDataset ds) {
    msplat_dataset_destroy_v2(ds, nullptr);
}

int msplat_dataset_num_train(MsplatDataset ds) {
    int count = 0;
    msplat_dataset_num_train_v2(ds, &count, nullptr);
    return count;
}

int msplat_dataset_num_test(MsplatDataset ds) {
    int count = 0;
    msplat_dataset_num_test_v2(ds, &count, nullptr);
    return count;
}

void msplat_dataset_camera_pose(MsplatDataset ds, int cameraIndex, float camToWorld[16]) {
    msplat_dataset_camera_pose_v2(ds, cameraIndex, camToWorld, nullptr);
}

MsplatTrainer msplat_trainer_create(MsplatDataset ds, MsplatConfig config) {
    MsplatTrainer trainer = nullptr;
    msplat_trainer_create_v2(ds, &config, sizeof(config), &trainer, nullptr);
    return trainer;
}

void msplat_trainer_destroy(MsplatTrainer t) {
    msplat_trainer_destroy_v2(t, nullptr);
}

MsplatStats msplat_trainer_step(MsplatTrainer t) {
    MsplatStats stats{};
    msplat_trainer_step_v2(t, &stats, nullptr);
    return stats;
}

void msplat_trainer_train(MsplatTrainer t) {
    msplat_trainer_train_v2(t, nullptr);
}

MsplatEvalMetrics msplat_trainer_evaluate(MsplatTrainer t) {
    MsplatEvalMetrics metrics{};
    msplat_trainer_evaluate_v2(t, &metrics, nullptr);
    return metrics;
}

MsplatPixelBuffer msplat_trainer_render(MsplatTrainer t, int cameraIndex, bool useTest) {
    MsplatPixelBuffer buffer{};
    msplat_trainer_render_v2(t, cameraIndex, useTest, &buffer, nullptr);
    return buffer;
}

MsplatPixelBuffer msplat_trainer_render_pose(MsplatTrainer t, const float camToWorld[16], int refCameraIndex) {
    MsplatPixelBuffer buffer{};
    msplat_trainer_render_pose_v2(t, camToWorld, refCameraIndex, &buffer, nullptr);
    return buffer;
}

void msplat_trainer_render_pose_to_buffer(MsplatTrainer t, const float camToWorld[16],
                                      int refCameraIndex, uint8_t* outRGBA,
                                      int* outWidth, int* outHeight) {
    msplat_trainer_render_pose_to_buffer_v2(
        t, camToWorld, refCameraIndex, outRGBA,
        outRGBA ? std::numeric_limits<size_t>::max() : 0,
        outWidth, outHeight, nullptr);
}

void msplat_trainer_export_ply(MsplatTrainer t, const char* path) {
    msplat_trainer_export_ply_v2(t, path, nullptr);
}

void msplat_trainer_export_splat(MsplatTrainer t, const char* path) {
    msplat_trainer_export_splat_v2(t, path, nullptr);
}

void msplat_trainer_export_spz(MsplatTrainer t, const char* path) {
    msplat_trainer_export_spz_v2(t, path, nullptr);
}

void msplat_trainer_save_checkpoint(MsplatTrainer t, const char* path) {
    msplat_trainer_save_checkpoint_v2(t, path, nullptr);
}

int msplat_trainer_load_checkpoint(MsplatTrainer t, const char* path) {
    int iteration = -1;
    msplat_trainer_load_checkpoint_v2(t, path, &iteration, nullptr);
    return iteration;
}

int msplat_trainer_splat_count(MsplatTrainer t) {
    int count = 0;
    msplat_trainer_splat_count_v2(t, &count, nullptr);
    return count;
}

int msplat_trainer_iteration(MsplatTrainer t) {
    int iteration = 0;
    msplat_trainer_iteration_v2(t, &iteration, nullptr);
    return iteration;
}

void msplat_sync(void) { msplat_sync_v2(nullptr); }
void msplat_cleanup(void) { msplat_cleanup_v2(nullptr); }
