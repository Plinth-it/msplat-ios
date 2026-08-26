// ObjC++ implementation of the Swift-facing C++ API.
// This is the ONLY file that touches internal C++ types (Model, Camera, MTensor).

#include "msplat_api.hpp"

#include "dataset_diagnostics.hpp"
#include "dataset_errors.hpp"
#include "model.hpp"
#include "input_data.hpp"
#include "msplat.hpp"
#include "ssim.hpp"
#include "memory_report.hpp"

#include <algorithm>
#include <atomic>
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
#include <utility>

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

    CameraTrainingTarget trainingTargetForTrainCamera(
        size_t trainIndex, int downscaleFactor) {
        return images.gpuTrainingTarget(
            data.cameras, trainIndices[trainIndex], downscaleFactor);
    }

    CameraTrainingTarget trainingTargetForTestCamera(
        size_t testIndex, int downscaleFactor) {
        return images.gpuTrainingTarget(
            data.cameras, testIndices[testIndex], downscaleFactor);
    }
};

namespace {

void initializeDataset(Dataset::Impl& impl, InputData data,
                       float downscaleFactor, bool evalMode, int testEvery,
                       const std::string& sourceDescription) {
    impl.data = std::move(data);
    if (impl.data.cameras.empty())
        throw std::runtime_error("Dataset has no cameras: " + sourceDescription);
    if (impl.data.points.count <= 0)
        throw std::runtime_error("Dataset has no sparse points: " + sourceDescription);
    impl.images = CameraImageCache(
        downscaleFactor, CameraImageCache::defaultBudgetBytes());

    // No image is decoded here. The first step that needs a camera loads it.
    if (evalMode) {
        auto [train, test] = impl.data.splitTrainTestIndices(testEvery);
        impl.trainIndices = std::move(train);
        impl.testIndices = std::move(test);
    } else {
        auto [train, valIdx] = impl.data.trainIndices(false);
        impl.trainIndices = std::move(train);
        (void)valIdx;
    }
}

} // namespace

Dataset::Dataset(const std::string& path, float downscaleFactor,
                 bool evalMode, int testEvery)
    : Dataset(path, downscaleFactor, evalMode, testEvery, false)
{}

Dataset::Dataset(const std::string& path, float downscaleFactor,
                 bool evalMode, int testEvery, bool discoverTrainingMasks)
    : impl(std::make_unique<Impl>())
{
    initializeDataset(
        *impl, inputDataFromX(path, "", discoverTrainingMasks), downscaleFactor,
                      evalMode, testEvery, path);
}

Dataset::Dataset(::DatasetDescriptor descriptor, float downscaleFactor,
                 bool evalMode, int testEvery)
    : impl(std::make_unique<Impl>())
{
    initializeDataset(*impl, inputDataFromDescriptor(std::move(descriptor)),
                      downscaleFactor, evalMode, testEvery, "descriptor");
}

Dataset::~Dataset() = default;
Dataset::Dataset(Dataset&&) noexcept = default;
Dataset& Dataset::operator=(Dataset&&) noexcept = default;

int Dataset::numTrain() const { return (int)impl->trainIndices.size(); }
int Dataset::numTest() const { return (int)impl->testIndices.size(); }
void Dataset::enableTrainingTargetPrefetch() noexcept {
    impl->images.enablePrefetch();
}
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

struct PreviewFrame::Impl {
    enum class Completion : uint8_t { Pending, Ready, Failed };

    Impl(id<MTLTexture> value, int frameWidth, int frameHeight)
        : texture([value retain]), width(frameWidth), height(frameHeight) {}

    ~Impl() { [texture release]; }

    void complete(bool succeeded, const char* message) noexcept {
        if (!succeeded) {
            try {
                std::lock_guard<std::mutex> lock(errorMutex);
                error = !message || message[0] == '\0'
                    ? "msplat: preview command buffer failed"
                    : message;
            } catch (...) {
                // Preserve a deterministic fallback even under allocation
                // pressure; completion handlers must not throw.
            }
        }
        completion.store(
            succeeded ? Completion::Ready : Completion::Failed,
            std::memory_order_release);
    }

    bool poll() const {
        const Completion value = completion.load(std::memory_order_acquire);
        if (value == Completion::Pending) return false;
        if (value == Completion::Ready) return true;

        std::lock_guard<std::mutex> lock(errorMutex);
        throw std::runtime_error(error.empty()
            ? "msplat: preview command buffer failed"
            : error);
    }

    id<MTLTexture> texture = nil;
    int width = 0;
    int height = 0;
    std::atomic<Completion> completion{Completion::Pending};
    mutable std::mutex errorMutex;
    std::string error;
};

PreviewFrame::PreviewFrame(std::shared_ptr<Impl> value)
    : impl(std::move(value)) {}

PreviewFrame::~PreviewFrame() = default;

bool PreviewFrame::poll() const {
    if (!impl) throw std::logic_error("Preview frame has no state");
    return impl->poll();
}

void* PreviewFrame::texture() const {
    if (!poll()) throw std::logic_error("Preview frame is not ready");
    return (__bridge void*)impl->texture;
}

int PreviewFrame::width() const { return impl ? impl->width : 0; }
int PreviewFrame::height() const { return impl ? impl->height : 0; }

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

    size_t cameraAfterCurrent() const {
        const size_t nextPosition = camIterPos + 1;
        if (nextPosition < camIndices.size())
            return camIndices[nextPosition];

        // Match the next epoch without advancing the live RNG or camera
        // sequence. The eventual currentCamera() call performs this same
        // shuffle after the committed step advances the cursor.
        std::vector<size_t> nextIndices = camIndices;
        std::mt19937 nextRng = rng;
        std::shuffle(nextIndices.begin(), nextIndices.end(), nextRng);
        return nextIndices.front();
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
        config.maxGaussians,
        config.refinePhotometricGains,
        config.refineCameraPoses,
        config.refineCameraPoses && !impl->ds->trainIndices.empty()
            ? static_cast<int>(impl->ds->trainIndices.front())
            : -1,
        config.cameraPoseConditioning,
        config.trainingMaskMode == TrainingMaskMode::Transparent,
        config.transparentAlphaLossWeight
    );

    impl->camIndices.resize(impl->ds->trainIndices.size());
    std::iota(impl->camIndices.begin(), impl->camIndices.end(), 0);
    impl->shuffleCameras();
    if (impl->ds->images.prefetchEnabled() && !impl->camIndices.empty()) {
        const size_t firstCamera = impl->currentCamera();
        impl->ds->images.prefetchTrainingTarget(
            impl->ds->data.cameras,
            impl->ds->trainIndices[firstCamera],
            impl->model->getDownscaleFactor(1));
    }
}

Trainer::~Trainer() {
    std::lock_guard lock(g_trainerTransactionMutex);
    if (impl && impl->ds)
        impl->ds->images.discardPrefetch();
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
        int ds = impl->model->getDownscaleFactor(nextStep);
        CameraTrainingTarget target =
            impl->ds->trainingTargetForTrainCamera(camIdx, ds);
        if (!target.image)
            throw std::runtime_error("Training target has no RGB image");
        Camera& cam = impl->ds->trainCamera(camIdx);

        msplat_training_step_mark_cpu_start(logicalStep);
        impl->model->fullIteration(
            cam, impl->ds->trainIndices[camIdx], nextStep, target,
            impl->config.ssimWeight);
        impl->model->schedulersStep(nextStep);
        impl->model->afterTrain(nextStep);

        MsplatTrainingStepDescriptor descriptor;
        descriptor.iteration = nextStep;
        descriptor.splatCount = impl->model->means.size(0);
        descriptor.modelCapacity = impl->model->buf_capacity;
        descriptor.effectiveWidth =
            static_cast<int32_t>(target.image->size(1));
        descriptor.effectiveHeight =
            static_cast<int32_t>(target.image->size(0));
        descriptor.activeShDegree = std::min(
            nextStep / impl->config.shDegreeInterval,
            impl->config.shDegree);
        cpuSubmitMs = msplat_training_step_submit(logicalStep, descriptor);

        if (impl->ds->images.prefetchEnabled() &&
            nextStep < impl->config.iterations) {
            const size_t nextCamIdx = impl->cameraAfterCurrent();
            impl->ds->images.prefetchTrainingTarget(
                impl->ds->data.cameras,
                impl->ds->trainIndices[nextCamIdx],
                impl->model->getDownscaleFactor(nextStep + 1));
        }
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
        CameraTrainingTarget target =
            impl->ds->trainingTargetForTestCamera(i, dsf);
        if (!target.image)
            throw std::runtime_error("Evaluation target has no RGB image");
        MTensor gtCpu = target.image->cpu();
        MTensor maskCpu;
        const MTensor* coverageMask = nullptr;
        if (target.coverageMask) {
            if (target.coverageMask == target.image) {
                // Preserve the packed-alpha marker across the GPU-to-CPU copy.
                coverageMask = &gtCpu;
            } else {
                maskCpu = target.coverageMask->cpu();
                coverageMask = &maskCpu;
            }
        }

        sumPsnr += psnr(
            rgbCpu, gtCpu, coverageMask, target.coverageUnits);
        sumSsim += ssim_eval(
            rgbCpu, gtCpu, coverageMask, target.coverageUnits);
        sumL1 += l1_loss(
            rgbCpu, gtCpu, coverageMask, target.coverageUnits);
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
    cam.setCameraToWorld(camToWorld);

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

    cam.setCameraToWorld(camToWorld);

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

std::unique_ptr<PreviewFrame> Trainer::renderFromPosePreview(
    const float camToWorld[16], int refCameraIndex) {
    std::lock_guard lock(g_trainerTransactionMutex);
    auto& indices = impl->ds->trainIndices;
    if (refCameraIndex < 0 || refCameraIndex >= (int)indices.size())
        throw std::invalid_argument("Reference camera index is out of range");

    Camera cam = impl->ds->images.ensureLoaded(
        impl->ds->data.cameras, indices[refCameraIndex]);
    const int downscale = impl->model->getDownscaleFactor(impl->currentStep);
    if (cam.width < downscale || cam.height < downscale)
        throw std::invalid_argument(
            "Training downscale produces a zero-sized image; reduce numDownscales");
    const int width = cam.width / downscale;
    const int height = cam.height / downscale;

    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
            MTLPixelFormatBGRA8Unorm
            width:static_cast<NSUInteger>(width)
            height:static_cast<NSUInteger>(height)
            mipmapped:NO];
    descriptor.textureType = MTLTextureType2D;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> texture = [msplat_device() newTextureWithDescriptor:descriptor];
    if (!texture)
        throw std::bad_alloc();
    texture.label = @"msplat completed preview";

    std::shared_ptr<PreviewFrame::Impl> frameState;
    try {
        frameState = std::make_shared<PreviewFrame::Impl>(
            texture, width, height);
    } catch (...) {
        [texture release];
        throw;
    }
    [texture release];

    cam.setCameraToWorld(camToWorld);
    MTensor rgb = impl->model->render(cam, impl->currentStep);
    if (rgb.size(1) != width || rgb.size(0) != height)
        throw std::runtime_error(
            "Rendered preview dimensions changed unexpectedly");

    msplat_submit_preview_texture(
        rgb, (__bridge void*)frameState->texture,
        [frameState](bool succeeded, const char* error) noexcept {
            frameState->complete(succeeded, error);
        });
    return std::unique_ptr<PreviewFrame>(
        new PreviewFrame(std::move(frameState)));
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
    // A constructor-prefetched step-one target cannot be reused after the
    // checkpoint changes both the logical step and camera sequence.
    impl->ds->images.discardPrefetch();
    const int loadedStep = impl->model->loadCheckpoint(path);
    impl->currentStep = loadedStep;
    msplat_training_telemetry_reset(impl->telemetry);
    // Re-shuffle cameras for resumed training
    impl->shuffleCameras();
    if (impl->ds->images.prefetchEnabled() &&
        loadedStep < impl->config.iterations &&
        !impl->camIndices.empty()) {
        const size_t firstCamera = impl->currentCamera();
        impl->ds->images.prefetchTrainingTarget(
            impl->ds->data.cameras,
            impl->ds->trainIndices[firstCamera],
            impl->model->getDownscaleFactor(loadedStep + 1));
    }
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
    metrics.countGpuTimeValid =
        snapshot.flags & MSPLAT_TRAINING_TELEMETRY_COUNT_GPU_TIMING_VALID;
    metrics.queueIdleTimeValid =
        snapshot.flags & MSPLAT_TRAINING_TELEMETRY_QUEUE_IDLE_TIMING_VALID;

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
    metrics.completed.imagePrepareMs =
        static_cast<float>(snapshot.completedStep.imagePrepareMs);
    metrics.completed.countGpuMs =
        static_cast<float>(snapshot.completedStep.countGpuMs);
    metrics.completed.countWaitWallMs =
        static_cast<float>(snapshot.completedStep.countWaitWallMs);
    metrics.completed.queueIdleMs =
        static_cast<float>(snapshot.completedStep.queueIdleMs);
    metrics.completed.postCountEncodeMs =
        static_cast<float>(snapshot.completedStep.postCountEncodeMs);
    metrics.completed.intersectionArenaGrowMs =
        static_cast<float>(snapshot.completedStep.intersectionArenaGrowMs);
    metrics.completed.maximumTileCount =
        snapshot.completedStep.maximumTileCount;
    metrics.completed.activeTileCount = snapshot.completedStep.activeTileCount;
    metrics.completed.trivialTileCount =
        snapshot.completedStep.trivialTileCount;
    metrics.completed.smallTileCount = snapshot.completedStep.smallTileCount;
    metrics.completed.mediumTileCount = snapshot.completedStep.mediumTileCount;
    metrics.completed.largeTileCount = snapshot.completedStep.largeTileCount;
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

uint32_t Trainer::poseRefinementStateCount() const {
    std::lock_guard lock(g_trainerTransactionMutex);
    return impl->model->poseRefinementStateCount();
}

PoseRefinementState Trainer::poseRefinementState(
    uint32_t canonicalCameraIndex) const {
    std::lock_guard lock(g_trainerTransactionMutex);
    if (canonicalCameraIndex >= impl->model->poseRefinementStateCount()) {
        throw std::invalid_argument(
            "Pose-refinement camera index is out of range");
    }

    // Pose deltas are updated by the training command buffer in shared Metal
    // storage. Keep this readback in the trainer transaction and wait before
    // dereferencing that storage on the CPU.
    msplat_gpu_sync();
    const ModelPoseRefinementState source =
        impl->model->poseRefinementState(canonicalCameraIndex);

    PoseRefinementState state;
    state.enabled = true;
    state.anchor = source.anchor;
    state.canonicalCameraIndex = canonicalCameraIndex;
    state.optimizerStepCount = source.optimizerStepCount;
    std::copy(source.geometry.poseDelta.begin(),
              source.geometry.poseDelta.end(), state.poseDelta);
    state.translationNorm = source.geometry.translationNorm;
    state.rotationNorm = source.geometry.rotationNorm;
    std::copy(source.geometry.correctedCameraToWorld.begin(),
              source.geometry.correctedCameraToWorld.end(),
              state.correctedCameraToWorld);
    state.frameId = source.frameId;
    state.frameIdLength = source.frameIdLength;
    return state;
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

msplat::PreviewFrame& previewFrameHandle(MsplatPreviewFrame handle) noexcept {
    return *static_cast<msplat::PreviewFrame*>(handle);
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
        } catch (const msplat::DatasetIOError& exception) {
            storeError(MSPLAT_STATUS_IO_ERROR, exception.what(), outError);
            return MSPLAT_STATUS_IO_ERROR;
        } catch (const msplat::InvalidDatasetError& exception) {
            storeError(MSPLAT_STATUS_INVALID_DATASET, exception.what(), outError);
            return MSPLAT_STATUS_INVALID_DATASET;
        } catch (const msplat::DatasetChangedError& exception) {
            storeError(MSPLAT_STATUS_INVALID_DATASET, exception.what(), outError);
            return MSPLAT_STATUS_INVALID_DATASET;
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

void validateDatasetCreationOptions(float downscaleFactor, bool evalMode,
                                    int testEvery) {
    require(std::isfinite(downscaleFactor) && downscaleFactor >= 1.0f &&
                downscaleFactor <= 32.0f,
            "downscaleFactor must be finite and in 1...32");
    require(testEvery > 0, "testEvery must be greater than zero");
    if (evalMode)
        require(testEvery >= 2, "testEvery must be at least 2 in eval mode");
}

template <typename Factory>
std::shared_ptr<msplat::Dataset> createCheckedDataset(Factory&& factory) {
    try {
        return factory();
    } catch (const std::invalid_argument& exception) {
        throw msplat::InvalidDatasetError(exception.what());
    }
}

MsplatStatus createDatasetFromPath(
    const char* path, float downscaleFactor, bool evalMode, int testEvery,
    bool discoverTrainingMasks, MsplatDataset* outDataset,
    MsplatErrorInfo* error) {
    if (outDataset) *outDataset = nullptr;
    return guarded(error, MSPLAT_STATUS_INVALID_DATASET, [&] {
        require(outDataset != nullptr, "outDataset must not be null");
        requirePath(path);
        validateDatasetCreationOptions(downscaleFactor, evalMode, testEvery);
        auto handle = std::make_unique<CApiDatasetHandle>();
        handle->dataset = createCheckedDataset([&] {
            return std::make_shared<msplat::Dataset>(
                std::string(path), downscaleFactor, evalMode, testEvery,
                discoverTrainingMasks);
        });
        *outDataset = static_cast<MsplatDataset>(handle.release());
    });
}

constexpr size_t kMaxDescriptorPointComponents =
    static_cast<size_t>(MSPLAT_DATASET_V5_MAX_POINTS) * 3;

void requireCountedPointer(const void* pointer, size_t count,
                           size_t maximumCount, const char* name) {
    if ((pointer == nullptr) != (count == 0)) {
        throw std::invalid_argument(
            std::string(name) + " pointer and count are inconsistent");
    }
    if (count > maximumCount) {
        throw std::invalid_argument(
            std::string(name) + " count exceeds the supported range");
    }
}

bool isRFC3629UTF8(const char* data, size_t length) noexcept {
    const auto* bytes = reinterpret_cast<const uint8_t*>(data);
    auto continuation = [](uint8_t byte) {
        return byte >= 0x80 && byte <= 0xbf;
    };

    size_t index = 0;
    while (index < length) {
        const uint8_t lead = bytes[index];
        if (lead <= 0x7f) {
            ++index;
            continue;
        }
        if (lead >= 0xc2 && lead <= 0xdf) {
            if (index + 1 >= length || !continuation(bytes[index + 1]))
                return false;
            index += 2;
            continue;
        }
        if (lead >= 0xe0 && lead <= 0xef) {
            if (index + 2 >= length || !continuation(bytes[index + 2]))
                return false;
            const uint8_t second = bytes[index + 1];
            if ((lead == 0xe0 && (second < 0xa0 || second > 0xbf)) ||
                (lead == 0xed && (second < 0x80 || second > 0x9f)) ||
                (lead != 0xe0 && lead != 0xed && !continuation(second))) {
                return false;
            }
            index += 3;
            continue;
        }
        if (lead >= 0xf0 && lead <= 0xf4) {
            if (index + 3 >= length ||
                !continuation(bytes[index + 2]) ||
                !continuation(bytes[index + 3])) {
                return false;
            }
            const uint8_t second = bytes[index + 1];
            if ((lead == 0xf0 && (second < 0x90 || second > 0xbf)) ||
                (lead == 0xf4 && (second < 0x80 || second > 0x8f)) ||
                (lead != 0xf0 && lead != 0xf4 && !continuation(second))) {
                return false;
            }
            index += 4;
            continue;
        }
        return false;
    }
    return true;
}

void validateStringView(const MsplatStringViewV5& view, const char* name) {
    requireCountedPointer(view.data, view.length,
                          MSPLAT_DATASET_V5_MAX_STRING_BYTES, name);
    if (view.length == 0) return;
    if (std::memchr(view.data, '\0', view.length) != nullptr) {
        throw std::invalid_argument(
            std::string(name) + " must not contain embedded NUL bytes");
    }
    if (!isRFC3629UTF8(view.data, view.length)) {
        throw std::invalid_argument(
            std::string(name) + " must contain valid RFC 3629 UTF-8");
    }
}

std::string copyStringView(const MsplatStringViewV5& view,
                           const char* name) {
    validateStringView(view, name);
    if (view.length == 0) return {};
    return std::string(view.data, view.length);
}

template <typename T>
std::vector<T> copyCountedArray(const T* pointer, size_t count,
                                size_t maximumCount, const char* name) {
    requireCountedPointer(pointer, count, maximumCount, name);
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::invalid_argument(
            std::string(name) + " byte size exceeds the supported range");
    }

    std::vector<T> result(count);
    if (count != 0)
        std::memcpy(result.data(), pointer, count * sizeof(T));
    return result;
}

::DatasetDescriptor copyDatasetDescriptor(
    const MsplatDatasetDescriptorV5& source) {
    requireCountedPointer(source.frames, source.frameCount,
                          MSPLAT_DATASET_V5_MAX_FRAMES, "frames");
    requireCountedPointer(source.pointXYZ, source.pointXYZCount,
                          kMaxDescriptorPointComponents, "pointXYZ");
    requireCountedPointer(source.pointRGB, source.pointRGBCount,
                          kMaxDescriptorPointComponents, "pointRGB");
    requireCountedPointer(source.pointSourceIds, source.pointSourceIdCount,
                          MSPLAT_DATASET_V5_MAX_POINTS, "pointSourceIds");
    requireCountedPointer(source.pointReprojectionErrors,
                          source.pointReprojectionErrorCount,
                          MSPLAT_DATASET_V5_MAX_POINTS,
                          "pointReprojectionErrors");
    requireCountedPointer(source.observations, source.observationCount,
                          MSPLAT_DATASET_V5_MAX_OBSERVATIONS, "observations");
    require(source.reserved[0] == 0 && source.reserved[1] == 0,
            "Dataset descriptor reserved fields must be zero");

    ::DatasetDescriptor result;
    result.frames.reserve(source.frameCount);
    for (size_t index = 0; index < source.frameCount; ++index) {
        const MsplatDatasetFrameV5& input = source.frames[index];
        require(input.reserved == 0, "Dataset frame reserved field must be zero");

        DatasetFrameDescriptor frame;
        frame.id = copyStringView(input.id, "frame id");
        frame.calibrationId = copyStringView(
            input.calibrationId, "frame calibrationId");
        frame.imagePath = copyStringView(input.imagePath, "frame imagePath");

        switch (input.rasterOrientation) {
            case MSPLAT_RASTER_ORIENTATION_ENCODED_PIXELS:
                frame.rasterOrientation = RasterOrientation::EncodedPixels;
                break;
            case MSPLAT_RASTER_ORIENTATION_EXIF_NORMALIZED:
                frame.rasterOrientation = RasterOrientation::ExifNormalized;
                break;
            default:
                throw std::invalid_argument(
                    "Dataset frame rasterOrientation is not recognized");
        }

        frame.calibration.width = input.calibration.width;
        frame.calibration.height = input.calibration.height;
        frame.calibration.fx = input.calibration.fx;
        frame.calibration.fy = input.calibration.fy;
        frame.calibration.cx = input.calibration.cx;
        frame.calibration.cy = input.calibration.cy;
        frame.calibration.k1 = input.calibration.k1;
        frame.calibration.k2 = input.calibration.k2;
        frame.calibration.k3 = input.calibration.k3;
        frame.calibration.p1 = input.calibration.p1;
        frame.calibration.p2 = input.calibration.p2;
        std::memcpy(frame.cameraToWorld.data(), input.cameraToWorld,
                    sizeof(input.cameraToWorld));
        result.frames.push_back(std::move(frame));
    }

    result.points.xyz = copyCountedArray(
        source.pointXYZ, source.pointXYZCount,
        kMaxDescriptorPointComponents, "pointXYZ");
    result.points.rgb = copyCountedArray(
        source.pointRGB, source.pointRGBCount,
        kMaxDescriptorPointComponents, "pointRGB");
    result.points.sourceIds = copyCountedArray(
        source.pointSourceIds, source.pointSourceIdCount,
        MSPLAT_DATASET_V5_MAX_POINTS, "pointSourceIds");
    result.points.reprojectionErrors = copyCountedArray(
        source.pointReprojectionErrors, source.pointReprojectionErrorCount,
        MSPLAT_DATASET_V5_MAX_POINTS, "pointReprojectionErrors");

    result.observations.reserve(source.observationCount);
    for (size_t index = 0; index < source.observationCount; ++index) {
        const MsplatSparseObservationV5& input = source.observations[index];
        require(input.reserved == 0,
                "Sparse observation reserved field must be zero");
        SparseObservation observation;
        observation.frameIndex = input.frameIndex;
        observation.frameObservationIndex = input.frameObservationIndex;
        observation.pointIndex = input.pointIndex;
        observation.x = input.x;
        observation.y = input.y;
        result.observations.push_back(observation);
    }

    result.provenance.adapter = copyStringView(
        source.provenanceAdapter, "provenance adapter");
    result.provenance.source = copyStringView(
        source.provenanceSource, "provenance source");
    return result;
}

void validateFrameMaskArray(const MsplatFrameMaskV6* frameMasks,
                            size_t frameMaskCount,
                            size_t frameMaskElementSize,
                            size_t descriptorFrameCount) {
    require(frameMaskElementSize == sizeof(MsplatFrameMaskV6),
            "Frame-mask element size does not match this msplat ABI");
    require(frameMaskCount == descriptorFrameCount,
            "Frame-mask count must match the dataset frame count");
    requireCountedPointer(frameMasks, frameMaskCount,
                          MSPLAT_DATASET_V5_MAX_FRAMES, "frameMasks");
    for (size_t index = 0; index < frameMaskCount; ++index) {
        const MsplatFrameMaskV6& input = frameMasks[index];
        require(input.reserved == 0 && input.reserved2[0] == 0 &&
                    input.reserved2[1] == 0,
                "Frame-mask reserved fields must be zero");

        validateStringView(input.maskPath, "frame mask path");
        if (input.maskPath.length == 0) {
            require(input.coverageChannel == MSPLAT_MASK_COVERAGE_LUMINANCE,
                    "An unmasked frame-mask slot must otherwise be zero");
            continue;
        }

        require(input.coverageChannel == MSPLAT_MASK_COVERAGE_LUMINANCE ||
                    input.coverageChannel == MSPLAT_MASK_COVERAGE_ALPHA,
                "Frame-mask coverage channel is not recognized");
    }
}

std::vector<std::optional<TrainingMaskDescriptor>> copyFrameMasks(
    const MsplatFrameMaskV6* frameMasks, size_t frameMaskCount) {
    std::vector<std::optional<TrainingMaskDescriptor>> result(frameMaskCount);
    for (size_t index = 0; index < frameMaskCount; ++index) {
        const MsplatFrameMaskV6& input = frameMasks[index];
        if (input.maskPath.length == 0) continue;

        TrainingMaskDescriptor mask;
        mask.path = std::string(input.maskPath.data, input.maskPath.length);
        switch (input.coverageChannel) {
            case MSPLAT_MASK_COVERAGE_LUMINANCE:
                mask.channel = TrainingMaskChannel::Luminance;
                break;
            case MSPLAT_MASK_COVERAGE_ALPHA:
                mask.channel = TrainingMaskChannel::Alpha;
                break;
            default:
                throw std::invalid_argument(
                    "Frame-mask coverage channel is not recognized");
        }
        result[index] = std::move(mask);
    }
    return result;
}

void copyReprojectionStatistics(
    const DatasetReprojectionStatistics& source,
    MsplatReprojectionErrorStatisticsV7& destination) noexcept {
    destination.sampleCount = source.sampleCount;
    destination.meanPixels = source.meanPixels;
    destination.rootMeanSquarePixels = source.rootMeanSquarePixels;
    destination.maximumPixels = source.maximumPixels;
}

void copyCaptureDiagnostics(
    const DatasetCaptureDiagnostics& source,
    MsplatDatasetCaptureDiagnosticsV7& destination,
    MsplatFrameCaptureDiagnosticsV7* frameDestinations) noexcept {
    destination.frameCount = source.frameCount;
    destination.pointCount = source.pointCount;
    destination.observationCount = source.observationCount;
    destination.linkedObservationCount = source.linkedObservationCount;
    destination.observedOutsideFrameCount =
        source.observedOutsideFrameCount;
    destination.reprojectedObservationCount =
        source.reprojectedObservationCount;
    destination.behindCameraObservationCount =
        source.behindCameraObservationCount;
    destination.nonFiniteProjectionCount = source.nonFiniteProjectionCount;
    destination.projectedOutsideFrameCount =
        source.projectedOutsideFrameCount;
    destination.observedPointCount = source.observedPointCount;
    destination.multiViewPointCount = source.multiViewPointCount;
    destination.maximumTrackLength = source.maximumTrackLength;
    destination.meanTrackLength = source.meanTrackLength;
    copyReprojectionStatistics(
        source.reprojectionError, destination.reprojectionError);
    copyReprojectionStatistics(
        source.sourcePointReprojectionError,
        destination.sourcePointReprojectionError);

    for (size_t index = 0; index < source.frames.size(); ++index) {
        const DatasetFrameCaptureDiagnostics& sourceFrame = source.frames[index];
        MsplatFrameCaptureDiagnosticsV7& destinationFrame =
            frameDestinations[index];
        destinationFrame.frameIndex = sourceFrame.frameIndex;
        destinationFrame.observationCount = sourceFrame.observationCount;
        destinationFrame.linkedObservationCount =
            sourceFrame.linkedObservationCount;
        destinationFrame.observedOutsideFrameCount =
            sourceFrame.observedOutsideFrameCount;
        destinationFrame.reprojectedObservationCount =
            sourceFrame.reprojectedObservationCount;
        destinationFrame.behindCameraObservationCount =
            sourceFrame.behindCameraObservationCount;
        destinationFrame.nonFiniteProjectionCount =
            sourceFrame.nonFiniteProjectionCount;
        destinationFrame.projectedOutsideFrameCount =
            sourceFrame.projectedOutsideFrameCount;
        copyReprojectionStatistics(
            sourceFrame.reprojectionError,
            destinationFrame.reprojectionError);
    }
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

void validateRefinementOptionsV8(
    const MsplatRefinementOptionsV8& options) {
    constexpr uint32_t knownFlags =
        MSPLAT_REFINEMENT_PHOTOMETRIC_RGB_GAINS |
        MSPLAT_REFINEMENT_CAMERA_POSE_DELTAS |
        MSPLAT_REFINEMENT_CAMERA_POSE_CAMP_CONDITIONING;
    require((options.flags & ~knownFlags) == 0u,
            "Refinement options contain unknown flags");
    require(
        (options.flags &
         MSPLAT_REFINEMENT_CAMERA_POSE_CAMP_CONDITIONING) == 0u ||
            (options.flags &
             MSPLAT_REFINEMENT_CAMERA_POSE_DELTAS) != 0u,
        "CamP conditioning requires camera-pose refinement");
    for (uint32_t reserved : options.reserved)
        require(reserved == 0u,
                "Refinement options reserved fields must be zero");
}

void validateTrainingMaskOptionsV11(
    const MsplatTrainingMaskOptionsV11& options) {
    require(options.mode == MSPLAT_TRAINING_MASK_MODE_COVERAGE ||
                options.mode == MSPLAT_TRAINING_MASK_MODE_TRANSPARENT,
            "Training mask mode is not recognized");
    require(std::isfinite(options.alphaLossWeight) &&
                options.alphaLossWeight >= 0.0f,
            "Training mask alpha loss weight must be finite and non-negative");
    for (uint32_t reserved : options.reserved)
        require(reserved == 0u,
                "Training mask options reserved fields must be zero");
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
    return createDatasetFromPath(
        path, downscaleFactor, evalMode, testEvery, false, outDataset, error);
}

MsplatStatus msplat_dataset_create_v10(
    const char* path, float downscaleFactor, bool evalMode, int32_t testEvery,
    bool discoverTrainingMasks, MsplatDataset* outDataset,
    MsplatErrorInfo* error) {
    return createDatasetFromPath(
        path, downscaleFactor, evalMode, static_cast<int>(testEvery),
        discoverTrainingMasks, outDataset, error);
}

MsplatStatus msplat_dataset_enable_training_target_prefetch_v14(
    MsplatDataset ds, MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(ds != nullptr, "Dataset handle must not be null");
        CApiDatasetHandle& handle = datasetHandle(ds);
        require(handle.dataset.use_count() == 1,
                "Training target prefetch must be enabled before trainer creation");
        handle.dataset->enableTrainingTargetPrefetch();
    });
}

MsplatStatus msplat_dataset_create_from_descriptor_v5(
    const MsplatDatasetDescriptorV5* descriptor,
    size_t descriptorSize,
    float downscaleFactor,
    bool evalMode,
    int32_t testEvery,
    MsplatDataset* outDataset,
    MsplatErrorInfo* error) {
    if (outDataset) *outDataset = nullptr;
    return guarded(error, MSPLAT_STATUS_INVALID_DATASET, [&] {
        require(outDataset != nullptr, "outDataset must not be null");
        require(descriptor != nullptr, "Dataset descriptor must not be null");
        require(descriptorSize == sizeof(MsplatDatasetDescriptorV5),
                "Dataset descriptor size does not match this msplat ABI");
        validateDatasetCreationOptions(
            downscaleFactor, evalMode, static_cast<int>(testEvery));

        ::DatasetDescriptor copied = copyDatasetDescriptor(*descriptor);
        auto handle = std::make_unique<CApiDatasetHandle>();
        handle->dataset = createCheckedDataset([&] {
            return std::make_shared<msplat::Dataset>(
                std::move(copied), downscaleFactor, evalMode,
                static_cast<int>(testEvery));
        });
        *outDataset = static_cast<MsplatDataset>(handle.release());
    });
}

MsplatStatus msplat_dataset_create_from_descriptor_v6(
    const MsplatDatasetDescriptorV5* descriptor,
    size_t descriptorSize,
    const MsplatFrameMaskV6* frameMasks,
    size_t frameMaskCount,
    size_t frameMaskElementSize,
    float downscaleFactor,
    bool evalMode,
    int32_t testEvery,
    MsplatDataset* outDataset,
    MsplatErrorInfo* error) {
    if (outDataset) *outDataset = nullptr;
    return guarded(error, MSPLAT_STATUS_INVALID_DATASET, [&] {
        require(outDataset != nullptr, "outDataset must not be null");
        require(descriptor != nullptr, "Dataset descriptor must not be null");
        require(descriptorSize == sizeof(MsplatDatasetDescriptorV5),
                "Dataset descriptor size does not match this msplat ABI");
        validateDatasetCreationOptions(
            downscaleFactor, evalMode, static_cast<int>(testEvery));
        validateFrameMaskArray(
            frameMasks, frameMaskCount, frameMaskElementSize,
            descriptor->frameCount);

        ::DatasetDescriptor copied = copyDatasetDescriptor(*descriptor);
        auto copiedMasks = copyFrameMasks(frameMasks, frameMaskCount);
        for (size_t index = 0; index < copiedMasks.size(); ++index) {
            copied.frames[index].trainingMask = std::move(copiedMasks[index]);
        }
        auto handle = std::make_unique<CApiDatasetHandle>();
        handle->dataset = createCheckedDataset([&] {
            return std::make_shared<msplat::Dataset>(
                std::move(copied), downscaleFactor, evalMode,
                static_cast<int>(testEvery));
        });
        *outDataset = static_cast<MsplatDataset>(handle.release());
    });
}

MsplatStatus msplat_dataset_capture_diagnostics_v7(
    const MsplatDatasetDescriptorV5* descriptor,
    size_t descriptorSize,
    MsplatDatasetCaptureDiagnosticsV7* outDiagnostics,
    size_t diagnosticsSize,
    MsplatFrameCaptureDiagnosticsV7* outFrames,
    size_t frameCount,
    size_t frameElementSize,
    MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(descriptor != nullptr, "Dataset descriptor must not be null");
        require(descriptorSize == sizeof(MsplatDatasetDescriptorV5),
                "Dataset descriptor size does not match this msplat ABI");
        require(outDiagnostics != nullptr,
                "Capture diagnostics output must not be null");
        require(diagnosticsSize == sizeof(MsplatDatasetCaptureDiagnosticsV7),
                "Capture diagnostics size does not match this msplat ABI");
        require(frameElementSize == sizeof(MsplatFrameCaptureDiagnosticsV7),
                "Frame diagnostics element size does not match this msplat ABI");
        require(descriptor->frameCount <= MSPLAT_DATASET_V5_MAX_FRAMES,
                "Dataset frame count exceeds the supported range");
        require(frameCount == descriptor->frameCount,
                "Frame diagnostics count must match the dataset frame count");
        requireCountedPointer(
            outFrames, frameCount, MSPLAT_DATASET_V5_MAX_FRAMES,
            "frame diagnostics");
        require(frameCount <= std::numeric_limits<size_t>::max() /
                    sizeof(MsplatFrameCaptureDiagnosticsV7),
                "Frame diagnostics byte size exceeds the supported range");

        std::memset(outDiagnostics, 0, sizeof(*outDiagnostics));
        if (frameCount != 0) {
            std::memset(
                outFrames, 0,
                frameCount * sizeof(MsplatFrameCaptureDiagnosticsV7));
        }

        ::DatasetDescriptor copied = copyDatasetDescriptor(*descriptor);
        DatasetCaptureDiagnostics diagnostics;
        try {
            diagnostics = analyzeDatasetCapture(copied);
        } catch (const std::invalid_argument& exception) {
            throw msplat::InvalidDatasetError(exception.what());
        }
        if (diagnostics.frames.size() != frameCount) {
            throw std::runtime_error(
                "Capture diagnostics frame count is inconsistent");
        }
        copyCaptureDiagnostics(diagnostics, *outDiagnostics, outFrames);
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
    const MsplatRefinementOptionsV8 refinementOptions =
        msplat_default_refinement_options_v8();
    return msplat_trainer_create_v8(
        ds, config, configSize, limits, limitsSize,
        &refinementOptions, sizeof(refinementOptions), outTrainer, error);
}

MsplatStatus msplat_refinement_options_validate_v8(
    const MsplatRefinementOptionsV8* options, size_t optionsSize,
    MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INVALID_ARGUMENT, [&] {
        require(options != nullptr, "Refinement options must not be null");
        require(optionsSize == sizeof(MsplatRefinementOptionsV8),
                "Refinement options size does not match this msplat ABI");
        validateRefinementOptionsV8(*options);
    });
}

MsplatStatus msplat_trainer_create_v8(
    MsplatDataset ds,
    const MsplatConfig* config, size_t configSize,
    const MsplatTrainingLimits* limits, size_t limitsSize,
    const MsplatRefinementOptionsV8* refinementOptions,
    size_t refinementOptionsSize,
    MsplatTrainer* outTrainer,
    MsplatErrorInfo* error) {
    const MsplatTrainingMaskOptionsV11 maskOptions =
        msplat_default_training_mask_options_v11();
    return msplat_trainer_create_v11(
        ds, config, configSize, limits, limitsSize,
        refinementOptions, refinementOptionsSize,
        &maskOptions, sizeof(maskOptions), outTrainer, error);
}

MsplatStatus msplat_training_mask_options_validate_v11(
    const MsplatTrainingMaskOptionsV11* options, size_t optionsSize,
    MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INVALID_ARGUMENT, [&] {
        require(options != nullptr, "Training mask options must not be null");
        require(optionsSize == sizeof(MsplatTrainingMaskOptionsV11),
                "Training mask options size does not match this msplat ABI");
        validateTrainingMaskOptionsV11(*options);
    });
}

MsplatStatus msplat_trainer_create_v11(
    MsplatDataset ds,
    const MsplatConfig* config, size_t configSize,
    const MsplatTrainingLimits* limits, size_t limitsSize,
    const MsplatRefinementOptionsV8* refinementOptions,
    size_t refinementOptionsSize,
    const MsplatTrainingMaskOptionsV11* maskOptions,
    size_t maskOptionsSize,
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
        require(refinementOptions != nullptr,
                "Refinement options must not be null");
        require(refinementOptionsSize == sizeof(MsplatRefinementOptionsV8),
                "Refinement options size does not match this msplat ABI");
        require(maskOptions != nullptr,
                "Training mask options must not be null");
        require(maskOptionsSize == sizeof(MsplatTrainingMaskOptionsV11),
                "Training mask options size does not match this msplat ABI");
        require(outTrainer != nullptr, "outTrainer must not be null");
        validateConfig(*config);
        validateTrainingLimits(*limits);
        validateRefinementOptionsV8(*refinementOptions);
        validateTrainingMaskOptionsV11(*maskOptions);
        require(!(maskOptions->mode ==
                      MSPLAT_TRAINING_MASK_MODE_TRANSPARENT &&
                  (refinementOptions->flags &
                   MSPLAT_REFINEMENT_PHOTOMETRIC_RGB_GAINS) != 0u),
                "Transparent training masks cannot be combined with photometric gain refinement");
        auto dataset = datasetHandle(ds).dataset;
        require(dataset->numTrain() > 0, "Dataset has no training cameras");
        auto cfg = configFromC(*config);
        cfg.maxGaussians = limits->maxGaussians;
        cfg.refinePhotometricGains =
            (refinementOptions->flags &
             MSPLAT_REFINEMENT_PHOTOMETRIC_RGB_GAINS) != 0u;
        cfg.refineCameraPoses =
            (refinementOptions->flags &
             MSPLAT_REFINEMENT_CAMERA_POSE_DELTAS) != 0u;
        cfg.cameraPoseConditioning =
            (refinementOptions->flags &
             MSPLAT_REFINEMENT_CAMERA_POSE_CAMP_CONDITIONING) != 0u
                ? msplat::CameraPoseConditioning::CamP
                : msplat::CameraPoseConditioning::Raw;
        cfg.trainingMaskMode =
            maskOptions->mode == MSPLAT_TRAINING_MASK_MODE_TRANSPARENT
                ? msplat::TrainingMaskMode::Transparent
                : msplat::TrainingMaskMode::Coverage;
        cfg.transparentAlphaLossWeight = maskOptions->alphaLossWeight;
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

MsplatStatus msplat_trainer_render_pose_preview_v13(
    MsplatTrainer t, const float camToWorld[16], int refCameraIndex,
    MsplatPreviewFrame* outFrame, MsplatErrorInfo* error) {
    if (outFrame) *outFrame = nullptr;
    return guarded(error, MSPLAT_STATUS_GPU_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        require(outFrame != nullptr, "outFrame must not be null");
        requirePose(camToWorld);
        auto frame = trainerHandle(t).trainer->renderFromPosePreview(
            camToWorld, refCameraIndex);
        require(frame != nullptr, "Preview frame creation returned no handle");
        *outFrame = frame.release();
    });
}

MsplatStatus msplat_preview_frame_poll_v13(
    MsplatPreviewFrame frame, bool* outReady, MsplatErrorInfo* error) {
    if (outReady) *outReady = false;
    return guarded(error, MSPLAT_STATUS_GPU_ERROR, [&] {
        require(frame != nullptr, "Preview frame must not be null");
        require(outReady != nullptr, "outReady must not be null");
        *outReady = previewFrameHandle(frame).poll();
    });
}

MsplatStatus msplat_preview_frame_texture_v13(
    MsplatPreviewFrame frame, MsplatMTLTextureRef* outTexture,
    int* outWidth, int* outHeight, MsplatErrorInfo* error) {
    if (outTexture) *outTexture = nil;
    if (outWidth) *outWidth = 0;
    if (outHeight) *outHeight = 0;
    return guarded(error, MSPLAT_STATUS_GPU_ERROR, [&] {
        require(frame != nullptr, "Preview frame must not be null");
        require(outTexture != nullptr, "outTexture must not be null");
        require(outWidth != nullptr && outHeight != nullptr,
                "Output dimensions must not be null");
        auto& value = previewFrameHandle(frame);
        require(value.poll(), "Preview frame is not ready");
        *outTexture = (__bridge id<MTLTexture>)value.texture();
        *outWidth = value.width();
        *outHeight = value.height();
    });
}

MsplatStatus msplat_preview_frame_destroy_v13(
    MsplatPreviewFrame frame, MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(frame != nullptr, "Preview frame must not be null");
        delete static_cast<msplat::PreviewFrame*>(frame);
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

MsplatStatus msplat_trainer_metrics_v12(
    MsplatTrainer t, MsplatTrainingMetricsV12* outMetrics, size_t outputSize,
    MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(outputSize == sizeof(MsplatTrainingMetricsV12),
                "Training metrics v12 size does not match this msplat ABI");
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
        if (metrics.countGpuTimeValid) {
            outMetrics->flags |=
                MSPLAT_TRAINING_METRICS_COUNT_GPU_TIME_VALID;
        }
        if (metrics.queueIdleTimeValid) {
            outMetrics->flags |=
                MSPLAT_TRAINING_METRICS_QUEUE_IDLE_TIME_VALID;
        }

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
        MsplatCompletedTrainingStepV12& destination = outMetrics->completed;
        destination.iteration = completed.iteration;
        destination.splatCount = completed.splatCount;
        destination.modelCapacity = completed.modelCapacity;
        destination.effectiveWidth = completed.effectiveWidth;
        destination.effectiveHeight = completed.effectiveHeight;
        destination.activeSHDegree = completed.activeSHDegree;
        destination.cpuSubmitMs = completed.cpuSubmitMs;
        destination.gpuExecutionMs = completed.gpuExecutionMs;
        destination.endToEndMs = completed.endToEndMs;
        destination.loss = completed.loss;
        destination.overflowKinds = completed.overflowKinds;
        destination.retainedPackedIntersectionCount =
            completed.retainedPackedIntersectionCount;
        destination.packedIntersectionCapacity =
            completed.packedIntersectionCapacity;
        destination.imagePrepareMs = completed.imagePrepareMs;
        destination.countGpuMs = completed.countGpuMs;
        destination.countWaitWallMs = completed.countWaitWallMs;
        destination.postCountEncodeMs = completed.postCountEncodeMs;
        destination.intersectionArenaGrowMs = completed.intersectionArenaGrowMs;
        destination.maximumTileCount = completed.maximumTileCount;
        destination.activeTileCount = completed.activeTileCount;
        destination.trivialTileCount = completed.trivialTileCount;
        destination.smallTileCount = completed.smallTileCount;
        destination.mediumTileCount = completed.mediumTileCount;
        destination.largeTileCount = completed.largeTileCount;
        destination.queueIdleMs = completed.queueIdleMs;

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

MsplatStatus msplat_trainer_pose_refinement_count_v15(
    MsplatTrainer t, uint32_t* outCount, MsplatErrorInfo* error) {
    if (outCount) *outCount = 0u;
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(t != nullptr, "Trainer handle must not be null");
        require(outCount != nullptr,
                "Pose-refinement count output must not be null");
        *outCount = trainerHandle(t).trainer->poseRefinementStateCount();
    });
}

MsplatStatus msplat_trainer_pose_refinement_state_v15(
    MsplatTrainer t, uint32_t canonicalCameraIndex,
    MsplatPoseRefinementStateV15* outState, size_t outputSize,
    MsplatErrorInfo* error) {
    return guarded(error, MSPLAT_STATUS_INTERNAL_ERROR, [&] {
        require(outputSize == sizeof(MsplatPoseRefinementStateV15),
                "Pose-refinement state size does not match this msplat ABI");
        require(outState != nullptr,
                "Pose-refinement state output must not be null");
        *outState = {};
        require(t != nullptr, "Trainer handle must not be null");

        const msplat::PoseRefinementState state =
            trainerHandle(t).trainer->poseRefinementState(
                canonicalCameraIndex);
        if (state.enabled)
            outState->flags |= MSPLAT_POSE_REFINEMENT_STATE_ENABLED;
        if (state.anchor)
            outState->flags |= MSPLAT_POSE_REFINEMENT_STATE_ANCHOR;
        outState->canonicalCameraIndex = state.canonicalCameraIndex;
        outState->optimizerStepCount = state.optimizerStepCount;
        std::memcpy(outState->poseDelta, state.poseDelta,
                    sizeof(outState->poseDelta));
        outState->translationNorm = state.translationNorm;
        outState->rotationNorm = state.rotationNorm;
        std::memcpy(outState->correctedCameraToWorld,
                    state.correctedCameraToWorld,
                    sizeof(outState->correctedCameraToWorld));
        outState->frameId = state.frameId;
        outState->frameIdLength =
            static_cast<uint64_t>(state.frameIdLength);
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
