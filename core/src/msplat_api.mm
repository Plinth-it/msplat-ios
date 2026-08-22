// ObjC++ implementation of the Swift-facing C++ API.
// This is the ONLY file that touches internal C++ types (Model, Camera, MTensor).

#include "msplat_api.hpp"

#include "model.hpp"
#include "input_data.hpp"
#include "msplat.hpp"
#include "ssim.hpp"
#include "memory_report.hpp"

#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstdlib>
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

// ── Trainer::Impl ───────────────────────────────────────────────────────────

struct Trainer::Impl {
    std::unique_ptr<Model> model;
    Config config;
    Dataset::Impl* ds = nullptr;
    int currentStep = 0;

    // Camera iteration
    std::vector<size_t> camIndices;
    size_t camIterPos = 0;
    std::mt19937 rng{42};

    void shuffleCameras() {
        std::shuffle(camIndices.begin(), camIndices.end(), rng);
        camIterPos = 0;
    }

    size_t nextCamera() {
        if (camIterPos >= camIndices.size()) shuffleCameras();
        return camIndices[camIterPos++];
    }
};

Trainer::Trainer(Dataset& dataset, const Config& config)
    : impl(std::make_unique<Impl>())
{
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
        config.stopDensifyAt
    );

    impl->camIndices.resize(impl->ds->trainIndices.size());
    std::iota(impl->camIndices.begin(), impl->camIndices.end(), 0);
    impl->shuffleCameras();
}

Trainer::~Trainer() = default;

Stats Trainer::step() {
    impl->currentStep++;
    size_t camIdx = impl->nextCamera();
    Camera& cam = impl->ds->trainCamera(camIdx);

    int ds = impl->model->getDownscaleFactor(impl->currentStep);
    MTensor& gt = impl->ds->gpuImageForTrainCamera(camIdx, ds);

    auto t0 = std::chrono::high_resolution_clock::now();

    impl->model->fullIteration(cam, impl->currentStep, gt, impl->config.ssimWeight);
    impl->model->schedulersStep(impl->currentStep);
    impl->model->afterTrain(impl->currentStep);
    msplat_commit();

    reportMemory(impl->currentStep, (int)impl->model->means.size(0),
                 impl->model->estimatedGpuBytes(),
                 impl->ds->images.cachedBytes(),
                 impl->ds->images.budgetBytes());

    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;

    Stats s;
    s.iteration = impl->currentStep;
    s.splatCount = (int)impl->model->means.size(0);
    s.msPerStep = ms;
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
    float* buf = (float*)malloc(h * w * 3 * sizeof(float));
    memcpy(buf, rgbCpu.data_ptr(), h * w * 3 * sizeof(float));

    return PixelBuffer(buf, w, h);
}

PixelBuffer Trainer::renderFromPose(const float camToWorld[16], int refCameraIndex) {
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
    float* buf = (float*)malloc(h * w * 3 * sizeof(float));
    memcpy(buf, rgbCpu.data_ptr(), h * w * 3 * sizeof(float));
    return PixelBuffer(buf, w, h);
}

void Trainer::renderFromPoseToBuffer(const float camToWorld[16], int refCameraIndex,
                                  uint8_t* outRGBA, int* outWidth, int* outHeight) {
    auto& indices = impl->ds->trainIndices;
    if (refCameraIndex < 0 || refCameraIndex >= (int)indices.size()) {
        *outWidth = 0; *outHeight = 0; return;
    }

    Camera cam = impl->ds->images.ensureLoaded(impl->ds->data.cameras, indices[refCameraIndex]);
    memcpy(cam.camToWorld, camToWorld, 16 * sizeof(float));
    cam.cachedViewMat = MTensor();
    cam.cachedProjViewMat = MTensor();

    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();

    int h = (int)rgb.size(0), w = (int)rgb.size(1);
    *outWidth = w;
    *outHeight = h;
    if (!outRGBA) return;

    // Read directly from GPU tensor (unified memory on Apple Silicon)
    const float* src = (const float*)rgb.data_ptr();
    int n = w * h;
    for (int i = 0; i < n; i++) {
        outRGBA[i * 4]     = (uint8_t)(fminf(fmaxf(src[i*3],   0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 1] = (uint8_t)(fminf(fmaxf(src[i*3+1], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 2] = (uint8_t)(fminf(fmaxf(src[i*3+2], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 3] = 255;
    }
}

void Trainer::exportPly(const std::string& path) {
    impl->model->savePly(path, impl->currentStep);
}

void Trainer::exportSplat(const std::string& path) {
    impl->model->saveSplat(path);
}

void Trainer::saveCheckpoint(const std::string& path) {
    impl->model->saveCheckpoint(path, impl->currentStep);
}

int Trainer::loadCheckpoint(const std::string& path) {
    impl->currentStep = impl->model->loadCheckpoint(path);
    // Re-shuffle cameras for resumed training
    impl->shuffleCameras();
    return impl->currentStep;
}

int Trainer::splatCount() const {
    return (int)impl->model->means.size(0);
}

int Trainer::iteration() const {
    return impl->currentStep;
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

void sync() { msplat_gpu_sync(); }
void cleanup() { cleanup_msplat_metal(); }

} // namespace msplat

// ── C API (for Swift interop) ───────────────────────────────────────────────

#include "msplat_c_api.h"

static msplat::Config configFromC(MsplatConfig c) {
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

MsplatDataset msplat_dataset_create(const char* path, float downscaleFactor,
                                     bool evalMode, int testEvery) {
    auto* ds = new msplat::Dataset(std::string(path), downscaleFactor, evalMode, testEvery);
    return static_cast<MsplatDataset>(ds);
}

void msplat_dataset_destroy(MsplatDataset ds) {
    delete static_cast<msplat::Dataset*>(ds);
}

int msplat_dataset_num_train(MsplatDataset ds) {
    return static_cast<msplat::Dataset*>(ds)->numTrain();
}

int msplat_dataset_num_test(MsplatDataset ds) {
    return static_cast<msplat::Dataset*>(ds)->numTest();
}

void msplat_dataset_camera_pose(MsplatDataset ds, int cameraIndex, float camToWorld[16]) {
    static_cast<msplat::Dataset*>(ds)->cameraPose(cameraIndex, camToWorld);
}

MsplatTrainer msplat_trainer_create(MsplatDataset ds, MsplatConfig config) {
    auto* dataset = static_cast<msplat::Dataset*>(ds);
    auto cfg = configFromC(config);
    auto* trainer = new msplat::Trainer(*dataset, cfg);
    return static_cast<MsplatTrainer>(trainer);
}

void msplat_trainer_destroy(MsplatTrainer t) {
    delete static_cast<msplat::Trainer*>(t);
}

MsplatStats msplat_trainer_step(MsplatTrainer t) {
    auto stats = static_cast<msplat::Trainer*>(t)->step();
    return MsplatStats{stats.iteration, stats.splatCount, stats.msPerStep};
}

void msplat_trainer_train(MsplatTrainer t) {
    static_cast<msplat::Trainer*>(t)->train(0);
}

MsplatEvalMetrics msplat_trainer_evaluate(MsplatTrainer t) {
    auto m = static_cast<msplat::Trainer*>(t)->evaluate();
    return MsplatEvalMetrics{m.psnr, m.ssim, m.l1, m.numTest, m.numGaussians};
}

MsplatPixelBuffer msplat_trainer_render(MsplatTrainer t, int cameraIndex, bool useTest) {
    auto buf = static_cast<msplat::Trainer*>(t)->render(cameraIndex, useTest);
    MsplatPixelBuffer result{buf.data, buf.width, buf.height};
    buf.data = nullptr; // Transfer ownership to caller
    return result;
}

MsplatPixelBuffer msplat_trainer_render_pose(MsplatTrainer t, const float camToWorld[16], int refCameraIndex) {
    auto buf = static_cast<msplat::Trainer*>(t)->renderFromPose(camToWorld, refCameraIndex);
    MsplatPixelBuffer result{buf.data, buf.width, buf.height};
    buf.data = nullptr;
    return result;
}

void msplat_trainer_render_pose_to_buffer(MsplatTrainer t, const float camToWorld[16],
                                      int refCameraIndex, uint8_t* outRGBA,
                                      int* outWidth, int* outHeight) {
    static_cast<msplat::Trainer*>(t)->renderFromPoseToBuffer(
        camToWorld, refCameraIndex, outRGBA, outWidth, outHeight);
}

void msplat_trainer_export_ply(MsplatTrainer t, const char* path) {
    static_cast<msplat::Trainer*>(t)->exportPly(std::string(path));
}

void msplat_trainer_export_splat(MsplatTrainer t, const char* path) {
    static_cast<msplat::Trainer*>(t)->exportSplat(std::string(path));
}

void msplat_trainer_save_checkpoint(MsplatTrainer t, const char* path) {
    static_cast<msplat::Trainer*>(t)->saveCheckpoint(std::string(path));
}

int msplat_trainer_load_checkpoint(MsplatTrainer t, const char* path) {
    return static_cast<msplat::Trainer*>(t)->loadCheckpoint(std::string(path));
}

int msplat_trainer_splat_count(MsplatTrainer t) {
    return static_cast<msplat::Trainer*>(t)->splatCount();
}

int msplat_trainer_iteration(MsplatTrainer t) {
    return static_cast<msplat::Trainer*>(t)->iteration();
}

void msplat_sync(void) { msplat::sync(); }
void msplat_cleanup(void) { msplat::cleanup(); }
