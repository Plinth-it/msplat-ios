#include "bindings.h"
#include "dataset_descriptor.hpp"
#include "loaders.hpp"
#include "msplat_api.hpp"
#include "ssim.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    "line " + std::to_string(__LINE__) + ": " + #condition); } while (false)

constexpr int kWidth = 64;
constexpr int kHeight = 32;
constexpr std::array<uint8_t, 3> kBackground = {32, 64, 96};

struct TempDirectory {
    std::filesystem::path path;

    TempDirectory() {
        std::string pattern = (std::filesystem::temp_directory_path() /
            "msplat-transparent-evaluation-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        const char* created = mkdtemp(writable.data());
        CHECK(created != nullptr);
        path = created;
    }

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

Image uniformImage(float value) {
    Image image;
    image.width = kWidth;
    image.height = kHeight;
    image.data.assign(kWidth * kHeight * 3, value);
    return image;
}

DatasetDescriptor makeDescriptor(const TempDirectory& temporary) {
    DatasetDescriptor descriptor;
    descriptor.provenance = {"transparent-evaluation-test", "synthetic"};
    for (int index = 0; index < 2; ++index) {
        DatasetFrameDescriptor frame;
        frame.id = "frame-" + std::to_string(index);
        frame.calibrationId = "camera";
        frame.imagePath = (temporary.path / "image.png").string();
        frame.trainingMask = TrainingMaskDescriptor{
            (temporary.path / "mask.png").string(),
            TrainingMaskChannel::Luminance};
        frame.calibration = {kWidth, kHeight, 40.0f, 40.0f, 32.0f, 16.0f};
        frame.cameraToWorld[3] = index * 0.1f;
        descriptor.frames.push_back(frame);
    }
    // Separate compact groups stay on either side of the mask boundary.
    // The exterior group initially matches the background exactly.
    for (int group = 0; group < 2; ++group) {
        for (int point = 0; point < 4; ++point) {
            descriptor.points.xyz.insert(descriptor.points.xyz.end(), {
                (group == 0 ? -0.65f : 0.65f) + (point % 2) * 0.05f,
                (point / 2) * 0.05f, -2.0f});
            for (int channel = 0; channel < 3; ++channel)
                descriptor.points.rgb.push_back(
                    group == 0 ? 255 : kBackground[channel]);
        }
    }
    return descriptor;
}

msplat::Config makeConfig(msplat::TrainingMaskMode mode) {
    msplat::Config config;
    config.iterations = 1;
    config.numDownscales = 0;
    config.shDegree = 0;
    config.stopDensifyAt = 0;
    config.trainingMaskMode = mode;
    for (int channel = 0; channel < 3; ++channel)
        config.bgColor[channel] = kBackground[channel] / 255.0f;
    return config;
}

msplat::EvalMetrics checkEvaluation(
    DatasetDescriptor descriptor, const msplat::Config& config,
    const MTensor& expectedTarget, const MTensor* metricMask = nullptr) {
    msplat::Dataset dataset(std::move(descriptor), 1.0f, true, 2);
    CHECK(dataset.numTrain() == 1);
    CHECK(dataset.numTest() == 1);
    msplat::Trainer trainer(dataset, config);
    const msplat::PixelBuffer pixels = trainer.render(0, true);
    CHECK(pixels.width == kWidth && pixels.height == kHeight);
    MTensor rendered({kHeight, kWidth, 3}, DType::Float32);
    std::memcpy(rendered.data_ptr(), pixels.data, rendered.nbytes());

    const msplat::EvalMetrics metrics = trainer.evaluate();
    CHECK(metrics.numTest == 1);
    CHECK(metrics.numGaussians == 8);
    CHECK(std::abs(metrics.l1 - l1_loss(
        rendered, expectedTarget, metricMask)) < 1e-6f);
    CHECK(std::abs(metrics.psnr - psnr(
        rendered, expectedTarget, metricMask)) < 1e-3f);
    CHECK(std::abs(metrics.ssim - ssim_eval(
        rendered, expectedTarget, metricMask, 0)) < 1e-5f);
    return metrics;
}

void checkTransparentEvaluation() {
    TempDirectory temporary;
    DatasetDescriptor descriptor = makeDescriptor(temporary);
    const std::string imagePath = descriptor.frames[0].imagePath;
    const std::string maskPath = descriptor.frames[0].trainingMask->path;
    Image source = uniformImage(1.0f);
    Image coverage = uniformImage(0.0f);
    constexpr float alpha = 128.0f / 255.0f;
    for (int y = 0; y < kHeight; ++y)
        for (int x = 0; x < kWidth / 2; ++x)
            for (int channel = 0; channel < 3; ++channel)
                coverage.data[(y * kWidth + x) * 3 + channel] = alpha;
    imwriteRGB(imagePath, source);
    imwriteRGB(maskPath, coverage);

    const msplat::Config transparent = makeConfig(
        msplat::TrainingMaskMode::Transparent);
    // Derive source colors whose composited target matches this deterministic
    // initial model. The only expected residual is 8-bit image quantization.
    {
        msplat::Dataset dataset(descriptor, 1.0f, true, 2);
        msplat::Trainer trainer(dataset, transparent);
        const msplat::PixelBuffer pixels = trainer.render(0, true);
        CHECK(pixels.width == kWidth && pixels.height == kHeight);
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth / 2; ++x) {
                for (int channel = 0; channel < 3; ++channel) {
                    const int offset = (y * kWidth + x) * 3 + channel;
                    const float background = transparent.bgColor[channel];
                    source.data[offset] =
                        (pixels.data[offset] - background) / alpha + background;
                    CHECK(source.data[offset] >= 0.0f &&
                          source.data[offset] <= 1.0f);
                }
            }
        }
    }
    imwriteRGB(imagePath, source);
    const RGBA8Image decoded = imreadRGBA8(
        imagePath, inspectImageSource(imagePath), kWidth, kHeight, false);
    const CoverageMask decodedMask = imreadCoverageMask(
        maskPath, inspectImageSource(maskPath), kWidth, kHeight, false,
        TrainingMaskChannel::Luminance);
    MTensor rawTarget({kHeight, kWidth, 3}, DType::Float32);
    MTensor compositedTarget({kHeight, kWidth, 3}, DType::Float32);
    MTensor mask({kHeight, kWidth}, DType::UInt8);
    for (int pixel = 0; pixel < kWidth * kHeight; ++pixel) {
        const uint8_t expectedCoverage = pixel % kWidth < kWidth / 2 ? 128 : 0;
        CHECK(decodedMask.data[pixel] == expectedCoverage);
        mask.data<uint8_t>()[pixel] = expectedCoverage;
        const float coverageValue = expectedCoverage / 255.0f;
        for (int channel = 0; channel < 3; ++channel) {
            const float rgb = decoded.data[pixel * 4 + channel] / 255.0f;
            rawTarget.data<float>()[pixel * 3 + channel] = rgb;
            compositedTarget.data<float>()[pixel * 3 + channel] =
                coverageValue * rgb + (1.0f - coverageValue) *
                    transparent.bgColor[channel];
        }
    }
    const msplat::EvalMetrics ideal = checkEvaluation(
        descriptor, transparent, compositedTarget);
    CHECK(ideal.l1 < 1e-3f);
    CHECK(ideal.psnr > 60.0f);

    const msplat::Config coverageConfig = makeConfig(
        msplat::TrainingMaskMode::Coverage);
    const msplat::EvalMetrics foreground = checkEvaluation(
        descriptor, coverageConfig, rawTarget, &mask);
    CHECK(foreground.l1 > ideal.l1 + 1e-4f);

    DatasetDescriptor artifact = descriptor;
    std::fill(artifact.points.rgb.begin() + 4 * 3,
              artifact.points.rgb.end(), 255);
    const msplat::EvalMetrics exterior = checkEvaluation(
        artifact, transparent, compositedTarget);
    CHECK(exterior.l1 > ideal.l1 + 1e-4f);
    CHECK(exterior.psnr < ideal.psnr);
    CHECK(exterior.ssim < ideal.ssim);
    const msplat::EvalMetrics ignoredExterior = checkEvaluation(
        artifact, coverageConfig, rawTarget, &mask);
    CHECK(std::abs(ignoredExterior.l1 - foreground.l1) < 1e-6f);

    // Transparent configuration alone must not composite unmasked frames.
    for (auto& frame : descriptor.frames) frame.trainingMask.reset();
    checkEvaluation(descriptor, transparent, rawTarget);
    checkEvaluation(descriptor, coverageConfig, rawTarget);
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 2)
                throw std::invalid_argument("Expected the metallib path");
            msplat_set_metallib_path_checked(argv[1]);
            checkTransparentEvaluation();
            cleanup_msplat_metal();
            return 0;
        } catch (const std::exception& error) {
            if (std::string(error.what()) ==
                "msplat: no Metal device is available") {
                std::cerr << "SKIP: " << error.what() << '\n';
                return 77;
            }
            std::cerr << error.what() << '\n';
            cleanup_msplat_metal();
            return 1;
        }
    }
}
