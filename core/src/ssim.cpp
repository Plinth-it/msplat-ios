// SSIM — CPU evaluation and window creation
// Ported from https://github.com/Po-Hsun-Su/pytorch-ssim (MIT)

#include "ssim.hpp"
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

struct MetricInput {
    int height;
    int width;
    int channels;
    int64_t pixels;
    const uint8_t* coverage;
    uint64_t coverageUnits;
    bool targetIsRGBA8;
};

float targetChannel(const MTensor& target, const MetricInput& input,
                    int64_t pixel, int channel) {
    if (input.targetIsRGBA8) {
        return target.data<uint8_t>()[pixel * 4 + channel] /
            255.0f;
    }
    return target.data<float>()[pixel * 3 + channel];
}

MetricInput validateMetricInput(const MTensor& rendered, const MTensor& gt,
                                const MTensor* coverageMask,
                                uint64_t suppliedCoverageUnits) {
    if (!rendered.defined() || !gt.defined())
        throw std::invalid_argument("Metric images must be defined");
    if (rendered.dtype() != DType::Float32 || rendered.ndim() != 3 ||
        rendered.size(2) != 3) {
        throw std::invalid_argument(
            "Rendered metric image must use float32 RGB");
    }
    const bool targetShapeMatches = gt.ndim() == 3 &&
        gt.size(0) == rendered.size(0) && gt.size(1) == rendered.size(1);
    const bool targetIsRGBA8 = targetShapeMatches &&
        gt.dtype() == DType::UInt8 && gt.size(2) == 4;
    const bool targetIsFloatRGB = targetShapeMatches &&
        gt.dtype() == DType::Float32 && gt.size(2) == 3;
    if (!targetIsRGBA8 && !targetIsFloatRGB) {
        throw std::invalid_argument(
            "Metric target must be matching float32 RGB or uint8 RGBA");
    }

    const int64_t height64 = rendered.size(0);
    const int64_t width64 = rendered.size(1);
    if (height64 <= 0 || width64 <= 0 ||
        height64 > std::numeric_limits<int>::max() ||
        width64 > std::numeric_limits<int>::max() ||
        height64 > std::numeric_limits<int>::max() / width64) {
        throw std::invalid_argument("Metric image dimensions are invalid");
    }
    const int64_t pixels = height64 * width64;
    if (static_cast<uint64_t>(pixels) >
        std::numeric_limits<uint64_t>::max() / 255u) {
        throw std::invalid_argument("Metric coverage denominator overflows");
    }

    MetricInput input{
        static_cast<int>(height64), static_cast<int>(width64), 3, pixels,
        nullptr, static_cast<uint64_t>(pixels) * 255u, targetIsRGBA8};
    if (!coverageMask) {
        if (suppliedCoverageUnits != 0 &&
            suppliedCoverageUnits != input.coverageUnits) {
            throw std::invalid_argument(
                "Unmasked metric coverage denominator is inconsistent");
        }
        return input;
    }

    if (!coverageMask->defined() || coverageMask->dtype() != DType::UInt8 ||
        coverageMask->ndim() != 2 || coverageMask->size(0) != height64 ||
        coverageMask->size(1) != width64) {
        throw std::invalid_argument(
            "Metric coverage mask must be uint8 with shape (height, width)");
    }

    input.coverage = coverageMask->data<uint8_t>();
    uint64_t calculatedCoverageUnits = 0;
    for (int64_t index = 0; index < pixels; ++index)
        calculatedCoverageUnits += input.coverage[index];
    if (calculatedCoverageUnits == 0)
        throw std::invalid_argument("Metric coverage mask has zero coverage");
    if (suppliedCoverageUnits != 0 &&
        suppliedCoverageUnits != calculatedCoverageUnits) {
        throw std::invalid_argument(
            "Metric coverage denominator does not match the mask");
    }
    input.coverageUnits = calculatedCoverageUnits;
    return input;
}

} // namespace

float psnr(const MTensor& rendered, const MTensor& gt,
           const MTensor* coverageMask, uint64_t coverageUnits) {
    const MetricInput input = validateMetricInput(
        rendered, gt, coverageMask, coverageUnits);
    const float* renderedData = rendered.data<float>();

    double mse = 0.0;
    for (int64_t pixel = 0; pixel < input.pixels; ++pixel) {
        const double coverage = input.coverage
            ? input.coverage[pixel]
            : 1.0;
        for (int channel = 0; channel < input.channels; ++channel) {
            const int64_t index = pixel * input.channels + channel;
            const double difference = renderedData[index] -
                targetChannel(gt, input, pixel, channel);
            mse += coverage * difference * difference;
        }
    }
    const double denominator = input.coverage
        ? static_cast<double>(input.coverageUnits) * input.channels
        : static_cast<double>(input.pixels) * input.channels;
    mse /= denominator;
    return 10.0f * std::log10(1.0 / mse);
}

float l1_loss(const MTensor& rendered, const MTensor& gt,
              const MTensor* coverageMask, uint64_t coverageUnits) {
    const MetricInput input = validateMetricInput(
        rendered, gt, coverageMask, coverageUnits);
    const float* renderedData = rendered.data<float>();

    double sum = 0.0;
    for (int64_t pixel = 0; pixel < input.pixels; ++pixel) {
        const double coverage = input.coverage
            ? input.coverage[pixel]
            : 1.0;
        for (int channel = 0; channel < input.channels; ++channel) {
            const int64_t index = pixel * input.channels + channel;
            sum += coverage *
                std::abs(renderedData[index] -
                         targetChannel(gt, input, pixel, channel));
        }
    }
    const double denominator = input.coverage
        ? static_cast<double>(input.coverageUnits) * input.channels
        : static_cast<double>(input.pixels) * input.channels;
    return static_cast<float>(sum / denominator);
}

std::vector<float> createSSIMWindow(int windowSize, float sigma) {
    // 1D Gaussian
    std::vector<float> g(windowSize);
    float sum = 0;
    for (int i = 0; i < windowSize; i++) {
        float x = (float)(i - windowSize / 2);
        g[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        sum += g[i];
    }
    for (int i = 0; i < windowSize; i++) g[i] /= sum;

    // 2D = outer product
    std::vector<float> w(windowSize * windowSize);
    for (int i = 0; i < windowSize; i++)
        for (int j = 0; j < windowSize; j++)
            w[i * windowSize + j] = g[i] * g[j];
    return w;
}

// Separable Gaussian blur on a single-channel (H, W) image.
// Writes result to `out`. `tmp` is scratch space (H * W floats).
static void gaussianBlur(const float* in, float* out, float* tmp,
                         int H, int W, const float* kernel, int kSize) {
    int pad = kSize / 2;

    // Horizontal pass: in → tmp
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sum = 0;
            for (int k = 0; k < kSize; k++) {
                int sx = x + k - pad;
                if (sx < 0) sx = 0;
                if (sx >= W) sx = W - 1;
                sum += in[y * W + sx] * kernel[k];
            }
            tmp[y * W + x] = sum;
        }
    }

    // Vertical pass: tmp → out
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sum = 0;
            for (int k = 0; k < kSize; k++) {
                int sy = y + k - pad;
                if (sy < 0) sy = 0;
                if (sy >= H) sy = H - 1;
                sum += tmp[sy * W + x] * kernel[k];
            }
            out[y * W + x] = sum;
        }
    }
}

float ssim_eval(const MTensor& rendered, const MTensor& gt,
                int windowSize, float sigma) {
    return ssim_eval(
        rendered, gt, nullptr, 0, windowSize, sigma);
}

float ssim_eval(const MTensor& rendered, const MTensor& gt,
                const MTensor* coverageMask, uint64_t coverageUnits,
                int windowSize, float sigma) {
    const MetricInput input = validateMetricInput(
        rendered, gt, coverageMask, coverageUnits);
    const int H = input.height;
    const int W = input.width;
    const int C = input.channels;
    const int HW = static_cast<int>(input.pixels);

    // 1D Gaussian kernel
    std::vector<float> kernel(windowSize);
    float ksum = 0;
    for (int i = 0; i < windowSize; i++) {
        float x = (float)(i - windowSize / 2);
        kernel[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        ksum += kernel[i];
    }
    for (int i = 0; i < windowSize; i++) kernel[i] /= ksum;

    const float* r = rendered.data<float>();

    // Scratch buffers
    std::vector<float> r_ch(HW), g_ch(HW), rr(HW), gg(HW), rg(HW);
    std::vector<float> mu1(HW), mu2(HW), s_rr(HW), s_gg(HW), s_rg(HW);
    std::vector<float> tmp(HW);

    const float C1 = 0.01f * 0.01f;
    const float C2 = 0.03f * 0.03f;
    double ssim_sum = 0;
    int count = 0;

    for (int c = 0; c < C; c++) {
        // Extract channel (HWC → planar)
        for (int i = 0; i < HW; i++) {
            r_ch[i] = r[i * C + c];
            g_ch[i] = targetChannel(gt, input, i, c);
            rr[i] = r_ch[i] * r_ch[i];
            gg[i] = g_ch[i] * g_ch[i];
            rg[i] = r_ch[i] * g_ch[i];
        }

        gaussianBlur(r_ch.data(), mu1.data(), tmp.data(), H, W, kernel.data(), windowSize);
        gaussianBlur(g_ch.data(), mu2.data(), tmp.data(), H, W, kernel.data(), windowSize);
        gaussianBlur(rr.data(), s_rr.data(), tmp.data(), H, W, kernel.data(), windowSize);
        gaussianBlur(gg.data(), s_gg.data(), tmp.data(), H, W, kernel.data(), windowSize);
        gaussianBlur(rg.data(), s_rg.data(), tmp.data(), H, W, kernel.data(), windowSize);

        for (int i = 0; i < HW; i++) {
            float m1sq = mu1[i] * mu1[i];
            float m2sq = mu2[i] * mu2[i];
            float m12  = mu1[i] * mu2[i];
            float sig1sq = s_rr[i] - m1sq;
            float sig2sq = s_gg[i] - m2sq;
            float sig12  = s_rg[i] - m12;

            float num = (2.0f * m12 + C1) * (2.0f * sig12 + C2);
            float den = (m1sq + m2sq + C1) * (sig1sq + sig2sq + C2);
            const double value = num / den;
            ssim_sum += input.coverage
                ? static_cast<double>(input.coverage[i]) * value
                : value;
            count++;
        }
    }

    const double denominator = input.coverage
        ? static_cast<double>(input.coverageUnits) * C
        : count;
    return static_cast<float>(ssim_sum / denominator);
}
