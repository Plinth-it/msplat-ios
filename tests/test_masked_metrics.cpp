#include "ssim.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

namespace {

bool nearlyEqual(float lhs, float rhs, float tolerance = 1e-5f) {
    return std::abs(lhs - rhs) <= tolerance;
}

MTensor makeImage(int height, int width) {
    return MTensor({height, width, 3}, DType::Float32);
}

MTensor makeMask(int height, int width, uint8_t coverage) {
    MTensor mask({height, width}, DType::UInt8);
    for (int index = 0; index < height * width; ++index)
        mask.data<uint8_t>()[index] = coverage;
    return mask;
}

} // namespace

int main() {
    constexpr int height = 13;
    constexpr int width = 17;
    MTensor rendered = makeImage(height, width);
    MTensor gt = makeImage(height, width);
    for (int64_t index = 0; index < rendered.numel(); ++index) {
        rendered.data<float>()[index] =
            static_cast<float>(index % 17) / 20.0f;
        gt.data<float>()[index] =
            static_cast<float>(index % 13) / 19.0f;
    }

    const float unmaskedPsnr = psnr(rendered, gt);
    const float unmaskedL1 = l1_loss(rendered, gt);
    const float unmaskedSsim = ssim_eval(rendered, gt);

    MTensor fullMask = makeMask(height, width, 255);
    const uint64_t fullUnits =
        static_cast<uint64_t>(height) * width * 255u;
    CHECK(nearlyEqual(
        psnr(rendered, gt, &fullMask, fullUnits), unmaskedPsnr));
    CHECK(nearlyEqual(
        l1_loss(rendered, gt, &fullMask, fullUnits), unmaskedL1));
    CHECK(nearlyEqual(
        ssim_eval(rendered, gt, &fullMask, fullUnits), unmaskedSsim));

    MTensor fractionalMask = makeMask(height, width, 37);
    const uint64_t fractionalUnits =
        static_cast<uint64_t>(height) * width * 37u;
    CHECK(nearlyEqual(
        psnr(rendered, gt, &fractionalMask, fractionalUnits), unmaskedPsnr));
    CHECK(nearlyEqual(
        l1_loss(rendered, gt, &fractionalMask, fractionalUnits), unmaskedL1));
    CHECK(nearlyEqual(
        ssim_eval(rendered, gt, &fractionalMask, fractionalUnits),
        unmaskedSsim));

    MTensor twoPixelRendered = makeImage(1, 2);
    MTensor twoPixelGt = makeImage(1, 2);
    for (int channel = 0; channel < 3; ++channel) {
        twoPixelRendered.data<float>()[channel] = 1.0f;
        twoPixelRendered.data<float>()[3 + channel] = 0.5f;
        twoPixelGt.data<float>()[channel] = 0.0f;
        twoPixelGt.data<float>()[3 + channel] = 0.0f;
    }
    MTensor regionMask({1, 2}, DType::UInt8);
    regionMask.data<uint8_t>()[0] = 255;
    regionMask.data<uint8_t>()[1] = 128;
    constexpr uint64_t regionUnits = 383;
    const float expectedL1 = (255.0f + 128.0f * 0.5f) / 383.0f;
    const float expectedMse = (255.0f + 128.0f * 0.25f) / 383.0f;
    CHECK(nearlyEqual(
        l1_loss(twoPixelRendered, twoPixelGt, &regionMask, regionUnits),
        expectedL1));
    CHECK(nearlyEqual(
        psnr(twoPixelRendered, twoPixelGt, &regionMask, regionUnits),
        10.0f * std::log10(1.0f / expectedMse)));

    // The mask weights SSIM window centers. It does not hide error samples
    // inside a covered center's Gaussian window.
    MTensor impulseRendered = makeImage(15, 15);
    MTensor impulseGt = makeImage(15, 15);
    impulseRendered.zero();
    impulseGt.zero();
    for (int channel = 0; channel < 3; ++channel)
        impulseRendered.data<float>()[((7 * 15 + 7) * 3) + channel] = 1.0f;
    MTensor nearCenter = makeMask(15, 15, 0);
    nearCenter.data<uint8_t>()[7 * 15 + 8] = 255;
    MTensor farCenter = makeMask(15, 15, 0);
    farCenter.data<uint8_t>()[0] = 255;
    CHECK(ssim_eval(impulseRendered, impulseGt, &nearCenter, 255) < 0.99f);
    CHECK(nearlyEqual(
        ssim_eval(impulseRendered, impulseGt, &farCenter, 255), 1.0f));

    bool denominatorRejected = false;
    try {
        (void)l1_loss(rendered, gt, &fullMask, fullUnits - 1);
    } catch (const std::invalid_argument&) {
        denominatorRejected = true;
    }
    CHECK(denominatorRejected);

    MTensor emptyMask = makeMask(height, width, 0);
    bool emptyRejected = false;
    try {
        (void)ssim_eval(rendered, gt, &emptyMask, 0);
    } catch (const std::invalid_argument&) {
        emptyRejected = true;
    }
    CHECK(emptyRejected);

    return 0;
}
