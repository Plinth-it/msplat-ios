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

    // Evaluation renders remain float32 RGB, while cached training targets are
    // compact uint8 RGBA. A byte-equivalent float target must produce the same
    // metrics, and the padding alpha byte must never participate.
    MTensor mixedRendered = makeImage(height, width);
    MTensor floatTarget = makeImage(height, width);
    MTensor rgbaTarget({height, width, 4}, DType::UInt8);
    MTensor differentAlphaTarget({height, width, 4}, DType::UInt8);
    MTensor mixedMask({height, width}, DType::UInt8);
    uint64_t mixedCoverageUnits = 0;
    for (int64_t pixel = 0; pixel < height * width; ++pixel) {
        for (int channel = 0; channel < 3; ++channel) {
            const uint8_t targetByte = static_cast<uint8_t>(
                (pixel * 53 + channel * 79 + 17) % 256);
            floatTarget.data<float>()[pixel * 3 + channel] =
                targetByte / 255.0f;
            rgbaTarget.data<uint8_t>()[pixel * 4 + channel] = targetByte;
            differentAlphaTarget.data<uint8_t>()[pixel * 4 + channel] =
                targetByte;
            mixedRendered.data<float>()[pixel * 3 + channel] =
                static_cast<float>(
                    (pixel * 29 + channel * 41 + 11) % 251) / 250.0f;
        }
        rgbaTarget.data<uint8_t>()[pixel * 4 + 3] =
            static_cast<uint8_t>((pixel * 7) % 256);
        differentAlphaTarget.data<uint8_t>()[pixel * 4 + 3] =
            static_cast<uint8_t>(255 - (pixel * 13) % 256);
        const uint8_t coverage = static_cast<uint8_t>((pixel * 37) % 256);
        mixedMask.data<uint8_t>()[pixel] = coverage;
        mixedCoverageUnits += coverage;
    }

    const float floatPsnr = psnr(mixedRendered, floatTarget);
    const float floatL1 = l1_loss(mixedRendered, floatTarget);
    const float floatSsim = ssim_eval(mixedRendered, floatTarget);
    CHECK(nearlyEqual(psnr(mixedRendered, rgbaTarget), floatPsnr));
    CHECK(nearlyEqual(l1_loss(mixedRendered, rgbaTarget), floatL1));
    CHECK(nearlyEqual(ssim_eval(mixedRendered, rgbaTarget), floatSsim));
    CHECK(nearlyEqual(
        psnr(mixedRendered, differentAlphaTarget), floatPsnr));
    CHECK(nearlyEqual(
        l1_loss(mixedRendered, differentAlphaTarget), floatL1));
    CHECK(nearlyEqual(
        ssim_eval(mixedRendered, differentAlphaTarget), floatSsim));

    const float maskedFloatPsnr = psnr(
        mixedRendered, floatTarget, &mixedMask, mixedCoverageUnits);
    const float maskedFloatL1 = l1_loss(
        mixedRendered, floatTarget, &mixedMask, mixedCoverageUnits);
    const float maskedFloatSsim = ssim_eval(
        mixedRendered, floatTarget, &mixedMask, mixedCoverageUnits);
    CHECK(nearlyEqual(
        psnr(mixedRendered, rgbaTarget, &mixedMask, mixedCoverageUnits),
        maskedFloatPsnr));
    CHECK(nearlyEqual(
        l1_loss(mixedRendered, rgbaTarget, &mixedMask, mixedCoverageUnits),
        maskedFloatL1));
    CHECK(nearlyEqual(
        ssim_eval(mixedRendered, rgbaTarget, &mixedMask, mixedCoverageUnits),
        maskedFloatSsim));
    CHECK(nearlyEqual(
        psnr(mixedRendered, differentAlphaTarget,
             &mixedMask, mixedCoverageUnits),
        maskedFloatPsnr));
    CHECK(nearlyEqual(
        l1_loss(mixedRendered, differentAlphaTarget,
                &mixedMask, mixedCoverageUnits),
        maskedFloatL1));
    CHECK(nearlyEqual(
        ssim_eval(mixedRendered, differentAlphaTarget,
                  &mixedMask, mixedCoverageUnits),
        maskedFloatSsim));

    // Camera cache targets carry the same coverage in their RGBA alpha byte
    // and signal that layout by passing the target itself as the mask.
    MTensor packedMaskTarget = rgbaTarget;
    for (int64_t pixel = 0; pixel < height * width; ++pixel) {
        packedMaskTarget.data<uint8_t>()[pixel * 4 + 3] =
            mixedMask.data<uint8_t>()[pixel];
    }
    CHECK(nearlyEqual(
        psnr(mixedRendered, packedMaskTarget,
             &packedMaskTarget, mixedCoverageUnits),
        maskedFloatPsnr));
    CHECK(nearlyEqual(
        l1_loss(mixedRendered, packedMaskTarget,
                &packedMaskTarget, mixedCoverageUnits),
        maskedFloatL1));
    CHECK(nearlyEqual(
        ssim_eval(mixedRendered, packedMaskTarget,
                  &packedMaskTarget, mixedCoverageUnits),
        maskedFloatSsim));

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

    // Transparent targets composite soft coverage over the configured
    // background, while their metrics include every pixel in the frame.
    const float background[3] = {0.2f, 0.4f, 0.6f};
    MTensor transparentSource = makeImage(1, 3);
    MTensor transparentRGBA({1, 3, 4}, DType::UInt8);
    MTensor transparentMask({1, 3}, DType::UInt8);
    MTensor idealTransparent = makeImage(1, 3);
    const uint8_t alphaBytes[3] = {255, 128, 0};
    for (int pixel = 0; pixel < 3; ++pixel) {
        transparentMask.data<uint8_t>()[pixel] = alphaBytes[pixel];
        transparentRGBA.data<uint8_t>()[pixel * 4 + 3] = alphaBytes[pixel];
        const float alpha = alphaBytes[pixel] / 255.0f;
        for (int channel = 0; channel < 3; ++channel) {
            transparentSource.data<float>()[pixel * 3 + channel] = 1.0f;
            transparentRGBA.data<uint8_t>()[pixel * 4 + channel] = 255;
            idealTransparent.data<float>()[pixel * 3 + channel] =
                alpha + (1.0f - alpha) * background[channel];
        }
    }
    MTensor compositedFloat = composite_metric_target(
        transparentSource, transparentMask, background, regionUnits);
    MTensor compositedRGBA = composite_metric_target(
        transparentRGBA, transparentMask, background, regionUnits);
    MTensor compositedPacked = composite_metric_target(
        transparentRGBA, transparentRGBA, background, regionUnits);
    for (const MTensor* composited :
         {&compositedFloat, &compositedRGBA, &compositedPacked}) {
        CHECK(nearlyEqual(l1_loss(idealTransparent, *composited), 0.0f));
        CHECK(psnr(idealTransparent, *composited) > 120.0f);
        CHECK(nearlyEqual(
            ssim_eval(idealTransparent, *composited), 1.0f, 1e-4f));
    }
    // Keep Coverage semantics: the same rendered soft edge is compared with
    // foreground RGB, not the composited target.
    CHECK(l1_loss(idealTransparent, transparentSource,
                  &transparentMask, regionUnits) > 0.05f);
    idealTransparent.data<float>()[6] += 0.25f;
    CHECK(nearlyEqual(
        l1_loss(idealTransparent, compositedPacked), 0.25f / 9.0f));
    CHECK(psnr(idealTransparent, compositedPacked) < 25.0f);
    CHECK(ssim_eval(idealTransparent, compositedPacked) < 0.99f);
    CHECK(nearlyEqual(l1_loss(
        idealTransparent, compositedPacked, &transparentMask, regionUnits),
        0.0f));

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
