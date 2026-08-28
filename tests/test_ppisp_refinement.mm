#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const char* expression, int line) {
    throw std::runtime_error(
        "line " + std::to_string(line) + ": " + expression);
}

#define CHECK(condition) \
    do { if (!(condition)) fail(#condition, __LINE__); } while (false)

constexpr uint32_t kWidth = 3;
constexpr uint32_t kHeight = 2;
constexpr uint32_t kParameterCount = 9;
constexpr uint32_t kAppearanceNonfinite = 1u << 2;
constexpr float kMaxAbsExposure = 4.0f;
constexpr float kMaxAbsColorParameter = 4.0f;

struct ScopedObjCRelease {
    id object = nil;
    ~ScopedObjCRelease() { [object release]; }
};

std::runtime_error metalError(const char* operation, NSError* error) {
    std::string message = operation;
    const char* description =
        error ? error.localizedDescription.UTF8String : nullptr;
    if (description) {
        message += ": ";
        message += description;
    }
    return std::runtime_error(message);
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, const void* bytes,
                         size_t byteCount) {
    id<MTLBuffer> buffer = [device newBufferWithBytes:bytes
        length:byteCount options:MTLResourceStorageModeShared];
    if (!buffer)
        throw std::runtime_error("Failed to allocate a Metal test buffer");
    return buffer;
}

id<MTLComputePipelineState> makePipeline(
    id<MTLDevice> device, id<MTLLibrary> library, const char* name) {
    NSString* functionName = [NSString stringWithUTF8String:name];
    CHECK(functionName != nil);
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    CHECK(function != nil);
    ScopedObjCRelease functionOwner{function};
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline)
        throw metalError("Failed to create PPISP test pipeline", error);
    return pipeline;
}

bool nearlyEqual(float actual, float expected,
                 float absoluteTolerance = 3.0e-4f,
                 float relativeTolerance = 6.0e-3f) {
    return std::abs(actual - expected) <= absoluteTolerance +
        relativeTolerance * std::max(std::abs(actual), std::abs(expected));
}

struct BackwardResult {
    std::vector<float> rendererGradient;
    std::array<float, kParameterCount> parameterGradient{};
    uint32_t status = 0;
};

struct AdamResult {
    std::vector<float> parameters;
    std::vector<float> firstMoments;
    std::vector<float> secondMoments;
    uint32_t status = 0;
};

class PpispHarness {
public:
    explicit PpispHarness(const char* metallibPath) {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) throw std::runtime_error("No Metal device");
        queue_ = [device_ newCommandQueue];
        CHECK(queue_ != nil);
        NSString* path = [NSString stringWithUTF8String:metallibPath];
        CHECK(path != nil);
        NSError* error = nil;
        library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:path]
            error:&error];
        if (!library_)
            throw metalError("Failed to load the PPISP test metallib", error);
        forward_ = makePipeline(
            device_, library_, "ppisp_frame_forward_kernel");
        backward_ = makePipeline(
            device_, library_, "ppisp_frame_backward_kernel");
        adam_ = makePipeline(device_, library_, "ppisp_frame_adam_kernel");
    }

    ~PpispHarness() {
        [adam_ release];
        [backward_ release];
        [forward_ release];
        [library_ release];
        [queue_ release];
        [device_ release];
    }

    std::vector<float> forward(const std::vector<float>& rendererRgb,
                               const std::vector<float>& parameters,
                               uint32_t frameIndex) const {
        CHECK(rendererRgb.size() == kWidth * kHeight * 3u);
        CHECK(parameters.size() % kParameterCount == 0u);
        const uint32_t parameterOffset = frameIndex * kParameterCount;
        CHECK(parameterOffset + kParameterCount <= parameters.size());
        std::vector<float> output(rendererRgb.size(), 0.0f);
        ScopedObjCRelease renderer{makeBuffer(
            device_, rendererRgb.data(), rendererRgb.size() * sizeof(float))};
        ScopedObjCRelease corrected{makeBuffer(
            device_, output.data(), output.size() * sizeof(float))};
        ScopedObjCRelease params{makeBuffer(
            device_, parameters.data(), parameters.size() * sizeof(float))};
        const std::array<uint32_t, 2> imageSize = {kWidth, kHeight};

        encodeAndWait([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:forward_];
            [encoder setBuffer:renderer.object offset:0 atIndex:0];
            [encoder setBuffer:corrected.object offset:0 atIndex:1];
            [encoder setBuffer:params.object offset:0 atIndex:2];
            [encoder setBytes:&parameterOffset
                       length:sizeof(parameterOffset) atIndex:3];
            [encoder setBytes:imageSize.data()
                       length:sizeof(imageSize) atIndex:4];
            [encoder setBytes:&kMaxAbsExposure
                       length:sizeof(kMaxAbsExposure) atIndex:5];
            [encoder setBytes:&kMaxAbsColorParameter
                       length:sizeof(kMaxAbsColorParameter) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
        });
        std::memcpy(output.data(), [corrected.object contents],
                    output.size() * sizeof(float));
        return output;
    }

    BackwardResult backward(
        const std::vector<float>& rendererRgb,
        const std::vector<float>& correctedGradient,
        const std::vector<float>& parameters,
        uint32_t frameIndex) const {
        CHECK(rendererRgb.size() == kWidth * kHeight * 3u);
        CHECK(correctedGradient.size() == rendererRgb.size());
        const uint32_t parameterOffset = frameIndex * kParameterCount;
        CHECK(parameterOffset + kParameterCount <= parameters.size());
        BackwardResult result;
        result.rendererGradient = rendererRgb;
        uint32_t status = 0;
        ScopedObjCRelease renderer{makeBuffer(
            device_, rendererRgb.data(), rendererRgb.size() * sizeof(float))};
        ScopedObjCRelease gradient{makeBuffer(
            device_, correctedGradient.data(),
            correctedGradient.size() * sizeof(float))};
        ScopedObjCRelease params{makeBuffer(
            device_, parameters.data(), parameters.size() * sizeof(float))};
        ScopedObjCRelease paramGradient{makeBuffer(
            device_, result.parameterGradient.data(),
            result.parameterGradient.size() * sizeof(float))};
        ScopedObjCRelease statusBuffer{makeBuffer(
            device_, &status, sizeof(status))};
        const std::array<uint32_t, 2> imageSize = {kWidth, kHeight};

        encodeAndWait([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:backward_];
            [encoder setBuffer:renderer.object offset:0 atIndex:0];
            [encoder setBuffer:gradient.object offset:0 atIndex:1];
            [encoder setBuffer:params.object offset:0 atIndex:2];
            [encoder setBuffer:paramGradient.object offset:0 atIndex:3];
            [encoder setBytes:&parameterOffset
                       length:sizeof(parameterOffset) atIndex:4];
            [encoder setBytes:imageSize.data()
                       length:sizeof(imageSize) atIndex:5];
            [encoder setBytes:&kMaxAbsExposure
                       length:sizeof(kMaxAbsExposure) atIndex:6];
            [encoder setBytes:&kMaxAbsColorParameter
                       length:sizeof(kMaxAbsColorParameter) atIndex:7];
            [encoder setBuffer:statusBuffer.object offset:0 atIndex:8];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
        });
        std::memcpy(result.rendererGradient.data(), [renderer.object contents],
                    result.rendererGradient.size() * sizeof(float));
        std::memcpy(result.parameterGradient.data(),
                    [paramGradient.object contents],
                    result.parameterGradient.size() * sizeof(float));
        std::memcpy(&result.status, [statusBuffer.object contents],
                    sizeof(result.status));
        return result;
    }

    AdamResult adam(const std::vector<float>& parameters,
                    const std::vector<float>& firstMoments,
                    const std::vector<float>& secondMoments,
                    const std::array<float, kParameterCount>& gradient,
                    uint32_t frameIndex, float stepSize,
                    float regularization, uint32_t status,
                    uint32_t attemptGatingEnabled) const {
        CHECK(parameters.size() == firstMoments.size());
        CHECK(parameters.size() == secondMoments.size());
        const uint32_t parameterOffset = frameIndex * kParameterCount;
        CHECK(parameterOffset + kParameterCount <= parameters.size());
        AdamResult result{parameters, firstMoments, secondMoments, status};
        ScopedObjCRelease params{makeBuffer(
            device_, parameters.data(), parameters.size() * sizeof(float))};
        ScopedObjCRelease grad{makeBuffer(
            device_, gradient.data(), gradient.size() * sizeof(float))};
        ScopedObjCRelease first{makeBuffer(
            device_, firstMoments.data(), firstMoments.size() * sizeof(float))};
        ScopedObjCRelease second{makeBuffer(
            device_, secondMoments.data(), secondMoments.size() * sizeof(float))};
        ScopedObjCRelease statusBuffer{makeBuffer(
            device_, &status, sizeof(status))};
        constexpr float beta1 = 0.9f;
        constexpr float beta2 = 0.999f;
        constexpr float biasCorrection2Sqrt = 0.0316227766f;
        constexpr float epsilon = 1.0e-8f;

        encodeAndWait([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:adam_];
            [encoder setBuffer:params.object offset:0 atIndex:0];
            [encoder setBuffer:grad.object offset:0 atIndex:1];
            [encoder setBuffer:first.object offset:0 atIndex:2];
            [encoder setBuffer:second.object offset:0 atIndex:3];
            [encoder setBytes:&parameterOffset
                       length:sizeof(parameterOffset) atIndex:4];
            [encoder setBytes:&stepSize length:sizeof(stepSize) atIndex:5];
            [encoder setBytes:&beta1 length:sizeof(beta1) atIndex:6];
            [encoder setBytes:&beta2 length:sizeof(beta2) atIndex:7];
            [encoder setBytes:&biasCorrection2Sqrt
                       length:sizeof(biasCorrection2Sqrt) atIndex:8];
            [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:9];
            [encoder setBytes:&regularization
                       length:sizeof(regularization) atIndex:10];
            [encoder setBytes:&kMaxAbsExposure
                       length:sizeof(kMaxAbsExposure) atIndex:11];
            [encoder setBytes:&kMaxAbsColorParameter
                       length:sizeof(kMaxAbsColorParameter) atIndex:12];
            [encoder setBuffer:statusBuffer.object offset:0 atIndex:13];
            [encoder setBytes:&attemptGatingEnabled
                       length:sizeof(attemptGatingEnabled) atIndex:14];
            [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        });
        std::memcpy(result.parameters.data(), [params.object contents],
                    result.parameters.size() * sizeof(float));
        std::memcpy(result.firstMoments.data(), [first.object contents],
                    result.firstMoments.size() * sizeof(float));
        std::memcpy(result.secondMoments.data(), [second.object contents],
                    result.secondMoments.size() * sizeof(float));
        std::memcpy(&result.status, [statusBuffer.object contents],
                    sizeof(result.status));
        return result;
    }

private:
    template <typename Encode>
    void encodeAndWait(Encode&& encode) const {
        id<MTLCommandBuffer> commandBuffer = [queue_ commandBuffer];
        CHECK(commandBuffer != nil);
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        CHECK(encoder != nil);
        encode(encoder);
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status == MTLCommandBufferStatusError)
            throw metalError("PPISP test command buffer failed",
                             commandBuffer.error);
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    id<MTLComputePipelineState> forward_ = nil;
    id<MTLComputePipelineState> backward_ = nil;
    id<MTLComputePipelineState> adam_ = nil;
};

float dotLoss(const std::vector<float>& image,
              const std::vector<float>& gradient) {
    CHECK(image.size() == gradient.size());
    float value = 0.0f;
    for (size_t index = 0; index < image.size(); ++index)
        value += image[index] * gradient[index];
    return value;
}

void checkForwardAndBackward(PpispHarness& harness) {
    const std::vector<float> renderer = {
        0.12f, 0.35f, 0.77f, 0.80f, 0.20f, 0.40f,
        0.50f, 0.90f, 0.10f, 0.33f, 0.44f, 0.55f,
        0.21f, 0.62f, 0.43f, 0.71f, 0.18f, 0.29f,
    };
    std::vector<float> parameters(2 * kParameterCount, 0.0f);
    const auto identity = harness.forward(renderer, parameters, 0);
    for (size_t index = 0; index < renderer.size(); ++index)
        // The donor's chromaticity normalization deliberately adds 1e-5 to
        // its intensity divisor, so zero latent color is identity to that
        // stabilizer's numerical tolerance rather than bit-exact identity.
        CHECK(nearlyEqual(identity[index], renderer[index], 3.0e-5f, 3.0e-5f));

    parameters[kParameterCount] = 1.0f;
    const auto doubled = harness.forward(renderer, parameters, 1);
    for (size_t index = 0; index < renderer.size(); ++index)
        CHECK(nearlyEqual(doubled[index], 2.0f * renderer[index],
                          3.0e-5f, 3.0e-5f));

    const std::array<float, kParameterCount> active = {
        0.20f, 0.13f, -0.09f, 0.07f, -0.04f,
        -0.11f, 0.05f, 0.03f, -0.06f,
    };
    std::copy(active.begin(), active.end(),
              parameters.begin() + kParameterCount);
    const std::vector<float> upstream = {
        0.70f, -0.20f, 0.40f, -0.30f, 0.60f, 0.10f,
        0.25f, -0.45f, 0.80f, -0.55f, 0.35f, -0.15f,
        0.42f, 0.17f, -0.63f, -0.27f, 0.51f, 0.31f,
    };
    const BackwardResult backward =
        harness.backward(renderer, upstream, parameters, 1);
    CHECK(backward.status == 0u);

    constexpr float epsilon = 1.0e-3f;
    for (uint32_t parameter = 0; parameter < kParameterCount; ++parameter) {
        auto plus = parameters;
        auto minus = parameters;
        plus[kParameterCount + parameter] += epsilon;
        minus[kParameterCount + parameter] -= epsilon;
        const float numerical = (
            dotLoss(harness.forward(renderer, plus, 1), upstream) -
            dotLoss(harness.forward(renderer, minus, 1), upstream)) /
            (2.0f * epsilon);
        CHECK(nearlyEqual(backward.parameterGradient[parameter], numerical,
                          9.0e-4f, 1.5e-2f));
    }

    for (size_t component = 0; component < renderer.size(); ++component) {
        auto plus = renderer;
        auto minus = renderer;
        plus[component] += epsilon;
        minus[component] -= epsilon;
        const float numerical = (
            dotLoss(harness.forward(plus, parameters, 1), upstream) -
            dotLoss(harness.forward(minus, parameters, 1), upstream)) /
            (2.0f * epsilon);
        CHECK(nearlyEqual(backward.rendererGradient[component], numerical,
                          9.0e-4f, 1.5e-2f));
    }

    // PPISP receives the unclamped loss cotangent, but the cotangent handed
    // back to the canonical raster result must still honor its upper clamp.
    auto clampRenderer = renderer;
    clampRenderer[0] = 1.0f;
    std::vector<float> identityParameters(2 * kParameterCount, 0.0f);
    const BackwardResult clampBackward = harness.backward(
        clampRenderer, upstream, identityParameters, 1);
    CHECK(clampBackward.rendererGradient[0] == 0.0f);
    CHECK(std::abs(clampBackward.rendererGradient[1]) > 1.0e-4f);

    parameters.assign(2 * kParameterCount, 0.0f);
    parameters[kParameterCount + 3] =
        std::numeric_limits<float>::quiet_NaN();
    const auto safe = harness.forward(renderer, parameters, 1);
    for (size_t index = 0; index < renderer.size(); ++index)
        CHECK(nearlyEqual(safe[index], renderer[index], 3.0e-5f, 3.0e-5f));
}

void checkAdam(PpispHarness& harness) {
    std::vector<float> parameters(2 * kParameterCount, 0.25f);
    std::vector<float> firstMoments(parameters.size(), 0.0f);
    std::vector<float> secondMoments(parameters.size(), 0.0f);
    std::array<float, kParameterCount> gradient{};
    for (uint32_t index = 0; index < kParameterCount; ++index)
        gradient[index] = index % 2u == 0u ? -100.0f : 100.0f;
    const AdamResult clamped = harness.adam(
        parameters, firstMoments, secondMoments, gradient, 1,
        100.0f, 0.0f, 0u, 1u);
    for (uint32_t index = 0; index < kParameterCount; ++index) {
        CHECK(clamped.parameters[index] == parameters[index]);
        const float expected = index % 2u == 0u ? 4.0f : -4.0f;
        CHECK(nearlyEqual(clamped.parameters[kParameterCount + index],
                          expected, 2.0e-6f, 2.0e-6f));
    }

    gradient.fill(0.0f);
    const AdamResult regularized = harness.adam(
        parameters, firstMoments, secondMoments, gradient, 1,
        1.0e-2f, 1.0e-2f, 0u, 1u);
    for (uint32_t index = 0; index < kParameterCount; ++index) {
        CHECK(std::abs(regularized.parameters[kParameterCount + index]) <
              std::abs(parameters[kParameterCount + index]));
    }

    gradient.fill(1.0f);
    const AdamResult retryGated = harness.adam(
        parameters, firstMoments, secondMoments, gradient, 1,
        1.0f, 0.0f, 1u, 1u);
    CHECK(retryGated.parameters == parameters);
    CHECK(retryGated.firstMoments == firstMoments);
    CHECK(retryGated.secondMoments == secondMoments);

    gradient.fill(1.0f);
    gradient[4] = std::numeric_limits<float>::quiet_NaN();
    const AdamResult nonfinite = harness.adam(
        parameters, firstMoments, secondMoments, gradient, 1,
        1.0f, 0.0f, 0u, 0u);
    CHECK((nonfinite.status & kAppearanceNonfinite) != 0u);
    CHECK(nonfinite.parameters == parameters);
    CHECK(nonfinite.firstMoments == firstMoments);
    CHECK(nonfinite.secondMoments == secondMoments);
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        id<MTLDevice> probeDevice = MTLCreateSystemDefaultDevice();
        if (!probeDevice) {
            std::cerr << "SKIP: no Metal device\n";
            return 77;
        }
        [probeDevice release];
        try {
            if (argc != 2)
                throw std::invalid_argument("Expected the metallib path");
            PpispHarness harness(argv[1]);
            checkForwardAndBackward(harness);
            checkAdam(harness);
            std::cout << "PPISP Metal refinement tests passed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}
