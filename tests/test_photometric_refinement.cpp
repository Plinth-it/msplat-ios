#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>

namespace {

constexpr double kMaxAbsLogGain = 1.3862943611198906; // log(4)

bool close(double actual, double expected, double tolerance = 1.0e-10) {
    return std::abs(actual - expected) <= tolerance;
}

bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

double gainFromLog(double logGain) {
    return std::exp(std::clamp(logGain, -kMaxAbsLogGain, kMaxAbsLogGain));
}

struct CameraAdam {
    std::array<std::array<double, 3>, 2> parameters{};
    std::array<std::array<double, 3>, 2> firstMoments{};
    std::array<std::array<double, 3>, 2> secondMoments{};
    std::array<unsigned, 2> visits{};
};

void update(CameraAdam& optimizer, size_t camera,
            const std::array<double, 3>& dataGradient,
            double regularization = 0.0, double learningRate = 1.0e-3) {
    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    constexpr double epsilon = 1.0e-8;

    const unsigned step = ++optimizer.visits[camera];
    const double biasCorrection1 = 1.0 - std::pow(beta1, step);
    const double biasCorrection2 = 1.0 - std::pow(beta2, step);

    for (size_t channel = 0; channel < 3; ++channel) {
        double& parameter = optimizer.parameters[camera][channel];
        double& firstMoment = optimizer.firstMoments[camera][channel];
        double& secondMoment = optimizer.secondMoments[camera][channel];
        const double gradient =
            dataGradient[channel] + regularization * parameter;
        firstMoment = beta1 * firstMoment + (1.0 - beta1) * gradient;
        secondMoment = beta2 * secondMoment +
            (1.0 - beta2) * gradient * gradient;
        const double denominator =
            std::sqrt(secondMoment) / std::sqrt(biasCorrection2) + epsilon;
        parameter = std::clamp(
            parameter - (learningRate / biasCorrection1) *
                firstMoment / denominator,
            -kMaxAbsLogGain, kMaxAbsLogGain);
    }
}

bool sameRow(const CameraAdam& lhs, const CameraAdam& rhs, size_t camera) {
    for (size_t channel = 0; channel < 3; ++channel) {
        if (!close(lhs.parameters[camera][channel],
                   rhs.parameters[camera][channel]) ||
            !close(lhs.firstMoments[camera][channel],
                   rhs.firstMoments[camera][channel]) ||
            !close(lhs.secondMoments[camera][channel],
                   rhs.secondMoments[camera][channel])) {
            return false;
        }
    }
    return lhs.visits[camera] == rhs.visits[camera];
}

bool testGainBoundsAndDecomposition() {
    bool ok = true;
    ok &= expect(close(gainFromLog(-100.0), 0.25),
                 "log gain must clamp to a 0.25x lower bound");
    ok &= expect(close(gainFromLog(100.0), 4.0),
                 "log gain must clamp to a 4x upper bound");
    ok &= expect(close(gainFromLog(std::log(1.5)), 1.5),
                 "in-range log gain must round-trip through exp");

    const std::array<double, 3> logGains{0.3, -0.15, 0.6};
    const double exposure =
        (logGains[0] + logGains[1] + logGains[2]) / 3.0;
    std::array<double, 3> colorResidual{};
    for (size_t channel = 0; channel < 3; ++channel)
        colorResidual[channel] = logGains[channel] - exposure;

    ok &= expect(close(colorResidual[0] + colorResidual[1] + colorResidual[2],
                       0.0),
                 "channel residuals must have zero mean");
    for (size_t channel = 0; channel < 3; ++channel) {
        ok &= expect(close(exposure + colorResidual[channel],
                           logGains[channel]),
                     "exposure and channel residual must reconstruct log gain");
    }
    return ok;
}

bool testLogGainChainRule() {
    constexpr double raw = 0.63;
    constexpr double target = 0.21;
    constexpr double logGain = 0.27;
    constexpr double epsilon = 1.0e-6;
    auto loss = [](double value) {
        const double adjusted = raw * std::exp(value);
        const double residual = adjusted - target;
        return 0.5 * residual * residual;
    };

    const double adjusted = raw * std::exp(logGain);
    const double adjustedGradient = adjusted - target;
    const double analytic = adjustedGradient * adjusted;
    const double finiteDifference =
        (loss(logGain + epsilon) - loss(logGain - epsilon)) /
        (2.0 * epsilon);
    return expect(close(analytic, finiteDifference, 1.0e-9),
                  "dL/dlogGain must equal dL/dadjusted times adjusted RGB");
}

bool testPureL1GradientsAndIdentity() {
    constexpr double raw = 0.4;
    constexpr double target = 0.8;
    constexpr double logGain = 0.2;
    constexpr double epsilon = 1.0e-6;
    const double gain = std::exp(logGain);
    const double adjusted = raw * gain;
    const double adjustedGradient = adjusted < target ? -1.0 : 1.0;

    auto lossFromRaw = [&](double value) {
        return std::abs(value * gain - target);
    };
    auto lossFromLogGain = [&](double value) {
        return std::abs(raw * std::exp(value) - target);
    };
    const double rawFiniteDifference =
        (lossFromRaw(raw + epsilon) - lossFromRaw(raw - epsilon)) /
        (2.0 * epsilon);
    const double logFiniteDifference =
        (lossFromLogGain(logGain + epsilon) -
         lossFromLogGain(logGain - epsilon)) /
        (2.0 * epsilon);

    bool ok = true;
    ok &= expect(close(rawFiniteDifference, adjustedGradient * gain, 1.0e-9),
                 "L1 dL/draw must chain through the RGB gain");
    ok &= expect(close(logFiniteDifference,
                       adjustedGradient * adjusted, 1.0e-9),
                 "L1 dL/dlogGain must use adjusted RGB");
    ok &= expect(close(gainFromLog(0.0), 1.0),
                 "disabled zero log gain must preserve rendered RGB exactly");
    return ok;
}

bool testPerCameraAdamIsolation() {
    const std::array<double, 3> camera0Gradient{0.6, -0.25, 0.1};
    const std::array<double, 3> camera1Gradient{-0.4, 0.7, -0.2};
    CameraAdam alternating;
    CameraAdam isolated0;
    CameraAdam isolated1;
    const CameraAdam untouched;

    update(alternating, 0, camera0Gradient);
    bool ok = expect(sameRow(alternating, untouched, 1),
                     "updating camera 0 must leave camera 1 inactive");
    const CameraAdam afterCamera0 = alternating;
    update(alternating, 1, camera1Gradient);
    update(isolated0, 0, camera0Gradient);
    update(isolated1, 1, camera1Gradient);

    ok &= expect(sameRow(alternating, isolated0, 0),
                 "camera 0 first update must match an isolated first update");
    ok &= expect(sameRow(alternating, isolated1, 1),
                 "camera 1 first update must match an isolated first update");
    ok &= expect(sameRow(alternating, afterCamera0, 0),
                 "updating camera 1 must not mutate camera 0 state");
    return ok;
}

bool testRegularizationAndClamp() {
    bool ok = true;
    CameraAdam regularized;
    regularized.parameters[0] = {0.4, -0.3, 0.2};
    const auto before = regularized.parameters[0];
    update(regularized, 0, {0.0, 0.0, 0.0}, 1.0e-2);
    for (size_t channel = 0; channel < 3; ++channel) {
        ok &= expect(std::abs(regularized.parameters[0][channel]) <
                         std::abs(before[channel]),
                     "L2-only update must pull log gain toward zero");
    }

    CameraAdam upper;
    upper.parameters[0] = {
        kMaxAbsLogGain - 0.01, kMaxAbsLogGain - 0.01,
        kMaxAbsLogGain - 0.01};
    update(upper, 0, {-1.0, -1.0, -1.0}, 0.0, 10.0);
    CameraAdam lower;
    lower.parameters[0] = {
        -kMaxAbsLogGain + 0.01, -kMaxAbsLogGain + 0.01,
        -kMaxAbsLogGain + 0.01};
    update(lower, 0, {1.0, 1.0, 1.0}, 0.0, 10.0);
    for (size_t channel = 0; channel < 3; ++channel) {
        ok &= expect(close(upper.parameters[0][channel], kMaxAbsLogGain),
                     "Adam must clamp positive log gain");
        ok &= expect(close(lower.parameters[0][channel], -kMaxAbsLogGain),
                     "Adam must clamp negative log gain");
    }
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= testGainBoundsAndDecomposition();
    ok &= testLogGainChainRule();
    ok &= testPureL1GradientsAndIdentity();
    ok &= testPerCameraAdamIsolation();
    ok &= testRegularizationAndClamp();
    if (ok) std::cout << "Photometric refinement math tests passed\n";
    return ok ? 0 : 1;
}
