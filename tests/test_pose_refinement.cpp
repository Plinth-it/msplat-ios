#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Vec2 = std::array<double, 2>;
using Vec3 = std::array<double, 3>;
using Mat2 = std::array<double, 4>;
using Mat23 = std::array<double, 6>;
using Mat3 = std::array<double, 9>;

constexpr double kMaxTranslation = 0.05;
constexpr double kMaxRotation = 0.05235987755982989; // 3 degrees

struct Rigid {
    Mat3 rotation{};
    Vec3 translation{};
};

[[noreturn]] void fail(const char* expression, int line) {
    throw std::runtime_error(
        "line " + std::to_string(line) + ": " + expression);
}

#define CHECK(condition) \
    do { if (!(condition)) fail(#condition, __LINE__); } while (false)

bool near(double lhs, double rhs, double absoluteTolerance = 1.0e-9,
          double relativeTolerance = 1.0e-9) {
    return std::abs(lhs - rhs) <= absoluteTolerance +
        relativeTolerance * std::max(std::abs(lhs), std::abs(rhs));
}

void checkNear(const Vec3& actual, const Vec3& expected,
               double tolerance = 1.0e-9) {
    for (size_t axis = 0; axis < 3; ++axis)
        CHECK(near(actual[axis], expected[axis], tolerance, tolerance));
}

Mat3 identity3() {
    return {1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0};
}

Vec3 add(const Vec3& lhs, const Vec3& rhs) {
    return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

Vec3 scale(const Vec3& value, double factor) {
    return {factor * value[0], factor * value[1], factor * value[2]};
}

double dot(const Vec3& lhs, const Vec3& rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

double norm(const Vec3& value) {
    return std::sqrt(dot(value, value));
}

Vec3 cross(const Vec3& lhs, const Vec3& rhs) {
    return {
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    };
}

Mat3 add(const Mat3& lhs, const Mat3& rhs) {
    Mat3 result{};
    for (size_t index = 0; index < result.size(); ++index)
        result[index] = lhs[index] + rhs[index];
    return result;
}

Mat3 scale(const Mat3& value, double factor) {
    Mat3 result{};
    for (size_t index = 0; index < result.size(); ++index)
        result[index] = factor * value[index];
    return result;
}

Mat3 multiply(const Mat3& lhs, const Mat3& rhs) {
    Mat3 result{};
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            for (size_t inner = 0; inner < 3; ++inner) {
                result[row * 3 + column] +=
                    lhs[row * 3 + inner] * rhs[inner * 3 + column];
            }
        }
    }
    return result;
}

Vec3 multiply(const Mat3& matrix, const Vec3& vector) {
    return {
        matrix[0] * vector[0] + matrix[1] * vector[1] +
            matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] +
            matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] +
            matrix[8] * vector[2],
    };
}

Mat3 transpose(const Mat3& matrix) {
    return {
        matrix[0], matrix[3], matrix[6],
        matrix[1], matrix[4], matrix[7],
        matrix[2], matrix[5], matrix[8],
    };
}

Vec3 column(const Mat3& matrix, size_t index) {
    return {matrix[index], matrix[3 + index], matrix[6 + index]};
}

double determinant(const Mat3& matrix) {
    return
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
        matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
        matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

Mat3 skew(const Vec3& vector) {
    return {
        0.0, -vector[2], vector[1],
        vector[2], 0.0, -vector[0],
        -vector[1], vector[0], 0.0,
    };
}

Mat3 so3Exp(const Vec3& rotation) {
    const double theta2 = dot(rotation, rotation);
    double a = 0.0;
    double b = 0.0;
    if (theta2 < 1.0e-8) {
        const double theta4 = theta2 * theta2;
        a = 1.0 - theta2 / 6.0 + theta4 / 120.0;
        b = 0.5 - theta2 / 24.0 + theta4 / 720.0;
    } else {
        const double theta = std::sqrt(theta2);
        a = std::sin(theta) / theta;
        b = (1.0 - std::cos(theta)) / theta2;
    }
    const Mat3 omega = skew(rotation);
    return add(identity3(), add(scale(omega, a),
                                scale(multiply(omega, omega), b)));
}

Vec3 se3ExpTranslation(const Vec3& translation,
                       const Vec3& rotation) {
    const double theta2 = dot(rotation, rotation);
    double b = 0.0;
    double c = 0.0;
    if (theta2 < 1.0e-8) {
        const double theta4 = theta2 * theta2;
        b = 0.5 - theta2 / 24.0 + theta4 / 720.0;
        c = 1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0;
    } else {
        const double theta = std::sqrt(theta2);
        b = (1.0 - std::cos(theta)) / theta2;
        c = (theta - std::sin(theta)) / (theta2 * theta);
    }
    const Mat3 omega = skew(rotation);
    const Mat3 jacobian = add(
        identity3(), add(scale(omega, b),
                         scale(multiply(omega, omega), c)));
    return multiply(jacobian, translation);
}

Rigid se3Exp(const Vec3& translation, const Vec3& rotation) {
    return {so3Exp(rotation), se3ExpTranslation(translation, rotation)};
}

Vec3 so3Log(const Mat3& rotation) {
    const Vec3 sineAxis = {
        0.5 * (rotation[7] - rotation[5]),
        0.5 * (rotation[2] - rotation[6]),
        0.5 * (rotation[3] - rotation[1]),
    };
    const double sine = norm(sineAxis);
    const double cosine = std::clamp(
        0.5 * (rotation[0] + rotation[4] + rotation[8] - 1.0),
        -1.0, 1.0);
    if (sine < 1.0e-12) return sineAxis;
    const double angle = std::atan2(sine, cosine);
    return scale(sineAxis, angle / sine);
}

Rigid compose(const Rigid& lhs, const Rigid& rhs) {
    return {
        multiply(lhs.rotation, rhs.rotation),
        add(multiply(lhs.rotation, rhs.translation), lhs.translation),
    };
}

Rigid inverse(const Rigid& transform) {
    const Mat3 inverseRotation = transpose(transform.rotation);
    return {inverseRotation,
            scale(multiply(inverseRotation, transform.translation), -1.0)};
}

Vec3 transformPoint(const Rigid& transform, const Vec3& point) {
    return add(multiply(transform.rotation, point), transform.translation);
}

Rigid viewFromOpenGLCameraToWorld(const Rigid& cameraToWorld) {
    const Mat3 flip = {
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
    };
    const Mat3 rotation = multiply(flip, transpose(cameraToWorld.rotation));
    return {rotation,
            scale(multiply(rotation, cameraToWorld.translation), -1.0)};
}

Vec3 cameraPosition(const Rigid& view) {
    return scale(multiply(transpose(view.rotation), view.translation), -1.0);
}

Vec2 project(const Vec3& viewPoint, double fx, double fy,
             double cx, double cy) {
    return {
        fx * viewPoint[0] / viewPoint[2] + cx - 0.5,
        fy * viewPoint[1] / viewPoint[2] + cy - 0.5,
    };
}

Vec3 projectionVjp(const Vec3& viewPoint, double fx, double fy,
                   const Vec2& upstream) {
    const double reciprocalZ = 1.0 / viewPoint[2];
    const double reciprocalZ2 = reciprocalZ * reciprocalZ;
    return {
        fx * reciprocalZ * upstream[0],
        fy * reciprocalZ * upstream[1],
        -fx * viewPoint[0] * reciprocalZ2 * upstream[0] -
            fy * viewPoint[1] * reciprocalZ2 * upstream[1],
    };
}

void checkRigid(const Rigid& transform, double tolerance = 1.0e-10) {
    const Mat3 product = multiply(
        transform.rotation, transpose(transform.rotation));
    const Mat3 identity = identity3();
    for (size_t index = 0; index < product.size(); ++index)
        CHECK(near(product[index], identity[index], tolerance, tolerance));
    CHECK(near(determinant(transform.rotation), 1.0,
               tolerance, tolerance));
    for (double value : transform.rotation) CHECK(std::isfinite(value));
    for (double value : transform.translation) CHECK(std::isfinite(value));
}

Rigid projectBounds(Rigid correction) {
    const double translationNorm = norm(correction.translation);
    if (translationNorm > kMaxTranslation) {
        correction.translation = scale(
            correction.translation, kMaxTranslation / translationNorm);
    }
    Vec3 rotation = so3Log(correction.rotation);
    const double rotationNorm = norm(rotation);
    if (rotationNorm > kMaxRotation)
        rotation = scale(rotation, kMaxRotation / rotationNorm);
    correction.rotation = so3Exp(rotation);
    return correction;
}

double regularizer(const Rigid& correction, double lambda) {
    double rotationDistance2 = 0.0;
    const Mat3 identity = identity3();
    for (size_t index = 0; index < identity.size(); ++index) {
        const double difference = correction.rotation[index] - identity[index];
        rotationDistance2 += difference * difference;
    }
    return 0.5 * lambda * dot(correction.translation,
                              correction.translation) /
            (kMaxTranslation * kMaxTranslation) +
        0.25 * lambda * rotationDistance2 /
            (kMaxRotation * kMaxRotation);
}

std::array<double, 6> regularizerGradient(
    const Rigid& correction, double lambda) {
    return {
        lambda * correction.translation[0] /
            (kMaxTranslation * kMaxTranslation),
        lambda * correction.translation[1] /
            (kMaxTranslation * kMaxTranslation),
        lambda * correction.translation[2] /
            (kMaxTranslation * kMaxTranslation),
        lambda * 0.5 *
            (correction.rotation[7] - correction.rotation[5]) /
            (kMaxRotation * kMaxRotation),
        lambda * 0.5 *
            (correction.rotation[2] - correction.rotation[6]) /
            (kMaxRotation * kMaxRotation),
        lambda * 0.5 *
            (correction.rotation[3] - correction.rotation[1]) /
            (kMaxRotation * kMaxRotation),
    };
}

Rigid perturb(const Rigid& transform, size_t component, double amount) {
    Vec3 translation{};
    Vec3 rotation{};
    if (component < 3)
        translation[component] = amount;
    else
        rotation[component - 3] = amount;
    return compose(se3Exp(translation, rotation), transform);
}

void testCoordinateConvention() {
    const Rigid publicIdentity{identity3(), {0.0, 0.0, 0.0}};
    const Rigid declaredView = viewFromOpenGLCameraToWorld(publicIdentity);
    const Mat3 expectedFlip = {
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
    };
    CHECK(declaredView.rotation == expectedFlip);
    CHECK(declaredView.translation == Vec3({0.0, 0.0, 0.0}));

    const Vec3 point{0.2, -0.1, -2.0};
    const Vec3 viewPoint = transformPoint(declaredView, point);
    checkNear(viewPoint, {0.2, 0.1, 2.0});
    const Vec2 pixel = project(viewPoint, 400.0, 300.0, 320.0, 240.0);
    CHECK(near(pixel[0], 359.5));
    CHECK(near(pixel[1], 254.5));

    const Rigid correction{identity3(), {0.01, 0.0, 0.0}};
    const Rigid refinedView = compose(correction, declaredView);
    checkNear(cameraPosition(refinedView), {-0.01, 0.0, 0.0});
    const Vec2 refinedPixel = project(
        transformPoint(refinedView, point),
        400.0, 300.0, 320.0, 240.0);
    CHECK(near(refinedPixel[0], 361.5));
    CHECK(near(refinedPixel[1], 254.5));
}

void testExponentialAndRigidity() {
    const Vec3 translation{0.02, -0.01, 0.03};
    const Rigid pureTranslation = se3Exp(translation, {0.0, 0.0, 0.0});
    CHECK(pureTranslation.rotation == identity3());
    CHECK(pureTranslation.translation == translation);

    const Vec3 tinyRotation{1.0e-9, -2.0e-9, 3.0e-9};
    const Rigid tiny = se3Exp(translation, tinyRotation);
    const Mat3 omega = skew(tinyRotation);
    const Vec3 expectedTinyTranslation = add(
        translation,
        add(scale(multiply(omega, translation), 0.5),
            scale(multiply(multiply(omega, omega), translation), 1.0 / 6.0)));
    checkNear(tiny.translation, expectedTinyTranslation, 1.0e-15);
    checkRigid(tiny, 1.0e-12);

    const Vec3 rotation{0.2, -0.1, 0.05};
    const Rigid transform = se3Exp(translation, rotation);
    checkRigid(transform);
    const Rigid analyticInverse = se3Exp(scale(translation, -1.0),
                                         scale(rotation, -1.0));
    const Rigid product = compose(transform, analyticInverse);
    checkRigid(product);
    for (size_t index = 0; index < 9; ++index)
        CHECK(near(product.rotation[index], identity3()[index], 1.0e-11));
    checkNear(product.translation, {0.0, 0.0, 0.0}, 1.0e-11);
    const Rigid directInverse = inverse(transform);
    for (size_t index = 0; index < 9; ++index)
        CHECK(near(analyticInverse.rotation[index],
                   directInverse.rotation[index], 1.0e-11));
    checkNear(analyticInverse.translation,
              directInverse.translation, 1.0e-11);
}

void testProjectedPointGradient() {
    const Rigid view{{
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
    }, {0.0, 0.0, 0.0}};
    const Vec3 point{0.2, -0.1, -2.0};
    const Vec2 upstream{0.7, -1.2};
    const Vec3 viewPoint = transformPoint(view, point);
    const Vec3 translationGradient =
        projectionVjp(viewPoint, 400.0, 300.0, upstream);
    const Vec3 rotationGradient = cross(viewPoint, translationGradient);
    checkNear(translationGradient, {140.0, -180.0, -5.0}, 1.0e-10);
    checkNear(rotationGradient, {359.5, 281.0, -50.0}, 1.0e-10);

    const auto objective = [&](const Rigid& candidate) {
        const Vec2 pixel = project(
            transformPoint(candidate, point),
            400.0, 300.0, 320.0, 240.0);
        return pixel[0] * upstream[0] + pixel[1] * upstream[1];
    };
    const std::array<double, 6> analytic = {
        translationGradient[0], translationGradient[1],
        translationGradient[2], rotationGradient[0],
        rotationGradient[1], rotationGradient[2],
    };
    constexpr double epsilon = 1.0e-6;
    for (size_t component = 0; component < analytic.size(); ++component) {
        const double finiteDifference =
            (objective(perturb(view, component, epsilon)) -
             objective(perturb(view, component, -epsilon))) /
            (2.0 * epsilon);
        CHECK(near(finiteDifference, analytic[component],
                   2.0e-7, 2.0e-9));
    }
}

double covarianceObjective(const Rigid& view, const Vec3& point,
                           const Mat3& worldCovariance,
                           double fx, double fy, const Mat2& upstream) {
    const Vec3 p = transformPoint(view, point);
    const Mat23 jacobian = {
        fx / p[2], 0.0, -fx * p[0] / (p[2] * p[2]),
        0.0, fy / p[2], -fy * p[1] / (p[2] * p[2]),
    };
    const Mat3 cameraCovariance = multiply(
        multiply(view.rotation, worldCovariance),
        transpose(view.rotation));
    Mat2 projected{};
    for (size_t row = 0; row < 2; ++row) {
        for (size_t columnIndex = 0; columnIndex < 2; ++columnIndex) {
            for (size_t lhs = 0; lhs < 3; ++lhs) {
                for (size_t rhs = 0; rhs < 3; ++rhs) {
                    projected[row * 2 + columnIndex] +=
                        jacobian[row * 3 + lhs] *
                        cameraCovariance[lhs * 3 + rhs] *
                        jacobian[columnIndex * 3 + rhs];
                }
            }
        }
    }
    double result = 0.0;
    for (size_t index = 0; index < projected.size(); ++index)
        result += upstream[index] * projected[index];
    return result;
}

std::array<double, 6> covarianceGradient(
    const Rigid& view, const Vec3& point, const Mat3& worldCovariance,
    double fx, double fy, const Mat2& upstream) {
    const Vec3 p = transformPoint(view, point);
    const double reciprocalZ = 1.0 / p[2];
    const double reciprocalZ2 = reciprocalZ * reciprocalZ;
    const double reciprocalZ3 = reciprocalZ2 * reciprocalZ;
    const Mat23 jacobian = {
        fx * reciprocalZ, 0.0, -fx * p[0] * reciprocalZ2,
        0.0, fy * reciprocalZ, -fy * p[1] * reciprocalZ2,
    };
    const Mat3 cameraCovariance = multiply(
        multiply(view.rotation, worldCovariance),
        transpose(view.rotation));

    Mat23 vJacobian{};
    for (size_t row = 0; row < 2; ++row) {
        for (size_t columnIndex = 0; columnIndex < 3; ++columnIndex) {
            for (size_t otherRow = 0; otherRow < 2; ++otherRow) {
                for (size_t inner = 0; inner < 3; ++inner) {
                    vJacobian[row * 3 + columnIndex] +=
                        2.0 * upstream[row * 2 + otherRow] *
                        jacobian[otherRow * 3 + inner] *
                        cameraCovariance[inner * 3 + columnIndex];
                }
            }
        }
    }
    const Vec3 positionGradient = {
        -fx * reciprocalZ2 * vJacobian[2],
        -fy * reciprocalZ2 * vJacobian[5],
        -fx * reciprocalZ2 * vJacobian[0] +
            2.0 * fx * p[0] * reciprocalZ3 * vJacobian[2] -
            fy * reciprocalZ2 * vJacobian[4] +
            2.0 * fy * p[1] * reciprocalZ3 * vJacobian[5],
    };

    Mat3 vCameraCovariance{};
    for (size_t row = 0; row < 3; ++row) {
        for (size_t columnIndex = 0; columnIndex < 3; ++columnIndex) {
            for (size_t lhs = 0; lhs < 2; ++lhs) {
                for (size_t rhs = 0; rhs < 2; ++rhs) {
                    vCameraCovariance[row * 3 + columnIndex] +=
                        jacobian[lhs * 3 + row] *
                        upstream[lhs * 2 + rhs] *
                        jacobian[rhs * 3 + columnIndex];
                }
            }
        }
    }
    const Mat3 vRotation = scale(
        multiply(multiply(vCameraCovariance, view.rotation),
                 worldCovariance),
        2.0);
    Vec3 directRotation{};
    for (size_t columnIndex = 0; columnIndex < 3; ++columnIndex) {
        directRotation = add(
            directRotation,
            cross(column(view.rotation, columnIndex),
                  column(vRotation, columnIndex)));
    }
    const Vec3 rotationGradient = add(
        cross(p, positionGradient), directRotation);
    return {
        positionGradient[0], positionGradient[1], positionGradient[2],
        rotationGradient[0], rotationGradient[1], rotationGradient[2],
    };
}

void testCovarianceGradient() {
    const Rigid view{
        so3Exp({0.1, -0.05, 0.02}),
        {0.2, -0.1, 0.3},
    };
    const Vec3 point{0.3, -0.2, 1.7};
    const Mat3 covariance = {
        0.04, 0.005, -0.002,
        0.005, 0.02, 0.003,
        -0.002, 0.003, 0.03,
    };
    const Mat2 upstream = {0.7, -0.2, -0.2, 0.4};
    const std::array<double, 6> analytic = covarianceGradient(
        view, point, covariance, 300.0, 280.0, upstream);
    const std::array<double, 6> golden = {
        160.506022837995, -106.722860941448, -883.707180965911,
        478.451849968269, 695.882328799996, -216.818000995399,
    };
    for (size_t component = 0; component < analytic.size(); ++component)
        CHECK(near(analytic[component], golden[component], 2.0e-8, 2.0e-10));

    constexpr double epsilon = 1.0e-6;
    for (size_t component = 0; component < analytic.size(); ++component) {
        const double finiteDifference =
            (covarianceObjective(
                 perturb(view, component, epsilon), point, covariance,
                 300.0, 280.0, upstream) -
             covarianceObjective(
                 perturb(view, component, -epsilon), point, covariance,
                 300.0, 280.0, upstream)) /
            (2.0 * epsilon);
        CHECK(near(finiteDifference, analytic[component],
                   5.0e-5, 5.0e-8));
    }
}

void testRegularizationAndBounds() {
    constexpr double lambda = 1.0e-3;
    const Rigid correction{
        so3Exp({0.02, -0.01, 0.03}),
        {0.02, -0.03, 0.01},
    };
    const std::array<double, 6> analytic =
        regularizerGradient(correction, lambda);
    constexpr double epsilon = 1.0e-7;
    for (size_t component = 0; component < analytic.size(); ++component) {
        const double finiteDifference =
            (regularizer(perturb(correction, component, epsilon), lambda) -
             regularizer(perturb(correction, component, -epsilon), lambda)) /
            (2.0 * epsilon);
        CHECK(near(finiteDifference, analytic[component],
                   5.0e-9, 5.0e-8));
    }

    Rigid bounded = projectBounds({
        so3Exp({0.4, -0.2, 0.1}),
        {0.3, -0.4, 0.5},
    });
    CHECK(near(norm(bounded.translation), kMaxTranslation, 1.0e-12));
    CHECK(near(norm(so3Log(bounded.rotation)), kMaxRotation, 1.0e-12));
    checkRigid(bounded);

    for (size_t iteration = 0; iteration < 2000; ++iteration) {
        const Vec3 translationStep{
            0.01 * std::sin(static_cast<double>(iteration)),
            -0.008,
            0.006,
        };
        const Vec3 rotationStep{0.01, -0.007, 0.004};
        bounded = projectBounds(compose(
            se3Exp(translationStep, rotationStep), bounded));
        CHECK(norm(bounded.translation) <= kMaxTranslation + 1.0e-12);
        CHECK(norm(so3Log(bounded.rotation)) <= kMaxRotation + 1.0e-12);
        checkRigid(bounded, 2.0e-10);
    }
}

void testDescentSignAndCanonicalIndexIsolation() {
    const Rigid declaredView{{
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
    }, {0.0, 0.0, 0.0}};
    const Vec3 point{0.2, -0.1, -2.0};
    const double targetX = 359.5;
    Rigid correction{identity3(), {0.01, 0.0, 0.0}};
    const auto loss = [&](const Rigid& candidateCorrection) {
        const Vec2 pixel = project(
            transformPoint(compose(candidateCorrection, declaredView), point),
            400.0, 300.0, 320.0, 240.0);
        const double residual = pixel[0] - targetX;
        return 0.5 * residual * residual;
    };
    const Rigid refinedView = compose(correction, declaredView);
    const Vec3 p = transformPoint(refinedView, point);
    const double residual = project(p, 400.0, 300.0, 320.0, 240.0)[0] -
        targetX;
    const Vec3 translationGradient = projectionVjp(
        p, 400.0, 300.0, {residual, 0.0});
    const Vec3 rotationGradient = cross(p, translationGradient);
    constexpr double learningRate = 1.0e-6;
    const Rigid descentIncrement = se3Exp(
        scale(translationGradient, -learningRate),
        scale(rotationGradient, -learningRate));
    const Rigid updated = projectBounds(compose(descentIncrement, correction));
    CHECK(loss(updated) < loss(correction));

    std::array<Rigid, 5> cameraCorrections{};
    for (Rigid& value : cameraCorrections)
        value = {identity3(), {0.0, 0.0, 0.0}};
    const std::array<size_t, 3> trainIndices{4, 1, 3};
    const size_t anchorCameraIndex = trainIndices.front();
    const size_t localCameraIndex = 1;
    const size_t canonicalCameraIndex = trainIndices[localCameraIndex];
    if (canonicalCameraIndex != anchorCameraIndex)
        cameraCorrections[canonicalCameraIndex] = updated;
    CHECK(norm(cameraCorrections[canonicalCameraIndex].translation) > 0.0);
    CHECK(cameraCorrections[anchorCameraIndex].rotation == identity3());
    CHECK(cameraCorrections[anchorCameraIndex].translation ==
          Vec3({0.0, 0.0, 0.0}));
    for (size_t index : {size_t{0}, size_t{2}, size_t{3}}) {
        CHECK(cameraCorrections[index].rotation == identity3());
        CHECK(cameraCorrections[index].translation ==
              Vec3({0.0, 0.0, 0.0}));
    }
}

} // namespace

int main() {
    try {
        testCoordinateConvention();
        testExponentialAndRigidity();
        testProjectedPointGradient();
        testCovarianceGradient();
        testRegularizationAndBounds();
        testDescentSignAndCanonicalIndexIsolation();
        std::cout << "Pose refinement math tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "msplat pose refinement test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
