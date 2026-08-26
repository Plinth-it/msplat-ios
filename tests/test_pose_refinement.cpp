#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../core/include/camp_pose_point_sampler.hpp"
#include "../core/include/camp_pose_preconditioner.hpp"
#include "../core/include/pose_refinement_state.hpp"

namespace {

using Vec2 = std::array<double, 2>;
using Vec3 = std::array<double, 3>;
using Mat2 = std::array<double, 4>;
using Mat23 = std::array<double, 6>;
using Mat3 = std::array<double, 9>;
using Mat6 = msplat::detail::CampPoseMatrix;

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

Rigid openGLCameraToWorldFromView(const Rigid& view) {
    const Mat3 flip = {
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
    };
    const Rigid openCVCameraToWorld = inverse(view);
    return {
        multiply(openCVCameraToWorld.rotation, flip),
        openCVCameraToWorld.translation,
    };
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

void testPoseStateReadbackGeometry() {
    CHECK(msplat::detail::poseRefinementStateCount(false, 3) == 0u);
    CHECK(msplat::detail::poseRefinementStateCount(true, 3) == 3u);
    bool invalidCountRejected = false;
    try {
        (void)msplat::detail::poseRefinementStateCount(true, 0);
    } catch (const std::invalid_argument&) {
        invalidCountRejected = true;
    }
    CHECK(invalidCountRejected);

    constexpr float normalizationScale = 2.5f;
    const float normalizationTranslation[3] = {10.0f, -4.0f, 3.0f};
    const Vec3 originalPosition{11.2, -3.5, 4.1};
    const Vec3 normalizedPosition{
        (originalPosition[0] - normalizationTranslation[0]) *
            normalizationScale,
        (originalPosition[1] - normalizationTranslation[1]) *
            normalizationScale,
        (originalPosition[2] - normalizationTranslation[2]) *
            normalizationScale,
    };
    const Mat3 baseRotation = so3Exp({0.1, -0.05, 0.02});
    float baseCameraToWorld[16] = {
        static_cast<float>(baseRotation[0]),
        static_cast<float>(baseRotation[1]),
        static_cast<float>(baseRotation[2]),
        static_cast<float>(normalizedPosition[0]),
        static_cast<float>(baseRotation[3]),
        static_cast<float>(baseRotation[4]),
        static_cast<float>(baseRotation[5]),
        static_cast<float>(normalizedPosition[1]),
        static_cast<float>(baseRotation[6]),
        static_cast<float>(baseRotation[7]),
        static_cast<float>(baseRotation[8]),
        static_cast<float>(normalizedPosition[2]),
        0.0f, 0.0f, 0.0f, 1.0f,
    };

    const float anchorDelta[6] = {};
    const auto anchor = msplat::detail::makePoseRefinementGeometry(
        baseCameraToWorld, anchorDelta, normalizationScale,
        normalizationTranslation);
    CHECK(anchor.translationNorm == 0.0f);
    CHECK(anchor.rotationNorm == 0.0f);
    for (float value : anchor.poseDelta) CHECK(value == 0.0f);
    for (size_t row = 0; row < 3; ++row) {
        for (size_t columnIndex = 0; columnIndex < 3; ++columnIndex) {
            CHECK(near(
                anchor.correctedCameraToWorld[row * 4 + columnIndex],
                baseRotation[row * 3 + columnIndex], 2.0e-6, 2.0e-6));
        }
        CHECK(near(anchor.correctedCameraToWorld[row * 4 + 3],
                   originalPosition[row], 2.0e-6, 2.0e-6));
    }
    CHECK(anchor.correctedCameraToWorld[12] == 0.0f);
    CHECK(anchor.correctedCameraToWorld[13] == 0.0f);
    CHECK(anchor.correctedCameraToWorld[14] == 0.0f);
    CHECK(anchor.correctedCameraToWorld[15] == 1.0f);

    const float normalizedDelta[6] = {
        0.025f, -0.0125f, 0.005f,
        0.02f, -0.01f, 0.03f,
    };
    const auto corrected = msplat::detail::makePoseRefinementGeometry(
        baseCameraToWorld, normalizedDelta, normalizationScale,
        normalizationTranslation);
    const Rigid baseCameraToWorldRigid{baseRotation, normalizedPosition};
    const Rigid declaredView =
        viewFromOpenGLCameraToWorld(baseCameraToWorldRigid);
    const Rigid correction{
        so3Exp({normalizedDelta[3], normalizedDelta[4], normalizedDelta[5]}),
        {normalizedDelta[0], normalizedDelta[1], normalizedDelta[2]},
    };
    Rigid expected = openGLCameraToWorldFromView(
        compose(correction, declaredView));
    for (size_t axis = 0; axis < 3; ++axis) {
        expected.translation[axis] =
            expected.translation[axis] / normalizationScale +
            normalizationTranslation[axis];
    }

    for (size_t row = 0; row < 3; ++row) {
        for (size_t columnIndex = 0; columnIndex < 3; ++columnIndex) {
            CHECK(near(
                corrected.correctedCameraToWorld[row * 4 + columnIndex],
                expected.rotation[row * 3 + columnIndex],
                2.0e-6, 2.0e-6));
        }
        CHECK(near(corrected.correctedCameraToWorld[row * 4 + 3],
                   expected.translation[row], 2.0e-6, 2.0e-6));
        CHECK(near(corrected.poseDelta[row],
                   normalizedDelta[row] / normalizationScale,
                   1.0e-7, 1.0e-7));
        CHECK(near(corrected.poseDelta[row + 3], normalizedDelta[row + 3],
                   1.0e-7, 1.0e-7));
    }
    CHECK(near(corrected.translationNorm,
               std::sqrt(
                   corrected.poseDelta[0] * corrected.poseDelta[0] +
                   corrected.poseDelta[1] * corrected.poseDelta[1] +
                   corrected.poseDelta[2] * corrected.poseDelta[2]),
               1.0e-7, 1.0e-7));
    CHECK(near(corrected.rotationNorm,
               std::sqrt(
                   normalizedDelta[3] * normalizedDelta[3] +
                   normalizedDelta[4] * normalizedDelta[4] +
                   normalizedDelta[5] * normalizedDelta[5]),
               1.0e-7, 1.0e-7));
}

Mat6 multiply6(const Mat6& lhs, const Mat6& rhs) {
    Mat6 result{};
    for (size_t row = 0; row < 6; ++row) {
        for (size_t columnIndex = 0; columnIndex < 6; ++columnIndex) {
            for (size_t inner = 0; inner < 6; ++inner) {
                result[row * 6 + columnIndex] +=
                    lhs[row * 6 + inner] * rhs[inner * 6 + columnIndex];
            }
        }
    }
    return result;
}

void testCampProjectionJacobian() {
    constexpr std::array<double, 3> point{0.4, -0.25, 2.3};
    constexpr double focalX = 720.0;
    constexpr double focalY = 690.0;
    constexpr double width = 1280.0;
    constexpr double height = 720.0;
    const auto analytic = msplat::detail::campPoseProjectionJacobian(
        point, focalX, focalY, width, height);

    const auto normalizedProjection = [&](size_t parameter, double amount) {
        Vec3 translation{};
        Vec3 rotation{};
        if (parameter < 3)
            translation[parameter] = amount;
        else
            rotation[parameter - 3] = amount;
        const Vec3 transformed = transformPoint(
            se3Exp(translation, rotation), point);
        return Vec2{
            focalX * transformed[0] / transformed[2] /
                std::max(width, height),
            focalY * transformed[1] / transformed[2] /
                std::max(width, height),
        };
    };

    constexpr double epsilon = 1.0e-6;
    for (size_t parameter = 0; parameter < 6; ++parameter) {
        const Vec2 positive = normalizedProjection(parameter, epsilon);
        const Vec2 negative = normalizedProjection(parameter, -epsilon);
        for (size_t output = 0; output < 2; ++output) {
            const double finiteDifference =
                (positive[output] - negative[output]) / (2.0 * epsilon);
            CHECK(near(finiteDifference, analytic[output * 6 + parameter],
                       2.0e-10, 2.0e-9));
        }
    }
}

void testCampFullInverseSquareRoot() {
    const std::array<Vec3, 9> points = {{
        {-0.7, -0.3, 1.2}, {0.4, -0.5, 1.5}, {0.8, 0.6, 2.0},
        {-0.2, 0.7, 2.4}, {0.1, -0.1, 3.1}, {-1.0, 0.2, 3.5},
        {0.5, 0.9, 4.0}, {-0.6, -0.8, 4.5}, {1.1, -0.4, 5.0},
    }};
    Mat6 hessianSum{};
    for (const Vec3& point : points) {
        msplat::detail::accumulateCampPoseApproximateHessian(
            hessianSum,
            msplat::detail::campPoseProjectionJacobian(
                point, 810.0, 790.0, 1440.0, 1080.0));
    }
    const Mat6 hessian =
        msplat::detail::finishCampPoseApproximateHessian(
            hessianSum, points.size());
    const Mat6 damped =
        msplat::detail::dampCampPoseApproximateHessian(hessian);
    const Mat6 preconditioner =
        msplat::detail::campPosePreconditionerFromHessian(hessian);
    const Mat6 repeated =
        msplat::detail::campPosePreconditionerFromHessian(hessian);
    CHECK(preconditioner == repeated);

    bool hasTranslationRotationCoupling = false;
    for (size_t row = 0; row < 6; ++row) {
        for (size_t columnIndex = 0; columnIndex < 6; ++columnIndex) {
            CHECK(std::isfinite(preconditioner[row * 6 + columnIndex]));
            CHECK(near(preconditioner[row * 6 + columnIndex],
                       preconditioner[columnIndex * 6 + row],
                       1.0e-12, 1.0e-12));
            if (row < 3 && columnIndex >= 3 &&
                std::abs(preconditioner[row * 6 + columnIndex]) > 1.0e-7) {
                hasTranslationRotationCoupling = true;
            }
        }
    }
    CHECK(hasTranslationRotationCoupling);

    const Mat6 whitened = multiply6(
        multiply6(preconditioner, damped), preconditioner);
    for (size_t row = 0; row < 6; ++row) {
        for (size_t columnIndex = 0; columnIndex < 6; ++columnIndex) {
            CHECK(near(whitened[row * 6 + columnIndex],
                       row == columnIndex ? 1.0 : 0.0,
                       2.0e-10, 2.0e-10));
        }
    }

    const msplat::detail::CampPoseVector latent = {
        0.2, -0.1, 0.05, 0.01, -0.02, 0.03,
    };
    const auto metric = msplat::detail::applyCampPosePreconditioner(
        preconditioner, latent);
    for (double value : metric) CHECK(std::isfinite(value));
}

void testCampDampingAndValidation() {
    Mat6 diagonal{};
    const std::array<double, 6> values = {1.0, 4.0, 9.0, 16.0, 25.0, 36.0};
    for (size_t axis = 0; axis < 6; ++axis)
        diagonal[axis * 6 + axis] = values[axis];
    const Mat6 damped = msplat::detail::dampCampPoseApproximateHessian(
        diagonal, 1.0e-8, 0.1);
    const Mat6 preconditioner =
        msplat::detail::campPosePreconditionerFromHessian(
            diagonal, 1.0e-8, 0.1);
    for (size_t row = 0; row < 6; ++row) {
        CHECK(near(damped[row * 6 + row], 1.1 * values[row]));
        CHECK(near(preconditioner[row * 6 + row],
                   1.0 / std::sqrt(1.1 * values[row]),
                   1.0e-12, 1.0e-12));
        for (size_t columnIndex = 0; columnIndex < 6; ++columnIndex) {
            if (row != columnIndex)
                CHECK(preconditioner[row * 6 + columnIndex] == 0.0);
        }
    }

    const Mat6 zeroHessian{};
    const Mat6 zeroPreconditioner =
        msplat::detail::campPosePreconditionerFromHessian(zeroHessian);
    for (size_t row = 0; row < 6; ++row) {
        for (size_t columnIndex = 0; columnIndex < 6; ++columnIndex) {
            CHECK(near(zeroPreconditioner[row * 6 + columnIndex],
                       row == columnIndex ? 1.0e4 : 0.0,
                       1.0e-9, 1.0e-12));
        }
    }

    bool asymmetricRejected = false;
    try {
        Mat6 asymmetric = diagonal;
        asymmetric[1] = 0.25;
        (void)msplat::detail::campPosePreconditionerFromHessian(asymmetric);
    } catch (const std::invalid_argument&) {
        asymmetricRejected = true;
    }
    CHECK(asymmetricRejected);

    bool nonfiniteRejected = false;
    try {
        Mat6 nonfinite = diagonal;
        nonfinite[0] = std::numeric_limits<double>::infinity();
        (void)msplat::detail::campPosePreconditionerFromHessian(nonfinite);
    } catch (const std::invalid_argument&) {
        nonfiniteRejected = true;
    }
    CHECK(nonfiniteRejected);
}

msplat::detail::CampPoseCameraGeometry campTestCamera() {
    msplat::detail::CampPoseCameraGeometry camera;
    camera.cameraToWorld = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    };
    camera.focalX = 100.0;
    camera.focalY = 100.0;
    camera.principalX = 50.0;
    camera.principalY = 50.0;
    camera.width = 100;
    camera.height = 100;
    camera.deterministicKey = 0x12345678ULL;
    return camera;
}

void testCampRepresentativeVisibilityAndConvention() {
    const std::vector<float> xyz = {
         0.0f,  0.0f, -2.0f,   // center, visible
         0.5f,  0.5f, -2.0f,   // upper-right, visible after Y flip
         0.0f,  0.0f,  2.0f,   // behind
         2.0f,  0.0f, -1.0f,   // outside the right edge
         0.0f,  0.0f, -0.005f, // clipped by the renderer near plane
    };
    const msplat::detail::CampPoseWorldPointPool pool{
        xyz.data(), nullptr, xyz.size() / 3,
    };
    const auto camera = campTestCamera();
    const auto sample = msplat::detail::sampleCampPoseRepresentativePoints(
        camera, pool, 256);
    const auto repeated = msplat::detail::sampleCampPoseRepresentativePoints(
        camera, pool, 256);
    CHECK(sample.viewPoints == repeated.viewPoints);
    CHECK(sample.sourcePointIndices == repeated.sourcePointIndices);
    CHECK(sample.visiblePointCount == 2);
    CHECK(sample.selectedVisiblePointCount == 2);
    CHECK(sample.fallbackPointCount == 254);
    CHECK(sample.viewPoints.size() == 256);
    CHECK(sample.sourcePointIndices.size() == 256);

    bool foundCenter = false;
    bool foundUpperRight = false;
    for (size_t index = 0; index < sample.viewPoints.size(); ++index) {
        const Vec3& point = sample.viewPoints[index];
        CHECK(point[2] > msplat::detail::kCampRepresentativeNearPlane);
        const double pixelX = camera.focalX * point[0] / point[2] +
            camera.principalX;
        const double pixelY = camera.focalY * point[1] / point[2] +
            camera.principalY;
        CHECK(pixelX >= 0.0 && pixelX < camera.width);
        CHECK(pixelY >= 0.0 && pixelY < camera.height);
        if (sample.sourcePointIndices[index] == 0) {
            checkNear(point, {0.0, 0.0, 2.0});
            foundCenter = true;
        } else if (sample.sourcePointIndices[index] == 1) {
            checkNear(point, {0.5, -0.5, 2.0});
            foundUpperRight = true;
        } else {
            CHECK(sample.sourcePointIndices[index] ==
                  std::numeric_limits<size_t>::max());
            CHECK(pixelX >= 0.5 && pixelX <= camera.width - 0.5);
            CHECK(pixelY >= 0.5 && pixelY <= camera.height - 0.5);
        }
    }
    CHECK(foundCenter);
    CHECK(foundUpperRight);

    auto rotatedCamera = camera;
    rotatedCamera.cameraToWorld = {
         0.0, 0.0, 1.0, 10.0,
         0.0, 1.0, 0.0, 20.0,
        -1.0, 0.0, 0.0, 30.0,
         0.0, 0.0, 0.0,  1.0,
    };
    rotatedCamera.focalX = 100.0;
    rotatedCamera.focalY = 100.0;
    rotatedCamera.principalX = 320.0;
    rotatedCamera.principalY = 240.0;
    rotatedCamera.width = 640;
    rotatedCamera.height = 480;
    const std::array<float, 3> rotatedXYZ = {6.0f, 18.0f, 29.0f};
    const msplat::detail::CampPoseWorldPointPool rotatedPool{
        rotatedXYZ.data(), nullptr, 1,
    };
    const auto rotated = msplat::detail::sampleCampPoseRepresentativePoints(
        rotatedCamera, rotatedPool, 256);
    CHECK(rotated.visiblePointCount == 1);
    const auto source = std::find(
        rotated.sourcePointIndices.begin(),
        rotated.sourcePointIndices.end(), size_t{0});
    CHECK(source != rotated.sourcePointIndices.end());
    checkNear(rotated.viewPoints[static_cast<size_t>(
        source - rotated.sourcePointIndices.begin())], {1.0, 2.0, 4.0});
}

void testCampRepresentativeSpatialCapAndStableIDs() {
    auto camera = campTestCamera();
    camera.width = 640;
    camera.height = 480;
    camera.focalX = 500.0;
    camera.focalY = 500.0;
    camera.principalX = 320.0;
    camera.principalY = 240.0;

    std::vector<float> xyz;
    std::vector<uint64_t> stableIds;
    xyz.reserve(16 * 16 * 3 * 3);
    stableIds.reserve(16 * 16 * 3);
    for (size_t pass = 0; pass < 3; ++pass) {
        for (size_t cellY = 0; cellY < 16; ++cellY) {
            for (size_t cellX = 0; cellX < 16; ++cellX) {
                const double pixelX =
                    (static_cast<double>(cellX) + 0.35 + 0.1 * pass) * 40.0;
                const double pixelY =
                    (static_cast<double>(cellY) + 0.35 + 0.1 * pass) * 30.0;
                const double depth = 1.5 + 0.25 * pass;
                const double viewX =
                    (pixelX - camera.principalX) * depth / camera.focalX;
                const double viewY =
                    (pixelY - camera.principalY) * depth / camera.focalY;
                xyz.insert(xyz.end(), {
                    static_cast<float>(viewX),
                    static_cast<float>(-viewY),
                    static_cast<float>(-depth),
                });
                stableIds.push_back(
                    1000 + pass * 256 + cellY * 16 + cellX);
            }
        }
    }
    const msplat::detail::CampPoseWorldPointPool pool{
        xyz.data(), stableIds.data(), stableIds.size(),
    };
    const auto sample = msplat::detail::sampleCampPoseRepresentativePoints(
        camera, pool);
    CHECK(sample.visiblePointCount == 768);
    CHECK(sample.selectedVisiblePointCount == 512);
    CHECK(sample.fallbackPointCount == 0);
    std::array<size_t, 256> selectedPerCell{};
    for (const Vec3& point : sample.viewPoints) {
        const double pixelX = camera.focalX * point[0] / point[2] +
            camera.principalX;
        const double pixelY = camera.focalY * point[1] / point[2] +
            camera.principalY;
        const size_t cellX = std::min<size_t>(15,
            static_cast<size_t>(pixelX * 16.0 / camera.width));
        const size_t cellY = std::min<size_t>(15,
            static_cast<size_t>(pixelY * 16.0 / camera.height));
        ++selectedPerCell[cellY * 16 + cellX];
    }
    for (size_t count : selectedPerCell) CHECK(count == 2);

    std::vector<float> reversedXYZ;
    std::vector<uint64_t> reversedIDs;
    reversedXYZ.reserve(xyz.size());
    reversedIDs.reserve(stableIds.size());
    for (size_t reverseIndex = stableIds.size(); reverseIndex-- > 0;) {
        reversedXYZ.insert(reversedXYZ.end(), {
            xyz[reverseIndex * 3], xyz[reverseIndex * 3 + 1],
            xyz[reverseIndex * 3 + 2],
        });
        reversedIDs.push_back(stableIds[reverseIndex]);
    }
    const msplat::detail::CampPoseWorldPointPool reversedPool{
        reversedXYZ.data(), reversedIDs.data(), reversedIDs.size(),
    };
    const auto reordered = msplat::detail::sampleCampPoseRepresentativePoints(
        camera, reversedPool);
    CHECK(sample.viewPoints == reordered.viewPoints);

    const auto built = msplat::detail::buildCampPosePreconditioner(
        camera, pool);
    const auto rebuilt = msplat::detail::buildCampPosePreconditioner(
        camera, reversedPool);
    CHECK(built.hessian == rebuilt.hessian);
    CHECK(built.preconditioner == rebuilt.preconditioner);
    for (double value : built.preconditioner) CHECK(std::isfinite(value));

    auto doubledRaster = camera;
    doubledRaster.width *= 2;
    doubledRaster.height *= 2;
    doubledRaster.focalX *= 2.0;
    doubledRaster.focalY *= 2.0;
    doubledRaster.principalX *= 2.0;
    doubledRaster.principalY *= 2.0;
    const auto doubled = msplat::detail::buildCampPosePreconditioner(
        doubledRaster, pool);
    for (size_t index = 0; index < built.hessian.size(); ++index) {
        CHECK(near(built.hessian[index], doubled.hessian[index],
                   1.0e-13, 1.0e-13));
        CHECK(near(built.preconditioner[index], doubled.preconditioner[index],
                   1.0e-11, 1.0e-11));
    }
}

void testCampRepresentativeHaltonFallbackAndBounds() {
    const auto camera = campTestCamera();
    const msplat::detail::CampPoseWorldPointPool emptyPool{};
    const auto sample = msplat::detail::sampleCampPoseRepresentativePoints(
        camera, emptyPool);
    const auto repeated = msplat::detail::sampleCampPoseRepresentativePoints(
        camera, emptyPool);
    CHECK(sample.viewPoints == repeated.viewPoints);
    CHECK(sample.sourcePointIndices == repeated.sourcePointIndices);
    CHECK(sample.visiblePointCount == 0);
    CHECK(sample.selectedVisiblePointCount == 0);
    CHECK(sample.fallbackPointCount == 512);
    CHECK(sample.viewPoints.size() == 512);
    for (size_t sourceIndex : sample.sourcePointIndices) {
        CHECK(sourceIndex == std::numeric_limits<size_t>::max());
    }

    const auto built = msplat::detail::buildCampPosePreconditioner(
        camera, emptyPool);
    const auto rebuilt = msplat::detail::buildCampPosePreconditioner(
        camera, emptyPool);
    CHECK(built.hessian == rebuilt.hessian);
    CHECK(built.preconditioner == rebuilt.preconditioner);
    CHECK(built.fallbackPointCount == 512);
    for (double value : built.preconditioner) CHECK(std::isfinite(value));

    bool smallTargetRejected = false;
    try {
        (void)msplat::detail::sampleCampPoseRepresentativePoints(
            camera, emptyPool, 255);
    } catch (const std::invalid_argument&) {
        smallTargetRejected = true;
    }
    CHECK(smallTargetRejected);

    const float placeholder[3] = {};
    const msplat::detail::CampPoseWorldPointPool oversizedPool{
        placeholder, nullptr,
        msplat::detail::kCampMaximumWorldPointPoolCount + 1,
    };
    bool oversizedPoolRejected = false;
    try {
        (void)msplat::detail::sampleCampPoseRepresentativePoints(
            camera, oversizedPool);
    } catch (const std::invalid_argument&) {
        oversizedPoolRejected = true;
    }
    CHECK(oversizedPoolRejected);
}

void testCampWorldPointPoolCapUsesStableIDs() {
    const size_t pointCount =
        msplat::detail::kCampMaximumWorldPointPoolCount + 17;
    std::vector<uint64_t> stableIds(pointCount);
    for (size_t point = 0; point < pointCount; ++point)
        stableIds[point] = 10'000'000 + point * 13;

    const auto selected =
        msplat::detail::selectCampPoseWorldPointPoolIndices(
            pointCount, stableIds.data());
    CHECK(selected.size() ==
          msplat::detail::kCampMaximumWorldPointPoolCount);
    std::vector<uint64_t> selectedIds;
    selectedIds.reserve(selected.size());
    for (size_t index : selected)
        selectedIds.push_back(stableIds[index]);

    std::reverse(stableIds.begin(), stableIds.end());
    const auto reordered =
        msplat::detail::selectCampPoseWorldPointPoolIndices(
            pointCount, stableIds.data());
    std::vector<uint64_t> reorderedIds;
    reorderedIds.reserve(reordered.size());
    for (size_t index : reordered)
        reorderedIds.push_back(stableIds[index]);
    CHECK(selectedIds == reorderedIds);
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
        testPoseStateReadbackGeometry();
        testCampProjectionJacobian();
        testCampFullInverseSquareRoot();
        testCampDampingAndValidation();
        testCampRepresentativeVisibilityAndConvention();
        testCampRepresentativeSpatialCapAndStableIDs();
        testCampRepresentativeHaltonFallbackAndBounds();
        testCampWorldPointPoolCapUsesStableIDs();
        std::cout << "Pose refinement math tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "msplat pose refinement test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
