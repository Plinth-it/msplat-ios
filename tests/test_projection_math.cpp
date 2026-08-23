#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Matrix4 = std::array<double, 16>;
using Vector2 = std::array<double, 2>;
using Vector3 = std::array<double, 3>;

[[noreturn]] void fail(const char *expression, int line) {
    throw std::runtime_error(
        "line " + std::to_string(line) + ": " + expression);
}

#define CHECK(condition) \
    do { if (!(condition)) fail(#condition, __LINE__); } while (false)

bool near(double lhs, double rhs, double tolerance = 1.0e-6) {
    return std::abs(lhs - rhs) <= tolerance;
}

std::array<double, 4> transform(
    const Matrix4 &matrix, const Vector3 &point) {
    return {
        matrix[0] * point[0] + matrix[1] * point[1] +
            matrix[2] * point[2] + matrix[3],
        matrix[4] * point[0] + matrix[5] * point[1] +
            matrix[6] * point[2] + matrix[7],
        matrix[8] * point[0] + matrix[9] * point[1] +
            matrix[10] * point[2] + matrix[11],
        matrix[12] * point[0] + matrix[13] * point[1] +
            matrix[14] * point[2] + matrix[15],
    };
}

Vector2 projectRaster(
    const Matrix4 &matrix, const Vector3 &point,
    double width, double height, double cx, double cy) {
    const auto homogeneous = transform(matrix, point);
    return {
        0.5 * width * homogeneous[0] / homogeneous[3] + cx - 0.5,
        0.5 * height * homogeneous[1] / homogeneous[3] + cy - 0.5,
    };
}

Vector3 projectRasterVJP(
    const Matrix4 &matrix, const Vector3 &point,
    double width, double height, const Vector2 &upstream) {
    const auto homogeneous = transform(matrix, point);
    const double reciprocalW = 1.0 / homogeneous[3];
    const double vx = 0.5 * width * upstream[0];
    const double vy = 0.5 * height * upstream[1];
    const double vHomogeneousW =
        -(vx * homogeneous[0] + vy * homogeneous[1]) *
        reciprocalW * reciprocalW;
    const double vHomogeneousX = vx * reciprocalW;
    const double vHomogeneousY = vy * reciprocalW;
    return {
        matrix[0] * vHomogeneousX + matrix[4] * vHomogeneousY +
            matrix[12] * vHomogeneousW,
        matrix[1] * vHomogeneousX + matrix[5] * vHomogeneousY +
            matrix[13] * vHomogeneousW,
        matrix[2] * vHomogeneousX + matrix[6] * vHomogeneousY +
            matrix[14] * vHomogeneousW,
    };
}

double projectionObjective(
    const Matrix4 &matrix, const Vector3 &point,
    double width, double height, const Vector2 &upstream) {
    const Vector2 raster =
        projectRaster(matrix, point, width, height, 17.0, 23.0);
    return raster[0] * upstream[0] + raster[1] * upstream[1];
}

void checkGeneralProjectionVJP() {
    const Matrix4 matrix = {
        1.25, 0.10, -0.05, 0.20,
       -0.20, 1.10,  0.03, -0.10,
        0.00, 0.00,  1.00, 0.00,
        0.15, -0.05, 0.90, 1.30,
    };
    const Vector3 point = {0.2, -0.3, 0.7};
    const Vector2 upstream = {0.7, -1.2};
    const Vector3 gradient =
        projectRasterVJP(matrix, point, 640.0, 480.0, upstream);
    const Vector3 expected = {
        162.647575709, -146.300246755, -59.780368531,
    };
    for (size_t axis = 0; axis < 3; ++axis)
        CHECK(near(gradient[axis], expected[axis], 1.0e-8));

    constexpr double epsilon = 1.0e-6;
    for (size_t axis = 0; axis < 3; ++axis) {
        Vector3 before = point;
        Vector3 after = point;
        before[axis] -= epsilon;
        after[axis] += epsilon;
        const double finiteDifference =
            (projectionObjective(matrix, after, 640.0, 480.0, upstream) -
             projectionObjective(matrix, before, 640.0, 480.0, upstream)) /
            (2.0 * epsilon);
        CHECK(near(gradient[axis], finiteDifference, 2.0e-7));
    }
}

void checkPhysicalCameraProjection() {
    // Identity OpenGL camera: the runtime view transform flips Y and Z.
    const Matrix4 projectionView = {
        1.25, 0.0, 0.0, 0.0,
        0.0, -1.25, 0.0, 0.0,
        0.0, 0.0, -1.000002, -0.001000002,
        0.0, 0.0, -1.0, 0.0,
    };
    const Vector3 point = {0.2, -0.1, -2.0};
    const Vector2 raster =
        projectRaster(projectionView, point, 640.0, 480.0, 320.0, 240.0);
    CHECK(near(raster[0], 359.5));
    CHECK(near(raster[1], 254.5));

    const Vector3 gradient = projectRasterVJP(
        projectionView, point, 640.0, 480.0, {1.0, 1.0});
    CHECK(near(gradient[0], 200.0));
    CHECK(near(gradient[1], -150.0));
    CHECK(near(gradient[2], 27.5));
}

double clampedCovarianceTrace(const Vector3 &point) {
    constexpr double fx = 100.0;
    constexpr double fy = 80.0;
    constexpr double limitX = 1.3;
    constexpr double limitY = 0.9;
    const double ratioX =
        std::clamp(point[0] / point[2], -limitX, limitX);
    const double ratioY =
        std::clamp(point[1] / point[2], -limitY, limitY);
    const double j00 = fx / point[2];
    const double j11 = fy / point[2];
    const double j20 = -fx * ratioX / point[2];
    const double j21 = -fy * ratioY / point[2];
    return j00 * j00 + j20 * j20 + j11 * j11 + j21 * j21 + 0.6;
}

Vector3 clampedCovarianceTraceVJP(const Vector3 &point) {
    constexpr double fx = 100.0;
    constexpr double fy = 80.0;
    constexpr double limitX = 1.3;
    constexpr double limitY = 0.9;
    const double rawRatioX = point[0] / point[2];
    const double rawRatioY = point[1] / point[2];
    const double ratioX = std::clamp(rawRatioX, -limitX, limitX);
    const double ratioY = std::clamp(rawRatioY, -limitY, limitY);
    const double ratioGradientX =
        rawRatioX >= -limitX && rawRatioX <= limitX ? 1.0 : 0.0;
    const double ratioGradientY =
        rawRatioY >= -limitY && rawRatioY <= limitY ? 1.0 : 0.0;
    const double reciprocalZ = 1.0 / point[2];
    const double reciprocalZ2 = reciprocalZ * reciprocalZ;
    const double reciprocalZ3 = reciprocalZ2 * reciprocalZ;
    const double clampedX = point[2] * ratioX;
    const double clampedY = point[2] * ratioY;
    const double j00 = fx * reciprocalZ;
    const double j11 = fy * reciprocalZ;
    const double j20 = -fx * clampedX * reciprocalZ2;
    const double j21 = -fy * clampedY * reciprocalZ2;
    const double vJ00 = 2.0 * j00;
    const double vJ11 = 2.0 * j11;
    const double vJ20 = 2.0 * j20;
    const double vJ21 = 2.0 * j21;
    return {
        -ratioGradientX * fx * reciprocalZ2 * vJ20,
        -ratioGradientY * fy * reciprocalZ2 * vJ21,
        -fx * reciprocalZ2 * vJ00 +
            (1.0 + ratioGradientX) * fx * clampedX *
                reciprocalZ3 * vJ20 -
            fy * reciprocalZ2 * vJ11 +
            (1.0 + ratioGradientY) * fy * clampedY *
                reciprocalZ3 * vJ21,
    };
}

void checkClampedCovarianceVJP() {
    constexpr double epsilon = 1.0e-6;
    const auto checkPoint = [](const Vector3 &point) {
        const Vector3 analytic = clampedCovarianceTraceVJP(point);
        for (size_t axis = 0; axis < 3; ++axis) {
            Vector3 before = point;
            Vector3 after = point;
            before[axis] -= epsilon;
            after[axis] += epsilon;
            const double finiteDifference =
                (clampedCovarianceTrace(after) -
                 clampedCovarianceTrace(before)) /
                (2.0 * epsilon);
            CHECK(near(analytic[axis], finiteDifference, 2.0e-5));
        }
    };

    checkPoint({2.0, 0.25, 1.0});   // X clamped, Y differentiable.
    checkPoint({0.5, -1.2, 1.0});  // X differentiable, Y clamped.
    checkPoint({0.5, 0.25, 1.0});  // Both differentiable.
}

} // namespace

int main() {
    try {
        checkGeneralProjectionVJP();
        checkPhysicalCameraProjection();
        checkClampedCovarianceVJP();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "msplat projection math test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
