#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace msplat::detail {

// Formulation follows CamP's camera_delta.py at upstream commit
// 8e6d57e3aee34235faf3ef99decca0994efe66c9. The small fixed-size numerical
// implementation here is native to msplat and has no runtime dependency.
// CamP conditions a latent camera update before it is applied in the camera's
// metric parameter space. This helper is deliberately fixed to msplat's pose
// order: camera-space left translation [tx, ty, tz], followed by left
// axis-angle rotation [rx, ry, rz]. Matrices are row-major.
using CampPoseVector = std::array<double, 6>;
using CampPoseMatrix = std::array<double, 36>;
using CampProjectionJacobian = std::array<double, 12>;

constexpr double kCampAbsoluteDamping = 1.0e-8;
constexpr double kCampRelativeDamping = 0.1;

inline double& campMatrixAt(
    CampPoseMatrix& matrix, size_t row, size_t column) {
    return matrix[row * 6 + column];
}

inline double campMatrixAt(
    const CampPoseMatrix& matrix, size_t row, size_t column) {
    return matrix[row * 6 + column];
}

inline void validateCampPoseMatrix(const CampPoseMatrix& matrix) {
    double maximumMagnitude = 0.0;
    for (double value : matrix) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "CamP pose matrix must contain only finite values");
        }
        maximumMagnitude = std::max(maximumMagnitude, std::abs(value));
    }

    const double symmetryTolerance =
        1.0e-12 * std::max(1.0, maximumMagnitude);
    for (size_t row = 0; row < 6; ++row) {
        if (campMatrixAt(matrix, row, row) < 0.0) {
            throw std::invalid_argument(
                "CamP approximate Hessian diagonal must be nonnegative");
        }
        for (size_t column = row + 1; column < 6; ++column) {
            if (std::abs(campMatrixAt(matrix, row, column) -
                         campMatrixAt(matrix, column, row)) >
                symmetryTolerance) {
                throw std::invalid_argument(
                    "CamP pose matrix must be symmetric");
            }
        }
    }
}

/// Returns the normalized-pixel projection Jacobian for a point already in
/// camera/view space. The parameter columns are [tx, ty, tz, rx, ry, rz].
/// This is the zero-delta derivative of
///   project(exp([translation, rotation]) * viewPoint) / max(width, height).
inline CampProjectionJacobian campPoseProjectionJacobian(
    const std::array<double, 3>& viewPoint, double focalX, double focalY,
    double imageWidth, double imageHeight) {
    for (double value : viewPoint) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "CamP view point must contain only finite values");
        }
    }
    if (!std::isfinite(focalX) || !std::isfinite(focalY) ||
        focalX <= 0.0 || focalY <= 0.0) {
        throw std::invalid_argument(
            "CamP focal lengths must be finite and positive");
    }
    if (!std::isfinite(imageWidth) || !std::isfinite(imageHeight) ||
        imageWidth <= 0.0 || imageHeight <= 0.0) {
        throw std::invalid_argument(
            "CamP image dimensions must be finite and positive");
    }

    const double x = viewPoint[0];
    const double y = viewPoint[1];
    const double z = viewPoint[2];
    if (std::abs(z) <= 1.0e-12) {
        throw std::invalid_argument(
            "CamP view point depth must be nonzero");
    }

    const double inverseImageScale =
        1.0 / std::max(imageWidth, imageHeight);
    const double inverseZ = 1.0 / z;
    const double inverseZ2 = inverseZ * inverseZ;
    const std::array<double, 6> projectionDerivative = {
        focalX * inverseZ * inverseImageScale,
        0.0,
        -focalX * x * inverseZ2 * inverseImageScale,
        0.0,
        focalY * inverseZ * inverseImageScale,
        -focalY * y * inverseZ2 * inverseImageScale,
    };

    // At zero, d(exp(rotation) * p + translation) / d(rotation) is
    // -skew(p). Rows below are dp/d[tx,ty,tz,rx,ry,rz].
    constexpr std::array<double, 9> identityDerivatives = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const std::array<double, 9> rotationDerivatives = {
        0.0, z, -y,
        -z, 0.0, x,
        y, -x, 0.0,
    };

    CampProjectionJacobian result{};
    for (size_t output = 0; output < 2; ++output) {
        for (size_t parameter = 0; parameter < 6; ++parameter) {
            double derivative = 0.0;
            for (size_t axis = 0; axis < 3; ++axis) {
                const double pointDerivative = parameter < 3
                    ? identityDerivatives[axis * 3 + parameter]
                    : rotationDerivatives[axis * 3 + parameter - 3];
                derivative += projectionDerivative[output * 3 + axis] *
                    pointDerivative;
            }
            result[output * 6 + parameter] = derivative;
        }
    }
    return result;
}

inline void accumulateCampPoseApproximateHessian(
    CampPoseMatrix& hessianSum, const CampProjectionJacobian& jacobian) {
    for (double value : jacobian) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "CamP projection Jacobian must contain only finite values");
        }
    }
    for (size_t row = 0; row < 6; ++row) {
        for (size_t column = row; column < 6; ++column) {
            double value = 0.0;
            for (size_t output = 0; output < 2; ++output) {
                value += jacobian[output * 6 + row] *
                    jacobian[output * 6 + column];
            }
            campMatrixAt(hessianSum, row, column) += value;
            if (row != column)
                campMatrixAt(hessianSum, column, row) += value;
        }
    }
}

inline CampPoseMatrix finishCampPoseApproximateHessian(
    CampPoseMatrix hessianSum, size_t validPointCount) {
    if (validPointCount == 0) {
        throw std::invalid_argument(
            "CamP approximate Hessian requires at least one valid point");
    }
    const double inverseCount = 1.0 / static_cast<double>(validPointCount);
    for (double& value : hessianSum) value *= inverseCount;
    validateCampPoseMatrix(hessianSum);
    return hessianSum;
}

/// Applies CamP's per-parameter damping:
///   H_damped[i,i] = H[i,i] + max(mu, lambda * H[i,i]).
inline CampPoseMatrix dampCampPoseApproximateHessian(
    const CampPoseMatrix& hessian,
    double absoluteDamping = kCampAbsoluteDamping,
    double relativeDamping = kCampRelativeDamping) {
    validateCampPoseMatrix(hessian);
    if (!std::isfinite(absoluteDamping) || absoluteDamping <= 0.0 ||
        !std::isfinite(relativeDamping) || relativeDamping < 0.0) {
        throw std::invalid_argument(
            "CamP damping must be finite, with positive absolute and "
            "nonnegative relative values");
    }

    CampPoseMatrix result = hessian;
    for (size_t parameter = 0; parameter < 6; ++parameter) {
        const double diagonal = campMatrixAt(hessian, parameter, parameter);
        campMatrixAt(result, parameter, parameter) +=
            std::max(absoluteDamping, relativeDamping * diagonal);
    }
    return result;
}

/// Computes a deterministic full 6x6 symmetric inverse square root of the
/// damped approximate Hessian. A cyclic Jacobi eigensolver keeps this init-time
/// CPU path dependency-free and deterministic.
inline CampPoseMatrix campPosePreconditionerFromHessian(
    const CampPoseMatrix& hessian,
    double absoluteDamping = kCampAbsoluteDamping,
    double relativeDamping = kCampRelativeDamping) {
    CampPoseMatrix eigenSystem = dampCampPoseApproximateHessian(
        hessian, absoluteDamping, relativeDamping);
    CampPoseMatrix eigenvectors{};
    for (size_t axis = 0; axis < 6; ++axis)
        campMatrixAt(eigenvectors, axis, axis) = 1.0;

    constexpr size_t maximumSweeps = 64;
    bool converged = false;
    for (size_t sweep = 0; sweep < maximumSweeps; ++sweep) {
        double diagonalScale = 1.0;
        double maximumOffDiagonal = 0.0;
        for (size_t row = 0; row < 6; ++row) {
            diagonalScale = std::max(
                diagonalScale,
                std::abs(campMatrixAt(eigenSystem, row, row)));
            for (size_t column = row + 1; column < 6; ++column) {
                maximumOffDiagonal = std::max(
                    maximumOffDiagonal,
                    std::abs(campMatrixAt(eigenSystem, row, column)));
            }
        }
        if (maximumOffDiagonal <= 1.0e-14 * diagonalScale) {
            converged = true;
            break;
        }

        for (size_t first = 0; first < 5; ++first) {
            for (size_t second = first + 1; second < 6; ++second) {
                const double crossValue =
                    campMatrixAt(eigenSystem, first, second);
                if (std::abs(crossValue) <= 1.0e-16 * diagonalScale)
                    continue;

                const double firstValue =
                    campMatrixAt(eigenSystem, first, first);
                const double secondValue =
                    campMatrixAt(eigenSystem, second, second);
                const double tau =
                    (secondValue - firstValue) / (2.0 * crossValue);
                const double tangent = tau == 0.0
                    ? 1.0
                    : std::copysign(
                        1.0 / (std::abs(tau) +
                               std::hypot(1.0, tau)), tau);
                const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
                const double sine = tangent * cosine;

                for (size_t axis = 0; axis < 6; ++axis) {
                    if (axis == first || axis == second) continue;
                    const double axisFirst =
                        campMatrixAt(eigenSystem, axis, first);
                    const double axisSecond =
                        campMatrixAt(eigenSystem, axis, second);
                    const double rotatedFirst =
                        cosine * axisFirst - sine * axisSecond;
                    const double rotatedSecond =
                        sine * axisFirst + cosine * axisSecond;
                    campMatrixAt(eigenSystem, axis, first) = rotatedFirst;
                    campMatrixAt(eigenSystem, first, axis) = rotatedFirst;
                    campMatrixAt(eigenSystem, axis, second) = rotatedSecond;
                    campMatrixAt(eigenSystem, second, axis) = rotatedSecond;
                }
                campMatrixAt(eigenSystem, first, first) =
                    firstValue - tangent * crossValue;
                campMatrixAt(eigenSystem, second, second) =
                    secondValue + tangent * crossValue;
                campMatrixAt(eigenSystem, first, second) = 0.0;
                campMatrixAt(eigenSystem, second, first) = 0.0;

                for (size_t axis = 0; axis < 6; ++axis) {
                    const double vectorFirst =
                        campMatrixAt(eigenvectors, axis, first);
                    const double vectorSecond =
                        campMatrixAt(eigenvectors, axis, second);
                    campMatrixAt(eigenvectors, axis, first) =
                        cosine * vectorFirst - sine * vectorSecond;
                    campMatrixAt(eigenvectors, axis, second) =
                        sine * vectorFirst + cosine * vectorSecond;
                }
            }
        }
    }
    if (!converged) {
        double diagonalScale = 1.0;
        double maximumOffDiagonal = 0.0;
        for (size_t row = 0; row < 6; ++row) {
            diagonalScale = std::max(
                diagonalScale,
                std::abs(campMatrixAt(eigenSystem, row, row)));
            for (size_t column = row + 1; column < 6; ++column) {
                maximumOffDiagonal = std::max(
                    maximumOffDiagonal,
                    std::abs(campMatrixAt(eigenSystem, row, column)));
            }
        }
        converged = maximumOffDiagonal <= 1.0e-14 * diagonalScale;
    }
    if (!converged) {
        throw std::runtime_error(
            "CamP inverse-square-root eigensolver did not converge");
    }

    CampPoseVector inverseSquareRoots{};
    for (size_t axis = 0; axis < 6; ++axis) {
        const double eigenvalue =
            campMatrixAt(eigenSystem, axis, axis);
        if (!std::isfinite(eigenvalue) || eigenvalue <= 0.0) {
            throw std::invalid_argument(
                "CamP damped approximate Hessian must be positive definite");
        }
        inverseSquareRoots[axis] = 1.0 / std::sqrt(eigenvalue);
    }

    CampPoseMatrix result{};
    for (size_t row = 0; row < 6; ++row) {
        for (size_t column = row; column < 6; ++column) {
            double value = 0.0;
            for (size_t axis = 0; axis < 6; ++axis) {
                value += campMatrixAt(eigenvectors, row, axis) *
                    inverseSquareRoots[axis] *
                    campMatrixAt(eigenvectors, column, axis);
            }
            campMatrixAt(result, row, column) = value;
            campMatrixAt(result, column, row) = value;
        }
    }
    validateCampPoseMatrix(result);
    return result;
}

inline CampPoseVector applyCampPosePreconditioner(
    const CampPoseMatrix& preconditioner, const CampPoseVector& latent) {
    validateCampPoseMatrix(preconditioner);
    CampPoseVector result{};
    for (size_t row = 0; row < 6; ++row) {
        if (!std::isfinite(latent[row])) {
            throw std::invalid_argument(
                "CamP latent pose must contain only finite values");
        }
        for (size_t column = 0; column < 6; ++column) {
            result[row] += campMatrixAt(preconditioner, row, column) *
                latent[column];
        }
    }
    return result;
}

} // namespace msplat::detail
