#include "dataset_descriptor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

constexpr float kPoseTolerance = 1.0e-3f;

[[noreturn]] void reject(const std::string &reason) {
    throw std::invalid_argument("Invalid dataset descriptor: " + reason);
}

void validateCalibration(const CameraCalibration &calibration,
                         const std::string &frameId) {
    if (calibration.width <= 0 || calibration.height <= 0)
        reject("frame '" + frameId + "' has non-positive dimensions");

    const float values[] = {
        calibration.fx, calibration.fy, calibration.cx, calibration.cy,
        calibration.k1, calibration.k2, calibration.k3,
        calibration.p1, calibration.p2,
    };
    for (float value : values) {
        if (!std::isfinite(value))
            reject("frame '" + frameId + "' has non-finite calibration");
    }
    if (calibration.fx <= 0.0f || calibration.fy <= 0.0f)
        reject("frame '" + frameId + "' has non-positive focal length");
}

float rotationDot(const std::array<float, 16> &matrix,
                  int lhsRow, int rhsRow) {
    float result = 0.0f;
    for (int column = 0; column < 3; ++column)
        result += matrix[lhsRow * 4 + column] *
                  matrix[rhsRow * 4 + column];
    return result;
}

void validatePose(const std::array<float, 16> &matrix,
                  const std::string &frameId) {
    for (float value : matrix) {
        if (!std::isfinite(value))
            reject("frame '" + frameId + "' has a non-finite pose");
    }

    if (std::abs(matrix[12]) > kPoseTolerance ||
        std::abs(matrix[13]) > kPoseTolerance ||
        std::abs(matrix[14]) > kPoseTolerance ||
        std::abs(matrix[15] - 1.0f) > kPoseTolerance) {
        reject("frame '" + frameId + "' pose has an invalid bottom row");
    }

    for (int row = 0; row < 3; ++row) {
        if (std::abs(rotationDot(matrix, row, row) - 1.0f) >
            kPoseTolerance) {
            reject("frame '" + frameId + "' pose rotation is not orthonormal");
        }
        for (int other = row + 1; other < 3; ++other) {
            if (std::abs(rotationDot(matrix, row, other)) > kPoseTolerance) {
                reject("frame '" + frameId +
                       "' pose rotation is not orthonormal");
            }
        }
    }

    const float determinant =
        matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9]) -
        matrix[1] * (matrix[4] * matrix[10] - matrix[6] * matrix[8]) +
        matrix[2] * (matrix[4] * matrix[9] - matrix[5] * matrix[8]);
    if (std::abs(determinant - 1.0f) > kPoseTolerance)
        reject("frame '" + frameId + "' pose rotation is not right-handed");
}

} // namespace

void validateDatasetDescriptor(const DatasetDescriptor &descriptor) {
    if (descriptor.frames.empty())
        reject("frames must not be empty");
    if (descriptor.points.empty())
        reject("points must not be empty");
    if (descriptor.provenance.adapter.empty())
        reject("provenance adapter must not be empty");
    if (descriptor.provenance.source.empty())
        reject("provenance source must not be empty");
    if (descriptor.frames.size() >
        static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        reject("frame count exceeds the supported range");
    }

    std::unordered_set<std::string> frameIds;
    for (const DatasetFrameDescriptor &frame : descriptor.frames) {
        if (frame.id.empty())
            reject("frame IDs must not be empty");
        if (!frameIds.insert(frame.id).second)
            reject("frame ID '" + frame.id + "' is duplicated");
        if (frame.calibrationId.empty())
            reject("frame '" + frame.id + "' calibration ID must not be empty");
        if (frame.imagePath.empty())
            reject("frame '" + frame.id + "' image path must not be empty");

        switch (frame.rasterOrientation) {
            case RasterOrientation::EncodedPixels:
            case RasterOrientation::ExifNormalized:
                break;
            default:
                reject("frame '" + frame.id + "' has an unknown raster orientation");
        }

        validateCalibration(frame.calibration, frame.id);
        validatePose(frame.cameraToWorld, frame.id);
    }

    const SparsePointSet &points = descriptor.points;
    if (points.xyz.size() % 3 != 0)
        reject("point XYZ array length must be a multiple of three");
    for (float value : points.xyz) {
        if (!std::isfinite(value))
            reject("point XYZ values must be finite");
    }
    if (points.rgb.size() != points.xyz.size())
        reject("point RGB array length must match the XYZ array length");

    const size_t pointCount = points.count();
    if (pointCount > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        reject("point count exceeds the supported range");
    if (!points.sourceIds.empty() && points.sourceIds.size() != pointCount)
        reject("point source ID array length must match the point count");
    if (!points.sourceIds.empty()) {
        std::vector<uint64_t> sourceIds = points.sourceIds;
        std::sort(sourceIds.begin(), sourceIds.end());
        if (std::adjacent_find(sourceIds.begin(), sourceIds.end()) !=
            sourceIds.end()) {
            reject("point source IDs must be unique");
        }
    }

    if (!points.reprojectionErrors.empty() &&
        points.reprojectionErrors.size() != pointCount) {
        reject("point reprojection-error array length must match the point count");
    }
    for (float error : points.reprojectionErrors) {
        if (!std::isfinite(error) || error < 0.0f)
            reject("point reprojection errors must be finite and non-negative");
    }

    for (const SparseObservation &observation : descriptor.observations) {
        if (static_cast<size_t>(observation.frameIndex) >=
            descriptor.frames.size()) {
            reject("observation frame index is out of range");
        }
        if (observation.pointIndex < -1 ||
            (observation.pointIndex >= 0 &&
             static_cast<size_t>(observation.pointIndex) >= pointCount)) {
            reject("observation point index is out of range");
        }
        if (!std::isfinite(observation.x) || !std::isfinite(observation.y))
            reject("observation coordinates must be finite");
    }
}
