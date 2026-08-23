#include "dataset_diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace {

class ReprojectionAccumulator {
public:
    void add(double value) noexcept {
        ++statistics.sampleCount;
        const double count = static_cast<double>(statistics.sampleCount);
        statistics.meanPixels += (value - statistics.meanPixels) / count;
        statistics.maximumPixels = std::max(
            statistics.maximumPixels, value);

        if (value == 0.0) return;
        if (squareScale < value) {
            const double ratio = squareScale / value;
            scaledSquareSum = 1.0 + scaledSquareSum * ratio * ratio;
            squareScale = value;
        } else {
            const double ratio = value / squareScale;
            scaledSquareSum += ratio * ratio;
        }
    }

    DatasetReprojectionStatistics finish() const noexcept {
        DatasetReprojectionStatistics result = statistics;
        if (result.sampleCount != 0 && squareScale != 0.0) {
            result.rootMeanSquarePixels = squareScale * std::sqrt(
                scaledSquareSum / static_cast<double>(result.sampleCount));
        }
        return result;
    }

private:
    DatasetReprojectionStatistics statistics;
    double squareScale = 0.0;
    double scaledSquareSum = 0.0;
};

enum class ProjectionResult {
    Projected,
    BehindCamera,
    NonFinite,
};

struct PointTrackState {
    uint32_t lastFramePlusOne = 0;
    uint32_t frameCount = 0;
};

ProjectionResult projectPoint(
    const DatasetFrameDescriptor &frame,
    const float *point,
    double &pixelX,
    double &pixelY) noexcept {
    const auto &matrix = frame.cameraToWorld;
    const double dx = static_cast<double>(point[0]) - matrix[3];
    const double dy = static_cast<double>(point[1]) - matrix[7];
    const double dz = static_cast<double>(point[2]) - matrix[11];

    // The descriptor stores an OpenGL camera-to-world transform. Transpose its
    // rotation to get world-to-camera, then flip Y and Z into the calibration's
    // OpenCV x-right/y-down/z-forward frame.
    const double cameraX = matrix[0] * dx + matrix[4] * dy + matrix[8] * dz;
    const double cameraY = -(
        matrix[1] * dx + matrix[5] * dy + matrix[9] * dz);
    const double cameraZ = -(
        matrix[2] * dx + matrix[6] * dy + matrix[10] * dz);
    if (!std::isfinite(cameraX) || !std::isfinite(cameraY) ||
        !std::isfinite(cameraZ)) {
        return ProjectionResult::NonFinite;
    }
    if (!(cameraZ > 0.0)) return ProjectionResult::BehindCamera;

    const double x = cameraX / cameraZ;
    const double y = cameraY / cameraZ;
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const CameraCalibration &calibration = frame.calibration;
    const double radial = 1.0 + calibration.k1 * r2 +
        calibration.k2 * r4 + calibration.k3 * r6;
    const double distortedX = x * radial + 2.0 * calibration.p1 * x * y +
        calibration.p2 * (r2 + 2.0 * x * x);
    const double distortedY = y * radial +
        calibration.p1 * (r2 + 2.0 * y * y) +
        2.0 * calibration.p2 * x * y;
    pixelX = calibration.fx * distortedX + calibration.cx;
    pixelY = calibration.fy * distortedY + calibration.cy;
    if (!std::isfinite(r2) || !std::isfinite(r4) || !std::isfinite(r6) ||
        !std::isfinite(radial) || !std::isfinite(distortedX) ||
        !std::isfinite(distortedY) || !std::isfinite(pixelX) ||
        !std::isfinite(pixelY)) {
        return ProjectionResult::NonFinite;
    }
    return ProjectionResult::Projected;
}

bool isInsideFrame(double x, double y,
                   const CameraCalibration &calibration) noexcept {
    return x >= 0.0 && y >= 0.0 &&
        x < static_cast<double>(calibration.width) &&
        y < static_cast<double>(calibration.height);
}

} // namespace

DatasetCaptureDiagnostics analyzeDatasetCapture(
    const DatasetDescriptor &descriptor) {
    validateDatasetDescriptor(descriptor);

    DatasetCaptureDiagnostics result;
    result.frameCount = descriptor.frames.size();
    result.pointCount = descriptor.points.count();
    result.observationCount = descriptor.observations.size();
    result.frames.resize(descriptor.frames.size());
    for (size_t index = 0; index < result.frames.size(); ++index)
        result.frames[index].frameIndex = static_cast<uint32_t>(index);

    std::vector<PointTrackState> pointTracks(descriptor.points.count());
    ReprojectionAccumulator reprojection;
    std::vector<ReprojectionAccumulator> frameReprojection(
        descriptor.frames.size());

    for (const SparseObservation &observation : descriptor.observations) {
        const size_t frameIndex = observation.frameIndex;
        const DatasetFrameDescriptor &frame = descriptor.frames[frameIndex];
        DatasetFrameCaptureDiagnostics &frameResult = result.frames[frameIndex];
        ++frameResult.observationCount;

        if (!isInsideFrame(observation.x, observation.y, frame.calibration)) {
            ++result.observedOutsideFrameCount;
            ++frameResult.observedOutsideFrameCount;
        }
        if (observation.pointIndex < 0) continue;

        ++result.linkedObservationCount;
        ++frameResult.linkedObservationCount;
        const size_t pointIndex = static_cast<size_t>(observation.pointIndex);
        PointTrackState &track = pointTracks[pointIndex];
        const uint32_t framePlusOne = observation.frameIndex + 1;
        if (track.lastFramePlusOne != framePlusOne) {
            track.lastFramePlusOne = framePlusOne;
            ++track.frameCount;
        }

        double projectedX = 0.0;
        double projectedY = 0.0;
        switch (projectPoint(
            frame, &descriptor.points.xyz[pointIndex * 3],
            projectedX, projectedY)) {
            case ProjectionResult::BehindCamera:
                ++result.behindCameraObservationCount;
                ++frameResult.behindCameraObservationCount;
                continue;
            case ProjectionResult::NonFinite:
                ++result.nonFiniteProjectionCount;
                ++frameResult.nonFiniteProjectionCount;
                continue;
            case ProjectionResult::Projected:
                break;
        }

        const double error = std::hypot(
            projectedX - observation.x,
            projectedY - observation.y);
        if (!std::isfinite(error)) {
            ++result.nonFiniteProjectionCount;
            ++frameResult.nonFiniteProjectionCount;
            continue;
        }

        ++result.reprojectedObservationCount;
        ++frameResult.reprojectedObservationCount;
        if (!isInsideFrame(projectedX, projectedY, frame.calibration)) {
            ++result.projectedOutsideFrameCount;
            ++frameResult.projectedOutsideFrameCount;
        }
        reprojection.add(error);
        frameReprojection[frameIndex].add(error);
    }

    uint64_t trackLengthSum = 0;
    for (const PointTrackState &track : pointTracks) {
        const uint32_t trackLength = track.frameCount;
        if (trackLength == 0) continue;
        ++result.observedPointCount;
        if (trackLength >= 2) ++result.multiViewPointCount;
        result.maximumTrackLength = std::max(
            result.maximumTrackLength, trackLength);
        trackLengthSum += trackLength;
    }
    if (result.observedPointCount != 0) {
        result.meanTrackLength = static_cast<double>(trackLengthSum) /
            static_cast<double>(result.observedPointCount);
    }

    result.reprojectionError = reprojection.finish();
    for (size_t index = 0; index < result.frames.size(); ++index) {
        result.frames[index].reprojectionError =
            frameReprojection[index].finish();
    }

    ReprojectionAccumulator sourceReprojection;
    for (float error : descriptor.points.reprojectionErrors)
        sourceReprojection.add(error);
    result.sourcePointReprojectionError = sourceReprojection.finish();
    return result;
}
