#include "dataset_diagnostics.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

namespace {

constexpr double kTightTolerance = 1.0e-9;
constexpr double kFloatProjectionTolerance = 1.0e-4;

bool near(double actual, double expected,
          double tolerance = kTightTolerance) {
    return std::abs(actual - expected) <= tolerance;
}

DatasetFrameDescriptor identityFrame(uint32_t index) {
    DatasetFrameDescriptor frame;
    frame.id = "frame-" + std::to_string(index);
    frame.calibrationId = "camera-" + std::to_string(index);
    frame.imagePath = "images/" + std::to_string(index) + ".jpg";
    frame.calibration = {
        640, 480,
        100.0f, 100.0f,
        320.0f, 240.0f,
    };
    return frame;
}

DatasetDescriptor descriptorWithFrames(uint32_t frameCount) {
    DatasetDescriptor descriptor;
    for (uint32_t index = 0; index < frameCount; ++index)
        descriptor.frames.push_back(identityFrame(index));
    descriptor.provenance = {"test", "synthetic"};
    return descriptor;
}

void addPoint(DatasetDescriptor &descriptor, float x, float y, float z) {
    descriptor.points.xyz.insert(descriptor.points.xyz.end(), {x, y, z});
    descriptor.points.rgb.insert(descriptor.points.rgb.end(), {255, 255, 255});
}

int checkProjectionConventions() {
    DatasetDescriptor descriptor = descriptorWithFrames(2);

    // The second camera is translated and rotated +90 degrees about world Y.
    // Its OpenGL camera-to-world basis columns are right, up, and back.
    descriptor.frames[1].cameraToWorld = {
        0.0f, 0.0f, 1.0f, 10.0f,
        0.0f, 1.0f, 0.0f, 20.0f,
       -1.0f, 0.0f, 0.0f, 30.0f,
        0.0f, 0.0f, 0.0f,  1.0f,
    };

    // Identity camera: Z is back and Y is up, so the second point projects
    // down in the image. No half-pixel adjustment applies to observations.
    addPoint(descriptor, 0.0f, 0.0f, -2.0f);  // (320, 240)
    addPoint(descriptor, 1.0f, -2.0f, -4.0f); // (345, 290)

    // In camera 1 this is (x=1, y=2, z=4), despite its world transform.
    addPoint(descriptor, 6.0f, 18.0f, 29.0f); // (345, 290)
    descriptor.observations = {
        {0, 0, 0, 320.0f, 240.0f},
        {0, 1, 1, 345.0f, 290.0f},
        {1, 0, 2, 345.0f, 290.0f},
    };

    const DatasetCaptureDiagnostics result = analyzeDatasetCapture(descriptor);
    CHECK(result.frameCount == 2);
    CHECK(result.pointCount == 3);
    CHECK(result.observationCount == 3);
    CHECK(result.linkedObservationCount == 3);
    CHECK(result.reprojectedObservationCount == 3);
    CHECK(result.behindCameraObservationCount == 0);
    CHECK(result.nonFiniteProjectionCount == 0);
    CHECK(result.observedOutsideFrameCount == 0);
    CHECK(result.projectedOutsideFrameCount == 0);
    CHECK(result.observedPointCount == 3);
    CHECK(result.multiViewPointCount == 0);
    CHECK(result.maximumTrackLength == 1);
    CHECK(near(result.meanTrackLength, 1.0));
    CHECK(result.reprojectionError.sampleCount == 3);
    CHECK(near(result.reprojectionError.meanPixels, 0.0));
    CHECK(near(result.reprojectionError.rootMeanSquarePixels, 0.0));
    CHECK(near(result.reprojectionError.maximumPixels, 0.0));
    CHECK(result.sourcePointReprojectionError.sampleCount == 0);
    CHECK(result.frames.size() == 2);

    CHECK(result.frames[0].frameIndex == 0);
    CHECK(result.frames[0].observationCount == 2);
    CHECK(result.frames[0].linkedObservationCount == 2);
    CHECK(result.frames[0].reprojectedObservationCount == 2);
    CHECK(result.frames[0].reprojectionError.sampleCount == 2);
    CHECK(near(result.frames[0].reprojectionError.maximumPixels, 0.0));

    CHECK(result.frames[1].frameIndex == 1);
    CHECK(result.frames[1].observationCount == 1);
    CHECK(result.frames[1].linkedObservationCount == 1);
    CHECK(result.frames[1].reprojectedObservationCount == 1);
    CHECK(result.frames[1].reprojectionError.sampleCount == 1);
    CHECK(near(result.frames[1].reprojectionError.maximumPixels, 0.0));
    return 0;
}

int checkBrownConradyAndKnownResidual() {
    DatasetDescriptor descriptor = descriptorWithFrames(1);
    descriptor.frames[0].calibration = {
        640, 480,
        100.0f, 200.0f,
        320.0f, 240.0f,
        0.1f, 0.01f, 0.001f,
        0.02f, -0.03f,
    };

    // Normalized coordinates are x=0.5 and y=0.25. Brown-Conrady projects
    // this point to approximately (369.675354, 291.862854). The observation
    // is offset by (3, 4), making the expected residual exactly five pixels.
    addPoint(descriptor, 1.0f, -0.5f, -2.0f);
    descriptor.points.reprojectionErrors = {2.0f};
    descriptor.observations = {
        {0, 0, 0, 372.675354f, 295.862854f},
    };

    const DatasetCaptureDiagnostics result = analyzeDatasetCapture(descriptor);
    CHECK(result.reprojectionError.sampleCount == 1);
    CHECK(near(result.reprojectionError.meanPixels, 5.0,
               kFloatProjectionTolerance));
    CHECK(near(result.reprojectionError.rootMeanSquarePixels, 5.0,
               kFloatProjectionTolerance));
    CHECK(near(result.reprojectionError.maximumPixels, 5.0,
               kFloatProjectionTolerance));

    CHECK(result.sourcePointReprojectionError.sampleCount == 1);
    CHECK(near(result.sourcePointReprojectionError.meanPixels, 2.0));
    CHECK(near(result.sourcePointReprojectionError.rootMeanSquarePixels, 2.0));
    CHECK(near(result.sourcePointReprojectionError.maximumPixels, 2.0));
    return 0;
}

int checkTracksBoundsAndPerFrameStatistics() {
    DatasetDescriptor descriptor = descriptorWithFrames(2);
    addPoint(descriptor, 0.0f, 0.0f, -2.0f);  // projects inside both frames
    addPoint(descriptor, 10.0f, 0.0f, -2.0f); // projects to x=820
    addPoint(descriptor, 0.0f, 0.0f, -3.0f);  // deliberately unobserved
    descriptor.points.reprojectionErrors = {1.0f, 2.0f, 3.0f};

    descriptor.observations = {
        {0, 0, 0, 320.0f, 240.0f},
        {0, 1, -1, -10.0f, 700.0f},
        {0, 2, 1, 820.0f, 240.0f},
        {1, 0, 0, 323.0f, 244.0f},
    };

    const DatasetCaptureDiagnostics result = analyzeDatasetCapture(descriptor);
    CHECK(result.observationCount == 4);
    CHECK(result.linkedObservationCount == 3);
    CHECK(result.observedOutsideFrameCount == 2);
    CHECK(result.reprojectedObservationCount == 3);
    CHECK(result.behindCameraObservationCount == 0);
    CHECK(result.nonFiniteProjectionCount == 0);
    CHECK(result.projectedOutsideFrameCount == 1);

    CHECK(result.observedPointCount == 2);
    CHECK(result.multiViewPointCount == 1);
    CHECK(result.maximumTrackLength == 2);
    CHECK(near(result.meanTrackLength, 1.5));

    CHECK(result.reprojectionError.sampleCount == 3);
    CHECK(near(result.reprojectionError.meanPixels, 5.0 / 3.0));
    CHECK(near(result.reprojectionError.rootMeanSquarePixels,
               std::sqrt(25.0 / 3.0)));
    CHECK(near(result.reprojectionError.maximumPixels, 5.0));

    CHECK(result.sourcePointReprojectionError.sampleCount == 3);
    CHECK(near(result.sourcePointReprojectionError.meanPixels, 2.0));
    CHECK(near(result.sourcePointReprojectionError.rootMeanSquarePixels,
               std::sqrt(14.0 / 3.0)));
    CHECK(near(result.sourcePointReprojectionError.maximumPixels, 3.0));

    CHECK(result.frames.size() == 2);
    const DatasetFrameCaptureDiagnostics &first = result.frames[0];
    CHECK(first.frameIndex == 0);
    CHECK(first.observationCount == 3);
    CHECK(first.linkedObservationCount == 2);
    CHECK(first.observedOutsideFrameCount == 2);
    CHECK(first.reprojectedObservationCount == 2);
    CHECK(first.projectedOutsideFrameCount == 1);
    CHECK(first.reprojectionError.sampleCount == 2);
    CHECK(near(first.reprojectionError.meanPixels, 0.0));

    const DatasetFrameCaptureDiagnostics &second = result.frames[1];
    CHECK(second.frameIndex == 1);
    CHECK(second.observationCount == 1);
    CHECK(second.linkedObservationCount == 1);
    CHECK(second.observedOutsideFrameCount == 0);
    CHECK(second.reprojectedObservationCount == 1);
    CHECK(second.projectedOutsideFrameCount == 0);
    CHECK(second.reprojectionError.sampleCount == 1);
    CHECK(near(second.reprojectionError.meanPixels, 5.0));
    CHECK(near(second.reprojectionError.rootMeanSquarePixels, 5.0));
    CHECK(near(second.reprojectionError.maximumPixels, 5.0));
    return 0;
}

int checkUnprojectableObservations() {
    DatasetDescriptor descriptor = descriptorWithFrames(1);
    descriptor.frames[0].calibration.k3 =
        std::numeric_limits<float>::max();

    addPoint(descriptor, 0.0f, 0.0f, 1.0f); // behind the identity camera
    addPoint(descriptor, std::numeric_limits<float>::max(), 0.0f, -1.0f);
    descriptor.observations = {
        {0, 0, 0, 320.0f, 240.0f},
        {0, 1, 1, 320.0f, 240.0f},
    };

    const DatasetCaptureDiagnostics result = analyzeDatasetCapture(descriptor);
    CHECK(result.observationCount == 2);
    CHECK(result.linkedObservationCount == 2);
    CHECK(result.reprojectedObservationCount == 0);
    CHECK(result.behindCameraObservationCount == 1);
    CHECK(result.nonFiniteProjectionCount == 1);
    CHECK(result.projectedOutsideFrameCount == 0);
    CHECK(result.reprojectionError.sampleCount == 0);
    CHECK(near(result.reprojectionError.meanPixels, 0.0));
    CHECK(near(result.reprojectionError.rootMeanSquarePixels, 0.0));
    CHECK(near(result.reprojectionError.maximumPixels, 0.0));

    CHECK(result.frames.size() == 1);
    CHECK(result.frames[0].behindCameraObservationCount == 1);
    CHECK(result.frames[0].nonFiniteProjectionCount == 1);
    CHECK(result.frames[0].reprojectionError.sampleCount == 0);
    return 0;
}

int checkEmptyOptionalDiagnostics() {
    DatasetDescriptor descriptor = descriptorWithFrames(1);
    addPoint(descriptor, 0.0f, 0.0f, -2.0f);

    const DatasetCaptureDiagnostics result = analyzeDatasetCapture(descriptor);
    CHECK(result.frameCount == 1);
    CHECK(result.pointCount == 1);
    CHECK(result.observationCount == 0);
    CHECK(result.linkedObservationCount == 0);
    CHECK(result.observedOutsideFrameCount == 0);
    CHECK(result.reprojectedObservationCount == 0);
    CHECK(result.behindCameraObservationCount == 0);
    CHECK(result.nonFiniteProjectionCount == 0);
    CHECK(result.projectedOutsideFrameCount == 0);
    CHECK(result.observedPointCount == 0);
    CHECK(result.multiViewPointCount == 0);
    CHECK(result.maximumTrackLength == 0);
    CHECK(near(result.meanTrackLength, 0.0));
    CHECK(result.reprojectionError.sampleCount == 0);
    CHECK(result.sourcePointReprojectionError.sampleCount == 0);
    CHECK(near(result.sourcePointReprojectionError.meanPixels, 0.0));
    CHECK(near(result.sourcePointReprojectionError.rootMeanSquarePixels, 0.0));
    CHECK(near(result.sourcePointReprojectionError.maximumPixels, 0.0));

    CHECK(result.frames.size() == 1);
    CHECK(result.frames[0].frameIndex == 0);
    CHECK(result.frames[0].observationCount == 0);
    CHECK(result.frames[0].linkedObservationCount == 0);
    CHECK(result.frames[0].reprojectionError.sampleCount == 0);
    return 0;
}

int checkTrackLengthsCountDistinctFrames() {
    DatasetDescriptor descriptor = descriptorWithFrames(2);
    addPoint(descriptor, 0.0f, 0.0f, -2.0f);
    descriptor.observations = {
        {0, 0, 0, 320.0f, 240.0f},
        {0, 1, 0, 320.0f, 240.0f},
        {1, 0, 0, 320.0f, 240.0f},
    };

    const DatasetCaptureDiagnostics result = analyzeDatasetCapture(descriptor);
    CHECK(result.linkedObservationCount == 3);
    CHECK(result.observedPointCount == 1);
    CHECK(result.multiViewPointCount == 1);
    CHECK(result.maximumTrackLength == 2);
    CHECK(near(result.meanTrackLength, 2.0));
    return 0;
}

} // namespace

int main() {
    int result = checkProjectionConventions();
    if (result != 0) return result;
    result = checkBrownConradyAndKnownResidual();
    if (result != 0) return result;
    result = checkTracksBoundsAndPerFrameStatistics();
    if (result != 0) return result;
    result = checkUnprojectableObservations();
    if (result != 0) return result;
    result = checkEmptyOptionalDiagnostics();
    if (result != 0) return result;
    return checkTrackLengthsCountDistinctFrames();
}
