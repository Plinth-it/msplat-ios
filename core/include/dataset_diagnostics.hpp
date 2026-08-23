#ifndef MSPLAT_DATASET_DIAGNOSTICS_H
#define MSPLAT_DATASET_DIAGNOSTICS_H

#include "dataset_descriptor.hpp"

#include <cstdint>
#include <vector>

struct DatasetReprojectionStatistics {
    uint64_t sampleCount = 0;
    double meanPixels = 0.0;
    double rootMeanSquarePixels = 0.0;
    double maximumPixels = 0.0;
};

struct DatasetFrameCaptureDiagnostics {
    uint32_t frameIndex = 0;
    uint64_t observationCount = 0;
    uint64_t linkedObservationCount = 0;
    uint64_t observedOutsideFrameCount = 0;
    uint64_t reprojectedObservationCount = 0;
    uint64_t behindCameraObservationCount = 0;
    uint64_t nonFiniteProjectionCount = 0;
    uint64_t projectedOutsideFrameCount = 0;
    DatasetReprojectionStatistics reprojectionError;
};

struct DatasetCaptureDiagnostics {
    uint64_t frameCount = 0;
    uint64_t pointCount = 0;
    uint64_t observationCount = 0;
    uint64_t linkedObservationCount = 0;
    uint64_t observedOutsideFrameCount = 0;
    uint64_t reprojectedObservationCount = 0;
    uint64_t behindCameraObservationCount = 0;
    uint64_t nonFiniteProjectionCount = 0;
    uint64_t projectedOutsideFrameCount = 0;
    uint64_t observedPointCount = 0;
    uint64_t multiViewPointCount = 0;
    uint32_t maximumTrackLength = 0;
    double meanTrackLength = 0.0;
    DatasetReprojectionStatistics reprojectionError;
    DatasetReprojectionStatistics sourcePointReprojectionError;
    std::vector<DatasetFrameCaptureDiagnostics> frames;
};

/// Analyzes only immutable descriptor metadata. This does not decode assets,
/// normalize the scene, create a dataset handle, or initialize Metal.
DatasetCaptureDiagnostics analyzeDatasetCapture(
    const DatasetDescriptor &descriptor);

#endif // MSPLAT_DATASET_DIAGNOSTICS_H
