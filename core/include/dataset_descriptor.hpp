#ifndef MSPLAT_DATASET_DESCRIPTOR_H
#define MSPLAT_DATASET_DESCRIPTOR_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class RasterOrientation : uint8_t {
    EncodedPixels = 0,
    ExifNormalized = 1,
};

struct CameraCalibration {
    int32_t width = 0;
    int32_t height = 0;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float k1 = 0.0f;
    float k2 = 0.0f;
    float k3 = 0.0f;
    float p1 = 0.0f;
    float p2 = 0.0f;
};

struct DatasetFrameDescriptor {
    std::string id;
    std::string calibrationId;
    std::string imagePath;
    // Names the pixel frame used by calibration and cameraToWorld. The pose is
    // a rigid, row-major OpenGL camera-to-world transform (Y-up, Z-back).
    RasterOrientation rasterOrientation = RasterOrientation::EncodedPixels;
    CameraCalibration calibration;
    std::array<float, 16> cameraToWorld = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
};

struct SparsePointSet {
    std::vector<float> xyz;
    std::vector<uint8_t> rgb;
    std::vector<uint64_t> sourceIds;
    std::vector<float> reprojectionErrors;

    size_t count() const noexcept { return xyz.size() / 3; }
    bool empty() const noexcept { return xyz.empty(); }
};

struct SparseObservation {
    uint32_t frameIndex = 0;
    int32_t pointIndex = -1;
    float x = 0.0f;
    float y = 0.0f;
};

struct DatasetProvenance {
    std::string adapter;
    std::string source;
};

struct DatasetDescriptor {
    std::vector<DatasetFrameDescriptor> frames;
    SparsePointSet points;
    std::vector<SparseObservation> observations;
    DatasetProvenance provenance;
};

void validateDatasetDescriptor(const DatasetDescriptor &descriptor);

#endif // MSPLAT_DATASET_DESCRIPTOR_H
