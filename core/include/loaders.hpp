#ifndef LOADERS_H
#define LOADERS_H

#include "dataset_descriptor.hpp"
#include "input_data.hpp"

// Format-specific loaders
namespace loaders {
    DatasetDescriptor loadColmap(const std::string &projectRoot, const std::string &imageSourcePath = "");
    DatasetDescriptor loadNerfstudio(const std::string &projectRoot);
    DatasetDescriptor loadPolycam(const std::string &projectRoot);
}

// PLY point cloud reader
SparsePointSet readPly(const std::string &path);

// COLMAP binary point cloud reader
SparsePointSet readColmapPoints(const std::string &path);

// Image I/O
struct ImageSourceInfo {
    int rawWidth = 0;
    int rawHeight = 0;
    int orientedWidth = 0;
    int orientedHeight = 0;
    int exifOrientation = 1;
};

/// Reads dimensions and EXIF orientation without decoding the image pixels.
ImageSourceInfo inspectImageSource(const std::string &path);

/// Decodes directly near the requested size, optionally applies EXIF
/// orientation, then renders into an exact-size sRGB float32 RGB image in
/// [0, 1]. `sourceInfo` must come from `inspectImageSource` for the same
/// unchanged file. Callers must only request orientation normalization when
/// their camera calibration uses the orientation-normalized pixel frame.
Image imreadRGB(const std::string &path, const ImageSourceInfo &sourceInfo,
                int targetWidth, int targetHeight,
                bool applyExifOrientation);
Image resizeArea(const Image &src, int dstW, int dstH);  // box-filter downscale
void imwriteRGB(const std::string &path, const Image &img);  // save as PNG

// Undistortion (Brown-Conrady model, alpha=0 crop)
struct UndistortResult {
    Image image;
    float fx, fy, cx, cy;  // updated intrinsics after crop
    int width, height;      // cropped dimensions
};
UndistortResult undistortImage(const Image &src,
    float fx, float fy, float cx, float cy,
    float k1, float k2, float p1, float p2, float k3);

// Pose utilities
void autoScaleAndCenter(InputData &data);

// Gaussian PLY/splat I/O (trained scene export/import)
struct GaussianParams {
    MTensor &means, &scales, &quats, &featuresDc, &featuresRest, &opacities;
    float scale;          // CRS scale factor
    float translation[3]; // CRS translation
    bool keepCrs;
};

void saveGaussianPly(const std::string &path, GaussianParams &p, int step);
void saveGaussianSplat(const std::string &path, GaussianParams &p);
void saveGaussianSpz(const std::string &path, GaussianParams &p);

struct LoadedGaussians {
    MTensor means, scales, quats, featuresDc, featuresRest, opacities;
    int step;
};
LoadedGaussians loadGaussianPly(const std::string &path, float scale, const float translation[3], bool keepCrs);

#endif
