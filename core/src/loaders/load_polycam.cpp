#include "loaders.hpp"
#include "dataset_errors.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace fs = std::filesystem;
using json = nlohmann::json;

static std::optional<fs::path> findImage(
    const fs::path &dir, const std::string &stem) {
    for (auto ext : {".png", ".jpg", ".jpeg", ".JPG"}) {
        auto p = dir / (stem + ext);
        if (fs::is_regular_file(p)) return p;
    }
    return std::nullopt;
}

static float requiredFloatValue(
    const json &camera, const std::string &key, const fs::path &path) {
    const auto value = camera.find(key);
    if (value == camera.end() || !value->is_number()) {
        throw msplat::InvalidDatasetError(
            "Polycam camera " + path.string() +
            " is missing numeric field " + key);
    }

    const double number = value->get<double>();
    if (!std::isfinite(number) ||
        number < -std::numeric_limits<float>::max() ||
        number > std::numeric_limits<float>::max()) {
        throw msplat::InvalidDatasetError(
            "Polycam camera " + path.string() +
            " has an invalid field " + key);
    }
    return static_cast<float>(number);
}

static int32_t requiredDimensionValue(
    const json &camera, const std::string &key, const fs::path &path) {
    const auto value = camera.find(key);
    if (value == camera.end() ||
        (!value->is_number_integer() && !value->is_number_unsigned())) {
        throw msplat::InvalidDatasetError(
            "Polycam camera " + path.string() +
            " is missing integer field " + key);
    }

    int64_t number = 0;
    try {
        number = value->get<int64_t>();
    } catch (const json::exception &) {
        throw msplat::InvalidDatasetError(
            "Polycam camera " + path.string() +
            " has an invalid field " + key);
    }
    if (number <= 0 || number > std::numeric_limits<int32_t>::max()) {
        throw msplat::InvalidDatasetError(
            "Polycam camera " + path.string() +
            " has an invalid field " + key);
    }
    return static_cast<int32_t>(number);
}

static void appendCameraIds(
    const fs::path &directory, std::vector<std::string> &ids) {
    if (!fs::is_directory(directory)) return;
    for (const auto &entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json")
            ids.push_back(entry.path().stem().string());
    }
}

DatasetDescriptor loaders::loadPolycam(const std::string &projectRoot) {
    DatasetDescriptor data;
    data.provenance.adapter = "polycam";
    data.provenance.source = projectRoot;
    fs::path root(projectRoot);
    const fs::path correctedCamerasDir =
        root / "keyframes" / "corrected_cameras";
    const fs::path correctedImagesDir =
        root / "keyframes" / "corrected_images";
    const fs::path rawCamerasDir = root / "keyframes" / "cameras";
    const fs::path rawImagesDir = root / "keyframes" / "images";
    const bool hasFolderLayout =
        fs::is_directory(correctedCamerasDir) ||
        fs::is_directory(rawCamerasDir);

    if (hasFolderLayout) {
        // Layout 1: individual camera JSONs
        std::vector<std::string> frameIds;
        appendCameraIds(rawCamerasDir, frameIds);
        appendCameraIds(correctedCamerasDir, frameIds);
        std::sort(frameIds.begin(), frameIds.end());
        frameIds.erase(
            std::unique(frameIds.begin(), frameIds.end()), frameIds.end());
        if (frameIds.empty()) {
            throw msplat::InvalidDatasetError(
                "Polycam keyframes contain no camera JSON files");
        }

        for (const std::string &frameId : frameIds) {
            const fs::path correctedCamera =
                correctedCamerasDir / (frameId + ".json");
            const fs::path rawCamera = rawCamerasDir / (frameId + ".json");
            const std::optional<fs::path> correctedImage =
                findImage(correctedImagesDir, frameId);
            const std::optional<fs::path> rawImage =
                findImage(rawImagesDir, frameId);

            fs::path cameraPath;
            std::optional<fs::path> imagePath;
            if (fs::is_regular_file(correctedCamera) && correctedImage) {
                cameraPath = correctedCamera;
                imagePath = correctedImage;
            } else if (fs::is_regular_file(rawCamera) && rawImage) {
                cameraPath = rawCamera;
                imagePath = rawImage;
            } else {
                throw msplat::InvalidDatasetError(
                    "Polycam frame '" + frameId +
                    "' has no complete corrected or raw camera/image pair");
            }

            const fs::path &jp = cameraPath;
            std::ifstream f(jp);
            json j = json::parse(f);

            DatasetFrameDescriptor frame;
            frame.id = frameId;
            frame.calibrationId = frame.id;
            frame.calibration.width =
                requiredDimensionValue(j, "width", jp);
            frame.calibration.height =
                requiredDimensionValue(j, "height", jp);
            frame.calibration.fx = requiredFloatValue(j, "fx", jp);
            frame.calibration.fy = requiredFloatValue(j, "fy", jp);
            frame.calibration.cx = requiredFloatValue(j, "cx", jp);
            frame.calibration.cy = requiredFloatValue(j, "cy", jp);
            frame.rasterOrientation = RasterOrientation::EncodedPixels;

            // Polycam writes the upper three rows of its ARKit camera-to-world
            // transform as t_00...t_23. ARKit and msplat both use a
            // right-handed, Y-up camera with -Z forward, so no camera-axis
            // flip or matrix inversion applies here.
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 4; ++column) {
                    const std::string key =
                        "t_" + std::to_string(row) +
                        std::to_string(column);
                    frame.cameraToWorld[row * 4 + column] =
                        requiredFloatValue(j, key, jp);
                }
            }
            frame.cameraToWorld[12] = 0.0f;
            frame.cameraToWorld[13] = 0.0f;
            frame.cameraToWorld[14] = 0.0f;
            frame.cameraToWorld[15] = 1.0f;

            frame.imagePath = imagePath->string();
            data.frames.push_back(std::move(frame));
        }
    } else if (fs::exists(root / "cameras.json")) {
        // Layout 2: single cameras.json (dispatcher already confirmed it exists)
        std::ifstream f(root / "cameras.json");
        json j = json::parse(f);

        auto &frames = j.contains("frames") ? j["frames"] : j;
        for (auto &frameJson : frames) {
            DatasetFrameDescriptor frame;
            frame.calibration.width = frameJson.value("width", 0);
            frame.calibration.height = frameJson.value("height", 0);
            frame.calibration.fx = frameJson.value("fx", 0.0f);
            frame.calibration.fy = frameJson.value("fy", 0.0f);
            frame.calibration.cx = frameJson.value("cx", (float)frame.calibration.width / 2.0f);
            frame.calibration.cy = frameJson.value("cy", (float)frame.calibration.height / 2.0f);
            frame.rasterOrientation = RasterOrientation::EncodedPixels;

            if (!frameJson.contains("transform_matrix")) {
                throw std::runtime_error(
                    "Polycam frame " + std::to_string(data.frames.size()) +
                    " is missing transform_matrix");
            }
            auto &tm = frameJson["transform_matrix"];
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    frame.cameraToWorld[r*4+c] = tm[r][c].get<float>();

            std::string sourcePath = frameJson.value("file_path", "");
            frame.id = sourcePath.empty()
                ? "frame-" + std::to_string(data.frames.size())
                : sourcePath;
            frame.calibrationId = frame.id;
            frame.imagePath = sourcePath;
            if (!frame.imagePath.empty() && frame.imagePath[0] != '/')
                frame.imagePath = (root / frame.imagePath).string();
            data.frames.push_back(std::move(frame));
        }
    } else {
        throw msplat::InvalidDatasetError(
            "Polycam dataset must contain paired corrected_cameras and "
            "corrected_images directories, paired cameras and images "
            "directories, or cameras.json");
    }

    // Point cloud
    for (auto p : {"keyframes/point_cloud.ply", "point_cloud.ply", "sparse.ply"}) {
        auto path = (root / p).string();
        if (fs::exists(path)) { data.points = readPly(path); break; }
    }

    return data;
}
