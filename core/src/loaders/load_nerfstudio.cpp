#include "loaders.hpp"
#include <nlohmann/json.hpp>
#include <cctype>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <utility>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Try adding common image extensions if file doesn't exist
static std::string resolveImagePath(const std::string &path) {
    if (fs::exists(path)) return path;
    for (auto ext : {".png", ".jpg", ".jpeg", ".JPG"})
        if (fs::exists(path + ext)) return path + ext;
    return path;
}

static bool isInside(const fs::path &root, const fs::path &candidate) {
    auto rootComponent = root.begin();
    auto candidateComponent = candidate.begin();
    for (; rootComponent != root.end() && candidateComponent != candidate.end();
         ++rootComponent, ++candidateComponent) {
        if (*rootComponent != *candidateComponent) return false;
    }
    return rootComponent == root.end();
}

static void validateCameraModel(const json &container,
                                const std::string &context) {
    const auto modelJson = container.find("camera_model");
    if (modelJson == container.end() || modelJson->is_null()) return;
    if (!modelJson->is_string()) {
        throw std::runtime_error(
            context + " camera_model must be a string");
    }

    const std::string declaredModel = modelJson->get<std::string>();
    std::string normalizedModel = declaredModel;
    std::transform(
        normalizedModel.begin(), normalizedModel.end(), normalizedModel.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::toupper(character));
        });
    if (normalizedModel == "PINHOLE" ||
        normalizedModel == "PERSPECTIVE" ||
        normalizedModel == "OPENCV") {
        return;
    }

    throw std::runtime_error(
        context + " uses unsupported camera_model '" + declaredModel +
        "'; supported models are PINHOLE, PERSPECTIVE, and OPENCV");
}

static fs::path resolveMaskPath(const fs::path &projectRoot,
                                const std::string &path) {
    if (path.empty())
        throw std::runtime_error("Nerfstudio frame has an empty mask_path");

    std::error_code error;
    const fs::path canonicalRoot = fs::canonical(projectRoot, error);
    if (error)
        throw std::runtime_error("Nerfstudio dataset root could not be resolved");

    fs::path candidate(path);
    if (!candidate.is_absolute()) candidate = projectRoot / candidate;
    const fs::path canonicalCandidate = fs::canonical(candidate, error);
    if (error || !fs::is_regular_file(canonicalCandidate)) {
        throw std::runtime_error(
            "Nerfstudio mask_path must reference a regular file: " + path);
    }
    if (!isInside(canonicalRoot, canonicalCandidate)) {
        throw std::runtime_error(
            "Nerfstudio mask_path must stay inside the dataset root: " + path);
    }
    return canonicalCandidate;
}

DatasetDescriptor loaders::loadNerfstudio(const std::string &projectRoot) {
    std::ifstream f((fs::path(projectRoot) / "transforms.json").string());
    json j = json::parse(f);
    validateCameraModel(j, "Nerfstudio dataset");

    // Global defaults (overridden per-frame if present)
    int gW = j.value("w", 0), gH = j.value("h", 0);
    float gFx = j.value("fl_x", 0.0f), gFy = j.value("fl_y", 0.0f);
    float gCx = j.value("cx", 0.0f), gCy = j.value("cy", 0.0f);
    float gK1 = j.value("k1", 0.0f), gK2 = j.value("k2", 0.0f), gK3 = j.value("k3", 0.0f);
    float gP1 = j.value("p1", 0.0f), gP2 = j.value("p2", 0.0f);

    DatasetDescriptor data;
    data.provenance.adapter = "nerfstudio";
    data.provenance.source = projectRoot;

    size_t maskedFrameCount = 0;
    for (auto &frameJson : j["frames"]) {
        std::string fp = frameJson["file_path"].get<std::string>();
        if (fp.empty())
            throw std::runtime_error("Nerfstudio frame has an empty file_path");
        validateCameraModel(frameJson, "Nerfstudio frame '" + fp + "'");

        DatasetFrameDescriptor frame;
        frame.calibration.width = frameJson.value("w", gW);
        frame.calibration.height = frameJson.value("h", gH);
        frame.calibration.fx = frameJson.value("fl_x", gFx);
        frame.calibration.fy = frameJson.value("fl_y", gFy);
        frame.calibration.cx = frameJson.value("cx", gCx);
        frame.calibration.cy = frameJson.value("cy", gCy);
        frame.calibration.k1 = frameJson.value("k1", gK1);
        frame.calibration.k2 = frameJson.value("k2", gK2);
        frame.calibration.k3 = frameJson.value("k3", gK3);
        frame.calibration.p1 = frameJson.value("p1", gP1);
        frame.calibration.p2 = frameJson.value("p2", gP2);

        frame.id = fp;
        frame.calibrationId = fp;
        fs::path imagePath(fp);
        if (!imagePath.is_absolute()) imagePath = fs::path(projectRoot) / imagePath;
        frame.imagePath = resolveImagePath(imagePath.lexically_normal().string());
        frame.rasterOrientation = RasterOrientation::EncodedPixels;

        if (frameJson.contains("mask_path")) {
            if (!frameJson["mask_path"].is_string()) {
                throw std::runtime_error(
                    "Nerfstudio frame mask_path must be a string");
            }
            const std::string maskPath =
                frameJson["mask_path"].get<std::string>();
            frame.trainingMask = TrainingMaskDescriptor{
                resolveMaskPath(fs::path(projectRoot), maskPath).string(),
                TrainingMaskChannel::Automatic};
            ++maskedFrameCount;
        }

        // transform_matrix is 4x4 c2w (OpenGL convention)
        auto &tm = frameJson["transform_matrix"];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                frame.cameraToWorld[r*4+c] = tm[r][c].get<float>();

        data.frames.push_back(std::move(frame));
    }

    if (maskedFrameCount != 0 && maskedFrameCount != data.frames.size()) {
        throw std::runtime_error(
            "Nerfstudio mask_path must be present for every frame or no frames");
    }

    std::sort(data.frames.begin(), data.frames.end(),
        [](const DatasetFrameDescriptor &a, const DatasetFrameDescriptor &b) {
            return a.imagePath < b.imagePath;
        });

    // Point cloud
    if (j.contains("ply_file_path")) {
        std::string p = j["ply_file_path"].get<std::string>();
        if (!p.empty()) {
            fs::path pointPath(p);
            if (!pointPath.is_absolute())
                pointPath = fs::path(projectRoot) / pointPath;
            pointPath = pointPath.lexically_normal();
            if (fs::exists(pointPath)) data.points = readPly(pointPath.string());
        }
    }
    if (data.points.empty()) {
        for (auto p : {"sparse/0/points3D.ply", "points3D.ply"}) {
            auto path = (fs::path(projectRoot) / p).string();
            if (fs::exists(path)) { data.points = readPly(path); break; }
        }
    }

    return data;
}
