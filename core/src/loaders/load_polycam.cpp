#include "loaders.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <utility>

namespace fs = std::filesystem;
using json = nlohmann::json;

static std::string findImage(const fs::path &dir, const std::string &stem) {
    for (auto ext : {".png", ".jpg", ".jpeg", ".JPG"}) {
        auto p = dir / (stem + ext);
        if (fs::exists(p)) return p.string();
    }
    return (dir / (stem + ".png")).string();
}

DatasetDescriptor loaders::loadPolycam(const std::string &projectRoot) {
    DatasetDescriptor data;
    data.provenance.adapter = "polycam";
    data.provenance.source = projectRoot;
    fs::path root(projectRoot);
    fs::path keyframesDir = root / "keyframes" / "corrected_cameras";
    fs::path imagesDir = root / "keyframes" / "corrected_images";

    if (fs::exists(keyframesDir)) {
        // Layout 1: individual camera JSONs
        std::vector<fs::path> jsonFiles;
        for (auto &entry : fs::directory_iterator(keyframesDir))
            if (entry.path().extension() == ".json") jsonFiles.push_back(entry.path());
        std::sort(jsonFiles.begin(), jsonFiles.end());

        for (auto &jp : jsonFiles) {
            std::ifstream f(jp);
            json j = json::parse(f);

            DatasetFrameDescriptor frame;
            frame.id = jp.stem().string();
            frame.calibrationId = frame.id;
            frame.calibration.width = j.value("width", 0);
            frame.calibration.height = j.value("height", 0);
            frame.calibration.fx = j.value("fx", 0.0f);
            frame.calibration.fy = j.value("fy", 0.0f);
            frame.calibration.cx = j.value("cx", (float)frame.calibration.width / 2.0f);
            frame.calibration.cy = j.value("cy", (float)frame.calibration.height / 2.0f);
            frame.rasterOrientation = RasterOrientation::EncodedPixels;

            float R[9], T[3];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    R[r*3+c] = j.value("R_" + std::to_string(r) + std::to_string(c), 0.0f);
            T[0] = j.value("t_20", 0.0f);
            T[1] = j.value("t_21", 0.0f);
            T[2] = j.value("t_22", 0.0f);

            // c2w with OpenGL Y/Z flip
            frame.cameraToWorld[0]  =  R[0]; frame.cameraToWorld[1]  =  R[1]; frame.cameraToWorld[2]  =  R[2]; frame.cameraToWorld[3]  =  T[0];
            frame.cameraToWorld[4]  = -R[3]; frame.cameraToWorld[5]  = -R[4]; frame.cameraToWorld[6]  = -R[5]; frame.cameraToWorld[7]  = -T[1];
            frame.cameraToWorld[8]  = -R[6]; frame.cameraToWorld[9]  = -R[7]; frame.cameraToWorld[10] = -R[8]; frame.cameraToWorld[11] = -T[2];
            frame.cameraToWorld[12] = 0;     frame.cameraToWorld[13] = 0;     frame.cameraToWorld[14] = 0;     frame.cameraToWorld[15] = 1;

            frame.imagePath = findImage(imagesDir, frame.id);
            data.frames.push_back(std::move(frame));
        }
    } else {
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
    }

    // Point cloud
    for (auto p : {"keyframes/point_cloud.ply", "point_cloud.ply", "sparse.ply"}) {
        auto path = (root / p).string();
        if (fs::exists(path)) { data.points = readPly(path); break; }
    }

    return data;
}
