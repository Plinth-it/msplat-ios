#include "loaders.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <utility>

namespace fs = std::filesystem;

// Quaternion [w,x,y,z] → row-major 3x3 rotation matrix
static void quatToRotMat(const double q[4], float R[9]) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double n = std::sqrt(w*w + x*x + y*y + z*z);
    w /= n; x /= n; y /= n; z /= n;

    R[0] = (float)(1 - 2*(y*y + z*z));  R[1] = (float)(2*(x*y - w*z));      R[2] = (float)(2*(x*z + w*y));
    R[3] = (float)(2*(x*y + w*z));      R[4] = (float)(1 - 2*(x*x + z*z));  R[5] = (float)(2*(y*z - w*x));
    R[6] = (float)(2*(x*z - w*y));      R[7] = (float)(2*(y*z + w*x));      R[8] = (float)(1 - 2*(x*x + y*y));
}

enum ColmapModel { SIMPLE_PINHOLE=0, PINHOLE=1, SIMPLE_RADIAL=2, RADIAL=3, OPENCV=4 };

struct ColmapCamera {
    uint32_t id;
    int model;
    int width, height;
    float fx, fy, cx, cy;
    float k1, k2, p1, p2;
};

struct ColmapImage {
    uint32_t id;
    uint32_t camId;
    double quat[4]; // w, x, y, z
    double t[3];    // world-to-camera translation
    std::string filename;
};

static std::unordered_map<uint32_t, ColmapCamera> readCamerasBin(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    uint64_t n;
    f.read(reinterpret_cast<char*>(&n), 8);

    std::unordered_map<uint32_t, ColmapCamera> cams;
    for (uint64_t i = 0; i < n; i++) {
        ColmapCamera c = {};
        uint32_t model;
        uint64_t w, h;
        f.read(reinterpret_cast<char*>(&c.id), 4);
        f.read(reinterpret_cast<char*>(&model), 4);
        f.read(reinterpret_cast<char*>(&w), 8);
        f.read(reinterpret_cast<char*>(&h), 8);
        c.model = (int)model;
        c.width = (int)w;
        c.height = (int)h;

        auto rd = [&]() -> double { double v; f.read(reinterpret_cast<char*>(&v), 8); return v; };

        switch (c.model) {
            case SIMPLE_PINHOLE: c.fx = c.fy = (float)rd(); c.cx = (float)rd(); c.cy = (float)rd(); break;
            case PINHOLE:        c.fx = (float)rd(); c.fy = (float)rd(); c.cx = (float)rd(); c.cy = (float)rd(); break;
            case SIMPLE_RADIAL:  c.fx = c.fy = (float)rd(); c.cx = (float)rd(); c.cy = (float)rd(); c.k1 = (float)rd(); break;
            case RADIAL:         c.fx = c.fy = (float)rd(); c.cx = (float)rd(); c.cy = (float)rd(); c.k1 = (float)rd(); c.k2 = (float)rd(); break;
            case OPENCV:         c.fx = (float)rd(); c.fy = (float)rd(); c.cx = (float)rd(); c.cy = (float)rd();
                                 c.k1 = (float)rd(); c.k2 = (float)rd(); c.p1 = (float)rd(); c.p2 = (float)rd(); break;
            default: throw std::runtime_error("Unsupported COLMAP camera model: " + std::to_string(c.model));
        }
        cams[c.id] = c;
    }
    return cams;
}

static std::vector<ColmapImage> readImagesBin(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    uint64_t n;
    f.read(reinterpret_cast<char*>(&n), 8);

    std::vector<ColmapImage> images;
    images.reserve(n);

    for (uint64_t i = 0; i < n; i++) {
        ColmapImage img = {};
        f.read(reinterpret_cast<char*>(&img.id), 4);
        f.read(reinterpret_cast<char*>(img.quat), 32); // 4 doubles
        f.read(reinterpret_cast<char*>(img.t), 24);     // 3 doubles
        f.read(reinterpret_cast<char*>(&img.camId), 4);

        char ch;
        while (f.read(&ch, 1) && ch != '\0') img.filename += ch;

        uint64_t numPts2D;
        f.read(reinterpret_cast<char*>(&numPts2D), 8);
        f.seekg(numPts2D * 24, std::ios::cur);

        images.push_back(img);
    }
    return images;
}

// ── Text model ──────────────────────────────────────────────────────────────
// COLMAP writes the same model as either .bin or .txt. The text form is what
// tooling that assembles a reconstruction by hand tends to produce, and what
// COLMAP itself writes with --output_type TXT.

static bool nextRecord(std::istream &f, std::string &line) {
    while (std::getline(f, line)) {
        size_t start = line.find_first_not_of(" \t\r");
        if (start == std::string::npos || line[start] == '#') continue;
        return true;
    }
    return false;
}

static int colmapModelId(const std::string &name) {
    if (name == "SIMPLE_PINHOLE") return SIMPLE_PINHOLE;
    if (name == "PINHOLE")        return PINHOLE;
    if (name == "SIMPLE_RADIAL")  return SIMPLE_RADIAL;
    if (name == "RADIAL")         return RADIAL;
    if (name == "OPENCV")         return OPENCV;
    throw std::runtime_error("Unsupported COLMAP camera model: " + name);
}

static std::unordered_map<uint32_t, ColmapCamera> readCamerasText(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + path);

    std::unordered_map<uint32_t, ColmapCamera> cams;
    std::string line;
    while (nextRecord(f, line)) {
        std::istringstream ls(line);
        ColmapCamera c = {};
        std::string model;
        ls >> c.id >> model >> c.width >> c.height;
        c.model = colmapModelId(model);

        auto rd = [&]() -> float {
            double v;
            if (!(ls >> v)) throw std::runtime_error("Truncated camera record in " + path);
            return (float)v;
        };
        switch (c.model) {
            case SIMPLE_PINHOLE: c.fx = c.fy = rd(); c.cx = rd(); c.cy = rd(); break;
            case PINHOLE:        c.fx = rd(); c.fy = rd(); c.cx = rd(); c.cy = rd(); break;
            case SIMPLE_RADIAL:  c.fx = c.fy = rd(); c.cx = rd(); c.cy = rd(); c.k1 = rd(); break;
            case RADIAL:         c.fx = c.fy = rd(); c.cx = rd(); c.cy = rd(); c.k1 = rd(); c.k2 = rd(); break;
            case OPENCV:         c.fx = rd(); c.fy = rd(); c.cx = rd(); c.cy = rd();
                                 c.k1 = rd(); c.k2 = rd(); c.p1 = rd(); c.p2 = rd(); break;
        }
        cams[c.id] = c;
    }
    return cams;
}

static std::vector<ColmapImage> readImagesText(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + path);

    std::vector<ColmapImage> images;
    std::string line;
    while (nextRecord(f, line)) {
        std::istringstream ls(line);
        ColmapImage img = {};
        ls >> img.id
           >> img.quat[0] >> img.quat[1] >> img.quat[2] >> img.quat[3]
           >> img.t[0] >> img.t[1] >> img.t[2]
           >> img.camId;
        if (!ls) throw std::runtime_error("Truncated image record in " + path);

        // The name is the rest of the line, not a token — COLMAP permits
        // spaces in it and writes it unquoted.
        std::getline(ls, img.filename);
        size_t start = img.filename.find_first_not_of(" \t");
        size_t end = img.filename.find_last_not_of(" \t\r");
        img.filename = (start == std::string::npos)
            ? std::string() : img.filename.substr(start, end - start + 1);

        // Every record is two lines. The second holds the 2D observations,
        // which are not needed here, and is blank for an image with none —
        // so it has to be consumed unconditionally rather than skipped as
        // whitespace by the next nextRecord().
        std::getline(f, line);

        images.push_back(img);
    }
    return images;
}

static SparsePointSet readPointsText(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + path);

    SparsePointSet pts;
    std::string line;
    while (nextRecord(f, line)) {
        std::istringstream ls(line);
        uint64_t pointId;
        double x, y, z;
        int r, g, b;
        double error;
        ls >> pointId >> x >> y >> z >> r >> g >> b >> error;
        if (!ls) throw std::runtime_error("Truncated point record in " + path);
        if (r < 0 || r > 255 || g < 0 || g > 255 || b < 0 || b > 255)
            throw std::runtime_error("Invalid point color in " + path);

        pts.xyz.push_back((float)x);
        pts.xyz.push_back((float)y);
        pts.xyz.push_back((float)z);
        pts.rgb.push_back((uint8_t)r);
        pts.rgb.push_back((uint8_t)g);
        pts.rgb.push_back((uint8_t)b);
        pts.sourceIds.push_back(pointId);
        pts.reprojectionErrors.push_back((float)error);
    }
    return pts;
}

// w2c rotation + translation → 4x4 c2w row-major with OpenGL Y/Z flip
static void w2cToCamToWorld(const double quat[4], const double t[3], float out[16]) {
    float R[9];
    quatToRotMat(quat, R);

    // R^T (transpose = inverse for rotation)
    float Ri[9] = { R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8] };

    // -R^T * t
    float Ti[3] = {
        -(Ri[0]*(float)t[0] + Ri[1]*(float)t[1] + Ri[2]*(float)t[2]),
        -(Ri[3]*(float)t[0] + Ri[4]*(float)t[1] + Ri[5]*(float)t[2]),
        -(Ri[6]*(float)t[0] + Ri[7]*(float)t[1] + Ri[8]*(float)t[2])
    };

    // OpenGL flip: negate columns 1,2 (camera Y-down→Y-up, Z-fwd→Z-back)
    out[0]  = Ri[0]; out[1]  = -Ri[1]; out[2]  = -Ri[2]; out[3]  = Ti[0];
    out[4]  = Ri[3]; out[5]  = -Ri[4]; out[6]  = -Ri[5]; out[7]  = Ti[1];
    out[8]  = Ri[6]; out[9]  = -Ri[7]; out[10] = -Ri[8]; out[11] = Ti[2];
    out[12] = 0;     out[13] = 0;      out[14] = 0;      out[15] = 1;
}

DatasetDescriptor loaders::loadColmap(const std::string &projectRoot, const std::string &imageSourcePath) {
    // Find sparse dir — dispatcher already confirmed a cameras file exists
    fs::path root(projectRoot);
    auto hasModel = [](const fs::path &dir) {
        return fs::exists(dir / "cameras.bin") || fs::exists(dir / "cameras.txt");
    };
    fs::path sparse = hasModel(root) ? root : root / "sparse" / "0";

    std::string imageDir = !imageSourcePath.empty() ? imageSourcePath
        : fs::exists(root / "images") ? (root / "images").string()
        : projectRoot;

    // Chosen per file rather than once for the model: COLMAP writes all three
    // in the same format, but a dataset assembled by hand often mixes them.
    auto cameras = fs::exists(sparse / "cameras.bin")
        ? readCamerasBin((sparse / "cameras.bin").string())
        : readCamerasText((sparse / "cameras.txt").string());
    auto images = fs::exists(sparse / "images.bin")
        ? readImagesBin((sparse / "images.bin").string())
        : readImagesText((sparse / "images.txt").string());

    std::sort(images.begin(), images.end(),
        [](const ColmapImage &a, const ColmapImage &b) { return a.filename < b.filename; });

    DatasetDescriptor data;
    data.provenance.adapter = "colmap";
    data.provenance.source = projectRoot;
    data.frames.reserve(images.size());

    for (auto &img : images) {
        auto it = cameras.find(img.camId);
        if (it == cameras.end()) {
            throw std::runtime_error(
                "COLMAP image " + std::to_string(img.id) +
                " references unknown camera " + std::to_string(img.camId));
        }
        auto &cc = it->second;

        DatasetFrameDescriptor frame;
        frame.id = std::to_string(img.id);
        frame.calibrationId = std::to_string(img.camId);
        frame.imagePath = (fs::path(imageDir) / img.filename).string();
        frame.rasterOrientation = RasterOrientation::EncodedPixels;
        frame.calibration.width = cc.width;
        frame.calibration.height = cc.height;
        frame.calibration.fx = cc.fx;
        frame.calibration.fy = cc.fy;
        frame.calibration.cx = cc.cx;
        frame.calibration.cy = cc.cy;
        frame.calibration.k1 = cc.k1;
        frame.calibration.k2 = cc.k2;
        frame.calibration.p1 = cc.p1;
        frame.calibration.p2 = cc.p2;
        w2cToCamToWorld(img.quat, img.t, frame.cameraToWorld.data());
        data.frames.push_back(std::move(frame));
    }

    // Point cloud
    if (fs::exists(sparse / "points3D.bin"))
        data.points = readColmapPoints((sparse / "points3D.bin").string());
    else if (fs::exists(sparse / "points3D.txt"))
        data.points = readPointsText((sparse / "points3D.txt").string());
    else if (fs::exists(sparse / "points3D.ply"))
        data.points = readPly((sparse / "points3D.ply").string());

    return data;
}
