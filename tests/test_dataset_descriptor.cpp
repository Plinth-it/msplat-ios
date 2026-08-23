#include "dataset_descriptor.hpp"
#include "input_data.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

namespace {

namespace fs = std::filesystem;

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        std::string pattern =
            (fs::temp_directory_path() / "msplat-descriptor-test-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        const char *created = mkdtemp(writable.data());
        if (!created) throw std::runtime_error("could not create temporary directory");
        path = created;
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

DatasetDescriptor validDescriptor() {
    DatasetDescriptor descriptor;

    DatasetFrameDescriptor first;
    first.id = "frame-0";
    first.calibrationId = "camera-0";
    first.imagePath = "images/0.jpg";
    first.calibration = {640, 480, 500.0f, 501.0f, 320.0f, 240.0f,
                         0.01f, -0.02f, 0.001f, 0.002f, -0.003f};

    DatasetFrameDescriptor second;
    second.id = "frame-1";
    second.calibrationId = "camera-1";
    second.imagePath = "images/1.jpg";
    second.rasterOrientation = RasterOrientation::ExifNormalized;
    second.calibration = {480, 640, 501.0f, 500.0f, 240.0f, 320.0f};
    second.cameraToWorld = {
        0.0f, -1.0f, 0.0f, 1.0f,
        1.0f,  0.0f, 0.0f, 2.0f,
        0.0f,  0.0f, 1.0f, 3.0f,
        0.0f,  0.0f, 0.0f, 1.0f,
    };

    descriptor.frames = {first, second};
    descriptor.points.xyz = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 2.0f};
    descriptor.points.rgb = {255, 0, 0, 0, 255, 0};
    descriptor.points.sourceIds = {42, 84};
    descriptor.points.reprojectionErrors = {0.25f, 0.0f};
    // Pixel coordinates are deliberately outside the calibrated canvas. The
    // descriptor preserves source observations without clipping them.
    descriptor.observations = {
        {0, 0, -20.0f, 900.0f},
        {1, -1, 12.0f, 18.0f},
    };
    descriptor.provenance = {"test", "synthetic"};
    return descriptor;
}

bool accepts(const DatasetDescriptor &descriptor) {
    try {
        validateDatasetDescriptor(descriptor);
    } catch (...) {
        return false;
    }
    return true;
}

bool rejectsMaterialization(DatasetDescriptor descriptor) {
    try {
        (void)inputDataFromDescriptor(std::move(descriptor));
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

bool writePointPly(const fs::path &path) {
    std::ofstream points(path);
    points << "ply\n"
           << "format ascii 1.0\n"
           << "element vertex 2\n"
           << "property float x\n"
           << "property float y\n"
           << "property float z\n"
           << "property uchar red\n"
           << "property uchar green\n"
           << "property uchar blue\n"
           << "end_header\n"
           << "1 2 3 255 0 0\n"
           << "-1 0 2 0 255 0\n";
    return static_cast<bool>(points);
}

template <typename T>
void writeBinary(std::ofstream &stream, T value) {
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void writeColmapBinaryImage(std::ofstream &stream, uint32_t id,
                            uint32_t cameraId, const std::string &filename) {
    writeBinary(stream, id);
    for (double value : {1.0, 0.0, 0.0, 0.0}) writeBinary(stream, value);
    for (double value : {0.0, 0.0, 0.0}) writeBinary(stream, value);
    writeBinary(stream, cameraId);
    stream.write(filename.c_str(),
                 static_cast<std::streamsize>(filename.size() + 1));
    writeBinary(stream, uint64_t{0});
}

void writeColmapBinaryPoint(std::ofstream &stream, uint64_t id,
                            double x, double y, double z,
                            uint8_t red, uint8_t green, uint8_t blue,
                            double error) {
    writeBinary(stream, id);
    writeBinary(stream, x);
    writeBinary(stream, y);
    writeBinary(stream, z);
    writeBinary(stream, red);
    writeBinary(stream, green);
    writeBinary(stream, blue);
    writeBinary(stream, error);
    writeBinary(stream, uint64_t{0});
}

bool hasExpectedColmapMetadata(const DatasetDescriptor &descriptor) {
    return descriptor.provenance.adapter == "colmap" &&
           descriptor.frames.size() == 2 &&
           descriptor.frames[0].id == "10" &&
           descriptor.frames[1].id == "20" &&
           descriptor.frames[0].calibrationId == "7" &&
           descriptor.frames[1].calibrationId == "7" &&
           fs::path(descriptor.frames[0].imagePath).filename() == "a.jpg" &&
           fs::path(descriptor.frames[1].imagePath).filename() == "b.jpg" &&
           descriptor.points.sourceIds == std::vector<uint64_t>({900, 800}) &&
           descriptor.points.reprojectionErrors ==
               std::vector<float>({0.25f, 0.5f});
}

bool checkColmapTextAdapter() {
    TempDirectory temporary;
    {
        std::ofstream cameras(temporary.path / "cameras.txt");
        cameras << "7 PINHOLE 640 480 500 501 320 240\n";
        if (!cameras) return false;
    }
    {
        std::ofstream images(temporary.path / "images.txt");
        images << "20 1 0 0 0 0 0 0 7 b.jpg\n\n"
               << "10 1 0 0 0 0 0 0 7 a.jpg\n\n";
        if (!images) return false;
    }
    {
        std::ofstream points(temporary.path / "points3D.txt");
        points << "900 1 2 3 255 0 0 0.25\n"
               << "800 -1 0 2 0 255 0 0.5\n";
        if (!points) return false;
    }

    return hasExpectedColmapMetadata(
        datasetDescriptorFromX(temporary.path.string()));
}

bool checkColmapBinaryAdapter() {
    TempDirectory temporary;
    {
        std::ofstream cameras(temporary.path / "cameras.bin", std::ios::binary);
        writeBinary(cameras, uint64_t{1});
        writeBinary(cameras, uint32_t{7});
        writeBinary(cameras, uint32_t{1}); // PINHOLE
        writeBinary(cameras, uint64_t{640});
        writeBinary(cameras, uint64_t{480});
        for (double value : {500.0, 501.0, 320.0, 240.0})
            writeBinary(cameras, value);
        if (!cameras) return false;
    }
    {
        std::ofstream images(temporary.path / "images.bin", std::ios::binary);
        writeBinary(images, uint64_t{2});
        writeColmapBinaryImage(images, 20, 7, "b.jpg");
        writeColmapBinaryImage(images, 10, 7, "a.jpg");
        if (!images) return false;
    }
    {
        std::ofstream points(temporary.path / "points3D.bin", std::ios::binary);
        writeBinary(points, uint64_t{2});
        writeColmapBinaryPoint(points, 900, 1.0, 2.0, 3.0,
                               255, 0, 0, 0.25);
        writeColmapBinaryPoint(points, 800, -1.0, 0.0, 2.0,
                               0, 255, 0, 0.5);
        if (!points) return false;
    }

    return hasExpectedColmapMetadata(
        datasetDescriptorFromX(temporary.path.string()));
}

bool checkNerfstudioAdapter() {
    TempDirectory temporary;
    {
        std::ofstream transforms(temporary.path / "transforms.json");
        transforms << R"({
  "w": 4,
  "h": 3,
  "fl_x": 2.0,
  "fl_y": 2.5,
  "cx": 2.0,
  "cy": 1.5,
  "ply_file_path": "points3D.ply",
  "frames": [
    {"file_path": "./images/b.png", "transform_matrix":
      [[1,0,0,2], [0,1,0,0], [0,0,1,0], [0,0,0,1]]},
    {"file_path": "./images/a.png", "transform_matrix":
      [[1,0,0,-2], [0,1,0,0], [0,0,1,0], [0,0,0,1]]}
  ]
})";
        if (!transforms) return false;
    }
    if (!writePointPly(temporary.path / "points3D.ply")) return false;

    DatasetDescriptor descriptor = datasetDescriptorFromX(temporary.path.string());
    if (descriptor.provenance.adapter != "nerfstudio" ||
        descriptor.frames.size() != 2 || descriptor.points.count() != 2 ||
        descriptor.frames[0].id != "./images/a.png" ||
        descriptor.frames[1].id != "./images/b.png" ||
        descriptor.frames[0].imagePath !=
            (temporary.path / "images/a.png").string()) {
        return false;
    }

    InputData materialized = inputDataFromX(temporary.path.string());
    return materialized.metadata.frameIds ==
               std::vector<std::string>({"./images/a.png", "./images/b.png"}) &&
           materialized.metadata.provenance.adapter == "nerfstudio" &&
           materialized.points.count == 2 &&
           std::abs(materialized.scale - 0.5f) < 1e-6f;
}

bool checkPolycamLayout1Adapter() {
    TempDirectory temporary;
    const fs::path cameraDirectory =
        temporary.path / "keyframes/corrected_cameras";
    const fs::path imageDirectory =
        temporary.path / "keyframes/corrected_images";
    fs::create_directories(cameraDirectory);
    fs::create_directories(imageDirectory);

    const auto writeCamera = [&](const std::string &id, float translation) {
        std::ofstream camera(cameraDirectory / (id + ".json"));
        camera << "{\"width\":4,\"height\":3,\"fx\":2,\"fy\":2.5,"
               << "\"cx\":2,\"cy\":1.5,"
               << "\"R_00\":1,\"R_11\":1,\"R_22\":1,"
               << "\"t_20\":" << translation << "}";
        return static_cast<bool>(camera);
    };
    if (!writeCamera("b", 1.0f) || !writeCamera("a", -1.0f)) return false;
    std::ofstream(imageDirectory / "a.jpg").close();
    std::ofstream(imageDirectory / "b.png").close();
    if (!writePointPly(temporary.path / "keyframes/point_cloud.ply"))
        return false;

    const DatasetDescriptor descriptor =
        datasetDescriptorFromX(temporary.path.string());
    return descriptor.provenance.adapter == "polycam" &&
           descriptor.frames.size() == 2 &&
           descriptor.frames[0].id == "a" &&
           descriptor.frames[1].id == "b" &&
           descriptor.frames[0].calibrationId == "a" &&
           descriptor.frames[1].calibrationId == "b" &&
           descriptor.frames[0].imagePath ==
               (imageDirectory / "a.jpg").string() &&
           descriptor.frames[1].imagePath ==
               (imageDirectory / "b.png").string() &&
           descriptor.points.count() == 2;
}

bool checkPolycamLayout2Adapter() {
    TempDirectory temporary;
    {
        std::ofstream cameras(temporary.path / "cameras.json");
        cameras << R"({"frames":[
  {"file_path":"images/z.png","width":4,"height":3,"fx":2,"fy":2.5,
   "transform_matrix":[[1,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
  {"file_path":"images/a.png","width":4,"height":3,"fx":2,"fy":2.5,
   "transform_matrix":[[1,0,0,-1],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
]})";
        if (!cameras) return false;
    }
    if (!writePointPly(temporary.path / "point_cloud.ply")) return false;

    const DatasetDescriptor descriptor =
        datasetDescriptorFromX(temporary.path.string());
    return descriptor.provenance.adapter == "polycam" &&
           descriptor.frames.size() == 2 &&
           descriptor.frames[0].id == "images/z.png" &&
           descriptor.frames[1].id == "images/a.png" &&
           descriptor.frames[0].calibrationId == "images/z.png" &&
           descriptor.frames[1].calibrationId == "images/a.png" &&
           descriptor.frames[0].imagePath ==
               (temporary.path / "images/z.png").string() &&
           descriptor.points.count() == 2;
}

bool checkPolycamMissingTransformRejected() {
    TempDirectory temporary;
    {
        std::ofstream cameras(temporary.path / "cameras.json");
        cameras << R"([{"file_path":"image.png","width":4,"height":3,
                        "fx":2,"fy":2.5}])";
        if (!cameras) return false;
    }

    try {
        (void)datasetDescriptorFromX(temporary.path.string());
    } catch (const std::runtime_error &error) {
        return std::string(error.what()).find("missing transform_matrix") !=
               std::string::npos;
    }
    return false;
}

template <typename Mutation>
bool rejects(const DatasetDescriptor &valid, Mutation mutation) {
    DatasetDescriptor descriptor = valid;
    mutation(descriptor);
    try {
        validateDatasetDescriptor(descriptor);
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    CHECK(checkColmapTextAdapter());
    CHECK(checkColmapBinaryAdapter());
    CHECK(checkNerfstudioAdapter());
    CHECK(checkPolycamLayout1Adapter());
    CHECK(checkPolycamLayout2Adapter());
    CHECK(checkPolycamMissingTransformRejected());

    const DatasetDescriptor valid = validDescriptor();
    CHECK(accepts(valid));
    CHECK(valid.points.count() == 2);
    CHECK(!valid.points.empty());
    CHECK(SparsePointSet{}.count() == 0);
    CHECK(SparsePointSet{}.empty());

    InputData materialized = inputDataFromDescriptor(valid);
    CHECK(materialized.cameras.size() == 2);
    CHECK(materialized.points.count == 2);
    CHECK(materialized.metadata.frameIds ==
          std::vector<std::string>({"frame-0", "frame-1"}));
    CHECK(materialized.metadata.calibrationIds ==
          std::vector<std::string>({"camera-0", "camera-1"}));
    CHECK(materialized.metadata.pointSourceIds ==
          std::vector<uint64_t>({42, 84}));
    CHECK(materialized.metadata.pointReprojectionErrors ==
          std::vector<float>({0.25f, 0.0f}));
    CHECK(materialized.metadata.observations.size() == 2);
    CHECK(materialized.metadata.provenance.adapter == "test");
    CHECK(materialized.metadata.provenance.source == "synthetic");
    CHECK(materialized.cameras[1].rasterOrientation ==
          RasterOrientation::ExifNormalized);
    CHECK(std::abs(materialized.scale - (2.0f / 3.0f)) < 1e-6f);
    CHECK(std::abs(materialized.translation[0] - 0.5f) < 1e-6f);
    CHECK(std::abs(materialized.translation[1] - 1.0f) < 1e-6f);
    CHECK(std::abs(materialized.translation[2] - 1.5f) < 1e-6f);
    CHECK(std::abs(materialized.points.xyz[0] - (1.0f / 3.0f)) < 1e-6f);
    CHECK(std::abs(materialized.points.xyz[1] - (2.0f / 3.0f)) < 1e-6f);
    CHECK(std::abs(materialized.points.xyz[2] - 1.0f) < 1e-6f);

    DatasetDescriptor extremeCoordinates = valid;
    for (auto &frame : extremeCoordinates.frames) {
        frame.cameraToWorld[3] = -std::numeric_limits<float>::max();
        frame.cameraToWorld[7] = 0.0f;
        frame.cameraToWorld[11] = 0.0f;
    }
    extremeCoordinates.points.xyz[0] = std::numeric_limits<float>::max();
    CHECK(rejectsMaterialization(std::move(extremeCoordinates)));

    DatasetDescriptor withoutOptionalPointMetadata = valid;
    withoutOptionalPointMetadata.points.sourceIds.clear();
    withoutOptionalPointMetadata.points.reprojectionErrors.clear();
    CHECK(accepts(withoutOptionalPointMetadata));

    CHECK(rejects(valid, [](auto &value) { value.frames.clear(); }));
    CHECK(rejects(valid, [](auto &value) { value.points.xyz.clear(); }));
    CHECK(rejects(valid, [](auto &value) { value.frames[0].id.clear(); }));
    CHECK(rejects(valid, [](auto &value) {
        value.frames[1].id = value.frames[0].id;
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.frames[0].calibrationId.clear();
    }));
    CHECK(rejects(valid, [](auto &value) { value.frames[0].imagePath.clear(); }));
    CHECK(rejects(valid, [](auto &value) {
        value.frames[0].rasterOrientation = static_cast<RasterOrientation>(2);
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.provenance.adapter.clear();
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.provenance.source.clear();
    }));

    CHECK(rejects(valid, [](auto &value) {
        value.frames[0].calibration.width = 0;
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.frames[0].calibration.height = -1;
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.frames[0].calibration.fx = 0.0f;
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.frames[0].calibration.fy = -1.0f;
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.frames[0].calibration.cx =
            std::numeric_limits<float>::infinity();
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.frames[0].calibration.k3 =
            std::numeric_limits<float>::quiet_NaN();
    }));

    CHECK(rejects(valid, [](auto &value) {
        value.frames[0].cameraToWorld[3] =
            std::numeric_limits<float>::quiet_NaN();
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.frames[0].cameraToWorld[12] = 0.01f;
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.frames[0].cameraToWorld[0] = 2.0f;
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.frames[0].cameraToWorld[0] = -1.0f;
    }));

    CHECK(rejects(valid, [](auto &value) { value.points.xyz.pop_back(); }));
    CHECK(rejects(valid, [](auto &value) {
        value.points.xyz[0] = std::numeric_limits<float>::quiet_NaN();
    }));
    CHECK(rejects(valid, [](auto &value) { value.points.rgb.pop_back(); }));
    CHECK(rejects(valid, [](auto &value) { value.points.sourceIds.pop_back(); }));
    CHECK(rejects(valid, [](auto &value) {
        value.points.sourceIds[1] = value.points.sourceIds[0];
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.points.reprojectionErrors.pop_back();
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.points.reprojectionErrors[0] = -0.1f;
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.points.reprojectionErrors[0] =
            std::numeric_limits<float>::infinity();
    }));

    CHECK(rejects(valid, [](auto &value) {
        value.observations[0].frameIndex = 2;
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.observations[0].pointIndex = -2;
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.observations[0].pointIndex = 2;
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.observations[0].x = std::numeric_limits<float>::quiet_NaN();
    }));
    CHECK(rejects(valid, [](auto &value) {
        value.observations[0].y = std::numeric_limits<float>::infinity();
    }));

    return 0;
}
