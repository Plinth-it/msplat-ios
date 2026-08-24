#include "dataset_descriptor.hpp"
#include "loaders.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

namespace {

namespace fs = std::filesystem;

constexpr const char *kTextCameras =
    "7 PINHOLE 640 480 500 501 320 240\n";

// Source image order is 100, 200 while filename order is a.jpg, z.jpg. This
// makes every observation-to-frame assertion exercise the source-ID mapping.
constexpr const char *kTextImages =
    "100 1 0 0 0 0 0 0 7 z.jpg\n"
    "11 12 900 13 14 -1\n"
    "200 1 0 0 0 0 0 0 7 a.jpg\n"
    "21 22 800 23 24 900\n";

constexpr const char *kTextPoints =
    "900 1 2 3 255 0 0 0.25 100 0 200 1\n"
    "800 -1 0 2 0 255 0 0.5 200 0\n";

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        std::string pattern =
            (fs::temp_directory_path() / "msplat-colmap-test-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        const char *created = mkdtemp(writable.data());
        if (!created)
            throw std::runtime_error("could not create temporary directory");
        path = created;
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

bool writeText(const fs::path &path, const std::string &contents) {
    std::ofstream stream(path);
    stream << contents;
    return static_cast<bool>(stream);
}

bool writeTextModel(const fs::path &root, const std::string &cameras,
                    const std::string &images, const std::string &points) {
    return writeText(root / "cameras.txt", cameras) &&
           writeText(root / "images.txt", images) &&
           writeText(root / "points3D.txt", points);
}

bool writeFileCreatingParents(const fs::path &path,
                              const std::string &contents = "mask") {
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    return !error && writeText(path, contents);
}

const DatasetFrameDescriptor *frameWithId(
    const DatasetDescriptor &descriptor, const std::string &id) {
    for (const DatasetFrameDescriptor &frame : descriptor.frames) {
        if (frame.id == id) return &frame;
    }
    return nullptr;
}

bool maskEquals(const DatasetFrameDescriptor *frame, const fs::path &path) {
    return frame && frame->trainingMask &&
           frame->trainingMask->path == path.string() &&
           frame->trainingMask->channel == TrainingMaskChannel::Automatic;
}

bool checkTrainingMaskDiscoveryIsOptInAndPartial() {
    TempDirectory temporary;
    const fs::path maskPath =
        temporary.path / "assets" / "MaSkS" / "images" / "A.PnG";
    std::error_code directoryError;
    fs::create_directories(temporary.path / "images", directoryError);
    if (!writeTextModel(temporary.path, kTextCameras, kTextImages, kTextPoints) ||
        directoryError ||
        !writeFileCreatingParents(maskPath)) {
        return false;
    }

    try {
        const DatasetDescriptor disabled =
            loaders::loadColmap(temporary.path.string());
        if (disabled.frames.size() != 2 ||
            disabled.frames[0].trainingMask ||
            disabled.frames[1].trainingMask) {
            return false;
        }

        const DatasetDescriptor enabled = datasetDescriptorFromX(
            temporary.path.string(), "", true);
        validateDatasetDescriptor(enabled);
        return maskEquals(frameWithId(enabled, "200"), maskPath) &&
               frameWithId(enabled, "100") &&
               !frameWithId(enabled, "100")->trainingMask;
    } catch (...) {
        return false;
    }
}

bool checkTrainingMaskFormatsAndDeterminism() {
    TempDirectory temporary;
    if (!writeTextModel(temporary.path, kTextCameras, kTextImages, kTextPoints)) {
        return false;
    }

    const fs::path sameStemA = temporary.path / "masks" / "a.png";
    const fs::path sameStemB = temporary.path / "masks" / "A.webp";
    const fs::path fullFilename = temporary.path / "masks" / "z.jpg.mask";
    const fs::path maskSuffix = temporary.path / "masks" / "z.mask.tiff";
    if (!writeFileCreatingParents(sameStemA) ||
        !writeFileCreatingParents(sameStemB) ||
        !writeFileCreatingParents(fullFilename) ||
        !writeFileCreatingParents(maskSuffix)) {
        return false;
    }

    const fs::path expectedA = sameStemA.generic_string() <
            sameStemB.generic_string() ? sameStemA : sameStemB;
    const fs::path expectedZ = fullFilename.generic_string() <
            maskSuffix.generic_string() ? fullFilename : maskSuffix;

    try {
        const DatasetDescriptor first = loaders::loadColmap(
            temporary.path.string(), "", true);
        const DatasetDescriptor second = loaders::loadColmap(
            temporary.path.string(), "", true);
        if (!maskEquals(frameWithId(first, "200"), expectedA) ||
            !maskEquals(frameWithId(first, "100"), expectedZ) ||
            !maskEquals(frameWithId(second, "200"), expectedA) ||
            !maskEquals(frameWithId(second, "100"), expectedZ)) {
            return false;
        }

        std::error_code error;
        fs::remove(fullFilename, error);
        if (error) return false;
        const DatasetDescriptor suffixOnly = loaders::loadColmap(
            temporary.path.string(), "", true);
        return maskEquals(frameWithId(suffixOnly, "100"), maskSuffix);
    } catch (...) {
        return false;
    }
}

bool checkTrainingMaskNestedSuffixMatching() {
    TempDirectory temporary;
    const std::string images =
        "100 1 0 0 0 0 0 0 7 capture/nested/frame.JPG\n\n"
        "200 1 0 0 0 0 0 0 7 unmatched/other.jpeg\n\n";
    const std::string points = "900 1 2 3 255 0 0 0.25\n";
    const fs::path nestedMask =
        temporary.path / "masks" / "nested" / "FRAME.custom";
    const fs::path directoryOnly =
        temporary.path / "masks" / "unmatched" / "other.png";
    if (!writeTextModel(temporary.path, kTextCameras, images, points) ||
        !writeFileCreatingParents(nestedMask)) {
        return false;
    }
    std::error_code error;
    fs::create_directories(directoryOnly, error);
    if (error) return false;

    try {
        const DatasetDescriptor descriptor = loaders::loadColmap(
            temporary.path.string(), "", true);
        validateDatasetDescriptor(descriptor);
        return maskEquals(frameWithId(descriptor, "100"), nestedMask) &&
               frameWithId(descriptor, "200") &&
               !frameWithId(descriptor, "200")->trainingMask;
    } catch (...) {
        return false;
    }
}

bool writePointPly(const fs::path &path) {
    return writeText(path,
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
        "1 2 3 255 0 0\n");
}

bool rejectsRuntimeError(const fs::path &root) {
    try {
        (void)loaders::loadColmap(root.string());
    } catch (const std::runtime_error &) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

bool rejectedTextModel(const std::string &cameras,
                       const std::string &images,
                       const std::string &points) {
    TempDirectory temporary;
    return writeTextModel(temporary.path, cameras, images, points) &&
           rejectsRuntimeError(temporary.path);
}

const SparseObservation *findObservation(const DatasetDescriptor &descriptor,
                                         uint32_t frameIndex,
                                         uint32_t frameObservationIndex) {
    for (const SparseObservation &observation : descriptor.observations) {
        if (observation.frameIndex == frameIndex &&
            observation.frameObservationIndex == frameObservationIndex) {
            return &observation;
        }
    }
    return nullptr;
}

bool observationEquals(const DatasetDescriptor &descriptor,
                       uint32_t frameIndex, uint32_t frameObservationIndex,
                       int32_t pointIndex, float x, float y) {
    const SparseObservation *observation =
        findObservation(descriptor, frameIndex, frameObservationIndex);
    return observation && observation->pointIndex == pointIndex &&
           observation->x == x && observation->y == y;
}

bool hasExpectedModel(const DatasetDescriptor &descriptor) {
    if (descriptor.provenance.adapter != "colmap" ||
        descriptor.frames.size() != 2 || descriptor.points.count() != 2 ||
        descriptor.observations.size() != 4 ||
        descriptor.frames[0].id != "200" ||
        descriptor.frames[1].id != "100" ||
        descriptor.frames[0].calibrationId != "7" ||
        descriptor.frames[1].calibrationId != "7" ||
        fs::path(descriptor.frames[0].imagePath).filename() != "a.jpg" ||
        fs::path(descriptor.frames[1].imagePath).filename() != "z.jpg" ||
        descriptor.points.sourceIds != std::vector<uint64_t>({900, 800}) ||
        descriptor.points.reprojectionErrors !=
            std::vector<float>({0.25f, 0.5f})) {
        return false;
    }

    // Point indices follow points3D record order, not the numeric source ID.
    return observationEquals(descriptor, 0, 0, 1, 21.0f, 22.0f) &&
           observationEquals(descriptor, 0, 1, 0, 23.0f, 24.0f) &&
           observationEquals(descriptor, 1, 0, 0, 11.0f, 12.0f) &&
           observationEquals(descriptor, 1, 1, -1, 13.0f, 14.0f);
}

bool checkTextHappyPath() {
    TempDirectory temporary;
    if (!writeTextModel(temporary.path, kTextCameras, kTextImages,
                        kTextPoints)) {
        return false;
    }

    try {
        const DatasetDescriptor descriptor =
            loaders::loadColmap(temporary.path.string());
        validateDatasetDescriptor(descriptor);
        return hasExpectedModel(descriptor);
    } catch (...) {
        return false;
    }
}

bool checkTextSemanticRejections() {
    const std::string unknownPointImages =
        "100 1 0 0 0 0 0 0 7 z.jpg\n"
        "11 12 999\n";
    if (!rejectedTextModel(kTextCameras, unknownPointImages,
                           "900 1 2 3 255 0 0 0.25\n")) {
        return false;
    }

    const std::string noObservationImage =
        "100 1 0 0 0 0 0 0 7 z.jpg\n\n";
    if (!rejectedTextModel(kTextCameras, noObservationImage,
            "900 1 2 3 255 0 0 0.25 999 0\n")) {
        return false;
    }
    if (!rejectedTextModel(kTextCameras, noObservationImage,
            "900 1 2 3 255 0 0 0.25 100 1\n")) {
        return false;
    }

    const std::string mismatchedImages =
        "100 1 0 0 0 0 0 0 7 z.jpg\n"
        "11 12 800 13 14 900\n";
    if (!rejectedTextModel(kTextCameras, mismatchedImages,
            "900 1 2 3 255 0 0 0.25 100 0\n"
            "800 -1 0 2 0 255 0 0.5 100 1\n")) {
        return false;
    }

    const std::string linkedImage =
        "100 1 0 0 0 0 0 0 7 z.jpg\n"
        "11 12 900\n";
    if (!rejectedTextModel(kTextCameras, linkedImage,
            "900 1 2 3 255 0 0 0.25\n")) {
        return false;
    }
    if (!rejectedTextModel(kTextCameras, linkedImage,
            "900 1 2 3 255 0 0 0.25 100 0 100 0\n")) {
        return false;
    }
    return true;
}

bool checkTextDuplicateIdRejections() {
    const std::string noObservationImage =
        "100 1 0 0 0 0 0 0 7 z.jpg\n\n";
    const std::string untrackedPoint =
        "900 1 2 3 255 0 0 0.25\n";

    if (!rejectedTextModel(
            "7 PINHOLE 640 480 500 501 320 240\n"
            "7 PINHOLE 640 480 500 501 320 240\n",
            noObservationImage, untrackedPoint)) {
        return false;
    }
    if (!rejectedTextModel(kTextCameras,
            "100 1 0 0 0 0 0 0 7 z.jpg\n\n"
            "100 1 0 0 0 0 0 0 7 a.jpg\n\n",
            untrackedPoint)) {
        return false;
    }
    if (!rejectedTextModel(kTextCameras, noObservationImage,
            "900 1 2 3 255 0 0 0.25\n"
            "900 -1 0 2 0 255 0 0.5\n")) {
        return false;
    }
    return true;
}

bool checkTextTruncationRejections() {
    const std::string truncatedTriples =
        "100 1 0 0 0 0 0 0 7 z.jpg\n"
        "11 12 900 13 14\n";
    if (!rejectedTextModel(kTextCameras, truncatedTriples,
                           "900 1 2 3 255 0 0 0.25\n")) {
        return false;
    }

    const std::string noObservationImage =
        "100 1 0 0 0 0 0 0 7 z.jpg\n\n";
    if (!rejectedTextModel(kTextCameras, noObservationImage,
            "900 1 2 3 255 0 0 0.25 100\n")) {
        return false;
    }

    const std::string ambiguousSentinelImage =
        "100 1 0 0 0 0 0 0 7 z.jpg\n"
        "11 12 18446744073709551615\n";
    return rejectedTextModel(kTextCameras, ambiguousSentinelImage,
                             "900 1 2 3 255 0 0 0.25\n");
}

bool checkOversizedTextRecordRejected() {
    TempDirectory temporary;
    std::string oversizedRecord(16 * 1024 * 1024 + 1, 'x');
    return writeText(temporary.path / "cameras.txt", oversizedRecord) &&
           rejectsRuntimeError(temporary.path);
}

bool checkPlyLinkedObservationRejected() {
    TempDirectory temporary;
    const std::string linkedImage =
        "100 1 0 0 0 0 0 0 7 z.jpg\n"
        "11 12 900\n";
    return writeText(temporary.path / "cameras.txt", kTextCameras) &&
           writeText(temporary.path / "images.txt", linkedImage) &&
           writePointPly(temporary.path / "points3D.ply") &&
           rejectsRuntimeError(temporary.path);
}

template <typename T>
void writePod(std::ofstream &stream, const T &value) {
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

struct BinaryCamera {
    uint32_t id = 7;
    uint64_t width = 640;
    uint64_t height = 480;
};

struct BinaryObservation {
    double x;
    double y;
    int64_t pointId;
};

struct BinaryImage {
    uint32_t id;
    uint32_t cameraId;
    std::string filename;
    std::vector<BinaryObservation> observations;
};

struct BinaryTrack {
    uint32_t imageId;
    uint32_t point2DIndex;
};

struct BinaryPoint {
    uint64_t id;
    double x;
    double y;
    double z;
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    double error;
    std::vector<BinaryTrack> tracks;
};

std::vector<BinaryCamera> baselineCameras() {
    return {{7, 640, 480}};
}

std::vector<BinaryImage> baselineImages() {
    return {
        {100, 7, "z.jpg", {{11.0, 12.0, 900}, {13.0, 14.0, -1}}},
        {200, 7, "a.jpg", {{21.0, 22.0, 800}, {23.0, 24.0, 900}}},
    };
}

std::vector<BinaryPoint> baselinePoints() {
    return {
        {900, 1.0, 2.0, 3.0, 255, 0, 0, 0.25,
         {{100, 0}, {200, 1}}},
        {800, -1.0, 0.0, 2.0, 0, 255, 0, 0.5, {{200, 0}}},
    };
}

bool writeBinaryCameras(const fs::path &path,
                        const std::vector<BinaryCamera> &cameras) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    writePod(stream, static_cast<uint64_t>(cameras.size()));
    for (const BinaryCamera &camera : cameras) {
        writePod(stream, camera.id);
        writePod(stream, uint32_t{1}); // PINHOLE
        writePod(stream, camera.width);
        writePod(stream, camera.height);
        for (double parameter : {500.0, 501.0, 320.0, 240.0})
            writePod(stream, parameter);
    }
    return static_cast<bool>(stream);
}

void writeBinaryImageHeader(std::ofstream &stream, const BinaryImage &image,
                            bool terminateFilename) {
    writePod(stream, image.id);
    for (double value : {1.0, 0.0, 0.0, 0.0}) writePod(stream, value);
    for (double value : {0.0, 0.0, 0.0}) writePod(stream, value);
    writePod(stream, image.cameraId);
    stream.write(image.filename.data(),
                 static_cast<std::streamsize>(image.filename.size()));
    if (terminateFilename) writePod(stream, char{0});
}

bool writeBinaryImages(const fs::path &path,
                       const std::vector<BinaryImage> &images) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    writePod(stream, static_cast<uint64_t>(images.size()));
    for (const BinaryImage &image : images) {
        writeBinaryImageHeader(stream, image, true);
        writePod(stream, static_cast<uint64_t>(image.observations.size()));
        for (const BinaryObservation &observation : image.observations) {
            writePod(stream, observation.x);
            writePod(stream, observation.y);
            writePod(stream, observation.pointId);
        }
    }
    return static_cast<bool>(stream);
}

void writeBinaryPointRecord(std::ofstream &stream, const BinaryPoint &point,
                            bool includeTracks = true) {
    writePod(stream, point.id);
    writePod(stream, point.x);
    writePod(stream, point.y);
    writePod(stream, point.z);
    writePod(stream, point.red);
    writePod(stream, point.green);
    writePod(stream, point.blue);
    writePod(stream, point.error);
    if (!includeTracks) return;
    writePod(stream, static_cast<uint64_t>(point.tracks.size()));
    for (const BinaryTrack &track : point.tracks) {
        writePod(stream, track.imageId);
        writePod(stream, track.point2DIndex);
    }
}

bool writeBinaryPoints(const fs::path &path,
                       const std::vector<BinaryPoint> &points) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    writePod(stream, static_cast<uint64_t>(points.size()));
    for (const BinaryPoint &point : points)
        writeBinaryPointRecord(stream, point);
    return static_cast<bool>(stream);
}

bool writeBaselineBinaryModel(const fs::path &root) {
    return writeBinaryCameras(root / "cameras.bin", baselineCameras()) &&
           writeBinaryImages(root / "images.bin", baselineImages()) &&
           writeBinaryPoints(root / "points3D.bin", baselinePoints());
}

template <typename Mutation>
bool rejectedBinaryMutation(Mutation mutation) {
    TempDirectory temporary;
    if (!writeBaselineBinaryModel(temporary.path) ||
        !mutation(temporary.path)) {
        return false;
    }
    return rejectsRuntimeError(temporary.path);
}

bool checkBinaryHappyPath() {
    TempDirectory temporary;
    if (!writeBaselineBinaryModel(temporary.path)) return false;

    try {
        const DatasetDescriptor descriptor =
            loaders::loadColmap(temporary.path.string());
        validateDatasetDescriptor(descriptor);
        return hasExpectedModel(descriptor);
    } catch (...) {
        return false;
    }
}

bool checkBinaryDuplicateIdRejections() {
    if (!rejectedBinaryMutation([](const fs::path &root) {
            auto cameras = baselineCameras();
            cameras.push_back(cameras.front());
            return writeBinaryCameras(root / "cameras.bin", cameras);
        })) {
        return false;
    }
    if (!rejectedBinaryMutation([](const fs::path &root) {
            auto images = baselineImages();
            images.push_back(images.front());
            return writeBinaryImages(root / "images.bin", images);
        })) {
        return false;
    }
    return rejectedBinaryMutation([](const fs::path &root) {
        auto points = baselinePoints();
        points.push_back(points.front());
        return writeBinaryPoints(root / "points3D.bin", points);
    });
}

bool checkBinaryTruncationRejections() {
    if (!rejectedBinaryMutation([](const fs::path &root) {
            std::ofstream stream(root / "cameras.bin",
                                 std::ios::binary | std::ios::trunc);
            writePod(stream, uint64_t{1});
            writePod(stream, uint32_t{7});
            return static_cast<bool>(stream);
        })) {
        return false;
    }
    if (!rejectedBinaryMutation([](const fs::path &root) {
            std::ofstream stream(root / "images.bin",
                                 std::ios::binary | std::ios::trunc);
            writePod(stream, uint64_t{1});
            writePod(stream, uint32_t{100});
            return static_cast<bool>(stream);
        })) {
        return false;
    }
    return rejectedBinaryMutation([](const fs::path &root) {
        std::ofstream stream(root / "points3D.bin",
                             std::ios::binary | std::ios::trunc);
        writePod(stream, uint64_t{1});
        writePod(stream, uint64_t{900});
        return static_cast<bool>(stream);
    });
}

bool checkEveryBinaryPrefixRejected() {
    TempDirectory temporary;
    for (const char *filename : {
             "cameras.bin", "images.bin", "points3D.bin"}) {
        if (!writeBaselineBinaryModel(temporary.path)) return false;

        const fs::path path = temporary.path / filename;
        std::ifstream source(path, std::ios::binary);
        const std::vector<char> bytes{
            std::istreambuf_iterator<char>(source),
            std::istreambuf_iterator<char>()};
        if (!source.is_open() || bytes.empty()) return false;

        for (size_t prefixLength = 0; prefixLength < bytes.size();
             ++prefixLength) {
            std::ofstream prefix(path, std::ios::binary | std::ios::trunc);
            prefix.write(bytes.data(),
                         static_cast<std::streamsize>(prefixLength));
            prefix.close();
            if (!prefix || !rejectsRuntimeError(temporary.path)) return false;
        }
    }
    return true;
}

bool checkBinaryMaliciousCountRejections() {
    constexpr uint64_t malicious = std::numeric_limits<uint64_t>::max();

    if (!rejectedBinaryMutation([](const fs::path &root) {
            std::ofstream stream(root / "cameras.bin",
                                 std::ios::binary | std::ios::trunc);
            writePod(stream, std::numeric_limits<uint64_t>::max());
            return static_cast<bool>(stream);
        })) {
        return false;
    }
    if (!rejectedBinaryMutation([](const fs::path &root) {
            std::ofstream stream(root / "images.bin",
                                 std::ios::binary | std::ios::trunc);
            writePod(stream, std::numeric_limits<uint64_t>::max());
            return static_cast<bool>(stream);
        })) {
        return false;
    }
    if (!rejectedBinaryMutation([](const fs::path &root) {
            std::ofstream stream(root / "points3D.bin",
                                 std::ios::binary | std::ios::trunc);
            writePod(stream, std::numeric_limits<uint64_t>::max());
            return static_cast<bool>(stream);
        })) {
        return false;
    }
    if (!rejectedBinaryMutation([malicious](const fs::path &root) {
            std::ofstream stream(root / "images.bin",
                                 std::ios::binary | std::ios::trunc);
            writePod(stream, uint64_t{1});
            const BinaryImage image{100, 7, "z.jpg", {}};
            writeBinaryImageHeader(stream, image, true);
            writePod(stream, malicious);
            return static_cast<bool>(stream);
        })) {
        return false;
    }
    return rejectedBinaryMutation([malicious](const fs::path &root) {
        std::ofstream stream(root / "points3D.bin",
                             std::ios::binary | std::ios::trunc);
        writePod(stream, uint64_t{1});
        writeBinaryPointRecord(stream, baselinePoints().front(), false);
        writePod(stream, malicious);
        return static_cast<bool>(stream);
    });
}

bool checkBinaryUnterminatedFilenameRejected() {
    return rejectedBinaryMutation([](const fs::path &root) {
        std::ofstream stream(root / "images.bin",
                             std::ios::binary | std::ios::trunc);
        writePod(stream, uint64_t{1});
        const BinaryImage image{100, 7, "unterminated.jpg", {}};
        writeBinaryImageHeader(stream, image, false);
        return static_cast<bool>(stream);
    });
}

bool checkBinaryOversizedDimensionsRejected() {
    return rejectedBinaryMutation([](const fs::path &root) {
        auto cameras = baselineCameras();
        cameras.front().width =
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1;
        return writeBinaryCameras(root / "cameras.bin", cameras);
    });
}

} // namespace

int main() {
    CHECK(checkTextHappyPath());
    CHECK(checkBinaryHappyPath());
    CHECK(checkTrainingMaskDiscoveryIsOptInAndPartial());
    CHECK(checkTrainingMaskFormatsAndDeterminism());
    CHECK(checkTrainingMaskNestedSuffixMatching());
    CHECK(checkTextSemanticRejections());
    CHECK(checkTextDuplicateIdRejections());
    CHECK(checkTextTruncationRejections());
    CHECK(checkOversizedTextRecordRejected());
    CHECK(checkPlyLinkedObservationRejected());
    CHECK(checkBinaryDuplicateIdRejections());
    CHECK(checkBinaryTruncationRejections());
    CHECK(checkEveryBinaryPrefixRejected());
    CHECK(checkBinaryMaliciousCountRejections());
    CHECK(checkBinaryUnterminatedFilenameRejected());
    CHECK(checkBinaryOversizedDimensionsRejected());
    return 0;
}
