#include "loaders.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr uint64_t kInvalidPointId = std::numeric_limits<uint64_t>::max();
constexpr uint64_t kMaxFilenameBytes = 1024 * 1024;
constexpr size_t kMaxTextRecordBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxDescriptorCount =
    static_cast<uint64_t>(std::numeric_limits<int32_t>::max());

enum ColmapModel {
    SIMPLE_PINHOLE = 0,
    PINHOLE = 1,
    SIMPLE_RADIAL = 2,
    RADIAL = 3,
    OPENCV = 4,
};

struct ColmapCamera {
    uint32_t id = 0;
    int32_t model = 0;
    int32_t width = 0;
    int32_t height = 0;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float k1 = 0.0f;
    float k2 = 0.0f;
    float p1 = 0.0f;
    float p2 = 0.0f;
};

struct ColmapImageObservation {
    float x = 0.0f;
    float y = 0.0f;
    uint64_t pointId = kInvalidPointId;
};

struct ColmapImage {
    uint32_t id = 0;
    uint32_t cameraId = 0;
    double quaternion[4] = {};
    double translation[3] = {};
    std::string filename;
    std::vector<ColmapImageObservation> observations;
};

struct ColmapTrackValidation {
    std::unordered_map<uint32_t, size_t> imageIndexById;
    std::vector<std::vector<uint8_t>> markers;
    size_t totalObservationCount = 0;
};

[[noreturn]] void malformed(const std::string &path, const std::string &detail) {
    throw std::runtime_error("Malformed COLMAP file " + path + ": " + detail);
}

size_t checkedSize(uint64_t value, size_t maximum,
                   const std::string &path, const std::string &label) {
    if (value > static_cast<uint64_t>(maximum)) {
        malformed(path, label + " is too large");
    }
    return static_cast<size_t>(value);
}

size_t checkedProduct(size_t lhs, size_t rhs, size_t maximum,
                      const std::string &path, const std::string &label) {
    if (rhs != 0 && lhs > maximum / rhs) {
        malformed(path, label + " size overflows");
    }
    return lhs * rhs;
}

float checkedFloat(double value, const std::string &path,
                   const std::string &label) {
    constexpr double limit = static_cast<double>(std::numeric_limits<float>::max());
    if (!std::isfinite(value) || value < -limit || value > limit) {
        malformed(path, label + " is not a finite float");
    }
    return static_cast<float>(value);
}

int32_t checkedDimension(uint64_t value, const std::string &path,
                         const std::string &label) {
    if (value == 0 || value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        malformed(path, label + " is outside the supported range");
    }
    return static_cast<int32_t>(value);
}

class BinaryReader {
public:
    explicit BinaryReader(const std::string &path) : path_(path), stream_(path, std::ios::binary) {
        if (!stream_.is_open()) {
            throw std::runtime_error("Cannot open " + path);
        }
        stream_.seekg(0, std::ios::end);
        const std::streampos end = stream_.tellg();
        if (end < 0) {
            malformed(path_, "cannot determine file size");
        }
        remaining_ = static_cast<uint64_t>(end);
        stream_.seekg(0, std::ios::beg);
        if (!stream_) {
            malformed(path_, "cannot seek to the beginning");
        }
    }

    uint64_t remaining() const noexcept { return remaining_; }

    template <typename T>
    T read(const std::string &label) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "BinaryReader only supports trivially copyable values");
        T value{};
        readBytes(&value, sizeof(T), label);
        return value;
    }

    std::string readCString(const std::string &label) {
        std::string value;
        value.reserve(static_cast<size_t>(std::min<uint64_t>(remaining_, 256)));
        while (remaining_ != 0) {
            const char byte = read<char>(label);
            if (byte == '\0') {
                return value;
            }
            if (value.size() == kMaxFilenameBytes) {
                malformed(path_, label + " exceeds the supported length");
            }
            value.push_back(byte);
        }
        malformed(path_, label + " is not NUL-terminated");
    }

    void requireRecords(uint64_t count, uint64_t maximum, uint64_t minimumBytes,
                        const std::string &label) const {
        if (count > maximum) {
            malformed(path_, label + " count is outside the supported range");
        }
        if (minimumBytes != 0 && count > remaining_ / minimumBytes) {
            malformed(path_, label + " count exceeds the remaining file size");
        }
    }

    void requireEnd(const std::string &label) const {
        if (remaining_ != 0) {
            malformed(path_, "unexpected trailing bytes after " + label);
        }
    }

private:
    void readBytes(void *destination, size_t byteCount, const std::string &label) {
        if (byteCount > remaining_) {
            malformed(path_, "truncated " + label);
        }
        stream_.read(static_cast<char *>(destination),
                     static_cast<std::streamsize>(byteCount));
        if (!stream_ || static_cast<size_t>(stream_.gcount()) != byteCount) {
            malformed(path_, "truncated " + label);
        }
        remaining_ -= byteCount;
    }

    std::string path_;
    std::ifstream stream_;
    uint64_t remaining_ = 0;
};

uint64_t parseUnsigned(const std::string &token, uint64_t maximum,
                       const std::string &path, const std::string &label) {
    uint64_t value = 0;
    if (token.empty() || token.front() == '-') {
        malformed(path, "invalid " + label);
    }
    const char *begin = token.data();
    const char *end = begin + token.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc() || result.ptr != end || value > maximum) {
        malformed(path, "invalid " + label);
    }
    return value;
}

uint32_t parseUint32(const std::string &token, const std::string &path,
                     const std::string &label) {
    return static_cast<uint32_t>(
        parseUnsigned(token, std::numeric_limits<uint32_t>::max(), path, label));
}

uint64_t parsePointReference(const std::string &token, const std::string &path) {
    if (token == "-1") {
        return kInvalidPointId;
    }
    return parseUnsigned(token, kInvalidPointId - 1, path, "point3D ID");
}

double readFiniteDouble(std::istringstream &stream, const std::string &path,
                        const std::string &label) {
    double value = 0.0;
    if (!(stream >> value) || !std::isfinite(value)) {
        malformed(path, "invalid " + label);
    }
    return value;
}

std::string readToken(std::istringstream &stream, const std::string &path,
                      const std::string &label) {
    std::string token;
    if (!(stream >> token)) {
        malformed(path, "missing " + label);
    }
    return token;
}

void requireNoExtraTokens(std::istringstream &stream, const std::string &path,
                          const std::string &label) {
    std::string extra;
    if (stream >> extra) {
        malformed(path, "unexpected data after " + label);
    }
}

bool readBoundedLine(std::istream &stream, std::string &line,
                     const std::string &path, const std::string &label) {
    line.clear();
    char byte = 0;
    while (stream.get(byte)) {
        if (byte == '\n') {
            return true;
        }
        if (line.size() == kMaxTextRecordBytes) {
            malformed(path, label + " exceeds the supported line length");
        }
        line.push_back(byte);
    }
    if (stream.bad()) {
        malformed(path, "failed while reading " + label);
    }
    return !line.empty();
}

bool nextRecord(std::istream &stream, std::string &line,
                const std::string &path, const std::string &label) {
    while (readBoundedLine(stream, line, path, label)) {
        const size_t start = line.find_first_not_of(" \t\r");
        if (start == std::string::npos || line[start] == '#') {
            continue;
        }
        return true;
    }
    return false;
}

int32_t colmapModelId(const std::string &name) {
    if (name == "SIMPLE_PINHOLE") return SIMPLE_PINHOLE;
    if (name == "PINHOLE") return PINHOLE;
    if (name == "SIMPLE_RADIAL") return SIMPLE_RADIAL;
    if (name == "RADIAL") return RADIAL;
    if (name == "OPENCV") return OPENCV;
    throw std::runtime_error("Unsupported COLMAP camera model: " + name);
}

size_t cameraParameterCount(int32_t model) {
    switch (model) {
        case SIMPLE_PINHOLE: return 3;
        case PINHOLE: return 4;
        case SIMPLE_RADIAL: return 4;
        case RADIAL: return 5;
        case OPENCV: return 8;
        default:
            throw std::runtime_error(
                "Unsupported COLMAP camera model: " + std::to_string(model));
    }
}

void assignCameraParameters(ColmapCamera &camera, const double *parameters,
                            const std::string &path) {
    auto parameter = [&](size_t index, const std::string &label) {
        return checkedFloat(parameters[index], path, label);
    };
    switch (camera.model) {
        case SIMPLE_PINHOLE:
            camera.fx = camera.fy = parameter(0, "focal length");
            camera.cx = parameter(1, "principal point x");
            camera.cy = parameter(2, "principal point y");
            break;
        case PINHOLE:
            camera.fx = parameter(0, "focal length x");
            camera.fy = parameter(1, "focal length y");
            camera.cx = parameter(2, "principal point x");
            camera.cy = parameter(3, "principal point y");
            break;
        case SIMPLE_RADIAL:
            camera.fx = camera.fy = parameter(0, "focal length");
            camera.cx = parameter(1, "principal point x");
            camera.cy = parameter(2, "principal point y");
            camera.k1 = parameter(3, "radial distortion k1");
            break;
        case RADIAL:
            camera.fx = camera.fy = parameter(0, "focal length");
            camera.cx = parameter(1, "principal point x");
            camera.cy = parameter(2, "principal point y");
            camera.k1 = parameter(3, "radial distortion k1");
            camera.k2 = parameter(4, "radial distortion k2");
            break;
        case OPENCV:
            camera.fx = parameter(0, "focal length x");
            camera.fy = parameter(1, "focal length y");
            camera.cx = parameter(2, "principal point x");
            camera.cy = parameter(3, "principal point y");
            camera.k1 = parameter(4, "radial distortion k1");
            camera.k2 = parameter(5, "radial distortion k2");
            camera.p1 = parameter(6, "tangential distortion p1");
            camera.p2 = parameter(7, "tangential distortion p2");
            break;
        default:
            throw std::runtime_error(
                "Unsupported COLMAP camera model: " + std::to_string(camera.model));
    }
    if (!(camera.fx > 0.0f) || !(camera.fy > 0.0f)) {
        malformed(path, "camera focal lengths must be positive");
    }
}

std::unordered_map<uint32_t, ColmapCamera> readCamerasBinary(const std::string &path) {
    BinaryReader reader(path);
    const uint64_t count = reader.read<uint64_t>("camera count");
    reader.requireRecords(count, kMaxDescriptorCount, 48, "camera");

    std::unordered_map<uint32_t, ColmapCamera> cameras;
    cameras.reserve(checkedSize(count, cameras.max_size(), path, "camera count"));
    for (uint64_t index = 0; index < count; ++index) {
        ColmapCamera camera;
        camera.id = reader.read<uint32_t>("camera ID");
        camera.model = reader.read<int32_t>("camera model");
        camera.width = checkedDimension(reader.read<uint64_t>("camera width"), path,
                                        "camera width");
        camera.height = checkedDimension(reader.read<uint64_t>("camera height"), path,
                                         "camera height");

        double parameters[8] = {};
        const size_t parameterCount = cameraParameterCount(camera.model);
        for (size_t parameterIndex = 0; parameterIndex < parameterCount; ++parameterIndex) {
            parameters[parameterIndex] = reader.read<double>("camera parameter");
        }
        assignCameraParameters(camera, parameters, path);
        if (!cameras.emplace(camera.id, camera).second) {
            malformed(path, "duplicate camera ID " + std::to_string(camera.id));
        }
    }
    reader.requireEnd("camera records");
    return cameras;
}

std::unordered_map<uint32_t, ColmapCamera> readCamerasText(const std::string &path) {
    std::ifstream stream(path);
    if (!stream.is_open()) {
        throw std::runtime_error("Cannot open " + path);
    }

    std::unordered_map<uint32_t, ColmapCamera> cameras;
    std::string line;
    while (nextRecord(stream, line, path, "camera record")) {
        if (cameras.size() == kMaxDescriptorCount) {
            malformed(path, "camera count is outside the supported range");
        }
        std::istringstream record(line);
        ColmapCamera camera;
        camera.id = parseUint32(readToken(record, path, "camera ID"), path, "camera ID");
        camera.model = colmapModelId(readToken(record, path, "camera model"));
        camera.width = checkedDimension(
            parseUnsigned(readToken(record, path, "camera width"),
                          std::numeric_limits<uint64_t>::max(), path, "camera width"),
            path, "camera width");
        camera.height = checkedDimension(
            parseUnsigned(readToken(record, path, "camera height"),
                          std::numeric_limits<uint64_t>::max(), path, "camera height"),
            path, "camera height");

        double parameters[8] = {};
        const size_t parameterCount = cameraParameterCount(camera.model);
        for (size_t index = 0; index < parameterCount; ++index) {
            parameters[index] = readFiniteDouble(record, path, "camera parameter");
        }
        requireNoExtraTokens(record, path, "camera record");
        assignCameraParameters(camera, parameters, path);
        if (!cameras.emplace(camera.id, camera).second) {
            malformed(path, "duplicate camera ID " + std::to_string(camera.id));
        }
    }
    return cameras;
}

void validateImagePose(const ColmapImage &image, const std::string &path) {
    for (double component : image.quaternion) {
        if (!std::isfinite(component)) {
            malformed(path, "image quaternion is not finite");
        }
    }
    for (double component : image.translation) {
        if (!std::isfinite(component)) {
            malformed(path, "image translation is not finite");
        }
    }
    const double norm = std::hypot(
        std::hypot(image.quaternion[0], image.quaternion[1]),
        std::hypot(image.quaternion[2], image.quaternion[3]));
    if (!(norm > 0.0) || !std::isfinite(norm)) {
        malformed(path, "image quaternion has invalid length");
    }
}

std::vector<ColmapImage> readImagesBinary(const std::string &path) {
    BinaryReader reader(path);
    const uint64_t count = reader.read<uint64_t>("image count");
    reader.requireRecords(count, kMaxDescriptorCount, 73, "image");

    std::vector<ColmapImage> images;
    images.reserve(checkedSize(count, images.max_size(), path, "image count"));
    std::unordered_set<uint32_t> imageIds;
    imageIds.reserve(images.capacity());

    for (uint64_t index = 0; index < count; ++index) {
        ColmapImage image;
        image.id = reader.read<uint32_t>("image ID");
        for (double &component : image.quaternion) {
            component = reader.read<double>("image quaternion");
        }
        for (double &component : image.translation) {
            component = reader.read<double>("image translation");
        }
        image.cameraId = reader.read<uint32_t>("image camera ID");
        image.filename = reader.readCString("image filename");
        if (image.filename.empty()) {
            malformed(path, "image filename is empty");
        }
        validateImagePose(image, path);

        const uint64_t observationCount = reader.read<uint64_t>("image observation count");
        reader.requireRecords(observationCount,
                              std::numeric_limits<uint32_t>::max(), 24,
                              "image observation");
        image.observations.reserve(checkedSize(
            observationCount, image.observations.max_size(), path,
            "image observation count"));
        for (uint64_t observationIndex = 0;
             observationIndex < observationCount; ++observationIndex) {
            ColmapImageObservation observation;
            observation.x = checkedFloat(reader.read<double>("observation x"), path,
                                         "observation x");
            observation.y = checkedFloat(reader.read<double>("observation y"), path,
                                         "observation y");
            observation.pointId = reader.read<uint64_t>("observation point3D ID");
            image.observations.push_back(observation);
        }

        if (!imageIds.insert(image.id).second) {
            malformed(path, "duplicate image ID " + std::to_string(image.id));
        }
        images.push_back(std::move(image));
    }
    reader.requireEnd("image records");
    return images;
}

std::vector<ColmapImage> readImagesText(const std::string &path) {
    std::ifstream stream(path);
    if (!stream.is_open()) {
        throw std::runtime_error("Cannot open " + path);
    }

    std::vector<ColmapImage> images;
    std::unordered_set<uint32_t> imageIds;
    std::string line;
    while (nextRecord(stream, line, path, "image record")) {
        if (images.size() == kMaxDescriptorCount) {
            malformed(path, "image count is outside the supported range");
        }
        std::istringstream record(line);
        ColmapImage image;
        image.id = parseUint32(readToken(record, path, "image ID"), path, "image ID");
        for (double &component : image.quaternion) {
            component = readFiniteDouble(record, path, "image quaternion");
        }
        for (double &component : image.translation) {
            component = readFiniteDouble(record, path, "image translation");
        }
        image.cameraId = parseUint32(
            readToken(record, path, "image camera ID"), path, "image camera ID");

        std::getline(record, image.filename);
        const size_t start = image.filename.find_first_not_of(" \t");
        const size_t end = image.filename.find_last_not_of(" \t\r");
        image.filename = start == std::string::npos
            ? std::string()
            : image.filename.substr(start, end - start + 1);
        if (image.filename.empty()) {
            malformed(path, "image filename is empty");
        }
        if (image.filename.size() > kMaxFilenameBytes) {
            malformed(path, "image filename exceeds the supported length");
        }
        validateImagePose(image, path);

        if (!readBoundedLine(stream, line, path, "2D observation record")) {
            malformed(path, "missing 2D observation line for image " +
                                std::to_string(image.id));
        }
        std::istringstream observationLine(line);
        while (true) {
            double x = 0.0;
            if (!(observationLine >> x)) {
                if (!observationLine.eof()) {
                    malformed(path, "invalid observation x");
                }
                break;
            }
            const double y = readFiniteDouble(observationLine, path, "observation y");
            const std::string pointId = readToken(
                observationLine, path, "observation point3D ID");
            if (image.observations.size() == std::numeric_limits<uint32_t>::max()) {
                malformed(path, "too many observations for image " +
                                    std::to_string(image.id));
            }
            image.observations.push_back({
                checkedFloat(x, path, "observation x"),
                checkedFloat(y, path, "observation y"),
                parsePointReference(pointId, path),
            });
        }

        if (!imageIds.insert(image.id).second) {
            malformed(path, "duplicate image ID " + std::to_string(image.id));
        }
        images.push_back(std::move(image));
    }
    return images;
}

ColmapTrackValidation makeTrackValidation(
    const std::vector<ColmapImage> &images, const std::string &path) {
    ColmapTrackValidation validation;
    validation.imageIndexById.reserve(images.size());
    validation.markers.reserve(images.size());
    for (size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex) {
        if (!validation.imageIndexById.emplace(images[imageIndex].id,
                                                imageIndex).second) {
            malformed(path, "duplicate image ID " +
                                std::to_string(images[imageIndex].id));
        }
        if (images[imageIndex].observations.size() >
            std::numeric_limits<size_t>::max() -
                validation.totalObservationCount) {
            malformed(path, "total image observation count overflows");
        }
        validation.totalObservationCount +=
            images[imageIndex].observations.size();
        validation.markers.emplace_back(
            images[imageIndex].observations.size(), uint8_t{0});
    }
    return validation;
}

void validateAndMarkTrack(const std::vector<ColmapImage> &images,
                          ColmapTrackValidation &validation,
                          uint64_t pointId, uint32_t imageId,
                          uint32_t observationIndex,
                          const std::string &path) {
    const auto image = validation.imageIndexById.find(imageId);
    if (image == validation.imageIndexById.end()) {
        malformed(path, "point " + std::to_string(pointId) +
                            " track references unknown image " +
                            std::to_string(imageId));
    }
    const size_t imageIndex = image->second;
    if (observationIndex >= images[imageIndex].observations.size()) {
        malformed(path, "point " + std::to_string(pointId) +
                            " track references image " +
                            std::to_string(imageId) + " observation " +
                            std::to_string(observationIndex) +
                            " outside its observation list");
    }

    const ColmapImageObservation &observation =
        images[imageIndex].observations[observationIndex];
    if (observation.pointId == kInvalidPointId) {
        malformed(path, "point " + std::to_string(pointId) +
                            " track references an unlinked image observation");
    }
    if (observation.pointId != pointId) {
        malformed(path, "point track and image observation disagree for image " +
                            std::to_string(imageId) + " observation " +
                            std::to_string(observationIndex));
    }

    uint8_t &marker = validation.markers[imageIndex][observationIndex];
    if (marker != 0) {
        malformed(path, "duplicate track entry for image " +
                            std::to_string(imageId) + " observation " +
                            std::to_string(observationIndex));
    }
    marker = 1;
}

SparsePointSet readPointsBinary(const std::string &path,
                                const std::vector<ColmapImage> &images,
                                ColmapTrackValidation &trackValidation) {
    BinaryReader reader(path);
    const uint64_t count = reader.read<uint64_t>("point count");
    reader.requireRecords(count, kMaxDescriptorCount, 51, "point");

    SparsePointSet points;
    const size_t pointCount = checkedSize(
        count, points.sourceIds.max_size(), path, "point count");
    const size_t coordinateCount = checkedProduct(
        pointCount, 3, points.xyz.max_size(), path, "point coordinate");
    if (coordinateCount > points.rgb.max_size()) {
        malformed(path, "point color size overflows");
    }
    if (pointCount > points.reprojectionErrors.max_size()) {
        malformed(path, "point reprojection-error size overflows");
    }
    points.xyz.reserve(coordinateCount);
    points.rgb.reserve(coordinateCount);
    points.sourceIds.reserve(pointCount);
    points.reprojectionErrors.reserve(pointCount);
    std::unordered_set<uint64_t> pointIds;
    pointIds.reserve(pointCount);

    for (uint64_t index = 0; index < count; ++index) {
        const uint64_t pointId = reader.read<uint64_t>("point ID");
        if (pointId == kInvalidPointId) {
            malformed(path, "point ID UINT64_MAX is reserved for unlinked observations");
        }
        if (!pointIds.insert(pointId).second) {
            malformed(path, "duplicate point ID " + std::to_string(pointId));
        }

        for (size_t component = 0; component < 3; ++component) {
            points.xyz.push_back(checkedFloat(
                reader.read<double>("point coordinate"), path, "point coordinate"));
        }
        for (size_t component = 0; component < 3; ++component) {
            points.rgb.push_back(reader.read<uint8_t>("point color"));
        }
        const double error = reader.read<double>("point reprojection error");
        if (error < 0.0) {
            malformed(path, "point reprojection error is negative");
        }
        points.sourceIds.push_back(pointId);
        points.reprojectionErrors.push_back(
            checkedFloat(error, path, "point reprojection error"));

        const uint64_t trackCount = reader.read<uint64_t>("point track count");
        reader.requireRecords(trackCount, std::numeric_limits<uint32_t>::max(), 8,
                              "point track");
        for (uint64_t trackIndex = 0; trackIndex < trackCount; ++trackIndex) {
            const uint32_t imageId =
                reader.read<uint32_t>("track image ID");
            const uint32_t observationIndex =
                reader.read<uint32_t>("track observation index");
            validateAndMarkTrack(images, trackValidation, pointId, imageId,
                                 observationIndex, path);
        }
    }
    reader.requireEnd("point records");
    return points;
}

SparsePointSet readPointsText(const std::string &path,
                              const std::vector<ColmapImage> &images,
                              ColmapTrackValidation &trackValidation) {
    std::ifstream stream(path);
    if (!stream.is_open()) {
        throw std::runtime_error("Cannot open " + path);
    }

    SparsePointSet points;
    std::unordered_set<uint64_t> pointIds;
    std::string line;
    while (nextRecord(stream, line, path, "point record")) {
        if (points.sourceIds.size() == kMaxDescriptorCount) {
            malformed(path, "point count is outside the supported range");
        }
        std::istringstream record(line);
        const uint64_t pointId = parseUnsigned(
            readToken(record, path, "point ID"), kInvalidPointId - 1,
            path, "point ID");
        if (!pointIds.insert(pointId).second) {
            malformed(path, "duplicate point ID " + std::to_string(pointId));
        }

        for (size_t component = 0; component < 3; ++component) {
            points.xyz.push_back(checkedFloat(
                readFiniteDouble(record, path, "point coordinate"), path,
                "point coordinate"));
        }
        for (size_t component = 0; component < 3; ++component) {
            const uint64_t color = parseUnsigned(
                readToken(record, path, "point color"), 255, path, "point color");
            points.rgb.push_back(static_cast<uint8_t>(color));
        }
        const double error = readFiniteDouble(record, path, "point reprojection error");
        if (error < 0.0) {
            malformed(path, "point reprojection error is negative");
        }
        points.sourceIds.push_back(pointId);
        points.reprojectionErrors.push_back(
            checkedFloat(error, path, "point reprojection error"));

        uint64_t trackCount = 0;
        while (true) {
            std::string imageId;
            if (!(record >> imageId)) {
                if (!record.eof()) {
                    malformed(path, "invalid track image ID");
                }
                break;
            }
            const std::string observationIndex = readToken(
                record, path, "track observation index");
            if (trackCount == std::numeric_limits<uint32_t>::max()) {
                malformed(path, "point track count is outside the supported range");
            }
            validateAndMarkTrack(
                images, trackValidation, pointId,
                parseUint32(imageId, path, "track image ID"),
                parseUint32(observationIndex, path, "track observation index"),
                path);
            ++trackCount;
        }
    }
    return points;
}

void worldToCameraToCameraToWorld(const ColmapImage &image,
                                  std::array<float, 16> &output,
                                  const std::string &path) {
    const double norm = std::hypot(
        std::hypot(image.quaternion[0], image.quaternion[1]),
        std::hypot(image.quaternion[2], image.quaternion[3]));
    if (!(norm > 0.0) || !std::isfinite(norm)) {
        malformed(path, "image quaternion has invalid length");
    }
    const double w = image.quaternion[0] / norm;
    const double x = image.quaternion[1] / norm;
    const double y = image.quaternion[2] / norm;
    const double z = image.quaternion[3] / norm;

    const double rotation[9] = {
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
        2.0 * (x * z + w * y), 2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    };
    const double inverse[9] = {
        rotation[0], rotation[3], rotation[6],
        rotation[1], rotation[4], rotation[7],
        rotation[2], rotation[5], rotation[8],
    };
    const double inverseTranslation[3] = {
        -(inverse[0] * image.translation[0] + inverse[1] * image.translation[1] +
          inverse[2] * image.translation[2]),
        -(inverse[3] * image.translation[0] + inverse[4] * image.translation[1] +
          inverse[5] * image.translation[2]),
        -(inverse[6] * image.translation[0] + inverse[7] * image.translation[1] +
          inverse[8] * image.translation[2]),
    };
    const double values[16] = {
        inverse[0], -inverse[1], -inverse[2], inverseTranslation[0],
        inverse[3], -inverse[4], -inverse[5], inverseTranslation[1],
        inverse[6], -inverse[7], -inverse[8], inverseTranslation[2],
        0.0, 0.0, 0.0, 1.0,
    };
    for (size_t index = 0; index < output.size(); ++index) {
        output[index] = checkedFloat(values[index], path, "camera-to-world transform");
    }
}

void appendObservationsAndFinishTrackValidation(
    DatasetDescriptor &descriptor,
    std::vector<ColmapImage> &images,
    const SparsePointSet *colmapPoints,
    ColmapTrackValidation *trackValidation,
    bool usesPlyPoints,
    const std::string &source) {
    std::unordered_map<uint64_t, uint32_t> pointIndexById;
    if (colmapPoints != nullptr) {
        pointIndexById.reserve(colmapPoints->sourceIds.size());
        for (size_t index = 0; index < colmapPoints->sourceIds.size(); ++index) {
            pointIndexById.emplace(colmapPoints->sourceIds[index],
                                   static_cast<uint32_t>(index));
        }
    }

    size_t totalObservationCount = 0;
    if (trackValidation != nullptr) {
        if (colmapPoints == nullptr ||
            trackValidation->markers.size() != images.size()) {
            throw std::runtime_error(
                "Internal COLMAP track-validation state mismatch in " + source);
        }
        totalObservationCount = trackValidation->totalObservationCount;
        trackValidation->imageIndexById.clear();
        trackValidation->imageIndexById.rehash(0);
    } else {
        for (const ColmapImage &image : images) {
            if (image.observations.size() >
                descriptor.observations.max_size() - totalObservationCount) {
                throw std::runtime_error(
                    "COLMAP observation count is too large in " + source);
            }
            totalObservationCount += image.observations.size();
        }
    }
    if (totalObservationCount > descriptor.observations.max_size()) {
        throw std::runtime_error("COLMAP observation count is too large in " + source);
    }
    // Allocate the exact retained size once. Geometric vector growth can
    // otherwise leave nearly 2x observation capacity in InputData and briefly
    // hold both old and new allocations during a late reallocation.
    descriptor.observations.reserve(totalObservationCount);

    for (size_t frameIndex = 0; frameIndex < images.size(); ++frameIndex) {
        ColmapImage &image = images[frameIndex];
        std::vector<uint8_t> *markers = trackValidation == nullptr
            ? nullptr : &trackValidation->markers[frameIndex];
        if (markers != nullptr && markers->size() != image.observations.size()) {
            throw std::runtime_error(
                "Internal COLMAP observation-marker size mismatch in " + source);
        }
        for (size_t observationIndex = 0;
             observationIndex < image.observations.size(); ++observationIndex) {
            const ColmapImageObservation &sourceObservation =
                image.observations[observationIndex];
            int32_t pointIndex = -1;
            if (sourceObservation.pointId != kInvalidPointId) {
                if (usesPlyPoints) {
                    throw std::runtime_error(
                        "COLMAP image observations cannot be linked when points3D.ply "
                        "is used: PLY has no COLMAP point IDs or tracks");
                }
                const auto point = pointIndexById.find(sourceObservation.pointId);
                if (point == pointIndexById.end()) {
                    throw std::runtime_error(
                        "COLMAP image " + std::to_string(image.id) + " observation " +
                        std::to_string(observationIndex) + " references unknown point " +
                        std::to_string(sourceObservation.pointId));
                }
                if (markers != nullptr && (*markers)[observationIndex] == 0) {
                    throw std::runtime_error(
                        "COLMAP image " + std::to_string(image.id) +
                        " observation " + std::to_string(observationIndex) +
                        " is linked but missing from its point track");
                }
                pointIndex = static_cast<int32_t>(point->second);
            } else if (markers != nullptr && (*markers)[observationIndex] != 0) {
                throw std::runtime_error(
                    "COLMAP image " + std::to_string(image.id) +
                    " unlinked observation " +
                    std::to_string(observationIndex) +
                    " unexpectedly appears in a point track");
            }

            SparseObservation observation;
            observation.frameIndex = static_cast<uint32_t>(frameIndex);
            observation.frameObservationIndex = static_cast<uint32_t>(observationIndex);
            observation.pointIndex = pointIndex;
            observation.x = sourceObservation.x;
            observation.y = sourceObservation.y;
            descriptor.observations.push_back(observation);
        }
        std::vector<ColmapImageObservation>().swap(image.observations);
        if (markers != nullptr) {
            std::vector<uint8_t>().swap(*markers);
        }
    }
}

} // namespace

DatasetDescriptor loaders::loadColmap(const std::string &projectRoot,
                                      const std::string &imageSourcePath) {
    const fs::path root(projectRoot);
    const auto hasModel = [](const fs::path &directory) {
        return fs::exists(directory / "cameras.bin") ||
               fs::exists(directory / "cameras.txt");
    };
    const fs::path sparse = hasModel(root) ? root : root / "sparse" / "0";

    const std::string imageDirectory = !imageSourcePath.empty()
        ? imageSourcePath
        : fs::exists(root / "images") ? (root / "images").string() : projectRoot;

    const bool usesBinaryCameras = fs::exists(sparse / "cameras.bin");
    const bool usesBinaryImages = fs::exists(sparse / "images.bin");
    const auto cameras = usesBinaryCameras
        ? readCamerasBinary((sparse / "cameras.bin").string())
        : readCamerasText((sparse / "cameras.txt").string());
    auto images = usesBinaryImages
        ? readImagesBinary((sparse / "images.bin").string())
        : readImagesText((sparse / "images.txt").string());

    std::sort(images.begin(), images.end(), [](const ColmapImage &lhs,
                                                const ColmapImage &rhs) {
        if (lhs.filename != rhs.filename) {
            return lhs.filename < rhs.filename;
        }
        return lhs.id < rhs.id;
    });

    DatasetDescriptor descriptor;
    descriptor.provenance.adapter = "colmap";
    descriptor.provenance.source = projectRoot;
    descriptor.frames.reserve(images.size());
    const std::string poseSource = (usesBinaryImages
        ? sparse / "images.bin" : sparse / "images.txt").string();
    for (const ColmapImage &image : images) {
        const auto camera = cameras.find(image.cameraId);
        if (camera == cameras.end()) {
            throw std::runtime_error(
                "COLMAP image " + std::to_string(image.id) +
                " references unknown camera " + std::to_string(image.cameraId));
        }

        DatasetFrameDescriptor frame;
        frame.id = std::to_string(image.id);
        frame.calibrationId = std::to_string(image.cameraId);
        frame.imagePath = (fs::path(imageDirectory) / image.filename).string();
        frame.rasterOrientation = RasterOrientation::EncodedPixels;
        frame.calibration.width = camera->second.width;
        frame.calibration.height = camera->second.height;
        frame.calibration.fx = camera->second.fx;
        frame.calibration.fy = camera->second.fy;
        frame.calibration.cx = camera->second.cx;
        frame.calibration.cy = camera->second.cy;
        frame.calibration.k1 = camera->second.k1;
        frame.calibration.k2 = camera->second.k2;
        frame.calibration.p1 = camera->second.p1;
        frame.calibration.p2 = camera->second.p2;
        worldToCameraToCameraToWorld(image, frame.cameraToWorld, poseSource);
        descriptor.frames.push_back(std::move(frame));
    }

    ColmapTrackValidation trackValidation;
    ColmapTrackValidation *trackValidationState = nullptr;
    const SparsePointSet *colmapPoints = nullptr;
    bool usesPlyPoints = false;
    if (fs::exists(sparse / "points3D.bin")) {
        const std::string path = (sparse / "points3D.bin").string();
        trackValidation = makeTrackValidation(images, path);
        descriptor.points = readPointsBinary(path, images, trackValidation);
        trackValidationState = &trackValidation;
        colmapPoints = &descriptor.points;
    } else if (fs::exists(sparse / "points3D.txt")) {
        const std::string path = (sparse / "points3D.txt").string();
        trackValidation = makeTrackValidation(images, path);
        descriptor.points = readPointsText(path, images, trackValidation);
        trackValidationState = &trackValidation;
        colmapPoints = &descriptor.points;
    } else if (fs::exists(sparse / "points3D.ply")) {
        descriptor.points = readPly((sparse / "points3D.ply").string());
        usesPlyPoints = true;
    }

    appendObservationsAndFinishTrackValidation(
        descriptor, images, colmapPoints, trackValidationState,
        usesPlyPoints, projectRoot);
    return descriptor;
}
