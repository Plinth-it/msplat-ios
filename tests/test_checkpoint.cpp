#include "model.hpp"
#include "atomic_output.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

namespace {

constexpr uint32_t checkpointMagic = 0x4C50534D;
constexpr uint32_t checkpointVersionV1 = 1;
constexpr uint32_t checkpointVersionV2 = 2;

template <typename T>
void appendScalar(std::vector<uint8_t> &bytes, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto *source = reinterpret_cast<const uint8_t*>(&value);
    bytes.insert(bytes.end(), source, source + sizeof(value));
}

uint64_t tensorBytes(const std::vector<int64_t> &shape) {
    uint64_t elements = 1;
    for (int64_t dimension : shape)
        elements *= static_cast<uint64_t>(dimension);
    return elements * sizeof(float);
}

void appendTensor(std::vector<uint8_t> &bytes,
                  const std::vector<int64_t> &shape,
                  size_t *byteCountOffset = nullptr) {
    appendScalar(bytes, static_cast<uint32_t>(shape.size()));
    for (int64_t dimension : shape)
        appendScalar(bytes, dimension);
    if (byteCountOffset)
        *byteCountOffset = bytes.size();
    const uint64_t payloadBytes = tensorBytes(shape);
    appendScalar(bytes, payloadBytes);
    bytes.resize(bytes.size() + static_cast<size_t>(payloadBytes), 0);
}

std::vector<uint8_t> validCheckpoint(size_t &firstTensorByteCountOffset) {
    constexpr uint32_t numPoints = 2;
    constexpr uint32_t shDegree = 1;
    const std::array<std::vector<int64_t>, Model::N_ADAM_GROUPS> shapes = {{
        {numPoints, 3},
        {numPoints, 3},
        {numPoints, 4},
        {numPoints, 3},
        {numPoints, 3, 3},
        {numPoints, 1},
    }};

    std::vector<uint8_t> bytes;
    appendScalar(bytes, checkpointMagic);
    appendScalar(bytes, checkpointVersionV1);
    appendScalar(bytes, uint32_t{17});
    appendScalar(bytes, numPoints);
    appendScalar(bytes, shDegree);
    appendScalar(bytes, uint32_t{17});
    for (int group = 0; group < Model::N_ADAM_GROUPS; ++group)
        appendScalar(bytes, 0.001f);
    appendScalar(bytes, 0.00016f);
    appendScalar(bytes, 0.0000016f);

    for (int copy = 0; copy < 3; ++copy) {
        for (int group = 0; group < Model::N_ADAM_GROUPS; ++group) {
            appendTensor(bytes, shapes[group],
                         copy == 0 && group == 0 ? &firstTensorByteCountOffset : nullptr);
        }
    }
    return bytes;
}

void appendString(std::vector<uint8_t> &bytes, const std::string &value) {
    appendScalar(bytes, static_cast<uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

std::vector<uint8_t> validV2Checkpoint(
    bool photometricEnabled,
    size_t &firstTensorByteCountOffset,
    size_t &photometricFlagOffset,
    size_t &firstPhotometricStepOffset,
    size_t &firstPhotometricTensorByteCountOffset,
    const std::array<std::string, 2> &frameIds = {"frame-0", "frame-1"}) {
    std::vector<uint8_t> bytes = validCheckpoint(firstTensorByteCountOffset);
    std::memcpy(bytes.data() + sizeof(uint32_t), &checkpointVersionV2,
                sizeof(checkpointVersionV2));
    photometricFlagOffset = bytes.size();
    appendScalar(bytes, photometricEnabled ? uint32_t{1} : uint32_t{0});
    firstPhotometricStepOffset = 0;
    firstPhotometricTensorByteCountOffset = 0;
    if (!photometricEnabled)
        return bytes;

    appendScalar(bytes, uint32_t{2});
    for (const std::string &frameId : frameIds)
        appendString(bytes, frameId);
    firstPhotometricStepOffset = bytes.size();
    appendScalar(bytes, uint32_t{3});
    appendScalar(bytes, uint32_t{7});
    for (int copy = 0; copy < 3; ++copy) {
        appendTensor(bytes, {2, 3},
                     copy == 0
                         ? &firstPhotometricTensorByteCountOffset
                         : nullptr);
    }
    return bytes;
}

void writeFile(const fs::path &path, const std::vector<uint8_t> &bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("could not create checkpoint test file");
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!stream)
        throw std::runtime_error("could not write checkpoint test file");
}

std::string readTextFile(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("could not open atomic-output test file");
    return std::string(std::istreambuf_iterator<char>(stream),
                       std::istreambuf_iterator<char>());
}

bool rejects(const fs::path &path, const std::string &messageFragment) {
    try {
        validateCheckpointFile(path.string());
    } catch (const std::runtime_error &error) {
        return std::string(error.what()).find(messageFragment) != std::string::npos;
    }
    return false;
}

struct RemoveFiles {
    std::vector<fs::path> paths;
    ~RemoveFiles() {
        std::error_code ignored;
        for (const fs::path &path : paths)
            fs::remove(path, ignored);
    }
};

} // namespace

int main() {
    const fs::path prefix = fs::temp_directory_path() /
        ("msplat-checkpoint-test-" + std::to_string(static_cast<long long>(getpid())));
    const fs::path validPath = prefix.string() + "-valid.msplat";
    const fs::path truncatedPath = prefix.string() + "-truncated.msplat";
    const fs::path badByteCountPath = prefix.string() + "-byte-count.msplat";
    const fs::path maxStepPath = prefix.string() + "-max-step.msplat";
    const fs::path maxAdamStepPath = prefix.string() + "-max-adam-step.msplat";
    const fs::path zeroMeansRatePath = prefix.string() + "-zero-means-rate.msplat";
    const fs::path validV2DisabledPath = prefix.string() + "-valid-v2-disabled.msplat";
    const fs::path validV2EnabledPath = prefix.string() + "-valid-v2-enabled.msplat";
    const fs::path invalidPhotoFlagPath = prefix.string() + "-photo-flag.msplat";
    const fs::path duplicatePhotoIdPath = prefix.string() + "-photo-id.msplat";
    const fs::path maxPhotoStepPath = prefix.string() + "-photo-step.msplat";
    const fs::path inconsistentPhotoStepsPath =
        prefix.string() + "-photo-step-sum.msplat";
    const fs::path badPhotoTensorPath = prefix.string() + "-photo-tensor.msplat";
    const fs::path hugePhotoCountPath = prefix.string() + "-photo-count.msplat";
    const fs::path truncatedPhotoPayloadPath =
        prefix.string() + "-photo-payload.msplat";
    const fs::path trailingV2Path = prefix.string() + "-trailing-v2.msplat";
    const fs::path atomicDestination = prefix.string() + "-atomic.bin";
    const fs::path atomicSentinel = atomicDestination.string() + ".tmp";
    RemoveFiles cleanup{{validPath, truncatedPath, badByteCountPath,
                         maxStepPath, maxAdamStepPath, zeroMeansRatePath,
                         validV2DisabledPath, validV2EnabledPath,
                         invalidPhotoFlagPath, duplicatePhotoIdPath,
                         maxPhotoStepPath, inconsistentPhotoStepsPath,
                         badPhotoTensorPath, hugePhotoCountPath,
                         truncatedPhotoPayloadPath, trailingV2Path,
                         atomicDestination, atomicSentinel}};

    size_t firstTensorByteCountOffset = 0;
    const std::vector<uint8_t> valid = validCheckpoint(firstTensorByteCountOffset);
    writeFile(validPath, valid);
    validateCheckpointFile(validPath.string());

    std::vector<uint8_t> truncated(valid.begin(), valid.begin() + 10);
    writeFile(truncatedPath, truncated);
    CHECK(rejects(truncatedPath, "truncated"));

    std::vector<uint8_t> badByteCount = valid;
    uint64_t declaredBytes = 0;
    std::memcpy(&declaredBytes, badByteCount.data() + firstTensorByteCountOffset,
                sizeof(declaredBytes));
    ++declaredBytes;
    std::memcpy(badByteCount.data() + firstTensorByteCountOffset, &declaredBytes,
                sizeof(declaredBytes));
    writeFile(badByteCountPath, badByteCount);
    CHECK(rejects(badByteCountPath, "byte count"));

    std::vector<uint8_t> maxStep = valid;
    const uint32_t maxInt = static_cast<uint32_t>(std::numeric_limits<int>::max());
    std::memcpy(maxStep.data() + 8, &maxInt, sizeof(maxInt));
    writeFile(maxStepPath, maxStep);
    CHECK(rejects(maxStepPath, "step exceeds"));

    std::vector<uint8_t> maxAdamStep = valid;
    std::memcpy(maxAdamStep.data() + 20, &maxInt, sizeof(maxInt));
    writeFile(maxAdamStepPath, maxAdamStep);
    CHECK(rejects(maxAdamStepPath, "Adam step count"));

    std::vector<uint8_t> zeroMeansRate = valid;
    const float zero = 0.0f;
    std::memcpy(zeroMeansRate.data() + 48, &zero, sizeof(zero));
    writeFile(zeroMeansRatePath, zeroMeansRate);
    CHECK(rejects(zeroMeansRatePath, "means learning rates"));

    size_t v2TensorOffset = 0;
    size_t photoFlagOffset = 0;
    size_t photoStepOffset = 0;
    size_t photoTensorOffset = 0;
    const std::vector<uint8_t> validV2Disabled = validV2Checkpoint(
        false, v2TensorOffset, photoFlagOffset, photoStepOffset,
        photoTensorOffset);
    writeFile(validV2DisabledPath, validV2Disabled);
    validateCheckpointFile(validV2DisabledPath.string());

    const std::vector<uint8_t> validV2Enabled = validV2Checkpoint(
        true, v2TensorOffset, photoFlagOffset, photoStepOffset,
        photoTensorOffset);
    writeFile(validV2EnabledPath, validV2Enabled);
    validateCheckpointFile(validV2EnabledPath.string());

    std::vector<uint8_t> invalidPhotoFlag = validV2Disabled;
    const uint32_t invalidFlag = 2;
    std::memcpy(invalidPhotoFlag.data() + photoFlagOffset, &invalidFlag,
                sizeof(invalidFlag));
    writeFile(invalidPhotoFlagPath, invalidPhotoFlag);
    CHECK(rejects(invalidPhotoFlagPath, "enabled flag"));

    size_t ignoredOffset = 0;
    std::vector<uint8_t> duplicatePhotoId = validV2Checkpoint(
        true, ignoredOffset, ignoredOffset, ignoredOffset, ignoredOffset,
        {"same", "same"});
    writeFile(duplicatePhotoIdPath, duplicatePhotoId);
    CHECK(rejects(duplicatePhotoIdPath, "not unique"));

    std::vector<uint8_t> maxPhotoStep = validV2Enabled;
    const uint32_t maxPhotoCount = std::numeric_limits<uint32_t>::max();
    std::memcpy(maxPhotoStep.data() + photoStepOffset, &maxPhotoCount,
                sizeof(maxPhotoCount));
    writeFile(maxPhotoStepPath, maxPhotoStep);
    CHECK(rejects(maxPhotoStepPath, "step count"));

    std::vector<uint8_t> inconsistentPhotoSteps = validV2Enabled;
    const uint32_t tenVisits = 10;
    std::memcpy(inconsistentPhotoSteps.data() + photoStepOffset, &tenVisits,
                sizeof(tenVisits));
    std::memcpy(inconsistentPhotoSteps.data() + photoStepOffset + sizeof(uint32_t),
                &tenVisits, sizeof(tenVisits));
    writeFile(inconsistentPhotoStepsPath, inconsistentPhotoSteps);
    CHECK(rejects(inconsistentPhotoStepsPath, "inconsistent with Adam"));

    std::vector<uint8_t> badPhotoTensor = validV2Enabled;
    uint64_t photoBytes = 0;
    std::memcpy(&photoBytes, badPhotoTensor.data() + photoTensorOffset,
                sizeof(photoBytes));
    ++photoBytes;
    std::memcpy(badPhotoTensor.data() + photoTensorOffset, &photoBytes,
                sizeof(photoBytes));
    writeFile(badPhotoTensorPath, badPhotoTensor);
    CHECK(rejects(badPhotoTensorPath, "byte count"));

    std::vector<uint8_t> hugePhotoCount = valid;
    std::memcpy(hugePhotoCount.data() + sizeof(uint32_t),
                &checkpointVersionV2, sizeof(checkpointVersionV2));
    appendScalar(hugePhotoCount, uint32_t{1});
    appendScalar(hugePhotoCount, uint32_t{1'000'001});
    writeFile(hugePhotoCountPath, hugePhotoCount);
    CHECK(rejects(hugePhotoCountPath, "camera count"));

    std::vector<uint8_t> truncatedPhotoPayload = valid;
    std::memcpy(truncatedPhotoPayload.data() + sizeof(uint32_t),
                &checkpointVersionV2, sizeof(checkpointVersionV2));
    appendScalar(truncatedPhotoPayload, uint32_t{1});
    appendScalar(truncatedPhotoPayload, uint32_t{1'000'000});
    writeFile(truncatedPhotoPayloadPath, truncatedPhotoPayload);
    CHECK(rejects(truncatedPhotoPayloadPath,
                  "truncated photometric checkpoint payload"));

    std::vector<uint8_t> trailingV2 = validV2Disabled;
    trailingV2.push_back(0);
    writeFile(trailingV2Path, trailingV2);
    CHECK(rejects(trailingV2Path, "trailing data"));

    {
        std::ofstream sentinel(atomicSentinel, std::ios::binary | std::ios::trunc);
        sentinel << "do not replace";
    }
    {
        msplat::detail::AtomicOutputFile output(atomicDestination.string());
        CHECK(output.temporary() != atomicSentinel);
        std::ofstream stream(output.temporary(), std::ios::binary | std::ios::trunc);
        stream << "new output";
        stream.flush();
        CHECK(static_cast<bool>(stream));
        stream.close();
        CHECK(static_cast<bool>(stream));
        output.commit("test");
    }
    CHECK(readTextFile(atomicSentinel) == "do not replace");
    CHECK(readTextFile(atomicDestination) == "new output");

    return 0;
}
