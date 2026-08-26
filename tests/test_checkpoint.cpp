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
constexpr uint32_t checkpointVersionV3 = 3;
constexpr uint32_t checkpointVersionV4 = 4;

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
                  size_t *byteCountOffset = nullptr,
                  size_t *dataOffset = nullptr) {
    appendScalar(bytes, static_cast<uint32_t>(shape.size()));
    for (int64_t dimension : shape)
        appendScalar(bytes, dimension);
    if (byteCountOffset)
        *byteCountOffset = bytes.size();
    const uint64_t payloadBytes = tensorBytes(shape);
    appendScalar(bytes, payloadBytes);
    if (dataOffset)
        *dataOffset = bytes.size();
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

struct V3CheckpointFixture {
    std::vector<uint8_t> bytes;
    size_t poseFlagOffset = 0;
    size_t cameraCountOffset = 0;
    size_t anchorIndexOffset = 0;
    size_t firstPoseStepOffset = 0;
    size_t basePosesByteCountOffset = 0;
    size_t poseOptimizerPayloadOffset = 0;
    size_t poseDeltaByteCountOffset = 0;
    size_t firstMomentByteCountOffset = 0;
    size_t secondMomentByteCountOffset = 0;
};

V3CheckpointFixture validV3Checkpoint(
    bool photometricEnabled,
    bool poseEnabled,
    const std::array<std::string, 2> &poseFrameIds = {"frame-0", "frame-1"},
    const std::array<std::string, 2> &photometricFrameIds = {"frame-0", "frame-1"}) {
    size_t ignored = 0;
    V3CheckpointFixture fixture;
    fixture.bytes = validV2Checkpoint(
        photometricEnabled, ignored, ignored, ignored, ignored,
        photometricFrameIds);
    std::memcpy(fixture.bytes.data() + sizeof(uint32_t),
                &checkpointVersionV3, sizeof(checkpointVersionV3));

    fixture.poseFlagOffset = fixture.bytes.size();
    appendScalar(fixture.bytes, poseEnabled ? uint32_t{1} : uint32_t{0});
    if (!poseEnabled)
        return fixture;

    fixture.cameraCountOffset = fixture.bytes.size();
    appendScalar(fixture.bytes, uint32_t{2});
    fixture.anchorIndexOffset = fixture.bytes.size();
    appendScalar(fixture.bytes, uint32_t{0});
    for (const std::string &frameId : poseFrameIds)
        appendString(fixture.bytes, frameId);

    // The anchored camera is never updated. The second camera's count remains
    // below the global Adam step count stored in the v1-compatible prefix.
    fixture.firstPoseStepOffset = fixture.bytes.size();
    appendScalar(fixture.bytes, uint32_t{0});
    appendScalar(fixture.bytes, uint32_t{10});

    size_t basePoseDataOffset = 0;
    appendTensor(fixture.bytes, {2, 16},
                 &fixture.basePosesByteCountOffset, &basePoseDataOffset);
    const std::array<float, 32> basePoses = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        1, 0, 0, 1,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    std::memcpy(fixture.bytes.data() + basePoseDataOffset,
                basePoses.data(), sizeof(basePoses));

    fixture.poseOptimizerPayloadOffset = fixture.bytes.size();
    appendTensor(fixture.bytes, {2, 6},
                 &fixture.poseDeltaByteCountOffset);
    appendTensor(fixture.bytes, {2, 6},
                 &fixture.firstMomentByteCountOffset);
    appendTensor(fixture.bytes, {2, 6},
                 &fixture.secondMomentByteCountOffset);
    return fixture;
}

struct V4CheckpointFixture {
    std::vector<uint8_t> bytes;
    size_t conditioningOffset = 0;
    size_t firstReadinessOffset = 0;
    size_t preconditionerByteCountOffset = 0;
};

V4CheckpointFixture validV4Checkpoint() {
    const V3CheckpointFixture v3 = validV3Checkpoint(false, true);
    V4CheckpointFixture fixture;
    fixture.bytes = v3.bytes;
    std::memcpy(fixture.bytes.data() + sizeof(uint32_t),
                &checkpointVersionV4, sizeof(checkpointVersionV4));

    std::vector<uint8_t> extension;
    fixture.conditioningOffset = v3.poseOptimizerPayloadOffset;
    appendScalar(extension, uint32_t{1});
    fixture.firstReadinessOffset =
        v3.poseOptimizerPayloadOffset + extension.size();
    appendScalar(extension, uint8_t{1});
    appendScalar(extension, uint8_t{1});
    size_t preconditionerDataOffset = 0;
    size_t relativeByteCountOffset = 0;
    appendTensor(extension, {2, 36}, &relativeByteCountOffset,
                 &preconditionerDataOffset);
    fixture.preconditionerByteCountOffset =
        v3.poseOptimizerPayloadOffset + relativeByteCountOffset;

    std::array<float, 72> preconditioners{};
    for (size_t camera = 0; camera < 2; ++camera) {
        for (size_t axis = 0; axis < 6; ++axis)
            preconditioners[camera * 36 + axis * 6 + axis] = 1.0f;
    }
    std::memcpy(extension.data() + preconditionerDataOffset,
                preconditioners.data(), sizeof(preconditioners));
    fixture.bytes.insert(
        fixture.bytes.begin() +
            static_cast<std::ptrdiff_t>(v3.poseOptimizerPayloadOffset),
        extension.begin(), extension.end());
    return fixture;
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
    const fs::path validV3DisabledPath = prefix.string() + "-valid-v3-disabled.msplat";
    const fs::path validV3PosePath = prefix.string() + "-valid-v3-pose.msplat";
    const fs::path validV3BothPath = prefix.string() + "-valid-v3-both.msplat";
    const fs::path poseCorruptPath = prefix.string() + "-pose-corrupt.msplat";
    const fs::path truncatedPosePayloadPath =
        prefix.string() + "-pose-payload.msplat";
    const fs::path trailingV3Path = prefix.string() + "-trailing-v3.msplat";
    const fs::path validV4Path = prefix.string() + "-valid-v4.msplat";
    const fs::path poseConditioningCorruptPath =
        prefix.string() + "-pose-conditioning-corrupt.msplat";
    const fs::path truncatedV4Path =
        prefix.string() + "-pose-conditioning-truncated.msplat";
    const fs::path atomicDestination = prefix.string() + "-atomic.bin";
    const fs::path atomicSentinel = atomicDestination.string() + ".tmp";
    RemoveFiles cleanup{{validPath, truncatedPath, badByteCountPath,
                         maxStepPath, maxAdamStepPath, zeroMeansRatePath,
                         validV2DisabledPath, validV2EnabledPath,
                         invalidPhotoFlagPath, duplicatePhotoIdPath,
                         maxPhotoStepPath, inconsistentPhotoStepsPath,
                         badPhotoTensorPath, hugePhotoCountPath,
                         truncatedPhotoPayloadPath, trailingV2Path,
                         validV3DisabledPath, validV3PosePath,
                         validV3BothPath, poseCorruptPath,
                         truncatedPosePayloadPath, trailingV3Path,
                         validV4Path, poseConditioningCorruptPath,
                         truncatedV4Path,
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

    // Version 3 retains the complete v2 prefix and adds an independently
    // optional pose-refinement payload. Exercise disabled, pose-only, and
    // combined photometric+pose states so the prefix remains compatible.
    const V3CheckpointFixture validV3Disabled =
        validV3Checkpoint(false, false);
    writeFile(validV3DisabledPath, validV3Disabled.bytes);
    validateCheckpointFile(validV3DisabledPath.string());

    const V3CheckpointFixture validV3Pose = validV3Checkpoint(false, true);
    writeFile(validV3PosePath, validV3Pose.bytes);
    validateCheckpointFile(validV3PosePath.string());

    const V3CheckpointFixture validV3Both = validV3Checkpoint(true, true);
    writeFile(validV3BothPath, validV3Both.bytes);
    validateCheckpointFile(validV3BothPath.string());

    std::vector<uint8_t> invalidPoseFlag = validV3Disabled.bytes;
    std::memcpy(invalidPoseFlag.data() + validV3Disabled.poseFlagOffset,
                &invalidFlag, sizeof(invalidFlag));
    writeFile(poseCorruptPath, invalidPoseFlag);
    CHECK(rejects(poseCorruptPath, "pose enabled flag"));

    std::vector<uint8_t> zeroPoseCameraCount = validV3Pose.bytes;
    const uint32_t zeroCount = 0;
    std::memcpy(zeroPoseCameraCount.data() + validV3Pose.cameraCountOffset,
                &zeroCount, sizeof(zeroCount));
    writeFile(poseCorruptPath, zeroPoseCameraCount);
    CHECK(rejects(poseCorruptPath, "pose camera count"));

    std::vector<uint8_t> hugePoseCameraCount = validV3Pose.bytes;
    const uint32_t tooManyCameras = 1'000'001;
    std::memcpy(hugePoseCameraCount.data() + validV3Pose.cameraCountOffset,
                &tooManyCameras, sizeof(tooManyCameras));
    writeFile(poseCorruptPath, hugePoseCameraCount);
    CHECK(rejects(poseCorruptPath, "pose camera count"));

    std::vector<uint8_t> invalidAnchor = validV3Pose.bytes;
    const uint32_t outOfRangeAnchor = 2;
    std::memcpy(invalidAnchor.data() + validV3Pose.anchorIndexOffset,
                &outOfRangeAnchor, sizeof(outOfRangeAnchor));
    writeFile(poseCorruptPath, invalidAnchor);
    CHECK(rejects(poseCorruptPath, "pose anchor index"));

    const V3CheckpointFixture duplicatePoseId = validV3Checkpoint(
        false, true, {"same", "same"});
    writeFile(poseCorruptPath, duplicatePoseId.bytes);
    CHECK(rejects(poseCorruptPath, "pose frame IDs are not unique"));

    const V3CheckpointFixture emptyPoseId = validV3Checkpoint(
        false, true, {"", "frame-1"});
    writeFile(poseCorruptPath, emptyPoseId.bytes);
    CHECK(rejects(poseCorruptPath, "pose frame ID length"));

    const V3CheckpointFixture reorderedPoseIds = validV3Checkpoint(
        true, true, {"frame-1", "frame-0"});
    writeFile(poseCorruptPath, reorderedPoseIds.bytes);
    CHECK(rejects(poseCorruptPath, "camera identities"));

    std::vector<uint8_t> maxPoseStep = validV3Pose.bytes;
    std::memcpy(maxPoseStep.data() + validV3Pose.firstPoseStepOffset +
                    sizeof(uint32_t),
                &maxPhotoCount, sizeof(maxPhotoCount));
    writeFile(poseCorruptPath, maxPoseStep);
    CHECK(rejects(poseCorruptPath, "pose step count"));

    std::vector<uint8_t> inconsistentPoseSteps = validV3Pose.bytes;
    std::memcpy(inconsistentPoseSteps.data() + validV3Pose.firstPoseStepOffset,
                &tenVisits, sizeof(tenVisits));
    std::memcpy(inconsistentPoseSteps.data() + validV3Pose.firstPoseStepOffset +
                    sizeof(uint32_t),
                &tenVisits, sizeof(tenVisits));
    writeFile(poseCorruptPath, inconsistentPoseSteps);
    CHECK(rejects(poseCorruptPath, "pose step counts"));

    std::vector<uint8_t> updatedPoseAnchor = validV3Pose.bytes;
    const uint32_t oneAnchorVisit = 1;
    std::memcpy(updatedPoseAnchor.data() + validV3Pose.firstPoseStepOffset,
                &oneAnchorVisit, sizeof(oneAnchorVisit));
    writeFile(poseCorruptPath, updatedPoseAnchor);
    CHECK(rejects(poseCorruptPath, "pose anchor step count"));

    std::vector<uint8_t> badBasePoseShape = validV3Pose.bytes;
    const int64_t badBasePoseWidth = 15;
    std::memcpy(badBasePoseShape.data() +
                    validV3Pose.basePosesByteCountOffset - sizeof(int64_t),
                &badBasePoseWidth, sizeof(badBasePoseWidth));
    writeFile(poseCorruptPath, badBasePoseShape);
    CHECK(rejects(poseCorruptPath, "incompatible shape"));

    std::vector<uint8_t> badPoseDeltaShape = validV3Pose.bytes;
    const int64_t badPoseDeltaWidth = 7;
    std::memcpy(badPoseDeltaShape.data() +
                    validV3Pose.poseDeltaByteCountOffset - sizeof(int64_t),
                &badPoseDeltaWidth, sizeof(badPoseDeltaWidth));
    writeFile(poseCorruptPath, badPoseDeltaShape);
    CHECK(rejects(poseCorruptPath, "incompatible shape"));

    std::vector<uint8_t> badPoseMomentBytes = validV3Pose.bytes;
    uint64_t poseMomentBytes = 0;
    std::memcpy(&poseMomentBytes,
                badPoseMomentBytes.data() +
                    validV3Pose.secondMomentByteCountOffset,
                sizeof(poseMomentBytes));
    ++poseMomentBytes;
    std::memcpy(badPoseMomentBytes.data() +
                    validV3Pose.secondMomentByteCountOffset,
                &poseMomentBytes, sizeof(poseMomentBytes));
    writeFile(poseCorruptPath, badPoseMomentBytes);
    CHECK(rejects(poseCorruptPath, "byte count"));

    // Reject attacker-sized counts before reserving per-camera containers.
    std::vector<uint8_t> truncatedPosePayload = validV3Disabled.bytes;
    truncatedPosePayload.resize(validV3Disabled.poseFlagOffset);
    appendScalar(truncatedPosePayload, uint32_t{1});
    appendScalar(truncatedPosePayload, uint32_t{1'000'000});
    appendScalar(truncatedPosePayload, uint32_t{0});
    writeFile(truncatedPosePayloadPath, truncatedPosePayload);
    CHECK(rejects(truncatedPosePayloadPath,
                  "truncated pose checkpoint payload"));

    std::vector<uint8_t> trailingV3 = validV3Pose.bytes;
    trailingV3.push_back(0);
    writeFile(trailingV3Path, trailingV3);
    CHECK(rejects(trailingV3Path, "trailing data"));

    // Version 4 preserves the v3 pose prefix, then records the conditioning
    // mode, one readiness byte per camera, and the exact full 6x6 matrices
    // before the existing delta and Adam tensors.
    const V4CheckpointFixture validV4 = validV4Checkpoint();
    writeFile(validV4Path, validV4.bytes);
    validateCheckpointFile(validV4Path.string());

    std::vector<uint8_t> invalidConditioning = validV4.bytes;
    const uint32_t invalidConditioningMode = 2;
    std::memcpy(
        invalidConditioning.data() + validV4.conditioningOffset,
        &invalidConditioningMode, sizeof(invalidConditioningMode));
    writeFile(poseConditioningCorruptPath, invalidConditioning);
    CHECK(rejects(poseConditioningCorruptPath,
                  "require CamP conditioning"));

    std::vector<uint8_t> rawV4Conditioning = validV4.bytes;
    const uint32_t rawConditioningMode = 0;
    std::memcpy(
        rawV4Conditioning.data() + validV4.conditioningOffset,
        &rawConditioningMode, sizeof(rawConditioningMode));
    writeFile(poseConditioningCorruptPath, rawV4Conditioning);
    CHECK(rejects(poseConditioningCorruptPath,
                  "require CamP conditioning"));

    std::vector<uint8_t> invalidReadiness = validV4.bytes;
    const uint8_t invalidReadinessFlag = 2;
    std::memcpy(
        invalidReadiness.data() + validV4.firstReadinessOffset,
        &invalidReadinessFlag, sizeof(invalidReadinessFlag));
    writeFile(poseConditioningCorruptPath, invalidReadiness);
    CHECK(rejects(poseConditioningCorruptPath,
                  "preconditioner readiness flag"));

    std::vector<uint8_t> missingOptimizedPreconditioner = validV4.bytes;
    const uint8_t notReady = 0;
    std::memcpy(
        missingOptimizedPreconditioner.data() +
            validV4.firstReadinessOffset + 1,
        &notReady, sizeof(notReady));
    writeFile(poseConditioningCorruptPath,
              missingOptimizedPreconditioner);
    CHECK(rejects(poseConditioningCorruptPath,
                  "inconsistent with optimizer steps"));

    std::vector<uint8_t> badPreconditionerShape = validV4.bytes;
    const int64_t badPreconditionerWidth = 35;
    std::memcpy(
        badPreconditionerShape.data() +
            validV4.preconditionerByteCountOffset - sizeof(int64_t),
        &badPreconditionerWidth, sizeof(badPreconditionerWidth));
    writeFile(poseConditioningCorruptPath, badPreconditionerShape);
    CHECK(rejects(poseConditioningCorruptPath, "incompatible shape"));

    std::vector<uint8_t> badPreconditionerBytes = validV4.bytes;
    uint64_t preconditionerBytes = 0;
    std::memcpy(
        &preconditionerBytes,
        badPreconditionerBytes.data() +
            validV4.preconditionerByteCountOffset,
        sizeof(preconditionerBytes));
    ++preconditionerBytes;
    std::memcpy(
        badPreconditionerBytes.data() +
            validV4.preconditionerByteCountOffset,
        &preconditionerBytes, sizeof(preconditionerBytes));
    writeFile(poseConditioningCorruptPath, badPreconditionerBytes);
    CHECK(rejects(poseConditioningCorruptPath, "byte count"));

    std::vector<uint8_t> truncatedV4(
        validV4.bytes.begin(),
        validV4.bytes.begin() +
            static_cast<std::ptrdiff_t>(validV4.firstReadinessOffset + 1));
    writeFile(truncatedV4Path, truncatedV4);
    CHECK(rejects(truncatedV4Path,
                  "truncated pose checkpoint payload"));

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
