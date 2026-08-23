#include "msplat_c_api.h"
#include "dataset_errors.hpp"
#include "input_data.hpp"
#include "loaders.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unistd.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static_assert(sizeof(MsplatSubmittedTrainingStep) == 32);
static_assert(alignof(MsplatSubmittedTrainingStep) == 4);
static_assert(sizeof(MsplatCompletedTrainingStep) == 64);
static_assert(alignof(MsplatCompletedTrainingStep) == 8);
static_assert(sizeof(MsplatTrainingMetrics) == 136);
static_assert(alignof(MsplatTrainingMetrics) == 8);
static_assert(offsetof(MsplatTrainingMetrics, submitted) == 8);
static_assert(offsetof(MsplatTrainingMetrics, completed) == 40);
static_assert(offsetof(MsplatTrainingMetrics, overflowedCompletedSteps) == 104);
static_assert(offsetof(MsplatTrainingMetrics, lastFailedIteration) == 132);
static_assert(sizeof(MsplatTrainingMemoryMetrics) == 96);
static_assert(alignof(MsplatTrainingMemoryMetrics) == 8);
static_assert(offsetof(MsplatTrainingMemoryMetrics, trainerModelBufferBytes) == 8);
static_assert(offsetof(MsplatTrainingMemoryMetrics, trainingGpuImageCacheMisses) == 88);

static_assert(sizeof(void*) == 8);
static_assert(sizeof(size_t) == 8);
static_assert(std::is_standard_layout<MsplatStringViewV5>::value);
static_assert(sizeof(MsplatStringViewV5) == 16);
static_assert(alignof(MsplatStringViewV5) == 8);
static_assert(offsetof(MsplatStringViewV5, data) == 0);
static_assert(offsetof(MsplatStringViewV5, length) == 8);

static_assert(std::is_standard_layout<MsplatCameraCalibrationV5>::value);
static_assert(sizeof(MsplatCameraCalibrationV5) == 44);
static_assert(alignof(MsplatCameraCalibrationV5) == 4);
static_assert(offsetof(MsplatCameraCalibrationV5, width) == 0);
static_assert(offsetof(MsplatCameraCalibrationV5, height) == 4);
static_assert(offsetof(MsplatCameraCalibrationV5, fx) == 8);
static_assert(offsetof(MsplatCameraCalibrationV5, fy) == 12);
static_assert(offsetof(MsplatCameraCalibrationV5, cx) == 16);
static_assert(offsetof(MsplatCameraCalibrationV5, cy) == 20);
static_assert(offsetof(MsplatCameraCalibrationV5, k1) == 24);
static_assert(offsetof(MsplatCameraCalibrationV5, k2) == 28);
static_assert(offsetof(MsplatCameraCalibrationV5, k3) == 32);
static_assert(offsetof(MsplatCameraCalibrationV5, p1) == 36);
static_assert(offsetof(MsplatCameraCalibrationV5, p2) == 40);

static_assert(std::is_standard_layout<MsplatDatasetFrameV5>::value);
static_assert(sizeof(MsplatDatasetFrameV5) == 168);
static_assert(alignof(MsplatDatasetFrameV5) == 8);
static_assert(offsetof(MsplatDatasetFrameV5, id) == 0);
static_assert(offsetof(MsplatDatasetFrameV5, calibrationId) == 16);
static_assert(offsetof(MsplatDatasetFrameV5, imagePath) == 32);
static_assert(offsetof(MsplatDatasetFrameV5, rasterOrientation) == 48);
static_assert(offsetof(MsplatDatasetFrameV5, reserved) == 52);
static_assert(offsetof(MsplatDatasetFrameV5, calibration) == 56);
static_assert(offsetof(MsplatDatasetFrameV5, cameraToWorld) == 100);

static_assert(std::is_standard_layout<MsplatSparseObservationV5>::value);
static_assert(sizeof(MsplatSparseObservationV5) == 24);
static_assert(alignof(MsplatSparseObservationV5) == 4);
static_assert(offsetof(MsplatSparseObservationV5, frameIndex) == 0);
static_assert(offsetof(MsplatSparseObservationV5, frameObservationIndex) == 4);
static_assert(offsetof(MsplatSparseObservationV5, pointIndex) == 8);
static_assert(offsetof(MsplatSparseObservationV5, reserved) == 12);
static_assert(offsetof(MsplatSparseObservationV5, x) == 16);
static_assert(offsetof(MsplatSparseObservationV5, y) == 20);

static_assert(std::is_standard_layout<MsplatDatasetDescriptorV5>::value);
static_assert(sizeof(MsplatDatasetDescriptorV5) == 144);
static_assert(alignof(MsplatDatasetDescriptorV5) == 8);
static_assert(offsetof(MsplatDatasetDescriptorV5, frames) == 0);
static_assert(offsetof(MsplatDatasetDescriptorV5, frameCount) == 8);
static_assert(offsetof(MsplatDatasetDescriptorV5, pointXYZ) == 16);
static_assert(offsetof(MsplatDatasetDescriptorV5, pointXYZCount) == 24);
static_assert(offsetof(MsplatDatasetDescriptorV5, pointRGB) == 32);
static_assert(offsetof(MsplatDatasetDescriptorV5, pointRGBCount) == 40);
static_assert(offsetof(MsplatDatasetDescriptorV5, pointSourceIds) == 48);
static_assert(offsetof(MsplatDatasetDescriptorV5, pointSourceIdCount) == 56);
static_assert(offsetof(MsplatDatasetDescriptorV5, pointReprojectionErrors) == 64);
static_assert(offsetof(MsplatDatasetDescriptorV5, pointReprojectionErrorCount) == 72);
static_assert(offsetof(MsplatDatasetDescriptorV5, observations) == 80);
static_assert(offsetof(MsplatDatasetDescriptorV5, observationCount) == 88);
static_assert(offsetof(MsplatDatasetDescriptorV5, provenanceAdapter) == 96);
static_assert(offsetof(MsplatDatasetDescriptorV5, provenanceSource) == 112);
static_assert(offsetof(MsplatDatasetDescriptorV5, reserved) == 128);
static_assert(MSPLAT_DATASET_V5_MAX_STRING_BYTES == 1048576u);
static_assert(MSPLAT_DATASET_V5_MAX_FRAMES == 1000000u);
static_assert(MSPLAT_DATASET_V5_MAX_POINTS == 100000000u);
static_assert(MSPLAT_DATASET_V5_MAX_OBSERVATIONS == 100000000u);

namespace {

template <size_t N>
MsplatStringViewV5 stringView(const char (&value)[N]) {
    return {value, N - 1};
}

void setIdentityPose(float pose[16], float x) {
    const float identity[16] = {
        1.0f, 0.0f, 0.0f, x,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    std::memcpy(pose, identity, sizeof(identity));
}

struct DescriptorFixture {
    char frameIds[2][8] = {"frame-0", "frame-1"};
    char calibrationIds[2][9] = {"camera-0", "camera-1"};
    char imagePaths[2][12] = {"image-0.png", "image-1.png"};
    char provenanceAdapter[7] = "native";
    char provenanceSource[6] = "stack";

    MsplatDatasetFrameV5 frames[2] = {};
    float pointXYZ[6] = {-1.0f, 0.0f, 2.0f, 1.0f, 0.0f, 2.0f};
    uint8_t pointRGB[6] = {255, 0, 0, 0, 255, 0};
    uint64_t pointSourceIds[2] = {42, 84};
    float pointReprojectionErrors[2] = {0.25f, 0.5f};
    MsplatSparseObservationV5 observations[3] = {};
    MsplatDatasetDescriptorV5 descriptor = {};

    DescriptorFixture() {
        for (size_t index = 0; index < 2; ++index) {
            frames[index].id = {
                frameIds[index], std::strlen(frameIds[index])};
            frames[index].calibrationId = {
                calibrationIds[index], std::strlen(calibrationIds[index])};
            frames[index].imagePath = {
                imagePaths[index], std::strlen(imagePaths[index])};
            frames[index].rasterOrientation = index == 0
                ? MSPLAT_RASTER_ORIENTATION_ENCODED_PIXELS
                : MSPLAT_RASTER_ORIENTATION_EXIF_NORMALIZED;
            frames[index].calibration = {
                640, 480, 500.0f, 501.0f, 320.0f, 240.0f,
                0.01f, -0.02f, -0.003f, 0.001f, 0.002f,
            };
            setIdentityPose(frames[index].cameraToWorld,
                            index == 0 ? -1.0f : 1.0f);
        }

        observations[0] = {0, 0, 0, 0, 10.0f, 20.0f};
        observations[1] = {0, 1, -1, 0, 30.0f, 40.0f};
        observations[2] = {1, 0, 1, 0, 15.0f, 25.0f};

        descriptor.frames = frames;
        descriptor.frameCount = 2;
        descriptor.pointXYZ = pointXYZ;
        descriptor.pointXYZCount = 6;
        descriptor.pointRGB = pointRGB;
        descriptor.pointRGBCount = 6;
        descriptor.pointSourceIds = pointSourceIds;
        descriptor.pointSourceIdCount = 2;
        descriptor.pointReprojectionErrors = pointReprojectionErrors;
        descriptor.pointReprojectionErrorCount = 2;
        descriptor.observations = observations;
        descriptor.observationCount = 3;
        descriptor.provenanceAdapter = stringView(provenanceAdapter);
        descriptor.provenanceSource = stringView(provenanceSource);
    }

    DescriptorFixture(const DescriptorFixture&) = delete;
    DescriptorFixture& operator=(const DescriptorFixture&) = delete;
};

MsplatStatus createDescriptor(const MsplatDatasetDescriptorV5* descriptor,
                              size_t descriptorSize,
                              MsplatDataset* outDataset,
                              MsplatErrorInfo* error) {
    return msplat_dataset_create_from_descriptor_v5(
        descriptor, descriptorSize, 1.0f, false, 8, outDataset, error);
}

template <typename Mutation>
bool rejectsDescriptorAs(MsplatStatus expected, Mutation mutation) {
    DescriptorFixture fixture;
    mutation(fixture);

    MsplatDataset dataset = reinterpret_cast<MsplatDataset>(uintptr_t{1});
    MsplatErrorInfo error{};
    const MsplatStatus status = createDescriptor(
        &fixture.descriptor, sizeof(fixture.descriptor), &dataset, &error);
    return status == expected && dataset == nullptr && error.status == status &&
           error.message[0] != '\0';
}

bool createsAndDestroysDescriptor(DescriptorFixture &fixture);

template <size_t N>
bool rejectsFrameIdUTF8(const uint8_t (&bytes)[N]) {
    return rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [&](auto &value) {
        value.frames[0].id = {
            reinterpret_cast<const char*>(bytes), N};
    });
}

template <size_t N>
bool acceptsFrameIdUTF8(const uint8_t (&bytes)[N]) {
    DescriptorFixture fixture;
    fixture.frames[0].id = {
        reinterpret_cast<const char*>(bytes), N};
    return createsAndDestroysDescriptor(fixture);
}

template <typename Exception, typename Operation>
bool throwsExactly(Operation operation) {
    try {
        operation();
    } catch (const Exception&) {
        return true;
    } catch (...) {
    }
    return false;
}

struct CorruptImageFixture {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("msplat-c-api-corrupt-" + std::to_string(getpid()) + ".img");

    CorruptImageFixture() {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream << "not an image";
        if (!stream) throw std::runtime_error("Could not write corrupt image fixture");
    }

    ~CorruptImageFixture() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

bool createsAndDestroysDescriptor(DescriptorFixture &fixture) {
    MsplatDataset dataset = nullptr;
    MsplatErrorInfo error{};
    if (createDescriptor(&fixture.descriptor, sizeof(fixture.descriptor),
                         &dataset, &error) != MSPLAT_STATUS_OK ||
        dataset == nullptr || error.status != MSPLAT_STATUS_OK) {
        return false;
    }

    int trainCount = 0;
    const bool valid =
        msplat_dataset_num_train_v2(dataset, &trainCount, &error) ==
            MSPLAT_STATUS_OK &&
        trainCount == 2;
    const bool destroyed =
        msplat_dataset_destroy_v2(dataset, &error) == MSPLAT_STATUS_OK;
    return valid && destroyed;
}

MsplatStatus createFromStackAndOverwriteInputs(MsplatDataset *outDataset,
                                                MsplatErrorInfo *error) {
    DescriptorFixture fixture;
    const MsplatStatus status = createDescriptor(
        &fixture.descriptor, sizeof(fixture.descriptor), outDataset, error);
    if (status != MSPLAT_STATUS_OK) {
        return status;
    }

    std::memset(fixture.frameIds, 'x', sizeof(fixture.frameIds));
    std::memset(fixture.calibrationIds, 'x', sizeof(fixture.calibrationIds));
    std::memset(fixture.imagePaths, 'x', sizeof(fixture.imagePaths));
    std::memset(fixture.provenanceAdapter, 'x',
                sizeof(fixture.provenanceAdapter));
    std::memset(fixture.provenanceSource, 'x',
                sizeof(fixture.provenanceSource));
    for (float &value : fixture.pointXYZ) value = 12345.0f;
    for (uint8_t &value : fixture.pointRGB) value = 7;
    for (uint64_t &value : fixture.pointSourceIds) value = 999;
    for (float &value : fixture.pointReprojectionErrors) value = -1.0f;
    for (MsplatSparseObservationV5 &observation : fixture.observations) {
        observation.frameIndex = 99;
        observation.pointIndex = 99;
    }
    setIdentityPose(fixture.frames[0].cameraToWorld, 12345.0f);
    fixture.descriptor = {};
    return status;
}

} // namespace

int main() {
    CHECK(msplat_abi_version() == MSPLAT_ABI_VERSION);

    MsplatErrorInfo error{};

    // ABI v5 accepts stack-backed storage and synchronously deep-copies it.
    MsplatDataset copiedDataset = nullptr;
    CHECK(createFromStackAndOverwriteInputs(&copiedDataset, &error) ==
          MSPLAT_STATUS_OK);
    CHECK(copiedDataset != nullptr);
    int copiedTrainCount = 0;
    CHECK(msplat_dataset_num_train_v2(
              copiedDataset, &copiedTrainCount, &error) == MSPLAT_STATUS_OK);
    CHECK(copiedTrainCount == 2);
    float copiedPose[16] = {};
    CHECK(msplat_dataset_camera_pose_v2(
              copiedDataset, 0, copiedPose, &error) == MSPLAT_STATUS_OK);
    CHECK(copiedPose[0] == 1.0f);
    CHECK(copiedPose[5] == 1.0f);
    CHECK(copiedPose[10] == 1.0f);
    CHECK(copiedPose[15] == 1.0f);
    CHECK(copiedPose[3] == -1.0f);
    CHECK(msplat_dataset_destroy_v2(copiedDataset, &error) ==
          MSPLAT_STATUS_OK);

    // Optional metadata and correspondence arrays accept either a complete
    // pointer/count pair or the canonical absent representation.
    DescriptorFixture presentOptionalArrays;
    CHECK(createsAndDestroysDescriptor(presentOptionalArrays));
    DescriptorFixture absentOptionalArrays;
    absentOptionalArrays.descriptor.pointSourceIds = nullptr;
    absentOptionalArrays.descriptor.pointSourceIdCount = 0;
    absentOptionalArrays.descriptor.pointReprojectionErrors = nullptr;
    absentOptionalArrays.descriptor.pointReprojectionErrorCount = 0;
    absentOptionalArrays.descriptor.observations = nullptr;
    absentOptionalArrays.descriptor.observationCount = 0;
    CHECK(createsAndDestroysDescriptor(absentOptionalArrays));

    DescriptorFixture descriptorSizeFixture;
    MsplatDataset descriptorDataset =
        reinterpret_cast<MsplatDataset>(uintptr_t{1});
    CHECK(createDescriptor(
              &descriptorSizeFixture.descriptor,
              sizeof(descriptorSizeFixture.descriptor) - 1,
              &descriptorDataset, &error) == MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(descriptorDataset == nullptr);
    CHECK(error.status == MSPLAT_STATUS_INVALID_ARGUMENT);
    descriptorDataset = reinterpret_cast<MsplatDataset>(uintptr_t{1});
    CHECK(createDescriptor(
              &descriptorSizeFixture.descriptor,
              sizeof(descriptorSizeFixture.descriptor) + 1,
              &descriptorDataset, &error) == MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(descriptorDataset == nullptr);
    CHECK(error.status == MSPLAT_STATUS_INVALID_ARGUMENT);

    descriptorDataset = reinterpret_cast<MsplatDataset>(uintptr_t{1});
    CHECK(createDescriptor(nullptr, sizeof(MsplatDatasetDescriptorV5),
                           &descriptorDataset, &error) ==
          MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(descriptorDataset == nullptr);
    CHECK(createDescriptor(&descriptorSizeFixture.descriptor,
                           sizeof(descriptorSizeFixture.descriptor),
                           nullptr, &error) ==
          MSPLAT_STATUS_INVALID_ARGUMENT);

    // Pointer/count coherence is an outer ABI representation requirement.
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.frames = nullptr;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.frameCount = 0;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.pointXYZ = nullptr;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.pointXYZCount = 0;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.pointRGB = nullptr;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.pointRGBCount = 0;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.pointSourceIds = nullptr;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.pointSourceIdCount = 0;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.pointReprojectionErrors = nullptr;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.pointReprojectionErrorCount = 0;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.observations = nullptr;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.observationCount = 0;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.frames[0].id.data = nullptr;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.frames[0].id.length = 0;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.frameIds[0][1] = '\0';
    }));

    // Every string view is strict RFC 3629 UTF-8, not merely a byte string.
    const uint8_t overlongEncoding[] = {0xc0, 0xaf};
    const uint8_t surrogateEncoding[] = {0xed, 0xa0, 0x80};
    const uint8_t truncatedEncoding[] = {0xe2, 0x82};
    const uint8_t aboveUnicodeMaximum[] = {0xf4, 0x90, 0x80, 0x80};
    CHECK(rejectsFrameIdUTF8(overlongEncoding));
    CHECK(rejectsFrameIdUTF8(surrogateEncoding));
    CHECK(rejectsFrameIdUTF8(truncatedEncoding));
    CHECK(rejectsFrameIdUTF8(aboveUnicodeMaximum));
    const uint8_t maximumUnicodeScalar[] = {0xf4, 0x8f, 0xbf, 0xbf};
    CHECK(acceptsFrameIdUTF8(maximumUnicodeScalar));

    // Exported limits are enforced before native code reads beyond the first
    // caller-provided element.
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.frames[0].id.length =
            static_cast<size_t>(MSPLAT_DATASET_V5_MAX_STRING_BYTES) + 1;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.frameCount =
            static_cast<size_t>(MSPLAT_DATASET_V5_MAX_FRAMES) + 1;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.pointXYZCount =
            static_cast<size_t>(MSPLAT_DATASET_V5_MAX_POINTS) * 3 + 1;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.pointSourceIdCount =
            static_cast<size_t>(MSPLAT_DATASET_V5_MAX_POINTS) + 1;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.observationCount =
            static_cast<size_t>(MSPLAT_DATASET_V5_MAX_OBSERVATIONS) + 1;
    }));

    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.descriptor.reserved[0] = 1;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.frames[0].reserved = 1;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.observations[0].reserved = 1;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_ARGUMENT, [](auto &value) {
        value.frames[0].rasterOrientation = 2;
    }));

    // Once the representation is copied, canonical descriptor failures are
    // dataset errors rather than caller ABI errors.
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.frames[1].id = value.frames[0].id;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.pointSourceIds[1] = value.pointSourceIds[0];
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.frames[0].calibration.fx = 0.0f;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.frames[0].cameraToWorld[0] = 2.0f;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.pointXYZ[0] = std::numeric_limits<float>::quiet_NaN();
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.pointReprojectionErrors[0] = -1.0f;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.descriptor.pointXYZCount = 5;
        value.descriptor.pointRGBCount = 5;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.descriptor.pointSourceIdCount = 1;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.descriptor.pointReprojectionErrorCount = 1;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.observations[0].frameIndex = 2;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.observations[0].pointIndex = 2;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        const MsplatSparseObservationV5 observation = value.observations[1];
        value.observations[1] = value.observations[2];
        value.observations[2] = observation;
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.frames[0].id = {nullptr, 0};
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.frames[0].calibrationId = {nullptr, 0};
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.frames[0].imagePath = {nullptr, 0};
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.descriptor.provenanceAdapter = {nullptr, 0};
    }));
    CHECK(rejectsDescriptorAs(MSPLAT_STATUS_INVALID_DATASET, [](auto &value) {
        value.descriptor.provenanceSource = {nullptr, 0};
    }));

    // Lazy asset failures carry typed classifications through the native
    // loader. The C guarded step/evaluate seam catches these same types before
    // applying its GPU fallback status.
    const std::string missingImage =
        (std::filesystem::temp_directory_path() /
         ("msplat-c-api-missing-" + std::to_string(getpid()) + ".png"))
            .string();
    CHECK(throwsExactly<msplat::DatasetIOError>([&] {
        (void)inspectImageSource(missingImage);
    }));
    CorruptImageFixture corruptImage;
    CHECK(throwsExactly<msplat::DatasetIOError>([&] {
        (void)inspectImageSource(corruptImage.path.string());
    }));
    Camera invalidDimensions;
    invalidDimensions.filePath = missingImage;
    invalidDimensions.width = -1;
    invalidDimensions.height = 1;
    CHECK(throwsExactly<msplat::InvalidDatasetError>([&] {
        invalidDimensions.loadImage(1.0f);
    }));

    MsplatDataset dataset = reinterpret_cast<MsplatDataset>(1);
    MsplatStatus status = msplat_dataset_create_v2(
        nullptr, 1.0f, false, 8, &dataset, &error);
    CHECK(status == MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(dataset == nullptr);
    CHECK(error.status == status);
    CHECK(error.message[0] != '\0');

    int count = 99;
    status = msplat_dataset_num_train_v2(nullptr, &count, &error);
    CHECK(status == MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(count == 0);

    MsplatStats stats{1, 2, 3.0f};
    status = msplat_trainer_step_v2(nullptr, &stats, &error);
    CHECK(status == MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(stats.iteration == 0);
    CHECK(stats.splatCount == 0);
    CHECK(stats.msPerStep == 0.0f);

    MsplatTrainingMetrics trainingMetrics;
    std::memset(&trainingMetrics, 0x5a, sizeof(trainingMetrics));
    status = msplat_trainer_metrics_v4(
        nullptr, &trainingMetrics, sizeof(trainingMetrics), &error);
    CHECK(status == MSPLAT_STATUS_INVALID_ARGUMENT);
    MsplatTrainingMetrics zeroTrainingMetrics{};
    CHECK(std::memcmp(&trainingMetrics, &zeroTrainingMetrics,
                      sizeof(trainingMetrics)) == 0);

    std::memset(&trainingMetrics, 0x5a, sizeof(trainingMetrics));
    MsplatTrainingMetrics unchangedTrainingMetrics = trainingMetrics;
    status = msplat_trainer_metrics_v4(
        nullptr, &trainingMetrics, sizeof(trainingMetrics) - 1, &error);
    CHECK(status == MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(std::memcmp(&trainingMetrics, &unchangedTrainingMetrics,
                      sizeof(trainingMetrics)) == 0);
    status = msplat_trainer_metrics_v4(
        nullptr, &trainingMetrics, sizeof(trainingMetrics) + 1, &error);
    CHECK(status == MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(std::memcmp(&trainingMetrics, &unchangedTrainingMetrics,
                      sizeof(trainingMetrics)) == 0);
    status = msplat_trainer_metrics_v4(
        nullptr, nullptr, sizeof(trainingMetrics), &error);
    CHECK(status == MSPLAT_STATUS_INVALID_ARGUMENT);

    MsplatTrainingMemoryMetrics memoryMetrics;
    std::memset(&memoryMetrics, 0x5a, sizeof(memoryMetrics));
    status = msplat_trainer_memory_metrics_v4(
        nullptr, &memoryMetrics, sizeof(memoryMetrics), &error);
    CHECK(status == MSPLAT_STATUS_INVALID_ARGUMENT);
    MsplatTrainingMemoryMetrics zeroMemoryMetrics{};
    CHECK(std::memcmp(&memoryMetrics, &zeroMemoryMetrics,
                      sizeof(memoryMetrics)) == 0);

    std::memset(&memoryMetrics, 0x5a, sizeof(memoryMetrics));
    MsplatTrainingMemoryMetrics unchangedMemoryMetrics = memoryMetrics;
    status = msplat_trainer_memory_metrics_v4(
        nullptr, &memoryMetrics, sizeof(memoryMetrics) - 1, &error);
    CHECK(status == MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(std::memcmp(&memoryMetrics, &unchangedMemoryMetrics,
                      sizeof(memoryMetrics)) == 0);
    status = msplat_trainer_memory_metrics_v4(
        nullptr, &memoryMetrics, sizeof(memoryMetrics) + 1, &error);
    CHECK(status == MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(std::memcmp(&memoryMetrics, &unchangedMemoryMetrics,
                      sizeof(memoryMetrics)) == 0);
    status = msplat_trainer_memory_metrics_v4(
        nullptr, nullptr, sizeof(memoryMetrics), &error);
    CHECK(status == MSPLAT_STATUS_INVALID_ARGUMENT);

    status = msplat_set_metallib_path_v2("", &error);
    CHECK(status == MSPLAT_STATUS_INVALID_ARGUMENT);

    MsplatConfig config = msplat_default_config();
    CHECK(msplat_config_validate_v2(&config, sizeof(config), &error) ==
          MSPLAT_STATUS_OK);
    config.resolutionSchedule = 0;
    CHECK(msplat_config_validate_v2(&config, sizeof(config), &error) ==
          MSPLAT_STATUS_INVALID_ARGUMENT);
    config = msplat_default_config();
    config.refineEvery = 0;
    CHECK(msplat_config_validate_v2(&config, sizeof(config), &error) ==
          MSPLAT_STATUS_INVALID_ARGUMENT);
    config = msplat_default_config();
    config.shDegreeInterval = 0;
    CHECK(msplat_config_validate_v2(&config, sizeof(config), &error) ==
          MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(msplat_config_validate_v2(&config, sizeof(config) - 1, &error) ==
          MSPLAT_STATUS_INVALID_ARGUMENT);

    MsplatTrainingLimits limits = msplat_default_training_limits();
    CHECK(limits.maxGaussians == -1);
    CHECK(msplat_training_limits_validate_v3(&limits, sizeof(limits), &error) ==
          MSPLAT_STATUS_OK);
    limits.maxGaussians = 750000;
    CHECK(msplat_training_limits_validate_v3(&limits, sizeof(limits), &error) ==
          MSPLAT_STATUS_OK);
    limits.maxGaussians = 0;
    CHECK(msplat_training_limits_validate_v3(&limits, sizeof(limits), &error) ==
          MSPLAT_STATUS_INVALID_ARGUMENT);
    limits.maxGaussians = -2;
    CHECK(msplat_training_limits_validate_v3(&limits, sizeof(limits), &error) ==
          MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(msplat_training_limits_validate_v3(&limits, sizeof(limits) - 1, &error) ==
          MSPLAT_STATUS_INVALID_ARGUMENT);

    MsplatTrainer trainer = reinterpret_cast<MsplatTrainer>(1);
    config = msplat_default_config();
    limits = msplat_default_training_limits();
    CHECK(msplat_trainer_create_v3(
              nullptr, &config, sizeof(config), &limits, sizeof(limits),
              &trainer, &error) == MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(trainer == nullptr);

    // Legacy entry points route through the same exception boundary.
    CHECK(msplat_dataset_create(nullptr, 1.0f, false, 8) == nullptr);
    CHECK(msplat_last_status() == MSPLAT_STATUS_INVALID_ARGUMENT);
    CHECK(std::strlen(msplat_last_error_message()) > 0);

    MsplatPixelBuffer buffer{};
    buffer.data = static_cast<float*>(std::malloc(sizeof(float)));
    buffer.width = 1;
    buffer.height = 1;
    msplat_pixel_buffer_free(&buffer);
    CHECK(buffer.data == nullptr);
    CHECK(buffer.width == 0);
    CHECK(buffer.height == 0);

    // Failed Metal initialization is recoverable and retryable; neither call
    // may abort or let an exception cross the C boundary.
    status = msplat_set_metallib_path_v2(
        "/path-that-does-not-exist/msplat/default.metallib", &error);
    CHECK(status == MSPLAT_STATUS_OK);
    status = msplat_sync_v2(&error);
    CHECK(status == MSPLAT_STATUS_GPU_ERROR);
    CHECK(error.message[0] != '\0');
    status = msplat_sync_v2(&error);
    CHECK(status == MSPLAT_STATUS_GPU_ERROR);

    status = msplat_cleanup_v2(&error);
    CHECK(status == MSPLAT_STATUS_OK);
    return 0;
}
