#include "msplat_c_api.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <type_traits>

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

int main() {
    CHECK(msplat_abi_version() == MSPLAT_ABI_VERSION);

    MsplatErrorInfo error{};
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
