#include "msplat_c_api.h"

#include <cstdlib>
#include <cstring>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

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
