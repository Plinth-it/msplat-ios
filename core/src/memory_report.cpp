#include "memory_report.hpp"
#include "bindings.h"

#include <cstdio>
#include <cstdlib>
#include <mach/mach.h>

namespace msplat {

namespace {

size_t currentPhysFootprintBytes() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t result = task_info(mach_task_self(), TASK_VM_INFO,
                                     (task_info_t)&info, &count);
    return result == KERN_SUCCESS ? (size_t)info.phys_footprint : 0;
}

int logInterval() {
    static int interval = [] {
        if (const char *env = std::getenv("MSPLAT_MEM_LOG_EVERY")) {
            return std::atoi(env);
        }
        return 0;
    }();
    return interval;
}

double mb(size_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

} // namespace

void reportMemory(int step, int splatCount, size_t modelBytes,
                  size_t imageBytes, size_t imageBudgetBytes) {
    int every = logInterval();
    if (every <= 0) return;
    if (step != 1 && step % every != 0) return;

    size_t tempBytes = msplat_cached_tensor_bytes();
    size_t accounted = modelBytes + imageBytes + tempBytes;
    fprintf(stderr,
            "MSPLAT_MEM step=%d splats=%d phys=%.1fMB accounted=%.1fMB "
            "model=%.1fMB temp=%.1fMB images=%.1fMB imageBudget=%.1fMB\n",
            step, splatCount,
            mb(currentPhysFootprintBytes()), mb(accounted),
            mb(modelBytes), mb(tempBytes), mb(imageBytes), mb(imageBudgetBytes));
    fflush(stderr);
}

} // namespace msplat
