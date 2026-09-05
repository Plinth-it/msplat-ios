#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace msplat {

inline bool isDensificationStep(int64_t step, int64_t trainingCameras,
                                int64_t refineEvery, int64_t warmupLength,
                                int64_t resetInterval, int64_t stopStep) {
    return step % refineEvery == 0 && step > warmupLength &&
        step < stopStep && step % resetInterval > trainingCameras + refineEvery;
}

struct DensificationSchedule {
    int firstStep = 0; // Zero when no growth opportunity is scheduled.
    int eventCount = 0;
};

// Count opportunities, not actual splits: growth still depends on gradients
// and the Gaussian budget. stopStep is the resolved, exclusive cutoff.
inline DensificationSchedule densificationSchedule(
    int iterations, int trainingCameras, int refineEvery, int warmupLength,
    int resetAlphaEvery, int stopStep) {
    if (iterations <= 0 || trainingCameras <= 0 || refineEvery <= 0 ||
        warmupLength < 0 || resetAlphaEvery <= 0 || stopStep < 0) {
        throw std::invalid_argument("Invalid densification schedule");
    }
    const int64_t resetInterval = int64_t{resetAlphaEvery} * refineEvery;
    const int64_t lastStep = std::min<int64_t>(iterations, int64_t{stopStep} - 1);
    DensificationSchedule result;
    for (int64_t step = refineEvery; step <= lastStep; step += refineEvery) {
        if (!isDensificationStep(step, trainingCameras, refineEvery,
                                 warmupLength, resetInterval, stopStep)) continue;
        if (result.eventCount == 0) result.firstStep = static_cast<int>(step);
        ++result.eventCount;
    }
    return result;
}

} // namespace msplat
