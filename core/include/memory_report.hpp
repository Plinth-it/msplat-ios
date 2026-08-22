#ifndef MSPLAT_MEMORY_REPORT_H
#define MSPLAT_MEMORY_REPORT_H

#include <cstddef>

namespace msplat {

/// Emits one MSPLAT_MEM line to stderr when MSPLAT_MEM_LOG_EVERY says this step
/// is due, and does nothing when that variable is unset.
///
/// The figure it leads with is phys_footprint, not RSS. RSS counts pages that
/// were freed but not yet returned to the OS, so it stays flat across a change
/// that halves live allocation — and on iOS phys_footprint is what jetsam
/// judges the process on anyway. The breakdown that follows is there because
/// the total alone never says which of the three pools moved.
void reportMemory(int step, int splatCount, size_t modelBytes,
                  size_t imageBytes, size_t imageBudgetBytes);

} // namespace msplat

#endif // MSPLAT_MEMORY_REPORT_H
