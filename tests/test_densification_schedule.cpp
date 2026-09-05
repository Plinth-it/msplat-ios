#include "densification_schedule.hpp"

#include <climits>
#include <iostream>
#include <stdexcept>

#define CHECK(condition) do { if (!(condition)) { \
    std::cerr << "line " << __LINE__ << ": " #condition "\n"; return 1; \
} } while (false)

int main() {
    using msplat::densificationSchedule;
    const auto longRun = densificationSchedule(30000, 2799, 100, 500, 30, 15000);
    CHECK(longRun.firstStep == 2900);
    CHECK(longRun.eventCount == 5);
    const auto noLongGrowth = densificationSchedule(30000, 2800, 100, 500, 30, 15000);
    CHECK(noLongGrowth.firstStep == 0 && noLongGrowth.eventCount == 0);
    const auto shortRun = densificationSchedule(2000, 799, 100, 500, 30, 1000);
    CHECK(shortRun.firstStep == 900 && shortRun.eventCount == 1);
    CHECK(densificationSchedule(2000, 800, 100, 500, 30, 1000).eventCount == 0);
    CHECK(densificationSchedule(2000, 100, 100, 500, 30, 1000).eventCount == 4);
    // An explicit cutoff can include the final iteration, but never its own step.
    CHECK(densificationSchedule(900, 799, 100, 500, 30, 900).eventCount == 0);
    CHECK(densificationSchedule(900, 799, 100, 500, 30, 901).eventCount == 1);
    CHECK(densificationSchedule(2000, 100, 100, 500, 30, 0).eventCount == 0);
    CHECK(densificationSchedule(2000, 100, 100, 500, 30, 1).eventCount == 0);
    CHECK(densificationSchedule(2000, 100, 100, 1000, 30, 1000).eventCount == 0);
    // No signed overflow when configuration or camera counts approach int32 limits.
    CHECK(densificationSchedule(2000, INT_MAX, 100, 0, 30, INT_MAX).eventCount == 0);
    CHECK(densificationSchedule(2000, 1, INT_MAX, 0, INT_MAX, INT_MAX).eventCount == 0);
    bool rejected = false;
    try { (void)densificationSchedule(2000, 0, 100, 500, 30, 1000); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    return 0;
}
