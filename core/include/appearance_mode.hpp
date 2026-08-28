#pragma once

#include <cstdint>

namespace msplat {

/// Appearance compensation applied only while evaluating the training loss.
/// Canonical rendering, evaluation, and Gaussian export remain unchanged.
enum class AppearanceMode : uint32_t {
    None = 0,
    RgbGains = 1,
    PPISP = 2,
};

} // namespace msplat
