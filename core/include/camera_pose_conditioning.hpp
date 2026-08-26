#pragma once

#include <cstdint>

namespace msplat {

/// Coordinate conditioning for the optional camera-pose optimizer.
/// Raw preserves the established bounded SE(3) update exactly. CamP applies a
/// fixed per-camera projection-Jacobian preconditioner around that update.
enum class CameraPoseConditioning : uint32_t {
    Raw = 0,
    CamP = 1,
};

} // namespace msplat
