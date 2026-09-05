#ifndef SSIM_H
#define SSIM_H

#include <cmath>
#include <cstdint>
#include <vector>
#include "metal_tensor.hpp"

// CPU image metrics and SSIM window creation.
// Training uses Metal kernels (ssim_h/v_fwd/bwd) directly.

// Build the float32 RGB target used by transparent training. Coverage may be
// standalone UInt8 or packed target alpha. Evaluate this result without a mask
// so that exterior pixels participate in the metrics too.
MTensor composite_metric_target(const MTensor& gt, const MTensor& coverageMask,
                                const float background[3],
                                uint64_t coverageUnits = 0);

// Rendered images are (H, W, 3) float32 in [0,1]. Targets may be matching
// float32 RGB or compact uint8 RGBA; alpha is never sampled as color. An
// optional coverage mask is either a distinct (H, W) uint8 tensor or the RGBA
// target itself, in which case coverage comes from alpha. Coverage weights
// metric centers without changing the pixels sampled by an SSIM window. A zero
// coverageUnits value asks the CPU helper to calculate and validate the
// denominator from the mask.
float psnr(const MTensor& rendered, const MTensor& gt,
           const MTensor* coverageMask = nullptr,
           uint64_t coverageUnits = 0);
float l1_loss(const MTensor& rendered, const MTensor& gt,
              const MTensor* coverageMask = nullptr,
              uint64_t coverageUnits = 0);

// Create 11x11 Gaussian window for Metal SSIM loss kernel.
// Returns flat float vector (121 elements).
std::vector<float> createSSIMWindow(int windowSize = 11, float sigma = 1.5f);

// CPU SSIM evaluation for metrics (separable Gaussian blur).
float ssim_eval(const MTensor& rendered, const MTensor& gt,
                int windowSize = 11, float sigma = 1.5f);
float ssim_eval(const MTensor& rendered, const MTensor& gt,
                const MTensor* coverageMask, uint64_t coverageUnits,
                int windowSize = 11, float sigma = 1.5f);

#endif
