#ifndef SSIM_H
#define SSIM_H

#include <cmath>
#include <cstdint>
#include <vector>
#include "metal_tensor.hpp"

// CPU image metrics and SSIM window creation.
// Training uses Metal kernels (ssim_h/v_fwd/bwd) directly.

// Images are (H, W, 3) float32 in [0,1]. An optional coverage mask is
// (H, W) uint8. Coverage weights metric centers without changing the pixels
// sampled by an SSIM window. A zero coverageUnits value asks the CPU helper to
// calculate and validate the denominator from the mask.
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
