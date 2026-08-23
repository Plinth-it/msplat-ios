# msplat-ios

3D Gaussian Splatting training on iPhone. A fork of
[rayanht/msplat](https://github.com/rayanht/msplat), which built the engine:
the full training pipeline as fused Metal compute kernels, with no dependency
beyond system frameworks.

This fork reduces model and transient growth, adds a byte-budgeted image cache,
enforces an optional hard Gaussian population limit, fixes correctness bugs,
and provides an iOS build plus a COLMAP-to-PLY example app. Its Swift
`TrainingPlan` also validates resolution stages and derives a conservative
peak-memory estimate before a session is created. ABI v4 telemetry keeps CPU
submission progress separate from completed GPU work and exposes categorized
runtime memory and rasterizer overflow state. ABI v5 lets Plinth pass a
canonical, caller-owned dataset directly through a checked deep-copy boundary.

## Memory

| | Before | After | Measured on |
|---|---|---|---|
| Model buffers | 979.1 MB | 239.7 MB | garden, 2000 iters, 2 downscales |
| Transient buffers | 561.4 MB | 353.5 MB (pre cache split) | garden, 4 resolution levels, step 600+ |
| Training images | whole dataset | 512 MB budget (iOS default) | `MSPLAT_IMAGE_CACHE_MB` |

- Densification reserved `3 × num_active` for the case where every gaussian
  splits. The real population after a refine step is about 1.2× the count
  before it. Classification now runs first, so the grow asks for what will be
  written.
- Densification flags, prefixes, compact scratch, random samples, and gradient
  statistics are released after topology growth stops. They are never
  allocated for a cutoff at the first step, including checkpoint resumes that
  have already crossed the boundary.
- Depth-chunk buffers were carried to the end of training after chunking turned
  off. They are released.
- Render-only calls retain just the shared forward cache. Loss, SSIM, and
  backward workspaces are allocated on the first training step, and unused
  prefix, spherical-harmonics gradient, and SSIM-window buffers were removed.
- The active SSIM derivative workspace stores only its nine live FP32 values,
  and the final pass overwrites rendered RGB with its gradient. This removes
  36 bytes per pixel from the training cache (23.73 MiB at 960×720) without
  reducing numerical precision.
- Training images were all decoded up front. A byte-budgeted LRU holds what fits
  and reloads the rest.
- `maxGaussians` bounds the active population and its backing buffers. When a
  densification step has more eligible candidates than remaining capacity, the
  highest normalized-gradient candidates are retained; pruning still runs at
  the limit. Oversized initial models, PLY imports, and checkpoints are rejected
  before their Gaussian storage is allocated.

`TrainingPlan` derives its estimate from the selected Gaussian ceiling, SH
degree, source dimensions, resolution stages, an estimated exact-intersection
arena, native image cache, and an additional headroom allowance. The
intersection estimate scales a padded 16-intersections-per-Gaussian baseline
at 960×720 by tile count. That baseline is above the measured 10.4 at 960×720
and scales to 64 at 1920×1440, above the measured 43.5. The runtime still counts
and sizes each frame exactly. ImageIO is asked to decode a
thumbnail at the selected input resolution, and the app-owned decode allowance
therefore scales with that target. It is intentionally conservative, but it is
not a jetsam guarantee: codec-private surfaces, Metal driver state, framework
allocations, other process memory, and changing system pressure are outside the
model. A valid `MSPLAT_IMAGE_CACHE_MB` override is reflected automatically; use
`memoryEstimate(imageCacheBudgetBytes:)` to evaluate another budget explicitly.
The model term remains the topology-enabled peak; runtime model storage falls
after the densification cutoff.

## Correctness

**Rasterizer intersections.** The packed buffers were once sized
`num_points × 16`, and later used fixed 2,048-entry bins per tile. Both designs
could drop intersections or write beyond their useful capacity. The current
pipeline projects and counts every tile intersection, waits for that count,
builds checked exact offsets, grows compact arenas, and sorts the complete range
with a bitonic fast path through 2,048 entries and a deterministic radix path
through the explicit 65,536-per-tile work limit. Index, allocation, or work-limit
failures stop the step before rasterization and optimizer work instead of
training against an incomplete frame.

**Chunk buffers across a resolution change.** The guard compared against a
dimension already updated earlier in the same step, so kernels addressed
6056760 elements of a 5749380-element buffer. PSNR 20.56 → 21.68.

**Non-idempotent intrinsics.** `loadImage` rewrote focal lengths and distortion
coefficients in place, zeroing the coefficients after undistorting — harmless
until a cache evicts and reloads. Intrinsics are re-derived from what the
dataset declared on every call.

**Raster geometry.** Calibration and sparse observations use image-edge
coordinates, where the upper-left pixel center is `(0.5, 0.5)`; CPU image
arrays and Metal rasterization use zero-based sample indices. Resize,
Brown-Conrady rectification, crop offsets, and renderer projection cross that
half-pixel boundary explicitly. Descriptor observations remain in immutable
source-raster coordinates after lazy camera loading changes the effective
training raster.

**Image orientation and color.** COLMAP extracts features in encoded raster
coordinates with EXIF reorientation disabled. Its adapter therefore validates
EXIF orientation but deliberately preserves the raw pixel frame; applying a
display transform without updating intrinsics and poses would corrupt the
calibration. ImageIO converts decoded thumbnails into an explicit sRGB canvas.
The canonical dataset descriptor records that pixel-frame choice explicitly;
existing file adapters use encoded pixels, while calibration-aware native
adapters can opt into tested EXIF-normalized materialization. Low-level decode
tests cover all eight EXIF pixel transforms; camera loading rejects mirrored
orientations 2, 4, 5, and 7 because a positive-focal, right-handed camera
cannot represent that reflection. Normalize such assets before describing
them, or preserve their encoded raster.

## Additions

- COLMAP text models (`cameras.txt` / `images.txt` / `points3D.txt`)
- `.spz` export
- `--stop-densify-at`, stopping topology growth after a chosen step
- `--max-gaussians`, a hard Gaussian population and backing-buffer ceiling
  (`-1` keeps the legacy unlimited behavior)
- ABI v3 checked trainer creation with a size-validated
  `MsplatTrainingLimits` structure; ABI v2 creation remains available and
  unlimited
- ABI v4 query-only training and memory snapshots. Existing step APIs and
  `MsplatStats` retain their submission-only behavior and layout.
- ABI v5 checked canonical-dataset creation with synchronous ownership of
  frame calibration, image paths, sparse points, observations, and provenance;
  the existing folder-based ABI v2 entry point remains available.
- Swift `TrainingPlan` validation, resolved per-stage dimensions, and a
  code-derived peak-memory estimate
- Target-resolution ImageIO thumbnail decoding with checked dimensions,
  explicit sRGB conversion, and raw-coordinate EXIF handling for COLMAP
- A validated canonical descriptor shared by the COLMAP, Nerfstudio, and
  Polycam adapters, preserving source frame/calibration identity, sparse-point
  IDs, reprojection errors, image observations, and adapter provenance; COLMAP
  track reciprocity is checked before the descriptor is materialized
- Polycam raw exports read the documented row-major `t_00...t_23` ARKit
  camera-to-world transform directly, preferring a complete corrected
  camera/image pair per frame while falling back to that frame's raw pair when
  optimization output is partial
- XCFramework slices for `macos-arm64`, `ios-arm64` and
  `ios-arm64_x86_64-simulator`, each with its own metallib

## Quick start

```sh
./scripts/build-xcframework.sh
open examples/ios/MsplatExample.xcodeproj
```

See [examples/ios/README.md](examples/ios/README.md).

Swift currently uses the locally built `MsplatCore.xcframework`: add the
package at `msplat-ios/swift` after running the build script above. Tagged
remote SwiftPM distribution still needs a release URL and checksum.

```swift
import Foundation
import Msplat

@MsplatRuntimeActor
func train() throws {
    let plan = try TrainingPlan(
        // Maximum source dimensions, before input decoding.
        inputDimensions: TrainingImageDimensions(width: 4_032, height: 3_024),
        inputDecodeScale: 2,
        iterationBudget: 2_000,
        stages: [
            try TrainingResolutionStage(
                iterations: 1...1_000,
                downscaleFactor: 2
            ),
            try TrainingResolutionStage(
                iterations: 1_001...2_000,
                downscaleFactor: 1
            ),
        ],
        targetSHDegree: 2,
        maximumGaussianCount: 400_000
    )

    print("Conservative peak: \(plan.estimatedPeakMemory / 1_048_576) MiB")

    var baseConfig = TrainingConfig()
    baseConfig.stopDensifyAt = 750

    let session = try MsplatSession(
        datasetURL: URL(fileURLWithPath: "path/to/colmap/"),
        trainingPlan: plan,
        baseConfig: baseConfig
    )
    defer { try? session.close() }

    for _ in 0..<2_000 {
        let stats = try session.step()
        print("step=\(stats.iteration) splats=\(stats.splatCount)")
    }
    let telemetry = try session.trainingMetrics()
    if let completed = telemetry.completed,
       let gpuMs = completed.gpuExecutionMs {
        print("GPU completed step \(completed.iteration): \(gpuMs) ms")
    }
    try session.exportPLY(to: URL(fileURLWithPath: "output.ply"))
}
```

`MsplatSession` is the checked, throwing API and serializes the engine's
process-global Metal state. The legacy `GaussianDataset` and `GaussianTrainer`
types remain available for source compatibility.

Plinth can also construct a Swift `DatasetDescriptor` from calibrated frame
URLs, optional per-frame soft training masks, sparse points, and optional
observations, then pass it to
`MsplatSession(dataset:securityScopedResourceURLs:options:config:)`. The native
ABI v5/v6 path copies every descriptor and mask-sidecar buffer synchronously.
Luminance masks use premultiplied Rec. 709 coverage; alpha masks require an
alpha channel. Both remain soft UInt8 coverage through resize and
undistortion, must match the image in its selected encoded or EXIF-normalized
raster frame, and normalize training loss over covered RGB units. Callers
should provide the selected folder or bookmark roots whose security scopes
cover every lazily decoded image and mask URL; the session releases those
scopes after its native trainer and dataset are destroyed. Set
`TrainingPlan.includesTrainingMasks` through its initializer when planning a
masked dataset so the estimate includes full-source mask decoding and paired
CPU/GPU mask caches.

Before creating a session, a canonical descriptor can be checked against its
sparse correspondences without reading any image or mask and without starting
Metal:

```swift
let capture = try descriptor.captureDiagnostics()
if let residual = capture.reprojectionError {
    print("Reprojection RMS: \(residual.rootMeanSquarePixels) px")
}
```

The report keeps recomputed observation residuals separate from optional
source-reported point errors. Aggregate and per-frame counts also identify
unlinked observations, points behind the declared cameras, non-finite
projections, and observed or predicted coordinates outside the raster. Track
lengths count distinct observing frames per observed point; msplat deliberately
supplies no subjective pass/fail threshold, leaving capture policy to the
integrating app.

`step()` returns a submission receipt: its `cpuSubmitMs` measures active CPU
encoding and submission time with required synchronous GPU waits excluded.
`trainingMetrics()` is a non-draining poll whose submitted and completed
descriptors may name different iterations. A completed descriptor is published
only after every command buffer in that logical step succeeds;
`gpuExecutionMs` sums their Metal GPU intervals, while `endToEndMs` is
completion-observed wall latency, including queueing, completion-handler
scheduling, and any required synchronous readbacks.
`memoryMetrics()` separates model, render-transient, training-transient,
telemetry-readback, and image-cache bytes from process `phys_footprint` and iOS
available memory. The buffer categories are logical allocations, not a claim
about physical residency.

CLI:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/msplat path/to/dataset -n 7000 --max-gaussians 400000 --eval
```

## Reference

Input: COLMAP (binary or text), Nerfstudio, Polycam.
Output: `.ply`, `.splat`, `.spz`.

| Variable | Effect |
|---|---|
| `MSPLAT_IMAGE_CACHE_MB` | Image cache budget. Default 512 on iOS, 2048 elsewhere. |
| `MSPLAT_MEM_LOG_EVERY` | Memory breakdown every N steps. |
| `MSPLAT_ISECT_LOG` | Intersection count against capacity at each sample. |

```
MSPLAT_MEM step=1000 splats=232313 phys=2908.4MB accounted=2863.5MB \
           model=239.7MB temp=576.1MB images=2047.7MB imageBudget=2048.0MB
```

`phys` is `phys_footprint`, what jetsam counts. `model` is gaussian parameters
and Adam state, `temp` the cached per-iteration buffers, `images` the decoded
training images. This captured example predates the render/training cache split;
use a device run for current footprint values.

## Building

```sh
git clone https://github.com/frs0n/msplat-ios.git && cd msplat-ios

cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j  # CLI + static lib
pip install -e .                                                     # Python module
./scripts/build-xcframework.sh                                       # XCFramework + metallibs
```

macOS 15+ and Xcode; iOS builds target 18.0. zlib, used for `.spz`, ships with
the OS.

`MsplatCore.xcframework` and the metallibs are build outputs, not committed. Run
`scripts/build-xcframework.sh` before building the Swift package or the example
app, and again after any C API change — the Swift side otherwise links a stale
header without complaint.

## Reproducibility

Training is not bit-reproducible: the rasterizer backward scatters gradients
with relaxed float atomics, so accumulation order varies between runs and
gaussians sitting exactly on `densifyGradThresh` flip. Over seven runs of garden
at 2000 iterations, PSNR fell in 26.75–26.93 and the final gaussian count in
231.6K–237.1K.

## Engine

One training step uses an exact-count command buffer followed by the remaining
work. The synchronization between them is the correctness boundary that sizes
the compact intersection arena:

```
Forward:
  project_and_sh_forward     fused projection + SH + exact per-tile counts
  CPU checked prefix         exact offsets and grow-only arena sizing
  exact scatter + sort       compact checked tile ranges + packing
  nd_rasterize_forward       per-pixel alpha compositing (16×16 tiles)
  ssim_h_fwd + fused_v_h_bwd separable 11-tap SSIM + L1 loss

Backward:
  ssim_v_bwd                 in-place final SSIM image gradient
  rasterize_backward         per-pixel backward compositing
  project_and_sh_backward    fused projection + SH VJP + SH Adam update
  fused_adam (×4 groups)     optimizer step
  accumulate_grad_stats      gradient norms for densification
```

Upstream's [README](https://github.com/rayanht/msplat) covers the design behind
this and carries M4 Max benchmarks against gsplat. This fork has not re-run
them.

## License

Apache 2.0, as upstream, whose copyright the LICENSE file carries.
