# msplat-ios

3D Gaussian Splatting training on iPhone. A fork of
[rayanht/msplat](https://github.com/rayanht/msplat), which built the engine:
the full training pipeline as fused Metal compute kernels, with no dependency
beyond system frameworks.

This fork reduces model and transient growth, adds a byte-budgeted image cache,
enforces an optional hard Gaussian population limit, fixes correctness bugs,
and provides an iOS build plus a COLMAP/Nerfstudio-to-PLY example app. Its Swift
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
- Periodic opacity resets now clamp the active opacity logits and clear their
  Adam moments in the ordered Metal command buffer. They no longer synchronize
  the GPU for a CPU scan and two host-side buffer clears.
- Depth-chunk buffers were carried to the end of training after chunking turned
  off. They are released.
- Render-only calls retain just the shared forward cache. Loss, SSIM, and
  backward workspaces are allocated on the first training step, and unused
  prefix, spherical-harmonics gradient, and SSIM-window buffers were removed.
- The active SSIM derivative workspace stores only its nine live FP32 values,
  and the final pass overwrites rendered RGB with its gradient. This removes
  36 bytes per pixel from the training cache (23.73 MiB at 960×720) without
  reducing numerical precision.
- Training images were all decoded up front as CPU and GPU Float32 RGB copies.
  A byte-budgeted LRU now retains one tightly packed UInt8 RGBA GPU target per
  resident camera, releases decoded CPU pixels after upload, and reloads only
  on an eviction or resolution-stage transition. Loss kernels convert the RGB
  bytes to float during their existing tile loads. Masked targets reuse the
  alpha byte for soft coverage. Coverage mode additionally caches one activity
  byte per 16x16 render tile, expanded by the five-pixel SSIM halo, so exact
  intersection packing omits inactive tiles and leaves their raster bins empty.
  Transparent mode retains the full-frame path. Swift sessions can opt in with
  `DatasetOptions.prefetchTrainingTargets`; native clients can use ABI v14 or
  set exactly `MSPLAT_CAMERA_PREFETCH=1`. One detached CPU target for the next
  shuffled camera and resolution is prepared while the current Metal step runs.
  GPU upload and LRU mutation remain on the serialized training thread, and
  library prefetch remains off by default.
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
and sizes each frame exactly. The second 8-byte key arena is allocated only after
a tile exceeds the 2,048-entry bitonic path; bitonic-only resolution stages use
44 rather than 52 runtime bytes per arena slot, while the planner retains the
52-byte worst case. The synchronized layout pass also precomputes every tile
range and bucket-orders tiles with more than one entry. Exact sorting skips
empty and already-sorted single-entry tiles, uses one 32-thread group for 2-32
entries, and reserves the 256-thread path for larger ranges. ImageIO is asked
to decode a thumbnail at the selected input resolution, and the app-owned
decode allowance therefore
scales with that target. It is intentionally conservative, but it is not a
jetsam guarantee: codec-private surfaces, Metal driver state, framework
allocations, other process memory, and changing system pressure are outside the
model. A valid
`MSPLAT_IMAGE_CACHE_MB` override is reflected automatically; use
`memoryEstimate(imageCacheBudgetBytes:)` to evaluate another budget explicitly.
The model term remains the topology-enabled peak; runtime model storage falls
after the densification cutoff.

The monolithic forward rasterizer defaults to its established 8x8 threadgroup.
Native benchmark processes can set `MSPLAT_RASTER_VARIANT=16x8` or
`MSPLAT_RASTER_VARIANT=16x16` before the first Metal call to compare larger
threadgroups. These experimental overrides do not change chunked forward or
backward rasterization, which remain 8x8; select a shipping default only from
representative physical-device performance and thermal results.

Intersection attributes likewise remain packed by default. A benchmark process
can set `MSPLAT_INTERSECTION_ATTRIBUTES=gather` before the first Metal call to
skip the three sorted float3 copies and have rasterization gather XY, conic, raw
color, and the already projected sigmoid opacity through each sorted key's
Gaussian ID. This reduces the bitonic-only arena from 44 to 8 bytes per slot and
the radix arena from 52 to 16 bytes per slot, saving 36 bytes per retained
intersection (about 450 MB at a 12.5-million-slot capacity), and removes the pack
dispatch. Indirect reads may trade bandwidth capacity for locality, especially
in the fixed 8x8 backward path, so keep `packed` as the shipping mode until
physical-device throughput and sustained-thermal A/B results justify a change.

Exact tile counting likewise keeps the established per-intersection enumeration
as its default. A benchmark process can set
`MSPLAT_TILE_COUNT_MODE=difference` before the first Metal call to record four
signed corners per visible Gaussian and reconstruct the exact tile counts with
horizontal and vertical scans. The experimental mode preserves coverage-tile
masking and the synchronized CPU layout/allocation safety boundary; it does not
remove the per-step host wait. Keep `enumerated` as the shipping mode until
fixed-snapshot physical-device and sustained-thermal measurements justify a
default change.

Exact tile layout likewise remains on its established CPU implementation by
default. `MSPLAT_TILE_LAYOUT_MODE=gpu` is a correctness-first experiment that
builds the same inclusive offsets, tile bins, stable small/medium/large sort
lists, and checked summary metadata with one GPU thread. It composes with both
tile-count modes, but the host still waits for completion, validates the final
offset and metadata, and sizes the arena before submitting rasterization and
optimizer work.

`MSPLAT_TRAINING_ARENA_MODE=retry` is the next correctness-first experiment. It
forces GPU tile layout, reuses the grow-only intersection high-water mark, and
keeps the normal count/layout and training work in one GPU attempt. If the
arena, radix scratch, or raster chunks are too small, persistent model updates
are suppressed, the host grows the required resources, and the same logical
step is retried. The initial attempt at a resolution still bootstraps the arena
through the synchronized exact path, and steady-state retry mode synchronizes at
attempt retirement before committing CPU training state. It therefore removes
the mid-step sizing barrier and its GPU idle gap, but does not yet provide
multi-step queue depth. Recovered attempts remain visible as packed-capacity
overflow events, and their resource-growth and encoding costs are accumulated
into the completed logical step's telemetry. Keep `exact` as the shipping mode
until physical-device throughput, memory, and sustained-thermal measurements
justify a default change.

The separable SSIM derivative path likewise keeps its established staged mode
by default. `MSPLAT_SSIM_MODE=fused` uses a 16x8 terminal threadgroup to retain
the vertical and horizontal derivative fields in one reusable shared-memory
tile. It removes the 9-float-per-pixel derivative buffer and the final SSIM
dispatch, saving 36 bytes per training pixel plus its write/read round trip.
The public training-plan estimate continues to budget staged storage even when
fused mode is selected. Keep `staged` as the shipping mode until representative
physical-device and sustained-thermal A/B results show that the larger halo
work is a throughput win as well as a memory win.

Densification split offsets keep the established libc++ CPU normal stream by
default. `MSPLAT_DENSIFY_RANDOM_MODE=gpu` replaces the periodic host fill with
a stateless counter-based Box-Muller stream generated directly by each split
thread from the logical step and dense split ordinal. The experimental path
avoids the sample-buffer write/read traffic but deliberately retains that
capacity-sized allocation so CPU and GPU modes remain directly comparable.
Because the random sequence changes the training trajectory, keep `cpu` as the
shipping mode until fixed-checkpoint quality, densification-step timing, and
sustained physical-device thermal results justify promotion. Resume with the
same mode when comparing a continued run; the mode is process configuration,
not checkpoint metadata.

## Correctness

**Rasterizer intersections.** The packed buffers were once sized
`num_points × 16`, and later used fixed 2,048-entry bins per tile. Both designs
could drop intersections or write beyond their useful capacity. The current
pipeline projects and counts every tile intersection, waits for that count,
builds checked exact offsets, grows compact arenas, and sorts the complete range
with a 32-thread small-range path, a bitonic fast path through 2,048 entries,
and a deterministic radix path through the explicit 65,536-per-tile work limit.
Index, allocation, or work-limit failures stop the step before rasterization
and optimizer work instead of
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
Training retains those encoded numerical values as RGBA8 and normalizes them
manually; it does not use an sRGB texture conversion that would change the loss
space.
The canonical dataset descriptor records that pixel-frame choice explicitly;
existing file adapters use encoded pixels, while calibration-aware native
adapters can opt into tested EXIF-normalized materialization. Low-level decode
tests cover all eight EXIF pixel transforms; camera loading rejects mirrored
orientations 2, 4, 5, and 7 because a positive-focal, right-handed camera
cannot represent that reflection. Normalize such assets before describing
them, or preserve their encoded raster.

**Photometric refinement.** Training can optionally learn three bounded
log-domain RGB gains per canonical camera. Their mean is an exposure-like
offset and their zero-mean residual is a channel-balance correction. The input
pixels are sRGB encoded, so these are photometric gains rather than physical
linear-light exposure estimates. They affect training loss only: canonical
rendering, evaluation, and PLY/SPZ export remain unchanged. Checkpoint v2 keeps
the gains, Adam moments, per-camera visit counts, and exact frame IDs so a
resume cannot silently attach corrections to different cameras.

**Camera-pose refinement.** Training can optionally learn small, bounded
camera-space SE(3) corrections after warm-up. The first training camera is a
fixed anchor, and the geometry-only pose gradient deliberately detaches the SH
view-direction term. Imported poses and canonical rendering, evaluation, and
export remain unchanged. Checkpoint v3 preserves corrections, Adam moments,
per-camera visit counts, the anchor, exact frame IDs, and the immutable source
poses.

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
- ABI v8 opt-in per-camera photometric RGB-gain refinement, also exposed by
  Swift and the native/Python CLIs; it is disabled by default
- ABI v9 opt-in bounded camera-pose refinement, exposed by Swift and the
  native/Python CLIs; it is disabled by default and training-only
- ABI v10 opt-in mask-sidecar discovery for path-based dataset loading; the
  existing path entry point remains unmasked by default
- ABI v11 versioned training-mask treatment options. Coverage remains the
  default; transparent mode adds full-frame alpha supervision without changing
  the locked `MsplatConfig` layout or any earlier entry point.
- ABI v12 detailed count-barrier and tile-distribution telemetry through a new
  query structure; the ABI v4 telemetry layout and entry point remain unchanged.
- ABI v13 GPU-native preview submission through an immutable BGRA8Unorm Metal
  surface; existing CPU render entry points remain available.
- ABI v14 instance-scoped training-target prefetch exposed through Swift
  `DatasetOptions`; the native environment opt-in remains available.
- Swift `TrainingPlan` validation, resolved per-stage dimensions, and a
  code-derived peak-memory estimate
- Target-resolution ImageIO thumbnail decoding with checked dimensions,
  explicit sRGB conversion, compact RGBA8 training storage, and raw-coordinate
  EXIF handling for COLMAP
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
    let useTrainingMasks = true
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
        maximumGaussianCount: 400_000,
        includesTrainingMasks: useTrainingMasks
    )

    print("Conservative peak: \(plan.estimatedPeakMemory / 1_048_576) MiB")

    var baseConfig = TrainingConfig()
    baseConfig.stopDensifyAt = 750
    baseConfig.trainingMaskMode = .transparent

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

ABI v13 adds a GPU-native preview path. `MsplatSession.submitPreview(...)`
returns a `MetalPreviewSubmission`; calling `waitUntilReady()` returns a
`MetalPreviewSurface` only after its render has completed. The completed surface
owns an immutable BGRA8Unorm `MTLTexture`, so a Metal-backed view can display it
without copying pixels to the CPU or uploading them again. The texture's device
is the source of truth for the consuming Metal view and command queue, and the
surface must remain alive while the texture is displayed.

`renderRGBA` and the `PixelData` render methods remain available for callers that
need CPU-owned pixels. MsplatExample uses the Metal surface path, retains at most
one pending submission and the latest completed surface, and accounts for both
surfaces in its memory preflight. Submitting a fixed-camera preview still reaches
the renderer's existing exact-count CPU/GPU synchronization before the remaining
work is queued. The native surface avoids the later CPU readback and UIImage
re-upload, but preview submission is not fully nonblocking until tile count and
scan move entirely onto the GPU.

Path-based COLMAP loading can opt into mask discovery with
`DatasetOptions.discoverTrainingMasks`; plan-based sessions use
`TrainingPlan.includesTrainingMasks` for both discovery and their conservative
memory estimate. Both default to off. The native loader indexes regular files
below any case-insensitive `masks` path component. It matches an image such as
`images/foo.jpeg` to sidecars with the same stem (`masks/foo.png`), its full
name plus `.mask` (`masks/foo.jpeg.mask`), or a `.mask` stem
(`masks/foo.mask.png`). Nested mask directories may match a suffix of the image
directory, and frames without a match retain full coverage. An alpha-bearing
sidecar uses alpha; other color images use their first/red channel. In the
default `TrainingMaskMode.coverage`, mask value 0 excludes a pixel from RGB loss
and 255 gives it full coverage, with intermediate values providing soft
coverage. `TrainingMaskMode.transparent` instead treats those values as target
alpha. It composites source RGB over `TrainingConfig.bgColor`, supervises RGB
over the full frame, and adds `transparentAlphaLossWeight` times full-frame L1
loss between the mask and rendered alpha (`1 - transmittance`); the default
weight is 0.1. This directly penalizes exterior opacity while preserving soft
silhouette edges. The fixed configured background is used for deterministic,
race-free asynchronous training. Transparent mask treatment cannot currently be
combined with per-camera photometric gain refinement because those gains would
also act on the synthetic background. Frames without a matched mask remain
ordinary opaque RGB targets. Sidecars must currently match the source image dimensions;
unlike Brush, MSplat does not resize mismatched masks.

Plinth can also construct a Swift `DatasetDescriptor` from calibrated frame
URLs, optional per-frame soft training masks, sparse points, and optional
observations, then pass it to
`MsplatSession(dataset:securityScopedResourceURLs:options:config:)`. The native
ABI v5/v6 path copies every descriptor and mask-sidecar buffer synchronously.
Luminance masks use premultiplied Rec. 709 coverage; alpha masks require an
alpha channel. Both remain soft UInt8 masks through resize and undistortion and
must match the image in its selected encoded or EXIF-normalized raster frame.
Coverage mode normalizes over covered RGB units; transparent mode uses the
full-frame RGB and alpha objective described above. Callers
should provide the selected folder or bookmark roots whose security scopes
cover every lazily decoded image and mask URL; the session releases those
scopes after its native trainer and dataset are destroyed. Set
`TrainingPlan.includesTrainingMasks` through its initializer when planning a
masked dataset so the estimate includes full-source mask decoding and paired
image/mask decode transients; the resident GPU target packs coverage into RGBA
alpha.

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
scheduling, and any required synchronous readbacks. ABI v12 additionally
reports image preparation, exact-count GPU and wall-wait time, post-count CPU
encoding, intersection-arena growth, the exact intersection count, maximum
tile population, and trivial/small/medium/large tile counts. This makes the
per-step count barrier measurable without enabling the heavier stage profiler.
With camera prefetch enabled, decode work overlapped with the preceding Metal
step is outside the next step's `imagePrepareMs`; any remaining wait plus target
installation and GPU upload is still included.
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
| `MSPLAT_CAMERA_PREFETCH` | Set exactly `1` to predecode one upcoming training camera. Default off. |
| `MSPLAT_TILE_COUNT_MODE` | `enumerated` (default) or experimental exact `difference` counting. Set before first Metal use. |
| `MSPLAT_TILE_LAYOUT_MODE` | `cpu` (default) or experimental exact `gpu` layout. The GPU mode retains the synchronized host validation and arena-sizing boundary. Set before first Metal use. |
| `MSPLAT_TRAINING_ARENA_MODE` | `exact` (default) or experimental transactional `retry`. Retry forces GPU tile layout and replaces steady-state mid-step sizing with an end-of-attempt validation/retry boundary. Set before first Metal use. |
| `MSPLAT_SSIM_MODE` | `staged` (default) or experimental `fused` derivative processing. Fused removes the 36-byte-per-pixel derivative buffer and one dispatch. Set before first Metal use. |
| `MSPLAT_INTERSECTION_ATTRIBUTES` | `packed` (default) or experimental key-driven `gather`. Gather removes three float3 arrays and the pack dispatch, saving 36 bytes per arena slot. Set before first Metal use. |
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
  project_and_sh_forward     fused projection + SH + exact count contributions
  optional difference scans  exact per-tile counts for the experimental mode
  CPU checked prefix         exact offsets and grow-only arena sizing
  exact scatter + sort       compact checked tile ranges + optional packing
  nd_rasterize_forward       per-pixel alpha compositing (16×16 tiles)
  ssim_h_fwd + fused_v_h_bwd default staged 11-tap SSIM + L1 loss

Backward:
  ssim_v_bwd                 default staged final SSIM image gradient
  optional fused SSIM        folds both derivative passes into the loss pass
  rasterize_backward         per-pixel backward compositing
  sh_opacity_backward_adam   register SH VJP + SH/opacity Adam update
  project_backward_adam      register geometry VJP + Adam + densify stats
```

The optional difference scans only replace how the first command buffer derives
the per-tile counts. Both modes still synchronize before the checked CPU prefix,
use the same exact scatter/sort/raster stages, and fail before optimizer work if
the layout or arena cannot be represented safely.

The loss target entering `ssim_h_fwd` is a tightly packed UInt8 RGBA buffer;
RGB is sampled and converted with `byte / 255`. For camera masks, alpha carries
soft coverage and the same buffer drives coverage weighting or transparent
alpha supervision. Low-level callers may still supply a standalone UInt8 mask.

Upstream's [README](https://github.com/rayanht/msplat) covers the design behind
this and carries M4 Max benchmarks against gsplat. This fork has not re-run
them.

## License

Apache 2.0, as upstream, whose copyright the LICENSE file carries.
