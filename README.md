# msplat-ios

3D Gaussian Splatting training on iPhone. A fork of
[rayanht/msplat](https://github.com/rayanht/msplat), which built the engine:
the full training pipeline as fused Metal compute kernels, with no dependency
beyond system frameworks.

This fork reduces model and transient growth, adds a byte-budgeted image cache,
fixes three correctness bugs, and provides an iOS build plus a COLMAP-to-PLY
example app. It does not yet impose a hard model-memory or Gaussian-count cap.

## Memory

| | Before | After | Measured on |
|---|---|---|---|
| Model buffers | 979.1 MB | 239.7 MB | garden, 2000 iters, 2 downscales |
| Transient buffers | 561.4 MB | 353.5 MB | garden, 4 resolution levels, step 600+ |
| Training images | whole dataset | 512 MB budget (iOS default) | `MSPLAT_IMAGE_CACHE_MB` |

- Densification reserved `3 × num_active` for the case where every gaussian
  splits. The real population after a refine step is about 1.2× the count
  before it. Classification now runs first, so the grow asks for what will be
  written.
- Depth-chunk buffers were carried to the end of training after chunking turned
  off. They are released.
- Training images were all decoded up front. A byte-budgeted LRU holds what fits
  and reloads the rest.

## Correctness

**Rasterizer intersection overflow.** The packed buffers were sized
`num_points × 16`; the sort kernel wrote at offsets from a prefix sum over
every tile, bounded by nothing about `num_points`. The overflow landed in the
model, and far enough out faulted the GPU and killed the process's Metal
context. iPhone 15 Pro, 20852 gaussians: 906292 slots needed against 333136
allocated. The kernel now takes a capacity and clamps; buffers are sized from
what the GPU last reported needing.

**Chunk buffers across a resolution change.** The guard compared against a
dimension already updated earlier in the same step, so kernels addressed
6056760 elements of a 5749380-element buffer. PSNR 20.56 → 21.68.

**Non-idempotent intrinsics.** `loadImage` rewrote focal lengths and distortion
coefficients in place, zeroing the coefficients after undistorting — harmless
until a cache evicts and reloads. Intrinsics are re-derived from what the
dataset declared on every call.

## Additions

- COLMAP text models (`cameras.txt` / `images.txt` / `points3D.txt`)
- `.spz` export
- `--stop-densify-at`, stopping topology growth after a chosen step (not a hard
  Gaussian-count or memory cap)
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
    var config = TrainingConfig()
    config.iterations = 2_000
    config.numDownscales = 0 // train at the dataset's selected decode scale
    config.stopDensifyAt = 750

    let session = try MsplatSession(
        datasetURL: URL(fileURLWithPath: "path/to/colmap/"),
        options: DatasetOptions(downscaleFactor: 2.0),
        config: config
    )
    defer { try? session.close() }

    for _ in 0..<2_000 {
        let stats = try session.step()
        print("step=\(stats.iteration) splats=\(stats.splatCount)")
    }
    try session.exportPLY(to: URL(fileURLWithPath: "output.ply"))
}
```

`MsplatSession` is the checked, throwing API and serializes the engine's
process-global Metal state. The legacy `GaussianDataset` and `GaussianTrainer`
types remain available for source compatibility.

CLI:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/msplat path/to/dataset -n 7000 --eval
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
training images.

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

One training step, dispatched into a single Metal command encoder:

```
Forward:
  project_and_sh_forward     fused 3D→2D projection + spherical harmonics
  prefix_sum + scatter       gaussian→tile intersection mapping
  bitonic_sort_per_tile      tile-local depth sort + inline data packing
  nd_rasterize_forward       per-pixel alpha compositing (16×16 tiles)
  ssim_h_fwd + ssim_v_fwd    separable 11-tap SSIM + L1 loss

Backward:
  ssim_h_bwd + ssim_v_bwd    separable SSIM gradient
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
