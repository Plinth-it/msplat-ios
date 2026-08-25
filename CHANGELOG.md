# Changelog

## Unreleased

- Added ABI v2 checked C entry points with structured errors, input and buffer
  validation, ABI/config-size checks, and exception containment.
- Added ABI v3 checked trainer creation with a size-validated
  `MsplatTrainingLimits` structure. Its `maxGaussians` field provides an
  optional hard population and backing-buffer ceiling while ABI v2 creation
  retains the unlimited behavior.
- Added ABI v4 query-only telemetry for submitted versus completed training,
  per-logical-step GPU and end-to-end timing, completion-safe loss, typed
  rasterizer overflow incidence, categorized buffer ownership, image-cache
  hit/miss counts, `phys_footprint`, and iOS available memory. Legacy step
  statistics remain submission-only and ABI-compatible.
- Added ABI v5 checked creation from a canonical caller-owned dataset
  descriptor. The boundary deep-copies frame calibration and poses, image
  paths, sparse points, observations, and provenance; structural contract
  errors remain distinct from semantic dataset failures. Swift exposes the
  same immutable descriptor through the actor-isolated `MsplatSession` while
  retaining explicitly supplied security-scoped roots for lazy image access.
- Added ABI v6 optional per-frame training masks as a synchronously copied
  sidecar to the stable v5 descriptor. Luminance or alpha masks retain UInt8
  soft coverage through resize and undistortion, weight L1 and SSIM loss, and
  normalize completed loss telemetry by covered RGB units. Unmasked callers
  continue through ABI v5 without mask allocation or storage and with unchanged
  loss normalization.
- Added ABI v7 CPU-only capture diagnostics over canonical camera geometry,
  sparse points, and observation tracks. Swift descriptors can now preflight
  per-frame and aggregate reprojection residuals, source-reported point errors,
  track coverage, out-of-frame features, behind-camera points, and non-finite
  projections without decoding assets or initializing Metal.
- Added ABI v8 opt-in per-camera log-RGB photometric refinement with bounded,
  independently stepped Adam state. Because training pixels are sRGB encoded,
  the learned values are photometric gains rather than physical exposure.
  Canonical render, evaluation, and export paths remain unchanged; checkpoint
  v2 preserves gains, moments, per-camera visit counts, and exact frame IDs.
- Added ABI v9 opt-in camera-pose refinement with bounded, regularized SE(3)
  corrections, a fixed first-training-camera anchor, and per-camera Adam state.
  The initial geometry-only gradient detaches the SH view-direction term.
  Imported poses and canonical render, evaluation, and export remain unchanged;
  checkpoint v3 preserves pose state and validates frame IDs and source poses.
- Added ABI v10 opt-in training-mask discovery for path-based COLMAP datasets.
  Regular files under a case-insensitive `masks` directory use Brush-compatible
  stem, suffix-directory, and deterministic tie-breaking rules; partial mask
  sets remain valid. Discovered masks read alpha when present and the first color
  channel otherwise, while ABI v2 remains discovery-disabled and descriptor ABI
  v6 remains limited to explicit luminance or alpha coverage.
- Added ABI v11 versioned training-mask treatment options while preserving the
  locked `MsplatConfig` layout and coverage defaults for every earlier entry
  point. Transparent mode composites masked RGB over the configured background,
  normalizes RGB over the full frame, and adds a full-frame L1 objective on
  rendered alpha (`1 - transmittance`). Swift exposes the mode and alpha weight,
  and the sample defaults discovered masks to transparent treatment while
  retaining coverage-only selection. Transparent treatment rejects photometric
  gain refinement so synthetic-background pixels cannot dominate those gains.
  Chunked raster backward now keeps every pixel alive through cooperative
  threadgroup barriers, so dense tiles propagate gradients through all chunks.
- Added ABI v12 completed-step telemetry for image preparation, exact-count GPU
  duration and wall wait, post-count encoding, arena growth, and tile-density
  buckets. The ABI v4 telemetry structure and query remain byte-for-byte stable.
- Added ABI v13 GPU-native preview submissions. Swift
  `MsplatSession.submitPreview(...)` returns a `MetalPreviewSubmission`, whose
  `waitUntilReady()` produces a `MetalPreviewSurface` that owns an immutable
  BGRA8Unorm `MTLTexture`. Existing `renderRGBA` and `PixelData` APIs remain
  available. MsplatExample keeps at most one pending submission plus the latest
  completed surface, displays that texture without a CPU copy or UIImage
  re-upload, preserves the legacy UIImage display orientation, and includes both
  surfaces in its memory preflight. Fixed-camera submission still performs the
  renderer's existing exact-count synchronization; fully nonblocking preview
  submission depends on the planned GPU count/scan work.
- Added ABI v14 instance-scoped training-target prefetch enablement. Swift
  exposes it through `DatasetOptions.prefetchTrainingTargets`, and the iOS
  sample enables it for both masked and unmasked runs so their timing comparison
  remains fair. The existing environment opt-in remains available to native
  clients. The first shuffled target is scheduled when the trainer is created;
  later targets overlap the preceding Metal step while upload and LRU mutation
  stay serialized.
- Added an exact binary-grayscale PNG mask path for discovered Brush-style
  masks. Eligible 8-bit black/white masks decode into one source byte per pixel
  before the existing area filter; soft, profiled, alpha, and color masks retain
  the established RGBA/sRGB fallback and byte-for-byte target semantics.
- Removed the unused depth cotangent from RGB training, computed projected
  opacity once per visible Gaussian, and recovered backward Gaussian IDs directly
  from sorted keys. Exact-intersection storage drops from 56 to 52 bytes per
  entry without changing raster output or the training objective.
- Fused geometry backward directly into the mean, scale, and quaternion Adam
  updates, and folded opacity plus densification-stat updates into the terminal
  backward stage. The training cache no longer owns or clears the three geometry
  gradient tensors, removing 40 bytes per Gaussian and four standalone Adam plus
  one statistics dispatch. Geometry and opacity preserve full-population
  zero-gradient moment decay; SH retains its visibility/degree gates and now
  includes degree-four backward updates.
- Replaced retained Float32 RGB training images with tightly packed UInt8 RGBA
  targets. ImageIO decode, resolution pyramids, and Brown-Conrady correction
  remain byte-native; the SSIM/L1 kernels normalize RGB during their existing
  tile loads, while CPU evaluation accepts the same compact representation.
  Decoded CPU pixels are released after upload. Masked coverage is packed into
  the existing alpha byte. Coverage-mode targets also cache one UInt8 activity
  byte per 16x16 render tile, including the exact five-pixel SSIM halo, and use
  it to prune exact intersections without changing projection, dense loss, or
  transparent-mode execution.
- Corrected the image-edge versus array-index conversion in Brown-Conrady
  rectification, including alpha=0 crop endpoints and paired mask sampling.
  The renderer now uses an exact homogeneous divide, propagates the missing
  perspective-depth projection gradient, and differentiates EWA FOV clamps
  consistently with their forward pass. Synthetic pixel and finite-difference
  contracts lock these conventions; mirrored EXIF camera normalization now
  fails explicitly instead of producing pixels that cannot match its geometry.
- Replaced fixed 2,048-entry per-tile bins with a two-pass exact intersection
  pipeline: projection counts each tile, the host builds checked offsets and
  grows compact arenas, and exact-range bitonic/radix sorting preserves every
  intersection. Allocation, native-index, and the explicit 65,536-per-tile
  work-limit failures now abort before rasterization or Adam; the planner models
  the 52-byte exact arena without treating its estimate as a correctness cap.
- Enforced the Gaussian ceiling during initialization, densification, PLY
  import, and checkpoint restore. Capacity-constrained densification keeps the
  highest normalized-gradient candidates, and `--max-gaussians` exposes the
  limit in the native CLI and Python CLI.
- Added the actor-isolated, throwing `MsplatSession` Swift API while preserving
  the existing Swift and C symbols for source compatibility.
- Added Swift `TrainingPlan` validation for input decoding, exact native
  resolution-stage mapping, target SH degree, and the required Gaussian limit.
  Its code-derived peak-memory estimate includes native model/training buffers,
  image-cache insertion, target-resolution app-owned decode buffers, and
  recommended headroom; it remains a conservative planning aid, not a jetsam
  guarantee.
- Images now decode through ImageIO thumbnails at the requested input
  resolution into an explicit sRGB canvas. COLMAP paths validate EXIF metadata
  while preserving encoded raster coordinates so pixels, intrinsics, and poses
  remain in the same frame.
- COLMAP, Nerfstudio, and Polycam now materialize through one validated
  canonical dataset descriptor. It preserves stable frame/calibration IDs,
  COLMAP sparse-point IDs and reprojection errors, source provenance, and an
  explicit encoded or EXIF-normalized raster-coordinate policy.
- Corrected Polycam raw-export pose import to read the documented row-major
  `t_00...t_23` camera-to-world matrix without an extra axis flip. Each frame
  now uses corrected data only when both its corrected camera and image exist,
  otherwise falling back to its complete raw pair; incomplete pairs and
  missing required intrinsics are rejected explicitly.
- Camera render caches now fingerprint the complete camera-to-world pose and
  rebuild derived view/projection state after either explicit or legacy direct
  pose mutation. Runtime cameras also retain their immutable source
  calibration eagerly, keeping sparse observations separate from effective
  decoded-image geometry ahead of camera refinement.
- COLMAP image observations and sparse-point tracks are now retained as one
  canonical correspondence table. Text and binary imports reject inconsistent
  tracks, duplicate source IDs, truncated records, and unsafe source counts.
- Render-only calls now allocate only shared forward workspaces. Loss, SSIM,
  and backward buffers are added lazily by training, while unused transient
  prefix, spherical-harmonics gradient, and SSIM-window storage was removed;
  `TrainingPlan` memory estimates reflect the smaller live set.
- Compacted the active FP32 SSIM derivative workspace from 15 to 9 values per
  pixel and overwrote rendered RGB in place with its gradient, reducing the
  training cache by 36 bytes per pixel. Full 16-by-16 threadgroups now preserve
  SSIM shared-memory loads at image edges.
- Densification scratch and gradient-stat buffers are now omitted when topology
  growth is disabled from the first step and released, after GPU completion, at
  the configured cutoff. Checkpoint resumes past that cutoff do not recreate
  them.
- Checkpoint restore now requires the saved SH degree to match the configured
  training degree, preventing a planned memory/quality contract from changing
  silently during resume.
- Added Preview and Balanced plans to the iOS example. The app displays the
  resolved stages, SH degree, Gaussian ceiling, and estimated peak, then rejects
  a plan when that estimate exceeds the memory iOS currently reports available.
- Fixed Metal context construction, initialization races, resource teardown,
  full submitted-chain command completion checks, and recoverable encoder
  failures.
- Retained datasets for trainer lifetimes and removed automatic global cache
  cleanup from individual Swift trainer destruction; serialized native trainer
  transactions while the Metal engine remains process-global.
- Hardened checkpoint loading against corrupt shapes, sizes, truncation, and
  partial state replacement; checkpoint and model exports now use atomic
  temporary-file replacement, with native regression tests for invalid inputs.
- Made the iOS preview's single-resolution policy and CPU submission timing
  explicit, pinned CMake downloads by hash, and added an iOS simulator build to CI.
- The iOS example now advances progress from completed GPU iterations and
  displays CPU submission, GPU execution, end-to-end latency, loss, effective
  resolution/SH degree, tracked memory categories, cache hit rate, thermal
  state, and completed-step overflow warnings.

## v1.1.3 — Fused kernels + pre-allocated tile bins

- **Fused SH backward into Adam optimizer** — spherical harmonics gradients are now
  computed in registers and fed directly into Adam updates, eliminating a ~600 MB/iter
  device memory round-trip (at 1.5M gaussians).
- **Fused SSIM vertical-forward + horizontal-backward** — replaces two separate passes
  with a single kernel that recomputes V-conv from the H-buffer, saving ~130 MB/iter
  of intermediate buffer traffic.
- **Pre-allocated per-tile bins** — replaces the count→prefix-sum→scatter intersection
  pipeline with direct scatter to fixed-size per-tile bins. Eliminates 3 kernel
  dispatches and 3 memory barriers per iteration. `prefix_sort_pack` stage reduced
  from 19% to 10% of GPU time.
- **14–48% faster training** across mipnerf360 scenes. Improvement scales with gaussian
  count: garden 30K (3.5M gaussians) sees the largest speedup at 48%.
- **Per-stage GPU profiling** — `PROFILE_STAGES=1` enables Metal timestamp counter
  sampling per pipeline stage. Uses separate compute encoders on the same command buffer
  with `MTLComputePassDescriptor` for zero-overhead timestamp capture.
- **GPU timing instrumentation** — `PROFILE_GPU=1` adds completion handler timing to
  command buffers, reporting per-CB GPU execution time without affecting the
  `commitAndContinue` pipeline.

## v1.1.2

- Added `py.typed` marker (PEP 561) — type checkers now discover stubs automatically
- `TrainingConfig(bg_color=...)` now raises `ValueError` on wrong-size lists instead
  of silently falling back to the default

## v1.1.1

- Fixed `new[]`/`free()` mismatch in C API pixel buffer allocation — undefined
  behavior when Swift or other C callers freed render output with `free()`.
  Allocation now uses `malloc` consistently.
- Updated type stubs (`_core.pyi`) with `camera_pose` and `render_from_pose`
  methods added in v1.1.

## v1.1 — Arbitrary viewpoint rendering

- **`renderFromPose` API** — render from any camera-to-world matrix, not just dataset cameras.
  Uses intrinsics from a reference camera. Available across all surfaces:
  - C++: `trainer.renderFromPose(camToWorld, refCameraIndex)`
  - C API: `msplat_trainer_render_pose()`
  - Python: `trainer.render_from_pose(cam_to_world, ref_cam_idx=0)`
  - Swift: `trainer.renderFromPose(camToWorld:refCameraIndex:)`
- **`renderFromPoseToBuffer`** — zero-copy variant that writes directly into a
  caller-provided RGBA uint8 buffer. Eliminates intermediate float allocation for
  real-time display loops (400 FPS at full resolution on M4 Max).
  - C++: `trainer.renderFromPoseToBuffer(camToWorld, ref, outRGBA, &w, &h)`
  - C API: `msplat_trainer_render_pose_to_buffer()`
- **`cameraPose` accessor** — retrieve camera-to-world matrices from loaded datasets.
  - C++: `dataset.cameraPose(index, outMatrix)`
  - C API: `msplat_dataset_camera_pose()`
  - Python: `dataset.camera_pose(index)` → numpy `(4, 4)` float32
  - Swift: `dataset.cameraPose(at: index)` → `[Float]`
- **Demo app** (`demo/`) — macOS SwiftUI app for screen-recording hero videos.
  Live training with progress bar, then smooth circular camera orbit with FPS counter.

## v1.0 — Public release

Stable API across Python, Swift, and C++ surfaces.

## v0.6 — Bug fixes and API improvements

- Fixed SSIM Gaussian kernel — ported formula `floor((i - windowSize) / 2.0)`
  produced pairwise-duplicated values instead of a symmetric bell curve.
  Corrected to `i - windowSize / 2` in both Metal shader and CPU eval path.
- Fixed ASCII PLY reader — x coordinate used byte offset instead of token index,
  silently reading wrong values when x isn't the first property.
- Background color now configurable across all APIs (Python, Swift, C++, CLI).
  Default magenta `[0.613, 0.010, 0.398]` documented as intentional
  (high contrast for debugging under-reconstructed regions).
  - Python: `TrainingConfig(bg_color=[r, g, b])`
  - Swift: `config.bgColor = (r, g, b)`
  - CLI: `--bg-color R G B`
- `cleanup()` now safe to call multiple times (Python guard prevents double-free
  when manual call + atexit handler both fire)
- Added type stubs (`_core.pyi`) — IDEs now have autocompletion and type checking
  for the compiled extension module
- Documented `MTensor.view()` use-after-free risk (non-owning alias)

## v0.5 — Open-source cleanup

- Removed datasets from git (1+ GB of LFS-tracked files)
  - CI/release workflows now download garden dataset from Google Storage with caching
- Code quality fixes
  - `exit(1)` on image load failure → `throw std::runtime_error` (safe for library consumers)
  - Deduplicated `getCachedMTensorImage` (3 copies) into `Camera::getGPUImage()` method
  - Removed debug `printf` on metallib load, commented-out `printf` in Metal shader
  - Deleted dead `msplat_model.hpp` alias header
  - Error messages to `stderr` instead of `stdout`
- Python API improvements
  - Removed always-zero `loss`/`psnr` fields from `TrainingStats`
  - Added docstrings to all nanobind bindings (TrainingConfig, TrainingStats, Dataset, GaussianTrainer)
  - Fixed `requires-python` from `>=3.10` to `>=3.12` (only supported versions)
  - Fixed SPDX license / classifier conflict in `pyproject.toml`
- Swift package: added render and export PLY tests (3 → 5 tests)
- Apache 2.0 license
- Full PyPI metadata (author, classifiers, keywords, URLs)

## v0.4.1

- Swift XCFramework distribution: `scripts/build-xcframework.sh` builds a self-contained XCFramework
  - `msplat_set_metallib_path()` C API for explicit Metal library path configuration
  - Swift wrapper auto-configures metallib via `Bundle.module`
  - Replaced CMsplat bridge target with `.binaryTarget` pointing at XCFramework
- GitHub Actions CI/CD
  - `ci.yml`: build + test C++ CLI, Python wheels (3.12/3.13), Swift package on every push
  - `release.yml`: GitHub Releases + PyPI publishing on tagged commits
  - Version sync check (VERSION, pyproject.toml, `__init__.py`) gates all jobs
- Fixed OpenGL Y/Z flip in COLMAP pose conversion (negate columns, not rows)
- Removed `constants.hpp` — `APP_VERSION` from CMake, `PI` → `M_PI`

## v0.4 — Drop OpenCV dependency

- Replaced OpenCV with lightweight built-in implementations
  - `Image` struct (float32 RGB) replaces `cv::Mat` throughout
  - Area-based image resize (box filter) replaces `cv::resize(INTER_AREA)`
  - Brown-Conrady undistortion with alpha=0 crop replaces `cv::undistort`
  - CoreGraphics PNG writing replaces `cv::imwrite`
  - Dropped dead Linux/OpenCV fallback code (Metal is macOS-only)
- No external dependencies beyond system frameworks (Metal, CoreGraphics, ImageIO)
- Removed `brew install opencv` requirement

## v0.3 — Checkpoint system, clean-room loaders, CLI11

- Checkpoint save/resume (`trainer.save_checkpoint()` / `trainer.load_checkpoint()`)
  - Binary `.msplat` format: gaussian params + full Adam optimizer state
  - Bound in Python, Swift, and C API
  - 2 new tests (save/load + resume round-trip)
- Rewrote all dataset loaders from scratch
  - COLMAP binary format (cameras.bin, images.bin, points3D.bin)
  - Nerfstudio transforms.json
  - Polycam (keyframes/ and cameras.json layouts)
  - Dropped OpenSfM + OpenMVG (low adoption, trivially convertible to COLMAP)
  - PLY point cloud reader + COLMAP binary point reader
  - CoreGraphics image loading on macOS
- Moved Gaussian PLY/splat I/O out of model.cpp into `loaders/save_gaussians.cpp`
- Switched CLI from cxxopts to CLI11 (validation, subcommand-ready)
- Rewrote `utils.hpp` → `random_iter.hpp` (dropped `parallel_for`)
- Made `kdtree_tensor` header-only
- Removed `tensor_math.{cpp,hpp}` (unused)
- Loader code reorganized into `core/src/loaders/` subdirectory

## v0.2 — Swift Package + general cleanup

- Swift Package with C API bridge (3 tests: config, dataset loading, 10-step training)
- C API header (`msplat_c_api.h`) for Swift interop via opaque handles
- `msplat_api.{hpp,mm}` compiled into `libmsplat_core.a` (not SPM)

## v0.1 — Initial release

Standalone 3D Gaussian Splatting engine for Apple Silicon with 44 fused Metal
compute kernels and Python bindings via nanobind.

- C++ core with Metal backend (44 fused compute kernels)
- CMake build system: `libmsplat_core.a` static library + `msplat` CLI
- Python package (`pip install msplat`) via scikit-build-core + nanobind
- Full training pipeline: `GaussianTrainer.train()` with progress callbacks
- Multi-format dataset loading: COLMAP, Nerfstudio, OpenSfM, OpenMVG
- Evaluation on held-out test views (PSNR, SSIM, L1)
- PLY and .splat export
- Rendering API: `trainer.render(cam_idx)`
- Python CLI: `msplat-train path/to/dataset -n 7000 --eval`

### Numbers

Garden (mipnerf360), 7K steps, 24 test views:
- PSNR: 25.75 dB
- SSIM: 0.786
- 1.5M gaussians
- ~3 ms/iter at 4x downscale, ~17 ms/iter full res (M4 Max)
