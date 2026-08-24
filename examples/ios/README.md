# MsplatExample

An iOS app that imports a COLMAP reconstruction, trains it, and hands back a PLY.

## Running it

The Swift package links a prebuilt `MsplatCore.xcframework`, and that is a
build output rather than something committed. Build it first:

```sh
./scripts/build-xcframework.sh
```

Then open `examples/ios/MsplatExample.xcodeproj` and run. Set your own team
under Signing & Capabilities for a device build; the simulator needs no
signing.

## Getting a dataset onto the device

The app reads the folder you pick in place, so it works with anything the
Files app can reach — iCloud Drive, an external drive, or the app's own
folder. For the last one, with the device connected, drag a COLMAP folder
into `MsplatExample` under Finder's Files tab, then pick it from
`On My iPhone / MsplatExample`.

A COLMAP folder is one holding `cameras.bin` or `cameras.txt`, either at its
root or under `sparse/0`, alongside an `images` directory. Optional training
mask sidecars live below any case-insensitive `masks` path component.

After a folder is selected, the app counts regular files below those
directories on a background task and automatically enables **Use discovered
masks** when that count is nonzero. The switch remains manually available
during and after the advisory scan, and can disable discovery for a run. The displayed number is a
candidate-file count, not a matched-frame count: the native COLMAP loader owns
exact sidecar matching, and frames without a match train with full coverage.
For an image such as `images/foo.jpeg`, supported names include
`masks/foo.png`, `masks/foo.jpeg.mask`, and `masks/foo.mask.png`; matching is
case-insensitive and supports nested directory suffixes. An alpha-bearing mask
uses alpha, while other color masks use their first/red channel. Mask value 0
is transparent, 255 is opaque, and intermediate values preserve soft edges.
The app selects **Transparent** treatment by default: it composites source RGB
over the configured background and trains rendered alpha across the full frame,
which suppresses exterior floaters. **Coverage only** retains the earlier
behavior where mask values merely weight RGB loss and masked-out pixels provide
no opacity supervision. Frames without a matched mask remain opaque RGB targets
in either mode. Sidecars must match their source-image dimensions; unlike Brush,
MSplat does not resize mismatched masks.

## What it shows

Before loading the dataset, the app reads image dimensions from ImageIO
metadata and builds one of two explicit plans:

- Preview targets a 1,600-pixel longest edge, SH degree 1, one resolution
  stage, and a 250,000-Gaussian ceiling.
- Balanced targets a 1,920-pixel longest edge, trains its first half at an
  additional 2x downscale before moving to the final resolution, reaches SH
  degree 2, and has a 400,000-Gaussian ceiling.

For either profile, when the selected COLMAP model starts with more sparse
points, the ceiling rises only enough to preserve the input population and the
memory estimate is recomputed before training.

The plan screen shows each effective resolution stage, target SH degree,
initial Gaussian count, Gaussian ceiling, and a conservative code-derived
peak-memory estimate. The
estimate covers native model and training buffers, image-cache insertion,
target-resolution app-owned decode buffers, and recommended headroom. When mask
discovery is enabled, it also includes conservative source-mask decoding.
Training targets remain compact UInt8 RGBA buffers, with coverage packed into
alpha, plus one activity byte per 16x16 tile for masked coverage targets; decoded
CPU pixels are released after upload. The current formula reflects that compact
cache and removal of dead workspaces. The app refuses to start when that
estimate exceeds a nonzero `os_proc_available_memory` value at preflight (the
simulator reports zero and skips this comparison).

The sample enables depth-one CPU camera prefetch for both masked and unmasked
runs, keeping their timing comparison fair. It prepares the next training
target while the current Metal step runs. Library clients remain opt-in through
`DatasetOptions.prefetchTrainingTargets`; existing native clients can still set
exactly `MSPLAT_CAMERA_PREFETCH=1` in their environment.

This check is a planning aid, not a jetsam guarantee. Metal driver state,
framework allocations, other process memory, and changing system pressure are
not fully modeled. The model term intentionally covers the pre-cutoff peak;
densification-only state is released later. ImageIO now requests the selected
input resolution directly,
but a codec may still use private decoder surfaces that the estimate cannot
observe. COLMAP camera calibration uses encoded raster coordinates, so valid
EXIF orientation metadata is checked but intentionally not applied; an oriented
image provider must transform its calibration and pose together with its pixels.

During training the app distinguishes submitted from completed iterations and
advances its progress bar only from GPU completion. At each sampled preview it
polls the matching logical-step GPU execution and end-to-end time, loss,
effective resolution and SH degree, Gaussian count/capacity, rasterizer
overflow incidence, categorized native/image-cache memory, `phys_footprint`,
iOS available memory, cache hit rate, and thermal state. The preview render is
sampled rather than performed every iteration; because it synchronizes prior
work, the following telemetry poll is an authoritative completion snapshot.

The model/transient/image figures are logical owned buffers. They do not include
Metal driver state, codec-private surfaces, framework allocations, or allocator
overhead; `phys_footprint` is the process-wide device measurement. The
simulator does not provide meaningful jetsam headroom, and physical-device
profiling remains the validation gate for timing, memory pressure, and thermal
behavior.

The exported PLY goes to the app's Documents folder and is offered through the
share sheet.
