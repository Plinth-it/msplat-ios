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
root or under `sparse/0`, alongside an `images` directory.

## What it shows

Before loading the dataset, the app reads image dimensions from ImageIO
metadata and builds one of two explicit plans:

- Preview targets a 1,600-pixel longest edge, SH degree 1, one resolution
  stage, and a hard limit of 250,000 Gaussians.
- Balanced targets a 1,920-pixel longest edge, trains its first half at an
  additional 2x downscale before moving to the final resolution, reaches SH
  degree 2, and limits the population to 400,000 Gaussians.

The plan screen shows each effective resolution stage, target SH degree,
Gaussian ceiling, and a conservative code-derived peak-memory estimate. The
estimate covers native model and training buffers, image-cache insertion,
target-resolution app-owned decode buffers, and recommended headroom. The
current formula reflects the split cache and removal of dead workspaces. The app
refuses to start when that estimate exceeds a nonzero
`os_proc_available_memory` value at preflight (the simulator reports zero and
skips this comparison).

This check is a planning aid, not a jetsam guarantee. Metal driver state,
framework allocations, other process memory, and changing system pressure are
not fully modeled. The model term intentionally covers the pre-cutoff peak;
densification-only state is released later. ImageIO now requests the selected
input resolution directly,
but a codec may still use private decoder surfaces that the estimate cannot
observe. COLMAP camera calibration uses encoded raster coordinates, so valid
EXIF orientation metadata is checked but intentionally not applied; an oriented
image provider must transform its calibration and pose together with its pixels.

During training the app reports the step, Gaussian count, CPU
encode/submission time, `phys_footprint`, and the memory iOS currently reports
available. It also samples a rendered preview rather than rendering every
iteration.

The exported PLY goes to the app's Documents folder and is offered through the
share sheet.
