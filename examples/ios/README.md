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

Training reports the step, the gaussian count, milliseconds per step, and the
memory figures that matter on a phone: `phys_footprint`, which is what jetsam
counts, and `os_proc_available_memory`, which is how much headroom is left
before the app is killed. Resolution defaults to half, because full-resolution
captures are where a phone runs out of memory first.

The exported PLY goes to the app's Documents folder and is offered through the
share sheet.
