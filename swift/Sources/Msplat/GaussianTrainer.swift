import MsplatCore
import Foundation

/// Trains a 3D Gaussian Splatting scene on a dataset.
public class GaussianTrainer {
    private let handle: MsplatTrainer?
    private let dataset: GaussianDataset

    /// Create a trainer.
    /// - Parameters:
    ///   - dataset: The loaded dataset. Retained for the trainer's lifetime.
    ///   - config: Training configuration.
    public init(dataset: GaussianDataset, config: TrainingConfig = TrainingConfig()) {
        self.dataset = dataset
        handle = withNativeEngineLock {
            msplat_trainer_create(dataset.handle, config.toC())
        }
    }

    deinit {
        guard let handle else { return }
        withNativeEngineLock { msplat_trainer_destroy(handle) }
    }

    /// Run one training step.
    @discardableResult
    public func step() -> TrainingStats {
        withNativeEngineLock {
            TrainingStats(from: msplat_trainer_step(handle))
        }
    }

    /// Train for all remaining iterations (blocking, no progress callbacks).
    /// For progress reporting, use `step()` in a loop instead.
    public func train() {
        withNativeEngineLock { msplat_trainer_train(handle) }
    }

    /// Evaluate on held-out test views.
    public func evaluate() -> EvalMetrics {
        withNativeEngineLock {
            EvalMetrics(from: msplat_trainer_evaluate(handle))
        }
    }

    /// Render a camera view as RGB float32 pixel data.
    public func render(cameraIndex: Int, useTest: Bool = false) -> PixelData {
        guard let nativeIndex = Int32(exactly: cameraIndex) else { return .empty }
        let buf = withNativeEngineLock {
            msplat_trainer_render(handle, nativeIndex, useTest)
        }
        let count = Int(buf.width) * Int(buf.height) * 3
        let data = Array(UnsafeBufferPointer(start: buf.data, count: count))
        free(buf.data)
        return PixelData(pixels: data, width: Int(buf.width), height: Int(buf.height))
    }

    /// Render from an arbitrary camera-to-world pose (4x4 row-major, OpenGL convention).
    /// Uses intrinsics (focal length, resolution) from the given reference camera.
    public func renderFromPose(camToWorld: [Float], refCameraIndex: Int = 0) -> PixelData {
        guard camToWorld.count == 16, camToWorld.allSatisfy(\.isFinite),
              let nativeIndex = Int32(exactly: refCameraIndex) else { return .empty }
        let buf = withNativeEngineLock {
            camToWorld.withUnsafeBufferPointer { pointer in
                msplat_trainer_render_pose(handle, pointer.baseAddress, nativeIndex)
            }
        }
        let count = Int(buf.width) * Int(buf.height) * 3
        let data = Array(UnsafeBufferPointer(start: buf.data, count: count))
        free(buf.data)
        return PixelData(pixels: data, width: Int(buf.width), height: Int(buf.height))
    }

    /// Zero-copy render from an arbitrary camera pose into a pre-allocated RGBA uint8 buffer.
    /// For real-time display loops where allocation overhead matters.
    ///
    /// Pass `nil` for `rgba` to query dimensions without rendering (for buffer pre-allocation).
    /// Buffer must hold at least `width × height × 4` bytes.
    public func renderFromPoseToBuffer(camToWorld: [Float], refCameraIndex: Int = 0,
                                       rgba: UnsafeMutablePointer<UInt8>?,
                                       width: inout Int32, height: inout Int32) {
        guard camToWorld.count == 16, camToWorld.allSatisfy(\.isFinite),
              let nativeIndex = Int32(exactly: refCameraIndex) else {
            width = 0
            height = 0
            return
        }
        withNativeEngineLock {
            camToWorld.withUnsafeBufferPointer { pointer in
                msplat_trainer_render_pose_to_buffer(
                    handle, pointer.baseAddress, nativeIndex, rgba, &width, &height
                )
            }
        }
    }

    /// Export scene as PLY.
    public func exportPly(to path: String) {
        withNativeEngineLock { msplat_trainer_export_ply(handle, path) }
    }

    /// Export scene as .splat.
    public func exportSplat(to path: String) {
        withNativeEngineLock { msplat_trainer_export_splat(handle, path) }
    }

    /// Export scene as .spz.
    public func exportSpz(to path: String) {
        withNativeEngineLock { msplat_trainer_export_spz(handle, path) }
    }

    /// Save full training state for resume.
    public func saveCheckpoint(to path: String) {
        withNativeEngineLock { msplat_trainer_save_checkpoint(handle, path) }
    }

    /// Load checkpoint and resume training. Returns the saved iteration.
    @discardableResult
    public func loadCheckpoint(from path: String) -> Int {
        withNativeEngineLock {
            Int(msplat_trainer_load_checkpoint(handle, path))
        }
    }

    /// Current number of gaussians.
    public var splatCount: Int {
        withNativeEngineLock { Int(msplat_trainer_splat_count(handle)) }
    }

    /// Current training iteration.
    public var iteration: Int {
        withNativeEngineLock { Int(msplat_trainer_iteration(handle)) }
    }
}

/// RGB float32 pixel data from a render.
public struct PixelData: Sendable {
    public let pixels: [Float]  // RGB, HWC layout
    public let width: Int
    public let height: Int

    static let empty = PixelData(pixels: [], width: 0, height: 0)
}

/// Synchronize the GPU (wait for all commands to complete).
public func msplatSync() {
    withNativeEngineLock { msplat_sync() }
}

/// Release cached GPU resources after all trainers are idle.
public func msplatCleanup() {
    withNativeEngineLock { msplat_cleanup() }
}
