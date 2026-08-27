import MsplatCore
import Foundation

/// A metallib is compiled against one SDK and will not load on another, so the
/// package ships one per platform and picks here rather than at build time.
private var metallibResourceName: String {
    #if os(macOS)
    return "default-macos"
    #elseif targetEnvironment(simulator)
    return "default-iossimulator"
    #else
    return "default-ios"
    #endif
}

/// A loaded dataset of camera views for training.
@available(
    *,
    deprecated,
    message: "Use MsplatSession, which reports initialization failures and enforces exclusive native-engine ownership"
)
public class GaussianDataset {
    let handle: MsplatDataset

    /// Load a dataset from disk.
    /// - Parameters:
    ///   - path: Path to COLMAP, Nerfstudio, or other supported format.
    ///   - downscaleFactor: Image downscale factor (1.0 = full resolution).
    ///   - evalMode: If true, split cameras into train/test sets.
    ///   - testEvery: Hold out every Nth image for evaluation.
    public init(path: String, downscaleFactor: Float = 1.0,
                evalMode: Bool = false, testEvery: Int32 = 8) throws {
        try reserveNativeSession()
        let createdHandle: MsplatDataset
        do {
            createdHandle = try withConfiguredNativeEngine(metallibResourceName) {
                var dataset: MsplatDataset?
                var nativeError = MsplatErrorInfo()
                let status = msplat_dataset_create_v10(
                    path,
                    downscaleFactor,
                    evalMode,
                    testEvery,
                    false,
                    &dataset,
                    &nativeError
                )
                try checkNativeStatus(status, error: &nativeError)
                guard let dataset else {
                    throw MsplatError.internalFailure(
                        "Native dataset creation returned no handle"
                    )
                }
                return dataset
            }
        } catch {
            releaseNativeSession()
            throw error
        }
        handle = createdHandle
    }

    deinit {
        withNativeEngineLock {
            var nativeError = MsplatErrorInfo()
            _ = msplat_dataset_destroy_v2(handle, &nativeError)
        }
        releaseNativeSession()
    }

    /// Number of training cameras.
    public var numTrain: Int {
        withNativeEngineLock {
            Int(msplat_dataset_num_train(handle))
        }
    }

    /// Number of test cameras (0 if evalMode was false).
    public var numTest: Int {
        withNativeEngineLock {
            Int(msplat_dataset_num_test(handle))
        }
    }

    /// Get the camera-to-world pose (4x4 row-major, OpenGL convention) for a training camera.
    public func cameraPose(at index: Int) -> [Float] {
        var pose = [Float](repeating: 0, count: 16)
        guard let nativeIndex = Int32(exactly: index) else { return pose }
        withNativeEngineLock {
            pose.withUnsafeMutableBufferPointer { pointer in
                msplat_dataset_camera_pose(handle, nativeIndex, pointer.baseAddress)
            }
        }
        return pose
    }
}
