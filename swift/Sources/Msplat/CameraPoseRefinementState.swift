import Foundation
import MsplatCore

/// A read-only snapshot of one camera's bounded training-time pose correction.
///
/// Translation values use the dataset's original, pre-normalization length
/// units. Rotation values are axis-angle radians. The corrected transform is a
/// row-major OpenGL camera-to-world matrix in the dataset's original coordinate
/// system.
public struct CameraPoseRefinementState: Sendable, Equatable {
    public let isEnabled: Bool
    public let isAnchor: Bool
    public let canonicalCameraIndex: Int
    public let optimizerStepCount: Int
    public let translationDelta: [Float]
    public let rotationDelta: [Float]
    public let translationNorm: Float
    public let rotationNorm: Float
    public let correctedCameraToWorld: CameraPose
    public let frameID: String

    init(from native: MsplatPoseRefinementStateV15) throws {
        let poseDelta = withUnsafeBytes(of: native.poseDelta) {
            Array($0.bindMemory(to: Float.self))
        }
        guard poseDelta.count == 6 else {
            throw MsplatError.internalFailure(
                "Native pose-refinement state does not contain six delta values"
            )
        }
        guard poseDelta.allSatisfy(\.isFinite),
              native.translationNorm.isFinite,
              native.rotationNorm.isFinite else {
            throw MsplatError.internalFailure(
                "Native pose-refinement state contains non-finite correction values"
            )
        }

        let correctedElements = withUnsafeBytes(of: native.correctedCameraToWorld) {
            Array($0.bindMemory(to: Float.self))
        }
        let frameID = try Self.decodeFrameID(
            native.frameId,
            length: native.frameIdLength
        )

        isEnabled = native.flags & UInt32(MSPLAT_POSE_REFINEMENT_STATE_ENABLED) != 0
        isAnchor = native.flags & UInt32(MSPLAT_POSE_REFINEMENT_STATE_ANCHOR) != 0
        canonicalCameraIndex = Int(native.canonicalCameraIndex)
        optimizerStepCount = Int(native.optimizerStepCount)
        translationDelta = Array(poseDelta[0..<3])
        rotationDelta = Array(poseDelta[3..<6])
        translationNorm = native.translationNorm
        rotationNorm = native.rotationNorm
        correctedCameraToWorld = try CameraPose(elements: correctedElements)
        self.frameID = frameID
    }

    private static func decodeFrameID(
        _ data: UnsafePointer<CChar>?,
        length: UInt64
    ) throws -> String {
        guard let byteCount = Int(exactly: length) else {
            throw MsplatError.internalFailure(
                "Native pose-refinement frame ID is too large"
            )
        }
        guard byteCount > 0 else { return "" }
        guard let data else {
            throw MsplatError.internalFailure(
                "Native pose-refinement frame ID has no storage"
            )
        }

        let bytes = UnsafeRawPointer(data).assumingMemoryBound(to: UInt8.self)
        let buffer = UnsafeBufferPointer(start: bytes, count: byteCount)
        guard let frameID = String(bytes: buffer, encoding: .utf8) else {
            throw MsplatError.internalFailure(
                "Native pose-refinement frame ID is not valid UTF-8"
            )
        }
        return frameID
    }
}
