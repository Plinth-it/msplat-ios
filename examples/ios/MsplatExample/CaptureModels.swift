import Foundation
import Msplat

enum CaptureMode: String, CaseIterable, Codable, Identifiable, Sendable {
    case object
    case scene

    var id: Self { self }

    var title: String {
        switch self {
        case .object: "Object"
        case .scene: "Scene"
        }
    }

    var requiresSubjectSelection: Bool { self == .object }
}

struct CapturedDataset: Sendable {
    let rootURL: URL
    let descriptor: DatasetDescriptor
    let manifest: CaptureManifest
    let reviewSnapshot: CaptureReviewSnapshot
}

enum TrainingDatasetSource {
    case importedFolder(DatasetFolder)
    case captured(CapturedDataset)

    var name: String {
        switch self {
        case .importedFolder(let folder): folder.name
        case .captured(let capture): capture.rootURL.lastPathComponent
        }
    }

    var summary: String {
        switch self {
        case .importedFolder(let folder): folder.summary
        case .captured(let capture):
            "ARKit \(capture.manifest.mode.title), " +
                "\(capture.reviewSnapshot.frameCount) images, " +
                "\(capture.reviewSnapshot.pointCount.formatted()) points"
        }
    }

    var importedFolder: DatasetFolder? {
        guard case .importedFolder(let folder) = self else { return nil }
        return folder
    }

    var capturedDataset: CapturedDataset? {
        guard case .captured(let capture) = self else { return nil }
        return capture
    }

    var includesTrainingMasks: Bool {
        switch self {
        case .importedFolder:
            false
        case .captured(let capture):
            capture.descriptor.frames.contains { $0.trainingMask != nil }
        }
    }
}

struct CaptureCalibrationRecord: Codable, Equatable, Sendable {
    let width: Int
    let height: Int
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float

    func descriptorCalibration() throws -> DatasetCalibration {
        try DatasetCalibration(
            width: width,
            height: height,
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy
        )
    }
}

/// Rotation from ARKit's native camera raster into the captured interface
/// orientation. Persisted pixels, masks, calibration, and pose all use it.
enum CaptureDisplayOrientation: String, Codable, Equatable, Hashable, Sendable {
    case up
    case down
    case right
    case left
}

struct CaptureFrameRecord: Codable, Equatable, Identifiable, Sendable {
    let id: String
    let imagePath: String
    let maskPath: String?
    let softMaskPath: String?
    /// Source-to-persisted rotation applied atomically to this frame's raster,
    /// masks, calibration, and camera pose. Optional for format-v1 journals.
    let displayOrientation: CaptureDisplayOrientation?
    let calibration: CaptureCalibrationRecord
    /// Row-major OpenGL camera-to-world transform.
    let cameraToWorld: [Float]
    let timestamp: TimeInterval
    let exposureDurationSeconds: TimeInterval
    let trackingState: String
    let maskConfidence: Float?
    let fusedPointCount: Int

    func imageURL(under rootURL: URL) -> URL {
        rootURL.appending(path: imagePath)
    }

    func maskURL(under rootURL: URL) -> URL? {
        maskPath.map { rootURL.appending(path: $0) }
    }

    func softMaskURL(under rootURL: URL) -> URL? {
        softMaskPath.map { rootURL.appending(path: $0) }
    }
}

struct CaptureManifest: Codable, Equatable, Sendable {
    static let currentFormatVersion = 2

    let formatVersion: Int
    let id: UUID
    let createdAt: Date
    let mode: CaptureMode
    let coordinateSystem: String
    let pointCloudPath: String
    let frames: [CaptureFrameRecord]
}

struct CaptureReviewSnapshot: Equatable, Sendable {
    let frameCount: Int
    let pointCount: Int
    let minimumCX: Float
    let maximumCX: Float
    let minimumCY: Float
    let maximumCY: Float
}

enum CaptureFailure: LocalizedError, Sendable {
    case unsupportedDevice
    case cameraPermissionDenied
    case noCurrentFrame
    case subjectDepthUnavailable
    case subjectNotFound
    case invalidFrame(String)
    case insufficientCapture(frameCount: Int, pointCount: Int)
    case persistence(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedDevice:
            "AR world tracking is not available on this device."
        case .cameraPermissionDenied:
            "Camera access is required to capture a training dataset."
        case .noCurrentFrame:
            "ARKit has not produced a usable camera frame yet."
        case .subjectDepthUnavailable:
            "No confident scene depth was available at that point. Try a textured, opaque surface or use Scene mode."
        case .subjectNotFound:
            "Vision did not find a foreground subject at that point."
        case .invalidFrame(let message):
            "The candidate frame was rejected: \(message)"
        case .insufficientCapture(let frameCount, let pointCount):
            "Capture needs at least 3 frames and one initial point; it currently has \(frameCount) frames and \(pointCount) points."
        case .persistence(let message):
            "The capture could not be saved: \(message)"
        }
    }
}
