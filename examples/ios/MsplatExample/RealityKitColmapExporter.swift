import CoreImage
import Foundation
import ImageIO
@preconcurrency import RealityKit
import simd
import UniformTypeIdentifiers
import Vision

struct RealityKitColmapCamera: Equatable, Sendable {
    let width: Int
    let height: Int
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double
}

struct RealityKitColmapImageEntry: Sendable {
    let imageID: Int
    let filename: String
    let cameraToWorld: simd_float4x4
    let camera: RealityKitColmapCamera
    let orientation: RealityKitColmapEXIFOrientation
}

struct RealityKitColmapPoint: Equatable, Sendable {
    let position: SIMD3<Float>
    let color: SIMD3<Float>
}

struct RealityKitColmapExportResult: Sendable {
    let datasetDirectory: URL
    let alignedImageCount: Int
    let cameraCount: Int
    let pointCount: Int
    let maskCount: Int
}

enum RealityKitColmapExportStage: Sendable {
    case images
    case masks
    case sparseModel
    case publishing
}

enum RealityKitColmapExportError: LocalizedError, Sendable {
    case missingIntrinsics(sampleID: Int, filename: String)
    case missingSourceURL(sampleID: Int)
    case noAlignedImages
    case insufficientAlignedImages(Int)
    case noSparsePoints
    case invalidPose(sampleID: Int)
    case invalidSparsePoint(index: Int)
    case outputAlreadyExists
    case imageReadFailed(filename: String)
    case imageMetadataReadFailed(filename: String)
    case unsupportedOrientation(filename: String, orientation: Int)
    case imageEncodingFailed(filename: String)
    case invalidCamera(filename: String)
    case maskGenerationFailed(filename: String)
    case maskRenderFailed(filename: String)
    case maskEncodingFailed(filename: String)
    case emptyTrainingMask(filename: String)

    var errorDescription: String? {
        switch self {
        case .missingIntrinsics(let sampleID, let filename):
            "RealityKit returned no intrinsics for aligned sample \(sampleID) (\(filename))."
        case .missingSourceURL(let sampleID):
            "RealityKit returned a pose without a source image URL for sample \(sampleID)."
        case .noAlignedImages:
            "RealityKit did not align any input images."
        case .insufficientAlignedImages(let count):
            "RealityKit aligned only \(count) image(s); at least 3 are required for training."
        case .noSparsePoints:
            "RealityKit did not produce a point cloud, so the result cannot seed training."
        case .invalidPose(let sampleID):
            "RealityKit returned a non-finite camera pose for sample \(sampleID)."
        case .invalidSparsePoint(let index):
            "RealityKit returned a non-finite sparse point at index \(index)."
        case .outputAlreadyExists:
            "The generated alignment output path already exists. Try the export again."
        case .imageReadFailed(let filename):
            "The aligned source image could not be read: \(filename)."
        case .imageMetadataReadFailed(let filename):
            "The aligned source image has no usable dimensions or orientation: \(filename)."
        case .unsupportedOrientation(let filename, let orientation):
            "Mirrored EXIF orientation \(orientation) is unsupported for \(filename)."
        case .imageEncodingFailed(let filename):
            "The aligned image could not be preserved or converted to lossless sRGB PNG: \(filename)."
        case .invalidCamera(let filename):
            "RealityKit returned invalid camera intrinsics for \(filename)."
        case .maskGenerationFailed(let filename):
            "Vision could not generate a foreground training mask for \(filename)."
        case .maskRenderFailed(let filename):
            "The Vision training mask could not be rendered for \(filename)."
        case .maskEncodingFailed(let filename):
            "The Vision training mask could not be saved for \(filename)."
        case .emptyTrainingMask(let filename):
            "Vision found no foreground subject in \(filename), so a complete training-mask set could not be exported."
        }
    }
}

enum RealityKitColmapEXIFOrientation: Int, Sendable {
    case up = 1
    case upMirrored = 2
    case down = 3
    case downMirrored = 4
    case leftMirrored = 5
    case right = 6
    case rightMirrored = 7
    case left = 8

    var needsTransform: Bool { self != .up }

    var isMirrored: Bool {
        switch self {
        case .upMirrored, .downMirrored, .leftMirrored, .rightMirrored:
            true
        default:
            false
        }
    }
}

struct RealityKitColmapImageMetadata: Equatable, Sendable {
    let width: Int
    let height: Int
    let orientation: RealityKitColmapEXIFOrientation
}

enum RealityKitColmapPoseConverter {
    static func convertToColmapCamera(
        cameraToWorld matrix: simd_float4x4
    ) -> (qvec: [Double], tvec: [Double]) {
        let cameraToWorld: [[Double]] = [
            [Double(matrix[0][0]), Double(matrix[1][0]), Double(matrix[2][0]), Double(matrix[3][0])],
            [Double(matrix[0][1]), Double(matrix[1][1]), Double(matrix[2][1]), Double(matrix[3][1])],
            [Double(matrix[0][2]), Double(matrix[1][2]), Double(matrix[2][2]), Double(matrix[3][2])],
            [Double(matrix[0][3]), Double(matrix[1][3]), Double(matrix[2][3]), Double(matrix[3][3])],
        ]

        var worldToCamera = Array(
            repeating: Array(repeating: 0.0, count: 4),
            count: 4
        )
        for row in 0..<3 {
            for column in 0..<3 {
                worldToCamera[row][column] = cameraToWorld[column][row]
            }
        }
        for row in 0..<3 {
            worldToCamera[row][3] = -(
                worldToCamera[row][0] * cameraToWorld[0][3] +
                    worldToCamera[row][1] * cameraToWorld[1][3] +
                    worldToCamera[row][2] * cameraToWorld[2][3]
            )
        }

        // RealityKit cameras look down -Z with +Y up. COLMAP cameras look
        // down +Z with +Y down.
        var colmapWorldToCamera = worldToCamera
        for column in 0..<4 {
            colmapWorldToCamera[1][column] = -worldToCamera[1][column]
            colmapWorldToCamera[2][column] = -worldToCamera[2][column]
        }

        let rotation = (0..<3).map { row in
            (0..<3).map { column in colmapWorldToCamera[row][column] }
        }
        let translation = [
            colmapWorldToCamera[0][3],
            colmapWorldToCamera[1][3],
            colmapWorldToCamera[2][3],
        ]
        return (
            qvec: rotationMatrixToQuaternion(rotation),
            tvec: translation
        )
    }

    static func rotationMatrixToQuaternion(_ rotation: [[Double]]) -> [Double] {
        let rxx = rotation[0][0]
        let ryx = rotation[1][0]
        let rzx = rotation[2][0]
        let rxy = rotation[0][1]
        let ryy = rotation[1][1]
        let rzy = rotation[2][1]
        let rxz = rotation[0][2]
        let ryz = rotation[1][2]
        let rzz = rotation[2][2]

        let trace = rxx + ryy + rzz
        var qw: Double
        var qx: Double
        var qy: Double
        var qz: Double

        if trace > 0 {
            let scale = sqrt(trace + 1) * 2
            qw = 0.25 * scale
            qx = (rzy - ryz) / scale
            qy = (rxz - rzx) / scale
            qz = (ryx - rxy) / scale
        } else if rxx > ryy, rxx > rzz {
            let scale = sqrt(1 + rxx - ryy - rzz) * 2
            qw = (rzy - ryz) / scale
            qx = 0.25 * scale
            qy = (rxy + ryx) / scale
            qz = (rxz + rzx) / scale
        } else if ryy > rzz {
            let scale = sqrt(1 + ryy - rxx - rzz) * 2
            qw = (rxz - rzx) / scale
            qx = (rxy + ryx) / scale
            qy = 0.25 * scale
            qz = (ryz + rzy) / scale
        } else {
            let scale = sqrt(1 + rzz - rxx - ryy) * 2
            qw = (ryx - rxy) / scale
            qx = (rxz + rzx) / scale
            qy = (ryz + rzy) / scale
            qz = 0.25 * scale
        }

        if qw < 0 {
            qw = -qw
            qx = -qx
            qy = -qy
            qz = -qz
        }
        return [qw, qx, qy, qz]
    }
}

enum RealityKitColmapExportBuilder {
    static let seedModelNoticeFilename = "REALITYKIT_SEED_MODEL.md"

    static let seedModelNotice = """
    # RealityKit camera-and-point seed

    This export contains per-frame camera intrinsics, registered poses, and a colored point seed for msplat training.

    The COLMAP `images` records intentionally contain zero `POINTS2D` observations, and the `points3D` records intentionally contain zero tracks. This is not a feature-matched or bundle-adjusted SfM reconstruction. Run feature extraction, matching, triangulation, and bundle adjustment before using workflows that require observations or tracks.
    """

    static func maskFilename(forExportedImage filename: String) -> String {
        URL(fileURLWithPath: filename)
            .deletingPathExtension()
            .lastPathComponent + ".png"
    }

    static func exportImage(
        source: URL,
        to imagesDirectory: URL,
        imageID: Int,
        orientation: RealityKitColmapEXIFOrientation,
        context: CIContext
    ) throws -> String {
        let imageSource = CGImageSourceCreateWithURL(source as CFURL, nil)
        let sourceType = imageSource
            .flatMap(CGImageSourceGetType)
            .map { $0 as String }

        let preservedExtension: String? = if orientation == .up,
                                             sourceType == UTType.jpeg.identifier {
            "jpg"
        } else if orientation == .up,
                  sourceType == UTType.png.identifier {
            "png"
        } else {
            nil
        }
        let filename = String(
            format: "image_%06d.%@",
            imageID,
            preservedExtension ?? "png"
        )
        let destination = imagesDirectory.appending(path: filename)

        if preservedExtension != nil {
            do {
                try FileManager.default.copyItem(at: source, to: destination)
            } catch {
                throw RealityKitColmapExportError.imageEncodingFailed(
                    filename: source.lastPathComponent
                )
            }
            return filename
        }

        guard let imageSource,
              let decoded = CGImageSourceCreateImageAtIndex(
                imageSource,
                0,
                [kCGImageSourceShouldCache: false] as CFDictionary
              ) else {
            throw RealityKitColmapExportError.imageReadFailed(
                filename: source.lastPathComponent
            )
        }
        // Build from decoded pixels so the lossless output cannot inherit the
        // source EXIF orientation after that transform has already been baked.
        let image = CIImage(cgImage: decoded)
        let normalized = orientation.needsTransform
            ? image.oriented(forExifOrientation: Int32(orientation.rawValue))
            : image
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let data = context.pngRepresentation(
                of: normalized,
                format: .RGBA8,
                colorSpace: colorSpace,
                options: [:]
              ) else {
            throw RealityKitColmapExportError.imageEncodingFailed(
                filename: source.lastPathComponent
            )
        }
        try data.write(to: destination, options: .atomic)
        return filename
    }

    static func camera(
        _ camera: RealityKitColmapCamera,
        applying orientation: RealityKitColmapEXIFOrientation
    ) -> RealityKitColmapCamera {
        switch orientation {
        case .right:
            RealityKitColmapCamera(
                width: camera.height,
                height: camera.width,
                fx: camera.fy,
                fy: camera.fx,
                cx: Double(camera.height) - camera.cy,
                cy: camera.cx
            )
        case .left:
            RealityKitColmapCamera(
                width: camera.height,
                height: camera.width,
                fx: camera.fy,
                fy: camera.fx,
                cx: camera.cy,
                cy: Double(camera.width) - camera.cx
            )
        case .down:
            RealityKitColmapCamera(
                width: camera.width,
                height: camera.height,
                fx: camera.fx,
                fy: camera.fy,
                cx: Double(camera.width) - camera.cx,
                cy: Double(camera.height) - camera.cy
            )
        default:
            camera
        }
    }

    static func convertedPose(
        for entry: RealityKitColmapImageEntry
    ) -> (qvec: [Double], tvec: [Double]) {
        var conversion = RealityKitColmapPoseConverter.convertToColmapCamera(
            cameraToWorld: entry.cameraToWorld
        )
        guard let rasterRotation = cameraRotation(for: entry.orientation) else {
            return conversion
        }

        let originalRotation = quaternionToRotationMatrix(conversion.qvec)
        var rotated = Array(
            repeating: Array(repeating: 0.0, count: 3),
            count: 3
        )
        var translated = [0.0, 0.0, 0.0]
        for row in 0..<3 {
            for column in 0..<3 {
                for index in 0..<3 {
                    rotated[row][column] += rasterRotation[row][index] *
                        originalRotation[index][column]
                }
            }
            for index in 0..<3 {
                translated[row] += rasterRotation[row][index] *
                    conversion.tvec[index]
            }
        }

        conversion.qvec = RealityKitColmapPoseConverter
            .rotationMatrixToQuaternion(rotated)
        conversion.tvec = translated
        return conversion
    }

    static func cameraRecords(
        _ entries: [RealityKitColmapImageEntry]
    ) -> ([RealityKitColmapCamera], [Int: Int]) {
        var cameras: [RealityKitColmapCamera] = []
        cameras.reserveCapacity(entries.count)
        var assignments: [Int: Int] = [:]
        assignments.reserveCapacity(entries.count)

        for (index, entry) in entries.enumerated() {
            cameras.append(entry.camera)
            assignments[entry.imageID] = index + 1
        }

        return (cameras, assignments)
    }

    static func writeSparseFiles(
        to sparseDirectory: URL,
        entries: [RealityKitColmapImageEntry],
        points: [RealityKitColmapPoint]
    ) throws {
        guard !entries.isEmpty else {
            throw RealityKitColmapExportError.noAlignedImages
        }
        guard !points.isEmpty else {
            throw RealityKitColmapExportError.noSparsePoints
        }

        try FileManager.default.createDirectory(
            at: sparseDirectory,
            withIntermediateDirectories: true
        )
        let (cameras, assignments) = cameraRecords(entries)
        try writeCamerasText(to: sparseDirectory, cameras: cameras)
        try writeImagesText(
            to: sparseDirectory,
            entries: entries,
            cameraAssignments: assignments
        )
        try writePointsText(to: sparseDirectory, points: points)
        try writeCamerasBinary(to: sparseDirectory, cameras: cameras)
        try writeImagesBinary(
            to: sparseDirectory,
            entries: entries,
            cameraAssignments: assignments
        )
        try writePointsBinary(to: sparseDirectory, points: points)
        try (seedModelNotice + "\n").write(
            to: sparseDirectory.appending(path: seedModelNoticeFilename),
            atomically: true,
            encoding: .utf8
        )
    }

    private static func cameraRotation(
        for orientation: RealityKitColmapEXIFOrientation
    ) -> [[Double]]? {
        switch orientation {
        case .right:
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        case .left:
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        case .down:
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        default:
            nil
        }
    }

    private static func quaternionToRotationMatrix(_ quaternion: [Double])
        -> [[Double]] {
        let qw = quaternion[0]
        let qx = quaternion[1]
        let qy = quaternion[2]
        let qz = quaternion[3]
        return [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    }

    private static func writeCamerasText(
        to sparseDirectory: URL,
        cameras: [RealityKitColmapCamera]
    ) throws {
        var lines = [
            "# Camera list with one line of data per camera:",
            "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
            "# Number of cameras: \(cameras.count)",
        ]
        for (index, camera) in cameras.enumerated() {
            try Task.checkCancellation()
            lines.append(
                "\(index + 1) PINHOLE \(camera.width) \(camera.height) " +
                    "\(camera.fx) \(camera.fy) \(camera.cx) \(camera.cy)"
            )
        }
        try writeText(lines, to: sparseDirectory.appending(path: "cameras.txt"))
    }

    private static func writeImagesText(
        to sparseDirectory: URL,
        entries: [RealityKitColmapImageEntry],
        cameraAssignments: [Int: Int]
    ) throws {
        var lines = [
            "# Image list with two lines of data per image:",
            "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
            "# POINTS2D[] as (X, Y, POINT3D_ID)",
            "# RealityKit seed model: POINTS2D[] is intentionally empty.",
            "# Number of images: \(entries.count)",
        ]
        for entry in entries {
            try Task.checkCancellation()
            let pose = convertedPose(for: entry)
            let cameraID = cameraAssignments[entry.imageID] ?? 1
            lines.append(
                "\(entry.imageID) \(pose.qvec[0]) \(pose.qvec[1]) " +
                    "\(pose.qvec[2]) \(pose.qvec[3]) \(pose.tvec[0]) " +
                    "\(pose.tvec[1]) \(pose.tvec[2]) \(cameraID) \(entry.filename)"
            )
            lines.append("")
        }
        try writeText(lines, to: sparseDirectory.appending(path: "images.txt"))
    }

    private static func writePointsText(
        to sparseDirectory: URL,
        points: [RealityKitColmapPoint]
    ) throws {
        var lines = [
            "# 3D point list with one line of data per point:",
            "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]",
            "# RealityKit seed model: TRACK[] is intentionally empty.",
        ]
        for (index, point) in points.enumerated() {
            if index.isMultiple(of: 4_096) {
                try Task.checkCancellation()
            }
            lines.append(
                "\(index + 1) \(Double(point.position.x)) " +
                    "\(Double(point.position.y)) \(Double(point.position.z)) " +
                    "\(colorComponent(point.color.x)) " +
                    "\(colorComponent(point.color.y)) " +
                    "\(colorComponent(point.color.z)) 0.0"
            )
        }
        try writeText(lines, to: sparseDirectory.appending(path: "points3D.txt"))
    }

    private static func writeCamerasBinary(
        to sparseDirectory: URL,
        cameras: [RealityKitColmapCamera]
    ) throws {
        var data = Data()
        appendUInt64(&data, UInt64(cameras.count))
        for (index, camera) in cameras.enumerated() {
            try Task.checkCancellation()
            appendInt32(&data, Int32(index + 1))
            appendInt32(&data, 1) // PINHOLE
            appendUInt64(&data, UInt64(camera.width))
            appendUInt64(&data, UInt64(camera.height))
            appendFloat64(&data, camera.fx)
            appendFloat64(&data, camera.fy)
            appendFloat64(&data, camera.cx)
            appendFloat64(&data, camera.cy)
        }
        try data.write(
            to: sparseDirectory.appending(path: "cameras.bin"),
            options: .atomic
        )
    }

    private static func writeImagesBinary(
        to sparseDirectory: URL,
        entries: [RealityKitColmapImageEntry],
        cameraAssignments: [Int: Int]
    ) throws {
        var data = Data()
        appendUInt64(&data, UInt64(entries.count))
        for entry in entries {
            try Task.checkCancellation()
            let pose = convertedPose(for: entry)
            appendUInt32(&data, UInt32(entry.imageID))
            for value in pose.qvec { appendFloat64(&data, value) }
            for value in pose.tvec { appendFloat64(&data, value) }
            appendUInt32(&data, UInt32(cameraAssignments[entry.imageID] ?? 1))
            data.append(contentsOf: entry.filename.utf8)
            data.append(0)
            appendUInt64(&data, 0)
        }
        try data.write(
            to: sparseDirectory.appending(path: "images.bin"),
            options: .atomic
        )
    }

    private static func writePointsBinary(
        to sparseDirectory: URL,
        points: [RealityKitColmapPoint]
    ) throws {
        var data = Data()
        appendUInt64(&data, UInt64(points.count))
        for (index, point) in points.enumerated() {
            if index.isMultiple(of: 4_096) {
                try Task.checkCancellation()
            }
            appendUInt64(&data, UInt64(index + 1))
            appendFloat64(&data, Double(point.position.x))
            appendFloat64(&data, Double(point.position.y))
            appendFloat64(&data, Double(point.position.z))
            data.append(UInt8(colorComponent(point.color.x)))
            data.append(UInt8(colorComponent(point.color.y)))
            data.append(UInt8(colorComponent(point.color.z)))
            appendFloat64(&data, 0)
            appendUInt64(&data, 0)
        }
        try data.write(
            to: sparseDirectory.appending(path: "points3D.bin"),
            options: .atomic
        )
    }

    private static func writeText(_ lines: [String], to url: URL) throws {
        try (lines.joined(separator: "\n") + "\n")
            .write(to: url, atomically: true, encoding: .utf8)
    }

    private static func colorComponent(_ value: Float) -> Int {
        Int(min(max(Double(value), 0), 255))
    }

    private static func appendUInt32(_ data: inout Data, _ value: UInt32) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    private static func appendInt32(_ data: inout Data, _ value: Int32) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    private static func appendUInt64(_ data: inout Data, _ value: UInt64) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    private static func appendFloat64(_ data: inout Data, _ value: Double) {
        withUnsafeBytes(of: value.bitPattern.littleEndian) {
            data.append(contentsOf: $0)
        }
    }
}

#if !targetEnvironment(simulator)
enum RealityKitColmapExporter {
    private struct AlignedSample {
        let sampleID: Int
        let pose: PhotogrammetrySession.Pose
        let sourceURL: URL
        let metadata: RealityKitColmapImageMetadata
        let camera: RealityKitColmapCamera
    }

    static func export(
        poses: PhotogrammetrySession.Poses,
        pointCloud: PhotogrammetrySession.PointCloud,
        knownCamerasByFilename: [String: RealityKitColmapCamera],
        exportsTrainingMasks: Bool,
        to datasetDirectory: URL,
        progressHandler: @escaping @MainActor @Sendable (
            Double,
            RealityKitColmapExportStage
        ) -> Void
    ) async throws -> RealityKitColmapExportResult {
        try Task.checkCancellation()
        var points: [RealityKitColmapPoint] = []
        points.reserveCapacity(pointCloud.points.count)
        for (index, point) in pointCloud.points.enumerated() {
            if index.isMultiple(of: 4_096) {
                try Task.checkCancellation()
            }
            guard point.position.x.isFinite,
                  point.position.y.isFinite,
                  point.position.z.isFinite else {
                throw RealityKitColmapExportError.invalidSparsePoint(index: index)
            }
            points.append(
                RealityKitColmapPoint(
                    position: point.position,
                    color: SIMD3<Float>(
                        Float(point.color.x),
                        Float(point.color.y),
                        Float(point.color.z)
                    )
                )
            )
        }
        guard !points.isEmpty else {
            throw RealityKitColmapExportError.noSparsePoints
        }

        var samples: [AlignedSample] = []
        samples.reserveCapacity(poses.posesBySample.count)
        for (sampleID, pose) in poses.posesBySample {
            try Task.checkCancellation()
            guard let sourceURL = poses.urlsBySample[sampleID] else {
                throw RealityKitColmapExportError.missingSourceURL(
                    sampleID: sampleID
                )
            }
            guard isFinite(pose.transform.matrix) else {
                throw RealityKitColmapExportError.invalidPose(sampleID: sampleID)
            }
            let metadata = try imageMetadata(for: sourceURL)
            try validate(metadata.orientation, filename: sourceURL.lastPathComponent)
            let camera = try camera(
                for: pose,
                sampleID: sampleID,
                sourceURL: sourceURL,
                metadata: metadata,
                knownCamera: knownCamerasByFilename[sourceURL.lastPathComponent]
            )
            samples.append(AlignedSample(
                sampleID: sampleID,
                pose: pose,
                sourceURL: sourceURL,
                metadata: metadata,
                camera: camera
            ))
        }
        samples.sort { $0.sampleID < $1.sampleID }
        guard !samples.isEmpty else {
            throw RealityKitColmapExportError.noAlignedImages
        }
        guard samples.count >= 3 else {
            throw RealityKitColmapExportError.insufficientAlignedImages(
                samples.count
            )
        }

        let fileManager = FileManager.default
        let parentDirectory = datasetDirectory.deletingLastPathComponent()
        try fileManager.createDirectory(
            at: parentDirectory,
            withIntermediateDirectories: true
        )
        guard !fileManager.fileExists(atPath: datasetDirectory.path) else {
            throw RealityKitColmapExportError.outputAlreadyExists
        }
        let stagingDirectory = parentDirectory.appending(
            path: ".\(datasetDirectory.lastPathComponent).staging-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        let imagesDirectory = stagingDirectory.appending(
            path: "images",
            directoryHint: .isDirectory
        )
        let masksDirectory = stagingDirectory.appending(
            path: "masks",
            directoryHint: .isDirectory
        )
        let sparseDirectory = stagingDirectory
            .appending(path: "sparse", directoryHint: .isDirectory)
            .appending(path: "0", directoryHint: .isDirectory)
        try fileManager.createDirectory(
            at: imagesDirectory,
            withIntermediateDirectories: true
        )
        if exportsTrainingMasks {
            try fileManager.createDirectory(
                at: masksDirectory,
                withIntermediateDirectories: true
            )
        }
        var committed = false
        defer {
            if !committed {
                try? fileManager.removeItem(at: stagingDirectory)
            }
        }

        let context = CIContext(options: [.cacheIntermediates: false])
        var entries: [RealityKitColmapImageEntry] = []
        entries.reserveCapacity(samples.count)
        let totalWorkUnits = samples.count +
            (exportsTrainingMasks ? samples.count : 0) + 2
        var completedWorkUnits = 0
        await progressHandler(0, .images)
        for (index, sample) in samples.enumerated() {
            try Task.checkCancellation()
            let imageID = index + 1
            let filename = try RealityKitColmapExportBuilder.exportImage(
                source: sample.sourceURL,
                to: imagesDirectory,
                imageID: imageID,
                orientation: sample.metadata.orientation,
                context: context
            )
            entries.append(
                RealityKitColmapImageEntry(
                    imageID: imageID,
                    filename: filename,
                    cameraToWorld: sample.pose.transform.matrix,
                    camera: sample.camera,
                    orientation: sample.metadata.orientation
                )
            )
            completedWorkUnits += 1
            await progressHandler(
                Double(completedWorkUnits) / Double(totalWorkUnits),
                .images
            )
        }

        if exportsTrainingMasks {
            await progressHandler(
                Double(completedWorkUnits) / Double(totalWorkUnits),
                .masks
            )
            for entry in entries {
                try Task.checkCancellation()
                let source = imagesDirectory.appending(path: entry.filename)
                let destination = masksDirectory.appending(
                    path: RealityKitColmapExportBuilder.maskFilename(
                        forExportedImage: entry.filename
                    )
                )
                try await exportForegroundMask(
                    source: source,
                    destination: destination,
                    expectedWidth: entry.camera.width,
                    expectedHeight: entry.camera.height,
                    context: context
                )
                completedWorkUnits += 1
                await progressHandler(
                    Double(completedWorkUnits) / Double(totalWorkUnits),
                    .masks
                )
            }
        }

        try Task.checkCancellation()
        await progressHandler(
            Double(completedWorkUnits) / Double(totalWorkUnits),
            .sparseModel
        )
        try RealityKitColmapExportBuilder.writeSparseFiles(
            to: sparseDirectory,
            entries: entries,
            points: points
        )
        completedWorkUnits += 1
        await progressHandler(
            Double(completedWorkUnits) / Double(totalWorkUnits),
            .sparseModel
        )
        try Task.checkCancellation()
        await progressHandler(
            Double(completedWorkUnits) / Double(totalWorkUnits),
            .publishing
        )
        try fileManager.moveItem(at: stagingDirectory, to: datasetDirectory)
        committed = true
        completedWorkUnits += 1
        await progressHandler(
            Double(completedWorkUnits) / Double(totalWorkUnits),
            .publishing
        )
        return RealityKitColmapExportResult(
            datasetDirectory: datasetDirectory,
            alignedImageCount: entries.count,
            cameraCount: entries.count,
            pointCount: points.count,
            maskCount: exportsTrainingMasks ? entries.count : 0
        )
    }

    private static func camera(
        for pose: PhotogrammetrySession.Pose,
        sampleID: Int,
        sourceURL: URL,
        metadata: RealityKitColmapImageMetadata,
        knownCamera: RealityKitColmapCamera?
    ) throws -> RealityKitColmapCamera {
        let raw: RealityKitColmapCamera
        if let knownCamera {
            raw = knownCamera
        } else if #available(iOS 26.0, *), let intrinsics = pose.intrinsics {
            raw = RealityKitColmapCamera(
                width: metadata.width,
                height: metadata.height,
                fx: Double(intrinsics[0][0]),
                fy: Double(intrinsics[1][1]),
                cx: Double(intrinsics[2][0]),
                cy: Double(intrinsics[2][1])
            )
        } else {
            throw RealityKitColmapExportError.missingIntrinsics(
                sampleID: sampleID,
                filename: sourceURL.lastPathComponent
            )
        }
        let oriented = RealityKitColmapExportBuilder.camera(
            raw,
            applying: metadata.orientation
        )
        let expectedDimensions = switch metadata.orientation {
        case .left, .right:
            (width: metadata.height, height: metadata.width)
        default:
            (width: metadata.width, height: metadata.height)
        }
        guard oriented.width == expectedDimensions.width,
              oriented.height == expectedDimensions.height,
              oriented.height > 0,
              oriented.fx.isFinite,
              oriented.fy.isFinite,
              oriented.cx.isFinite,
              oriented.cy.isFinite,
              oriented.fx > 0,
              oriented.fy > 0 else {
            throw RealityKitColmapExportError.invalidCamera(
                filename: sourceURL.lastPathComponent
            )
        }
        return oriented
    }

    private static func isFinite(_ matrix: simd_float4x4) -> Bool {
        for column in 0..<4 {
            for row in 0..<4 where !matrix[column][row].isFinite {
                return false
            }
        }
        return true
    }

    private static func imageMetadata(
        for url: URL
    ) throws -> RealityKitColmapImageMetadata {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any] else {
            throw RealityKitColmapExportError.imageMetadataReadFailed(
                filename: url.lastPathComponent
            )
        }
        let rawOrientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?
            .intValue ?? 1
        guard let orientation = RealityKitColmapEXIFOrientation(
            rawValue: rawOrientation
        ),
              let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?
                .intValue,
              let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?
                .intValue,
              width > 0,
              height > 0 else {
            throw RealityKitColmapExportError.imageMetadataReadFailed(
                filename: url.lastPathComponent
            )
        }
        return RealityKitColmapImageMetadata(
            width: width,
            height: height,
            orientation: orientation
        )
    }

    private static func validate(
        _ orientation: RealityKitColmapEXIFOrientation,
        filename: String
    ) throws {
        guard !orientation.isMirrored else {
            throw RealityKitColmapExportError.unsupportedOrientation(
                filename: filename,
                orientation: orientation.rawValue
            )
        }
    }

    private static func exportForegroundMask(
        source: URL,
        destination: URL,
        expectedWidth: Int,
        expectedHeight: Int,
        context: CIContext
    ) async throws {
        let task = Task.detached(priority: .userInitiated) {
            try Task.checkCancellation()
            guard let imageSource = CGImageSourceCreateWithURL(
                source as CFURL,
                nil
            ),
                  let image = CGImageSourceCreateImageAtIndex(
                    imageSource,
                    0,
                    nil
                  ) else {
                throw RealityKitColmapExportError.imageReadFailed(
                    filename: source.lastPathComponent
                )
            }
            guard image.width == expectedWidth,
                  image.height == expectedHeight else {
                throw RealityKitColmapExportError.invalidCamera(
                    filename: source.lastPathComponent
                )
            }

            let request = VNGenerateForegroundInstanceMaskRequest()
            let handler = VNImageRequestHandler(cgImage: image, options: [:])
            do {
                try handler.perform([request])
            } catch {
                try Task.checkCancellation()
                throw RealityKitColmapExportError.maskGenerationFailed(
                    filename: source.lastPathComponent
                )
            }
            try Task.checkCancellation()
            guard let result = request.results?.first else {
                throw RealityKitColmapExportError.maskGenerationFailed(
                    filename: source.lastPathComponent
                )
            }
            guard !result.allInstances.isEmpty else {
                throw RealityKitColmapExportError.emptyTrainingMask(
                    filename: source.lastPathComponent
                )
            }

            let maskBuffer: CVPixelBuffer
            do {
                maskBuffer = try result.generateScaledMaskForImage(
                    forInstances: result.allInstances,
                    from: handler
                )
            } catch {
                try Task.checkCancellation()
                throw RealityKitColmapExportError.maskGenerationFailed(
                    filename: source.lastPathComponent
                )
            }
            try Task.checkCancellation()
            let maskImage = CIImage(cvPixelBuffer: maskBuffer)
                .applyingFilter(
                    "CIColorThreshold",
                    parameters: ["inputThreshold": 0.5]
                )
            guard let renderedMask = context.createCGImage(
                maskImage,
                from: maskImage.extent,
                format: .L8,
                colorSpace: CGColorSpaceCreateDeviceGray()
            ) else {
                throw RealityKitColmapExportError.maskRenderFailed(
                    filename: source.lastPathComponent
                )
            }
            guard renderedMask.width == image.width,
                  renderedMask.height == image.height,
                  renderedMask.bitsPerComponent == 8,
                  renderedMask.bitsPerPixel == 8,
                  renderedMask.alphaInfo == .none else {
                throw RealityKitColmapExportError.maskRenderFailed(
                    filename: source.lastPathComponent
                )
            }

            try Task.checkCancellation()
            guard let imageDestination = CGImageDestinationCreateWithURL(
                destination as CFURL,
                UTType.png.identifier as CFString,
                1,
                nil
            ) else {
                throw RealityKitColmapExportError.maskEncodingFailed(
                    filename: source.lastPathComponent
                )
            }
            CGImageDestinationAddImage(imageDestination, renderedMask, nil)
            guard CGImageDestinationFinalize(imageDestination) else {
                throw RealityKitColmapExportError.maskEncodingFailed(
                    filename: source.lastPathComponent
                )
            }
        }

        try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }
}
#endif
