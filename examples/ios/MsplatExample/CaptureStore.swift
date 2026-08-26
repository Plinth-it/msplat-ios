@preconcurrency import ARKit
import CoreImage
import CoreVideo
import Foundation
import ImageIO
import Msplat
import UniformTypeIdentifiers
@preconcurrency import Vision

final class OwnedPixelBuffer: @unchecked Sendable {
    let value: CVPixelBuffer

    init(_ value: CVPixelBuffer) {
        self.value = value
    }
}

struct CaptureFrameCandidate: Sendable {
    let image: OwnedPixelBuffer
    let depth: OwnedPixelBuffer?
    let confidence: OwnedPixelBuffer?
    let displayOrientation: CaptureDisplayOrientation
    let calibration: CaptureCalibrationRecord
    let cameraToWorld: simd_float4x4
    let timestamp: TimeInterval
    let exposureDuration: TimeInterval
    let trackingState: String
    let rawFeaturePoints: [SIMD3<Float>]
    let subjectWorldPosition: SIMD3<Float>?
}

struct CaptureCommit: Sendable {
    let record: CaptureFrameRecord
    let totalPointCount: Int
}

struct CaptureSubjectSelection: Sendable {
    let worldPosition: SIMD3<Float>
}

actor CaptureSubjectSelector {
    func selectSubject(
        in candidate: CaptureFrameCandidate,
        normalizedImagePoint: CGPoint
    ) throws -> CaptureSubjectSelection {
        guard (0...1).contains(normalizedImagePoint.x),
              (0...1).contains(normalizedImagePoint.y) else {
            throw CaptureFailure.subjectNotFound
        }
        guard let depthBuffer = candidate.depth?.value else {
            throw CaptureFailure.subjectDepthUnavailable
        }

        let handler = VNImageRequestHandler(
            cvPixelBuffer: candidate.image.value,
            orientation: .up
        )
        let request = VNGenerateForegroundInstanceMaskRequest()
        try handler.perform([request])
        guard let observation = request.results?.first,
              let hit = maskHit(
                in: observation.instanceMask,
                normalizedImagePoint: normalizedImagePoint
              ) else {
            throw CaptureFailure.subjectNotFound
        }

        guard let sample = medianDepth(
            in: depthBuffer,
            confidence: candidate.confidence?.value,
            instanceMask: observation.instanceMask,
            selectedLabel: hit.label,
            normalizedImagePoint: hit.normalizedPoint
        ) else {
            throw CaptureFailure.subjectDepthUnavailable
        }
        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)
        let depthPixel = SIMD2<Float>(
            Float(hit.normalizedPoint.x) * Float(width),
            Float(hit.normalizedPoint.y) * Float(height)
        )
        guard let worldPosition = CaptureGeometry.worldPoint(
            depth: sample,
            depthPixel: depthPixel,
            depthWidth: width,
            depthHeight: height,
            calibration: candidate.calibration,
            cameraToWorld: candidate.cameraToWorld
        ) else {
            throw CaptureFailure.subjectDepthUnavailable
        }
        return CaptureSubjectSelection(worldPosition: worldPosition)
    }
}

actor CaptureStore {
    private struct VoxelKey: Hashable, Comparable, Sendable {
        let x: Int32
        let y: Int32
        let z: Int32

        static func < (lhs: Self, rhs: Self) -> Bool {
            if lhs.x != rhs.x { return lhs.x < rhs.x }
            if lhs.y != rhs.y { return lhs.y < rhs.y }
            return lhs.z < rhs.z
        }
    }

    private struct FusedPoint: Sendable {
        var positionSum: SIMD3<Float>
        var redSum: UInt64
        var greenSum: UInt64
        var blueSum: UInt64
        var count: UInt32

        var position: SIMD3<Float> {
            positionSum / Float(max(count, 1))
        }

        var color: SIMD3<UInt8> {
            let divisor = UInt64(max(count, 1))
            return SIMD3(
                UInt8(clamping: Int(redSum / divisor)),
                UInt8(clamping: Int(greenSum / divisor)),
                UInt8(clamping: Int(blueSum / divisor))
            )
        }
    }

    private struct FusionDelta {
        let contributions: [VoxelKey: FusedPoint]
        let addedPointCount: Int
    }

    private struct MaskResult {
        let bytes: [UInt8]
        let binaryBytes: [UInt8]
        let width: Int
        let height: Int
        let confidence: Float
    }

    private struct NerfstudioManifest: Encodable {
        struct Frame: Encodable {
            let filePath: String
            let maskPath: String?
            let width: Int
            let height: Int
            let flX: Float
            let flY: Float
            let cx: Float
            let cy: Float
            let k1: Float
            let k2: Float
            let p1: Float
            let p2: Float
            let transformMatrix: [[Float]]

            enum CodingKeys: String, CodingKey {
                case filePath = "file_path"
                case maskPath = "mask_path"
                case width = "w"
                case height = "h"
                case flX = "fl_x"
                case flY = "fl_y"
                case cx, cy, k1, k2, p1, p2
                case transformMatrix = "transform_matrix"
            }
        }

        let cameraModel = "OPENCV"
        let plyFilePath: String
        let frames: [Frame]

        enum CodingKeys: String, CodingKey {
            case cameraModel = "camera_model"
            case plyFilePath = "ply_file_path"
            case frames
        }
    }

    private static let maximumPointCount = 150_000
    private let mode: CaptureMode
    private let captureID: UUID
    private let createdAt: Date
    private let rootURL: URL
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let voxelSize: Float
    private var frames: [CaptureFrameRecord] = []
    private var fusedPoints: [VoxelKey: FusedPoint] = [:]
    private var isDiscarded = false

    init(mode: CaptureMode, baseDirectory: URL = .documentsDirectory) throws {
        self.mode = mode
        captureID = UUID()
        createdAt = Date()
        voxelSize = mode == .object ? 0.005 : 0.02
        rootURL = baseDirectory
            .appending(path: "Captures", directoryHint: .isDirectory)
            .appending(
                path: "Capture-\(captureID.uuidString)",
                directoryHint: .isDirectory
            )

        do {
            for directory in ["images", "masks", "masks_soft", "pointcloud"] {
                try FileManager.default.createDirectory(
                    at: rootURL.appending(path: directory, directoryHint: .isDirectory),
                    withIntermediateDirectories: true
                )
            }
            FileManager.default.createFile(
                atPath: rootURL.appending(path: "capture_journal.jsonl").path,
                contents: nil
            )
        } catch {
            throw CaptureFailure.persistence(error.localizedDescription)
        }
    }

    func accept(_ candidate: CaptureFrameCandidate) throws -> CaptureCommit {
        guard !isDiscarded else { throw CancellationError() }
        let frameNumber = frames.count + 1
        let frameID = String(format: "frame_%06d", frameNumber)
        let imagePath = "images/\(frameID).png"
        let maskPath = mode == .object ? "masks/\(frameID).png" : nil
        let softMaskPath = mode == .object ? "masks_soft/\(frameID).png" : nil
        let imageURL = rootURL.appending(path: imagePath)
        let maskURL = maskPath.map { rootURL.appending(path: $0) }
        let softMaskURL = softMaskPath.map { rootURL.appending(path: $0) }

        let imageWidth = CVPixelBufferGetWidth(candidate.image.value)
        let imageHeight = CVPixelBufferGetHeight(candidate.image.value)
        guard imageWidth == candidate.calibration.width,
              imageHeight == candidate.calibration.height else {
            throw CaptureFailure.invalidFrame(
                "encoded raster \(imageWidth)x\(imageHeight) does not match calibration " +
                    "\(candidate.calibration.width)x\(candidate.calibration.height)"
            )
        }

        let mask: MaskResult?
        if mode == .object {
            guard let subjectWorldPosition = candidate.subjectWorldPosition else {
                throw CaptureFailure.invalidFrame("object capture has no selected subject")
            }
            mask = try objectMask(
                image: candidate.image.value,
                calibration: candidate.calibration,
                cameraToWorld: candidate.cameraToWorld,
                subjectWorldPosition: subjectWorldPosition
            )
        } else {
            mask = nil
        }

        let orientedGeometry = CaptureGeometry.orientedGeometry(
            calibration: candidate.calibration,
            cameraToWorld: candidate.cameraToWorld,
            orientation: candidate.displayOrientation
        )
        let orientedMask: MaskResult?
        if let mask {
            let soft = try CaptureGeometry.orientedInterleavedBytes(
                mask.bytes,
                width: mask.width,
                height: mask.height,
                components: 1,
                orientation: candidate.displayOrientation
            )
            let binary = try CaptureGeometry.orientedInterleavedBytes(
                mask.binaryBytes,
                width: mask.width,
                height: mask.height,
                components: 1,
                orientation: candidate.displayOrientation
            )
            guard soft.width == binary.width,
                  soft.height == binary.height,
                  soft.width == orientedGeometry.calibration.width,
                  soft.height == orientedGeometry.calibration.height else {
                throw CaptureFailure.invalidFrame(
                    "oriented RGB, mask, and calibration dimensions do not match"
                )
            }
            orientedMask = MaskResult(
                bytes: soft.bytes,
                binaryBytes: binary.bytes,
                width: soft.width,
                height: soft.height,
                confidence: mask.confidence
            )
        } else {
            orientedMask = nil
        }

        let fusion = try stageFusion(from: candidate, mask: mask)

        var committedURLs: [URL] = []
        var didCommit = false
        defer {
            if !didCommit {
                for url in committedURLs {
                    try? FileManager.default.removeItem(at: url)
                }
            }
        }

        try writeImage(
            candidate.image.value,
            orientation: candidate.displayOrientation,
            expectedWidth: orientedGeometry.calibration.width,
            expectedHeight: orientedGeometry.calibration.height,
            atomicallyTo: imageURL
        )
        committedURLs.append(imageURL)
        if let orientedMask, let maskURL, let softMaskURL {
            try writeGrayscalePNG(
                orientedMask.binaryBytes,
                width: orientedMask.width,
                height: orientedMask.height,
                atomicallyTo: maskURL
            )
            committedURLs.append(maskURL)
            try writeGrayscalePNG(
                orientedMask.bytes,
                width: orientedMask.width,
                height: orientedMask.height,
                atomicallyTo: softMaskURL
            )
            committedURLs.append(softMaskURL)
        }

        let record = CaptureFrameRecord(
            id: frameID,
            imagePath: imagePath,
            maskPath: maskPath,
            softMaskPath: softMaskPath,
            displayOrientation: candidate.displayOrientation,
            calibration: orientedGeometry.calibration,
            cameraToWorld: CaptureGeometry.rowMajorCameraToWorld(
                orientedGeometry.cameraToWorld
            ),
            timestamp: candidate.timestamp,
            exposureDurationSeconds: candidate.exposureDuration,
            trackingState: candidate.trackingState,
            maskConfidence: mask?.confidence,
            fusedPointCount: fusion.addedPointCount
        )
        try appendJournal(record)
        apply(fusion)
        frames.append(record)
        didCommit = true
        return CaptureCommit(record: record, totalPointCount: fusedPoints.count)
    }

    func finalize() throws -> CapturedDataset {
        guard !isDiscarded else { throw CancellationError() }
        guard frames.count >= 3, !fusedPoints.isEmpty else {
            throw CaptureFailure.insufficientCapture(
                frameCount: frames.count,
                pointCount: fusedPoints.count
            )
        }

        let orderedPoints = fusedPoints.sorted { $0.key < $1.key }.map(\.value)
        let pointCloudPath = "pointcloud/lidar_colored.ply"
        try writePointCloud(
            orderedPoints,
            atomicallyTo: rootURL.appending(path: pointCloudPath)
        )

        let manifest = CaptureManifest(
            formatVersion: CaptureManifest.currentFormatVersion,
            id: captureID,
            createdAt: createdAt,
            mode: mode,
            coordinateSystem: "arkit_world_meters",
            pointCloudPath: pointCloudPath,
            frames: frames
        )
        try writeJSON(manifest, atomicallyTo: rootURL.appending(path: "capture_metadata.json"))
        try writeNerfstudioManifest(pointCloudPath: pointCloudPath)

        var xyz: [Float] = []
        var rgb: [UInt8] = []
        xyz.reserveCapacity(orderedPoints.count * 3)
        rgb.reserveCapacity(orderedPoints.count * 3)
        for point in orderedPoints {
            let position = point.position
            let color = point.color
            xyz.append(contentsOf: [position.x, position.y, position.z])
            rgb.append(contentsOf: [color.x, color.y, color.z])
        }

        let descriptorFrames = try frames.map { frame in
            let trainingMask: DatasetTrainingMask?
            if let maskURL = frame.softMaskURL(under: rootURL) {
                trainingMask = try DatasetTrainingMask(
                    url: maskURL,
                    coverageChannel: .luminance
                )
            } else {
                trainingMask = nil
            }
            return try DatasetFrame(
                id: frame.id,
                calibrationID: frame.id,
                imageURL: frame.imageURL(under: rootURL),
                rasterOrientation: .encodedPixels,
                calibration: frame.calibration.descriptorCalibration(),
                cameraToWorld: CameraPose(elements: frame.cameraToWorld),
                trainingMask: trainingMask
            )
        }
        let descriptor = try DatasetDescriptor(
            frames: descriptorFrames,
            points: DatasetSparsePointSet(xyz: xyz, rgb: rgb),
            provenance: DatasetProvenance(
                adapter: "arkit-capture",
                source: rootURL.path
            )
        )
        let cxValues = frames.map(\.calibration.cx)
        let cyValues = frames.map(\.calibration.cy)
        guard let minimumCX = cxValues.min(), let maximumCX = cxValues.max(),
              let minimumCY = cyValues.min(), let maximumCY = cyValues.max() else {
            throw CaptureFailure.invalidFrame("capture contains no calibration values")
        }
        let review = CaptureReviewSnapshot(
            frameCount: frames.count,
            pointCount: orderedPoints.count,
            minimumCX: minimumCX,
            maximumCX: maximumCX,
            minimumCY: minimumCY,
            maximumCY: maximumCY
        )
        return CapturedDataset(
            rootURL: rootURL,
            descriptor: descriptor,
            manifest: manifest,
            reviewSnapshot: review
        )
    }

    func discard() throws {
        isDiscarded = true
        frames.removeAll(keepingCapacity: false)
        fusedPoints.removeAll(keepingCapacity: false)
        do {
            try removeIfPresent(rootURL)
        } catch {
            throw CaptureFailure.persistence(error.localizedDescription)
        }
    }

    private func objectMask(
        image: CVPixelBuffer,
        calibration: CaptureCalibrationRecord,
        cameraToWorld: simd_float4x4,
        subjectWorldPosition: SIMD3<Float>
    ) throws -> MaskResult {
        guard let projectedPoint = CaptureGeometry.normalizedImagePoint(
            worldPoint: subjectWorldPosition,
            calibration: calibration,
            cameraToWorld: cameraToWorld
        ),
              (-0.05...1.05).contains(projectedPoint.x),
              (-0.05...1.05).contains(projectedPoint.y) else {
            throw CaptureFailure.invalidFrame("selected subject is outside the image")
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: image, orientation: .up)
        let request = VNGenerateForegroundInstanceMaskRequest()
        try handler.perform([request])
        guard let observation = request.results?.first,
              let hit = maskHit(
                in: observation.instanceMask,
                normalizedImagePoint: projectedPoint
              ) else {
            throw CaptureFailure.invalidFrame("Vision could not track the selected subject")
        }
        let selectedInstances = IndexSet(integer: hit.label)
        let scaledMask = try observation.generateScaledMaskForImage(
            forInstances: selectedInstances,
            from: handler
        )
        let width = CVPixelBufferGetWidth(scaledMask)
        let height = CVPixelBufferGetHeight(scaledMask)
        guard width == calibration.width, height == calibration.height else {
            throw CaptureFailure.invalidFrame(
                "Vision mask dimensions do not match the encoded image"
            )
        }
        let bytes = try normalizedMaskBytes(from: scaledMask)
        let binary = bytes.map { $0 >= 128 ? UInt8.max : 0 }
        let covered = binary.reduce(into: 0) { count, value in
            if value != 0 { count += 1 }
        }
        guard covered > 0 else {
            throw CaptureFailure.invalidFrame("Vision returned an empty subject mask")
        }
        return MaskResult(
            bytes: bytes,
            binaryBytes: binary,
            width: width,
            height: height,
            confidence: Float(covered) / Float(binary.count)
        )
    }

    private func stageFusion(
        from candidate: CaptureFrameCandidate,
        mask: MaskResult?
    ) throws -> FusionDelta {
        var contributions: [VoxelKey: FusedPoint] = [:]
        var addedPointCount = 0

        func stage(point: SIMD3<Float>, color: SIMD3<UInt8>) {
            guard let key = voxelKey(for: point) else { return }
            if var current = contributions[key] {
                current.positionSum += point
                current.redSum += UInt64(color.x)
                current.greenSum += UInt64(color.y)
                current.blueSum += UInt64(color.z)
                current.count &+= 1
                contributions[key] = current
                return
            }

            let alreadyFused = fusedPoints[key] != nil
            guard alreadyFused ||
                    fusedPoints.count + addedPointCount < Self.maximumPointCount else {
                return
            }
            contributions[key] = FusedPoint(
                positionSum: point,
                redSum: UInt64(color.x),
                greenSum: UInt64(color.y),
                blueSum: UInt64(color.z),
                count: 1
            )
            if !alreadyFused { addedPointCount += 1 }
        }

        guard let depth = candidate.depth?.value else {
            guard mode == .scene else {
                throw CaptureFailure.invalidFrame(
                    "object capture requires scene depth for every accepted frame"
                )
            }
            for point in candidate.rawFeaturePoints {
                stage(point: point, color: SIMD3(repeating: 160))
            }
            return FusionDelta(
                contributions: contributions,
                addedPointCount: addedPointCount
            )
        }

        let depthWidth = CVPixelBufferGetWidth(depth)
        let depthHeight = CVPixelBufferGetHeight(depth)
        guard depthWidth > 0, depthHeight > 0,
              CVPixelBufferGetPixelFormatType(depth) ==
                kCVPixelFormatType_DepthFloat32 else {
            throw CaptureFailure.invalidFrame("scene depth is not Float32")
        }
        let imageWidth = candidate.calibration.width
        let imageHeight = candidate.calibration.height
        let rgba = try rgbaBytes(from: candidate.image.value)
        let confidence = candidate.confidence?.value
        if let confidence {
            guard CVPixelBufferGetWidth(confidence) == depthWidth,
                  CVPixelBufferGetHeight(confidence) == depthHeight,
                  CVPixelBufferGetPixelFormatType(confidence) ==
                    kCVPixelFormatType_OneComponent8 else {
                throw CaptureFailure.invalidFrame(
                    "scene-depth confidence does not match the depth map"
                )
            }
        }

        CVPixelBufferLockBaseAddress(depth, .readOnly)
        if let confidence {
            CVPixelBufferLockBaseAddress(confidence, .readOnly)
        }
        defer {
            if let confidence {
                CVPixelBufferUnlockBaseAddress(confidence, .readOnly)
            }
            CVPixelBufferUnlockBaseAddress(depth, .readOnly)
        }
        guard let depthBase = CVPixelBufferGetBaseAddress(depth) else {
            throw CaptureFailure.invalidFrame("scene depth has no readable storage")
        }
        let depthStride = CVPixelBufferGetBytesPerRow(depth) / MemoryLayout<Float>.stride
        let depthValues = depthBase.assumingMemoryBound(to: Float.self)
        let confidenceValues = confidence.flatMap(CVPixelBufferGetBaseAddress)?
            .assumingMemoryBound(to: UInt8.self)
        let confidenceStride = confidence.map(CVPixelBufferGetBytesPerRow) ?? 0
        let sampleStride = 3

        for y in stride(from: 0, to: depthHeight, by: sampleStride) {
            for x in stride(from: 0, to: depthWidth, by: sampleStride) {
                if let confidenceValues,
                   confidenceValues[y * confidenceStride + x] < 1 {
                    continue
                }
                let imageX = min(
                    imageWidth - 1,
                    Int((Float(x) + 0.5) * Float(imageWidth) / Float(depthWidth))
                )
                let imageY = min(
                    imageHeight - 1,
                    Int((Float(y) + 0.5) * Float(imageHeight) / Float(depthHeight))
                )
                if let mask,
                   mask.binaryBytes[imageY * mask.width + imageX] == 0 {
                    continue
                }
                let depthValue = depthValues[y * depthStride + x]
                guard let worldPoint = CaptureGeometry.worldPoint(
                    depth: depthValue,
                    depthPixel: SIMD2(Float(x) + 0.5, Float(y) + 0.5),
                    depthWidth: depthWidth,
                    depthHeight: depthHeight,
                    calibration: candidate.calibration,
                    cameraToWorld: candidate.cameraToWorld
                ) else {
                    continue
                }
                let pixelOffset = (imageY * imageWidth + imageX) * 4
                let color = SIMD3(
                    rgba[pixelOffset],
                    rgba[pixelOffset + 1],
                    rgba[pixelOffset + 2]
                )
                stage(point: worldPoint, color: color)
            }
        }
        return FusionDelta(
            contributions: contributions,
            addedPointCount: addedPointCount
        )
    }

    private func apply(_ fusion: FusionDelta) {
        for (key, contribution) in fusion.contributions {
            if var current = fusedPoints[key] {
                current.positionSum += contribution.positionSum
                current.redSum += contribution.redSum
                current.greenSum += contribution.greenSum
                current.blueSum += contribution.blueSum
                current.count &+= contribution.count
                fusedPoints[key] = current
            } else {
                fusedPoints[key] = contribution
            }
        }
    }

    private func voxelKey(for point: SIMD3<Float>) -> VoxelKey? {
        guard point.x.isFinite, point.y.isFinite, point.z.isFinite else {
            return nil
        }
        let scaled = point / voxelSize
        let components = [scaled.x.rounded(.down), scaled.y.rounded(.down), scaled.z.rounded(.down)]
        guard components.allSatisfy({ $0 >= Float(Int32.min) && $0 <= Float(Int32.max) }) else {
            return nil
        }
        return VoxelKey(
            x: Int32(components[0]),
            y: Int32(components[1]),
            z: Int32(components[2])
        )
    }

    private func writeImage(
        _ buffer: CVPixelBuffer,
        orientation: CaptureDisplayOrientation,
        expectedWidth: Int,
        expectedHeight: Int,
        atomicallyTo url: URL
    ) throws {
        let oriented = CIImage(cvPixelBuffer: buffer).oriented(
            orientation.cgImagePropertyOrientation
        )
        let extent = oriented.extent.integral
        guard Int(extent.width) == expectedWidth,
              Int(extent.height) == expectedHeight else {
            throw CaptureFailure.invalidFrame(
                "oriented RGB dimensions do not match transformed calibration"
            )
        }
        let image = oriented.transformed(by: CGAffineTransform(
            translationX: -extent.minX,
            y: -extent.minY
        ))
        let temporary = temporaryURL(for: url)
        try removeIfPresent(temporary)
        do {
            try context.writePNGRepresentation(
                of: image,
                to: temporary,
                format: .RGBA8,
                colorSpace: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
            )
            try FileManager.default.moveItem(at: temporary, to: url)
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw CaptureFailure.persistence(error.localizedDescription)
        }
    }

    private func rgbaBytes(from buffer: CVPixelBuffer) throws -> [UInt8] {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let image = CIImage(cvPixelBuffer: buffer)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        bytes.withUnsafeMutableBytes { storage in
            guard let baseAddress = storage.baseAddress else { return }
            context.render(
                image,
                toBitmap: baseAddress,
                rowBytes: width * 4,
                bounds: CGRect(x: 0, y: 0, width: width, height: height),
                format: .RGBA8,
                colorSpace: colorSpace
            )
        }
        return bytes
    }

    private func writeGrayscalePNG(
        _ bytes: [UInt8],
        width: Int,
        height: Int,
        atomicallyTo url: URL
    ) throws {
        guard bytes.count == width * height,
              let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 8,
                bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo(rawValue: 0),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            throw CaptureFailure.persistence("could not create a grayscale mask image")
        }
        let temporary = temporaryURL(for: url)
        try removeIfPresent(temporary)
        guard let destination = CGImageDestinationCreateWithURL(
            temporary as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw CaptureFailure.persistence("could not create a PNG destination")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            try? FileManager.default.removeItem(at: temporary)
            throw CaptureFailure.persistence("could not encode a PNG mask")
        }
        do {
            try FileManager.default.moveItem(at: temporary, to: url)
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw CaptureFailure.persistence(error.localizedDescription)
        }
    }

    private func appendJournal(_ record: CaptureFrameRecord) throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        var line = try encoder.encode(record)
        line.append(0x0A)
        let url = rootURL.appending(path: "capture_journal.jsonl")
        do {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
        } catch {
            throw CaptureFailure.persistence(error.localizedDescription)
        }
    }

    private func writeNerfstudioManifest(pointCloudPath: String) throws {
        let encodedFrames = frames.map { frame in
            let values = frame.cameraToWorld
            let matrix = stride(from: 0, to: 16, by: 4).map { start in
                Array(values[start..<(start + 4)])
            }
            return NerfstudioManifest.Frame(
                filePath: frame.imagePath,
                maskPath: frame.maskPath,
                width: frame.calibration.width,
                height: frame.calibration.height,
                flX: frame.calibration.fx,
                flY: frame.calibration.fy,
                cx: frame.calibration.cx,
                cy: frame.calibration.cy,
                k1: 0,
                k2: 0,
                p1: 0,
                p2: 0,
                transformMatrix: matrix
            )
        }
        try writeJSON(
            NerfstudioManifest(
                plyFilePath: pointCloudPath,
                frames: encodedFrames
            ),
            atomicallyTo: rootURL.appending(path: "transforms.json")
        )
    }

    private func writeJSON<Value: Encodable>(_ value: Value, atomicallyTo url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        encoder.keyEncodingStrategy = .convertToSnakeCase
        do {
            try encoder.encode(value).write(to: url, options: .atomic)
        } catch {
            throw CaptureFailure.persistence(error.localizedDescription)
        }
    }

    private func writePointCloud(
        _ points: [FusedPoint],
        atomicallyTo url: URL
    ) throws {
        let header = """
        ply
        format binary_little_endian 1.0
        element vertex \(points.count)
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header

        """
        guard let headerData = header.data(using: .utf8) else {
            throw CaptureFailure.persistence("could not encode the PLY header")
        }
        var data = headerData
        data.reserveCapacity(headerData.count + points.count * 15)
        for point in points {
            appendLittleEndian(point.position.x, to: &data)
            appendLittleEndian(point.position.y, to: &data)
            appendLittleEndian(point.position.z, to: &data)
            data.append(contentsOf: [point.color.x, point.color.y, point.color.z])
        }
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw CaptureFailure.persistence(error.localizedDescription)
        }
    }

    private func appendLittleEndian(_ value: Float, to data: inout Data) {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }

    private func temporaryURL(for destination: URL) -> URL {
        destination
            .deletingLastPathComponent()
            .appending(path: ".\(destination.lastPathComponent).tmp-\(UUID().uuidString)")
    }

    private func removeIfPresent(_ url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        try FileManager.default.removeItem(at: url)
    }
}

private extension CaptureDisplayOrientation {
    var cgImagePropertyOrientation: CGImagePropertyOrientation {
        switch self {
        case .up: .up
        case .right: .right
        case .down: .down
        case .left: .left
        }
    }
}

private func normalizedMaskBytes(from buffer: CVPixelBuffer) throws -> [UInt8] {
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let format = CVPixelBufferGetPixelFormatType(buffer)
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else {
        throw CaptureFailure.invalidFrame("Vision mask has no readable storage")
    }

    var output = [UInt8](repeating: 0, count: width * height)
    switch format {
    case kCVPixelFormatType_OneComponent8:
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let values = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<height {
            for x in 0..<width {
                output[y * width + x] = values[y * rowBytes + x]
            }
        }
    case kCVPixelFormatType_OneComponent32Float:
        let rowValues = CVPixelBufferGetBytesPerRow(buffer) / MemoryLayout<Float>.stride
        let values = base.assumingMemoryBound(to: Float.self)
        for y in 0..<height {
            for x in 0..<width {
                let value = max(0, min(1, values[y * rowValues + x]))
                output[y * width + x] = UInt8((value * 255).rounded())
            }
        }
    default:
        throw CaptureFailure.invalidFrame("Vision returned an unsupported mask format")
    }
    return output
}

private struct MaskHit {
    let label: Int
    let normalizedPoint: CGPoint
}

private func maskHit(
    in buffer: CVPixelBuffer,
    normalizedImagePoint: CGPoint
) -> MaskHit? {
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    guard width > 0, height > 0 else { return nil }
    let centerX = min(width - 1, max(0, Int(normalizedImagePoint.x * CGFloat(width))))
    let centerY = min(height - 1, max(0, Int(normalizedImagePoint.y * CGFloat(height))))
    let format = CVPixelBufferGetPixelFormatType(buffer)

    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }

    func value(x: Int, y: Int) -> Int {
        switch format {
        case kCVPixelFormatType_OneComponent8:
            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            return Int(base.assumingMemoryBound(to: UInt8.self)[y * rowBytes + x])
        case kCVPixelFormatType_OneComponent32Float:
            let rowValues = CVPixelBufferGetBytesPerRow(buffer) / MemoryLayout<Float>.stride
            let raw = base.assumingMemoryBound(to: Float.self)[y * rowValues + x]
            return raw.isFinite ? Int(raw.rounded()) : 0
        default:
            return 0
        }
    }

    func hit(x: Int, y: Int, label: Int) -> MaskHit {
        MaskHit(
            label: label,
            normalizedPoint: CGPoint(
                x: (CGFloat(x) + 0.5) / CGFloat(width),
                y: (CGFloat(y) + 0.5) / CGFloat(height)
            )
        )
    }

    let direct = value(x: centerX, y: centerY)
    if direct > 0 { return hit(x: centerX, y: centerY, label: direct) }
    for radius in 1...8 {
        let minimumX = max(0, centerX - radius)
        let maximumX = min(width - 1, centerX + radius)
        let minimumY = max(0, centerY - radius)
        let maximumY = min(height - 1, centerY + radius)
        for y in minimumY...maximumY {
            for x in minimumX...maximumX {
                let nearby = value(x: x, y: y)
                if nearby > 0 { return hit(x: x, y: y, label: nearby) }
            }
        }
    }
    return nil
}

private func medianDepth(
    in depth: CVPixelBuffer,
    confidence: CVPixelBuffer?,
    instanceMask: CVPixelBuffer,
    selectedLabel: Int,
    normalizedImagePoint: CGPoint
) -> Float? {
    guard selectedLabel > 0,
          CVPixelBufferGetPixelFormatType(depth) == kCVPixelFormatType_DepthFloat32 else {
        return nil
    }
    let width = CVPixelBufferGetWidth(depth)
    let height = CVPixelBufferGetHeight(depth)
    let maskWidth = CVPixelBufferGetWidth(instanceMask)
    let maskHeight = CVPixelBufferGetHeight(instanceMask)
    let maskFormat = CVPixelBufferGetPixelFormatType(instanceMask)
    guard width > 0, height > 0, maskWidth > 0, maskHeight > 0,
          maskFormat == kCVPixelFormatType_OneComponent8 ||
            maskFormat == kCVPixelFormatType_OneComponent32Float else {
        return nil
    }
    if let confidence {
        guard CVPixelBufferGetWidth(confidence) == width,
              CVPixelBufferGetHeight(confidence) == height,
              CVPixelBufferGetPixelFormatType(confidence) ==
                kCVPixelFormatType_OneComponent8 else {
            return nil
        }
    }
    let centerX = min(width - 1, max(0, Int(normalizedImagePoint.x * CGFloat(width))))
    let centerY = min(height - 1, max(0, Int(normalizedImagePoint.y * CGFloat(height))))

    CVPixelBufferLockBaseAddress(depth, .readOnly)
    CVPixelBufferLockBaseAddress(instanceMask, .readOnly)
    if let confidence {
        CVPixelBufferLockBaseAddress(confidence, .readOnly)
    }
    defer {
        if let confidence {
            CVPixelBufferUnlockBaseAddress(confidence, .readOnly)
        }
        CVPixelBufferUnlockBaseAddress(instanceMask, .readOnly)
        CVPixelBufferUnlockBaseAddress(depth, .readOnly)
    }
    guard let depthBase = CVPixelBufferGetBaseAddress(depth),
          let maskBase = CVPixelBufferGetBaseAddress(instanceMask) else {
        return nil
    }
    let depthValues = depthBase.assumingMemoryBound(to: Float.self)
    let depthStride = CVPixelBufferGetBytesPerRow(depth) / MemoryLayout<Float>.stride
    let confidenceValues = confidence.flatMap(CVPixelBufferGetBaseAddress)?
        .assumingMemoryBound(to: UInt8.self)
    let confidenceStride = confidence.map(CVPixelBufferGetBytesPerRow) ?? 0
    var values: [Float] = []

    func instanceLabel(depthX: Int, depthY: Int) -> Int {
        let maskX = min(
            maskWidth - 1,
            Int((Float(depthX) + 0.5) * Float(maskWidth) / Float(width))
        )
        let maskY = min(
            maskHeight - 1,
            Int((Float(depthY) + 0.5) * Float(maskHeight) / Float(height))
        )
        switch maskFormat {
        case kCVPixelFormatType_OneComponent8:
            let rowBytes = CVPixelBufferGetBytesPerRow(instanceMask)
            return Int(
                maskBase.assumingMemoryBound(to: UInt8.self)[maskY * rowBytes + maskX]
            )
        case kCVPixelFormatType_OneComponent32Float:
            let rowValues = CVPixelBufferGetBytesPerRow(instanceMask) /
                MemoryLayout<Float>.stride
            let raw = maskBase.assumingMemoryBound(to: Float.self)[
                maskY * rowValues + maskX
            ]
            return raw.isFinite ? Int(raw.rounded()) : 0
        default:
            return 0
        }
    }

    for y in max(0, centerY - 2)...min(height - 1, centerY + 2) {
        for x in max(0, centerX - 2)...min(width - 1, centerX + 2) {
            guard instanceLabel(depthX: x, depthY: y) == selectedLabel else {
                continue
            }
            if let confidenceValues,
               confidenceValues[y * confidenceStride + x] < 1 {
                continue
            }
            let value = depthValues[y * depthStride + x]
            if value.isFinite, value > 0 { values.append(value) }
        }
    }
    guard !values.isEmpty else { return nil }
    values.sort()
    return values[values.count / 2]
}
