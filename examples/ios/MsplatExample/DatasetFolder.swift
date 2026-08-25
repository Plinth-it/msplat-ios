import Foundation
import ImageIO
import Msplat

/// A COLMAP or Nerfstudio training dataset the user picked from Files.
///
/// The picked folder lives outside the app container, so it is only readable
/// while its security scope is held. Training reads images lazily from first
/// step to last — that is the whole point of the byte-budgeted image cache —
/// so the scope has to stay open for the entire session, not just the load.
final class DatasetFolder {
    enum Kind: String, Sendable {
        case colmap = "COLMAP"
        case nerfstudio = "Nerfstudio"
    }

    private struct NerfstudioManifest: Decodable, Sendable {
        struct Frame: Decodable, Sendable {
            let filePath: String
            let cameraModel: String?

            private enum CodingKeys: String, CodingKey {
                case filePath = "file_path"
                case cameraModel = "camera_model"
            }
        }

        let cameraModel: String?
        let plyFilePath: String?
        let frames: [Frame]

        private enum CodingKeys: String, CodingKey {
            case cameraModel = "camera_model"
            case plyFilePath = "ply_file_path"
            case frames
        }
    }

    private static let pointReadChunkSize = 64 * 1_024
    private static let maximumPointTextLineBytes = 16 * 1_024 * 1_024
    private static let minimumBinaryPointRecordBytes = 51
    private static let nerfstudioImageExtensions = [
        ".png", ".jpg", ".jpeg", ".JPG",
    ]

    let id = UUID()
    let url: URL
    let kind: Kind
    private var scoped = false

    /// Filenames that mark a directory as a COLMAP model, in either encoding.
    private static let modelFiles = ["cameras.bin", "cameras.txt"]

    init?(picked url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        guard let kind = Self.kind(at: url) else {
            if scoped {
                url.stopAccessingSecurityScopedResource()
            }
            return nil
        }

        self.url = url
        self.kind = kind
        self.scoped = scoped
    }

    deinit { release() }

    private func release() {
        if scoped {
            url.stopAccessingSecurityScopedResource()
            scoped = false
        }
    }

    var name: String { url.lastPathComponent }

    var supportsAutomaticTrainingMaskDiscovery: Bool {
        kind == .colmap
    }

    /// COLMAP puts the model either at the root or under sparse/0.
    static func modelDirectory(under root: URL) -> URL? {
        let fm = FileManager.default
        for candidate in [root, root.appending(path: "sparse/0")] {
            for name in modelFiles where fm.fileExists(atPath: candidate.appending(path: name).path) {
                return candidate
            }
        }
        return nil
    }

    private static func kind(at root: URL) -> Kind? {
        let fileManager = FileManager.default
        if fileManager.fileExists(
            atPath: root.appending(path: "transforms.json").path
        ) {
            return .nerfstudio
        }
        return modelDirectory(under: root) == nil ? nil : .colmap
    }

    private static func requiredKind(at root: URL) throws -> Kind {
        guard let kind = kind(at: root) else {
            throw MsplatError.invalidDataset(
                "No root transforms.json or COLMAP camera model was found."
            )
        }
        return kind
    }

    /// Reads only the sparse-model metadata needed to choose a safe Gaussian
    /// ceiling. Full point and track validation remains in the native loader.
    static func initialSparsePointCount(at root: URL) throws -> Int {
        try Task.checkCancellation()
        if try requiredKind(at: root) == .nerfstudio {
            let manifest = try nerfstudioManifest(at: root)
            let pointCloud = try nerfstudioPointCloudURL(
                at: root,
                manifest: manifest
            )
            return try plySparsePointCount(at: pointCloud)
        }

        guard let model = modelDirectory(under: root) else {
            throw MsplatError.invalidDataset(
                "No COLMAP camera model was found in the selected folder."
            )
        }

        let fileManager = FileManager.default
        let binary = model.appending(path: "points3D.bin")
        if fileManager.fileExists(atPath: binary.path) {
            return try binarySparsePointCount(at: binary)
        }

        let text = model.appending(path: "points3D.txt")
        if fileManager.fileExists(atPath: text.path) {
            return try textSparsePointCount(at: text)
        }

        let ply = model.appending(path: "points3D.ply")
        if fileManager.fileExists(atPath: ply.path) {
            return try plySparsePointCount(at: ply)
        }

        throw MsplatError.invalidDataset(
            "No points3D.bin, points3D.txt, or points3D.ply was found in the COLMAP model."
        )
    }

    private static func binarySparsePointCount(at url: URL) throws -> Int {
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values.isRegularFile == true,
              let fileSize = values.fileSize,
              fileSize >= MemoryLayout<UInt64>.size else {
            throw MsplatError.invalidDataset(
                "The COLMAP points3D.bin header is incomplete."
            )
        }

        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let header = try handle.read(upToCount: MemoryLayout<UInt64>.size) ?? Data()
        guard header.count == MemoryLayout<UInt64>.size else {
            throw MsplatError.invalidDataset(
                "The COLMAP points3D.bin header is incomplete."
            )
        }

        let count = header.enumerated().reduce(UInt64(0)) { value, byte in
            value | (UInt64(byte.element) << UInt64(byte.offset * 8))
        }
        let availableRecordBytes = fileSize - MemoryLayout<UInt64>.size
        guard count <= UInt64(availableRecordBytes / minimumBinaryPointRecordBytes) else {
            throw MsplatError.invalidDataset(
                "The COLMAP points3D.bin point count exceeds its file size."
            )
        }
        return try validatedSparsePointCount(count)
    }

    private static func textSparsePointCount(at url: URL) throws -> Int {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var count = UInt64(0)
        var lineLength = 0
        var lineIsData = false
        var lineWasClassified = false

        func finishLine() throws {
            if lineIsData {
                count += 1
                guard count <= UInt64(Int32.max) else {
                    throw MsplatError.invalidDataset(
                        "The COLMAP sparse-point count exceeds the supported range."
                    )
                }
            }
            lineLength = 0
            lineIsData = false
            lineWasClassified = false
        }

        while true {
            try Task.checkCancellation()
            let chunk = try handle.read(upToCount: pointReadChunkSize) ?? Data()
            if chunk.isEmpty { break }

            for byte in chunk {
                if byte == 0x0A {
                    try finishLine()
                    continue
                }

                lineLength += 1
                guard lineLength <= maximumPointTextLineBytes else {
                    throw MsplatError.invalidDataset(
                        "A COLMAP points3D.txt line exceeds the supported length."
                    )
                }
                if !lineWasClassified,
                   byte != 0x20,
                   byte != 0x09,
                   byte != 0x0D {
                    lineWasClassified = true
                    lineIsData = byte != 0x23
                }
            }
        }

        if lineLength > 0 {
            try finishLine()
        }
        return try validatedSparsePointCount(count)
    }

    private static func plySparsePointCount(at url: URL) throws -> Int {
        let fileName = url.lastPathComponent
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var line = Data()
        var isFirstLine = true
        var foundHeaderEnd = false
        var pointCount: UInt64?

        func processLine() throws -> Bool {
            let value = String(decoding: line, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if isFirstLine {
                isFirstLine = false
                guard value == "ply" else {
                    throw MsplatError.invalidDataset(
                        "\(fileName) is not a PLY file."
                    )
                }
            }

            let fields = value.split(whereSeparator: { $0.isWhitespace })
            if fields.count == 3,
               fields[0] == "element",
               fields[1] == "vertex" {
                guard let parsed = UInt64(fields[2]) else {
                    throw MsplatError.invalidDataset(
                        "\(fileName) has an invalid vertex count."
                    )
                }
                pointCount = parsed
            }
            if value == "end_header" {
                foundHeaderEnd = true
                return true
            }
            return false
        }

        var finished = false
        while !finished {
            try Task.checkCancellation()
            let chunk = try handle.read(upToCount: pointReadChunkSize) ?? Data()
            if chunk.isEmpty { break }

            for byte in chunk {
                if byte == 0x0A {
                    finished = try processLine()
                    line.removeAll(keepingCapacity: true)
                    if finished { break }
                } else {
                    line.append(byte)
                    guard line.count <= maximumPointTextLineBytes else {
                        throw MsplatError.invalidDataset(
                            "A \(fileName) header line exceeds the supported length."
                        )
                    }
                }
            }
        }

        if !finished, !line.isEmpty {
            _ = try processLine()
        }
        guard foundHeaderEnd, let pointCount else {
            throw MsplatError.invalidDataset(
                "\(fileName) has an incomplete header or no vertex count."
            )
        }
        return try validatedSparsePointCount(pointCount)
    }

    private static func validatedSparsePointCount(_ count: UInt64) throws -> Int {
        guard count > 0 else {
            throw MsplatError.invalidDataset(
                "The dataset point cloud contains no points."
            )
        }
        guard count <= UInt64(Int32.max) else {
            throw MsplatError.invalidDataset(
                "The dataset point count exceeds the supported range."
            )
        }
        return Int(count)
    }

    private static func nerfstudioManifest(
        at root: URL
    ) throws -> NerfstudioManifest {
        let manifestURL = root.appending(path: "transforms.json")
        let manifest: NerfstudioManifest
        do {
            let data = try Data(contentsOf: manifestURL)
            manifest = try JSONDecoder().decode(NerfstudioManifest.self, from: data)
        } catch {
            throw MsplatError.invalidDataset(
                "transforms.json could not be decoded: \(error.localizedDescription)"
            )
        }

        guard !manifest.frames.isEmpty else {
            throw MsplatError.invalidDataset(
                "transforms.json contains no frames."
            )
        }
        try validateNerfstudioCameraModel(
            manifest.cameraModel,
            context: "transforms.json"
        )
        for frame in manifest.frames {
            guard !frame.filePath.isEmpty else {
                throw MsplatError.invalidDataset(
                    "transforms.json contains a frame with an empty file_path."
                )
            }
            try validateNerfstudioCameraModel(
                frame.cameraModel ?? manifest.cameraModel,
                context: "frame '\(frame.filePath)'"
            )
        }
        try Task.checkCancellation()
        return manifest
    }

    private static func validateNerfstudioCameraModel(
        _ cameraModel: String?,
        context: String
    ) throws {
        guard let cameraModel else { return }
        switch cameraModel.uppercased() {
        case "PINHOLE", "PERSPECTIVE", "OPENCV":
            return
        case "OPENCV_FISHEYE":
            throw MsplatError.invalidDataset(
                "\(context) uses OPENCV_FISHEYE, which this sample does not support."
            )
        default:
            throw MsplatError.invalidDataset(
                "\(context) uses unsupported camera_model '\(cameraModel)'."
            )
        }
    }

    private static func nerfstudioPointCloudURL(
        at root: URL,
        manifest: NerfstudioManifest
    ) throws -> URL {
        var candidates: [URL] = []
        if let path = manifest.plyFilePath, !path.isEmpty {
            let pointCloud = try datasetURL(
                for: path,
                under: root,
                purpose: "point cloud"
            )
            if FileManager.default.fileExists(atPath: pointCloud.path) {
                guard isRegularFile(pointCloud) else {
                    throw MsplatError.invalidDataset(
                        "Nerfstudio ply_file_path must reference a regular file."
                    )
                }
                return pointCloud
            }
        }
        candidates.append(root.appending(path: "sparse/0/points3D.ply"))
        candidates.append(root.appending(path: "points3D.ply"))

        for candidate in candidates where isRegularFile(candidate) {
            return candidate
        }
        throw MsplatError.invalidDataset(
            "The Nerfstudio dataset needs a non-empty PLY referenced by " +
            "ply_file_path, sparse/0/points3D.ply, or points3D.ply."
        )
    }

    private static func datasetURL(
        for path: String,
        under root: URL,
        purpose: String
    ) throws -> URL {
        let standardizedRoot = root.standardizedFileURL
        let candidate = URL(fileURLWithPath: path, relativeTo: standardizedRoot)
            .standardizedFileURL
        let rootPath = standardizedRoot.path
        guard candidate.path == rootPath || candidate.path.hasPrefix(rootPath + "/") else {
            throw MsplatError.invalidDataset(
                "The Nerfstudio \(purpose) path must stay inside the selected folder."
            )
        }
        return candidate
    }

    private static func isRegularFile(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isRegularFileKey]))?
            .isRegularFile == true
    }

    /// Counts regular files below any case-insensitive `masks/` path component
    /// without claiming that every file matches a COLMAP frame. The native
    /// loader owns exact mask matching.
    static func countTrainingMaskCandidates(at root: URL) -> Int {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey]
        ) else {
            return 0
        }

        var count = 0
        let rootComponentCount = root.standardizedFileURL.pathComponents.count
        for case let candidate as URL in enumerator {
            if Task.isCancelled { return count }
            let relativeComponents = candidate.standardizedFileURL.pathComponents
                .dropFirst(rootComponentCount)
                .dropLast()
            guard relativeComponents.contains(where: {
                $0.compare(
                    "masks",
                    options: [.caseInsensitive, .literal]
                ) == .orderedSame
            }) else {
                continue
            }

            let values = try? candidate.resourceValues(
                forKeys: [.isRegularFileKey]
            )
            if values?.isRegularFile == true {
                count += 1
            }
        }
        return count
    }

    /// A short description of what was found, for the picker row.
    var summary: String {
        switch kind {
        case .nerfstudio:
            let imageCount = (try? Self.nerfstudioManifest(at: url))?.frames.count
            return imageCount.map { "Nerfstudio, \($0) images" } ?? "Nerfstudio manifest"
        case .colmap:
            guard let model = Self.modelDirectory(under: url) else {
                return "COLMAP model unavailable"
            }
            let encoding = FileManager.default.fileExists(
                atPath: model.appending(path: "cameras.bin").path
            ) ? "binary" : "text"
            let images = (try? FileManager.default.contentsOfDirectory(
                at: url.appending(path: "images"),
                includingPropertiesForKeys: nil
            ).count) ?? 0
            return images > 0
                ? "COLMAP \(encoding), \(images) images"
                : "COLMAP \(encoding)"
        }
    }

    /// Reads image headers only; ImageIO does not decode full pixel buffers.
    /// COLMAP calibrates against encoded pixel coordinates, so EXIF orientation
    /// is validated but intentionally not applied to these dimensions.
    static func maximumSourceDimensions(at datasetURL: URL) throws
        -> TrainingImageDimensions {
        switch try requiredKind(at: datasetURL) {
        case .colmap:
            return try maximumColmapSourceDimensions(at: datasetURL)
        case .nerfstudio:
            return try maximumNerfstudioSourceDimensions(at: datasetURL)
        }
    }

    private static func maximumColmapSourceDimensions(
        at datasetURL: URL
    ) throws -> TrainingImageDimensions {
        let imageDirectory = datasetURL.appending(path: "images")
        var enumerationFailed = false
        guard let enumerator = FileManager.default.enumerator(
            at: imageDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants],
            errorHandler: { _, _ in
                enumerationFailed = true
                return false
            }
        ) else {
            throw MsplatError.invalidDataset(
                "The dataset's images folder could not be enumerated."
            )
        }

        var maximumWidth = 0
        var maximumHeight = 0
        for case let imageURL as URL in enumerator {
            try Task.checkCancellation()
            let values = try? imageURL.resourceValues(forKeys: [.isRegularFileKey])
            guard values?.isRegularFile == true,
                  let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
                  let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                    as NSDictionary?,
                  let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
                  let height = properties[kCGImagePropertyPixelHeight] as? NSNumber else {
                continue
            }
            let orientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?
                .intValue ?? 1
            guard (1...8).contains(orientation) else {
                throw MsplatError.invalidDataset(
                    "An image has an invalid EXIF orientation value."
                )
            }
            maximumWidth = max(maximumWidth, width.intValue)
            maximumHeight = max(maximumHeight, height.intValue)
        }

        guard !enumerationFailed else {
            throw MsplatError.ioFailure(
                "Not every image in the dataset could be inspected safely."
            )
        }
        guard maximumWidth > 0, maximumHeight > 0 else {
            throw MsplatError.invalidDataset(
                "No readable image dimensions were found in the dataset's images folder."
            )
        }
        return try TrainingImageDimensions(width: maximumWidth, height: maximumHeight)
    }

    private static func maximumNerfstudioSourceDimensions(
        at datasetURL: URL
    ) throws -> TrainingImageDimensions {
        let manifest = try nerfstudioManifest(at: datasetURL)
        var maximumWidth = 0
        var maximumHeight = 0

        for frame in manifest.frames {
            try Task.checkCancellation()
            let imageURL = try nerfstudioImageURL(
                for: frame.filePath,
                under: datasetURL
            )
            guard let dimensions = try sourceDimensions(at: imageURL) else {
                throw MsplatError.invalidDataset(
                    "Nerfstudio image '\(frame.filePath)' could not be inspected."
                )
            }
            maximumWidth = max(maximumWidth, dimensions.width)
            maximumHeight = max(maximumHeight, dimensions.height)
        }

        return try TrainingImageDimensions(
            width: maximumWidth,
            height: maximumHeight
        )
    }

    private static func nerfstudioImageURL(
        for path: String,
        under root: URL
    ) throws -> URL {
        let imageURL = try datasetURL(for: path, under: root, purpose: "image")
        if isRegularFile(imageURL) {
            return imageURL
        }
        for suffix in nerfstudioImageExtensions {
            let candidate = URL(fileURLWithPath: imageURL.path + suffix)
            if isRegularFile(candidate) {
                return candidate
            }
        }
        throw MsplatError.invalidDataset(
            "Nerfstudio image '\(path)' was not found."
        )
    }

    private static func sourceDimensions(
        at imageURL: URL
    ) throws -> (width: Int, height: Int)? {
        guard isRegularFile(imageURL),
              let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as NSDictionary?,
              let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let height = properties[kCGImagePropertyPixelHeight] as? NSNumber else {
            return nil
        }
        let orientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?
            .intValue ?? 1
        guard (1...8).contains(orientation) else {
            throw MsplatError.invalidDataset(
                "An image has an invalid EXIF orientation value."
            )
        }
        return (width.intValue, height.intValue)
    }
}
