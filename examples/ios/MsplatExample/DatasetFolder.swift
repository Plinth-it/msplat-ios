import Foundation
import ImageIO
import Msplat

/// A COLMAP reconstruction the user picked from Files.
///
/// The picked folder lives outside the app container, so it is only readable
/// while its security scope is held. Training reads images lazily from first
/// step to last — that is the whole point of the byte-budgeted image cache —
/// so the scope has to stay open for the entire session, not just the load.
final class DatasetFolder {
    private static let pointReadChunkSize = 64 * 1_024
    private static let maximumPointTextLineBytes = 16 * 1_024 * 1_024
    private static let minimumBinaryPointRecordBytes = 51

    let id = UUID()
    let url: URL
    private var scoped = false

    /// Filenames that mark a directory as a COLMAP model, in either encoding.
    private static let modelFiles = ["cameras.bin", "cameras.txt"]

    init?(picked url: URL) {
        self.url = url
        scoped = url.startAccessingSecurityScopedResource()
        guard !Self.containsHigherPriorityDataset(at: url),
              Self.modelDirectory(under: url) != nil else {
            release()
            return nil
        }
    }

    deinit { release() }

    private func release() {
        if scoped {
            url.stopAccessingSecurityScopedResource()
            scoped = false
        }
    }

    var name: String { url.lastPathComponent }

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

    private static func containsHigherPriorityDataset(at root: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: root.appending(path: "transforms.json").path
        )
    }

    /// Reads only the sparse-model metadata needed to choose a safe Gaussian
    /// ceiling. Full point and track validation remains in the native loader.
    static func initialSparsePointCount(at root: URL) throws -> Int {
        try Task.checkCancellation()
        guard !containsHigherPriorityDataset(at: root) else {
            throw MsplatError.invalidDataset(
                "This sample expects a COLMAP-only folder without a root transforms.json."
            )
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
                    throw MsplatError.invalidDataset("points3D.ply is not a PLY file.")
                }
            }

            let fields = value.split(whereSeparator: { $0.isWhitespace })
            if fields.count == 3,
               fields[0] == "element",
               fields[1] == "vertex" {
                guard let parsed = UInt64(fields[2]) else {
                    throw MsplatError.invalidDataset(
                        "points3D.ply has an invalid vertex count."
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
                            "A points3D.ply header line exceeds the supported length."
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
                "points3D.ply has an incomplete header or no vertex count."
            )
        }
        return try validatedSparsePointCount(pointCount)
    }

    private static func validatedSparsePointCount(_ count: UInt64) throws -> Int {
        guard count > 0 else {
            throw MsplatError.invalidDataset(
                "The COLMAP model contains no sparse points."
            )
        }
        guard count <= UInt64(Int32.max) else {
            throw MsplatError.invalidDataset(
                "The COLMAP sparse-point count exceeds the supported range."
            )
        }
        return Int(count)
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
        guard let model = Self.modelDirectory(under: url) else { return "not a COLMAP folder" }
        let encoding = FileManager.default.fileExists(atPath: model.appending(path: "cameras.bin").path)
            ? "binary" : "text"
        let images = (try? FileManager.default.contentsOfDirectory(
            at: url.appending(path: "images"), includingPropertiesForKeys: nil).count) ?? 0
        return images > 0 ? "\(encoding) model, \(images) images" : "\(encoding) model"
    }

    /// Reads image headers only; ImageIO does not decode full pixel buffers.
    /// COLMAP calibrates against encoded pixel coordinates, so EXIF orientation
    /// is validated but intentionally not applied to these dimensions.
    static func maximumSourceDimensions(at datasetURL: URL) throws
        -> TrainingImageDimensions {
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
}
