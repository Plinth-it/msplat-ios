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
    let url: URL
    private var scoped = false

    /// Filenames that mark a directory as a COLMAP model, in either encoding.
    private static let modelFiles = ["cameras.bin", "cameras.txt"]

    init?(picked url: URL) {
        self.url = url
        scoped = url.startAccessingSecurityScopedResource()
        guard Self.modelDirectory(under: url) != nil else {
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
    /// The maxima are combined conservatively for preflight planning when a
    /// dataset happens to contain more than one camera resolution.
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
