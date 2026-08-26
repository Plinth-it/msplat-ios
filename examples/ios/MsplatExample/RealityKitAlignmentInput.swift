import Foundation
import ImageIO

/// A folder of source images whose security scope stays open for the full
/// RealityKit alignment run.
final class RealityKitAlignmentInput: Identifiable {
    enum Origin: Sendable {
        case imported
        case capture(CaptureMode)
    }

    private static let supportedImageExtensions: Set<String> = [
        "heic", "heif", "jpeg", "jpg", "png",
    ]

    let id = UUID()
    let selectedURL: URL
    let imagesURL: URL
    let imageURLs: [URL]
    let origin: Origin
    let knownCamerasByFilename: [String: RealityKitColmapCamera]
    private var scoped: Bool

    init?(picked url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        guard let resolved = Self.resolveImages(in: url) else {
            if scoped {
                url.stopAccessingSecurityScopedResource()
            }
            return nil
        }

        selectedURL = url
        imagesURL = resolved.directory
        imageURLs = resolved.images
        origin = .imported
        knownCamerasByFilename = [:]
        self.scoped = scoped
    }

    init?(capture: CapturedDataset) {
        let imagesURL = capture.rootURL.appending(
            path: "images",
            directoryHint: .isDirectory
        )
        guard let images = Self.imageFiles(in: imagesURL), !images.isEmpty else {
            return nil
        }

        selectedURL = capture.rootURL
        self.imagesURL = imagesURL
        imageURLs = images
        origin = .capture(capture.manifest.mode)
        knownCamerasByFilename = Dictionary(
            capture.manifest.frames.map { frame in
                (
                    frame.imageURL(under: capture.rootURL).lastPathComponent,
                    RealityKitColmapCamera(
                        width: frame.calibration.width,
                        height: frame.calibration.height,
                        fx: Double(frame.calibration.fx),
                        fy: Double(frame.calibration.fy),
                        cx: Double(frame.calibration.cx),
                        cy: Double(frame.calibration.cy)
                    )
                )
            },
            uniquingKeysWith: { _, newer in newer }
        )
        scoped = false
    }

    deinit {
        if scoped {
            selectedURL.stopAccessingSecurityScopedResource()
            scoped = false
        }
    }

    var name: String { selectedURL.lastPathComponent }

    var suggestedObjectMasking: Bool {
        switch origin {
        case .imported, .capture(.object): true
        case .capture(.scene): false
        }
    }

    var suggestedTrainingMaskExport: Bool {
        switch origin {
        case .imported, .capture(.object): true
        case .capture(.scene): false
        }
    }

    var originDescription: String {
        switch origin {
        case .imported: "Imported image folder"
        case .capture(let mode): "ARKit \(mode.title) capture"
        }
    }

    private static func resolveImages(
        in selectedURL: URL
    ) -> (directory: URL, images: [URL])? {
        let candidates = [
            selectedURL,
            selectedURL.appending(path: "images", directoryHint: .isDirectory),
        ]
        for candidate in candidates {
            if let images = imageFiles(in: candidate), !images.isEmpty {
                return (candidate, images)
            }
        }
        return nil
    }

    private static func imageFiles(in directory: URL) -> [URL]? {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        return entries.filter { url in
            guard supportedImageExtensions.contains(url.pathExtension.lowercased()),
                  (try? url.resourceValues(forKeys: [.isRegularFileKey]))?
                    .isRegularFile == true,
                  let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
                return false
            }
            return CGImageSourceGetCount(source) > 0
        }
        .sorted {
            $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent)
                == .orderedAscending
        }
    }
}
