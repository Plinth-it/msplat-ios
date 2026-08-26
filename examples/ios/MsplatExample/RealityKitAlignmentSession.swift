import Foundation
import ImageIO
import os
@preconcurrency import RealityKit
import SwiftUI

enum RealityKitAlignmentOrdering: String, CaseIterable, Identifiable, Sendable {
    case sequential = "Sequential"
    case unordered = "Unordered"

    var id: Self { self }
}

struct RealityKitAlignmentOptions: Equatable, Sendable {
    let usesObjectMaskingForAlignment: Bool
    let exportsTrainingMasks: Bool
    let ordering: RealityKitAlignmentOrdering
}

enum RealityKitAlignmentStageDescription {
    static func message(
        for stage: PhotogrammetrySession.Output.ProcessingStage
    ) -> String? {
        switch stage {
        case .preProcessing:
            "Preparing images…"
        case .imageAlignment:
            "Aligning camera poses…"
        case .pointCloudGeneration:
            "Generating sparse point cloud…"
        case .optimization:
            "Optimizing camera alignment…"
        case .meshGeneration, .textureMapping:
            "Finalizing camera and point-cloud alignment…"
        @unknown default:
            nil
        }
    }
}

@MainActor
final class RealityKitAlignmentResult {
    let folder: DatasetFolder
    let export: RealityKitColmapExportResult
    let invalidSampleCount: Int
    let skippedSampleCount: Int
    let wasDownsampled: Bool
    let stitchingWasIncomplete: Bool

    init(
        folder: DatasetFolder,
        export: RealityKitColmapExportResult,
        invalidSampleCount: Int,
        skippedSampleCount: Int,
        wasDownsampled: Bool,
        stitchingWasIncomplete: Bool
    ) {
        self.folder = folder
        self.export = export
        self.invalidSampleCount = invalidSampleCount
        self.skippedSampleCount = skippedSampleCount
        self.wasDownsampled = wasDownsampled
        self.stitchingWasIncomplete = stitchingWasIncomplete
    }
}

enum RealityKitAlignmentFailure: LocalizedError, Sendable {
    case unavailable
    case requiresPhysicalDevice
    case rawImageIntrinsicsRequireNewerOS
    case tooFewImages(Int)
    case tooManyImages(count: Int, maximum: Int)
    case imageTooLarge(dimension: Int, maximum: Int)
    case insufficientMemory(availableMB: Int, recommendedMB: Int)
    case request(String)
    case missingPoses
    case missingPointCloud
    case invalidExport

    var errorDescription: String? {
        switch self {
        case .unavailable:
            "RealityKit photogrammetry is unavailable on this device."
        case .requiresPhysicalDevice:
            "RealityKit COLMAP alignment requires a physical device."
        case .rawImageIntrinsicsRequireNewerOS:
            "Imported image alignment requires iOS 26 or later so RealityKit can return the calibrated intrinsics needed by COLMAP. ARKit captures carry their own intrinsics and can be aligned on iOS 18 or later."
        case .tooFewImages(let count):
            "Choose at least 3 readable images; this folder contains \(count)."
        case .tooManyImages(let count, let maximum):
            "This device supports at most \(maximum) input images, but the folder contains \(count)."
        case .imageTooLarge(let dimension, let maximum):
            "An input image has a \(dimension)-pixel edge; this device supports at most \(maximum) pixels."
        case .insufficientMemory(let availableMB, let recommendedMB):
            "RealityKit alignment needs about \(recommendedMB) MB free for this folder, but iOS currently reports \(availableMB) MB. Close other apps or use fewer images."
        case .request(let message):
            "RealityKit alignment failed: \(message)"
        case .missingPoses:
            "RealityKit finished without returning aligned camera poses."
        case .missingPointCloud:
            "RealityKit finished without returning a sparse point cloud."
        case .invalidExport:
            "The exported folder did not pass the sample app's COLMAP preflight."
        }
    }
}

private enum RealityKitAlignmentPreflight {
    private static let bytesPerDecodedPixel: UInt64 = 4
    private static let minimumAvailableMemory: UInt64 = 1_250 * 1_024 * 1_024
    private static let fixedSessionOverhead: UInt64 = 512 * 1_024 * 1_024
    private static let perImageFeatureBudget: UInt64 = 12 * 1_024 * 1_024

    static func validate(_ input: RealityKitAlignmentInput) throws {
        let imageCount = input.imageURLs.count
        guard imageCount >= 3 else {
            throw RealityKitAlignmentFailure.tooFewImages(imageCount)
        }

        let limits = PhotogrammetrySession.limits
        guard imageCount <= limits.maximumNumberOfInputImages else {
            throw RealityKitAlignmentFailure.tooManyImages(
                count: imageCount,
                maximum: limits.maximumNumberOfInputImages
            )
        }

        let decodedSizes = input.imageURLs.compactMap(imageMetrics)
        if let largestDimension = decodedSizes.map(\.largestDimension).max(),
           largestDimension > limits.maximumInputImageDimension {
            throw RealityKitAlignmentFailure.imageTooLarge(
                dimension: largestDimension,
                maximum: limits.maximumInputImageDimension
            )
        }

        let decodedBytes = decodedSizes.map(\.decodedBytes)
        let totalDecodedBytes = decodedBytes.reduce(UInt64(0), +)
        let largestImageBytes = decodedBytes.max() ?? 0
        let featureBytes = UInt64(imageCount) * perImageFeatureBudget
        let estimated = max(
            largestImageBytes * 5 + featureBytes,
            totalDecodedBytes / 4
        ) + fixedSessionOverhead
        let recommended = max(minimumAvailableMemory, estimated)
        let available = UInt64(os_proc_available_memory())
        guard available == 0 || available >= recommended else {
            throw RealityKitAlignmentFailure.insufficientMemory(
                availableMB: megabytes(available),
                recommendedMB: megabytes(recommended)
            )
        }
    }

    private static func imageMetrics(
        at url: URL
    ) -> (largestDimension: Int, decodedBytes: UInt64)? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any],
              let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?
                .intValue,
              let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?
                .intValue,
              width > 0,
              height > 0 else {
            return nil
        }
        let pixels = UInt64(width).multipliedReportingOverflow(by: UInt64(height))
        guard !pixels.overflow else { return nil }
        let bytes = pixels.partialValue.multipliedReportingOverflow(
            by: bytesPerDecodedPixel
        )
        guard !bytes.overflow else { return nil }
        return (max(width, height), bytes.partialValue)
    }

    private static func megabytes(_ bytes: UInt64) -> Int {
        Int((bytes + 1_048_575) / 1_048_576)
    }
}

@MainActor
final class RealityKitAlignmentSession: ObservableObject {
    private static let alignmentProgressWeight = 0.9
    private static let exportProgressWeight = 0.09

    enum Phase {
        case idle
        case preparing
        case aligning
        case exporting
        case finished(RealityKitAlignmentResult)
        case cancelled
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var progress = 0.0
    @Published private(set) var statusMessage = "Ready to align images."

    private var workTask: Task<Void, Never>?
#if !targetEnvironment(simulator)
    private var photogrammetrySession: PhotogrammetrySession?
#endif

    var isBusy: Bool {
        switch phase {
        case .preparing, .aligning, .exporting: true
        default: false
        }
    }

    func start(
        input: RealityKitAlignmentInput,
        options: RealityKitAlignmentOptions
    ) {
        guard !isBusy else { return }
        workTask?.cancel()
        progress = 0
        statusMessage = "Checking device capacity…"
        phase = .preparing

        workTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
#if targetEnvironment(simulator)
                throw RealityKitAlignmentFailure.requiresPhysicalDevice
#else
                try await run(input: input, options: options)
#endif
            } catch is CancellationError {
                phase = .cancelled
                statusMessage = "Alignment cancelled."
            } catch {
                phase = .failed(error.localizedDescription)
                statusMessage = "Alignment failed."
            }
#if !targetEnvironment(simulator)
            photogrammetrySession = nil
#endif
            workTask = nil
        }
    }

    func cancel() {
        guard isBusy else { return }
        workTask?.cancel()
#if !targetEnvironment(simulator)
        photogrammetrySession?.cancel()
#endif
    }

#if !targetEnvironment(simulator)
    private struct ProcessedAlignment {
        let poses: PhotogrammetrySession.Poses
        let pointCloud: PhotogrammetrySession.PointCloud
        let invalidSampleCount: Int
        let skippedSampleCount: Int
        let wasDownsampled: Bool
        let stitchingWasIncomplete: Bool
    }

    private func run(
        input: RealityKitAlignmentInput,
        options: RealityKitAlignmentOptions
    ) async throws {
        guard PhotogrammetrySession.isSupported else {
            throw RealityKitAlignmentFailure.unavailable
        }
        if input.knownCamerasByFilename.isEmpty {
            guard #available(iOS 26.0, *) else {
                throw RealityKitAlignmentFailure.rawImageIntrinsicsRequireNewerOS
            }
        }
        try RealityKitAlignmentPreflight.validate(input)
        try Task.checkCancellation()
        let processed = try await processAlignment(
            input: input,
            options: options
        )
        statusMessage = "Preparing COLMAP export…"
        try await Task.sleep(for: .milliseconds(300))
        try Task.checkCancellation()

        phase = .exporting
        progress = Self.alignmentProgressWeight
        statusMessage = options.exportsTrainingMasks
            ? "Writing aligned images and Vision training masks…"
            : "Writing aligned COLMAP dataset…"
        let root = URL.documentsDirectory
            .appending(path: "RealityKitAlignments", directoryHint: .isDirectory)
            .appending(
                path: "Alignment-\(UUID().uuidString)",
                directoryHint: .isDirectory
            )
        let export = try await RealityKitColmapExporter.export(
            poses: processed.poses,
            pointCloud: processed.pointCloud,
            knownCamerasByFilename: input.knownCamerasByFilename,
            exportsTrainingMasks: options.exportsTrainingMasks,
            to: root,
            progressHandler: { [weak self] exportProgress, stage in
                guard let self else { return }
                self.progress = Self.alignmentProgressWeight +
                    exportProgress * Self.exportProgressWeight
                self.statusMessage = switch stage {
                case .images:
                    "Normalizing registered images…"
                case .masks:
                    "Generating Vision training masks…"
                case .sparseModel:
                    "Writing camera poses and sparse point cloud…"
                case .publishing:
                    "Publishing aligned COLMAP dataset…"
                }
            }
        )
        statusMessage = "Validating exported COLMAP dataset…"
        guard let folder = DatasetFolder(picked: export.datasetDirectory) else {
            throw RealityKitAlignmentFailure.invalidExport
        }

        phase = .finished(
            RealityKitAlignmentResult(
                folder: folder,
                export: export,
                invalidSampleCount: processed.invalidSampleCount,
                skippedSampleCount: processed.skippedSampleCount,
                wasDownsampled: processed.wasDownsampled,
                stitchingWasIncomplete: processed.stitchingWasIncomplete
            )
        )
        progress = 1
        statusMessage = "Aligned COLMAP dataset ready."
    }

    private func processAlignment(
        input: RealityKitAlignmentInput,
        options: RealityKitAlignmentOptions
    ) async throws -> ProcessedAlignment {
        var configuration = PhotogrammetrySession.Configuration()
        configuration.isObjectMaskingEnabled =
            options.usesObjectMaskingForAlignment
        configuration.featureSensitivity = .high
        configuration.sampleOrdering = options.ordering == .sequential
            ? .sequential : .unordered

        let session = try PhotogrammetrySession(
            input: input.imagesURL,
            configuration: configuration
        )
        photogrammetrySession = session
        defer {
            session.cancel()
            photogrammetrySession = nil
        }

        var poses: PhotogrammetrySession.Poses?
        var pointCloud: PhotogrammetrySession.PointCloud?
        var poseProgress = 0.0
        var pointProgress = 0.0
        var invalidSampleCount = 0
        var skippedSampleCount = 0
        var wasDownsampled = false
        var stitchingWasIncomplete = false

        try session.process(requests: [.poses, .pointCloud])
        phase = .aligning
        statusMessage = "Aligning cameras and sparse points…"

        outputLoop: for try await output in session.outputs {
            try Task.checkCancellation()
            switch output {
            case .inputComplete:
                statusMessage = "Input accepted. Aligning cameras…"
            case .requestProgress(let request, let fractionComplete):
                if case .poses = request {
                    poseProgress = fractionComplete
                } else if case .pointCloud = request {
                    pointProgress = fractionComplete
                }
                progress = min(max((poseProgress + pointProgress) / 2, 0), 1) *
                    Self.alignmentProgressWeight
            case .requestProgressInfo(let request, let info):
                switch request {
                case .poses, .pointCloud:
                    if let stage = info.processingStage,
                       let message = RealityKitAlignmentStageDescription.message(
                        for: stage
                       ) {
                        statusMessage = message
                    }
                default:
                    break
                }
            case .requestComplete(_, let result):
                switch result {
                case .poses(let value):
                    poses = value
                    poseProgress = 1
                case .pointCloud(let value):
                    pointCloud = value
                    pointProgress = 1
                default:
                    break
                }
                progress = ((poseProgress + pointProgress) / 2) *
                    Self.alignmentProgressWeight
            case .requestError(_, let error):
                throw RealityKitAlignmentFailure.request(
                    String(describing: error)
                )
            case .processingComplete:
                break outputLoop
            case .processingCancelled:
                throw CancellationError()
            case .invalidSample:
                invalidSampleCount += 1
            case .skippedSample:
                skippedSampleCount += 1
            case .automaticDownsampling:
                wasDownsampled = true
            case .stitchingIncomplete:
                stitchingWasIncomplete = true
            @unknown default:
                break
            }
        }

        try Task.checkCancellation()
        guard let poses else {
            throw RealityKitAlignmentFailure.missingPoses
        }
        guard let pointCloud else {
            throw RealityKitAlignmentFailure.missingPointCloud
        }
        return ProcessedAlignment(
            poses: poses,
            pointCloud: pointCloud,
            invalidSampleCount: invalidSampleCount,
            skippedSampleCount: skippedSampleCount,
            wasDownsampled: wasDownsampled,
            stitchingWasIncomplete: stitchingWasIncomplete
        )
    }
#endif
}
