import Foundation
import Msplat
import SwiftUI
import os

@MainActor
final class TrainingSession: ObservableObject {
    enum QualityProfile: String, CaseIterable, Identifiable {
        case preview = "Preview"
        case balanced = "Balanced"

        var id: Self { self }

        var longestEdge: Int {
            switch self {
            case .preview: 1_600
            case .balanced: 1_920
            }
        }

        var shDegree: Int32 {
            switch self {
            case .preview: 1
            case .balanced: 2
            }
        }

        var maximumGaussianCount: Int {
            switch self {
            case .preview: 250_000
            case .balanced: 400_000
            }
        }
    }

    enum Phase: Equatable {
        case idle, planning, loading, training, cancelled, finished, failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var submittedIteration = 0
    @Published private(set) var completedIteration = 0
    @Published private(set) var splatCount = 0
    @Published private(set) var modelCapacity = 0
    @Published private(set) var cpuSubmitMs: Float = 0
    @Published private(set) var imagePrepareMs: Float = 0
    @Published private(set) var gpuExecutionMs: Float?
    @Published private(set) var endToEndMs: Float?
    @Published private(set) var loss: Float?
    @Published private(set) var effectiveWidth = 0
    @Published private(set) var effectiveHeight = 0
    @Published private(set) var activeSHDegree = 0
    @Published private(set) var retainedPackedIntersections: UInt64?
    @Published private(set) var packedIntersectionCapacity: UInt64?
    @Published private(set) var overflowKinds: RasterizerOverflowKinds = []
    @Published private(set) var overflowedCompletedSteps: UInt64 = 0
    @Published private(set) var memorySnapshot: TrainingMemorySnapshot?
    @Published private(set) var thermalState = "Nominal"
    @Published private(set) var preview: UIImage?
    @Published private(set) var trainingCameras = 0
    @Published private(set) var footprintMB = 0
    @Published private(set) var availableMB = 0
    @Published private(set) var exportedPly: URL?
    @Published private(set) var plannedStages: [ResolvedTrainingResolutionStage] = []
    @Published private(set) var estimatedPeakMB = 0
    @Published private(set) var plannedSHDegree = 0
    @Published private(set) var plannedInitialGaussians = 0
    @Published private(set) var plannedMaximumGaussians = 0

    /// Total steps. Kept modest by default: on a phone this is a battery and
    /// thermal budget as much as a quality one.
    @Published var iterations = 2_000
    @Published var qualityProfile: QualityProfile = .preview
    @Published var trainingMasksEnabled = false
    @Published var trainingMaskMode: TrainingMaskMode = .transparent

    private var worker: Task<Void, Never>?

    func start(folder: DatasetFolder) {
        guard phase == .idle || phase == .cancelled || phase == .finished || isFailed else { return }
        phase = .planning
        submittedIteration = 0
        completedIteration = 0
        splatCount = 0
        modelCapacity = 0
        cpuSubmitMs = 0
        imagePrepareMs = 0
        gpuExecutionMs = nil
        endToEndMs = nil
        loss = nil
        effectiveWidth = 0
        effectiveHeight = 0
        activeSHDegree = 0
        retainedPackedIntersections = nil
        packedIntersectionCapacity = nil
        overflowKinds = []
        overflowedCompletedSteps = 0
        memorySnapshot = nil
        thermalState = "Nominal"
        preview = nil
        trainingCameras = 0
        footprintMB = 0
        availableMB = 0
        exportedPly = nil
        plannedStages = []
        estimatedPeakMB = 0
        plannedSHDegree = 0
        plannedInitialGaussians = 0
        plannedMaximumGaussians = 0

        let steps = iterations
        let profile = qualityProfile
        let useTrainingMasks = trainingMasksEnabled
        let selectedTrainingMaskMode = trainingMaskMode

        worker = Task { [weak self] in
            guard let self else { return }
            await self.run(
                folder: folder,
                steps: steps,
                profile: profile,
                useTrainingMasks: useTrainingMasks,
                trainingMaskMode: selectedTrainingMaskMode
            )
        }
    }

    func cancel() { worker?.cancel() }

    private var isFailed: Bool { if case .failed = phase { return true }; return false }

    private func run(
        folder: DatasetFolder,
        steps: Int,
        profile: QualityProfile,
        useTrainingMasks: Bool,
        trainingMaskMode: TrainingMaskMode
    ) async {
        var session: MsplatSession?
        var resultURL: URL?
        var finalSplatCount = 0
        var failureMessage: String?
        var wasCancelled = false

        do {
            let datasetURL = folder.url
            let datasetScan = Task.detached(priority: .userInitiated) {
                let dimensions = try DatasetFolder.maximumSourceDimensions(
                    at: datasetURL
                )
                let initialGaussianCount = try DatasetFolder.initialSparsePointCount(
                    at: datasetURL
                )
                return (dimensions, initialGaussianCount)
            }
            let (sourceDimensions, initialGaussianCount) =
                try await withTaskCancellationHandler {
                    try await datasetScan.value
                } onCancel: {
                    datasetScan.cancel()
                }
            let plan = try Self.makePlan(
                sourceDimensions: sourceDimensions,
                initialGaussianCount: initialGaussianCount,
                steps: steps,
                profile: profile,
                includesTrainingMasks: useTrainingMasks
            )
            plannedStages = plan.resolvedStages
            estimatedPeakMB = Self.megabytes(plan.estimatedPeakMemory)
            plannedSHDegree = Int(plan.targetSHDegree)
            plannedInitialGaussians = initialGaussianCount
            plannedMaximumGaussians = plan.maximumGaussianCount
            availableMB = Self.availableMB()

            let availableBytes = Int64(availableMB) * 1_024 * 1_024
            guard availableBytes == 0 || plan.estimatedPeakMemory <= availableBytes else {
                throw MsplatError.outOfMemory(
                    "The \(profile.rawValue) plan estimates \(estimatedPeakMB) MB, " +
                    "but iOS currently reports \(availableMB) MB available. " +
                    (profile == .balanced
                        ? "Choose Preview, turn off masks, or close other apps."
                        : "Turn off masks, use a sparser COLMAP model, or close other apps.")
                )
            }

            phase = .loading
            try Task.checkCancellation()

            var baseConfig = TrainingConfig()
            baseConfig.trainingMaskMode = trainingMaskMode
            let activeSession = try await MsplatSession(
                datasetURL: folder.url,
                trainingPlan: plan,
                baseConfig: baseConfig
            )
            session = activeSession
            try Task.checkCancellation()

            let cameras = try await activeSession.numTrain
            guard cameras > 0 else {
                throw MsplatError.invalidDataset("No training cameras in that folder.")
            }
            trainingCameras = cameras
            phase = .training
            let previewPose = try await activeSession.cameraPose(at: 0)

            // Rendering a preview costs a full forward pass, so it is sampled
            // rather than done every step.
            let previewEvery = max(steps / 40, 10)

            for i in 0..<steps {
                try Task.checkCancellation()
                let stats = try await activeSession.step()

                if i % previewEvery == 0 || i == steps - 1 {
                    let frame = try await activeSession.renderRGBA(
                        pose: previewPose,
                        referenceCamera: 0
                    )
                    // Rendering synchronizes all previously submitted training
                    // work, so this poll can truthfully advance completed
                    // progress and attach timings to the matching iteration.
                    let telemetry = try await activeSession.trainingMetrics()
                    let memory = try await activeSession.memoryMetrics()
                    submittedIteration = telemetry.submitted?.iteration ?? stats.iteration
                    cpuSubmitMs = telemetry.submitted?.cpuSubmitMs ?? stats.cpuSubmitMs
                    if let completed = telemetry.completed {
                        completedIteration = completed.iteration
                        splatCount = completed.splatCount
                        modelCapacity = completed.modelCapacity
                        imagePrepareMs = completed.imagePrepareMs
                        gpuExecutionMs = completed.gpuExecutionMs
                        endToEndMs = completed.endToEndMs
                        loss = completed.loss
                        effectiveWidth = completed.effectiveWidth
                        effectiveHeight = completed.effectiveHeight
                        activeSHDegree = completed.activeSHDegree
                        retainedPackedIntersections =
                            completed.retainedPackedIntersectionCount
                        packedIntersectionCapacity =
                            completed.packedIntersectionCapacity
                        overflowKinds = completed.overflowKinds
                    }
                    overflowedCompletedSteps = telemetry.overflowedCompletedSteps
                    memorySnapshot = memory
                    preview = Self.image(from: frame)
                    footprintMB = memory.processPhysicalFootprintBytes.map(Self.megabytes) ?? 0
                    availableMB = memory.processAvailableBytes.map(Self.megabytes) ?? 0
                    thermalState = Self.thermalStateDescription()
                }
            }

            try Task.checkCancellation()
            let url = URL.documentsDirectory.appending(path: "msplat-scene.ply")
            try await activeSession.exportPLY(to: url)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MsplatError.ioFailure("Training finished, but the PLY file was not created.")
            }
            finalSplatCount = try await activeSession.splatCount
            resultURL = url
        } catch is CancellationError {
            // A cancelled run deliberately does not export a partial scene.
            wasCancelled = true
        } catch {
            failureMessage = error.localizedDescription
        }

        if let session {
            do {
                try await session.close()
            } catch {
                let closeMessage = "Could not release the training session: \(error.localizedDescription)"
                failureMessage = failureMessage.map { "\($0) \(closeMessage)" } ?? closeMessage
            }
        }

        if let failureMessage {
            phase = .failed(failureMessage)
        } else if wasCancelled {
            phase = .cancelled
        } else {
            splatCount = finalSplatCount
            exportedPly = resultURL
            phase = .finished
        }
        worker = nil
    }

    // MARK: - Helpers

    nonisolated static func makePlan(
        sourceDimensions: TrainingImageDimensions,
        initialGaussianCount: Int,
        steps: Int,
        profile: QualityProfile,
        includesTrainingMasks: Bool = false
    ) throws -> TrainingPlan {
        guard initialGaussianCount > 0 else {
            throw MsplatError.invalidDataset(
                "The COLMAP model must contain at least one sparse point."
            )
        }
        guard let iterationBudget = Int32(exactly: steps) else {
            throw MsplatError.invalidArgument("Iteration budget is outside the native range")
        }

        let sourceLongestEdge = max(sourceDimensions.width, sourceDimensions.height)
        let inputDecodeScale = min(
            32,
            max(1, Float(sourceLongestEdge) / Float(profile.longestEdge))
        )

        let stages: [TrainingResolutionStage]
        switch profile {
        case .preview:
            stages = [
                try TrainingResolutionStage(
                    iterations: 1...iterationBudget,
                    downscaleFactor: 1
                ),
            ]
        case .balanced:
            guard iterationBudget >= 2 else {
                throw MsplatError.invalidArgument(
                    "Balanced training requires at least two iterations"
                )
            }
            let coarseEnd = max(1, iterationBudget / 2)
            stages = [
                try TrainingResolutionStage(
                    iterations: 1...coarseEnd,
                    downscaleFactor: 2
                ),
                try TrainingResolutionStage(
                    iterations: (coarseEnd + 1)...iterationBudget,
                    downscaleFactor: 1
                ),
            ]
        }

        return try TrainingPlan(
            inputDimensions: sourceDimensions,
            inputDecodeScale: inputDecodeScale,
            iterationBudget: iterationBudget,
            stages: stages,
            targetSHDegree: profile.shDegree,
            maximumGaussianCount: max(
                profile.maximumGaussianCount,
                initialGaussianCount
            ),
            includesTrainingMasks: includesTrainingMasks
        )
    }

    private nonisolated static func megabytes(_ bytes: Int64) -> Int {
        Int((bytes + 1_048_575) / 1_048_576)
    }

    private nonisolated static func megabytes(_ bytes: UInt64) -> Int {
        Int((bytes + 1_048_575) / 1_048_576)
    }

    private nonisolated static func image(from frame: RGBAFrame) -> UIImage? {
        guard frame.width > 0, frame.height > 0,
              let provider = CGDataProvider(data: frame.data as CFData),
              let cg = CGImage(width: frame.width, height: frame.height,
                               bitsPerComponent: 8, bitsPerPixel: 32,
                               bytesPerRow: frame.bytesPerRow,
                               space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                               provider: provider, decode: nil,
                               shouldInterpolate: false, intent: .defaultIntent)
        else { return nil }
        return UIImage(cgImage: cg)
    }

    /// How much more the app may use before it is killed.
    private nonisolated static func availableMB() -> Int {
        Int(os_proc_available_memory()) >> 20
    }

    private nonisolated static func thermalStateDescription() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "Nominal"
        case .fair: "Fair"
        case .serious: "Serious"
        case .critical: "Critical"
        @unknown default: "Unknown"
        }
    }
}
