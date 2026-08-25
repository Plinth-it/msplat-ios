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
    @Published private(set) var preview: MetalPreviewSurface?
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

    private struct PendingPreview {
        let id: UInt64
        let task: Task<Void, Never>
    }

    private var worker: Task<Void, Never>?
    private var pendingPreview: PendingPreview?
    private var pendingPreviewFailure: Error?
    private var nextPreviewID: UInt64 = 0
    private var previewGeneration: UInt64 = 0

    func start(folder: DatasetFolder) {
        guard phase == .idle || phase == .cancelled || phase == .finished || isFailed else { return }
        phase = .planning
        submittedIteration = 0
        completedIteration = 0
        splatCount = 0
        modelCapacity = 0
        cpuSubmitMs = 0
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
        pendingPreviewFailure = nil
        previewGeneration &+= 1

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

    func cancel() {
        worker?.cancel()
        pendingPreview?.task.cancel()
    }

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
            let appEstimatedPeakMemory = try Self.appEstimatedPeakMemory(for: plan)
            estimatedPeakMB = Self.megabytes(appEstimatedPeakMemory)
            plannedSHDegree = Int(plan.targetSHDegree)
            plannedInitialGaussians = initialGaussianCount
            plannedMaximumGaussians = plan.maximumGaussianCount
            availableMB = Self.availableMB()

            let availableBytes = Int64(availableMB) * 1_024 * 1_024
            guard availableBytes == 0 || appEstimatedPeakMemory <= availableBytes else {
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

            // Preview submission is asynchronous, but still costs a full
            // forward pass. Keep one submission in flight and sample it less
            // frequently than the independent telemetry poll.
            let previewEvery = Self.previewInterval(for: steps)
            let telemetryEvery = max(steps / 40, 10)

            for i in 0..<steps {
                try Task.checkCancellation()
                try throwPendingPreviewFailure()
                let stats = try await activeSession.step()
                let isFinalStep = i == steps - 1

                if isFinalStep {
                    // Preserve the last completed surface while the final one
                    // finishes, and do not tear down the native session until
                    // its app-owned texture is ready.
                    await drainPendingPreview()
                    try throwPendingPreviewFailure()
                    try await submitPreviewIfIdle(
                        session: activeSession,
                        pose: previewPose
                    )
                    await drainPendingPreview()
                    try throwPendingPreviewFailure()
                } else if i % previewEvery == 0 {
                    try await submitPreviewIfIdle(
                        session: activeSession,
                        pose: previewPose
                    )
                }

                // This is a non-blocking snapshot. Submitted and completed
                // iterations may legitimately differ now that preview display
                // no longer drains the training queue.
                if i % telemetryEvery == 0 || isFinalStep {
                    try await refreshTelemetry(
                        session: activeSession,
                        fallback: stats
                    )
                }
            }

            try Task.checkCancellation()
            let url = URL.documentsDirectory.appending(path: "msplat-scene.ply")
            try await activeSession.exportPLY(to: url)
            try Task.checkCancellation()
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MsplatError.ioFailure("Training finished, but the PLY file was not created.")
            }
            finalSplatCount = try await activeSession.splatCount
            try Task.checkCancellation()
            resultURL = url
        } catch is CancellationError {
            // A cancelled run deliberately does not export a partial scene.
            wasCancelled = true
        } catch {
            failureMessage = error.localizedDescription
        }

        // An in-flight preview owns native GPU resources. Let it finish before
        // closing the session, even when the training task was cancelled.
        await drainPendingPreview()

        if let session {
            do {
                try await session.close()
            } catch {
                let closeMessage = "Could not release the training session: \(error.localizedDescription)"
                failureMessage = failureMessage.map { "\($0) \(closeMessage)" } ?? closeMessage
            }
        }

        if Task.isCancelled {
            wasCancelled = true
            resultURL = nil
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

    nonisolated static func previewInterval(for steps: Int) -> Int {
        max(steps / 20, 100)
    }

    /// Adds the two app-owned RGBA8 preview surfaces to the native plan's peak.
    nonisolated static func appEstimatedPeakMemory(for plan: TrainingPlan) throws -> Int64 {
        var largestPixelCount: Int64 = 0
        for stage in plan.resolvedStages {
            guard let width = Int64(exactly: stage.dimensions.width),
                  let height = Int64(exactly: stage.dimensions.height) else {
                throw MsplatError.outOfMemory("Preview surface dimensions are too large")
            }
            let (pixelCount, pixelCountOverflowed) = width.multipliedReportingOverflow(
                by: height
            )
            guard !pixelCountOverflowed else {
                throw MsplatError.outOfMemory("Preview surface size overflowed")
            }
            largestPixelCount = max(largestPixelCount, pixelCount)
        }

        // Two surfaces at four bytes per pixel let one stay on screen while
        // the next preview is produced.
        let (previewBytes, previewBytesOverflowed) = largestPixelCount
            .multipliedReportingOverflow(by: 2 * 4)
        guard !previewBytesOverflowed else {
            throw MsplatError.outOfMemory("Preview surface memory estimate overflowed")
        }
        let (total, totalOverflowed) = plan.estimatedPeakMemory.addingReportingOverflow(
            previewBytes
        )
        guard !totalOverflowed else {
            throw MsplatError.outOfMemory("Training memory estimate overflowed")
        }
        return total
    }

    private func submitPreviewIfIdle(
        session: MsplatSession,
        pose: CameraPose
    ) async throws {
        try throwPendingPreviewFailure()
        guard pendingPreview == nil else { return }

        let submission = try await session.submitPreview(
            pose: pose,
            referenceCamera: 0
        )
        try Task.checkCancellation()
        nextPreviewID &+= 1
        let id = nextPreviewID
        let generation = previewGeneration
        let task = Task { @MainActor [weak self] in
            let result: Result<MetalPreviewSurface, Error>
            do {
                result = .success(try await submission.waitUntilReady())
            } catch {
                result = .failure(error)
            }
            self?.completePendingPreview(
                id: id,
                generation: generation,
                result: result
            )
        }
        pendingPreview = PendingPreview(id: id, task: task)
    }

    private func completePendingPreview(
        id: UInt64,
        generation: UInt64,
        result: Result<MetalPreviewSurface, Error>
    ) {
        guard pendingPreview?.id == id else { return }
        pendingPreview = nil
        guard generation == previewGeneration else { return }

        switch result {
        case .success(let surface):
            preview = surface
        case .failure(let error):
            pendingPreviewFailure = error
        }
    }

    private func drainPendingPreview() async {
        guard let pending = pendingPreview else { return }
        await pending.task.value
        if pendingPreview?.id == pending.id {
            pendingPreview = nil
        }
    }

    private func throwPendingPreviewFailure() throws {
        if let pendingPreviewFailure {
            throw pendingPreviewFailure
        }
    }

    private func refreshTelemetry(
        session: MsplatSession,
        fallback stats: TrainingStats
    ) async throws {
        let telemetry = try await session.trainingMetrics()
        let memory = try await session.memoryMetrics()
        submittedIteration = telemetry.submitted?.iteration ?? stats.iteration
        cpuSubmitMs = telemetry.submitted?.cpuSubmitMs ?? stats.cpuSubmitMs
        if let completed = telemetry.completed {
            completedIteration = completed.iteration
            splatCount = completed.splatCount
            modelCapacity = completed.modelCapacity
            gpuExecutionMs = completed.gpuExecutionMs
            endToEndMs = completed.endToEndMs
            loss = completed.loss
            effectiveWidth = completed.effectiveWidth
            effectiveHeight = completed.effectiveHeight
            activeSHDegree = completed.activeSHDegree
            retainedPackedIntersections = completed.retainedPackedIntersectionCount
            packedIntersectionCapacity = completed.packedIntersectionCapacity
            overflowKinds = completed.overflowKinds
        }
        overflowedCompletedSteps = telemetry.overflowedCompletedSteps
        memorySnapshot = memory
        footprintMB = memory.processPhysicalFootprintBytes.map(Self.megabytes) ?? 0
        availableMB = memory.processAvailableBytes.map(Self.megabytes) ?? 0
        thermalState = Self.thermalStateDescription()
    }

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
