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
    @Published private(set) var queueIdleMs: Float?
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
    @Published private(set) var benchmarkResultURL: URL?
    @Published private(set) var benchmarkSummary: String?

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
    private var benchmarkRequestHandled = false

    func start(source: TrainingDatasetSource) {
        start(
            source: source,
            benchmark: nil,
            trainingMaskCandidateCount: nil
        )
    }

    func start(folder: DatasetFolder) {
        start(source: .importedFolder(folder))
    }

    /// Starts one launch-configured physical-device benchmark after dataset
    /// mask discovery has settled. Repeated SwiftUI task invocations are safe.
    func startBenchmarkIfRequested(
        folder: DatasetFolder,
        maskCandidateCount: Int?
    ) {
        guard !benchmarkRequestHandled,
              let benchmark = TrainingBenchmarkConfiguration.requested(),
              phase == .idle || phase == .cancelled || phase == .finished || isFailed else {
            return
        }
        if folder.supportsAutomaticTrainingMaskDiscovery,
           maskCandidateCount == nil {
            return
        }

        benchmarkRequestHandled = true
        iterations = benchmark.totalIterations
        qualityProfile = .preview
        start(
            source: .importedFolder(folder),
            benchmark: benchmark,
            trainingMaskCandidateCount: maskCandidateCount
        )
    }

    private func start(
        source: TrainingDatasetSource,
        benchmark: TrainingBenchmarkConfiguration?,
        trainingMaskCandidateCount: Int?
    ) {
        guard phase == .idle || phase == .cancelled || phase == .finished || isFailed else { return }
        phase = .planning
        submittedIteration = 0
        completedIteration = 0
        splatCount = 0
        modelCapacity = 0
        cpuSubmitMs = 0
        imagePrepareMs = 0
        gpuExecutionMs = nil
        queueIdleMs = nil
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
        benchmarkResultURL = nil
        benchmarkSummary = nil
        pendingPreviewFailure = nil
        previewGeneration &+= 1

        let steps = benchmark?.totalIterations ?? iterations
        let profile: QualityProfile = benchmark == nil ? qualityProfile : .preview
        let useTrainingMasks = source.capturedDataset?.descriptor.frames.contains {
            $0.trainingMask != nil
        } ?? (source.importedFolder?.hasNerfstudioTrainingMasks == true ||
              trainingMasksEnabled)
        let selectedTrainingMaskMode = source.capturedDataset?.manifest.mode == .object
            ? TrainingMaskMode.transparent : trainingMaskMode

        worker = Task { [weak self] in
            guard let self else { return }
            await self.run(
                source: source,
                steps: steps,
                profile: profile,
                useTrainingMasks: useTrainingMasks,
                trainingMaskMode: selectedTrainingMaskMode,
                benchmark: benchmark,
                trainingMaskCandidateCount: trainingMaskCandidateCount
            )
        }
    }

    func cancel() {
        worker?.cancel()
        pendingPreview?.task.cancel()
    }

    private var isFailed: Bool { if case .failed = phase { return true }; return false }

    private func run(
        source: TrainingDatasetSource,
        steps: Int,
        profile: QualityProfile,
        useTrainingMasks: Bool,
        trainingMaskMode: TrainingMaskMode,
        benchmark: TrainingBenchmarkConfiguration?,
        trainingMaskCandidateCount: Int?
    ) async {
        var session: MsplatSession?
        var resultURL: URL?
        var finalSplatCount = 0
        var failureMessage: String?
        var wasCancelled = false
        var benchmarkRecorder: TrainingBenchmarkRecorder?
        if let benchmark, let folder = source.importedFolder {
            benchmarkRecorder = TrainingBenchmarkRecorder(
                configuration: benchmark,
                folder: folder,
                trainingMaskCandidateCount: trainingMaskCandidateCount,
                trainingMasksEnabled: useTrainingMasks,
                trainingMaskMode: trainingMaskMode
            )
        }

        do {
            let (sourceDimensions, initialGaussianCount) = try await Self.scan(
                source: source
            )
            let benchmarkMaximumGaussianCount = try benchmark?
                .validatedGrowthMaximumGaussianCount(
                    initialGaussianCount: initialGaussianCount
                )
            let plan = try Self.makePlan(
                sourceDimensions: sourceDimensions,
                initialGaussianCount: initialGaussianCount,
                steps: steps,
                profile: profile,
                includesTrainingMasks: useTrainingMasks,
                maximumGaussianCountOverride: benchmarkMaximumGaussianCount
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
                        : "Turn off masks, use a sparser point cloud, or close other apps.")
                )
            }

            phase = .loading
            try Task.checkCancellation()

            let baseConfig = try Self.makeTrainingConfig(
                trainingMaskMode: trainingMaskMode,
                keepCrs: source.capturedDataset != nil,
                benchmark: benchmark
            )
            let activeSession: MsplatSession
            switch source {
            case .importedFolder(let folder):
                activeSession = try await MsplatSession(
                    datasetURL: folder.url,
                    trainingPlan: plan,
                    baseConfig: baseConfig,
                    prefetchTrainingTargets: true
                )
            case .captured(let capture):
                activeSession = try await MsplatSession(
                    dataset: capture.descriptor,
                    options: plan.makeDatasetOptions(
                        prefetchTrainingTargets: true
                    ),
                    config: plan.makeTrainingConfig(startingFrom: baseConfig),
                    maximumGaussianCount: plan.maximumGaussianCount
                )
            }
            session = activeSession
            try Task.checkCancellation()

            let cameras = try await activeSession.numTrain
            guard cameras > 0 else {
                throw MsplatError.invalidDataset("No training cameras in that folder.")
            }
            trainingCameras = cameras
            phase = .training
            let previewPose: CameraPose?
            if benchmark == nil {
                previewPose = try await activeSession.cameraPose(at: 0)
            } else {
                previewPose = nil
            }

            // Preview submission is asynchronous, but still costs a full
            // forward pass. Keep one submission in flight and sample it less
            // frequently than the independent telemetry poll.
            let previewEvery = Self.previewInterval(for: steps)
            let telemetryEvery = max(steps / 40, 10)
            var latestStats: TrainingStats?
            var measuredStartedAt: TimeInterval?

            for i in 0..<steps {
                try Task.checkCancellation()
                try throwPendingPreviewFailure()
                if let benchmark, i == benchmark.warmupIterations {
                    // Drain warm-up work so this interval measures only steady-state
                    // submissions through completion, without setup or export time.
                    msplatSync()
                    measuredStartedAt = ProcessInfo.processInfo.systemUptime
                }
                let nextIteration = i + 1
                if let benchmark,
                   !benchmark.fixedPopulation,
                   let latestStats,
                   Self.isDensificationStep(
                       nextIteration,
                       config: baseConfig,
                       cameraCount: cameras
                   ) {
                    // Preserve the completed descriptor immediately before a
                    // growth candidate. The transition sample can then prove
                    // that its own end-to-end time includes the capacity copy.
                    msplatSync()
                    let snapshot = try await telemetrySnapshot(
                        session: activeSession,
                        fallback: latestStats
                    )
                    benchmarkRecorder?.record(
                        telemetry: snapshot.telemetry,
                        memory: snapshot.memory,
                        thermalState: thermalState
                    )
                }
                let stats = try await activeSession.step()
                latestStats = stats
                let isFinalStep = i == steps - 1

                if benchmark != nil {
                    let snapshot = try await telemetrySnapshot(
                        session: activeSession,
                        fallback: stats
                    )
                    benchmarkRecorder?.record(
                        telemetry: snapshot.telemetry,
                        memory: snapshot.memory,
                        thermalState: thermalState
                    )
                } else if isFinalStep, let previewPose {
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
                } else if i % previewEvery == 0, let previewPose {
                    try await submitPreviewIfIdle(
                        session: activeSession,
                        pose: previewPose
                    )
                }

                // This is a non-blocking snapshot. Submitted and completed
                // iterations may legitimately differ now that preview display
                // no longer drains the training queue.
                if benchmark == nil && (i % telemetryEvery == 0 || isFinalStep) {
                    try await refreshTelemetry(
                        session: activeSession,
                        fallback: stats
                    )
                }
            }

            try Task.checkCancellation()
            let measuredElapsedSeconds: TimeInterval?
            if benchmark != nil, let measuredStartedAt {
                msplatSync()
                measuredElapsedSeconds = ProcessInfo.processInfo.systemUptime - measuredStartedAt
            } else {
                measuredElapsedSeconds = nil
            }
            if benchmark == nil {
                let url = source.capturedDataset?.rootURL.appending(
                    path: "trained-scene.ply"
                ) ?? URL.documentsDirectory.appending(path: "msplat-scene.ply")
                try await activeSession.exportPLY(to: url)
                try Task.checkCancellation()
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw MsplatError.ioFailure(
                        "Training finished, but the PLY file was not created."
                    )
                }
                resultURL = url
            }
            finalSplatCount = try await activeSession.splatCount
            if benchmarkRecorder != nil {
                guard let latestStats else {
                    throw MsplatError.internalFailure(
                        "Benchmark completed without submitting a training step."
                    )
                }
                let snapshot = try await telemetrySnapshot(
                    session: activeSession,
                    fallback: latestStats
                )
                benchmarkRecorder?.record(
                    telemetry: snapshot.telemetry,
                    memory: snapshot.memory,
                    thermalState: thermalState,
                    isFinalDescriptor: true
                )
                if var recorder = benchmarkRecorder {
                    guard let measuredElapsedSeconds else {
                        throw MsplatError.internalFailure(
                            "Benchmark measured interval was not started."
                        )
                    }
                    let output = try recorder.finish(
                        measuredElapsedSeconds: measuredElapsedSeconds
                    )
                    benchmarkRecorder = recorder
                    benchmarkResultURL = output.url
                    benchmarkSummary = output.summaryLine
                    print("MSPLAT_BENCHMARK_RESULT \(output.summaryLine)")
                }
            }
            try Task.checkCancellation()
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
            if benchmark != nil {
                let singleLine = failureMessage
                    .replacingOccurrences(of: "\n", with: " ")
                    .replacingOccurrences(of: "\r", with: " ")
                print("MSPLAT_BENCHMARK_FAILURE \(singleLine)")
            }
        } else if wasCancelled {
            phase = .cancelled
            if benchmark != nil {
                print("MSPLAT_BENCHMARK_FAILURE cancelled")
            }
        } else {
            splatCount = finalSplatCount
            exportedPly = resultURL
            phase = .finished
        }
        worker = nil
    }

    // MARK: - Helpers

    private static func scan(
        source: TrainingDatasetSource
    ) async throws -> (TrainingImageDimensions, Int) {
        switch source {
        case .importedFolder(let folder):
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
            return try await withTaskCancellationHandler {
                try await datasetScan.value
            } onCancel: {
                datasetScan.cancel()
            }
        case .captured(let capture):
            let maximumWidth = capture.descriptor.frames
                .map(\.calibration.width)
                .max() ?? 0
            let maximumHeight = capture.descriptor.frames
                .map(\.calibration.height)
                .max() ?? 0
            return (
                try TrainingImageDimensions(
                    width: maximumWidth,
                    height: maximumHeight
                ),
                capture.descriptor.points.count
            )
        }
    }

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
        _ = try await telemetrySnapshot(session: session, fallback: stats)
    }

    private func telemetrySnapshot(
        session: MsplatSession,
        fallback stats: TrainingStats
    ) async throws -> (
        telemetry: TrainingTelemetry,
        memory: TrainingMemorySnapshot
    ) {
        let telemetry = try await session.trainingMetrics()
        let memory = try await session.memoryMetrics()
        applyTelemetry(telemetry, memory: memory, fallback: stats)
        return (telemetry, memory)
    }

    private func applyTelemetry(
        _ telemetry: TrainingTelemetry,
        memory: TrainingMemorySnapshot,
        fallback stats: TrainingStats
    ) {
        submittedIteration = telemetry.submitted?.iteration ?? stats.iteration
        cpuSubmitMs = telemetry.submitted?.cpuSubmitMs ?? stats.cpuSubmitMs
        if let completed = telemetry.completed {
            completedIteration = completed.iteration
            splatCount = completed.splatCount
            modelCapacity = completed.modelCapacity
            imagePrepareMs = completed.imagePrepareMs
            gpuExecutionMs = completed.gpuExecutionMs
            queueIdleMs = completed.queueIdleMs
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
        includesTrainingMasks: Bool = false,
        maximumGaussianCountOverride: Int? = nil
    ) throws -> TrainingPlan {
        guard initialGaussianCount > 0 else {
            throw MsplatError.invalidDataset(
                "The dataset must contain at least one initial point."
            )
        }
        guard let iterationBudget = Int32(exactly: steps) else {
            throw MsplatError.invalidArgument("Iteration budget is outside the native range")
        }
        if let maximumGaussianCountOverride,
           maximumGaussianCountOverride <= initialGaussianCount {
            throw MsplatError.invalidArgument(
                "Maximum Gaussian count must exceed the initial population"
            )
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
            maximumGaussianCount: maximumGaussianCountOverride ??
                max(profile.maximumGaussianCount, initialGaussianCount),
            includesTrainingMasks: includesTrainingMasks
        )
    }

    nonisolated static func makeTrainingConfig(
        trainingMaskMode: TrainingMaskMode,
        keepCrs: Bool = false,
        benchmark: TrainingBenchmarkConfiguration?
    ) throws -> TrainingConfig {
        var config = TrainingConfig()
        config.trainingMaskMode = trainingMaskMode
        config.keepCrs = keepCrs
        guard let benchmark else { return config }

        if benchmark.fixedPopulation {
            config.stopDensifyAt = 0
            return config
        }

        guard let stopDensifyAt = Int32(exactly: benchmark.totalIterations + 1) else {
            throw MsplatError.invalidArgument(
                "Growth benchmark iteration budget is outside the native range"
            )
        }
        config.warmupLength = 0
        config.refineEvery = 25
        config.resetAlphaEvery = 100
        // Make the synthetic growth event deterministic enough to exercise
        // capacity preservation rather than depend on scene-specific tuning.
        config.densifyGradThresh = 0
        config.stopDensifyAt = stopDensifyAt
        return config
    }

    nonisolated static func isDensificationStep(
        _ step: Int,
        config: TrainingConfig,
        cameraCount: Int
    ) -> Bool {
        let refineEvery = Int(config.refineEvery)
        let resetInterval = Int(config.resetAlphaEvery) * refineEvery
        return step % refineEvery == 0 &&
            step > Int(config.warmupLength) &&
            step < Int(config.stopDensifyAt) &&
            step % resetInterval > cameraCount + refineEvery
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
