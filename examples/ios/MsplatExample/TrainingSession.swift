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
    @Published private(set) var iteration = 0
    @Published private(set) var splatCount = 0
    @Published private(set) var msPerStep: Float = 0
    @Published private(set) var preview: UIImage?
    @Published private(set) var trainingCameras = 0
    @Published private(set) var footprintMB = 0
    @Published private(set) var availableMB = 0
    @Published private(set) var exportedPly: URL?
    @Published private(set) var plannedStages: [ResolvedTrainingResolutionStage] = []
    @Published private(set) var estimatedPeakMB = 0
    @Published private(set) var plannedSHDegree = 0
    @Published private(set) var plannedMaximumGaussians = 0

    /// Total steps. Kept modest by default: on a phone this is a battery and
    /// thermal budget as much as a quality one.
    @Published var iterations = 2_000
    @Published var qualityProfile: QualityProfile = .preview

    private var worker: Task<Void, Never>?

    func start(folder: DatasetFolder) {
        guard phase == .idle || phase == .cancelled || phase == .finished || isFailed else { return }
        phase = .planning
        iteration = 0
        splatCount = 0
        msPerStep = 0
        preview = nil
        trainingCameras = 0
        footprintMB = 0
        availableMB = 0
        exportedPly = nil
        plannedStages = []
        estimatedPeakMB = 0
        plannedSHDegree = 0
        plannedMaximumGaussians = 0

        let steps = iterations
        let profile = qualityProfile

        worker = Task { [weak self] in
            guard let self else { return }
            await self.run(folder: folder, steps: steps, profile: profile)
        }
    }

    func cancel() { worker?.cancel() }

    private var isFailed: Bool { if case .failed = phase { return true }; return false }

    private func run(
        folder: DatasetFolder,
        steps: Int,
        profile: QualityProfile
    ) async {
        var session: MsplatSession?
        var resultURL: URL?
        var finalSplatCount = 0
        var failureMessage: String?
        var wasCancelled = false

        do {
            let datasetURL = folder.url
            let dimensionScan = Task.detached(priority: .userInitiated) {
                try DatasetFolder.maximumSourceDimensions(at: datasetURL)
            }
            let sourceDimensions = try await withTaskCancellationHandler {
                try await dimensionScan.value
            } onCancel: {
                dimensionScan.cancel()
            }
            let plan = try Self.makePlan(
                sourceDimensions: sourceDimensions,
                steps: steps,
                profile: profile
            )
            plannedStages = plan.resolvedStages
            estimatedPeakMB = Self.megabytes(plan.estimatedPeakMemory)
            plannedSHDegree = Int(plan.targetSHDegree)
            plannedMaximumGaussians = plan.maximumGaussianCount
            availableMB = Self.availableMB()

            let availableBytes = Int64(availableMB) * 1_024 * 1_024
            guard availableBytes == 0 || plan.estimatedPeakMemory <= availableBytes else {
                throw MsplatError.outOfMemory(
                    "The \(profile.rawValue) plan estimates \(estimatedPeakMB) MB, " +
                    "but iOS currently reports \(availableMB) MB available. " +
                    "Choose a smaller plan or close other apps."
                )
            }

            phase = .loading
            try Task.checkCancellation()

            let activeSession = try await MsplatSession(
                datasetURL: folder.url,
                trainingPlan: plan
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
                    iteration = stats.iteration
                    splatCount = stats.splatCount
                    msPerStep = stats.cpuSubmitMs
                    preview = Self.image(from: frame)
                    footprintMB = Self.footprintMB()
                    availableMB = Self.availableMB()
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

    private nonisolated static func makePlan(
        sourceDimensions: TrainingImageDimensions,
        steps: Int,
        profile: QualityProfile
    ) throws -> TrainingPlan {
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
            maximumGaussianCount: profile.maximumGaussianCount
        )
    }

    private nonisolated static func megabytes(_ bytes: Int64) -> Int {
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

    /// What jetsam actually counts against the app.
    private nonisolated static func footprintMB() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Int(info.phys_footprint) >> 20
    }

    /// How much more the app may use before it is killed.
    private nonisolated static func availableMB() -> Int {
        Int(os_proc_available_memory()) >> 20
    }
}
