import Foundation
import Msplat
import SwiftUI
import os

@MainActor
final class TrainingSession: ObservableObject {
    enum Phase: Equatable {
        case idle, loading, training, cancelled, finished, failed(String)
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

    /// Total steps. Kept modest by default: on a phone this is a battery and
    /// thermal budget as much as a quality one.
    @Published var iterations = 2_000
    /// Longest image edge the trainer works at. Full-resolution captures are
    /// where a phone runs out of memory first, so the default halves them.
    @Published var downscale: Float = 2

    private var worker: Task<Void, Never>?

    func start(folder: DatasetFolder) {
        guard phase == .idle || phase == .cancelled || phase == .finished || isFailed else { return }
        phase = .loading
        iteration = 0
        exportedPly = nil

        let datasetURL = folder.url
        let steps = iterations
        let scale = downscale

        worker = Task { [weak self] in
            guard let self else { return }
            await self.run(datasetURL: datasetURL, steps: steps, downscale: scale)
        }
    }

    func cancel() { worker?.cancel() }

    private var isFailed: Bool { if case .failed = phase { return true }; return false }

    private func run(datasetURL: URL, steps: Int, downscale: Float) async {
        var session: MsplatSession?
        var resultURL: URL?
        var finalSplatCount = 0
        var failureMessage: String?
        var wasCancelled = false

        do {
            var config = TrainingConfig()
            config.iterations = Int32(steps)
            // The UI's Full/Half/Quarter choice is the complete resolution
            // policy for this short mobile preview; do not apply the desktop
            // trainer's additional progressive 4x downscale.
            config.numDownscales = 0

            let activeSession = try await MsplatSession(
                datasetURL: datasetURL,
                options: DatasetOptions(downscaleFactor: downscale),
                config: config
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
