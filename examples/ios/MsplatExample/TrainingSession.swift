import Foundation
import Msplat
import SwiftUI
import os

@MainActor
final class TrainingSession: ObservableObject {
    enum Phase: Equatable {
        case idle, loading, training, finished, failed(String)
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

    private var worker: Thread?

    /// Written from the UI and read from the worker thread on every step, so
    /// it lives outside the actor — hopping to the main actor once per step
    /// just to ask whether to stop would serialise training against the UI.
    private final class CancelFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        var isSet: Bool { lock.withLock { value } }
        func set() { lock.withLock { value = true } }
        func reset() { lock.withLock { value = false } }
    }
    private nonisolated let cancelFlag = CancelFlag()

    func start(folder: DatasetFolder) {
        guard phase == .idle || phase == .finished || isFailed else { return }
        phase = .loading
        iteration = 0
        exportedPly = nil
        cancelFlag.reset()

        let path = folder.url.path(percentEncoded: false)
        let steps = iterations
        let scale = downscale

        let thread = Thread { [weak self] in
            self?.run(path: path, steps: steps, downscale: scale)
        }
        // The default 512KB secondary-thread stack is enough, but the loaders
        // recurse over directory trees; give them room.
        thread.stackSize = 4 << 20
        worker = thread
        thread.start()
    }

    func cancel() { cancelFlag.set() }

    private var isFailed: Bool { if case .failed = phase { return true }; return false }

    private nonisolated func run(path: String, steps: Int, downscale: Float) {
        autoreleasepool {
            let dataset = GaussianDataset(path: path, downscaleFactor: downscale)
            let cameras = dataset.numTrain
            guard cameras > 0 else {
                Task { @MainActor in self.phase = .failed("No training cameras in that folder.") }
                return
            }

            var config = TrainingConfig()
            config.iterations = Int32(steps)
            config.downscaleFactor = downscale
            let trainer = GaussianTrainer(dataset: dataset, config: config)

            Task { @MainActor in
                self.trainingCameras = cameras
                self.phase = .training
            }

            // Rendering a preview costs a full forward pass, so it is sampled
            // rather than done every step.
            let previewEvery = max(steps / 40, 10)

            for i in 0..<steps {
                if cancelFlag.isSet { break }
                let stats = trainer.step()

                if i % previewEvery == 0 || i == steps - 1 {
                    let image = Self.image(from: trainer.render(cameraIndex: 0))
                    let footprint = Self.footprintMB()
                    let available = Self.availableMB()
                    Task { @MainActor in
                        self.iteration = stats.iteration
                        self.splatCount = stats.splatCount
                        self.msPerStep = stats.msPerStep
                        self.preview = image
                        self.footprintMB = footprint
                        self.availableMB = available
                    }
                }
            }

            let url = URL.documentsDirectory.appending(path: "msplat-scene.ply")
            try? FileManager.default.removeItem(at: url)
            trainer.exportPly(to: url.path(percentEncoded: false))
            let finalCount = trainer.splatCount

            Task { @MainActor in
                self.splatCount = finalCount
                self.exportedPly = FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) ? url : nil
                self.phase = .finished
            }
        }
    }

    // MARK: - Helpers

    private nonisolated static func image(from pd: PixelData) -> UIImage? {
        let count = pd.width * pd.height
        guard count > 0 else { return nil }

        var rgba = [UInt8](repeating: 255, count: count * 4)
        pd.pixels.withUnsafeBufferPointer { src in
            for i in 0..<count {
                for c in 0..<3 {
                    rgba[i * 4 + c] = UInt8(min(max(src[i * 3 + c], 0), 1) * 255)
                }
            }
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cg = CGImage(width: pd.width, height: pd.height,
                               bitsPerComponent: 8, bitsPerPixel: 32,
                               bytesPerRow: pd.width * 4,
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
