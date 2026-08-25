import Msplat
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var session = TrainingSession()
    @State private var folder: DatasetFolder?
    @State private var picking = false
    @State private var pickError: String?
    @State private var trainingMaskCandidateCount: Int?
    @State private var trainingMaskSelectionWasEdited = false
    @State private var didAttemptDatasetRestore = false

    var body: some View {
        NavigationStack {
            Form {
                datasetSection
                if folder != nil { settingsSection }
                if session.phase != .idle { progressSection }
                if let ply = session.exportedPly { exportSection(ply) }
            }
            .navigationTitle("msplat")
            .fileImporter(isPresented: $picking, allowedContentTypes: [.folder]) { result in
                switch result {
                case .success(let url):
                    if let picked = DatasetFolder(picked: url) {
                        select(picked, persistBookmark: true)
                    } else {
                        pickError = "Choose a folder with root transforms.json, or COLMAP cameras.bin/cameras.txt at its root or in sparse/0."
                    }
                case .failure(let error):
                    pickError = error.localizedDescription
                }
            }
            .task {
                restoreLastDatasetIfNeeded()
            }
            .task(id: folder?.id) {
                guard let selectedFolder = folder else { return }
                await scanTrainingMasks(in: selectedFolder)
                startBenchmarkIfReady()
            }
            .onChange(of: scenePhase) { _, newPhase in
                if newPhase == .active {
                    startBenchmarkIfReady()
                }
            }
        }
    }

    @MainActor
    private func startBenchmarkIfReady() {
        guard scenePhase == .active, let folder else { return }
        session.startBenchmarkIfRequested(
            folder: folder,
            maskCandidateCount: trainingMaskCandidateCount
        )
    }

    @MainActor
    private func restoreLastDatasetIfNeeded() {
        guard !didAttemptDatasetRestore else { return }
        didAttemptDatasetRestore = true
        guard folder == nil else { return }
        let restored = DatasetFolder.benchmarkDatasetFromDocuments()
            ?? DatasetFolder.restoreLastPicked()
        guard let restored else {
            return
        }
        select(restored, persistBookmark: false)
    }

    @MainActor
    private func select(
        _ selectedFolder: DatasetFolder,
        persistBookmark: Bool
    ) {
        folder = selectedFolder
        trainingMaskCandidateCount = nil
        trainingMaskSelectionWasEdited = false
        session.trainingMasksEnabled = false
        session.trainingMaskMode = .transparent
        pickError = nil

        guard persistBookmark else { return }
        do {
            try selectedFolder.persistAsLastPicked()
        } catch {
            pickError = "The folder is selected, but it could not be remembered: " +
                error.localizedDescription
        }
    }

    private var datasetSection: some View {
        Section("Training dataset") {
            Button {
                picking = true
            } label: {
                Label(folder?.name ?? "Choose a folder…", systemImage: "folder")
            }
            .disabled(isBusy)

            if let folder {
                LabeledContent("Contents", value: folder.summary)
            }
            if let pickError {
                Text(pickError).font(.footnote).foregroundStyle(.red)
            }
        }
    }

    private var settingsSection: some View {
        Section {
            Stepper("Iterations: \(session.iterations)",
                    value: $session.iterations, in: 200...20_000, step: 200)
            Picker("Quality", selection: $session.qualityProfile) {
                ForEach(TrainingSession.QualityProfile.allCases) { profile in
                    Text(profile.rawValue).tag(profile)
                }
            }
            if folder?.supportsAutomaticTrainingMaskDiscovery == true {
                Toggle("Use discovered masks", isOn: trainingMasksBinding)
                    .disabled(isBusy)
                if session.trainingMasksEnabled {
                    Picker("Mask treatment", selection: $session.trainingMaskMode) {
                        Text("Transparent").tag(TrainingMaskMode.transparent)
                        Text("Coverage only").tag(TrainingMaskMode.coverage)
                    }
                    .disabled(isBusy)
                }
                LabeledContent(
                    "Mask candidates",
                    value: trainingMaskCandidateCount?.formatted() ?? "Scanning…"
                )
            } else if folder?.kind == .nerfstudio {
                Text("Automatic mask sidecars are not yet available for Nerfstudio imports.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Button(trainButtonTitle) {
                if let folder { session.start(folder: folder) }
            }
            .disabled(
                isBusy ||
                (folder?.supportsAutomaticTrainingMaskDiscovery == true &&
                 trainingMaskCandidateCount == nil &&
                 !trainingMaskSelectionWasEdited)
            )

            if isBusy {
                Button("Stop", role: .destructive) { session.cancel() }
            }
        } footer: {
            Text("Preview targets a 1,600-pixel edge, SH1, and a 250K Gaussian ceiling. Balanced targets 1,920 pixels, SH2, and a 400K ceiling when preflight memory permits. Either ceiling rises only enough to preserve a larger initial point cloud, and the memory estimate is recomputed.")
            if folder?.supportsAutomaticTrainingMaskDiscovery == true {
                Text("Mask candidates are regular files below any masks/ path component; the native loader decides which candidates match frames. Coverage only weights RGB loss and can skip off-mask tile work for throughput. Transparent supervises the full frame to suppress exterior floaters and is not expected to be faster.")
            }
        }
    }

    private var trainingMasksBinding: Binding<Bool> {
        Binding(
            get: { session.trainingMasksEnabled },
            set: { enabled in
                trainingMaskSelectionWasEdited = true
                session.trainingMasksEnabled = enabled
            }
        )
    }

    @MainActor
    private func scanTrainingMasks(in selectedFolder: DatasetFolder) async {
        guard selectedFolder.supportsAutomaticTrainingMaskDiscovery else {
            session.trainingMasksEnabled = false
            trainingMaskCandidateCount = 0
            return
        }

        let datasetURL = selectedFolder.url
        let scan = Task.detached(priority: .utility) {
            DatasetFolder.countTrainingMaskCandidates(at: datasetURL)
        }
        let count = await withTaskCancellationHandler {
            await scan.value
        } onCancel: {
            scan.cancel()
        }

        guard !Task.isCancelled, folder === selectedFolder else { return }
        trainingMaskCandidateCount = count
        if count > 0 && !trainingMaskSelectionWasEdited {
            session.trainingMasksEnabled = true
        }
    }

    private var progressSection: some View {
        Section("Progress") {
            if !session.plannedStages.isEmpty {
                ForEach(Array(session.plannedStages.enumerated()), id: \.offset) { _, stage in
                    LabeledContent(
                        "Steps \(stage.iterations.lowerBound)–\(stage.iterations.upperBound)",
                        value: "\(stage.dimensions.width) × \(stage.dimensions.height)"
                    )
                }
                LabeledContent("Final SH degree", value: "\(session.plannedSHDegree)")
                LabeledContent(
                    "Initial Gaussians",
                    value: session.plannedInitialGaussians.formatted()
                )
                LabeledContent(
                    "Gaussian limit",
                    value: session.plannedMaximumGaussians.formatted()
                )
                LabeledContent("Estimated peak", value: "\(session.estimatedPeakMB) MB")
            }

            switch session.phase {
            case .planning:
                LabeledContent("Status", value: "Planning device-safe training…")
            case .loading:
                LabeledContent("Status", value: "Loading dataset…")
            case .cancelled:
                LabeledContent("Status", value: "Cancelled")
            case .failed(let message):
                Text(message).foregroundStyle(.red)
            default:
                if let preview = session.preview {
                    MetalPreviewView(surface: preview)
                        .aspectRatio(
                            CGFloat(preview.width) / CGFloat(max(preview.height, 1)),
                            contentMode: .fit
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .listRowInsets(EdgeInsets())
                }
                ProgressView(value: Double(session.completedIteration),
                             total: Double(max(session.iterations, 1)))
                LabeledContent(
                    "Submitted",
                    value: "\(session.submittedIteration) / \(session.iterations)"
                )
                LabeledContent(
                    "Completed",
                    value: "\(session.completedIteration) / \(session.iterations)"
                )
                LabeledContent(
                    "Gaussians",
                    value: session.modelCapacity > 0
                        ? "\(session.splatCount.formatted()) / \(session.modelCapacity.formatted())"
                        : session.splatCount.formatted()
                )
                LabeledContent("CPU submit", value: duration(session.cpuSubmitMs))
                LabeledContent("Image prepare", value: duration(session.imagePrepareMs))
                LabeledContent("GPU execute", value: duration(session.gpuExecutionMs))
                LabeledContent("GPU gap", value: duration(session.queueIdleMs))
                LabeledContent("End to end", value: duration(session.endToEndMs))
                LabeledContent("Loss", value: session.loss.map {
                    String(format: "%.5f", $0)
                } ?? "—")
                LabeledContent(
                    "Training frame",
                    value: session.effectiveWidth > 0
                        ? "\(session.effectiveWidth) × \(session.effectiveHeight), SH\(session.activeSHDegree)"
                        : "—"
                )
                LabeledContent(
                    "Packed intersections",
                    value: packedIntersectionDescription
                )
                LabeledContent("Cameras", value: "\(session.trainingCameras)")
                LabeledContent("Process memory", value: session.footprintMB > 0
                    ? (session.availableMB > 0
                    ? "\(session.footprintMB) MB used, \(session.availableMB) MB free"
                    : "\(session.footprintMB) MB used")
                    : "—")
                if let memory = session.memorySnapshot {
                    LabeledContent(
                        "Tracked GPU buffers",
                        value: bytes(memory.trackedNativeBufferBytes)
                    )
                    LabeledContent(
                        "Model / transients",
                        value: "\(bytes(memory.trainerModelBufferBytes)) / " +
                            "\(bytes(memory.engineSharedTransientBufferBytes + memory.engineTrainingTransientBufferBytes))"
                    )
                    LabeledContent(
                        "Image cache CPU / GPU",
                        value: "\(bytes(memory.trainerImageCacheCpuBytes)) / " +
                            "\(bytes(memory.trainerImageCacheGpuBytes))"
                    )
                    LabeledContent(
                        "GPU image-cache hits",
                        value: memory.trainingGpuImageCacheHitRate.map {
                            String(format: "%.0f%%", $0 * 100)
                        } ?? "—"
                    )
                }
                LabeledContent("Thermal", value: session.thermalState)
                if session.overflowedCompletedSteps > 0 {
                    Text(
                        "Rasterizer overflow on \(session.overflowedCompletedSteps) completed " +
                        "step(s); latest source: \(overflowDescription)."
                    )
                    .foregroundStyle(.red)
                }
            }
        }
    }

    private func duration(_ milliseconds: Float?) -> String {
        milliseconds.map { String(format: "%.1f ms", $0) } ?? "—"
    }

    private func duration(_ milliseconds: Float) -> String {
        String(format: "%.1f ms", milliseconds)
    }

    private func bytes(_ count: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(clamping: count), countStyle: .memory)
    }

    private var overflowDescription: String {
        var sources: [String] = []
        if session.overflowKinds.contains(.tileCapacity) { sources.append("legacy tile cap") }
        if session.overflowKinds.contains(.packedCapacity) { sources.append("packed arena") }
        return sources.isEmpty ? "earlier step" : sources.joined(separator: " + ")
    }

    private var packedIntersectionDescription: String {
        guard let retained = session.retainedPackedIntersections,
              let capacity = session.packedIntersectionCapacity else {
            return "—"
        }
        return "\(retained.formatted()) / \(capacity.formatted()) used"
    }

    private var isBusy: Bool {
        switch session.phase {
        case .planning, .loading, .training: return true
        default: return false
        }
    }

    private var trainButtonTitle: String {
        switch session.phase {
        case .planning, .loading: "Preparing…"
        case .training: "Training…"
        default: "Train"
        }
    }

    private func exportSection(_ ply: URL) -> some View {
        Section("Result") {
            ShareLink(item: ply) {
                Label("Export \(ply.lastPathComponent)", systemImage: "square.and.arrow.up")
            }
        }
    }
}
