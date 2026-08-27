import Msplat
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    private enum Route: Identifiable {
        case capture
        case alignment(RealityKitAlignmentInput)

        var id: String {
            switch self {
            case .capture: "capture"
            case .alignment(let input): "alignment-\(input.id.uuidString)"
            }
        }
    }

    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var session = TrainingSession()
    @State private var source: TrainingDatasetSource?
    @State private var picking = false
    @State private var route: Route?
    @State private var pickError: String?
    @State private var trainingMaskCandidateCount: Int?
    @State private var trainingMaskSelectionWasEdited = false
    @State private var didAttemptDatasetRestore = false

    var body: some View {
        NavigationStack {
            Form {
                datasetSection
                if source != nil { settingsSection }
                if session.phase != .idle { progressSection }
                if let ply = session.exportedPly { exportSection(ply) }
            }
            .navigationTitle("msplat")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        route = .capture
                    } label: {
                        Label("Capture", systemImage: "camera.viewfinder")
                    }
                    .disabled(isBusy)
                }
            }
            .fullScreenCover(item: $route) { route in
                switch route {
                case .capture:
                    CaptureRootView { capture in
                        select(capture)
                    }
                case .alignment(let input):
                    RealityKitAlignmentView(input: input) { alignedFolder in
                        select(alignedFolder, persistBookmark: true)
                    }
                }
            }
            .fileImporter(isPresented: $picking, allowedContentTypes: [.folder]) { result in
                switch result {
                case .success(let url):
                    if let picked = DatasetFolder(picked: url) {
                        select(picked, persistBookmark: true)
                    } else if let input = RealityKitAlignmentInput(picked: url) {
                        pickError = nil
                        route = .alignment(input)
                    } else {
                        pickError = "Choose a COLMAP or Nerfstudio dataset, or a folder containing HEIC, JPEG, or PNG images for RealityKit alignment."
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

    private var folder: DatasetFolder? { source?.importedFolder }

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
        guard source == nil else { return }
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
        source = .importedFolder(selectedFolder)
        trainingMaskCandidateCount = nil
        trainingMaskSelectionWasEdited = false
        session.trainingMasksEnabled = false
        session.trainingMaskMode = .transparent
        session.refineCameraPosesEnabled = false
        session.cameraPoseConditioning = .raw
        pickError = nil

        guard persistBookmark else { return }
        do {
            try selectedFolder.persistAsLastPicked()
        } catch {
            pickError = "The folder is selected, but it could not be remembered: " +
                error.localizedDescription
        }
    }

    @MainActor
    private func select(_ capture: CapturedDataset) {
        source = .captured(capture)
        trainingMaskCandidateCount = capture.descriptor.frames.lazy
            .filter { $0.trainingMask != nil }
            .count
        trainingMaskSelectionWasEdited = true
        session.trainingMasksEnabled = capture.descriptor.frames.contains {
            $0.trainingMask != nil
        }
        session.trainingMaskMode = capture.manifest.mode == .object
            ? .transparent : .coverage
        session.refineCameraPosesEnabled = false
        session.cameraPoseConditioning = .raw
        pickError = nil
    }

    private var datasetSection: some View {
        Section("Training dataset") {
            Button {
                picking = true
            } label: {
                Label(source?.name ?? "Choose a folder…", systemImage: "folder")
            }
            .disabled(isBusy)

            if let source {
                LabeledContent("Contents", value: source.summary)
            }
            if let capture = source?.capturedDataset {
                Button {
                    if let input = RealityKitAlignmentInput(capture: capture) {
                        pickError = nil
                        route = .alignment(input)
                    } else {
                        pickError = "The capture no longer contains readable source images."
                    }
                } label: {
                    Label(
                        "Realign capture with RealityKit",
                        systemImage: "viewfinder"
                    )
                }
                .disabled(isBusy)
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
            if let capture = source?.capturedDataset {
                LabeledContent(
                    "Capture masks",
                    value: capture.descriptor.frames.contains {
                        $0.trainingMask != nil
                    } ? "Included" : "None"
                )
                Toggle(
                    "Refine camera poses",
                    isOn: $session.refineCameraPosesEnabled
                )
                .disabled(isBusy)
                if session.refineCameraPosesEnabled {
                    Picker(
                        "Pose optimizer",
                        selection: $session.cameraPoseConditioning
                    ) {
                        Text("Bounded SE(3)").tag(CameraPoseConditioning.raw)
                        Text("CamP-conditioned").tag(CameraPoseConditioning.camP)
                    }
                    .disabled(isBusy)
                }
            } else if folder?.supportsAutomaticTrainingMaskDiscovery == true {
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
                if folder?.hasNerfstudioTrainingMasks == true {
                    LabeledContent("Manifest masks", value: "Included")
                    Picker("Mask treatment", selection: $session.trainingMaskMode) {
                        Text("Transparent").tag(TrainingMaskMode.transparent)
                        Text("Coverage only").tag(TrainingMaskMode.coverage)
                    }
                    .disabled(isBusy)
                } else {
                    Text("Add mask_path to every frame in transforms.json to import Nerfstudio masks.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }

            Button(trainButtonTitle) {
                if let source { session.start(source: source) }
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
            if source?.capturedDataset != nil {
                Text("Pose refinement is an explicit A/B control for captured datasets. Bounded SE(3) preserves the baseline optimizer; CamP conditions the same bounded updates with a fixed per-camera projection metric. Neither mode changes the captured transforms.json.")
                if let requirement = poseRefinementBudgetRequirement {
                    Text(
                        "It requires at least \(requirement.minimumIterations) iterations: " +
                        "\(requirement.warmupIterations) warm-up iterations plus " +
                        "\(requirement.postWarmupCameraVisits) camera visits to complete " +
                        "one full post-warm-up shuffled pass."
                    )
                    if session.refineCameraPosesEnabled,
                       session.iterations < requirement.minimumIterations {
                        Text(
                            "Increase Iterations to at least " +
                            "\(requirement.minimumIterations) before training."
                        )
                        .foregroundStyle(.orange)
                    }
                }
            }
        }
    }

    private var poseRefinementBudgetRequirement: PoseRefinementBudgetRequirement? {
        guard let capture = source?.capturedDataset,
              var config = try? TrainingSession.makeTrainingConfig(
                  trainingMaskMode: capture.manifest.mode == .object
                      ? .transparent : session.trainingMaskMode,
                  keepCrs: true,
                  refineCameraPoses: true,
                  cameraPoseConditioning: session.cameraPoseConditioning,
                  benchmark: nil
              ),
              let iterations = Int32(exactly: session.iterations) else {
            return nil
        }
        config.iterations = iterations
        return try? TrainingSession.poseRefinementBudgetRequirement(
            config: config,
            trainingCameraCount: capture.descriptor.frames.count
        )
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
                    LabeledContent(
                        "Target prefetch waits",
                        value: memory.trainingTargetPrefetchWaitRate.map {
                            String(
                                format: "%.0f%% (%llu/%llu)",
                                $0 * 100,
                                memory.trainingTargetPrefetchWaited,
                                memory.trainingTargetPrefetchUsed
                            )
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
                if session.phase == .finished,
                   let summary = session.poseCorrectionSummary {
                    LabeledContent(
                        "Pose translation",
                        value: summary.translationDescription
                    )
                    LabeledContent(
                        "Pose rotation",
                        value: summary.rotationDescription
                    )
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
            if let refinedTransformsURL = session.refinedTransformsURL {
                ShareLink(item: refinedTransformsURL) {
                    Label(
                        "Export \(refinedTransformsURL.lastPathComponent)",
                        systemImage: "square.and.arrow.up"
                    )
                }
            }
        }
    }
}
