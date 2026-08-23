import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var session = TrainingSession()
    @State private var folder: DatasetFolder?
    @State private var picking = false
    @State private var pickError: String?

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
                        folder = picked
                        pickError = nil
                    } else {
                        pickError = "No cameras.bin or cameras.txt in that folder, or in its sparse/0."
                    }
                case .failure(let error):
                    pickError = error.localizedDescription
                }
            }
        }
    }

    private var datasetSection: some View {
        Section("COLMAP dataset") {
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

            Button(trainButtonTitle) {
                if let folder { session.start(folder: folder) }
            }
            .disabled(isBusy)

            if isBusy {
                Button("Stop", role: .destructive) { session.cancel() }
            }
        } footer: {
            Text("Preview targets a 1,600-pixel edge, SH1, and 250K Gaussians. Balanced targets 1,920 pixels, SH2, and 400K Gaussians when preflight memory permits.")
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
                    Image(uiImage: preview)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
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
                LabeledContent("GPU execute", value: duration(session.gpuExecutionMs))
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
        if session.overflowKinds.contains(.tileCapacity) { sources.append("tile cap") }
        if session.overflowKinds.contains(.packedCapacity) { sources.append("packed arena") }
        return sources.isEmpty ? "earlier step" : sources.joined(separator: " + ")
    }

    private var packedIntersectionDescription: String {
        guard let retained = session.retainedPackedIntersections,
              let capacity = session.packedIntersectionCapacity else {
            return "—"
        }
        return "\(retained.formatted()) / \(capacity.formatted()) retained"
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
