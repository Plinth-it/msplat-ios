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
            .disabled(session.phase == .training || session.phase == .loading)

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
            Picker("Resolution", selection: $session.downscale) {
                Text("Full").tag(Float(1))
                Text("Half").tag(Float(2))
                Text("Quarter").tag(Float(4))
            }

            Button(session.phase == .training ? "Training…" : "Train") {
                if let folder { session.start(folder: folder) }
            }
            .disabled(session.phase == .training || session.phase == .loading)

            if session.phase == .training {
                Button("Stop", role: .destructive) { session.cancel() }
            }
        } footer: {
            Text("Half resolution is the default: full-resolution captures are where a phone runs out of memory first.")
        }
    }

    private var progressSection: some View {
        Section("Progress") {
            switch session.phase {
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
                ProgressView(value: Double(session.iteration),
                             total: Double(max(session.iterations, 1)))
                LabeledContent("Step", value: "\(session.iteration) / \(session.iterations)")
                LabeledContent("Gaussians", value: session.splatCount.formatted())
                LabeledContent("CPU submit", value: String(format: "%.1f ms", session.msPerStep))
                LabeledContent("Cameras", value: "\(session.trainingCameras)")
                // os_proc_available_memory reports 0 in the simulator, which
                // has no jetsam limit — showing "0 MB left" there would read
                // as being out of memory rather than as not applicable.
                LabeledContent("Memory", value: session.availableMB > 0
                    ? "\(session.footprintMB) MB used, \(session.availableMB) MB free"
                    : "\(session.footprintMB) MB used")
            }
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
