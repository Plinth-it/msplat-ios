import SwiftUI

struct RealityKitAlignmentView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var session = RealityKitAlignmentSession()
    @State private var usesObjectMaskingForAlignment: Bool
    @State private var exportsTrainingMasks: Bool
    @State private var ordering = RealityKitAlignmentOrdering.sequential

    let input: RealityKitAlignmentInput
    let onUseDataset: (DatasetFolder) -> Void

    init(
        input: RealityKitAlignmentInput,
        onUseDataset: @escaping (DatasetFolder) -> Void
    ) {
        self.input = input
        self.onUseDataset = onUseDataset
        _usesObjectMaskingForAlignment = State(
            initialValue: input.suggestedObjectMasking
        )
        _exportsTrainingMasks = State(
            initialValue: input.suggestedTrainingMaskExport
        )
    }

    var body: some View {
        NavigationStack {
            Form {
                RealityKitAlignmentSourceSection(
                    name: input.name,
                    origin: input.originDescription,
                    imageCount: input.imageURLs.count
                )

                switch session.phase {
                case .idle, .cancelled, .failed:
                    RealityKitAlignmentOptionsSection(
                        usesObjectMaskingForAlignment:
                            $usesObjectMaskingForAlignment,
                        exportsTrainingMasks: $exportsTrainingMasks,
                        ordering: $ordering,
                        buttonTitle: retryButtonTitle,
                        errorMessage: failureMessage,
                        onStart: start
                    )
                case .preparing, .aligning, .exporting:
                    RealityKitAlignmentProgressSection(
                        progress: session.progress,
                        statusMessage: session.statusMessage
                    )
                case .finished(let result):
                    RealityKitAlignmentResultSection(
                        result: result,
                        onUseDataset: {
                            onUseDataset(result.folder)
                            dismiss()
                        }
                    )
                }
            }
            .navigationTitle("RealityKit alignment")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    if session.isBusy {
                        Button("Cancel", role: .cancel) {
                            session.cancel()
                        }
                    } else {
                        Button("Close") { dismiss() }
                    }
                }
            }
            .interactiveDismissDisabled(session.isBusy)
            .onDisappear {
                if session.isBusy {
                    session.cancel()
                }
            }
        }
    }

    private var retryButtonTitle: String {
        switch session.phase {
        case .failed, .cancelled: "Try alignment again"
        default: "Align and export COLMAP"
        }
    }

    private var failureMessage: String? {
        guard case .failed(let message) = session.phase else { return nil }
        return message
    }

    private func start() {
        session.start(
            input: input,
            options: RealityKitAlignmentOptions(
                usesObjectMaskingForAlignment:
                    usesObjectMaskingForAlignment,
                exportsTrainingMasks: exportsTrainingMasks,
                ordering: ordering
            )
        )
    }
}

private struct RealityKitAlignmentSourceSection: View {
    let name: String
    let origin: String
    let imageCount: Int

    var body: some View {
        Section {
            LabeledContent("Folder", value: name)
            LabeledContent("Source", value: origin)
            LabeledContent("Images", value: imageCount.formatted())
        } header: {
            Text("Source images")
        } footer: {
            Text("RealityKit registers the usable subset, estimates camera poses and a sparse colored point cloud, then the app writes a separate COLMAP dataset for training.")
        }
    }
}

private struct RealityKitAlignmentOptionsSection: View {
    @Binding var usesObjectMaskingForAlignment: Bool
    @Binding var exportsTrainingMasks: Bool
    @Binding var ordering: RealityKitAlignmentOrdering
    let buttonTitle: String
    let errorMessage: String?
    let onStart: () -> Void

    var body: some View {
        Section {
            Toggle(
                "Mask subject during alignment",
                isOn: $usesObjectMaskingForAlignment
            )
            Toggle(
                "Export Vision masks for training",
                isOn: $exportsTrainingMasks
            )
            Picker("Image order", selection: $ordering) {
                ForEach(RealityKitAlignmentOrdering.allCases) { ordering in
                    Text(ordering.rawValue).tag(ordering)
                }
            }

            if let errorMessage {
                Text(errorMessage)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }

            Button(action: onStart) {
                Label(buttonTitle, systemImage: "viewfinder")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
        } header: {
            Text("Alignment")
        } footer: {
            Text("Alignment masking and training-mask export are independent. Turn off alignment masking when full frames register better; Vision can still create masks afterward for Gaussian training.")
            Text("Sequential is best for a capture orbit. Use Unordered when filenames do not preserve capture order.")
            Text("Runs only on a physical device. Imported photos require iOS 26+ for calibrated pose intrinsics; ARKit captures include their own intrinsics and support iOS 18+.")
        }
    }
}

private struct RealityKitAlignmentProgressSection: View {
    let progress: Double
    let statusMessage: String

    var body: some View {
        Section {
            HStack {
                ProgressView(value: progress)
                    .accessibilityLabel("Alignment and export progress")
                Text(
                    progress,
                    format: .percent.precision(.fractionLength(0))
                )
                .font(.footnote.monospacedDigit())
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            }
            Text(statusMessage)
                .foregroundStyle(.secondary)
        } header: {
            Text("Alignment and export progress")
        } footer: {
            Text("Keep the app in the foreground. RealityKit alignment and msplat training do not run at the same time.")
        }
    }
}

private struct RealityKitAlignmentResultSection: View {
    let result: RealityKitAlignmentResult
    let onUseDataset: () -> Void

    var body: some View {
        Section {
            LabeledContent(
                "Registered images",
                value: result.export.alignedImageCount.formatted()
            )
            LabeledContent(
                "Cameras",
                value: result.export.cameraCount.formatted()
            )
            LabeledContent(
                "Sparse points",
                value: result.export.pointCount.formatted()
            )
            LabeledContent(
                "Training masks",
                value: result.export.maskCount.formatted()
            )

            if result.invalidSampleCount > 0 {
                Text("RealityKit rejected \(result.invalidSampleCount) input image(s).")
                    .foregroundStyle(.orange)
            }
            if result.skippedSampleCount > 0 {
                Text("RealityKit skipped \(result.skippedSampleCount) input image(s).")
                    .foregroundStyle(.orange)
            }
            if result.wasDownsampled {
                Text("RealityKit downsampled inputs because of device limits.")
                    .foregroundStyle(.orange)
            }
            if result.stitchingWasIncomplete {
                Text("RealityKit reported incomplete stitching; inspect the trained result before relying on it.")
                    .foregroundStyle(.orange)
            }

            Button(action: onUseDataset) {
                Label("Use for training", systemImage: "sparkles")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)

            ShareLink(item: result.export.datasetDirectory) {
                Label("Share COLMAP folder", systemImage: "square.and.arrow.up")
            }
        } header: {
            Text("RealityKit training seed")
        } footer: {
            Text("The original images and ARKit dataset remain unchanged. The aligned output is stored under Documents/RealityKitAlignments.")
            Text("This is a camera-and-point seed for msplat training. COLMAP image observations and point tracks are intentionally empty; it is not a feature-matched or bundle-adjusted SfM reconstruction.")
        }
    }
}
