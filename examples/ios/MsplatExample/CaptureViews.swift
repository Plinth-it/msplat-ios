@preconcurrency import ARKit
import SceneKit
import SwiftUI
import UIKit

struct CaptureRootView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = CaptureViewModel()
    let onUseDataset: (CapturedDataset) -> Void

    var body: some View {
        Group {
            switch model.state {
            case .preparing, .preflight, .capturing, .finalizing:
                cameraFlowView
            case .review(let capture):
                CaptureReviewView(
                    capture: capture,
                    onUseDataset: {
                        onUseDataset(capture)
                        dismiss()
                    },
                    onClose: { dismiss() }
                )
            case .failed(let message):
                ContentUnavailableView(
                    "Capture unavailable",
                    systemImage: "camera.fill.badge.exclamationmark",
                    description: Text(message)
                )
                .safeAreaInset(edge: .bottom) {
                    Button("Close") { dismiss() }
                        .buttonStyle(.borderedProminent)
                        .padding()
                }
            }
        }
        .background(.black)
        .task { model.prepareIfNeeded() }
        .onDisappear { model.stop() }
    }

    private var cameraFlowView: some View {
        ZStack {
            CaptureCameraPreview(
                session: model.engine.session,
                onInterfaceOrientationChange: model.updateInterfaceOrientation
            )
                .ignoresSafeArea()

            switch model.state {
            case .preparing:
                progressOverlay("Preparing camera…")
            case .preflight:
                cameraOverlay(isCapturing: false)
            case .capturing:
                cameraOverlay(isCapturing: true)
            case .finalizing:
                progressOverlay("Finalizing capture…")
            case .review, .failed:
                EmptyView()
            }
        }
    }

    private func progressOverlay(_ title: String) -> some View {
        ZStack(alignment: .topTrailing) {
            Color.black.opacity(0.35)
                .ignoresSafeArea()
            ProgressView(title)
                .tint(.white)
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            closeButton
                .padding()
        }
    }

    private func cameraOverlay(isCapturing: Bool) -> some View {
        ZStack {
            if !isCapturing, model.mode == .object {
                GeometryReader { geometry in
                    Color.clear
                        .contentShape(Rectangle())
                        .onTapGesture { location in
                            model.selectSubject(
                                at: location,
                                viewportSize: geometry.size
                            )
                        }
                }
            }

            VStack(spacing: 12) {
                HStack {
                    closeButton
                    Spacer()
                    if isCapturing {
                        Label(
                            "\(model.frames.count)",
                            systemImage: "circle.fill"
                        )
                        .font(.headline.monospacedDigit())
                        .foregroundStyle(.white, .red)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(.black.opacity(0.65), in: Capsule())
                    }
                }
                Spacer()
                controls(isCapturing: isCapturing)
            }
            .padding()
        }
    }

    private var closeButton: some View {
        Button {
            dismiss()
        } label: {
            Image(systemName: "xmark")
                .font(.headline)
                .frame(width: 44, height: 44)
                .background(.black.opacity(0.65), in: Circle())
                .foregroundStyle(.white)
        }
        .accessibilityLabel("Close capture")
    }

    private func controls(isCapturing: Bool) -> some View {
        VStack(spacing: 12) {
            if !isCapturing {
                Picker("Capture type", selection: $model.mode) {
                    ForEach(CaptureMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
            }

            VStack(spacing: 6) {
                Text(model.statusMessage)
                    .font(.headline)
                    .multilineTextAlignment(.center)
                if isCapturing {
                    Text("\(model.pointCount.formatted()) fused points")
                        .font(.subheadline.monospacedDigit())
                } else if model.mode == .object {
                    Label(
                        model.subjectSelected ? "Subject locked" : "Subject required",
                        systemImage: model.subjectSelected ? "scope" : "hand.tap"
                    )
                    .font(.subheadline)
                }
                if let rejection = model.latestRejection {
                    Text(rejection)
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .lineLimit(3)
                }
            }
            .foregroundStyle(.white)

            if isCapturing {
                Button {
                    model.stopCapture()
                } label: {
                    Label("Stop and review", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .disabled(!model.canStop)
            } else {
                Button {
                    model.startCapture()
                } label: {
                    Label("Start automatic capture", systemImage: "record.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!model.canStart)
            }
        }
        .padding()
        .background(.black.opacity(0.72), in: RoundedRectangle(cornerRadius: 18))
    }
}

private struct CaptureCameraPreview: UIViewRepresentable {
    let session: ARSession
    let onInterfaceOrientationChange: (UIInterfaceOrientation) -> Void

    func makeUIView(context: Context) -> ARSCNView {
        let view = CaptureARSCNView(frame: .zero)
        view.session = session
        view.scene = SCNScene()
        view.automaticallyUpdatesLighting = true
        view.onInterfaceOrientationChange = onInterfaceOrientationChange
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        if uiView.session !== session {
            uiView.session = session
        }
        if let uiView = uiView as? CaptureARSCNView {
            uiView.onInterfaceOrientationChange = onInterfaceOrientationChange
        }
    }
}

private final class CaptureARSCNView: ARSCNView {
    var onInterfaceOrientationChange: ((UIInterfaceOrientation) -> Void)?
    private var lastInterfaceOrientation: UIInterfaceOrientation = .unknown

    override func didMoveToWindow() {
        super.didMoveToWindow()
        publishInterfaceOrientationIfNeeded()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        publishInterfaceOrientationIfNeeded()
    }

    private func publishInterfaceOrientationIfNeeded() {
        guard let interfaceOrientation = window?.windowScene?.interfaceOrientation,
              interfaceOrientation != .unknown,
              interfaceOrientation != lastInterfaceOrientation else {
            return
        }
        lastInterfaceOrientation = interfaceOrientation
        onInterfaceOrientationChange?(interfaceOrientation)
    }
}

private struct CaptureReviewView: View {
    let capture: CapturedDataset
    let onUseDataset: () -> Void
    let onClose: () -> Void

    var body: some View {
        NavigationStack {
            List {
                Section("Captured frames") {
                    ScrollView(.horizontal) {
                        LazyHStack(spacing: 10) {
                            ForEach(capture.manifest.frames) { frame in
                                CaptureThumbnail(
                                    imageURL: frame.imageURL(under: capture.rootURL),
                                    label: frame.id
                                )
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    .scrollIndicators(.hidden)
                    .listRowInsets(EdgeInsets())
                }

                Section("Geometry") {
                    LabeledContent(
                        "Frames",
                        value: capture.reviewSnapshot.frameCount.formatted()
                    )
                    LabeledContent(
                        "Initial points",
                        value: capture.reviewSnapshot.pointCount.formatted()
                    )
                    LabeledContent(
                        "Principal point cx",
                        value: range(
                            capture.reviewSnapshot.minimumCX,
                            capture.reviewSnapshot.maximumCX
                        )
                    )
                    LabeledContent(
                        "Principal point cy",
                        value: range(
                            capture.reviewSnapshot.minimumCY,
                            capture.reviewSnapshot.maximumCY
                        )
                    )
                }

                Section {
                    Button(action: onUseDataset) {
                        Label("Use ARKit dataset", systemImage: "sparkles")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)

                    ShareLink(item: capture.rootURL) {
                        Label("Share Nerfstudio package", systemImage: "square.and.arrow.up")
                    }
                } footer: {
                    Text("RGB, per-frame intrinsics and poses, masks, the colored point seed, transforms.json, and the recovery journal are stored together.")
                    Text("After using the dataset, the training screen can optionally use RealityKit to estimate new camera poses and a sparse point cloud from its images before training.")
                }
            }
            .navigationTitle("Review capture")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close", action: onClose)
                }
            }
        }
    }

    private func range(_ minimum: Float, _ maximum: Float) -> String {
        String(format: "%.2f – %.2f", minimum, maximum)
    }
}

private struct CaptureThumbnail: View {
    let imageURL: URL
    let label: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let image = displayImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 132, height: 98)
                    .clipped()
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                ContentUnavailableView("Unavailable", systemImage: "photo")
                    .frame(width: 132, height: 98)
            }
            Text(label)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
        }
    }

    private var displayImage: UIImage? {
        UIImage(contentsOfFile: imageURL.path)
    }
}
