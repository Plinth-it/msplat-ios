@preconcurrency import ARKit
import Foundation
import SwiftUI
import UIKit

enum CaptureFlowState {
    case preparing
    case preflight
    case capturing
    case finalizing
    case review(CapturedDataset)
    case failed(String)
}

@MainActor
final class CaptureViewModel: ObservableObject {
    @Published private(set) var state: CaptureFlowState = .preparing
    @Published private(set) var frames: [CaptureFrameRecord] = []
    @Published private(set) var pointCount = 0
    @Published private(set) var subjectSelected = false
    @Published private(set) var statusMessage = "Starting ARKit…"
    @Published private(set) var latestRejection: String?
    @Published var mode: CaptureMode = .object {
        didSet {
            guard oldValue != mode else { return }
            actionTask?.cancel()
            actionTask = nil
            engine.resetSubject()
            subjectSelected = false
            latestRejection = nil
            statusMessage = mode == .object
                ? "Tap the object to select it."
                : "Move slowly, then start high-resolution capture."
        }
    }

    let engine: CaptureEngine
    private var eventTask: Task<Void, Never>?
    private var prepareTask: Task<Void, Never>?
    private var actionTask: Task<Void, Never>?

    init(engine: CaptureEngine = CaptureEngine()) {
        self.engine = engine
        eventTask = Task { @MainActor [weak self, events = engine.events] in
            for await event in events {
                guard let self, !Task.isCancelled else { return }
                handle(event)
            }
        }
    }

    deinit {
        eventTask?.cancel()
        prepareTask?.cancel()
        actionTask?.cancel()
    }

    var canStart: Bool {
        switch mode {
        case .object: subjectSelected
        case .scene: true
        }
    }

    var canStop: Bool { frames.count >= 3 && pointCount > 0 }

    func prepareIfNeeded() {
        guard prepareTask == nil else { return }
        prepareTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                try await engine.prepare()
                try Task.checkCancellation()
                state = .preflight
                statusMessage = mode == .object
                    ? "Tap the object to select it."
                    : "Move slowly, then start high-resolution capture."
            } catch is CancellationError {
                return
            } catch {
                state = .failed(error.localizedDescription)
            }
            prepareTask = nil
        }
    }

    func selectSubject(at point: CGPoint, viewportSize: CGSize) {
        guard case .preflight = state, mode == .object else { return }
        actionTask?.cancel()
        engine.resetSubject()
        subjectSelected = false
        statusMessage = "Finding subject…"
        actionTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                try await engine.selectSubject(at: point, viewportSize: viewportSize)
                try Task.checkCancellation()
                guard case .preflight = state, mode == .object else { return }
                subjectSelected = true
                latestRejection = nil
                statusMessage = "Subject selected. Move slowly around it."
            } catch is CancellationError {
                return
            } catch {
                subjectSelected = false
                statusMessage = "Tap the object to try again."
                latestRejection = error.localizedDescription
            }
            actionTask = nil
        }
    }

    func updateInterfaceOrientation(_ interfaceOrientation: UIInterfaceOrientation) {
        engine.updateInterfaceOrientation(interfaceOrientation)
    }

    func startCapture() {
        guard case .preflight = state, canStart else { return }
        do {
            try engine.startRecording(mode: mode)
            frames = []
            pointCount = 0
            latestRejection = nil
            statusMessage = "Move slowly; high-resolution frames are captured " +
                "automatically."
            state = .capturing
        } catch {
            latestRejection = error.localizedDescription
        }
    }

    func stopCapture() {
        guard case .capturing = state, canStop else { return }
        state = .finalizing
        statusMessage = "Writing point cloud and dataset…"
        actionTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let capture = try await engine.stopAndFinalize()
                try Task.checkCancellation()
                state = .review(capture)
            } catch is CancellationError {
                return
            } catch {
                state = .failed(error.localizedDescription)
            }
            actionTask = nil
        }
    }

    func stop() {
        prepareTask?.cancel()
        actionTask?.cancel()
        engine.stop()
    }

    private func handle(_ event: CaptureEngineEvent) {
        switch event {
        case .frameCommitted(let commit):
            frames.append(commit.record)
            pointCount = commit.totalPointCount
            latestRejection = nil
            statusMessage = "Captured \(frames.count) frame\(frames.count == 1 ? "" : "s")."
        case .frameQualityFallback(let message):
            latestRejection = message
        case .frameRejected(let message):
            latestRejection = message
        case .failed(let message):
            engine.stop()
            state = .failed(message)
        }
    }
}
