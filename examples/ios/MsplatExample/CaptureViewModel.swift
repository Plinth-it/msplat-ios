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

struct CaptureTaskGeneration {
    private var nextID: UInt64 = 0
    private(set) var currentID: UInt64?

    var isActive: Bool { currentID != nil }

    mutating func begin() -> UInt64 {
        nextID &+= 1
        currentID = nextID
        return nextID
    }

    mutating func invalidate() {
        currentID = nil
    }

    mutating func finish(_ id: UInt64) -> Bool {
        guard currentID == id else { return false }
        currentID = nil
        return true
    }
}

struct CaptureEventOrdering {
    private(set) var latestCommittedCandidateSequence: UInt64?
    private(set) var latestMessageCandidateSequence: UInt64?

    mutating func recordCommit(candidateSequence: UInt64) -> Bool {
        latestCommittedCandidateSequence = max(
            latestCommittedCandidateSequence ?? candidateSequence,
            candidateSequence
        )
        guard latestMessageCandidateSequence.map({ candidateSequence >= $0 }) ?? true else {
            return false
        }
        latestMessageCandidateSequence = candidateSequence
        return true
    }

    mutating func shouldApplyTelemetry(candidateSequence: UInt64) -> Bool {
        if let latestCommittedCandidateSequence,
           candidateSequence <= latestCommittedCandidateSequence {
            return false
        }
        guard latestMessageCandidateSequence.map({ candidateSequence > $0 }) ?? true else {
            return false
        }
        latestMessageCandidateSequence = candidateSequence
        return true
    }
}

@MainActor
final class CaptureViewModel: ObservableObject {
    @Published private(set) var state: CaptureFlowState = .preparing
    @Published private(set) var frames: [CaptureFrameRecord] = []
    @Published private(set) var frameCount = 0
    @Published private(set) var pointCount = 0
    @Published private(set) var subjectSelected = false
    @Published private(set) var statusMessage = "Starting ARKit…"
    @Published private(set) var latestRejection: String?
    @Published var mode: CaptureMode = .object {
        didSet {
            guard oldValue != mode else { return }
            cancelActionTask()
            engine.resetSubject()
            subjectSelected = false
            latestRejection = nil
            statusMessage = preflightStatusMessage
        }
    }

    let engine: CaptureEngine
    private var eventTask: Task<Void, Never>?
    private var telemetryTask: Task<Void, Never>?
    private var prepareTask: Task<Void, Never>?
    private var actionTask: Task<Void, Never>?
    private var prepareTaskGeneration = CaptureTaskGeneration()
    private var actionTaskGeneration = CaptureTaskGeneration()
    private var eventOrdering = CaptureEventOrdering()

    init(engine: CaptureEngine = CaptureEngine()) {
        self.engine = engine
        eventTask = Task { @MainActor [weak self, events = engine.controlEvents] in
            for await event in events {
                guard let self, !Task.isCancelled else { return }
                handle(event)
            }
        }
        telemetryTask = Task { @MainActor [weak self, events = engine.telemetryEvents] in
            for await event in events {
                guard let self, !Task.isCancelled else { return }
                handle(event)
            }
        }
    }

    deinit {
        eventTask?.cancel()
        telemetryTask?.cancel()
        prepareTask?.cancel()
        actionTask?.cancel()
    }

    var canStart: Bool {
        switch mode {
        case .object: subjectSelected
        case .scene: true
        }
    }

    var canStop: Bool { frameCount >= 3 && pointCount > 0 }

    func prepareIfNeeded() {
        guard !prepareTaskGeneration.isActive else { return }
        let operationID = prepareTaskGeneration.begin()
        prepareTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer { finishPrepareTask(operationID) }
            do {
                try await engine.prepare()
                try Task.checkCancellation()
                state = .preflight
                statusMessage = preflightStatusMessage
            } catch is CancellationError {
            } catch {
                guard !Task.isCancelled,
                      prepareTaskGeneration.currentID == operationID else {
                    return
                }
                state = .failed(error.localizedDescription)
            }
        }
    }

    func selectSubject(at point: CGPoint, viewportSize: CGSize) {
        guard case .preflight = state, mode == .object else { return }
        cancelActionTask()
        engine.resetSubject()
        subjectSelected = false
        statusMessage = "Finding subject…"
        let operationID = actionTaskGeneration.begin()
        actionTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer { finishActionTask(operationID) }
            do {
                try await engine.selectSubject(at: point, viewportSize: viewportSize)
                try Task.checkCancellation()
                guard case .preflight = state, mode == .object else { return }
                subjectSelected = true
                latestRejection = nil
                statusMessage = engine.supportsHighResolutionCapture
                    ? "Subject selected. Move slowly around it."
                    : "Subject selected. Camera-stream frames will be used."
            } catch is CancellationError {
            } catch {
                guard !Task.isCancelled,
                      actionTaskGeneration.currentID == operationID else {
                    return
                }
                subjectSelected = false
                statusMessage = "Tap the object to try again."
                latestRejection = error.localizedDescription
            }
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
            frameCount = 0
            pointCount = 0
            latestRejection = nil
            statusMessage = captureStatusMessage
            state = .capturing
        } catch {
            latestRejection = error.localizedDescription
        }
    }

    func stopCapture() {
        guard case .capturing = state, canStop else { return }
        cancelActionTask()
        state = .finalizing
        statusMessage = "Writing point cloud and dataset…"
        let operationID = actionTaskGeneration.begin()
        actionTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer { finishActionTask(operationID) }
            do {
                let capture = try await engine.stopAndFinalize()
                try Task.checkCancellation()
                state = .review(capture)
            } catch is CancellationError {
            } catch {
                guard !Task.isCancelled,
                      actionTaskGeneration.currentID == operationID else {
                    return
                }
                state = .failed(error.localizedDescription)
            }
        }
    }

    func stop() {
        cancelPrepareTask()
        cancelActionTask()
        engine.stop()
    }

    private func handle(_ event: CaptureEngineControlEvent) {
        switch event {
        case .frameCommitted(
            let candidateSequence,
            let commit,
            let qualityFallback
        ):
            let shouldUpdateMessage = eventOrdering.recordCommit(
                candidateSequence: candidateSequence
            )
            frames.append(commit.record)
            frameCount = commit.totalFrameCount
            pointCount = commit.totalPointCount
            if shouldUpdateMessage {
                latestRejection = qualityFallback
            }
            statusMessage = "Captured \(frameCount) frame\(frameCount == 1 ? "" : "s")."
        case .failed(let message):
            cancelPrepareTask()
            cancelActionTask()
            engine.stop()
            state = .failed(message)
        }
    }

    private func handle(_ event: CaptureEngineTelemetryEvent) {
        switch event {
        case .frameRejected(let candidateSequence, let message):
            guard eventOrdering.shouldApplyTelemetry(
                candidateSequence: candidateSequence
            ) else {
                return
            }
            latestRejection = message
        }
    }

    private var preflightStatusMessage: String {
        if mode == .object {
            return "Tap the object to select it."
        }
        if engine.supportsHighResolutionCapture {
            return "Move slowly, then start high-resolution capture."
        }
        return "High-resolution capture is unavailable. Move slowly, then start capture."
    }

    private var captureStatusMessage: String {
        if engine.supportsHighResolutionCapture {
            return "Move slowly; high-resolution frames are captured automatically."
        }
        return "Move slowly; camera-stream frames are captured automatically."
    }

    private func cancelPrepareTask() {
        prepareTask?.cancel()
        prepareTask = nil
        prepareTaskGeneration.invalidate()
    }

    private func finishPrepareTask(_ operationID: UInt64) {
        guard prepareTaskGeneration.finish(operationID) else { return }
        prepareTask = nil
    }

    private func cancelActionTask() {
        actionTask?.cancel()
        actionTask = nil
        actionTaskGeneration.invalidate()
    }

    private func finishActionTask(_ operationID: UInt64) {
        guard actionTaskGeneration.finish(operationID) else { return }
        actionTask = nil
    }
}
