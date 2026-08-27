@preconcurrency import ARKit
import AVFoundation
import CoreVideo
import Foundation
import OSLog
import Synchronization
import UIKit

enum CaptureEngineControlEvent: Sendable {
    case frameCommitted(
        candidateSequence: UInt64,
        commit: CaptureCommit,
        qualityFallback: String?
    )
    case failed(String)
}

enum CaptureEngineTelemetryEvent: Sendable {
    case frameRejected(candidateSequence: UInt64, message: String)
}

/// Retains one synchronized streaming observation without retaining `ARFrame`.
/// Admission bounds this to one snapshot through high-resolution commit/fallback.
struct CaptureFrameSnapshot: @unchecked Sendable {
    let image: CVPixelBuffer
    let depth: CVPixelBuffer?
    let confidence: CVPixelBuffer?
    let displayOrientation: CaptureDisplayOrientation
    let calibration: CaptureCalibrationRecord
    let cameraToWorld: simd_float4x4
    let timestamp: TimeInterval
    let exposureDuration: TimeInterval
    let trackingState: String
    let rawFeaturePoints: [SIMD3<Float>]
    let subjectWorldPosition: SIMD3<Float>?

    func materialize() throws -> CaptureFrameCandidate {
        CaptureFrameCandidate(
            image: try copyPixelBuffer(image),
            depth: try depth.map(copyPixelBuffer),
            confidence: try confidence.map(copyPixelBuffer),
            displayOrientation: displayOrientation,
            calibration: calibration,
            cameraToWorld: cameraToWorld,
            timestamp: timestamp,
            exposureDuration: exposureDuration,
            trackingState: trackingState,
            rawFeaturePoints: rawFeaturePoints,
            subjectWorldPosition: subjectWorldPosition
        )
    }
}

struct CaptureFrameAdmission: Sendable {
    fileprivate let generation: UInt64
    let candidateSequence: UInt64
    fileprivate let store: CaptureStore
    let requestsHighResolutionFrame: Bool
    let displayOrientation: CaptureDisplayOrientation
    fileprivate let cameraToWorld: simd_float4x4
    fileprivate let timestamp: TimeInterval
    fileprivate let subjectWorldPosition: SIMD3<Float>?
}

/// Mirrors ARKit's one-high-resolution-request-at-a-time contract. Unlike a
/// recording generation, this reservation survives cancellation until ARKit's
/// asynchronous request actually returns.
final class CaptureHighResolutionRequestGate: Sendable {
    private let isReserved = Mutex(false)

    func reserve() -> Bool {
        isReserved.withLock { isReserved in
            guard !isReserved else { return false }
            isReserved = true
            return true
        }
    }

    func release() {
        isReserved.withLock { isReserved in
            isReserved = false
        }
    }
}

/// Keeps frame admission and persistence bounded while only owned snapshots
/// cross the asynchronous storage boundary.
final class CaptureFrameAdmissionController: Sendable {
    private struct State {
        var generation: UInt64 = 0
        var nextCandidateSequence: UInt64 = 0
        var activeCandidateSequence: UInt64?
        var isRecording = false
        var store: CaptureStore?
        var usesHighResolutionCapture = false
        var subjectWorldPosition: SIMD3<Float>?
        var displayOrientation: CaptureDisplayOrientation?
        var lastAcceptedTransform: simd_float4x4?
        var lastAcceptedTimestamp: TimeInterval?
        var processingTask: Task<Void, Never>?
    }

    private let state = Mutex(State())

    func start(
        store: CaptureStore,
        subjectWorldPosition: SIMD3<Float>?,
        usesHighResolutionCapture: Bool = true
    ) {
        let previousTask = state.withLock { state in
            let previousTask = state.processingTask
            state.generation &+= 1
            state.activeCandidateSequence = nil
            state.isRecording = true
            state.store = store
            state.usesHighResolutionCapture = usesHighResolutionCapture
            state.subjectWorldPosition = subjectWorldPosition
            state.lastAcceptedTransform = nil
            state.lastAcceptedTimestamp = nil
            state.processingTask = nil
            return previousTask
        }
        previousTask?.cancel()
    }

    func updateDisplayOrientation(_ displayOrientation: CaptureDisplayOrientation) {
        state.withLock { state in
            state.displayOrientation = displayOrientation
        }
    }

    func stop() {
        state.withLock { state in
            state.isRecording = false
        }
    }

    func begin(
        cameraToWorld: simd_float4x4,
        timestamp: TimeInterval
    ) -> CaptureFrameAdmission? {
        state.withLock { state in
            guard state.isRecording,
                  state.activeCandidateSequence == nil,
                  let store = state.store,
                  let displayOrientation = state.displayOrientation else {
                return nil
            }
            if let lastAcceptedTransform = state.lastAcceptedTransform,
               let lastAcceptedTimestamp = state.lastAcceptedTimestamp {
                guard timestamp - lastAcceptedTimestamp >= 0.35 else {
                    return nil
                }
                let translation = CaptureGeometry.translationDistance(
                    from: lastAcceptedTransform,
                    to: cameraToWorld
                )
                let rotation = CaptureGeometry.rotationAngle(
                    from: lastAcceptedTransform,
                    to: cameraToWorld
                )
                guard translation >= 0.03 || rotation >= 5 * .pi / 180 else {
                    return nil
                }
            }

            state.nextCandidateSequence &+= 1
            let candidateSequence = state.nextCandidateSequence
            state.activeCandidateSequence = candidateSequence
            return CaptureFrameAdmission(
                generation: state.generation,
                candidateSequence: candidateSequence,
                store: store,
                requestsHighResolutionFrame: state.usesHighResolutionCapture,
                displayOrientation: displayOrientation,
                cameraToWorld: cameraToWorld,
                timestamp: timestamp,
                subjectWorldPosition: state.subjectWorldPosition
            )
        }
    }

    func register(
        task: Task<Void, Never>,
        for admission: CaptureFrameAdmission
    ) {
        let candidateIsActive = state.withLock { state in
            guard state.generation == admission.generation,
                  state.activeCandidateSequence == admission.candidateSequence else {
                return false
            }
            state.processingTask = task
            return true
        }
        if !candidateIsActive {
            task.cancel()
        }
    }

    @discardableResult
    func complete(
        _ admission: CaptureFrameAdmission,
        committed: Bool,
        acceptedCameraToWorld: simd_float4x4? = nil,
        acceptedTimestamp: TimeInterval? = nil
    ) -> Bool {
        state.withLock { state in
            guard state.generation == admission.generation,
                  state.activeCandidateSequence == admission.candidateSequence else {
                return false
            }
            if committed {
                state.lastAcceptedTransform = acceptedCameraToWorld
                    ?? admission.cameraToWorld
                state.lastAcceptedTimestamp = acceptedTimestamp
                    ?? admission.timestamp
            }
            state.activeCandidateSequence = nil
            state.processingTask = nil
            return true
        }
    }

    var processingTask: Task<Void, Never>? {
        state.withLock { state in state.processingTask }
    }

    @discardableResult
    func abort() -> Task<Void, Never>? {
        let task = state.withLock { state in
            let task = state.processingTask
            state.generation &+= 1
            state.activeCandidateSequence = nil
            state.isRecording = false
            state.store = nil
            state.usesHighResolutionCapture = false
            state.subjectWorldPosition = nil
            state.processingTask = nil
            return task
        }
        task?.cancel()
        return task
    }

    func finish() {
        abort()
    }
}

private final class CaptureFrameIngestor: NSObject,
    ARSessionDelegate,
    @unchecked Sendable {
    let queue = DispatchQueue(
        label: "io.github.msplat.capture.frames",
        qos: .userInitiated
    )

    private let controlContinuation:
        AsyncStream<CaptureEngineControlEvent>.Continuation
    private let telemetryContinuation:
        AsyncStream<CaptureEngineTelemetryEvent>.Continuation
    private let admission = CaptureFrameAdmissionController()
    private let highResolutionRequestGate = CaptureHighResolutionRequestGate()

    init(
        controlContinuation: AsyncStream<CaptureEngineControlEvent>.Continuation,
        telemetryContinuation: AsyncStream<CaptureEngineTelemetryEvent>.Continuation
    ) {
        self.controlContinuation = controlContinuation
        self.telemetryContinuation = telemetryContinuation
        super.init()
    }

    func start(
        store: CaptureStore,
        subjectWorldPosition: SIMD3<Float>?,
        usesHighResolutionCapture: Bool
    ) {
        admission.start(
            store: store,
            subjectWorldPosition: subjectWorldPosition,
            usesHighResolutionCapture: usesHighResolutionCapture
        )
    }

    func stop() {
        admission.stop()
    }

    @discardableResult
    func abort() -> Task<Void, Never>? {
        admission.abort()
    }

    func updateDisplayOrientation(_ displayOrientation: CaptureDisplayOrientation) {
        admission.updateDisplayOrientation(displayOrientation)
    }

    func finish() {
        admission.finish()
    }

    var processingTask: Task<Void, Never>? {
        admission.processingTask
    }

    func drain() async {
        await withCheckedContinuation { continuation in
            queue.async {
                continuation.resume()
            }
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard case .normal = frame.camera.trackingState else { return }
        guard let ticket = admission.begin(
            cameraToWorld: frame.camera.transform,
            timestamp: frame.timestamp
        ) else {
            return
        }
        if ticket.requestsHighResolutionFrame {
            guard highResolutionRequestGate.reserve() else {
                admission.complete(ticket, committed: false)
                return
            }
        }

        let streamingSnapshot: CaptureFrameSnapshot
        do {
            streamingSnapshot = try makeCaptureFrameSnapshot(
                from: frame,
                displayOrientation: ticket.displayOrientation,
                subjectWorldPosition: ticket.subjectWorldPosition
            )
        } catch {
            if ticket.requestsHighResolutionFrame {
                highResolutionRequestGate.release()
            }
            if admission.complete(ticket, committed: false) {
                telemetryContinuation.yield(.frameRejected(
                    candidateSequence: ticket.candidateSequence,
                    message: error.localizedDescription
                ))
            }
            return
        }

        let admission = admission
        let controlContinuation = controlContinuation
        let telemetryContinuation = telemetryContinuation
        let highResolutionRequestGate = highResolutionRequestGate
        let processingTask = Task(priority: .userInitiated) {
            do {
                var candidate: CaptureFrameCandidate
                var commitFallback: CaptureFrameSnapshot?
                var fallbackMessage: String?
                if ticket.requestsHighResolutionFrame {
                    do {
                        let highResolutionCandidate =
                            try await makeHighResolutionCaptureFrameCandidate(
                                session: session,
                                displayOrientation: ticket.displayOrientation,
                                subjectWorldPosition: ticket.subjectWorldPosition,
                                requestGate: highResolutionRequestGate
                        )
                        if captureCandidateHasUsableGeometry(highResolutionCandidate) {
                            candidate = highResolutionCandidate
                            commitFallback = streamingSnapshot
                        } else {
                            candidate = try streamingSnapshot.materialize()
                            fallbackMessage = "The high-resolution frame lacked " +
                                "usable geometry; captured a camera-stream frame instead."
                        }
                    } catch is CancellationError {
                        throw CancellationError()
                    } catch {
                        candidate = try streamingSnapshot.materialize()
                        fallbackMessage = "High-resolution capture failed; captured " +
                            "a camera-stream frame instead."
                    }
                } else {
                    candidate = try streamingSnapshot.materialize()
                }
                try Task.checkCancellation()
                let acceptance = try await acceptCaptureCandidate(
                    candidate,
                    streamingFallback: commitFallback,
                    in: ticket.store
                )
                candidate = acceptance.candidate
                if let commitFallbackMessage = acceptance.fallbackMessage {
                    fallbackMessage = commitFallbackMessage
                }
                try Task.checkCancellation()
                if admission.complete(
                    ticket,
                    committed: true,
                    acceptedCameraToWorld: candidate.cameraToWorld,
                    acceptedTimestamp: candidate.timestamp
                ) {
                    controlContinuation.yield(.frameCommitted(
                        candidateSequence: ticket.candidateSequence,
                        commit: acceptance.commit,
                        qualityFallback: fallbackMessage
                    ))
                }
            } catch is CancellationError {
                admission.complete(ticket, committed: false)
            } catch {
                if admission.complete(ticket, committed: false) {
                    telemetryContinuation.yield(.frameRejected(
                        candidateSequence: ticket.candidateSequence,
                        message: error.localizedDescription
                    ))
                }
            }
        }
        admission.register(task: processingTask, for: ticket)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        abort()
        controlContinuation.yield(.failed(error.localizedDescription))
    }
}

@MainActor
final class CaptureEngine: NSObject {
    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "io.github.msplat",
        category: "Capture"
    )

    let session = ARSession()
    let controlEvents: AsyncStream<CaptureEngineControlEvent>
    let telemetryEvents: AsyncStream<CaptureEngineTelemetryEvent>

    private let controlContinuation:
        AsyncStream<CaptureEngineControlEvent>.Continuation
    private let telemetryContinuation:
        AsyncStream<CaptureEngineTelemetryEvent>.Continuation
    private let frameIngestor: CaptureFrameIngestor
    private let subjectSelector = CaptureSubjectSelector()
    private let captureBaseDirectory: URL
    private var store: CaptureStore?
    private var cleanupTask: Task<Void, Never>?
    private var interfaceOrientation: UIInterfaceOrientation?
    private var displayOrientation: CaptureDisplayOrientation?
    private(set) var subjectWorldPosition: SIMD3<Float>?
    private(set) var supportsSceneDepth = false
    private(set) var supportsHighResolutionCapture = false
    private(set) var isRecording = false

    init(captureBaseDirectory: URL = .documentsDirectory) {
        let controlStream = AsyncStream.makeStream(
            of: CaptureEngineControlEvent.self,
            bufferingPolicy: .unbounded
        )
        let telemetryStream = AsyncStream.makeStream(
            of: CaptureEngineTelemetryEvent.self,
            bufferingPolicy: .bufferingNewest(32)
        )
        controlEvents = controlStream.stream
        telemetryEvents = telemetryStream.stream
        controlContinuation = controlStream.continuation
        telemetryContinuation = telemetryStream.continuation
        frameIngestor = CaptureFrameIngestor(
            controlContinuation: controlStream.continuation,
            telemetryContinuation: telemetryStream.continuation
        )
        self.captureBaseDirectory = captureBaseDirectory
        super.init()
        session.delegate = frameIngestor
        session.delegateQueue = frameIngestor.queue
    }

    deinit {
        controlContinuation.finish()
        telemetryContinuation.finish()
    }

    func prepare() async throws {
        guard ARWorldTrackingConfiguration.isSupported else {
            throw CaptureFailure.unsupportedDevice
        }
        let authorization = AVCaptureDevice.authorizationStatus(for: .video)
        let cameraAllowed: Bool
        switch authorization {
        case .authorized:
            cameraAllowed = true
        case .notDetermined:
            cameraAllowed = await AVCaptureDevice.requestAccess(for: .video)
        case .denied, .restricted:
            cameraAllowed = false
        @unknown default:
            cameraAllowed = false
        }
        guard cameraAllowed else { throw CaptureFailure.cameraPermissionDenied }
        try Task.checkCancellation()

        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        if let highResolutionVideoFormat = ARWorldTrackingConfiguration
            .recommendedVideoFormatForHighResolutionFrameCapturing {
            configuration.videoFormat = highResolutionVideoFormat
            supportsHighResolutionCapture = true
        } else {
            supportsHighResolutionCapture = false
        }
        let allDepthSemantics: ARConfiguration.FrameSemantics = [
            .sceneDepth,
            .smoothedSceneDepth,
        ]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(allDepthSemantics) {
            configuration.frameSemantics.formUnion(allDepthSemantics)
            supportsSceneDepth = true
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
            supportsSceneDepth = true
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(
            .smoothedSceneDepth
        ) {
            configuration.frameSemantics.insert(.smoothedSceneDepth)
            supportsSceneDepth = true
        } else {
            supportsSceneDepth = false
        }
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    func selectSubject(at viewPoint: CGPoint, viewportSize: CGSize) async throws {
        guard !isRecording else { throw CancellationError() }
        guard viewportSize.width > 0, viewportSize.height > 0 else {
            throw CaptureFailure.noCurrentFrame
        }
        guard supportsSceneDepth else {
            throw CaptureFailure.subjectDepthUnavailable
        }

        let viewNormalized = CGPoint(
            x: viewPoint.x / viewportSize.width,
            y: viewPoint.y / viewportSize.height
        )
        let snapshot = try makeSubjectSnapshot(viewportSize: viewportSize)
        let imageNormalized = viewNormalized.applying(
            snapshot.displayTransform.inverted()
        )
        let selection = try await subjectSelector.selectSubject(
            in: snapshot.candidate,
            normalizedImagePoint: imageNormalized
        )
        try Task.checkCancellation()
        guard !isRecording else { throw CancellationError() }
        subjectWorldPosition = selection.worldPosition
    }

    func updateInterfaceOrientation(_ interfaceOrientation: UIInterfaceOrientation) {
        guard let displayOrientation = CaptureDisplayOrientation(
            interfaceOrientation: interfaceOrientation
        ) else {
            return
        }
        self.interfaceOrientation = interfaceOrientation
        self.displayOrientation = displayOrientation
        frameIngestor.updateDisplayOrientation(displayOrientation)
    }

    func startRecording(mode: CaptureMode) throws {
        guard !isRecording else { return }
        guard store == nil, cleanupTask == nil else {
            throw CancellationError()
        }
        guard displayOrientation != nil else {
            throw CaptureFailure.noCurrentFrame
        }
        if mode.requiresSubjectSelection, subjectWorldPosition == nil {
            throw CaptureFailure.subjectNotFound
        }
        let store = try CaptureStore(
            mode: mode,
            baseDirectory: captureBaseDirectory
        )
        self.store = store
        frameIngestor.start(
            store: store,
            subjectWorldPosition: subjectWorldPosition,
            usesHighResolutionCapture: supportsHighResolutionCapture
        )
        isRecording = true
    }

    func stopAndFinalize() async throws -> CapturedDataset {
        guard let finalizingStore = store else {
            try Task.checkCancellation()
            throw CaptureFailure.persistence("capture storage was not initialized")
        }
        isRecording = false
        frameIngestor.stop()
        await frameIngestor.drain()
        try Task.checkCancellation()
        guard self.store === finalizingStore else { throw CancellationError() }
        if let processingTask = frameIngestor.processingTask {
            await processingTask.value
        }
        try Task.checkCancellation()
        guard self.store === finalizingStore else { throw CancellationError() }
        frameIngestor.finish()
        session.pause()
        let capture = try await finalizingStore.finalize()
        try Task.checkCancellation()
        guard self.store === finalizingStore else { throw CancellationError() }
        self.store = nil
        return capture
    }

    @discardableResult
    func stop() -> Task<Void, Never> {
        isRecording = false
        frameIngestor.stop()
        session.pause()
        if let cleanupTask {
            return cleanupTask
        }
        let abandonedStore = store
        store = nil
        let processingTask = frameIngestor.abort()
        let cleanupTask = Task { [weak self, frameIngestor, abandonedStore] in
            await frameIngestor.drain()
            if let processingTask {
                await processingTask.value
            }
            if let abandonedStore {
                do {
                    try await abandonedStore.discard()
                } catch {
                    Self.logger.error(
                        "Failed to discard unfinished capture: \(error.localizedDescription, privacy: .public)"
                    )
                }
            }
            self?.cleanupTask = nil
        }
        self.cleanupTask = cleanupTask
        return cleanupTask
    }

    func resetSubject() {
        guard !isRecording else { return }
        subjectWorldPosition = nil
    }

    private func makeSubjectSnapshot(
        viewportSize: CGSize
    ) throws -> (
        candidate: CaptureFrameCandidate,
        displayTransform: CGAffineTransform
    ) {
        guard let frame = session.currentFrame else {
            throw CaptureFailure.noCurrentFrame
        }
        guard let interfaceOrientation, let displayOrientation else {
            throw CaptureFailure.noCurrentFrame
        }
        let displayTransform = frame.displayTransform(
            for: interfaceOrientation,
            viewportSize: viewportSize
        )
        let candidate = try makeCaptureFrameCandidate(
            from: frame,
            displayOrientation: displayOrientation,
            subjectWorldPosition: subjectWorldPosition
        )
        return (candidate, displayTransform)
    }
}

struct CaptureCandidateAcceptance: Sendable {
    let commit: CaptureCommit
    let candidate: CaptureFrameCandidate
    let fallbackMessage: String?
}

func acceptCaptureCandidate(
    _ candidate: CaptureFrameCandidate,
    streamingFallback: CaptureFrameSnapshot?,
    in store: CaptureStore
) async throws -> CaptureCandidateAcceptance {
    do {
        let commit = try await store.accept(candidate)
        return CaptureCandidateAcceptance(
            commit: commit,
            candidate: candidate,
            fallbackMessage: nil
        )
    } catch let CaptureFailure.invalidFrame(message) {
        guard let streamingFallback else {
            throw CaptureFailure.invalidFrame(message)
        }
        try Task.checkCancellation()
        let fallbackCandidate = try streamingFallback.materialize()
        let commit = try await store.accept(fallbackCandidate)
        return CaptureCandidateAcceptance(
            commit: commit,
            candidate: fallbackCandidate,
            fallbackMessage: "The high-resolution frame was rejected; captured " +
                "a camera-stream frame instead."
        )
    }
}

private func makeHighResolutionCaptureFrameCandidate(
    session: ARSession,
    displayOrientation: CaptureDisplayOrientation,
    subjectWorldPosition: SIMD3<Float>?,
    requestGate: CaptureHighResolutionRequestGate
) async throws -> CaptureFrameCandidate {
    defer { requestGate.release() }
    try Task.checkCancellation()
    let frame = try await session.captureHighResolutionFrame()
    try Task.checkCancellation()
    guard case .normal = frame.camera.trackingState else {
        throw CaptureFailure.invalidFrame(
            "high-resolution frame did not have normal tracking"
        )
    }
    return try makeCaptureFrameCandidate(
        from: frame,
        displayOrientation: displayOrientation,
        subjectWorldPosition: subjectWorldPosition
    )
}

func captureCandidateHasUsableGeometry(
    _ candidate: CaptureFrameCandidate
) -> Bool {
    if candidate.subjectWorldPosition != nil {
        return candidate.depth != nil
    }
    return candidate.depth != nil || !candidate.rawFeaturePoints.isEmpty
}

private func makeCaptureFrameCandidate(
    from frame: ARFrame,
    displayOrientation: CaptureDisplayOrientation,
    subjectWorldPosition: SIMD3<Float>?
) throws -> CaptureFrameCandidate {
    try makeCaptureFrameSnapshot(
        from: frame,
        displayOrientation: displayOrientation,
        subjectWorldPosition: subjectWorldPosition
    ).materialize()
}

private func makeCaptureFrameSnapshot(
    from frame: ARFrame,
    displayOrientation: CaptureDisplayOrientation,
    subjectWorldPosition: SIMD3<Float>?
) throws -> CaptureFrameSnapshot {
    let imageWidth = CVPixelBufferGetWidth(frame.capturedImage)
    let imageHeight = CVPixelBufferGetHeight(frame.capturedImage)
    let resolution = frame.camera.imageResolution
    guard Int(resolution.width) == imageWidth,
          Int(resolution.height) == imageHeight else {
        throw CaptureFailure.invalidFrame(
            "ARKit imageResolution does not match capturedImage"
        )
    }
    let intrinsics = frame.camera.intrinsics
    let calibration = CaptureCalibrationRecord(
        width: imageWidth,
        height: imageHeight,
        fx: intrinsics.columns.0.x,
        fy: intrinsics.columns.1.y,
        cx: intrinsics.columns.2.x,
        cy: intrinsics.columns.2.y
    )
    let depthData = frame.sceneDepth ?? frame.smoothedSceneDepth
    let rawFeaturePoints: [SIMD3<Float>]
    if let cloud = frame.rawFeaturePoints {
        rawFeaturePoints = Array(cloud.points)
    } else {
        rawFeaturePoints = []
    }
    return CaptureFrameSnapshot(
        image: frame.capturedImage,
        depth: depthData?.depthMap,
        confidence: depthData?.confidenceMap,
        displayOrientation: displayOrientation,
        calibration: calibration,
        cameraToWorld: frame.camera.transform,
        timestamp: frame.timestamp,
        exposureDuration: frame.camera.exposureDuration,
        trackingState: "normal",
        rawFeaturePoints: rawFeaturePoints,
        subjectWorldPosition: subjectWorldPosition
    )
}

extension CaptureDisplayOrientation {
    init?(interfaceOrientation: UIInterfaceOrientation) {
        switch interfaceOrientation {
        case .portrait:
            self = .right
        case .portraitUpsideDown:
            self = .left
        case .landscapeLeft:
            self = .down
        case .landscapeRight:
            self = .up
        case .unknown:
            return nil
        @unknown default:
            return nil
        }
    }
}

private func copyPixelBuffer(_ source: CVPixelBuffer) throws -> OwnedPixelBuffer {
    let width = CVPixelBufferGetWidth(source)
    let height = CVPixelBufferGetHeight(source)
    let format = CVPixelBufferGetPixelFormatType(source)
    let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:]]
    var destination: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        format,
        attributes as CFDictionary,
        &destination
    )
    guard status == kCVReturnSuccess, let destination else {
        throw CaptureFailure.invalidFrame("could not copy the camera pixel buffer")
    }
    CVBufferPropagateAttachments(source, destination)
    CVPixelBufferLockBaseAddress(source, .readOnly)
    CVPixelBufferLockBaseAddress(destination, [])
    defer {
        CVPixelBufferUnlockBaseAddress(destination, [])
        CVPixelBufferUnlockBaseAddress(source, .readOnly)
    }

    let planeCount = CVPixelBufferGetPlaneCount(source)
    if planeCount > 0 {
        guard planeCount == CVPixelBufferGetPlaneCount(destination) else {
            throw CaptureFailure.invalidFrame("pixel-buffer plane count changed while copying")
        }
        for plane in 0..<planeCount {
            guard let sourceBase = CVPixelBufferGetBaseAddressOfPlane(source, plane),
                  let destinationBase = CVPixelBufferGetBaseAddressOfPlane(destination, plane) else {
                throw CaptureFailure.invalidFrame("pixel-buffer plane has no storage")
            }
            let sourceRowBytes = CVPixelBufferGetBytesPerRowOfPlane(source, plane)
            let destinationRowBytes = CVPixelBufferGetBytesPerRowOfPlane(destination, plane)
            let rowCount = CVPixelBufferGetHeightOfPlane(source, plane)
            let copyCount = min(sourceRowBytes, destinationRowBytes)
            for row in 0..<rowCount {
                destinationBase.advanced(by: row * destinationRowBytes).copyMemory(
                    from: sourceBase.advanced(by: row * sourceRowBytes),
                    byteCount: copyCount
                )
            }
        }
    } else {
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let destinationBase = CVPixelBufferGetBaseAddress(destination) else {
            throw CaptureFailure.invalidFrame("pixel buffer has no storage")
        }
        let sourceRowBytes = CVPixelBufferGetBytesPerRow(source)
        let destinationRowBytes = CVPixelBufferGetBytesPerRow(destination)
        let copyCount = min(sourceRowBytes, destinationRowBytes)
        for row in 0..<height {
            destinationBase.advanced(by: row * destinationRowBytes).copyMemory(
                from: sourceBase.advanced(by: row * sourceRowBytes),
                byteCount: copyCount
            )
        }
    }
    return OwnedPixelBuffer(destination)
}
