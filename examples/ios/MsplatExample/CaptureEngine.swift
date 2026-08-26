@preconcurrency import ARKit
import AVFoundation
import CoreVideo
import Foundation
import Synchronization
import UIKit

enum CaptureEngineEvent: Sendable {
    case frameCommitted(CaptureCommit)
    case frameQualityFallback(String)
    case frameRejected(String)
    case failed(String)
}

struct CaptureFrameAdmission: Sendable {
    fileprivate let generation: UInt64
    fileprivate let candidateID: UInt64
    fileprivate let store: CaptureStore
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
        var nextCandidateID: UInt64 = 0
        var activeCandidateID: UInt64?
        var isRecording = false
        var store: CaptureStore?
        var subjectWorldPosition: SIMD3<Float>?
        var displayOrientation: CaptureDisplayOrientation?
        var lastAcceptedTransform: simd_float4x4?
        var lastAcceptedTimestamp: TimeInterval?
        var processingTask: Task<Void, Never>?
    }

    private let state = Mutex(State())

    func start(store: CaptureStore, subjectWorldPosition: SIMD3<Float>?) {
        let previousTask = state.withLock { state in
            let previousTask = state.processingTask
            state.generation &+= 1
            state.nextCandidateID = 0
            state.activeCandidateID = nil
            state.isRecording = true
            state.store = store
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
                  state.activeCandidateID == nil,
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

            state.nextCandidateID &+= 1
            let candidateID = state.nextCandidateID
            state.activeCandidateID = candidateID
            return CaptureFrameAdmission(
                generation: state.generation,
                candidateID: candidateID,
                store: store,
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
                  state.activeCandidateID == admission.candidateID else {
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
                  state.activeCandidateID == admission.candidateID else {
                return false
            }
            if committed {
                state.lastAcceptedTransform = acceptedCameraToWorld
                    ?? admission.cameraToWorld
                state.lastAcceptedTimestamp = acceptedTimestamp
                    ?? admission.timestamp
            }
            state.activeCandidateID = nil
            state.processingTask = nil
            return true
        }
    }

    var processingTask: Task<Void, Never>? {
        state.withLock { state in state.processingTask }
    }

    func abort() {
        let task = state.withLock { state in
            let task = state.processingTask
            state.generation &+= 1
            state.activeCandidateID = nil
            state.isRecording = false
            state.store = nil
            state.subjectWorldPosition = nil
            state.processingTask = nil
            return task
        }
        task?.cancel()
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

    private let continuation: AsyncStream<CaptureEngineEvent>.Continuation
    private let admission = CaptureFrameAdmissionController()
    private let highResolutionRequestGate = CaptureHighResolutionRequestGate()

    init(continuation: AsyncStream<CaptureEngineEvent>.Continuation) {
        self.continuation = continuation
        super.init()
    }

    func start(store: CaptureStore, subjectWorldPosition: SIMD3<Float>?) {
        admission.start(
            store: store,
            subjectWorldPosition: subjectWorldPosition
        )
    }

    func stop() {
        admission.stop()
    }

    func abort() {
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
        guard highResolutionRequestGate.reserve() else {
            admission.complete(ticket, committed: false)
            return
        }

        let streamingCandidate: CaptureFrameCandidate
        do {
            streamingCandidate = try makeCaptureFrameCandidate(
                from: frame,
                displayOrientation: ticket.displayOrientation,
                subjectWorldPosition: ticket.subjectWorldPosition
            )
        } catch {
            highResolutionRequestGate.release()
            if admission.complete(ticket, committed: false) {
                continuation.yield(.frameRejected(error.localizedDescription))
            }
            return
        }

        let admission = admission
        let continuation = continuation
        let highResolutionRequestGate = highResolutionRequestGate
        let processingTask = Task(priority: .userInitiated) {
            do {
                let candidate: CaptureFrameCandidate
                let fallbackMessage: String?
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
                        fallbackMessage = nil
                    } else {
                        candidate = streamingCandidate
                        fallbackMessage = "The high-resolution frame lacked " +
                            "usable geometry; captured a camera-stream frame instead."
                    }
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    candidate = streamingCandidate
                    fallbackMessage = "High-resolution capture failed; captured " +
                        "a camera-stream frame instead."
                }
                try Task.checkCancellation()
                let commit = try await ticket.store.accept(candidate)
                try Task.checkCancellation()
                if admission.complete(
                    ticket,
                    committed: true,
                    acceptedCameraToWorld: candidate.cameraToWorld,
                    acceptedTimestamp: candidate.timestamp
                ) {
                    continuation.yield(.frameCommitted(commit))
                    if let fallbackMessage {
                        continuation.yield(.frameQualityFallback(fallbackMessage))
                    }
                }
            } catch is CancellationError {
                admission.complete(ticket, committed: false)
            } catch {
                if admission.complete(ticket, committed: false) {
                    continuation.yield(.frameRejected(error.localizedDescription))
                }
            }
        }
        admission.register(task: processingTask, for: ticket)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        abort()
        continuation.yield(.failed(error.localizedDescription))
    }
}

@MainActor
final class CaptureEngine: NSObject {
    let session = ARSession()
    let events: AsyncStream<CaptureEngineEvent>

    private let continuation: AsyncStream<CaptureEngineEvent>.Continuation
    private let frameIngestor: CaptureFrameIngestor
    private let subjectSelector = CaptureSubjectSelector()
    private var store: CaptureStore?
    private var interfaceOrientation: UIInterfaceOrientation?
    private var displayOrientation: CaptureDisplayOrientation?
    private(set) var subjectWorldPosition: SIMD3<Float>?
    private(set) var supportsSceneDepth = false
    private(set) var isRecording = false

    override init() {
        let stream = AsyncStream.makeStream(
            of: CaptureEngineEvent.self,
            bufferingPolicy: .bufferingNewest(32)
        )
        events = stream.stream
        continuation = stream.continuation
        frameIngestor = CaptureFrameIngestor(continuation: stream.continuation)
        super.init()
        session.delegate = frameIngestor
        session.delegateQueue = frameIngestor.queue
    }

    deinit {
        continuation.finish()
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
        guard let highResolutionVideoFormat = ARWorldTrackingConfiguration
            .recommendedVideoFormatForHighResolutionFrameCapturing else {
            throw CaptureFailure.highResolutionCaptureUnavailable
        }
        configuration.videoFormat = highResolutionVideoFormat
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
        guard displayOrientation != nil else {
            throw CaptureFailure.noCurrentFrame
        }
        if mode.requiresSubjectSelection, subjectWorldPosition == nil {
            throw CaptureFailure.subjectNotFound
        }
        let store = try CaptureStore(mode: mode)
        self.store = store
        frameIngestor.start(
            store: store,
            subjectWorldPosition: subjectWorldPosition
        )
        isRecording = true
    }

    func stopAndFinalize() async throws -> CapturedDataset {
        isRecording = false
        frameIngestor.stop()
        await frameIngestor.drain()
        if let processingTask = frameIngestor.processingTask {
            await processingTask.value
        }
        session.pause()
        defer {
            store = nil
            frameIngestor.finish()
        }
        guard let store else {
            throw CaptureFailure.persistence("capture storage was not initialized")
        }
        return try await store.finalize()
    }

    func stop() {
        isRecording = false
        frameIngestor.abort()
        session.pause()
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
    let image = try copyPixelBuffer(frame.capturedImage)
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
    let depth = try depthData.map { try copyPixelBuffer($0.depthMap) }
    let confidence = try depthData?.confidenceMap.map(copyPixelBuffer)
    let rawFeaturePoints: [SIMD3<Float>]
    if let cloud = frame.rawFeaturePoints {
        rawFeaturePoints = Array(cloud.points)
    } else {
        rawFeaturePoints = []
    }
    return CaptureFrameCandidate(
        image: image,
        depth: depth,
        confidence: confidence,
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
