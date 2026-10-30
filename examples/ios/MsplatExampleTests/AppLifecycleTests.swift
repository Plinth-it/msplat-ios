@preconcurrency import ARKit
import Foundation
@testable import MsplatExample
import simd
import XCTest

final class AppLifecycleTests: XCTestCase {
    private var temporaryDirectory: URL?

    override func setUpWithError() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(
                path: "msplat-example-lifecycle-\(UUID().uuidString)",
                directoryHint: .isDirectory
            )
        temporaryDirectory = directory
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
    }

    override func tearDownWithError() throws {
        if let temporaryDirectory {
            try FileManager.default.removeItem(at: temporaryDirectory)
        }
        temporaryDirectory = nil
    }

    // MARK: - Screen awake

    @MainActor
    func testScreenAwakeControllerKeepsAwakeUntilLastAssertionEnds() {
        let recorder = IdleTimerRecorder()
        let controller = ScreenAwakeController { recorder.values.append($0) }
        let training = controller.makeAssertion()
        let capture = controller.makeAssertion()

        training.isActive = true
        capture.isActive = true
        XCTAssertEqual(recorder.values, [true])
        XCTAssertTrue(controller.keepsScreenAwake)

        training.isActive = false
        XCTAssertEqual(recorder.values, [true])
        XCTAssertTrue(controller.keepsScreenAwake)

        capture.isActive = false
        XCTAssertEqual(recorder.values, [true, false])
        XCTAssertFalse(controller.keepsScreenAwake)
    }

    @MainActor
    func testScreenAwakeAssertionUpdatesAreIdempotent() {
        let recorder = IdleTimerRecorder()
        let controller = ScreenAwakeController { recorder.values.append($0) }
        let assertion = controller.makeAssertion()
        let other = controller.makeAssertion()

        assertion.isActive = true
        assertion.isActive = true
        other.isActive = true
        assertion.isActive = false
        assertion.isActive = false
        XCTAssertEqual(recorder.values, [true])

        other.isActive = false
        other.isActive = false
        assertion.isActive = true
        XCTAssertEqual(recorder.values, [true, false, true])
    }

    // MARK: - Training pause gate

    @MainActor
    func testTrainingPauseGatePassesThroughWhileActive() async throws {
        let gate = TrainingPauseGate()

        try await gate.waitUntilResumed()
        gate.pause()
        gate.resume()
        try await gate.waitUntilResumed()

        XCTAssertFalse(gate.isPaused)
        XCTAssertEqual(gate.waiterCount, 0)
    }

    @MainActor
    func testTrainingPauseGateHoldsWaiterUntilResumed() async throws {
        let gate = TrainingPauseGate()
        let progress = WaiterProgress()
        gate.pause()
        gate.pause()

        let waiter = Task { @MainActor in
            try await gate.waitUntilResumed()
            progress.didResume = true
        }
        while gate.waiterCount == 0 {
            await Task.yield()
        }
        XCTAssertTrue(gate.isPaused)
        XCTAssertFalse(progress.didResume)

        gate.resume()
        try await waiter.value

        XCTAssertTrue(progress.didResume)
        XCTAssertFalse(gate.isPaused)
        XCTAssertEqual(gate.waiterCount, 0)
    }

    @MainActor
    func testTrainingPauseGateCancellationEndsOnlyThatWait() async throws {
        let gate = TrainingPauseGate()
        gate.pause()

        let waiter = Task { @MainActor in
            try await gate.waitUntilResumed()
        }
        while gate.waiterCount == 0 {
            await Task.yield()
        }
        waiter.cancel()
        do {
            try await waiter.value
            XCTFail("A cancelled waiter must not resume normally")
        } catch is CancellationError {
        }

        XCTAssertTrue(gate.isPaused)
        XCTAssertEqual(gate.waiterCount, 0)
    }

    @MainActor
    func testTrainingPauseGateRejectsAlreadyCancelledWaiter() async {
        let gate = TrainingPauseGate()
        gate.pause()

        // The task cannot start until this test yields the main actor, so it
        // is cancelled before it registers as a waiter.
        let waiter = Task { @MainActor in
            try await gate.waitUntilResumed()
        }
        waiter.cancel()
        do {
            try await waiter.value
            XCTFail("A cancelled waiter must not resume normally")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }

        XCTAssertTrue(gate.isPaused)
        XCTAssertEqual(gate.waiterCount, 0)
    }

    // MARK: - ARKit interruptions

    func testCaptureAdmissionClosesOnInterruptionUntilRelocalized() throws {
        let controller = try makeRecordingAdmissionController()
        defer { controller.finish() }
        let beforeInterruption = try XCTUnwrap(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 1
        ))
        XCTAssertTrue(controller.complete(beforeInterruption, committed: true))

        let moved = translated(x: 0.1)
        XCTAssertEqual(controller.interrupt(), .interrupted)
        XCTAssertNil(controller.interrupt())
        XCTAssertNil(controller.begin(cameraToWorld: moved, timestamp: 2))

        // Normal tracking before the interruption ends proves nothing.
        XCTAssertNil(controller.observeTracking(.normal))
        XCTAssertNil(controller.begin(cameraToWorld: moved, timestamp: 2))

        XCTAssertEqual(controller.endInterruption(), .relocalizing)
        XCTAssertNil(controller.endInterruption())
        XCTAssertNil(controller.observeTracking(.relocalizing))
        XCTAssertNil(controller.observeTracking(.other))
        XCTAssertNil(controller.begin(cameraToWorld: moved, timestamp: 3))

        XCTAssertEqual(controller.observeTracking(.normal), .continuous)
        XCTAssertNil(controller.observeTracking(.normal))
        let afterRelocalization = try XCTUnwrap(controller.begin(
            cameraToWorld: moved,
            timestamp: 4
        ))
        XCTAssertTrue(controller.complete(afterRelocalization, committed: false))
    }

    func testCaptureAdmissionTreatsRelocalizingFramesAsInterruptionEnd() throws {
        let controller = try makeRecordingAdmissionController()
        defer { controller.finish() }

        XCTAssertEqual(controller.interrupt(), .interrupted)
        XCTAssertEqual(controller.observeTracking(.relocalizing), .relocalizing)
        XCTAssertNil(controller.endInterruption())
        XCTAssertEqual(controller.observeTracking(.normal), .continuous)

        XCTAssertEqual(controller.interrupt(), .interrupted)
        XCTAssertEqual(controller.continuity, .interrupted)
    }

    func testCaptureAdmissionEndsRecordingWhenTrackingReinitializes() throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let controller = try makeRecordingAdmissionController()
        defer { controller.finish() }

        XCTAssertEqual(controller.interrupt(), .interrupted)
        XCTAssertEqual(controller.endInterruption(), .relocalizing)
        XCTAssertEqual(controller.observeTracking(.initializing), .lost)

        // A new world origin can never be reconciled with this recording.
        XCTAssertNil(controller.observeTracking(.normal))
        XCTAssertNil(controller.interrupt())
        XCTAssertNil(controller.endInterruption())
        XCTAssertNil(controller.begin(
            cameraToWorld: translated(x: 0.1),
            timestamp: 5
        ))

        let nextStore = try CaptureStore(
            mode: .scene,
            baseDirectory: temporaryDirectory
        )
        controller.start(store: nextStore, subjectWorldPosition: nil)
        XCTAssertEqual(controller.continuity, .continuous)
        let nextRecording = try XCTUnwrap(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 6
        ))
        XCTAssertTrue(controller.complete(nextRecording, committed: false))
    }

    func testCaptureAdmissionKeepsPreInterruptionCandidateCommittable() throws {
        let controller = try makeRecordingAdmissionController()
        defer { controller.finish() }
        let inFlight = try XCTUnwrap(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 1
        ))
        XCTAssertTrue(controller.hasUninterruptedTracking(since: inFlight))

        XCTAssertEqual(controller.interrupt(), .interrupted)
        XCTAssertEqual(controller.endInterruption(), .relocalizing)
        XCTAssertEqual(controller.observeTracking(.normal), .continuous)

        // Its streaming snapshot still commits, but a high-resolution frame
        // delivered after the interruption must not replace it.
        XCTAssertFalse(controller.hasUninterruptedTracking(since: inFlight))
        XCTAssertTrue(controller.complete(inFlight, committed: true))

        let next = try XCTUnwrap(controller.begin(
            cameraToWorld: translated(x: 0.1),
            timestamp: 2
        ))
        XCTAssertTrue(controller.hasUninterruptedTracking(since: next))
        XCTAssertTrue(controller.complete(next, committed: false))
    }

    func testCaptureAdmissionIgnoresInterruptionsOutsideRecording() throws {
        let controller = CaptureFrameAdmissionController()

        XCTAssertNil(controller.interrupt())
        XCTAssertNil(controller.endInterruption())
        XCTAssertNil(controller.observeTracking(.initializing))
        XCTAssertEqual(controller.continuity, .continuous)
    }

    func testCaptureTrackingObservationMapsARKitStates() {
        XCTAssertEqual(CaptureTrackingObservation(.normal), .normal)
        XCTAssertEqual(
            CaptureTrackingObservation(.limited(.relocalizing)),
            .relocalizing
        )
        XCTAssertEqual(
            CaptureTrackingObservation(.limited(.initializing)),
            .initializing
        )
        XCTAssertEqual(
            CaptureTrackingObservation(.limited(.excessiveMotion)),
            .other
        )
        XCTAssertEqual(CaptureTrackingObservation(.notAvailable), .other)
    }

    func testCaptureDepthRequestsOnlyUnsmoothedDepthWhenSupported() {
        XCTAssertEqual(
            CaptureEngine.depthFrameSemantics { _ in true },
            .sceneDepth
        )
        XCTAssertEqual(
            CaptureEngine.depthFrameSemantics { $0 == .smoothedSceneDepth },
            .smoothedSceneDepth
        )
        XCTAssertNil(CaptureEngine.depthFrameSemantics { _ in false })
    }

    // MARK: - Helpers

    private func makeRecordingAdmissionController() throws -> CaptureFrameAdmissionController {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let store = try CaptureStore(
            mode: .scene,
            baseDirectory: temporaryDirectory
        )
        let controller = CaptureFrameAdmissionController()
        controller.start(store: store, subjectWorldPosition: nil)
        controller.updateDisplayOrientation(.right)
        return controller
    }

    private func translated(x: Float) -> simd_float4x4 {
        var transform = matrix_identity_float4x4
        transform.columns.3.x = x
        return transform
    }
}

@MainActor
private final class IdleTimerRecorder {
    var values: [Bool] = []
}

@MainActor
private final class WaiterProgress {
    var didResume = false
}
