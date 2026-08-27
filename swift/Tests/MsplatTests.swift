import XCTest
@testable import Msplat
import MsplatCore

final class MsplatTests: XCTestCase {

    static let gardenPath = "../datasets/mipnerf360/garden"

    func testConfigDefaults() {
        let config = TrainingConfig()
        XCTAssertEqual(config.iterations, 30_000)
        XCTAssertEqual(config.shDegree, 3)
        XCTAssertEqual(config.ssimWeight, 0.2, accuracy: 0.001)
        XCTAssertFalse(config.refinePhotometricGains)
        XCTAssertFalse(config.refineCameraPoses)
        XCTAssertEqual(config.cameraPoseConditioning, .raw)
        XCTAssertEqual(config.trainingMaskMode, .coverage)
        XCTAssertEqual(config.transparentAlphaLossWeight, 0.1, accuracy: 0.001)
        XCTAssertEqual(config.toRefinementOptionsV8().flags, 0)
        XCTAssertEqual(
            config.toTrainingMaskOptionsV11().mode,
            MSPLAT_TRAINING_MASK_MODE_COVERAGE.rawValue
        )
        XCTAssertNoThrow(try config.validate())
    }

    func testTransparentMaskModeMapsToVersionedNativeOptions() {
        var config = TrainingConfig()
        config.trainingMaskMode = .transparent
        config.transparentAlphaLossWeight = 0.25

        let options = config.toTrainingMaskOptionsV11()
        XCTAssertEqual(
            options.mode,
            MSPLAT_TRAINING_MASK_MODE_TRANSPARENT.rawValue
        )
        XCTAssertEqual(options.alphaLossWeight, 0.25, accuracy: 0.001)
        XCTAssertEqual(MemoryLayout<MsplatTrainingMaskOptionsV11>.size, 16)
    }

    func testPhotometricRefinementMapsToVersionedNativeOptions() {
        var config = TrainingConfig()
        config.refinePhotometricGains = true

        let options = config.toRefinementOptionsV8()
        XCTAssertEqual(
            options.flags,
            UInt32(MSPLAT_REFINEMENT_PHOTOMETRIC_RGB_GAINS)
        )
        XCTAssertEqual(MemoryLayout<MsplatRefinementOptionsV8>.size, 16)
    }

    func testCameraPoseRefinementMapsToVersionedNativeOptions() {
        var config = TrainingConfig()
        config.refineCameraPoses = true

        var options = config.toRefinementOptionsV8()
        XCTAssertEqual(
            options.flags,
            UInt32(MSPLAT_REFINEMENT_CAMERA_POSE_DELTAS)
        )

        config.refinePhotometricGains = true
        options = config.toRefinementOptionsV8()
        XCTAssertEqual(
            options.flags,
            UInt32(MSPLAT_REFINEMENT_PHOTOMETRIC_RGB_GAINS)
                | UInt32(MSPLAT_REFINEMENT_CAMERA_POSE_DELTAS)
        )

        config.cameraPoseConditioning = .camP
        options = config.toRefinementOptionsV8()
        XCTAssertEqual(
            options.flags,
            UInt32(MSPLAT_REFINEMENT_PHOTOMETRIC_RGB_GAINS)
                | UInt32(MSPLAT_REFINEMENT_CAMERA_POSE_DELTAS)
                | UInt32(MSPLAT_REFINEMENT_CAMERA_POSE_CAMP_CONDITIONING)
        )
        XCTAssertNoThrow(try config.validate())
    }

    func testNativePoseRefinementStateConversion() throws {
        XCTAssertEqual(MemoryLayout<MsplatPoseRefinementStateV15>.size, 128)

        var native = MsplatPoseRefinementStateV15()
        native.flags = UInt32(MSPLAT_POSE_REFINEMENT_STATE_ENABLED)
            | UInt32(MSPLAT_POSE_REFINEMENT_STATE_ANCHOR)
        native.canonicalCameraIndex = 7
        native.optimizerStepCount = 11
        copyFloats(
            [1, 2, 3, 0.1, 0.2, 0.3],
            into: &native.poseDelta
        )
        native.translationNorm = Float(14).squareRoot()
        native.rotationNorm = Float(0.14).squareRoot()
        let correctedPose: [Float] = [
            1, 0, 0, 4,
            0, 1, 0, 5,
            0, 0, 1, 6,
            0, 0, 0, 1,
        ]
        copyFloats(correctedPose, into: &native.correctedCameraToWorld)

        let frameIDBytes = Array("frame_000007".utf8)
        let state = try frameIDBytes.withUnsafeBufferPointer { bytes in
            native.frameId = bytes.baseAddress.map {
                UnsafeRawPointer($0).assumingMemoryBound(to: CChar.self)
            }
            native.frameIdLength = UInt64(bytes.count)
            return try CameraPoseRefinementState(from: native)
        }

        XCTAssertTrue(state.isEnabled)
        XCTAssertTrue(state.isAnchor)
        XCTAssertEqual(state.canonicalCameraIndex, 7)
        XCTAssertEqual(state.optimizerStepCount, 11)
        XCTAssertEqual(state.translationDelta, [1, 2, 3])
        XCTAssertEqual(state.rotationDelta, [0.1, 0.2, 0.3])
        XCTAssertEqual(
            state.translationNorm,
            Float(14).squareRoot(),
            accuracy: 0.000_001
        )
        XCTAssertEqual(
            state.rotationNorm,
            Float(0.14).squareRoot(),
            accuracy: 0.000_001
        )
        XCTAssertEqual(state.correctedCameraToWorld.elements, correctedPose)
        XCTAssertEqual(state.frameID, "frame_000007")
    }

    func testInvalidConfigIsRejected() {
        var invalidConfigs: [TrainingConfig] = []

        var config = TrainingConfig()
        config.iterations = 0
        invalidConfigs.append(config)

        config = TrainingConfig()
        config.shDegreeInterval = 0
        invalidConfigs.append(config)

        config = TrainingConfig()
        config.resolutionSchedule = 0
        invalidConfigs.append(config)

        config = TrainingConfig()
        config.refineEvery = 0
        invalidConfigs.append(config)

        config = TrainingConfig()
        config.resetAlphaEvery = 0
        invalidConfigs.append(config)

        config = TrainingConfig()
        config.numDownscales = 31
        invalidConfigs.append(config)

        config = TrainingConfig()
        config.ssimWeight = .nan
        invalidConfigs.append(config)

        config = TrainingConfig()
        config.bgColor = (0, .infinity, 0)
        invalidConfigs.append(config)

        config = TrainingConfig()
        config.transparentAlphaLossWeight = -.infinity
        invalidConfigs.append(config)

        config = TrainingConfig()
        config.transparentAlphaLossWeight = -0.1
        invalidConfigs.append(config)

        config = TrainingConfig()
        config.trainingMaskMode = .transparent
        config.refinePhotometricGains = true
        invalidConfigs.append(config)

        config = TrainingConfig()
        config.cameraPoseConditioning = .camP
        invalidConfigs.append(config)

        for invalidConfig in invalidConfigs {
            XCTAssertThrowsError(try invalidConfig.validate())
        }
    }

    func testCameraPoseValidation() throws {
        XCTAssertNoThrow(
            try CameraPose(elements: [Float](repeating: 0, count: 16))
        )
        XCTAssertThrowsError(
            try CameraPose(elements: [Float](repeating: 0, count: 15))
        )
        var nonFinite = [Float](repeating: 0, count: 16)
        nonFinite[3] = .nan
        XCTAssertThrowsError(try CameraPose(elements: nonFinite))
    }

    func testTrainingTelemetryConversionKeepsSubmissionAndCompletionSeparate() {
        var native = MsplatTrainingMetricsV12()
        native.flags = UInt32(MSPLAT_TRAINING_METRICS_HAS_SUBMITTED_STEP)
            | UInt32(MSPLAT_TRAINING_METRICS_HAS_COMPLETED_STEP)
            | UInt32(MSPLAT_TRAINING_METRICS_GPU_TIME_VALID)
            | UInt32(MSPLAT_TRAINING_METRICS_LOSS_VALID)
            | UInt32(MSPLAT_TRAINING_METRICS_INTERSECTIONS_VALID)
            | UInt32(MSPLAT_TRAINING_METRICS_COUNT_GPU_TIME_VALID)
            | UInt32(MSPLAT_TRAINING_METRICS_QUEUE_IDLE_TIME_VALID)
        native.submitted.iteration = 12
        native.submitted.splatCount = 120
        native.submitted.modelCapacity = 150
        native.submitted.effectiveWidth = 1_920
        native.submitted.effectiveHeight = 1_080
        native.submitted.activeSHDegree = 2
        native.submitted.cpuSubmitMs = 1.25
        native.completed.iteration = 10
        native.completed.splatCount = 100
        native.completed.modelCapacity = 125
        native.completed.effectiveWidth = 960
        native.completed.effectiveHeight = 540
        native.completed.activeSHDegree = 1
        native.completed.cpuSubmitMs = 1.5
        native.completed.gpuExecutionMs = 8.5
        native.completed.endToEndMs = 11.0
        native.completed.loss = 0.125
        native.completed.overflowKinds = UInt32(MSPLAT_RASTER_OVERFLOW_TILE_CAP)
            | UInt32(MSPLAT_RASTER_OVERFLOW_PACKED_CAPACITY)
        native.completed.retainedPackedIntersectionCount = 4_000_000_001
        native.completed.packedIntersectionCapacity = 5_000_000_001
        native.completed.imagePrepareMs = 2.25
        native.completed.countGpuMs = 3.5
        native.completed.countWaitWallMs = 6.75
        native.completed.queueIdleMs = 1.75
        native.completed.postCountEncodeMs = 0.5
        native.completed.intersectionArenaGrowMs = 0.25
        native.completed.maximumTileCount = 2_049
        native.completed.activeTileCount = 400
        native.completed.trivialTileCount = 100
        native.completed.smallTileCount = 200
        native.completed.mediumTileCount = 90
        native.completed.largeTileCount = 10
        native.overflowedCompletedSteps = 3
        native.tileCapOverflowedSteps = 2
        native.packedCapacityOverflowedSteps = 1
        native.lastOverflowIteration = 10

        let telemetry = TrainingTelemetry(from: native)
        XCTAssertEqual(telemetry.submitted?.iteration, 12)
        XCTAssertEqual(telemetry.submitted?.effectiveWidth, 1_920)
        XCTAssertEqual(telemetry.completed?.iteration, 10)
        XCTAssertEqual(telemetry.completed?.effectiveWidth, 960)
        XCTAssertEqual(telemetry.completed?.gpuExecutionMs, 8.5)
        XCTAssertEqual(telemetry.completed?.loss, 0.125)
        XCTAssertEqual(
            telemetry.completed?.overflowKinds,
            [.tileCapacity, .packedCapacity]
        )
        XCTAssertEqual(
            telemetry.completed?.retainedPackedIntersectionCount,
            4_000_000_001
        )
        XCTAssertEqual(telemetry.completed?.exactIntersectionCount, 4_000_000_001)
        XCTAssertEqual(telemetry.completed?.imagePrepareMs, 2.25)
        XCTAssertEqual(telemetry.completed?.countGpuMs, 3.5)
        XCTAssertEqual(telemetry.completed?.countWaitWallMs, 6.75)
        XCTAssertEqual(telemetry.completed?.queueIdleMs, 1.75)
        XCTAssertEqual(telemetry.completed?.postCountEncodeMs, 0.5)
        XCTAssertEqual(telemetry.completed?.intersectionArenaGrowMs, 0.25)
        XCTAssertEqual(telemetry.completed?.maximumTileCount, 2_049)
        XCTAssertEqual(telemetry.completed?.activeTileCount, 400)
        XCTAssertEqual(telemetry.completed?.trivialTileCount, 100)
        XCTAssertEqual(telemetry.completed?.smallTileCount, 200)
        XCTAssertEqual(telemetry.completed?.mediumTileCount, 90)
        XCTAssertEqual(telemetry.completed?.largeTileCount, 10)
        XCTAssertEqual(telemetry.overflowedCompletedSteps, 3)
        XCTAssertEqual(telemetry.lastOverflowIteration, 10)
    }

    func testTrainingTelemetryValidityFlagsControlOptionals() {
        var native = MsplatTrainingMetricsV12()
        native.completed.iteration = 7
        native.completed.gpuExecutionMs = 99
        native.completed.loss = 99
        native.completed.queueIdleMs = 99

        var telemetry = TrainingTelemetry(from: native)
        XCTAssertNil(telemetry.submitted)
        XCTAssertNil(telemetry.completed)

        native.flags = UInt32(MSPLAT_TRAINING_METRICS_HAS_COMPLETED_STEP)
        telemetry = TrainingTelemetry(from: native)
        XCTAssertEqual(telemetry.completed?.iteration, 7)
        XCTAssertNil(telemetry.completed?.gpuExecutionMs)
        XCTAssertNil(telemetry.completed?.loss)
        XCTAssertNil(telemetry.completed?.countGpuMs)
        XCTAssertNil(telemetry.completed?.queueIdleMs)

        native.flags |= UInt32(MSPLAT_TRAINING_METRICS_COUNT_GPU_TIME_VALID)
        native.completed.countGpuMs = 4.5
        telemetry = TrainingTelemetry(from: native)
        XCTAssertEqual(telemetry.completed?.countGpuMs, 4.5)
        XCTAssertNil(telemetry.completed?.queueIdleMs)

        native.flags |= UInt32(MSPLAT_TRAINING_METRICS_QUEUE_IDLE_TIME_VALID)
        native.completed.queueIdleMs = 1.25
        telemetry = TrainingTelemetry(from: native)
        XCTAssertEqual(telemetry.completed?.queueIdleMs, 1.25)
    }

    func testTrainingMemoryConversionPreservesCountsAndValidity() {
        var native = MsplatTrainingMemoryMetricsV17()
        native.flags = UInt32(MSPLAT_MEMORY_METRICS_PHYS_FOOTPRINT_VALID)
        native.trainerModelBufferBytes = 1
        native.engineSharedTransientBufferBytes = 2
        native.engineTrainingTransientBufferBytes = 3
        native.trainerTelemetryReadbackBytes = 4
        native.trainerImageCacheGpuBytes = 5
        native.trainerImageCacheCpuBytes = 6
        native.processPhysFootprintBytes = 5_000_000_001
        native.processAvailableBytes = 6_000_000_001
        native.trainingGpuImageCacheHits = 3
        native.trainingGpuImageCacheMisses = 1
        native.trainingTargetPrefetchScheduled = 8
        native.trainingTargetPrefetchUsed = 6
        native.trainingTargetPrefetchWaited = 4
        native.trainingTargetPrefetchDiscarded = 1

        let snapshot = TrainingMemorySnapshot(from: native)
        XCTAssertEqual(snapshot.trackedNativeBufferBytes, 15)
        XCTAssertEqual(snapshot.processPhysicalFootprintBytes, 5_000_000_001)
        XCTAssertNil(snapshot.processAvailableBytes)
        XCTAssertEqual(snapshot.trainingGpuImageCacheHitRate, 0.75)
        XCTAssertEqual(snapshot.trainingTargetPrefetchReady, 2)
        XCTAssertEqual(snapshot.trainingTargetPrefetchWaitRate, 2.0 / 3.0)
        XCTAssertEqual(snapshot.trainingTargetPrefetchDiscarded, 1)

        native.trainingGpuImageCacheHits = 0
        native.trainingGpuImageCacheMisses = 0
        XCTAssertNil(TrainingMemorySnapshot(from: native).trainingGpuImageCacheHitRate)

        native.trainingTargetPrefetchUsed = 0
        native.trainingTargetPrefetchWaited = 0
        XCTAssertNil(TrainingMemorySnapshot(from: native)
            .trainingTargetPrefetchWaitRate)
    }

    func testLoadDataset() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0,
            evalMode: true,
            testEvery: 8
        )
        XCTAssertGreaterThan(dataset.numTrain, 0)
        XCTAssertGreaterThan(dataset.numTest, 0)
    }

    func testTrainShort() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 10
        config.numDownscales = 0

        let trainer = GaussianTrainer(dataset: dataset, config: config)

        for _ in 0..<10 {
            let stats = trainer.step()
            XCTAssertGreaterThan(stats.splatCount, 0)
        }

        XCTAssertEqual(trainer.iteration, 10)
        XCTAssertGreaterThan(trainer.splatCount, 100_000)

        let telemetry = try trainer.trainingMetrics()
        XCTAssertEqual(telemetry.submitted?.iteration, 10)
        let completed = try XCTUnwrap(telemetry.completed)
        XCTAssertGreaterThan(completed.iteration, 0)
        XCTAssertLessThanOrEqual(completed.iteration, 10)
        XCTAssertGreaterThan(completed.exactIntersectionCount ?? 0, 0)
        XCTAssertGreaterThan(completed.maximumTileCount, 0)
        XCTAssertGreaterThanOrEqual(completed.imagePrepareMs, 0)
        XCTAssertGreaterThanOrEqual(completed.countWaitWallMs, 0)
        if let queueIdleMs = completed.queueIdleMs {
            XCTAssertGreaterThanOrEqual(queueIdleMs, 0)
        }
        XCTAssertGreaterThanOrEqual(completed.postCountEncodeMs, 0)
        XCTAssertGreaterThanOrEqual(completed.intersectionArenaGrowMs, 0)
        let tileCount = ((completed.effectiveWidth + 15) / 16)
            * ((completed.effectiveHeight + 15) / 16)
        XCTAssertEqual(
            completed.trivialTileCount + completed.smallTileCount
                + completed.mediumTileCount + completed.largeTileCount,
            tileCount
        )
        XCTAssertLessThanOrEqual(completed.activeTileCount, tileCount)
    }

    func testRender() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 5
        config.numDownscales = 0

        let trainer = GaussianTrainer(dataset: dataset, config: config)
        for _ in 0..<5 { trainer.step() }

        let rendered = trainer.render(cameraIndex: 0)
        XCTAssertGreaterThan(rendered.width, 0)
        XCTAssertGreaterThan(rendered.height, 0)
        XCTAssertEqual(rendered.pixels.count, rendered.width * rendered.height * 3)
    }

    func testExportPly() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 5
        config.numDownscales = 0

        let trainer = GaussianTrainer(dataset: dataset, config: config)
        for _ in 0..<5 { trainer.step() }

        let tmpPath = NSTemporaryDirectory() + "msplat_test_export.ply"
        trainer.exportPly(to: tmpPath)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tmpPath))

        let fileSize = try FileManager.default.attributesOfItem(atPath: tmpPath)[.size] as! Int
        XCTAssertGreaterThan(fileSize, 0)

        try FileManager.default.removeItem(atPath: tmpPath)
    }

    func testExportSpz() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 5
        config.numDownscales = 0

        let trainer = GaussianTrainer(dataset: dataset, config: config)
        for _ in 0..<5 { trainer.step() }

        let tmpPath = NSTemporaryDirectory() + "msplat_test_export.spz"
        trainer.exportSpz(to: tmpPath)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tmpPath))

        let data = try Data(contentsOf: URL(fileURLWithPath: tmpPath))
        XCTAssertGreaterThan(data.count, 100)
        XCTAssertEqual(data[0], 0x1f)
        XCTAssertEqual(data[1], 0x8b)

        try FileManager.default.removeItem(atPath: tmpPath)
    }
}

private func copyFloats<Tuple>(_ values: [Float], into tuple: inout Tuple) {
    withUnsafeMutableBytes(of: &tuple) { destination in
        values.withUnsafeBytes { source in
            XCTAssertEqual(destination.count, source.count)
            guard destination.count == source.count else { return }
            destination.copyBytes(from: source)
        }
    }
}
