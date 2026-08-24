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
        var native = MsplatTrainingMetrics()
        native.flags = UInt32(MSPLAT_TRAINING_METRICS_HAS_SUBMITTED_STEP)
            | UInt32(MSPLAT_TRAINING_METRICS_HAS_COMPLETED_STEP)
            | UInt32(MSPLAT_TRAINING_METRICS_GPU_TIME_VALID)
            | UInt32(MSPLAT_TRAINING_METRICS_LOSS_VALID)
            | UInt32(MSPLAT_TRAINING_METRICS_INTERSECTIONS_VALID)
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
        XCTAssertEqual(telemetry.overflowedCompletedSteps, 3)
        XCTAssertEqual(telemetry.lastOverflowIteration, 10)
    }

    func testTrainingTelemetryValidityFlagsControlOptionals() {
        var native = MsplatTrainingMetrics()
        native.completed.iteration = 7
        native.completed.gpuExecutionMs = 99
        native.completed.loss = 99

        var telemetry = TrainingTelemetry(from: native)
        XCTAssertNil(telemetry.submitted)
        XCTAssertNil(telemetry.completed)

        native.flags = UInt32(MSPLAT_TRAINING_METRICS_HAS_COMPLETED_STEP)
        telemetry = TrainingTelemetry(from: native)
        XCTAssertEqual(telemetry.completed?.iteration, 7)
        XCTAssertNil(telemetry.completed?.gpuExecutionMs)
        XCTAssertNil(telemetry.completed?.loss)
    }

    func testTrainingMemoryConversionPreservesCountsAndValidity() {
        var native = MsplatTrainingMemoryMetrics()
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

        let snapshot = TrainingMemorySnapshot(from: native)
        XCTAssertEqual(snapshot.trackedNativeBufferBytes, 15)
        XCTAssertEqual(snapshot.processPhysicalFootprintBytes, 5_000_000_001)
        XCTAssertNil(snapshot.processAvailableBytes)
        XCTAssertEqual(snapshot.trainingGpuImageCacheHitRate, 0.75)

        native.trainingGpuImageCacheHits = 0
        native.trainingGpuImageCacheMisses = 0
        XCTAssertNil(TrainingMemorySnapshot(from: native).trainingGpuImageCacheHitRate)
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
