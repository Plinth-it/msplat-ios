import XCTest
import Msplat

final class TrainingPlanTests: XCTestCase {
    func testPlanMapsExactlyToCurrentNativeScheduler() throws {
        let plan = try TrainingPlan(
            inputDimensions: TrainingImageDimensions(width: 1_920, height: 1_440),
            inputDecodeScale: 2,
            iterationBudget: 2_000,
            stages: [
                TrainingResolutionStage(iterations: 1...1_000, downscaleFactor: 2),
                TrainingResolutionStage(iterations: 1_001...2_000, downscaleFactor: 1),
            ],
            targetSHDegree: 2,
            maximumGaussianCount: 750_000
        )

        let options = try plan.makeDatasetOptions(evalMode: true, testEvery: 10)
        XCTAssertEqual(options, DatasetOptions(
            downscaleFactor: 2,
            evalMode: true,
            testEvery: 10
        ))

        let config = try plan.makeTrainingConfig()
        XCTAssertEqual(config.iterations, 2_000)
        XCTAssertEqual(config.numDownscales, 1)
        XCTAssertEqual(config.resolutionSchedule, 1_001)
        XCTAssertEqual(config.shDegree, 2)
        XCTAssertEqual(config.shDegreeInterval, 666)
        XCTAssertEqual(config.downscaleFactor, 1)
        XCTAssertEqual(plan.maximumGaussianCount, 750_000)
        XCTAssertEqual(
            plan.decodedInputDimensions,
            try TrainingImageDimensions(width: 960, height: 720)
        )
        XCTAssertEqual(plan.resolvedStages, [
            ResolvedTrainingResolutionStage(
                iterations: 1...1_000,
                downscaleFactor: 2,
                dimensions: try TrainingImageDimensions(width: 480, height: 360)
            ),
            ResolvedTrainingResolutionStage(
                iterations: 1_001...2_000,
                downscaleFactor: 1,
                dimensions: try TrainingImageDimensions(width: 960, height: 720)
            ),
        ])

        XCTAssertEqual(
            plan.memoryEstimate.imageCacheBudgetBytes,
            TrainingMemoryEstimate.configuredNativeImageCacheBudgetBytes
        )
        XCTAssertEqual(
            plan.estimatedPeakMemory,
            plan.memoryEstimate.estimatedPeakMemory
        )

        let estimate = try plan.memoryEstimate(
            imageCacheBudgetBytes: TrainingMemoryEstimate.defaultIOSImageCacheBudgetBytes
        )
        XCTAssertEqual(
            estimate.imageCacheBudgetBytes,
            TrainingMemoryEstimate.defaultIOSImageCacheBudgetBytes
        )
        XCTAssertEqual(estimate.modelStorageBytes, 441_002_932)
        XCTAssertEqual(estimate.modelLifecycleBytes, 564_002_944)
        XCTAssertEqual(estimate.stages.map(\.pixelCount), [172_800, 691_200])
        XCTAssertEqual(estimate.stages.map(\.tileCount), [690, 2_700])
        XCTAssertEqual(
            estimate.stages.map(\.estimatedIntersectionCount),
            [12_000_000, 12_000_000]
        )
        XCTAssertEqual(
            estimate.stages.map(\.intersectionCapacity),
            [15_000_000, 15_000_000]
        )
        XCTAssertEqual(estimate.stages.map(\.chunkCount), [1, 1])
        XCTAssertEqual(
            estimate.stages.map(\.trainingCacheBytes),
            [962_276_648, 1_041_105_608]
        )
        XCTAssertEqual(estimate.peakTrainingCacheBytes, 1_041_105_608)
        XCTAssertEqual(estimate.largestImageCacheEntryBytes, 16_588_800)
        XCTAssertEqual(estimate.imageDecodeTransientBytes, 24_883_200)
        XCTAssertEqual(estimate.imageInsertionPeakBytes, 561_754_112)
        XCTAssertEqual(estimate.codeDerivedBytes, 2_166_862_664)
        XCTAssertEqual(estimate.recommendedHeadroomBytes, 433_372_533)
        XCTAssertEqual(estimate.estimatedPeakMemory, 2_600_235_197)
        let customEstimate = try plan.memoryEstimate(
            imageCacheBudgetBytes: 64 * 1_024 * 1_024
        )
        XCTAssertEqual(customEstimate.imageInsertionPeakBytes, 91_992_064)
        XCTAssertEqual(customEstimate.codeDerivedBytes, 1_697_100_616)
        XCTAssertEqual(customEstimate.recommendedHeadroomBytes, 339_420_124)
        XCTAssertEqual(customEstimate.estimatedPeakMemory, 2_036_520_740)
    }

    func testSingleCoarseStageDoesNotTransitionWithinBudget() throws {
        let plan = try TrainingPlan(
            inputDimensions: TrainingImageDimensions(width: 4_000, height: 3_000),
            inputDecodeScale: 1,
            iterationBudget: 500,
            stages: [
                TrainingResolutionStage(iterations: 1...500, downscaleFactor: 4),
            ],
            targetSHDegree: 0,
            maximumGaussianCount: 100_000
        )

        let config = try plan.makeTrainingConfig()
        XCTAssertEqual(config.numDownscales, 2)
        XCTAssertEqual(config.resolutionSchedule, 501)
        XCTAssertEqual(plan.resolvedStages[0].dimensions,
                       try TrainingImageDimensions(width: 1_000, height: 750))
    }

    func testThreeStagesRepresentNativeOneBasedBoundaryOffset() throws {
        let plan = try TrainingPlan(
            inputDimensions: TrainingImageDimensions(width: 1_600, height: 1_200),
            inputDecodeScale: 1,
            iterationBudget: 300,
            stages: [
                TrainingResolutionStage(iterations: 1...100, downscaleFactor: 4),
                TrainingResolutionStage(iterations: 101...201, downscaleFactor: 2),
                TrainingResolutionStage(iterations: 202...300, downscaleFactor: 1),
            ],
            targetSHDegree: 2,
            maximumGaussianCount: 100_000
        )

        let config = try plan.makeTrainingConfig()
        XCTAssertEqual(config.resolutionSchedule, 101)
        XCTAssertEqual(plan.resolvedStages.map(\.dimensions), [
            try TrainingImageDimensions(width: 400, height: 300),
            try TrainingImageDimensions(width: 800, height: 600),
            try TrainingImageDimensions(width: 1_600, height: 1_200),
        ])
    }

    func testPlanRetainsUncontrolledTrainingConfiguration() throws {
        var base = TrainingConfig()
        base.ssimWeight = 0.35
        base.refineEvery = 50
        base.stopDensifyAt = 900
        base.downscaleFactor = 8

        let plan = try TrainingPlan(
            inputDimensions: TrainingImageDimensions(width: 800, height: 600),
            inputDecodeScale: 4,
            iterationBudget: 1_000,
            stages: [
                TrainingResolutionStage(iterations: 1...1_000, downscaleFactor: 1),
            ],
            targetSHDegree: 1,
            maximumGaussianCount: 100_000
        )

        let config = try plan.makeTrainingConfig(startingFrom: base)
        XCTAssertEqual(config.ssimWeight, 0.35)
        XCTAssertEqual(config.refineEvery, 50)
        XCTAssertEqual(config.stopDensifyAt, 900)
        XCTAssertEqual(config.downscaleFactor, 1)
    }

    func testRejectsStageGapOverlapAndIncompleteCoverage() throws {
        let dimensions = try TrainingImageDimensions(width: 1_000, height: 800)

        let invalidStages: [[TrainingResolutionStage]] = [
            [
                try TrainingResolutionStage(iterations: 1...100, downscaleFactor: 2),
                try TrainingResolutionStage(iterations: 102...200, downscaleFactor: 1),
            ],
            [
                try TrainingResolutionStage(iterations: 1...100, downscaleFactor: 2),
                try TrainingResolutionStage(iterations: 100...200, downscaleFactor: 1),
            ],
            [
                try TrainingResolutionStage(iterations: 1...199, downscaleFactor: 1),
            ],
        ]

        for stages in invalidStages {
            XCTAssertThrowsError(
                try TrainingPlan(
                    inputDimensions: dimensions,
                    inputDecodeScale: 1,
                    iterationBudget: 200,
                    stages: stages,
                    targetSHDegree: 1,
                    maximumGaussianCount: 100_000
                )
            )
        }
    }

    func testRejectsStagesNativeSchedulerCannotRepresentExactly() throws {
        let dimensions = try TrainingImageDimensions(width: 1_000, height: 800)
        let stages = [
            try TrainingResolutionStage(iterations: 1...100, downscaleFactor: 4),
            try TrainingResolutionStage(iterations: 101...200, downscaleFactor: 2),
            try TrainingResolutionStage(iterations: 201...300, downscaleFactor: 1),
        ]

        XCTAssertThrowsError(
            try TrainingPlan(
                inputDimensions: dimensions,
                inputDecodeScale: 1,
                iterationBudget: 300,
                stages: stages,
                targetSHDegree: 2,
                maximumGaussianCount: 100_000
            )
        )
    }

    func testRejectsInvalidDimensionsScaleAndStageOutput() throws {
        XCTAssertThrowsError(try TrainingImageDimensions(width: 0, height: 100))
        let dimensions = try TrainingImageDimensions(width: 32, height: 32)
        let stage = try TrainingResolutionStage(
            iterations: 1...10,
            downscaleFactor: 1
        )

        XCTAssertThrowsError(
            try TrainingPlan(
                inputDimensions: dimensions,
                inputDecodeScale: .nan,
                iterationBudget: 10,
                stages: [stage],
                targetSHDegree: 0,
                maximumGaussianCount: 100_000
            )
        )

        let oversizedStage = try TrainingResolutionStage(
            iterations: 1...10,
            downscaleFactor: 64
        )
        XCTAssertThrowsError(
            try TrainingPlan(
                inputDimensions: dimensions,
                inputDecodeScale: 1,
                iterationBudget: 10,
                stages: [oversizedStage],
                targetSHDegree: 0,
                maximumGaussianCount: 100_000
            )
        )
    }

    func testRejectsInvalidMaximumGaussianCountAndSHBudget() throws {
        let dimensions = try TrainingImageDimensions(width: 640, height: 480)
        let stage = try TrainingResolutionStage(
            iterations: 1...4,
            downscaleFactor: 1
        )

        XCTAssertThrowsError(
            try TrainingPlan(
                inputDimensions: dimensions,
                inputDecodeScale: 1,
                iterationBudget: 4,
                stages: [stage],
                targetSHDegree: 4,
                maximumGaussianCount: 100_000
            )
        )
        XCTAssertThrowsError(
            try TrainingPlan(
                inputDimensions: dimensions,
                inputDecodeScale: 1,
                iterationBudget: 4,
                stages: [stage],
                targetSHDegree: 0,
                maximumGaussianCount: 0
            )
        )
    }

    func testMemoryEstimateUsesMinimumHeadroomAndRejectsInvalidCacheBudget() throws {
        let plan = try TrainingPlan(
            inputDimensions: TrainingImageDimensions(width: 32, height: 32),
            inputDecodeScale: 1,
            iterationBudget: 10,
            stages: [
                TrainingResolutionStage(iterations: 1...10, downscaleFactor: 1),
            ],
            targetSHDegree: 0,
            maximumGaussianCount: 1
        )

        XCTAssertEqual(
            plan.memoryEstimate.recommendedHeadroomBytes,
            128 * 1_024 * 1_024
        )
        XCTAssertThrowsError(try plan.memoryEstimate(imageCacheBudgetBytes: 0))
    }

    func testIntersectionEstimateScalesMeasuredResolutionBaseline() throws {
        let plan = try TrainingPlan(
            inputDimensions: TrainingImageDimensions(width: 1_920, height: 1_440),
            inputDecodeScale: 1,
            iterationBudget: 1,
            stages: [
                TrainingResolutionStage(iterations: 1...1, downscaleFactor: 1),
            ],
            targetSHDegree: 0,
            maximumGaussianCount: 1
        )

        let stage = plan.memoryEstimate.stages[0]
        XCTAssertEqual(stage.tileCount, 10_800)
        XCTAssertEqual(stage.estimatedIntersectionCount, 64)
        XCTAssertEqual(stage.intersectionCapacity, 4_160)
    }

    func testRejectsNativeMemoryIndexOverflows() throws {
        XCTAssertThrowsError(
            try TrainingImageDimensions(width: 65_536, height: 65_536)
        )

        // The exact pipeline has no tileCount * 2,048 layout, so large legal
        // tile grids are planning inputs rather than false index overflows.
        let formerlyRejectedTileDimensions = try TrainingImageDimensions(
            width: 16_384,
            height: 16_384
        )
        let largeTilePlan = try TrainingPlan(
            inputDimensions: formerlyRejectedTileDimensions,
            inputDecodeScale: 1,
            iterationBudget: 10,
            stages: [
                TrainingResolutionStage(iterations: 1...10, downscaleFactor: 1),
            ],
            targetSHDegree: 0,
            maximumGaussianCount: 2_048
        )
        XCTAssertEqual(largeTilePlan.memoryEstimate.stages[0].tileCount,
                       1_048_576)
        XCTAssertGreaterThan(
            largeTilePlan.memoryEstimate.stages[0].intersectionCapacity,
            2_048 * 2_048
        )

        let smallDimensions = try TrainingImageDimensions(width: 64, height: 64)
        let estimateBeyondNativeRange = try TrainingPlan(
            inputDimensions: smallDimensions,
            inputDecodeScale: 1,
            iterationBudget: 10,
            stages: [
                TrainingResolutionStage(iterations: 1...10, downscaleFactor: 1),
            ],
            targetSHDegree: 0,
            maximumGaussianCount: 134_217_728
        )
        XCTAssertEqual(
            estimateBeyondNativeRange.memoryEstimate.stages[0]
                .estimatedIntersectionCount,
            2_147_483_648
        )
        XCTAssertEqual(
            estimateBeyondNativeRange.memoryEstimate.stages[0]
                .intersectionCapacity,
            Int64(Int32.max)
        )
        XCTAssertThrowsError(
            try TrainingPlan(
                inputDimensions: smallDimensions,
                inputDecodeScale: 1,
                iterationBudget: 10,
                stages: [
                    TrainingResolutionStage(iterations: 1...10, downscaleFactor: 1),
                ],
                targetSHDegree: 4,
                maximumGaussianCount: 60_000_000
            )
        )
    }

    func testRejectsMemoryArithmeticOverflow() throws {
        let plan = try TrainingPlan(
            inputDimensions: TrainingImageDimensions(width: 16, height: 16),
            inputDecodeScale: 1,
            iterationBudget: 10,
            stages: [
                TrainingResolutionStage(iterations: 1...10, downscaleFactor: 1),
            ],
            targetSHDegree: 0,
            maximumGaussianCount: 1
        )
        XCTAssertThrowsError(
            try plan.memoryEstimate(imageCacheBudgetBytes: Int64.max)
        )
    }
}
