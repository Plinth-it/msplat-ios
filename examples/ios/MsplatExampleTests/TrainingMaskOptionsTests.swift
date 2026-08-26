import CoreGraphics
import CoreVideo
import Foundation
import Msplat
@testable import MsplatExample
@preconcurrency import RealityKit
import simd
import UIKit
import UniformTypeIdentifiers
import XCTest

final class TrainingMaskOptionsTests: XCTestCase {
    private var temporaryDirectory: URL?

    override func setUpWithError() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "msplat-example-masks-\(UUID().uuidString)", directoryHint: .isDirectory)
        temporaryDirectory = directory
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try Data().write(to: directory.appending(path: "cameras.txt"))
    }

    override func tearDownWithError() throws {
        if let temporaryDirectory {
            try FileManager.default.removeItem(at: temporaryDirectory)
        }
        temporaryDirectory = nil
    }

    func testDatasetFolderCountsOnlyRegularMaskCandidatesRecursively() throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let masks = temporaryDirectory.appending(path: "masks", directoryHint: .isDirectory)
        let nested = masks.appending(path: "nested", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data([0]).write(to: masks.appending(path: "frame-a.png"))
        try Data([255]).write(to: nested.appending(path: "frame-b.png"))
        try Data([127]).write(to: masks.appending(path: ".hidden.png"))

        _ = try XCTUnwrap(DatasetFolder(picked: temporaryDirectory))

        XCTAssertEqual(
            DatasetFolder.countTrainingMaskCandidates(at: temporaryDirectory),
            3
        )
    }

    func testDatasetFolderWithoutMasksHasNoCandidates() throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        _ = try XCTUnwrap(DatasetFolder(picked: temporaryDirectory))

        XCTAssertEqual(
            DatasetFolder.countTrainingMaskCandidates(at: temporaryDirectory),
            0
        )
    }

    func testDatasetFolderFindsCaseInsensitiveMasksDirectory() throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let masks = temporaryDirectory.appending(
            path: "assets/MaSkS",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: masks, withIntermediateDirectories: true)
        try Data([255]).write(to: masks.appending(path: "frame.png"))

        _ = try XCTUnwrap(DatasetFolder(picked: temporaryDirectory))

        XCTAssertEqual(
            DatasetFolder.countTrainingMaskCandidates(at: temporaryDirectory),
            1
        )
    }

    func testDatasetFolderBookmarkRoundTrip() throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let suiteName = "msplat-example-bookmark-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let selected = try XCTUnwrap(DatasetFolder(picked: temporaryDirectory))

        try selected.persistAsLastPicked(in: defaults)
        let restored = try XCTUnwrap(
            DatasetFolder.restoreLastPicked(from: defaults)
        )

        XCTAssertEqual(restored.url.standardizedFileURL,
                       temporaryDirectory.standardizedFileURL)
        XCTAssertEqual(restored.kind, .colmap)
    }

    func testBenchmarkDatasetStaysInsideDocumentsDirectory() throws {
        let root = try XCTUnwrap(temporaryDirectory)
        let documents = root.appending(path: "Documents", directoryHint: .isDirectory)
        let dataset = documents.appending(path: "egg", directoryHint: .isDirectory)
        let outside = root.appending(path: "outside", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try Data().write(to: dataset.appending(path: "cameras.txt"))
        try Data().write(to: outside.appending(path: "cameras.txt"))
        let restored = try XCTUnwrap(
            DatasetFolder.benchmarkDatasetFromDocuments(
                environment: ["MSPLAT_BENCHMARK_DATASET": "egg"],
                documentsDirectory: documents
            )
        )

        XCTAssertEqual(restored.url.standardizedFileURL,
                       dataset.standardizedFileURL)
        XCTAssertNil(
            DatasetFolder.benchmarkDatasetFromDocuments(
                environment: ["MSPLAT_BENCHMARK_DATASET": "../outside"],
                documentsDirectory: documents
            )
        )

        let link = documents.appending(path: "linked-outside")
        try FileManager.default.createSymbolicLink(
            at: link,
            withDestinationURL: outside
        )
        XCTAssertNil(
            DatasetFolder.benchmarkDatasetFromDocuments(
                environment: ["MSPLAT_BENCHMARK_DATASET": "linked-outside"],
                documentsDirectory: documents
            )
        )
    }

    func testDatasetFolderPrefersNerfstudioOverColmap() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        let pointCloud = directory.appending(path: "cloud/sparse.ply")
        try writePointPLY(count: 4, to: pointCloud)
        try writeNerfstudioManifest(
            to: directory,
            plyFilePath: "cloud/sparse.ply"
        )

        let folder = try XCTUnwrap(DatasetFolder(picked: directory))

        XCTAssertEqual(folder.kind, .nerfstudio)
        XCTAssertEqual(folder.summary, "Nerfstudio, 1 images")
        XCTAssertFalse(folder.supportsAutomaticTrainingMaskDiscovery)
        XCTAssertEqual(
            try DatasetFolder.initialSparsePointCount(at: directory),
            4
        )
    }

    func testNerfstudioPointCountFallsBackToPoints3DPLY() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        try writePointPLY(
            count: 3,
            to: directory.appending(path: "points3D.ply")
        )
        try writeNerfstudioManifest(to: directory, plyFilePath: nil)

        XCTAssertEqual(
            try DatasetFolder.initialSparsePointCount(at: directory),
            3
        )
    }

    func testNerfstudioPreflightReadsOnlyReferencedImageDimensions() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        let images = directory.appending(path: "frames", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: images,
            withIntermediateDirectories: true
        )
        try writePNG(
            width: 12,
            height: 7,
            to: images.appending(path: "first.png")
        )
        try writePNG(
            width: 30,
            height: 20,
            to: images.appending(path: "unreferenced.png")
        )
        try writePointPLY(
            count: 2,
            to: directory.appending(path: "points3D.ply")
        )
        try writeNerfstudioManifest(
            to: directory,
            plyFilePath: nil,
            framePaths: ["./frames/first"]
        )

        let dimensions = try DatasetFolder.maximumSourceDimensions(at: directory)

        XCTAssertEqual(dimensions.width, 12)
        XCTAssertEqual(dimensions.height, 7)
    }

    func testNerfstudioPreflightRejectsPartialMaskCoverage() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        let images = directory.appending(path: "images", directoryHint: .isDirectory)
        let masks = directory.appending(path: "masks", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: images,
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: masks,
            withIntermediateDirectories: true
        )
        try writePNG(width: 12, height: 7, to: images.appending(path: "a.png"))
        try writePNG(width: 12, height: 7, to: images.appending(path: "b.png"))
        try writePNG(width: 12, height: 7, to: masks.appending(path: "a.png"))
        try writeNerfstudioManifest(
            to: directory,
            plyFilePath: nil,
            framePaths: ["images/a.png", "images/b.png"],
            maskPaths: ["masks/a.png", nil]
        )

        XCTAssertThrowsError(
            try DatasetFolder.maximumSourceDimensions(at: directory)
        ) { error in
            XCTAssertTrue(
                error.localizedDescription.contains(
                    "present for every frame or no frames"
                )
            )
        }
    }

    func testNerfstudioPreflightRejectsMaskDimensionMismatch() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        let images = directory.appending(path: "images", directoryHint: .isDirectory)
        let masks = directory.appending(path: "masks", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: images,
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: masks,
            withIntermediateDirectories: true
        )
        try writePNG(
            width: 12,
            height: 7,
            to: images.appending(path: "frame.png")
        )
        try writePNG(
            width: 11,
            height: 7,
            to: masks.appending(path: "frame.png")
        )
        try writeNerfstudioManifest(
            to: directory,
            plyFilePath: nil,
            maskPaths: ["masks/frame.png"]
        )
        let folder = try XCTUnwrap(DatasetFolder(picked: directory))

        XCTAssertTrue(folder.hasNerfstudioTrainingMasks)
        XCTAssertThrowsError(
            try DatasetFolder.maximumSourceDimensions(at: directory)
        ) { error in
            XCTAssertTrue(
                error.localizedDescription.contains("dimensions do not match")
            )
        }
    }

    func testNerfstudioFolderCreatesNativeSession() async throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        let images = directory.appending(path: "images", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: images,
            withIntermediateDirectories: true
        )
        try writePNG(
            width: 12,
            height: 7,
            to: images.appending(path: "frame.png")
        )
        try writePointPLY(
            count: 4,
            to: directory.appending(path: "points3D.ply")
        )
        try writeNerfstudioManifest(to: directory, plyFilePath: nil)

        let session = try await MsplatSession(
            datasetURL: directory,
            maximumGaussianCount: 4
        )
        do {
            let trainingCameraCount = try await session.numTrain
            XCTAssertEqual(trainingCameraCount, 1)
            try await session.close()
        } catch {
            try? await session.close()
            throw error
        }
    }

    func testNerfstudioPreflightRejectsFisheyeCameraModel() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        try writePointPLY(
            count: 2,
            to: directory.appending(path: "points3D.ply")
        )
        try writeNerfstudioManifest(
            to: directory,
            plyFilePath: nil,
            cameraModel: "OPENCV_FISHEYE"
        )

        XCTAssertThrowsError(
            try DatasetFolder.initialSparsePointCount(at: directory)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("OPENCV_FISHEYE"))
        }
    }

    func testNerfstudioPreflightRequiresInitialPointCloud() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        try writeNerfstudioManifest(to: directory, plyFilePath: nil)

        XCTAssertThrowsError(
            try DatasetFolder.initialSparsePointCount(at: directory)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("needs a non-empty PLY"))
        }
    }

    func testNerfstudioPreflightRejectsPointCloudOutsideSelectedFolder() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        try writeNerfstudioManifest(
            to: directory,
            plyFilePath: "../outside.ply"
        )

        XCTAssertThrowsError(
            try DatasetFolder.initialSparsePointCount(at: directory)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("selected folder"))
        }
    }

    func testTrainingPlanAccountsForSelectedMaskDiscovery() throws {
        let dimensions = try TrainingImageDimensions(width: 1_920, height: 1_440)
        let unmasked = try TrainingSession.makePlan(
            sourceDimensions: dimensions,
            initialGaussianCount: 100_000,
            steps: 2_000,
            profile: .preview
        )
        let masked = try TrainingSession.makePlan(
            sourceDimensions: dimensions,
            initialGaussianCount: 100_000,
            steps: 2_000,
            profile: .preview,
            includesTrainingMasks: true
        )

        XCTAssertFalse(unmasked.includesTrainingMasks)
        XCTAssertTrue(masked.includesTrainingMasks)
        XCTAssertGreaterThan(masked.estimatedPeakMemory, unmasked.estimatedPeakMemory)
    }

    @MainActor
    func testDiscoveredMasksDefaultToTransparentTreatment() {
        let session = TrainingSession()

        XCTAssertEqual(session.trainingMaskMode, .transparent)
    }

    @MainActor
    func testCameraPoseRefinementDefaultsOff() throws {
        let session = TrainingSession()
        let config = try TrainingSession.makeTrainingConfig(
            trainingMaskMode: .transparent,
            keepCrs: true,
            benchmark: nil
        )

        XCTAssertFalse(session.refineCameraPosesEnabled)
        XCTAssertFalse(config.refineCameraPoses)
    }

    func testBinarySparsePointCountTakesPriorityOverText() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        try writeBinaryPointHeader(count: 2, to: directory)
        try "1 0 0 0 0 0 0 0\n2 0 0 0 0 0 0 0\n3 0 0 0 0 0 0 0\n"
            .write(
                to: directory.appending(path: "points3D.txt"),
                atomically: true,
                encoding: .utf8
            )

        XCTAssertEqual(
            try DatasetFolder.initialSparsePointCount(at: directory),
            2
        )
    }

    func testTextSparsePointCountIgnoresCommentsAndBlankLines() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        try "# comment\r\n\r\n  # indented comment\n1 point\n2 point"
            .write(
                to: directory.appending(path: "points3D.txt"),
                atomically: true,
                encoding: .utf8
            )

        XCTAssertEqual(
            try DatasetFolder.initialSparsePointCount(at: directory),
            2
        )
    }

    func testPlySparsePointCountUsesVertexHeader() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        try """
        ply
        format ascii 1.0
        element vertex 3
        property float x
        property float y
        property float z
        end_header
        0 0 0
        1 1 1
        2 2 2
        """.write(
            to: directory.appending(path: "points3D.ply"),
            atomically: true,
            encoding: .utf8
        )

        XCTAssertEqual(
            try DatasetFolder.initialSparsePointCount(at: directory),
            3
        )
    }

    func testMalformedBinaryPointCountDoesNotFallBackToText() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        try writeBinaryPointHeader(count: 2, to: directory, recordCount: 0)
        try "1 point\n2 point\n"
            .write(
                to: directory.appending(path: "points3D.txt"),
                atomically: true,
                encoding: .utf8
            )

        XCTAssertThrowsError(
            try DatasetFolder.initialSparsePointCount(at: directory)
        )
    }

    func testPreviewPlanRaisesCapToPreserveEggInitialPopulation() throws {
        let dimensions = try TrainingImageDimensions(width: 3_024, height: 4_032)
        let plan = try TrainingSession.makePlan(
            sourceDimensions: dimensions,
            initialGaussianCount: 313_214,
            steps: 2_000,
            profile: .preview,
            includesTrainingMasks: true
        )
        let iOSMemory = try plan.memoryEstimate(
            imageCacheBudgetBytes: TrainingMemoryEstimate.defaultIOSImageCacheBudgetBytes
        )

        XCTAssertEqual(plan.maximumGaussianCount, 313_214)
        XCTAssertEqual(plan.resolvedStages.first?.dimensions.width, 1_200)
        XCTAssertEqual(plan.resolvedStages.first?.dimensions.height, 1_600)
        XCTAssertEqual(iOSMemory.intersectionAttributeStorage, .gather)
        XCTAssertEqual(iOSMemory.estimatedPeakMemory, 1_579_190_943)
        XCTAssertLessThan(
            iOSMemory.estimatedPeakMemory,
            3_351 * 1_024 * 1_024
        )
    }

    func testPreviewPlanKeepsDefaultCapForSmallerDataset() throws {
        let dimensions = try TrainingImageDimensions(width: 1_920, height: 1_440)
        let plan = try TrainingSession.makePlan(
            sourceDimensions: dimensions,
            initialGaussianCount: 100_000,
            steps: 2_000,
            profile: .preview
        )

        XCTAssertEqual(plan.maximumGaussianCount, 250_000)
    }

    func testAppPreflightAccountsForTwoRGBA8PreviewSurfaces() throws {
        let dimensions = try TrainingImageDimensions(width: 1_920, height: 1_440)
        let plan = try TrainingSession.makePlan(
            sourceDimensions: dimensions,
            initialGaussianCount: 100_000,
            steps: 2_000,
            profile: .preview
        )

        let appPeak = try TrainingSession.appEstimatedPeakMemory(for: plan)
        let largestStage = try XCTUnwrap(
            plan.resolvedStages.max {
                $0.dimensions.width * $0.dimensions.height
                    < $1.dimensions.width * $1.dimensions.height
            }
        )
        let expectedPreviewBytes = Int64(
            largestStage.dimensions.width * largestStage.dimensions.height * 2 * 4
        )

        XCTAssertEqual(appPeak - plan.estimatedPeakMemory, expectedPreviewBytes)
    }

    func testPreviewCadenceHasOneHundredStepFloor() {
        XCTAssertEqual(TrainingSession.previewInterval(for: 200), 100)
        XCTAssertEqual(TrainingSession.previewInterval(for: 2_000), 100)
        XCTAssertEqual(TrainingSession.previewInterval(for: 20_000), 1_000)
    }

    func testBenchmarkConfigurationDefaultsAndOverrides() throws {
        XCTAssertNil(TrainingBenchmarkConfiguration.requested(environment: [:]))

        let defaults = try XCTUnwrap(
            TrainingBenchmarkConfiguration.requested(environment: [
                "MSPLAT_BENCHMARK": "1",
            ])
        )
        XCTAssertEqual(defaults.label, "baseline")
        XCTAssertEqual(defaults.warmupIterations, 50)
        XCTAssertEqual(defaults.measuredIterations, 300)
        XCTAssertEqual(defaults.totalIterations, 350)
        XCTAssertTrue(defaults.fixedPopulation)
        XCTAssertNil(defaults.growthMaximumGaussianCount)

        let overridden = try XCTUnwrap(
            TrainingBenchmarkConfiguration.requested(environment: [
                "MSPLAT_BENCHMARK": "1",
                "MSPLAT_BENCHMARK_LABEL": "gather",
                "MSPLAT_BENCHMARK_WARMUP": "10",
                "MSPLAT_BENCHMARK_MEASURED": "25",
            ])
        )
        XCTAssertEqual(overridden.label, "gather")
        XCTAssertEqual(overridden.warmupIterations, 10)
        XCTAssertEqual(overridden.measuredIterations, 25)
        XCTAssertEqual(overridden.totalIterations, 35)

        let tooShort = try XCTUnwrap(
            TrainingBenchmarkConfiguration.requested(environment: [
                "MSPLAT_BENCHMARK": "1",
                "MSPLAT_BENCHMARK_WARMUP": "0",
                "MSPLAT_BENCHMARK_MEASURED": "1",
            ])
        )
        XCTAssertEqual(tooShort.warmupIterations, 50)
        XCTAssertEqual(tooShort.measuredIterations, 300)
    }

    func testGrowthBenchmarkConfigurationAndPlan() throws {
        let benchmark = try XCTUnwrap(
            TrainingBenchmarkConfiguration.requested(environment: [
                "MSPLAT_BENCHMARK": "1",
                "MSPLAT_BENCHMARK_LABEL": "growth",
                "MSPLAT_BENCHMARK_WARMUP": "10",
                "MSPLAT_BENCHMARK_MEASURED": "90",
                "MSPLAT_BENCHMARK_GROWTH": "1",
                "MSPLAT_BENCHMARK_GROWTH_MAX_GAUSSIANS": "400000",
            ])
        )
        XCTAssertFalse(benchmark.fixedPopulation)
        XCTAssertEqual(benchmark.growthMaximumGaussianCount, 400_000)
        XCTAssertEqual(benchmark.maximumMissingMeasuredIterations, 1)
        XCTAssertEqual(
            try benchmark.validatedGrowthMaximumGaussianCount(
                initialGaussianCount: 100_000
            ),
            400_000
        )

        let config = try TrainingSession.makeTrainingConfig(
            trainingMaskMode: .coverage,
            benchmark: benchmark
        )
        XCTAssertEqual(config.warmupLength, 0)
        XCTAssertEqual(config.refineEvery, 25)
        XCTAssertEqual(config.resetAlphaEvery, 100)
        XCTAssertEqual(config.densifyGradThresh, 0)
        XCTAssertEqual(config.stopDensifyAt, 101)
        XCTAssertFalse(
            TrainingSession.isDensificationStep(
                25,
                config: config,
                cameraCount: 10
            )
        )
        XCTAssertTrue(
            TrainingSession.isDensificationStep(
                50,
                config: config,
                cameraCount: 10
            )
        )
        XCTAssertFalse(
            TrainingSession.isDensificationStep(
                51,
                config: config,
                cameraCount: 10
            )
        )

        let dimensions = try TrainingImageDimensions(width: 1_920, height: 1_440)
        let configuredPlan = try TrainingSession.makePlan(
            sourceDimensions: dimensions,
            initialGaussianCount: 100_000,
            steps: benchmark.totalIterations,
            profile: .preview,
            maximumGaussianCountOverride: benchmark.growthMaximumGaussianCount
        )
        XCTAssertEqual(configuredPlan.maximumGaussianCount, 400_000)
        let exactOverridePlan = try TrainingSession.makePlan(
            sourceDimensions: dimensions,
            initialGaussianCount: 100_000,
            steps: benchmark.totalIterations,
            profile: .preview,
            maximumGaussianCountOverride: 200_000
        )
        XCTAssertEqual(exactOverridePlan.maximumGaussianCount, 200_000)
    }

    func testGrowthBenchmarkRejectsMissingOrNonIncreasingCap() throws {
        let missingCap = try XCTUnwrap(
            TrainingBenchmarkConfiguration.requested(environment: [
                "MSPLAT_BENCHMARK": "1",
                "MSPLAT_BENCHMARK_GROWTH": "1",
            ])
        )
        XCTAssertFalse(missingCap.fixedPopulation)
        XCTAssertThrowsError(
            try missingCap.validatedGrowthMaximumGaussianCount(
                initialGaussianCount: 100_000
            )
        )

        let nonIncreasingCap = try XCTUnwrap(
            TrainingBenchmarkConfiguration.requested(environment: [
                "MSPLAT_BENCHMARK": "1",
                "MSPLAT_BENCHMARK_GROWTH": "1",
                "MSPLAT_BENCHMARK_GROWTH_MAX_GAUSSIANS": "100000",
            ])
        )
        XCTAssertThrowsError(
            try nonIncreasingCap.validatedGrowthMaximumGaussianCount(
                initialGaussianCount: 100_000
            )
        )

        let noGrowthHeadroom = try XCTUnwrap(
            TrainingBenchmarkConfiguration.requested(environment: [
                "MSPLAT_BENCHMARK": "1",
                "MSPLAT_BENCHMARK_GROWTH": "1",
                "MSPLAT_BENCHMARK_GROWTH_MAX_GAUSSIANS": "125000",
            ])
        )
        XCTAssertThrowsError(
            try noGrowthHeadroom.validatedGrowthMaximumGaussianCount(
                initialGaussianCount: 100_000
            )
        )
    }

    func testFixedPopulationBenchmarkDisablesDensification() throws {
        let benchmark = try XCTUnwrap(
            TrainingBenchmarkConfiguration.requested(environment: [
                "MSPLAT_BENCHMARK": "1",
            ])
        )

        let config = try TrainingSession.makeTrainingConfig(
            trainingMaskMode: .transparent,
            refineCameraPoses: true,
            benchmark: benchmark
        )

        XCTAssertEqual(config.trainingMaskMode, .transparent)
        XCTAssertFalse(config.refineCameraPoses)
        XCTAssertEqual(config.cameraPoseConditioning, .raw)
        XCTAssertEqual(config.stopDensifyAt, 0)
        XCTAssertEqual(config.warmupLength, TrainingConfig().warmupLength)
        XCTAssertEqual(config.refineEvery, TrainingConfig().refineEvery)
        XCTAssertEqual(config.resetAlphaEvery, TrainingConfig().resetAlphaEvery)
        XCTAssertEqual(benchmark.maximumMissingMeasuredIterations, 0)
    }

    func testCapturedTrainingConfigPreservesMetricCoordinates() throws {
        let config = try TrainingSession.makeTrainingConfig(
            trainingMaskMode: .transparent,
            keepCrs: true,
            benchmark: nil
        )

        XCTAssertTrue(config.keepCrs)
        XCTAssertEqual(config.trainingMaskMode, .transparent)
        XCTAssertFalse(config.refinePhotometricGains)
        XCTAssertFalse(config.refineCameraPoses)
        XCTAssertEqual(config.cameraPoseConditioning, .raw)
    }

    func testCapturedTrainingConfigEnablesPoseRefinementWhenOptedIn() throws {
        let config = try TrainingSession.makeTrainingConfig(
            trainingMaskMode: .transparent,
            keepCrs: true,
            refineCameraPoses: true,
            cameraPoseConditioning: .camP,
            benchmark: nil
        )

        XCTAssertTrue(config.keepCrs)
        XCTAssertTrue(config.refineCameraPoses)
        XCTAssertEqual(config.cameraPoseConditioning, .camP)
    }

    func testPoseRefinementBudgetRejectsInsufficientPostWarmupCameraPass() throws {
        var config = try TrainingSession.makeTrainingConfig(
            trainingMaskMode: .transparent,
            refineCameraPoses: true,
            benchmark: nil
        )
        let cameraCount = 7
        let requirement = try TrainingSession.poseRefinementBudgetRequirement(
            config: config,
            trainingCameraCount: cameraCount
        )
        config.iterations = Int32(requirement.minimumIterations - 1)

        XCTAssertEqual(requirement.warmupIterations, 500)
        XCTAssertEqual(requirement.postWarmupCameraVisits, 11)
        XCTAssertEqual(requirement.minimumIterations, 511)
        XCTAssertThrowsError(try TrainingSession.validatePoseRefinementBudget(
            config: config,
            trainingCameraCount: cameraCount
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("at least 511 iterations"))
            XCTAssertTrue(
                error.localizedDescription.contains("one full post-warm-up shuffled pass")
            )
        }
    }

    func testPoseRefinementBudgetAcceptsWarmupPlusFullCameraPass() throws {
        var config = try TrainingSession.makeTrainingConfig(
            trainingMaskMode: .transparent,
            refineCameraPoses: true,
            benchmark: nil
        )
        let cameraCount = 7
        let requirement = try TrainingSession.poseRefinementBudgetRequirement(
            config: config,
            trainingCameraCount: cameraCount
        )
        config.iterations = Int32(requirement.minimumIterations)

        XCTAssertNoThrow(try TrainingSession.validatePoseRefinementBudget(
            config: config,
            trainingCameraCount: cameraCount
        ))
    }

    func testRefinedCaptureManifestWritesSiblingAndPreservesRawFields() throws {
        let rootURL = try XCTUnwrap(temporaryDirectory)
        let pointCloudPath = "pointcloud/lidar_colored.ply"
        let first = makeCaptureFrameRecord(
            id: "frame_000001",
            imagePath: "images/frame_000001.png",
            maskPath: "masks/frame_000001.png"
        )
        let second = makeCaptureFrameRecord(
            id: "frame_000002",
            imagePath: "images/frame_000002.png",
            maskPath: nil
        )
        let identity: [[Float]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        let sourceObject: [String: Any] = [
            "camera_model": "OPENCV",
            "ply_file_path": pointCloudPath,
            "capture_note": "preserve me",
            "frames": [
                [
                    "file_path": second.imagePath,
                    "w": 1_920,
                    "h": 1_440,
                    "fl_x": 1_200.0,
                    "fl_y": 1_210.0,
                    "cx": 960.0,
                    "cy": 720.0,
                    "transform_matrix": identity,
                ],
                [
                    "file_path": first.imagePath,
                    "mask_path": try XCTUnwrap(first.maskPath),
                    "w": 1_920,
                    "h": 1_440,
                    "fl_x": 1_200.0,
                    "fl_y": 1_210.0,
                    "cx": 960.0,
                    "cy": 720.0,
                    "transform_matrix": identity,
                ],
            ],
        ]
        let sourceData = try JSONSerialization.data(
            withJSONObject: sourceObject,
            options: [.prettyPrinted, .sortedKeys]
        )
        let sourceURL = rootURL.appending(path: "transforms.json")
        try sourceData.write(to: sourceURL)

        var firstPose = first.cameraToWorld
        firstPose[3] = 0.25
        var secondPose = second.cameraToWorld
        secondPose[7] = -0.5
        let output = try RefinedCaptureManifestExporter.write(
            rootURL: rootURL,
            frames: [first, second],
            pointCloudPath: pointCloudPath,
            corrections: [
                CapturePoseCorrection(
                    frameID: second.id,
                    isAnchor: false,
                    optimizerStepCount: 1,
                    correctedCameraToWorld: try CameraPose(elements: secondPose),
                    translationNorm: 0.004,
                    rotationNorm: 0.02
                ),
                CapturePoseCorrection(
                    frameID: first.id,
                    isAnchor: true,
                    optimizerStepCount: 0,
                    correctedCameraToWorld: try CameraPose(elements: firstPose),
                    translationNorm: 0.003,
                    rotationNorm: 0.01
                ),
            ]
        )

        XCTAssertEqual(try Data(contentsOf: sourceURL), sourceData)
        XCTAssertEqual(output.url.lastPathComponent, "transforms_refined.json")
        let refinedData = try Data(contentsOf: output.url)
        let refinedObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: refinedData) as? [String: Any]
        )
        XCTAssertEqual(refinedObject["ply_file_path"] as? String, pointCloudPath)
        XCTAssertEqual(refinedObject["capture_note"] as? String, "preserve me")
        let sourceFrames = try XCTUnwrap(sourceObject["frames"] as? [[String: Any]])
        let refinedFrames = try XCTUnwrap(refinedObject["frames"] as? [[String: Any]])
        let sourceByPath = Dictionary(uniqueKeysWithValues: sourceFrames.compactMap { frame in
            (frame["file_path"] as? String).map { ($0, frame) }
        })
        let refinedByPath = Dictionary(uniqueKeysWithValues: refinedFrames.compactMap { frame in
            (frame["file_path"] as? String).map { ($0, frame) }
        })
        for path in [first.imagePath, second.imagePath] {
            var sourceFrame = try XCTUnwrap(sourceByPath[path])
            var refinedFrame = try XCTUnwrap(refinedByPath[path])
            sourceFrame.removeValue(forKey: "transform_matrix")
            refinedFrame.removeValue(forKey: "transform_matrix")
            XCTAssertTrue(
                NSDictionary(dictionary: sourceFrame).isEqual(
                    NSDictionary(dictionary: refinedFrame)
                )
            )
        }
        let firstMatrix = try XCTUnwrap(
            refinedByPath[first.imagePath]?["transform_matrix"] as? [[NSNumber]]
        )
        let secondMatrix = try XCTUnwrap(
            refinedByPath[second.imagePath]?["transform_matrix"] as? [[NSNumber]]
        )
        XCTAssertEqual(firstMatrix[0][3].floatValue, 0.25, accuracy: 1e-6)
        XCTAssertEqual(secondMatrix[1][3].floatValue, -0.5, accuracy: 1e-6)
        XCTAssertEqual(output.summary.cameraCount, 2)
        XCTAssertEqual(output.summary.maximumTranslationMeters, 0.004, accuracy: 1e-6)
        XCTAssertEqual(
            output.summary.rmsTranslationMeters,
            Float(sqrt((0.003 * 0.003 + 0.004 * 0.004) / 2)),
            accuracy: 1e-6
        )
        XCTAssertEqual(output.summary.maximumRotationRadians, 0.02, accuracy: 1e-6)
        XCTAssertEqual(
            output.summary.rmsRotationRadians,
            Float(sqrt((0.01 * 0.01 + 0.02 * 0.02) / 2)),
            accuracy: 1e-6
        )
    }

    func testRefinedCaptureManifestRejectsDuplicateAndMissingFrameIDs() throws {
        let frame = makeCaptureFrameRecord(
            id: "frame_000001",
            imagePath: "images/frame_000001.png",
            maskPath: nil
        )
        let other = makeCaptureFrameRecord(
            id: "frame_000002",
            imagePath: "images/frame_000002.png",
            maskPath: nil
        )
        let sourceData = try JSONSerialization.data(withJSONObject: [
            "ply_file_path": "pointcloud/lidar_colored.ply",
            "frames": [
                ["file_path": frame.imagePath, "transform_matrix": identityTransform],
                ["file_path": other.imagePath, "transform_matrix": identityTransform],
            ],
        ])
        let correction = CapturePoseCorrection(
            frameID: frame.id,
            isAnchor: false,
            optimizerStepCount: 1,
            correctedCameraToWorld: try CameraPose(elements: frame.cameraToWorld),
            translationNorm: 0,
            rotationNorm: 0
        )

        XCTAssertThrowsError(try RefinedCaptureManifestExporter.makeManifestData(
            sourceData: sourceData,
            frames: [frame, other],
            pointCloudPath: "pointcloud/lidar_colored.ply",
            corrections: [correction, correction]
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("duplicate frame ID"))
        }
        XCTAssertThrowsError(try RefinedCaptureManifestExporter.makeManifestData(
            sourceData: sourceData,
            frames: [frame, other],
            pointCloudPath: "pointcloud/lidar_colored.ply",
            corrections: [correction]
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("missing frame IDs"))
        }
    }

    func testRefinedCaptureManifestRejectsAnyUnvisitedNonAnchorCamera() throws {
        let pose = try CameraPose(elements: identityTransform.flatMap { $0 })
        let anchor = CapturePoseCorrection(
            frameID: "frame_000001",
            isAnchor: true,
            optimizerStepCount: 0,
            correctedCameraToWorld: pose,
            translationNorm: 0,
            rotationNorm: 0
        )
        let unvisitedCamera = CapturePoseCorrection(
            frameID: "frame_000002",
            isAnchor: false,
            optimizerStepCount: 0,
            correctedCameraToWorld: pose,
            translationNorm: 0,
            rotationNorm: 0
        )
        let visitedCamera = CapturePoseCorrection(
            frameID: "frame_000003",
            isAnchor: false,
            optimizerStepCount: 1,
            correctedCameraToWorld: pose,
            translationNorm: 0,
            rotationNorm: 0
        )

        XCTAssertThrowsError(try RefinedCaptureManifestExporter.validateOptimizerVisits(
            in: [anchor, visitedCamera, unvisitedCamera]
        )) { error in
            XCTAssertTrue(
                error.localizedDescription.contains(
                    "non-anchor cameras received no optimizer steps: frame_000002"
                )
            )
        }

        XCTAssertNoThrow(try RefinedCaptureManifestExporter.validateOptimizerVisits(
            in: [anchor, visitedCamera]
        ))
    }

    func testGrowthBenchmarkRequiresMeasuredCapacityIncrease() {
        XCTAssertNil(
            TrainingBenchmarkRecorder.firstModelCapacityGrowth(
                [
                    (iteration: 1, capacity: 125_000),
                    (iteration: 2, capacity: 156_250),
                ],
                within: 3...5
            )
        )
        XCTAssertNil(
            TrainingBenchmarkRecorder.firstModelCapacityGrowth(
                [
                    (iteration: 1, capacity: 125_000),
                    (iteration: 3, capacity: 156_250),
                ],
                within: 2...5
            )
        )
        let growth = TrainingBenchmarkRecorder.firstModelCapacityGrowth(
            [
                (iteration: 1, capacity: 125_000),
                (iteration: 2, capacity: 156_250),
                (iteration: 3, capacity: 156_250),
                (iteration: 4, capacity: 195_312),
            ],
            within: 3...5
        )
        XCTAssertEqual(
            growth,
            TrainingBenchmarkCapacityGrowth(
                iteration: 4,
                previousCapacity: 156_250,
                newCapacity: 195_312
            )
        )
    }

    func testBenchmarkDistributionUsesMedianAndNearestRankP90() throws {
        let distribution = try XCTUnwrap(
            TrainingBenchmarkDistribution.make((1...10).map(Double.init))
        )

        XCTAssertEqual(distribution.count, 10)
        XCTAssertEqual(distribution.median, 5.5)
        XCTAssertEqual(distribution.p90, 9)
    }

    func testPreviewTextureTransformPreservesExtentAndFlipsVertically() {
        let width = 120
        let height = 80
        let transform = MetalPreviewView.textureToCoreImageTransform(
            height: height
        )
        let extent = CGRect(x: 0, y: 0, width: width, height: height)

        XCTAssertEqual(extent.applying(transform), extent)
        XCTAssertEqual(
            CGPoint(x: 12, y: 0).applying(transform),
            CGPoint(x: 12, y: height)
        )
        XCTAssertEqual(
            CGPoint(x: 12, y: height).applying(transform),
            CGPoint(x: 12, y: 0)
        )
    }

    func testCaptureGeometryRoundTripsRowMajorCameraTransform() throws {
        let matrix = simd_float4x4(
            SIMD4<Float>(1, 5, 9, 13),
            SIMD4<Float>(2, 6, 10, 14),
            SIMD4<Float>(3, 7, 11, 15),
            SIMD4<Float>(4, 8, 12, 16)
        )

        let rowMajor = CaptureGeometry.rowMajorCameraToWorld(matrix)
        XCTAssertEqual(rowMajor, (1...16).map(Float.init))

        let restored = try CaptureGeometry.cameraToWorld(fromRowMajor: rowMajor)
        for column in 0..<4 {
            XCTAssertEqual(restored[column], matrix[column])
        }
    }

    func testCaptureGeometryBackprojectsAndProjectsNativeRasterPoint() throws {
        let calibration = CaptureCalibrationRecord(
            width: 640,
            height: 480,
            fx: 500,
            fy: 520,
            cx: 320,
            cy: 240
        )
        var cameraToWorld = matrix_identity_float4x4
        cameraToWorld.columns.0 = SIMD4<Float>(0, 1, 0, 0)
        cameraToWorld.columns.1 = SIMD4<Float>(-1, 0, 0, 0)
        cameraToWorld.columns.3 = SIMD4<Float>(1, -2, 3, 1)
        let depthPixel = SIMD2<Float>(200, 90)

        let worldPoint = try XCTUnwrap(CaptureGeometry.worldPoint(
            depth: 2,
            depthPixel: depthPixel,
            depthWidth: 320,
            depthHeight: 240,
            calibration: calibration,
            cameraToWorld: cameraToWorld
        ))
        let normalizedPoint = try XCTUnwrap(
            CaptureGeometry.normalizedImagePoint(
                worldPoint: worldPoint,
                calibration: calibration,
                cameraToWorld: cameraToWorld
            )
        )

        XCTAssertEqual(
            normalizedPoint.x,
            CGFloat(depthPixel.x / 320),
            accuracy: 1e-6
        )
        XCTAssertEqual(
            normalizedPoint.y,
            CGFloat(depthPixel.y / 240),
            accuracy: 1e-6
        )
    }

    func testCaptureDisplayOrientationMapsFromInterfaceOrientation() {
        XCTAssertEqual(
            CaptureDisplayOrientation(interfaceOrientation: .portrait),
            .right
        )
        XCTAssertEqual(
            CaptureDisplayOrientation(interfaceOrientation: .portraitUpsideDown),
            .left
        )
        XCTAssertEqual(
            CaptureDisplayOrientation(interfaceOrientation: .landscapeLeft),
            .down
        )
        XCTAssertEqual(
            CaptureDisplayOrientation(interfaceOrientation: .landscapeRight),
            .up
        )
        XCTAssertNil(
            CaptureDisplayOrientation(interfaceOrientation: .unknown)
        )
    }

    func testCaptureOrientationTransformsCalibrationAndPreservesRays() throws {
        let calibration = CaptureCalibrationRecord(
            width: 6,
            height: 4,
            fx: 10,
            fy: 12,
            cx: 2.5,
            cy: 1.5
        )
        var cameraToWorld = matrix_identity_float4x4
        cameraToWorld.columns.0 = SIMD4<Float>(0, 1, 0, 0)
        cameraToWorld.columns.1 = SIMD4<Float>(-1, 0, 0, 0)
        cameraToWorld.columns.3 = SIMD4<Float>(1, -2, 3, 1)
        let sourcePixel = SIMD2<Float>(1.25, 2.75)
        let worldPoint = try XCTUnwrap(CaptureGeometry.worldPoint(
            depth: 2,
            depthPixel: sourcePixel,
            depthWidth: calibration.width,
            depthHeight: calibration.height,
            calibration: calibration,
            cameraToWorld: cameraToWorld
        ))
        let sourceNormalized = CGPoint(
            x: CGFloat(sourcePixel.x / Float(calibration.width)),
            y: CGFloat(sourcePixel.y / Float(calibration.height))
        )
        let expected: [CaptureDisplayOrientation: (
            calibration: CaptureCalibrationRecord,
            normalizedPoint: CGPoint
        )] = [
            .up: (
                calibration,
                sourceNormalized
            ),
            .right: (
                CaptureCalibrationRecord(
                    width: 4,
                    height: 6,
                    fx: 12,
                    fy: 10,
                    cx: 2.5,
                    cy: 2.5
                ),
                CGPoint(x: 1 - sourceNormalized.y, y: sourceNormalized.x)
            ),
            .down: (
                CaptureCalibrationRecord(
                    width: 6,
                    height: 4,
                    fx: 10,
                    fy: 12,
                    cx: 3.5,
                    cy: 2.5
                ),
                CGPoint(x: 1 - sourceNormalized.x, y: 1 - sourceNormalized.y)
            ),
            .left: (
                CaptureCalibrationRecord(
                    width: 4,
                    height: 6,
                    fx: 12,
                    fy: 10,
                    cx: 1.5,
                    cy: 3.5
                ),
                CGPoint(x: sourceNormalized.y, y: 1 - sourceNormalized.x)
            ),
        ]

        for orientation in [
            CaptureDisplayOrientation.up,
            .right,
            .down,
            .left,
        ] {
            let oriented = CaptureGeometry.orientedGeometry(
                calibration: calibration,
                cameraToWorld: cameraToWorld,
                orientation: orientation
            )
            XCTAssertEqual(oriented.calibration, expected[orientation]?.calibration)
            let projected = try XCTUnwrap(CaptureGeometry.normalizedImagePoint(
                worldPoint: worldPoint,
                calibration: oriented.calibration,
                cameraToWorld: oriented.cameraToWorld
            ))
            let expectedPoint = try XCTUnwrap(expected[orientation]?.normalizedPoint)
            XCTAssertEqual(projected.x, expectedPoint.x, accuracy: 1e-6)
            XCTAssertEqual(projected.y, expectedPoint.y, accuracy: 1e-6)
        }
    }

    func testCaptureOrientationRotatesInterleavedRasterBytes() throws {
        let source: [UInt8] = [
            1, 2, 3,
            4, 5, 6,
        ]
        let expected: [CaptureDisplayOrientation: (
            bytes: [UInt8],
            width: Int,
            height: Int
        )] = [
            .up: ([1, 2, 3, 4, 5, 6], 3, 2),
            .right: ([4, 1, 5, 2, 6, 3], 2, 3),
            .down: ([6, 5, 4, 3, 2, 1], 3, 2),
            .left: ([3, 6, 2, 5, 1, 4], 2, 3),
        ]

        for orientation in [
            CaptureDisplayOrientation.up,
            .right,
            .down,
            .left,
        ] {
            let result = try CaptureGeometry.orientedInterleavedBytes(
                source,
                width: 3,
                height: 2,
                components: 1,
                orientation: orientation
            )
            XCTAssertEqual(result.bytes, expected[orientation]?.bytes)
            XCTAssertEqual(result.width, expected[orientation]?.width)
            XCTAssertEqual(result.height, expected[orientation]?.height)
        }

        XCTAssertThrowsError(try CaptureGeometry.orientedInterleavedBytes(
            [1],
            width: 3,
            height: 2,
            components: 1,
            orientation: .right
        ))
    }

    func testCaptureFrameRecordOrientationRoundTripsAndDecodesLegacyJSON() throws {
        let record = CaptureFrameRecord(
            id: "frame_000001",
            imagePath: "images/frame_000001.png",
            maskPath: nil,
            softMaskPath: nil,
            displayOrientation: .right,
            calibration: CaptureCalibrationRecord(
                width: 1920,
                height: 1440,
                fx: 1_200,
                fy: 1_200,
                cx: 960,
                cy: 720
            ),
            cameraToWorld: Array(repeating: 0, count: 16),
            timestamp: 1,
            exposureDurationSeconds: 0.01,
            trackingState: "normal",
            maskConfidence: nil,
            fusedPointCount: 12
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let encoded = try encoder.encode(record)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        XCTAssertEqual(
            try decoder.decode(CaptureFrameRecord.self, from: encoded)
                .displayOrientation,
            .right
        )

        var legacyObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        legacyObject.removeValue(forKey: "display_orientation")
        let legacyData = try JSONSerialization.data(withJSONObject: legacyObject)
        XCTAssertNil(
            try decoder.decode(CaptureFrameRecord.self, from: legacyData)
                .displayOrientation
        )
        XCTAssertEqual(CaptureManifest.currentFormatVersion, 2)
    }

    func testCaptureStorePersistsPortraitRasterAndMatchingGeometry() async throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let pixelBuffer = try makeBGRAPixelBuffer(width: 3, height: 2)
        let calibration = CaptureCalibrationRecord(
            width: 3,
            height: 2,
            fx: 10,
            fy: 12,
            cx: 1.5,
            cy: 1
        )
        let store = try CaptureStore(
            mode: .scene,
            baseDirectory: temporaryDirectory
        )
        for index in 0..<3 {
            let candidate = CaptureFrameCandidate(
                image: OwnedPixelBuffer(pixelBuffer),
                depth: nil,
                confidence: nil,
                displayOrientation: .right,
                calibration: calibration,
                cameraToWorld: matrix_identity_float4x4,
                timestamp: Double(index),
                exposureDuration: 0.01,
                trackingState: "normal",
                rawFeaturePoints: [SIMD3<Float>(0, 0, -1)],
                subjectWorldPosition: nil
            )
            _ = try await store.accept(candidate)
        }

        let capture = try await store.finalize()
        let frame = try XCTUnwrap(capture.manifest.frames.first)
        let image = try XCTUnwrap(UIImage(
            contentsOfFile: frame.imageURL(under: capture.rootURL).path
        )).cgImage
        let cgImage = try XCTUnwrap(image)

        XCTAssertEqual(cgImage.width, 2)
        XCTAssertEqual(cgImage.height, 3)
        XCTAssertEqual(
            try topLeftRGBABytes(from: cgImage).enumerated().compactMap {
                $0.offset.isMultiple(of: 4) ? $0.element : nil
            },
            [112, 16, 144, 48, 176, 80]
        )
        XCTAssertEqual(frame.displayOrientation, .right)
        XCTAssertEqual(frame.calibration, CaptureCalibrationRecord(
            width: 2,
            height: 3,
            fx: 12,
            fy: 10,
            cx: 1,
            cy: 1.5
        ))
        XCTAssertEqual(capture.descriptor.frames.first?.calibration.width, 2)
        XCTAssertEqual(capture.descriptor.frames.first?.calibration.height, 3)

        let transformsData = try Data(contentsOf: capture.rootURL.appending(
            path: "transforms.json"
        ))
        let transforms = try XCTUnwrap(
            JSONSerialization.jsonObject(with: transformsData) as? [String: Any]
        )
        let frames = try XCTUnwrap(transforms["frames"] as? [[String: Any]])
        XCTAssertEqual(frames.first?["w"] as? Int, 2)
        XCTAssertEqual(frames.first?["h"] as? Int, 3)
        XCTAssertEqual(frames.first?["cx"] as? Double, 1)
        XCTAssertEqual(frames.first?["cy"] as? Double, 1.5)
    }

    func testCaptureFrameAdmissionStaysBoundedAndReleasesAfterCompletion() throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let store = try CaptureStore(
            mode: .scene,
            baseDirectory: temporaryDirectory
        )
        let controller = CaptureFrameAdmissionController()
        controller.start(store: store, subjectWorldPosition: nil)
        controller.updateDisplayOrientation(.right)
        defer { controller.finish() }

        let first = try XCTUnwrap(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 1
        ))
        for index in 1...100 {
            XCTAssertNil(controller.begin(
                cameraToWorld: matrix_identity_float4x4,
                timestamp: 1 + Double(index)
            ))
        }

        controller.complete(first, committed: true)
        var moved = matrix_identity_float4x4
        moved.columns.3.x = 0.04
        XCTAssertNil(controller.begin(
            cameraToWorld: moved,
            timestamp: 1.2
        ))
        XCTAssertNil(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 1.5
        ))

        let rejected = try XCTUnwrap(controller.begin(
            cameraToWorld: moved,
            timestamp: 1.5
        ))
        controller.complete(rejected, committed: false)
        let retry = try XCTUnwrap(controller.begin(
            cameraToWorld: moved,
            timestamp: 1.6
        ))
        controller.complete(retry, committed: true)

        controller.stop()
        var movedAgain = moved
        movedAgain.columns.3.x = 0.08
        XCTAssertNil(controller.begin(
            cameraToWorld: movedAgain,
            timestamp: 2
        ))
    }

    func testCaptureFrameAdmissionRejectsStaleGenerationWork() async throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let controller = CaptureFrameAdmissionController()
        let firstStore = try CaptureStore(
            mode: .scene,
            baseDirectory: temporaryDirectory
        )
        controller.start(store: firstStore, subjectWorldPosition: nil)
        controller.updateDisplayOrientation(.right)
        let stale = try XCTUnwrap(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 1
        ))

        let secondStore = try CaptureStore(
            mode: .scene,
            baseDirectory: temporaryDirectory
        )
        controller.start(store: secondStore, subjectWorldPosition: nil)
        defer { controller.finish() }
        let current = try XCTUnwrap(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 2
        ))

        XCTAssertFalse(controller.complete(stale, committed: true))
        XCTAssertNil(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 3
        ))

        let staleTask = Task<Void, Never> {
            while !Task.isCancelled {
                await Task.yield()
            }
        }
        controller.register(task: staleTask, for: stale)
        await staleTask.value
        XCTAssertTrue(staleTask.isCancelled)

        XCTAssertTrue(controller.complete(current, committed: false))
        let next = try XCTUnwrap(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 3
        ))
        XCTAssertTrue(controller.complete(next, committed: false))
    }

    func testCaptureFrameAdmissionRequiresAndSnapshotsDisplayOrientation() throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let store = try CaptureStore(
            mode: .scene,
            baseDirectory: temporaryDirectory
        )
        let controller = CaptureFrameAdmissionController()
        controller.start(store: store, subjectWorldPosition: nil)
        defer { controller.finish() }

        XCTAssertNil(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 1
        ))

        controller.updateDisplayOrientation(.right)
        let portrait = try XCTUnwrap(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 1
        ))
        controller.updateDisplayOrientation(.left)
        XCTAssertEqual(portrait.displayOrientation, .right)
        XCTAssertTrue(controller.complete(portrait, committed: false))

        let upsideDown = try XCTUnwrap(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 2
        ))
        XCTAssertEqual(upsideDown.displayOrientation, .left)
        XCTAssertTrue(controller.complete(upsideDown, committed: false))
    }

    private func writeBinaryPointHeader(
        count: UInt64,
        to directory: URL,
        recordCount: Int? = nil
    ) throws {
        let header = (0..<MemoryLayout<UInt64>.size).map { byte in
            UInt8(truncatingIfNeeded: count >> UInt64(byte * 8))
        }
        var data = Data(header)
        data.append(Data(
            repeating: 0,
            count: (recordCount ?? Int(count)) * 51
        ))
        try data.write(to: directory.appending(path: "points3D.bin"))
    }

    private func makeBGRAPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:],
        ]
        XCTAssertEqual(
            CVPixelBufferCreate(
                kCFAllocatorDefault,
                width,
                height,
                kCVPixelFormatType_32BGRA,
                attributes as CFDictionary,
                &pixelBuffer
            ),
            kCVReturnSuccess
        )
        let result = try XCTUnwrap(pixelBuffer)
        CVPixelBufferLockBaseAddress(result, [])
        defer { CVPixelBufferUnlockBaseAddress(result, []) }
        let rowBytes = CVPixelBufferGetBytesPerRow(result)
        let storage = try XCTUnwrap(CVPixelBufferGetBaseAddress(result))
            .assumingMemoryBound(to: UInt8.self)
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * rowBytes + x * 4
                let value = UInt8(16 + (y * width + x) * 32)
                storage[offset] = value
                storage[offset + 1] = value
                storage[offset + 2] = value
                storage[offset + 3] = .max
            }
        }
        return result
    }

    private var identityTransform: [[Float]] {
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    }

    private func makeCaptureFrameRecord(
        id: String,
        imagePath: String,
        maskPath: String?
    ) -> CaptureFrameRecord {
        CaptureFrameRecord(
            id: id,
            imagePath: imagePath,
            maskPath: maskPath,
            softMaskPath: nil,
            displayOrientation: .up,
            calibration: CaptureCalibrationRecord(
                width: 1_920,
                height: 1_440,
                fx: 1_200,
                fy: 1_210,
                cx: 960,
                cy: 720
            ),
            cameraToWorld: identityTransform.flatMap { $0 },
            timestamp: 0,
            exposureDurationSeconds: 0.01,
            trackingState: "normal",
            maskConfidence: nil,
            fusedPointCount: 1
        )
    }

    private func topLeftRGBABytes(from image: CGImage) throws -> [UInt8] {
        var bytes = [UInt8](
            repeating: 0,
            count: image.width * image.height * 4
        )
        let context = try XCTUnwrap(CGContext(
            data: &bytes,
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bytesPerRow: image.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue |
                CGBitmapInfo.byteOrder32Big.rawValue
        ))
        context.interpolationQuality = .none
        context.draw(
            image,
            in: CGRect(x: 0, y: 0, width: image.width, height: image.height)
        )
        return bytes
    }

    private func writeNerfstudioManifest(
        to directory: URL,
        plyFilePath: String?,
        cameraModel: String = "OPENCV",
        framePaths: [String] = ["images/frame.png"],
        maskPaths: [String?]? = nil
    ) throws {
        let transform: [[Double]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        let frames: [[String: Any]] = framePaths.enumerated().map { index, path in
            var frame: [String: Any] = [
                "file_path": path,
                "transform_matrix": transform,
            ]
            if let maskPaths,
               maskPaths.indices.contains(index),
               let maskPath = maskPaths[index] {
                frame["mask_path"] = maskPath
            }
            return frame
        }
        var manifest: [String: Any] = [
            "camera_model": cameraModel,
            "w": 12,
            "h": 7,
            "fl_x": 10.0,
            "fl_y": 10.0,
            "cx": 6.0,
            "cy": 3.5,
            "frames": frames,
        ]
        if let plyFilePath {
            manifest["ply_file_path"] = plyFilePath
        }
        let data = try JSONSerialization.data(
            withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(to: directory.appending(path: "transforms.json"))
    }

    private func writePointPLY(count: Int, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let points = (0..<count)
            .map { "\($0) 0 0 255 255 255" }
            .joined(separator: "\n")
        try """
        ply
        format ascii 1.0
        element vertex \(count)
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        \(points)
        """.write(to: url, atomically: true, encoding: .utf8)
    }

    private func writePNG(width: Int, height: Int, to url: URL) throws {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = try XCTUnwrap(CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        context.setFillColor(CGColor(
            colorSpace: colorSpace,
            components: [0.25, 0.5, 0.75, 1]
        ) ?? CGColor(gray: 0.5, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = try XCTUnwrap(context.makeImage())
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }
}

final class RealityKitColmapExportBuilderTests: XCTestCase {
    private var temporaryDirectory: URL?

    override func setUpWithError() throws {
        let directory = FileManager.default.temporaryDirectory.appending(
            path: "msplat-realitykit-colmap-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        temporaryDirectory = directory
    }

    override func tearDownWithError() throws {
        if let temporaryDirectory {
            try FileManager.default.removeItem(at: temporaryDirectory)
        }
        temporaryDirectory = nil
    }

    func testRealityKitIdentityPoseConvertsToColmapConvention() {
        let pose = RealityKitColmapPoseConverter.convertToColmapCamera(
            cameraToWorld: matrix_identity_float4x4
        )

        XCTAssertEqual(pose.qvec[0], 0, accuracy: 0.000_001)
        XCTAssertEqual(pose.qvec[1], 1, accuracy: 0.000_001)
        XCTAssertEqual(pose.qvec[2], 0, accuracy: 0.000_001)
        XCTAssertEqual(pose.qvec[3], 0, accuracy: 0.000_001)
        XCTAssertTrue(pose.tvec.allSatisfy { abs($0) < 0.000_001 })
    }

    func testRealityKitTranslatedPoseStaysInPointCloudFrame() {
        var cameraToWorld = matrix_identity_float4x4
        cameraToWorld.columns.3 = SIMD4<Float>(1, 2, 3, 1)

        let pose = RealityKitColmapPoseConverter.convertToColmapCamera(
            cameraToWorld: cameraToWorld
        )

        XCTAssertEqual(pose.tvec[0], -1, accuracy: 0.000_001)
        XCTAssertEqual(pose.tvec[1], 2, accuracy: 0.000_001)
        XCTAssertEqual(pose.tvec[2], 3, accuracy: 0.000_001)
    }

    func testAlignmentMaskAndTrainingMaskExportAreIndependent() {
        let combinations = [
            (alignmentMask: false, trainingMasks: false),
            (alignmentMask: false, trainingMasks: true),
            (alignmentMask: true, trainingMasks: false),
            (alignmentMask: true, trainingMasks: true),
        ]

        for combination in combinations {
            let options = RealityKitAlignmentOptions(
                usesObjectMaskingForAlignment: combination.alignmentMask,
                exportsTrainingMasks: combination.trainingMasks,
                ordering: .sequential
            )
            XCTAssertEqual(
                options.usesObjectMaskingForAlignment,
                combination.alignmentMask
            )
            XCTAssertEqual(
                options.exportsTrainingMasks,
                combination.trainingMasks
            )
        }
    }

    func testRealityKitInternalMeshStagesUseAlignmentSpecificCopy() {
        XCTAssertEqual(
            RealityKitAlignmentStageDescription.message(for: .meshGeneration),
            "Finalizing camera and point-cloud alignment…"
        )
        XCTAssertEqual(
            RealityKitAlignmentStageDescription.message(for: .textureMapping),
            "Finalizing camera and point-cloud alignment…"
        )
    }

    func testTrainingMaskFilenameMatchesExportedImageStem() {
        XCTAssertEqual(
            RealityKitColmapExportBuilder.maskFilename(
                forExportedImage: "image_000123.jpg"
            ),
            "image_000123.png"
        )
    }

    func testRightOrientedImageRotatesIntrinsicsIntoExportedRaster() {
        let raw = RealityKitColmapCamera(
            width: 4_032,
            height: 3_024,
            fx: 3_000,
            fy: 2_990,
            cx: 2_016,
            cy: 1_512
        )

        let camera = RealityKitColmapExportBuilder.camera(
            raw,
            applying: .right
        )

        XCTAssertEqual(camera.width, 3_024)
        XCTAssertEqual(camera.height, 4_032)
        XCTAssertEqual(camera.fx, 2_990, accuracy: 0.000_001)
        XCTAssertEqual(camera.fy, 3_000, accuracy: 0.000_001)
        XCTAssertEqual(camera.cx, 1_511, accuracy: 0.000_001)
        XCTAssertEqual(camera.cy, 2_016, accuracy: 0.000_001)
    }

    func testCameraGroupingUsesHalfPixelTolerance() {
        let entries = [
            imageEntry(id: 1, fx: 1_000, fy: 1_001, cx: 512, cy: 384),
            imageEntry(id: 2, fx: 1_000.4, fy: 1_001.3, cx: 512.5, cy: 383.7),
            imageEntry(id: 3, fx: 1_001, fy: 1_001, cx: 512, cy: 384),
        ]

        let (cameras, assignments) = RealityKitColmapExportBuilder
            .groupCameras(entries)

        XCTAssertEqual(cameras.count, 2)
        XCTAssertEqual(assignments[1], 1)
        XCTAssertEqual(assignments[2], 1)
        XCTAssertEqual(assignments[3], 2)
    }

    func testSparseWriterProducesTrainableColmapLayout() throws {
        let root = try XCTUnwrap(temporaryDirectory).appending(
            path: "Aligned",
            directoryHint: .isDirectory
        )
        let sparse = root
            .appending(path: "sparse", directoryHint: .isDirectory)
            .appending(path: "0", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: root.appending(path: "images", directoryHint: .isDirectory),
            withIntermediateDirectories: true
        )

        try RealityKitColmapExportBuilder.writeSparseFiles(
            to: sparse,
            entries: [
                imageEntry(id: 1),
                imageEntry(id: 2),
                imageEntry(id: 3),
            ],
            points: [
                RealityKitColmapPoint(
                    position: SIMD3<Float>(1, 2, 3),
                    color: SIMD3<Float>(10, 20, 30)
                ),
                RealityKitColmapPoint(
                    position: SIMD3<Float>(-1, -2, -3),
                    color: SIMD3<Float>(40, 50, 60)
                ),
            ]
        )

        let folder = try XCTUnwrap(DatasetFolder(picked: root))
        XCTAssertEqual(folder.kind, .colmap)
        XCTAssertEqual(try DatasetFolder.initialSparsePointCount(at: root), 2)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: sparse.appending(path: "cameras.bin").path
        ))
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: sparse.appending(path: "images.bin").path
        ))
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: sparse.appending(path: "points3D.bin").path
        ))

        let imagesText = try String(
            contentsOf: sparse.appending(path: "images.txt"),
            encoding: .utf8
        )
        XCTAssertTrue(imagesText.contains("1 0.0 1.0 0.0 0.0"))
        XCTAssertTrue(imagesText.contains("1 image_000001.jpg"))
    }

    func testSparseWriterRejectsEmptyPointCloud() throws {
        let sparse = try XCTUnwrap(temporaryDirectory).appending(
            path: "sparse/0",
            directoryHint: .isDirectory
        )

        XCTAssertThrowsError(
            try RealityKitColmapExportBuilder.writeSparseFiles(
                to: sparse,
                entries: [imageEntry(id: 1)],
                points: []
            )
        ) { error in
            guard case RealityKitColmapExportError.noSparsePoints = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testRawAlignmentInputFindsReadableImagesInNestedImagesFolder() throws {
        let root = try XCTUnwrap(temporaryDirectory).appending(
            path: "Raw",
            directoryHint: .isDirectory
        )
        let images = root.appending(path: "images", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: images,
            withIntermediateDirectories: true
        )
        try writePNG(to: images.appending(path: "frame_2.png"))
        try writePNG(to: images.appending(path: "frame_1.png"))
        try Data("not an image".utf8).write(
            to: images.appending(path: "broken.png")
        )

        let input = try XCTUnwrap(RealityKitAlignmentInput(picked: root))

        XCTAssertEqual(input.imagesURL.standardizedFileURL, images.standardizedFileURL)
        XCTAssertEqual(
            input.imageURLs.map(\.lastPathComponent),
            ["frame_1.png", "frame_2.png"]
        )
        XCTAssertTrue(input.knownCamerasByFilename.isEmpty)
    }

    private func imageEntry(
        id: Int,
        fx: Double = 1_000,
        fy: Double = 1_001,
        cx: Double = 512,
        cy: Double = 384
    ) -> RealityKitColmapImageEntry {
        RealityKitColmapImageEntry(
            imageID: id,
            filename: String(format: "image_%06d.jpg", id),
            cameraToWorld: matrix_identity_float4x4,
            camera: RealityKitColmapCamera(
                width: 1_024,
                height: 768,
                fx: fx,
                fy: fy,
                cx: cx,
                cy: cy
            ),
            orientation: .up
        )
    }

    private func writePNG(to url: URL) throws {
        let context = try XCTUnwrap(CGContext(
            data: nil,
            width: 4,
            height: 3,
            bitsPerComponent: 8,
            bytesPerRow: 16,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        let image = try XCTUnwrap(context.makeImage())
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }
}
