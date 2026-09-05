import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import Msplat
@testable import MsplatExample
@preconcurrency import RealityKit
import simd
import UIKit
import UniformTypeIdentifiers
@preconcurrency import Vision
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
        let translationSumSquares: Float = 0.003 * 0.003 + 0.004 * 0.004
        let expectedTranslationRMS = sqrt(translationSumSquares / 2)
        XCTAssertEqual(
            output.summary.rmsTranslationMeters,
            expectedTranslationRMS,
            accuracy: 1e-6
        )
        XCTAssertEqual(output.summary.maximumRotationRadians, 0.02, accuracy: 1e-6)
        let rotationSumSquares: Float = 0.01 * 0.01 + 0.02 * 0.02
        let expectedRotationRMS = sqrt(rotationSumSquares / 2)
        XCTAssertEqual(
            output.summary.rmsRotationRadians,
            expectedRotationRMS,
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
            let commit = try await store.accept(candidate)
            XCTAssertEqual(commit.totalFrameCount, index + 1)
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

    func testCaptureStoreDiscardRemovesCaptureDirectory() async throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let store = try CaptureStore(
            mode: .scene,
            baseDirectory: temporaryDirectory
        )
        let capturesDirectory = temporaryDirectory.appending(
            path: "Captures",
            directoryHint: .isDirectory
        )
        let captureDirectories = try FileManager.default.contentsOfDirectory(
            at: capturesDirectory,
            includingPropertiesForKeys: nil
        )
        let captureDirectory = try XCTUnwrap(captureDirectories.first)
        XCTAssertEqual(captureDirectories.count, 1)
        try Data([1, 2, 3]).write(
            to: captureDirectory.appending(path: "images/unfinished.png")
        )
        try Data([4, 5, 6]).write(
            to: captureDirectory.appending(path: "masks/unfinished.png")
        )

        try await store.discard()

        XCTAssertFalse(FileManager.default.fileExists(atPath: captureDirectory.path))
        try await store.discard()
    }

    func testObjectCaptureExportsSoftMasksWithMatchingOrientedGeometry() async throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        let image = try makeBGRAPixelBuffer(width: 3, height: 2)
        let depth = try makeDepthPixelBuffer(width: 3, height: 2)
        let soft: [UInt8] = [255, 64, 127, 128, 192, 0]
        let binary = soft.map { $0 >= 128 ? UInt8.max : 0 }
        let calibration = CaptureCalibrationRecord(
            width: 3, height: 2, fx: 10, fy: 12, cx: 1.5, cy: 1
        )
        let orientations: [(CaptureDisplayOrientation, [Int])] = [
            (.up, [0, 1, 2, 3, 4, 5]),
            (.right, [3, 0, 4, 1, 5, 2]),
            (.down, [5, 4, 3, 2, 1, 0]),
            (.left, [2, 5, 1, 4, 0, 3]),
        ]
        var unrotatedCoverage: [UInt8] = []

        for (orientation, sourceIndices) in orientations {
            let store = try CaptureStore(
                mode: .object,
                baseDirectory: directory,
                objectMaskGenerator: { _ in
                    CaptureStore.MaskResult(
                        bytes: soft, binaryBytes: binary,
                        width: 3, height: 2, confidence: 0.5
                    )
                }
            )
            for index in 0..<3 {
                _ = try await store.accept(CaptureFrameCandidate(
                    image: OwnedPixelBuffer(image),
                    depth: OwnedPixelBuffer(depth),
                    confidence: nil,
                    displayOrientation: orientation,
                    calibration: calibration,
                    cameraToWorld: matrix_identity_float4x4,
                    timestamp: Double(index),
                    exposureDuration: 0.01,
                    trackingState: "normal",
                    rawFeaturePoints: [],
                    subjectWorldPosition: SIMD3<Float>(0, 0, -1)
                ))
            }

            let capture = try await store.finalize()
            let folder = try XCTUnwrap(DatasetFolder(picked: capture.rootURL))
            XCTAssertEqual(folder.kind, .nerfstudio)
            XCTAssertTrue(folder.hasNerfstudioTrainingMasks)
            XCTAssertFalse(folder.supportsAutomaticTrainingMaskDiscovery)

            let data = try Data(contentsOf: capture.rootURL.appending(path: "transforms.json"))
            let manifest = try XCTUnwrap(
                JSONSerialization.jsonObject(with: data) as? [String: Any]
            )
            let exportedFrames = try XCTUnwrap(manifest["frames"] as? [[String: Any]])
            XCTAssertEqual(exportedFrames.count, capture.descriptor.frames.count)

            for index in exportedFrames.indices {
                let exported = exportedFrames[index]
                let record = capture.manifest.frames[index]
                let direct = capture.descriptor.frames[index]
                let directMask = try XCTUnwrap(direct.trainingMask)
                let exportedPath = try XCTUnwrap(exported["mask_path"] as? String)
                let exportedURL = capture.rootURL.appending(path: exportedPath)
                XCTAssertEqual(exportedPath, record.softMaskPath)
                XCTAssertNotEqual(exportedPath, record.maskPath)
                XCTAssertEqual(exportedURL, directMask.url)
                XCTAssertEqual(directMask.coverageChannel, .luminance)
                XCTAssertEqual(exported["w"] as? Int, direct.calibration.width)
                XCTAssertEqual(exported["h"] as? Int, direct.calibration.height)
                XCTAssertEqual(exported["fl_x"] as? Float, direct.calibration.fx)
                XCTAssertEqual(exported["fl_y"] as? Float, direct.calibration.fy)
                XCTAssertEqual(exported["cx"] as? Float, direct.calibration.cx)
                XCTAssertEqual(exported["cy"] as? Float, direct.calibration.cy)
                let transform = try XCTUnwrap(exported["transform_matrix"] as? [[Float]])
                XCTAssertEqual(transform.flatMap { $0 }, direct.cameraToWorld.elements)

                let softImage = try XCTUnwrap(UIImage(contentsOfFile: exportedURL.path)?.cgImage)
                let softPixels = try topLeftRGBABytes(from: softImage)
                    .enumerated().compactMap { $0.offset.isMultiple(of: 4) ? $0.element : nil }
                XCTAssertTrue(softPixels.contains { $0 > 0 && $0 < 255 })
                XCTAssertEqual(softImage.width, record.calibration.width)
                XCTAssertEqual(softImage.height, record.calibration.height)
                if orientation == .up { unrotatedCoverage = softPixels }
                XCTAssertEqual(softPixels, sourceIndices.map { unrotatedCoverage[$0] })

                let binaryURL = try XCTUnwrap(record.maskURL(under: capture.rootURL))
                let binaryImage = try XCTUnwrap(UIImage(contentsOfFile: binaryURL.path)?.cgImage)
                let binaryPixels = try topLeftRGBABytes(from: binaryImage)
                    .enumerated().compactMap { $0.offset.isMultiple(of: 4) ? $0.element : nil }
                XCTAssertEqual(binaryPixels, sourceIndices.map { binary[$0] })
            }
        }
    }

    @MainActor
    func testCaptureEngineStopDiscardsUnfinishedStore() async throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let engine = CaptureEngine(captureBaseDirectory: temporaryDirectory)
        engine.updateInterfaceOrientation(.portrait)
        try engine.startRecording(mode: .scene)

        let capturesDirectory = temporaryDirectory.appending(
            path: "Captures",
            directoryHint: .isDirectory
        )
        let captureDirectories = try FileManager.default.contentsOfDirectory(
            at: capturesDirectory,
            includingPropertiesForKeys: nil
        )
        let captureDirectory = try XCTUnwrap(captureDirectories.first)
        XCTAssertEqual(captureDirectories.count, 1)
        try Data([1, 2, 3]).write(
            to: captureDirectory.appending(path: "images/unfinished.png")
        )

        let cleanupTask = engine.stop()
        await cleanupTask.value

        XCTAssertFalse(FileManager.default.fileExists(atPath: captureDirectory.path))
    }

    @MainActor
    func testCaptureEngineWaitsForCleanupBeforeRestarting() async throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let engine = CaptureEngine(captureBaseDirectory: temporaryDirectory)
        engine.updateInterfaceOrientation(.portrait)
        try engine.startRecording(mode: .scene)

        let capturesDirectory = temporaryDirectory.appending(
            path: "Captures",
            directoryHint: .isDirectory
        )
        let firstCaptureDirectory = try XCTUnwrap(
            FileManager.default.contentsOfDirectory(
                at: capturesDirectory,
                includingPropertiesForKeys: nil
            ).first
        )

        let firstCleanupTask = engine.stop()
        XCTAssertThrowsError(try engine.startRecording(mode: .scene)) { error in
            XCTAssertTrue(error is CancellationError)
        }
        await firstCleanupTask.value

        XCTAssertFalse(
            FileManager.default.fileExists(atPath: firstCaptureDirectory.path)
        )
        try engine.startRecording(mode: .scene)
        let secondCaptureDirectories = try FileManager.default.contentsOfDirectory(
            at: capturesDirectory,
            includingPropertiesForKeys: nil
        )
        let secondCaptureDirectory = try XCTUnwrap(secondCaptureDirectories.first)
        XCTAssertEqual(secondCaptureDirectories.count, 1)
        XCTAssertNotEqual(secondCaptureDirectory, firstCaptureDirectory)
        XCTAssertTrue(engine.isRecording)

        let secondCleanupTask = engine.stop()
        await secondCleanupTask.value

        XCTAssertFalse(
            FileManager.default.fileExists(atPath: secondCaptureDirectory.path)
        )
    }

    @MainActor
    func testCaptureEngineFinalizationFailureRemainsDiscardable() async throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let engine = CaptureEngine(captureBaseDirectory: temporaryDirectory)
        engine.updateInterfaceOrientation(.portrait)
        try engine.startRecording(mode: .scene)

        let capturesDirectory = temporaryDirectory.appending(
            path: "Captures",
            directoryHint: .isDirectory
        )
        let captureDirectories = try FileManager.default.contentsOfDirectory(
            at: capturesDirectory,
            includingPropertiesForKeys: nil
        )
        let captureDirectory = try XCTUnwrap(captureDirectories.first)
        XCTAssertEqual(captureDirectories.count, 1)

        do {
            _ = try await engine.stopAndFinalize()
            XCTFail("Expected an empty capture to fail finalization")
        } catch CaptureFailure.insufficientCapture(let frameCount, let pointCount) {
            XCTAssertEqual(frameCount, 0)
            XCTAssertEqual(pointCount, 0)
        } catch {
            XCTFail("Unexpected finalization error: \(error)")
        }

        let cleanupTask = engine.stop()
        await cleanupTask.value

        XCTAssertFalse(FileManager.default.fileExists(atPath: captureDirectory.path))
    }

    func testCaptureTaskGenerationDoesNotLetCancelledWorkFinishReplacement() {
        var generation = CaptureTaskGeneration()
        let cancelledOperation = generation.begin()
        generation.invalidate()
        let replacementOperation = generation.begin()

        XCTAssertFalse(generation.finish(cancelledOperation))
        XCTAssertTrue(generation.isActive)
        XCTAssertEqual(generation.currentID, replacementOperation)
        XCTAssertTrue(generation.finish(replacementOperation))
        XCTAssertFalse(generation.isActive)
    }

    func testCaptureEventOrderingIgnoresRejectionOlderThanCommit() {
        var ordering = CaptureEventOrdering()

        XCTAssertTrue(ordering.shouldApplyTelemetry(candidateSequence: 1))
        XCTAssertTrue(ordering.recordCommit(candidateSequence: 2))
        XCTAssertFalse(ordering.shouldApplyTelemetry(candidateSequence: 1))
        XCTAssertFalse(ordering.shouldApplyTelemetry(candidateSequence: 2))
        XCTAssertTrue(ordering.shouldApplyTelemetry(candidateSequence: 3))

        XCTAssertFalse(ordering.recordCommit(candidateSequence: 1))
        XCTAssertEqual(ordering.latestCommittedCandidateSequence, 2)
    }

    func testCaptureEventOrderingPreservesNewerRejectionAcrossOlderCommit() {
        var ordering = CaptureEventOrdering()

        XCTAssertTrue(ordering.shouldApplyTelemetry(candidateSequence: 3))
        XCTAssertFalse(ordering.recordCommit(candidateSequence: 2))
        XCTAssertEqual(ordering.latestCommittedCandidateSequence, 2)
        XCTAssertEqual(ordering.latestMessageCandidateSequence, 3)

        XCTAssertTrue(ordering.recordCommit(candidateSequence: 4))
        XCTAssertEqual(ordering.latestMessageCandidateSequence, 4)
    }

    func testCaptureVisionNormalizationOnlyRetriesCandidateLocalErrors() throws {
        for candidateCode in [
            VNErrorCode.invalidImage,
            .invalidFormat,
            .outOfBoundsError,
        ] {
            let candidateError = NSError(
                domain: VNErrorDomain,
                code: candidateCode.rawValue
            )
            let normalized = normalizeCaptureCandidateVisionError(candidateError)
            let captureFailure = try XCTUnwrap(normalized as? CaptureFailure)
            guard case .invalidFrame = captureFailure else {
                return XCTFail("Expected malformed Vision input to reject the candidate")
            }
        }

        let cancellation = NSError(
            domain: VNErrorDomain,
            code: VNErrorCode.requestCancelled.rawValue
        )
        XCTAssertTrue(
            normalizeCaptureCandidateVisionError(cancellation) is CancellationError
        )

        for nonCandidateCode in [
            VNErrorCode.ioError,
            .invalidArgument,
            .internalError,
            .outOfMemory,
            .timeout,
        ] {
            let nonCandidateError = NSError(
                domain: VNErrorDomain,
                code: nonCandidateCode.rawValue
            )
            let preserved = normalizeCaptureCandidateVisionError(nonCandidateError)
                as NSError
            XCTAssertEqual(preserved.domain, VNErrorDomain)
            XCTAssertEqual(preserved.code, nonCandidateCode.rawValue)
        }
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

        XCTAssertGreaterThan(current.candidateSequence, stale.candidateSequence)

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

    func testCaptureFrameAdmissionAbortInvalidatesAndCancelsActiveWork() async throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let store = try CaptureStore(
            mode: .scene,
            baseDirectory: temporaryDirectory
        )
        let controller = CaptureFrameAdmissionController()
        controller.start(store: store, subjectWorldPosition: nil)
        controller.updateDisplayOrientation(.right)
        let active = try XCTUnwrap(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 1
        ))
        let activeTask = Task<Void, Never> {
            while !Task.isCancelled {
                await Task.yield()
            }
        }
        controller.register(task: activeTask, for: active)

        controller.abort()
        await activeTask.value

        XCTAssertTrue(activeTask.isCancelled)
        XCTAssertFalse(controller.complete(active, committed: true))
        XCTAssertNil(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 2
        ))
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

    func testCaptureFrameAdmissionCanUseStreamingFramesOnly() throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let store = try CaptureStore(
            mode: .scene,
            baseDirectory: temporaryDirectory
        )
        let controller = CaptureFrameAdmissionController()
        controller.start(
            store: store,
            subjectWorldPosition: nil,
            usesHighResolutionCapture: false
        )
        controller.updateDisplayOrientation(.right)
        defer { controller.finish() }

        let admission = try XCTUnwrap(controller.begin(
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 1
        ))

        XCTAssertFalse(admission.requestsHighResolutionFrame)
        XCTAssertTrue(controller.complete(admission, committed: false))
    }

    func testCaptureHighResolutionRequestGateAllowsOnlyOneInFlightRequest() {
        let gate = CaptureHighResolutionRequestGate()

        XCTAssertTrue(gate.reserve())
        XCTAssertFalse(gate.reserve())
        gate.release()
        XCTAssertTrue(gate.reserve())
        gate.release()
    }

    func testCaptureGeometryFallbackRequirementsForSceneAndObject() throws {
        let pixelBuffer = try makeBGRAPixelBuffer(width: 2, height: 2)
        let calibration = CaptureCalibrationRecord(
            width: 2,
            height: 2,
            fx: 2,
            fy: 2,
            cx: 1,
            cy: 1
        )
        func candidate(
            hasDepth: Bool,
            rawFeaturePoints: [SIMD3<Float>],
            subjectWorldPosition: SIMD3<Float>?
        ) -> CaptureFrameCandidate {
            CaptureFrameCandidate(
                image: OwnedPixelBuffer(pixelBuffer),
                depth: hasDepth ? OwnedPixelBuffer(pixelBuffer) : nil,
                confidence: nil,
                displayOrientation: .right,
                calibration: calibration,
                cameraToWorld: matrix_identity_float4x4,
                timestamp: 1,
                exposureDuration: 0.01,
                trackingState: "normal",
                rawFeaturePoints: rawFeaturePoints,
                subjectWorldPosition: subjectWorldPosition
            )
        }

        XCTAssertFalse(captureCandidateHasUsableGeometry(candidate(
            hasDepth: false,
            rawFeaturePoints: [],
            subjectWorldPosition: nil
        )))
        XCTAssertTrue(captureCandidateHasUsableGeometry(candidate(
            hasDepth: false,
            rawFeaturePoints: [SIMD3<Float>(0, 0, -1)],
            subjectWorldPosition: nil
        )))
        XCTAssertFalse(captureCandidateHasUsableGeometry(candidate(
            hasDepth: false,
            rawFeaturePoints: [SIMD3<Float>(0, 0, -1)],
            subjectWorldPosition: SIMD3<Float>(0, 0, -1)
        )))
        XCTAssertTrue(captureCandidateHasUsableGeometry(candidate(
            hasDepth: true,
            rawFeaturePoints: [],
            subjectWorldPosition: SIMD3<Float>(0, 0, -1)
        )))
    }

    func testCaptureAcceptanceRetriesStreamingCandidateAfterLocalRejection() async throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let pixelBuffer = try makeBGRAPixelBuffer(width: 2, height: 2)
        let validCalibration = CaptureCalibrationRecord(
            width: 2,
            height: 2,
            fx: 2,
            fy: 2,
            cx: 1,
            cy: 1
        )
        let invalidCalibration = CaptureCalibrationRecord(
            width: 3,
            height: 2,
            fx: 2,
            fy: 2,
            cx: 1,
            cy: 1
        )
        func candidate(
            image: OwnedPixelBuffer,
            calibration: CaptureCalibrationRecord
        ) -> CaptureFrameCandidate {
            CaptureFrameCandidate(
                image: image,
                depth: nil,
                confidence: nil,
                displayOrientation: .up,
                calibration: calibration,
                cameraToWorld: matrix_identity_float4x4,
                timestamp: 1,
                exposureDuration: 0.01,
                trackingState: "normal",
                rawFeaturePoints: [SIMD3<Float>(0, 0, -1)],
                subjectWorldPosition: nil
            )
        }
        let highResolutionCandidate = candidate(
            image: OwnedPixelBuffer(pixelBuffer),
            calibration: invalidCalibration
        )
        let streamingSnapshot = CaptureFrameSnapshot(
            image: pixelBuffer,
            depth: nil,
            confidence: nil,
            displayOrientation: .up,
            calibration: validCalibration,
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 1,
            exposureDuration: 0.01,
            trackingState: "normal",
            rawFeaturePoints: [SIMD3<Float>(0, 0, -1)],
            subjectWorldPosition: nil
        )
        let store = try CaptureStore(
            mode: .scene,
            baseDirectory: temporaryDirectory
        )

        let acceptance = try await acceptCaptureCandidate(
            highResolutionCandidate,
            streamingFallback: streamingSnapshot,
            in: store
        )

        XCTAssertEqual(acceptance.candidate.calibration, validCalibration)
        XCTAssertEqual(acceptance.commit.totalFrameCount, 1)
        XCTAssertNotNil(acceptance.fallbackMessage)
    }

    func testCaptureFrameSnapshotMaterializesAnOwnedImageCopy() throws {
        let pixelBuffer = try makeBGRAPixelBuffer(width: 2, height: 2)
        let snapshot = CaptureFrameSnapshot(
            image: pixelBuffer,
            depth: nil,
            confidence: nil,
            displayOrientation: .right,
            calibration: CaptureCalibrationRecord(
                width: 2,
                height: 2,
                fx: 2,
                fy: 2,
                cx: 1,
                cy: 1
            ),
            cameraToWorld: matrix_identity_float4x4,
            timestamp: 12,
            exposureDuration: 0.02,
            trackingState: "normal",
            rawFeaturePoints: [SIMD3<Float>(1, 2, 3)],
            subjectWorldPosition: nil
        )

        let candidate = try snapshot.materialize()
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        let source = try XCTUnwrap(CVPixelBufferGetBaseAddress(pixelBuffer))
            .assumingMemoryBound(to: UInt8.self)
        source[0] = 255
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])

        CVPixelBufferLockBaseAddress(candidate.image.value, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(candidate.image.value, .readOnly)
        }
        let copy = try XCTUnwrap(CVPixelBufferGetBaseAddress(candidate.image.value))
            .assumingMemoryBound(to: UInt8.self)
        XCTAssertEqual(copy[0], 16)
        XCTAssertEqual(candidate.timestamp, 12)
        XCTAssertEqual(candidate.rawFeaturePoints, [SIMD3<Float>(1, 2, 3)])
    }

    func testCaptureAdmissionUsesReturnedHighResolutionFrameAsBaseline() throws {
        let temporaryDirectory = try XCTUnwrap(temporaryDirectory)
        let store = try CaptureStore(
            mode: .scene,
            baseDirectory: temporaryDirectory
        )
        let controller = CaptureFrameAdmissionController()
        controller.start(store: store, subjectWorldPosition: nil)
        controller.updateDisplayOrientation(.right)
        defer { controller.finish() }

        let triggerTransform = matrix_identity_float4x4
        let first = try XCTUnwrap(controller.begin(
            cameraToWorld: triggerTransform,
            timestamp: 1
        ))
        var capturedTransform = triggerTransform
        capturedTransform.columns.3.x = 0.04
        XCTAssertTrue(controller.complete(
            first,
            committed: true,
            acceptedCameraToWorld: capturedTransform,
            acceptedTimestamp: 1.1
        ))

        XCTAssertNil(controller.begin(
            cameraToWorld: capturedTransform,
            timestamp: 1.5
        ))
        let next = try XCTUnwrap(controller.begin(
            cameraToWorld: triggerTransform,
            timestamp: 1.5
        ))
        XCTAssertTrue(controller.complete(next, committed: false))
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

    private func makeDepthPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        XCTAssertEqual(CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_DepthFloat32,
            nil, &buffer
        ), kCVReturnSuccess)
        let result = try XCTUnwrap(buffer)
        CVPixelBufferLockBaseAddress(result, [])
        defer { CVPixelBufferUnlockBaseAddress(result, []) }
        let rowStride = CVPixelBufferGetBytesPerRow(result) / MemoryLayout<Float>.stride
        let values = try XCTUnwrap(CVPixelBufferGetBaseAddress(result))
            .assumingMemoryBound(to: Float.self)
        for y in 0..<height {
            for x in 0..<width { values[y * rowStride + x] = 1 }
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

    func testImageExportPreservesUpOrientedJPEGAndPNGBytes() throws {
        let root = try XCTUnwrap(temporaryDirectory)
        let images = root.appending(
            path: "preserved-images",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(
            at: images,
            withIntermediateDirectories: true
        )
        let formats: [(type: UTType, sourceExtension: String, outputExtension: String)] = [
            (.jpeg, "jpeg", "jpg"),
            (.png, "png", "png"),
        ]

        for (index, format) in formats.enumerated() {
            let source = root.appending(
                path: "source-\(index).\(format.sourceExtension)"
            )
            try writeImage(to: source, type: format.type, orientation: .up)
            let sourceBytes = try Data(contentsOf: source)

            let filename = try RealityKitColmapExportBuilder.exportImage(
                source: source,
                to: images,
                imageID: index + 1,
                orientation: .up,
                context: CIContext(options: [.cacheIntermediates: false])
            )

            XCTAssertEqual(
                filename,
                String(
                    format: "image_%06d.%@",
                    index + 1,
                    format.outputExtension
                )
            )
            XCTAssertEqual(
                try Data(contentsOf: images.appending(path: filename)),
                sourceBytes
            )
        }
    }

    func testImageExportUsesLosslessSRGBPNGWhenOrientationChanges() throws {
        let root = try XCTUnwrap(temporaryDirectory)
        let images = root.appending(
            path: "normalized-images",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(
            at: images,
            withIntermediateDirectories: true
        )
        let source = root.appending(path: "rotated.jpeg")
        try writeImage(
            to: source,
            type: .jpeg,
            width: 64,
            height: 48,
            orientation: .right
        )
        let sourceBytes = try Data(contentsOf: source)

        let filename = try RealityKitColmapExportBuilder.exportImage(
            source: source,
            to: images,
            imageID: 7,
            orientation: .right,
            context: CIContext(options: [.cacheIntermediates: false])
        )

        XCTAssertEqual(filename, "image_000007.png")
        XCTAssertEqual(try Data(contentsOf: source), sourceBytes)
        let destination = images.appending(path: filename)
        let imageSource = try XCTUnwrap(
            CGImageSourceCreateWithURL(destination as CFURL, nil)
        )
        XCTAssertEqual(
            CGImageSourceGetType(imageSource) as String?,
            UTType.png.identifier
        )
        let image = try XCTUnwrap(
            CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
        )
        XCTAssertEqual(image.width, 48)
        XCTAssertEqual(image.height, 64)
        XCTAssertEqual(image.colorSpace?.name, CGColorSpace.sRGB)
        let properties = try XCTUnwrap(
            CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil)
                as? [CFString: Any]
        )
        XCTAssertEqual(
            (properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1,
            RealityKitColmapEXIFOrientation.up.rawValue
        )
    }

    func testEXIFOrientationPreservesProjectedRasterCoordinates() {
        let raw = RealityKitColmapCamera(
            width: 4_032,
            height: 3_024,
            fx: 3_007.25,
            fy: 2_983.75,
            cx: 1_987.5,
            cy: 1_493.25
        )
        let cameraPoints = [
            SIMD3<Double>(-0.42, -0.27, 1.1),
            SIMD3<Double>(0.31, -0.18, 1.7),
            SIMD3<Double>(-0.16, 0.39, 2.2),
            SIMD3<Double>(0.47, 0.29, 1.4),
        ]

        for orientation in [
            RealityKitColmapEXIFOrientation.up,
            .right,
            .down,
            .left,
        ] {
            let orientedCamera = RealityKitColmapExportBuilder.camera(
                raw,
                applying: orientation
            )
            let swapsDimensions = orientation == .right || orientation == .left
            XCTAssertEqual(
                orientedCamera.width,
                swapsDimensions ? raw.height : raw.width
            )
            XCTAssertEqual(
                orientedCamera.height,
                swapsDimensions ? raw.width : raw.height
            )

            for cameraPoint in cameraPoints {
                let rawPixel = project(cameraPoint, through: raw)
                let expected = orient(
                    rawPixel,
                    sourceWidth: raw.width,
                    sourceHeight: raw.height,
                    orientation: orientation
                )
                let actual = project(
                    orient(cameraPoint, orientation: orientation),
                    through: orientedCamera
                )

                XCTAssertEqual(
                    actual.x,
                    expected.x,
                    accuracy: 0.000_001,
                    "x projection for EXIF orientation \(orientation.rawValue)"
                )
                XCTAssertEqual(
                    actual.y,
                    expected.y,
                    accuracy: 0.000_001,
                    "y projection for EXIF orientation \(orientation.rawValue)"
                )
            }
        }
    }

    private func project(
        _ point: SIMD3<Double>,
        through camera: RealityKitColmapCamera
    ) -> SIMD2<Double> {
        SIMD2<Double>(
            camera.fx * point.x / point.z + camera.cx,
            camera.fy * point.y / point.z + camera.cy
        )
    }

    private func orient(
        _ point: SIMD3<Double>,
        orientation: RealityKitColmapEXIFOrientation
    ) -> SIMD3<Double> {
        switch orientation {
        case .right:
            SIMD3<Double>(-point.y, point.x, point.z)
        case .down:
            SIMD3<Double>(-point.x, -point.y, point.z)
        case .left:
            SIMD3<Double>(point.y, -point.x, point.z)
        default:
            point
        }
    }

    private func orient(
        _ point: SIMD2<Double>,
        sourceWidth: Int,
        sourceHeight: Int,
        orientation: RealityKitColmapEXIFOrientation
    ) -> SIMD2<Double> {
        switch orientation {
        case .right:
            SIMD2<Double>(Double(sourceHeight) - point.y, point.x)
        case .down:
            SIMD2<Double>(
                Double(sourceWidth) - point.x,
                Double(sourceHeight) - point.y
            )
        case .left:
            SIMD2<Double>(point.y, Double(sourceWidth) - point.x)
        default:
            point
        }
    }

    func testCameraRecordsPreservePerFrameIntrinsics() {
        let entries = [
            imageEntry(id: 1, fx: 1_000, fy: 1_001, cx: 512, cy: 384),
            imageEntry(id: 2, fx: 1_000.4, fy: 1_001.3, cx: 512.5, cy: 383.7),
            imageEntry(id: 3, fx: 1_001, fy: 1_001, cx: 512, cy: 384),
        ]

        let (cameras, assignments) = RealityKitColmapExportBuilder
            .cameraRecords(entries)

        XCTAssertEqual(cameras, entries.map(\.camera))
        XCTAssertEqual(assignments[1], 1)
        XCTAssertEqual(assignments[2], 2)
        XCTAssertEqual(assignments[3], 3)
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
        let camerasText = try String(
            contentsOf: sparse.appending(path: "cameras.txt"),
            encoding: .utf8
        )
        XCTAssertTrue(camerasText.contains("# Number of cameras: 3"))
        XCTAssertTrue(imagesText.contains("1 0.0 1.0 0.0 0.0"))
        XCTAssertTrue(imagesText.contains("1 image_000001.jpg"))
    }

    func testSparseWriterDocumentsAndSerializesObservationFreeSeedModel() throws {
        let sparse = try XCTUnwrap(temporaryDirectory).appending(
            path: "seed/sparse/0",
            directoryHint: .isDirectory
        )
        let entries = [imageEntry(id: 1), imageEntry(id: 2)]
        let points = [
            RealityKitColmapPoint(
                position: SIMD3<Float>(1, 2, 3),
                color: SIMD3<Float>(10, 20, 30)
            ),
            RealityKitColmapPoint(
                position: SIMD3<Float>(-1, -2, -3),
                color: SIMD3<Float>(40, 50, 60)
            ),
        ]

        try RealityKitColmapExportBuilder.writeSparseFiles(
            to: sparse,
            entries: entries,
            points: points
        )

        let notice = try String(
            contentsOf: sparse.appending(
                path: RealityKitColmapExportBuilder.seedModelNoticeFilename
            ),
            encoding: .utf8
        )
        XCTAssertTrue(notice.contains("camera-and-point seed"))
        XCTAssertTrue(notice.contains("zero `POINTS2D` observations"))
        XCTAssertTrue(notice.contains("zero tracks"))
        XCTAssertTrue(notice.contains("not a feature-matched or bundle-adjusted SfM reconstruction"))

        let imagesText = try String(
            contentsOf: sparse.appending(path: "images.txt"),
            encoding: .utf8
        )
        XCTAssertTrue(imagesText.contains("POINTS2D[] is intentionally empty"))
        let imageLines = imagesText.components(separatedBy: .newlines)
        let imageRecordIndexes = imageLines.indices.filter { index in
            let line = imageLines[index]
            return !line.isEmpty && !line.hasPrefix("#")
        }
        XCTAssertEqual(imageRecordIndexes.count, entries.count)
        for index in imageRecordIndexes {
            XCTAssertEqual(imageLines[index + 1], "")
        }

        let pointsText = try String(
            contentsOf: sparse.appending(path: "points3D.txt"),
            encoding: .utf8
        )
        XCTAssertTrue(pointsText.contains("TRACK[] is intentionally empty"))
        let pointRecords = pointsText.split(whereSeparator: \.isNewline)
            .filter { !$0.hasPrefix("#") }
        XCTAssertEqual(pointRecords.count, points.count)
        XCTAssertTrue(pointRecords.allSatisfy { $0.split(separator: " ").count == 8 })

        let imagesBinary = try Data(
            contentsOf: sparse.appending(path: "images.bin")
        )
        var imageOffset = 8
        XCTAssertEqual(littleEndianUInt64(in: imagesBinary, at: 0), UInt64(entries.count))
        for _ in entries {
            imageOffset += 64
            let filenameEnd = try XCTUnwrap(
                imagesBinary[imageOffset...].firstIndex(of: 0)
            )
            imageOffset = filenameEnd + 1
            XCTAssertEqual(
                littleEndianUInt64(in: imagesBinary, at: imageOffset),
                0
            )
            imageOffset += 8
        }
        XCTAssertEqual(imageOffset, imagesBinary.count)

        let pointsBinary = try Data(
            contentsOf: sparse.appending(path: "points3D.bin")
        )
        var pointOffset = 8
        XCTAssertEqual(littleEndianUInt64(in: pointsBinary, at: 0), UInt64(points.count))
        for _ in points {
            pointOffset += 43
            XCTAssertEqual(
                littleEndianUInt64(in: pointsBinary, at: pointOffset),
                0
            )
            pointOffset += 8
        }
        XCTAssertEqual(pointOffset, pointsBinary.count)
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

    private func littleEndianUInt64(in data: Data, at offset: Int) -> UInt64 {
        guard offset >= data.startIndex,
              offset + MemoryLayout<UInt64>.size <= data.endIndex else {
            XCTFail("UInt64 offset \(offset) exceeds \(data.count)-byte buffer")
            return .max
        }
        return (0..<MemoryLayout<UInt64>.size).reduce(into: UInt64(0)) {
            value, byteOffset in
            value |= UInt64(data[offset + byteOffset]) << (byteOffset * 8)
        }
    }

    private func writePNG(to url: URL) throws {
        try writeImage(to: url, type: .png, orientation: .up)
    }

    private func writeImage(
        to url: URL,
        type: UTType,
        width: Int = 4,
        height: Int = 3,
        orientation: RealityKitColmapEXIFOrientation
    ) throws {
        let context = try XCTUnwrap(CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        context.setFillColor(CGColor(red: 0.15, green: 0.35, blue: 0.75, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setFillColor(CGColor(red: 0.9, green: 0.2, blue: 0.1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: max(1, width / 2), height: 1))
        let image = try XCTUnwrap(context.makeImage())
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(
            url as CFURL,
            type.identifier as CFString,
            1,
            nil
        ))
        CGImageDestinationAddImage(
            destination,
            image,
            [kCGImagePropertyOrientation: orientation.rawValue] as CFDictionary
        )
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }
}
