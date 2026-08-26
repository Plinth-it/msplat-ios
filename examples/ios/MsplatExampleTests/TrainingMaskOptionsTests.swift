import CoreGraphics
import Foundation
import ImageIO
import Msplat
@testable import MsplatExample
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
            benchmark: benchmark
        )

        XCTAssertEqual(config.trainingMaskMode, .transparent)
        XCTAssertEqual(config.stopDensifyAt, 0)
        XCTAssertEqual(config.warmupLength, TrainingConfig().warmupLength)
        XCTAssertEqual(config.refineEvery, TrainingConfig().refineEvery)
        XCTAssertEqual(config.resetAlphaEvery, TrainingConfig().resetAlphaEvery)
        XCTAssertEqual(benchmark.maximumMissingMeasuredIterations, 0)
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

    private func writeNerfstudioManifest(
        to directory: URL,
        plyFilePath: String?,
        cameraModel: String = "OPENCV",
        framePaths: [String] = ["images/frame.png"]
    ) throws {
        let transform: [[Double]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        let frames: [[String: Any]] = framePaths.map { path in
            [
                "file_path": path,
                "transform_matrix": transform,
            ]
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
