import Foundation
import Msplat
@testable import MsplatExample
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

    func testDatasetFolderRejectsHigherPriorityNerfstudioMarker() throws {
        let directory = try XCTUnwrap(temporaryDirectory)
        try Data().write(to: directory.appending(path: "transforms.json"))

        XCTAssertNil(DatasetFolder(picked: directory))
        XCTAssertThrowsError(
            try DatasetFolder.initialSparsePointCount(at: directory)
        )
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
        XCTAssertEqual(iOSMemory.estimatedPeakMemory, 2_458_298_991)
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
}
