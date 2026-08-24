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

    func testTrainingPlanAccountsForSelectedMaskDiscovery() throws {
        let dimensions = try TrainingImageDimensions(width: 1_920, height: 1_440)
        let unmasked = try TrainingSession.makePlan(
            sourceDimensions: dimensions,
            steps: 2_000,
            profile: .preview
        )
        let masked = try TrainingSession.makePlan(
            sourceDimensions: dimensions,
            steps: 2_000,
            profile: .preview,
            includesTrainingMasks: true
        )

        XCTAssertFalse(unmasked.includesTrainingMasks)
        XCTAssertTrue(masked.includesTrainingMasks)
        XCTAssertGreaterThan(masked.estimatedPeakMemory, unmasked.estimatedPeakMemory)
    }
}
