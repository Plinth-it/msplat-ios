import CoreGraphics
import Foundation
import ImageIO
@testable import Msplat
import UniformTypeIdentifiers
import XCTest

final class PoseCheckpointTests: XCTestCase {
    func testPoseCheckpointSaveLoadAndResume() async throws {
        try await runPoseCheckpointRegression()
    }
}

@MsplatRuntimeActor
private func runPoseCheckpointRegression() throws {
    let fixtureDirectory = FileManager.default.temporaryDirectory
        .appendingPathComponent("msplat-pose-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
        at: fixtureDirectory,
        withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: fixtureDirectory) }

    let anchorImageURL = fixtureDirectory.appendingPathComponent("anchor.png")
    let movableImageURL = fixtureDirectory.appendingPathComponent("movable.png")
    try writePoseFixturePNG(to: anchorImageURL, horizontalShift: 0)
    try writePoseFixturePNG(to: movableImageURL, horizontalShift: 3)

    let descriptor = try makePoseFixtureDescriptor(
        anchorImageURL: anchorImageURL,
        movableImageURL: movableImageURL
    )
    let config = makePoseFixtureConfig()
    let firstCheckpointURL = fixtureDirectory.appendingPathComponent("first.msplat")
    let reloadedCheckpointURL = fixtureDirectory.appendingPathComponent("reloaded.msplat")
    let resumedCheckpointURL = fixtureDirectory.appendingPathComponent("resumed.msplat")

    let firstSnapshot: PoseCheckpointSnapshot
    do {
        let session = try MsplatSession(
            dataset: descriptor,
            options: DatasetOptions(prefetchTrainingTargets: true),
            config: config,
            maximumGaussianCount: 5
        )
        defer { try? session.close() }

        _ = try session.step()
        _ = try session.step()
        try session.saveCheckpoint(to: firstCheckpointURL)
        firstSnapshot = try readPoseCheckpoint(at: firstCheckpointURL)
    }

    assertPoseSnapshot(firstSnapshot, iteration: 2, expectedMovableSteps: 1)

    let reloadedSnapshot: PoseCheckpointSnapshot
    let resumedSnapshot: PoseCheckpointSnapshot
    do {
        let session = try MsplatSession(
            dataset: descriptor,
            options: DatasetOptions(prefetchTrainingTargets: true),
            config: config,
            maximumGaussianCount: 5
        )
        defer { try? session.close() }

        XCTAssertEqual(try session.loadCheckpoint(from: firstCheckpointURL), 2)
        try session.saveCheckpoint(to: reloadedCheckpointURL)
        reloadedSnapshot = try readPoseCheckpoint(at: reloadedCheckpointURL)

        _ = try session.step()
        _ = try session.step()
        try session.saveCheckpoint(to: resumedCheckpointURL)
        resumedSnapshot = try readPoseCheckpoint(at: resumedCheckpointURL)
    }

    XCTAssertEqual(reloadedSnapshot, firstSnapshot)
    assertPoseSnapshot(resumedSnapshot, iteration: 4, expectedMovableSteps: 2)
    XCTAssertNotEqual(resumedSnapshot.deltas, firstSnapshot.deltas)
    XCTAssertNotEqual(resumedSnapshot.firstMoments, firstSnapshot.firstMoments)
    XCTAssertNotEqual(resumedSnapshot.secondMoments, firstSnapshot.secondMoments)
}

private func makePoseFixtureConfig() -> TrainingConfig {
    var config = TrainingConfig()
    config.iterations = 4
    config.shDegree = 0
    config.shDegreeInterval = 1
    config.ssimWeight = 0
    config.numDownscales = 0
    config.resolutionSchedule = 1
    config.refineEvery = 100
    config.warmupLength = 0
    config.stopDensifyAt = 0
    config.stopScreenSizeAt = 0
    config.refineCameraPoses = true
    config.bgColor = (0, 0, 0)
    return config
}

private func makePoseFixtureDescriptor(
    anchorImageURL: URL,
    movableImageURL: URL
) throws -> DatasetDescriptor {
    let calibration = try DatasetCalibration(
        width: 32,
        height: 32,
        fx: 24,
        fy: 24,
        cx: 16,
        cy: 16
    )
    let identity = try CameraPose(elements: [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ])
    let frames = try [
        DatasetFrame(
            id: "anchor",
            calibrationID: "anchor",
            imageURL: anchorImageURL,
            rasterOrientation: .encodedPixels,
            calibration: calibration,
            cameraToWorld: identity
        ),
        DatasetFrame(
            id: "movable",
            calibrationID: "movable",
            imageURL: movableImageURL,
            rasterOrientation: .encodedPixels,
            calibration: calibration,
            cameraToWorld: identity
        ),
    ]
    let points = try DatasetSparsePointSet(
        xyz: [
            -0.45, -0.25, -2.60,
             0.00, -0.35, -2.20,
             0.42, -0.12, -2.80,
            -0.28,  0.32, -2.50,
             0.30,  0.28, -2.35,
        ],
        rgb: [
            255, 48, 32,
            32, 255, 64,
            32, 80, 255,
            255, 220, 32,
            220, 32, 255,
        ]
    )
    return try DatasetDescriptor(frames: frames, points: points)
}

private func writePoseFixturePNG(to url: URL, horizontalShift: Int) throws {
    let width = 32
    let height = 32
    var rgba = [UInt8](repeating: 0, count: width * height * 4)
    for y in 0..<height {
        for x in 0..<width {
            let sourceX = min(width - 1, max(0, x - horizontalShift))
            let offset = (y * width + x) * 4
            rgba[offset] = UInt8(sourceX * 255 / (width - 1))
            rgba[offset + 1] = UInt8(y * 255 / (height - 1))
            rgba[offset + 2] = (sourceX / 4 + y / 4).isMultiple(of: 2)
                ? 32
                : 224
            rgba[offset + 3] = 255
        }
    }

    guard let provider = CGDataProvider(data: Data(rgba) as CFData),
          let image = CGImage(
              width: width,
              height: height,
              bitsPerComponent: 8,
              bitsPerPixel: 32,
              bytesPerRow: width * 4,
              space: CGColorSpaceCreateDeviceRGB(),
              bitmapInfo: CGBitmapInfo(
                  rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
              ),
              provider: provider,
              decode: nil,
              shouldInterpolate: false,
              intent: .defaultIntent
          ),
          let destination = CGImageDestinationCreateWithURL(
              url as CFURL,
              UTType.png.identifier as CFString,
              1,
              nil
          ) else {
        throw PoseFixtureError.invalidPNG
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
        throw PoseFixtureError.invalidPNG
    }
}

private struct PoseCheckpointSnapshot: Equatable {
    let iteration: UInt32
    let adamStep: UInt32
    let frameIDs: [String]
    let anchorIndex: Int
    let stepCounts: [UInt32]
    let basePoses: [Float]
    let deltas: [Float]
    let firstMoments: [Float]
    let secondMoments: [Float]
}

private func readPoseCheckpoint(at url: URL) throws -> PoseCheckpointSnapshot {
    var reader = PoseCheckpointReader(data: try Data(contentsOf: url))
    return try reader.readSnapshot()
}

private func assertPoseSnapshot(
    _ snapshot: PoseCheckpointSnapshot,
    iteration: UInt32,
    expectedMovableSteps: UInt32
) {
    XCTAssertEqual(snapshot.iteration, iteration)
    XCTAssertEqual(snapshot.adamStep, iteration)
    XCTAssertEqual(snapshot.frameIDs, ["anchor", "movable"])
    XCTAssertEqual(snapshot.anchorIndex, 0)
    XCTAssertEqual(snapshot.stepCounts, [0, expectedMovableSteps])
    let identityPose: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    XCTAssertEqual(snapshot.basePoses, identityPose + identityPose)
    XCTAssertEqual(Array(snapshot.deltas.prefix(6)), [Float](repeating: 0, count: 6))
    XCTAssertEqual(
        Array(snapshot.firstMoments.prefix(6)),
        [Float](repeating: 0, count: 6)
    )
    XCTAssertEqual(
        Array(snapshot.secondMoments.prefix(6)),
        [Float](repeating: 0, count: 6)
    )

    let movableDelta = Array(snapshot.deltas[6..<12])
    let movableFirstMoment = Array(snapshot.firstMoments[6..<12])
    let movableSecondMoment = Array(snapshot.secondMoments[6..<12])
    XCTAssertTrue(
        (movableDelta + movableFirstMoment + movableSecondMoment)
            .allSatisfy(\.isFinite)
    )
    XCTAssertTrue(movableDelta.contains { $0 != 0 })
    XCTAssertTrue(movableFirstMoment.contains { $0 != 0 })
    XCTAssertTrue(movableSecondMoment.contains { $0 > 0 })
    XCTAssertLessThanOrEqual(vectorNorm(movableDelta[0..<3]), 0.050_001)
    XCTAssertLessThanOrEqual(vectorNorm(movableDelta[3..<6]), 0.052_361)
}

private func vectorNorm(_ values: ArraySlice<Float>) -> Float {
    sqrt(values.reduce(0) { $0 + $1 * $1 })
}

private struct PoseCheckpointReader {
    private let data: Data
    private var cursor = 0

    init(data: Data) {
        self.data = data
    }

    mutating func readSnapshot() throws -> PoseCheckpointSnapshot {
        guard try readUInt32() == 0x4C50_534D else {
            throw PoseFixtureError.invalidCheckpoint
        }
        guard try readUInt32() == 3 else {
            throw PoseFixtureError.invalidCheckpoint
        }
        let iteration = try readUInt32()
        _ = try readUInt32() // Gaussian count
        _ = try readUInt32() // SH degree
        let adamStep = try readUInt32()
        try skip(6 * MemoryLayout<Float>.size + 2 * MemoryLayout<Float>.size)

        for _ in 0..<18 {
            try skipTensor()
        }

        guard try readUInt32() == 0 else {
            throw PoseFixtureError.invalidCheckpoint
        }
        guard try readUInt32() == 1 else {
            throw PoseFixtureError.invalidCheckpoint
        }
        let cameraCount = try checkedInt(readUInt32())
        guard cameraCount == 2 else {
            throw PoseFixtureError.invalidCheckpoint
        }
        let anchorIndex = try checkedInt(readUInt32())
        let frameIDs = try (0..<cameraCount).map { _ in try readString() }
        let stepCounts = try (0..<cameraCount).map { _ in try readUInt32() }
        let basePoses = try readTensor(expectedShape: [cameraCount, 16])
        let deltas = try readTensor(expectedShape: [cameraCount, 6])
        let firstMoments = try readTensor(expectedShape: [cameraCount, 6])
        let secondMoments = try readTensor(expectedShape: [cameraCount, 6])
        guard cursor == data.count else {
            throw PoseFixtureError.invalidCheckpoint
        }

        return PoseCheckpointSnapshot(
            iteration: iteration,
            adamStep: adamStep,
            frameIDs: frameIDs,
            anchorIndex: anchorIndex,
            stepCounts: stepCounts,
            basePoses: basePoses,
            deltas: deltas,
            firstMoments: firstMoments,
            secondMoments: secondMoments
        )
    }

    private mutating func skipTensor() throws {
        let rank = try checkedInt(readUInt32())
        for _ in 0..<rank {
            _ = try readInt64()
        }
        try skip(try checkedInt(readUInt64()))
    }

    private mutating func readTensor(expectedShape: [Int]) throws -> [Float] {
        let rank = try checkedInt(readUInt32())
        guard rank == expectedShape.count else {
            throw PoseFixtureError.invalidCheckpoint
        }

        var elementCount = 1
        for expectedDimension in expectedShape {
            let dimension = try readInt64()
            guard dimension == Int64(expectedDimension) else {
                throw PoseFixtureError.invalidCheckpoint
            }
            let result = elementCount.multipliedReportingOverflow(
                by: expectedDimension
            )
            guard !result.overflow else {
                throw PoseFixtureError.invalidCheckpoint
            }
            elementCount = result.partialValue
        }

        let expectedBytes = elementCount.multipliedReportingOverflow(
            by: MemoryLayout<Float>.size
        )
        guard !expectedBytes.overflow,
              try readUInt64() == UInt64(expectedBytes.partialValue) else {
            throw PoseFixtureError.invalidCheckpoint
        }
        return try (0..<elementCount).map { _ in
            Float(bitPattern: try readUInt32())
        }
    }

    private mutating func readString() throws -> String {
        let length = try checkedInt(readUInt32())
        guard length > 0 else {
            throw PoseFixtureError.invalidCheckpoint
        }
        let range = try takeRange(byteCount: length)
        guard let value = String(data: data[range], encoding: .utf8) else {
            throw PoseFixtureError.invalidCheckpoint
        }
        return value
    }

    private mutating func readUInt32() throws -> UInt32 {
        let range = try takeRange(byteCount: MemoryLayout<UInt32>.size)
        return data[range].withUnsafeBytes {
            UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))
        }
    }

    private mutating func readUInt64() throws -> UInt64 {
        let range = try takeRange(byteCount: MemoryLayout<UInt64>.size)
        return data[range].withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
    }

    private mutating func readInt64() throws -> Int64 {
        Int64(bitPattern: try readUInt64())
    }

    private mutating func skip(_ byteCount: Int) throws {
        _ = try takeRange(byteCount: byteCount)
    }

    private mutating func takeRange(byteCount: Int) throws -> Range<Int> {
        guard byteCount >= 0, cursor <= data.count,
              byteCount <= data.count - cursor else {
            throw PoseFixtureError.invalidCheckpoint
        }
        let range = cursor..<(cursor + byteCount)
        cursor += byteCount
        return range
    }

    private func checkedInt<Value: BinaryInteger>(_ value: Value) throws -> Int {
        guard let result = Int(exactly: value) else {
            throw PoseFixtureError.invalidCheckpoint
        }
        return result
    }
}

private enum PoseFixtureError: Error {
    case invalidPNG
    case invalidCheckpoint
}
