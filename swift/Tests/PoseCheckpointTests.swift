import CoreGraphics
import Foundation
import ImageIO
import Metal
@testable import Msplat
import UniformTypeIdentifiers
import XCTest

final class PoseCheckpointTests: XCTestCase {
    func testPoseCheckpointSaveLoadAndResume() async throws {
        try await runPoseCheckpointRegression()
    }

    func testGPUPreviewMatchesLegacyRenderAndOutlivesSession() async throws {
        try await runGPUPreviewRegression()
    }

    func testCampPoseCheckpointRoundTripAndModeMismatch() async throws {
        try await runCampPoseCheckpointRegression()
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

    do {
        var disabledConfig = config
        disabledConfig.refineCameraPoses = false
        let session = try MsplatSession(
            dataset: descriptor,
            config: disabledConfig,
            maximumGaussianCount: 5
        )
        defer { try? session.close() }
        XCTAssertEqual(try session.cameraPoseRefinementStates(), [])
    }

    let firstSnapshot: PoseCheckpointSnapshot
    let firstStates: [CameraPoseRefinementState]
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
        firstStates = try session.cameraPoseRefinementStates()
        try session.saveCheckpoint(to: firstCheckpointURL)
        firstSnapshot = try readPoseCheckpoint(at: firstCheckpointURL)
    }

    assertPoseSnapshot(firstSnapshot, iteration: 2, expectedMovableSteps: 1)
    assertPoseStates(
        firstStates,
        checkpoint: firstSnapshot,
        expectedMovableSteps: 1
    )

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

@MsplatRuntimeActor
private func runGPUPreviewRegression() async throws {
    let fixtureDirectory = FileManager.default.temporaryDirectory
        .appendingPathComponent("msplat-preview-\(UUID().uuidString)", isDirectory: true)
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
    let session = try MsplatSession(
        dataset: descriptor,
        options: DatasetOptions(prefetchTrainingTargets: true),
        config: makePoseFixtureConfig(),
        maximumGaussianCount: 5
    )
    defer { try? session.close() }

    _ = try session.step()
    let pose = try session.cameraPose(at: 0)
    let legacyFrame = try session.renderRGBA(pose: pose, referenceCamera: 0)
    let surface = try await completedPreview(
        from: session,
        pose: pose,
        referenceCamera: 0
    )

    XCTAssertEqual(surface.width, legacyFrame.width)
    XCTAssertEqual(surface.height, legacyFrame.height)
    XCTAssertEqual(surface.texture.width, legacyFrame.width)
    XCTAssertEqual(surface.texture.height, legacyFrame.height)
    XCTAssertEqual(surface.texture.pixelFormat, .bgra8Unorm)
    XCTAssertTrue(surface.texture.usage.contains(.shaderRead))

    try session.close()

    let previewRGBA = try readPreviewRGBA(surface)
    assertRGBAParity(
        expected: legacyFrame.data,
        actual: previewRGBA,
        tolerance: 1
    )
}

@MsplatRuntimeActor
private func runCampPoseCheckpointRegression() throws {
    let fixtureDirectory = FileManager.default.temporaryDirectory
        .appendingPathComponent("msplat-camp-pose-\(UUID().uuidString)", isDirectory: true)
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
    var campConfig = makePoseFixtureConfig()
    campConfig.cameraPoseConditioning = .camP
    let checkpointURL = fixtureDirectory.appendingPathComponent("camp.msplat")
    let reloadedURL = fixtureDirectory.appendingPathComponent("camp-reloaded.msplat")
    let singularURL = fixtureDirectory.appendingPathComponent("camp-singular.msplat")
    let missingBasisURL = fixtureDirectory.appendingPathComponent("camp-missing-basis.msplat")

    let rawModelBytes: UInt64
    do {
        let session = try MsplatSession(
            dataset: descriptor,
            config: makePoseFixtureConfig(),
            maximumGaussianCount: 5
        )
        defer { try? session.close() }
        rawModelBytes = try session.memoryMetrics().trainerModelBufferBytes
    }

    do {
        let session = try MsplatSession(
            dataset: descriptor,
            config: campConfig,
            maximumGaussianCount: 5
        )
        defer { try? session.close() }
        let campModelBytes = try session.memoryMetrics().trainerModelBufferBytes
        XCTAssertEqual(campModelBytes, rawModelBytes + UInt64(2 * 36 * 4))
        _ = try session.step()
        _ = try session.step()
        try session.saveCheckpoint(to: checkpointURL)
    }

    let snapshot = try readPoseCheckpoint(at: checkpointURL)
    XCTAssertEqual(snapshot.version, 4)
    XCTAssertEqual(snapshot.conditioning, .camP)
    XCTAssertEqual(snapshot.preconditionerReady, [1, 1])
    XCTAssertEqual(snapshot.preconditioners.count, 72)
    XCTAssertTrue(snapshot.preconditioners.allSatisfy(\.isFinite))
    XCTAssertEqual(snapshot.stepCounts, [0, 1])
    for camera in 0..<2 {
        for row in 0..<6 {
            for column in 0..<6 {
                XCTAssertEqual(
                    snapshot.preconditioners[camera * 36 + row * 6 + column],
                    snapshot.preconditioners[camera * 36 + column * 6 + row],
                    accuracy: 0.000_01
                )
            }
        }
    }
    let movablePreconditioner = Array(snapshot.preconditioners[36..<72])
    let identityPreconditioner = (0..<36).map { index in
        Float(index / 6 == index % 6 ? 1 : 0)
    }
    XCTAssertNotEqual(movablePreconditioner, identityPreconditioner)

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

    guard let preconditionerDataOffset = snapshot.preconditionerDataOffset,
          let readinessDataOffset = snapshot.preconditionerReadyDataOffset else {
        XCTFail("CamP checkpoint offsets are missing")
        return
    }
    var singularData = try Data(contentsOf: checkpointURL)
    var zeroBits = Float.zero.bitPattern.littleEndian
    withUnsafeBytes(of: &zeroBits) { bytes in
        singularData.replaceSubrange(
            preconditionerDataOffset..<(preconditionerDataOffset + bytes.count),
            with: bytes
        )
    }
    try singularData.write(to: singularURL)

    var missingBasisData = try Data(contentsOf: checkpointURL)
    missingBasisData[readinessDataOffset + 1] = 0
    try missingBasisData.write(to: missingBasisURL)

    do {
        let session = try MsplatSession(
            dataset: descriptor,
            config: campConfig,
            maximumGaussianCount: 5
        )
        defer { try? session.close() }
        XCTAssertThrowsError(try session.loadCheckpoint(from: singularURL))
        XCTAssertThrowsError(try session.loadCheckpoint(from: missingBasisURL))
        XCTAssertEqual(try session.loadCheckpoint(from: checkpointURL), 2)
        try session.saveCheckpoint(to: reloadedURL)
    }
    XCTAssertEqual(
        try Data(contentsOf: reloadedURL),
        try Data(contentsOf: checkpointURL)
    )

    do {
        let session = try MsplatSession(
            dataset: descriptor,
            config: makePoseFixtureConfig(),
            maximumGaussianCount: 5
        )
        defer { try? session.close() }
        XCTAssertThrowsError(try session.loadCheckpoint(from: checkpointURL))
    }
}

@MsplatRuntimeActor
private func completedPreview(
    from session: MsplatSession,
    pose: CameraPose,
    referenceCamera: Int
) async throws -> MetalPreviewSurface {
    let submission = try session.submitPreview(
        pose: pose,
        referenceCamera: referenceCamera
    )
    let surface = try await submission.waitUntilReady()
    XCTAssertTrue(try submission.poll())
    return surface
}

private func readPreviewRGBA(_ surface: MetalPreviewSurface) throws -> Data {
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm,
        width: surface.width,
        height: surface.height,
        mipmapped: false
    )
    descriptor.storageMode = .shared
    descriptor.usage = [.shaderRead]

    let device = surface.texture.device
    let stagingTexture = try XCTUnwrap(
        device.makeTexture(descriptor: descriptor),
        "Could not allocate the preview readback texture"
    )
    let commandQueue = try XCTUnwrap(
        device.makeCommandQueue(),
        "Could not allocate the preview readback command queue"
    )
    let commandBuffer = try XCTUnwrap(
        commandQueue.makeCommandBuffer(),
        "Could not allocate the preview readback command buffer"
    )
    let blitEncoder = try XCTUnwrap(
        commandBuffer.makeBlitCommandEncoder(),
        "Could not allocate the preview readback blit encoder"
    )
    blitEncoder.copy(
        from: surface.texture,
        sourceSlice: 0,
        sourceLevel: 0,
        sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
        sourceSize: MTLSize(width: surface.width, height: surface.height, depth: 1),
        to: stagingTexture,
        destinationSlice: 0,
        destinationLevel: 0,
        destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
    )
    blitEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if let error = commandBuffer.error {
        throw PreviewFixtureError.readbackFailed(error.localizedDescription)
    }
    guard commandBuffer.status == .completed else {
        throw PreviewFixtureError.readbackFailed(
            "Command buffer finished with status \(commandBuffer.status.rawValue)"
        )
    }

    let byteCount = surface.width * surface.height * 4
    var bgra = Data(count: byteCount)
    try bgra.withUnsafeMutableBytes { bytes in
        guard let destination = bytes.baseAddress else {
            throw PreviewFixtureError.inaccessibleReadbackStorage
        }
        stagingTexture.getBytes(
            destination,
            bytesPerRow: surface.width * 4,
            from: MTLRegionMake2D(0, 0, surface.width, surface.height),
            mipmapLevel: 0
        )
    }

    var rgba = bgra
    rgba.withUnsafeMutableBytes { bytes in
        let channels = bytes.bindMemory(to: UInt8.self)
        for offset in stride(from: 0, to: channels.count, by: 4) {
            let blue = channels[offset]
            channels[offset] = channels[offset + 2]
            channels[offset + 2] = blue
        }
    }
    return rgba
}

private func assertRGBAParity(
    expected: Data,
    actual: Data,
    tolerance: Int
) {
    let expectedBytes = [UInt8](expected)
    let actualBytes = [UInt8](actual)
    XCTAssertEqual(actualBytes.count, expectedBytes.count)
    guard actualBytes.count == expectedBytes.count else { return }

    for (index, pair) in zip(expectedBytes, actualBytes).enumerated() {
        let difference = abs(Int(pair.0) - Int(pair.1))
        guard difference <= tolerance else {
            XCTFail(
                "Preview byte \(index) differs by \(difference): "
                    + "legacy=\(pair.0), preview=\(pair.1)"
            )
            return
        }
    }
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
    let version: UInt32
    let iteration: UInt32
    let adamStep: UInt32
    let frameIDs: [String]
    let anchorIndex: Int
    let stepCounts: [UInt32]
    let basePoses: [Float]
    let conditioning: CameraPoseConditioning?
    let preconditionerReady: [UInt8]
    let preconditioners: [Float]
    let preconditionerReadyDataOffset: Int?
    let preconditionerDataOffset: Int?
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
    XCTAssertEqual(snapshot.version, 3)
    XCTAssertNil(snapshot.conditioning)
    XCTAssertEqual(snapshot.preconditionerReady, [])
    XCTAssertEqual(snapshot.preconditioners, [])
    XCTAssertEqual(snapshot.iteration, iteration)
    XCTAssertEqual(snapshot.adamStep, iteration)
    XCTAssertEqual(snapshot.frameIDs, ["anchor", "movable"])
    XCTAssertEqual(snapshot.anchorIndex, 0)
    XCTAssertEqual(snapshot.stepCounts, [0, expectedMovableSteps])
    let identityPose = poseFixtureIdentity()
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

private func assertPoseStates(
    _ states: [CameraPoseRefinementState],
    checkpoint: PoseCheckpointSnapshot,
    expectedMovableSteps: Int
) {
    XCTAssertEqual(states.map(\.canonicalCameraIndex), [0, 1])
    XCTAssertEqual(states.map(\.frameID), ["anchor", "movable"])
    XCTAssertEqual(states.map(\.isEnabled), [true, true])
    XCTAssertEqual(states.map(\.isAnchor), [true, false])
    XCTAssertEqual(states.map(\.optimizerStepCount), [0, expectedMovableSteps])
    guard states.count == 2 else { return }
    let anchor = states[0]
    let movable = states[1]
    XCTAssertEqual(anchor.translationDelta, [Float](repeating: 0, count: 3))
    XCTAssertEqual(anchor.rotationDelta, [Float](repeating: 0, count: 3))
    XCTAssertEqual(anchor.translationNorm, 0)
    XCTAssertEqual(anchor.rotationNorm, 0)
    assertFloatElementsEqual(
        anchor.correctedCameraToWorld.elements,
        poseFixtureIdentity(),
        accuracy: 0.000_001
    )
    XCTAssertTrue(movable.translationDelta.allSatisfy(\.isFinite))
    XCTAssertTrue(movable.rotationDelta.allSatisfy(\.isFinite))
    XCTAssertTrue(movable.translationNorm.isFinite)
    XCTAssertTrue(movable.rotationNorm.isFinite)
    XCTAssertEqual(
        movable.translationNorm,
        vectorNorm(movable.translationDelta[...]),
        accuracy: 0.000_001
    )
    XCTAssertEqual(
        movable.rotationNorm,
        vectorNorm(movable.rotationDelta[...]),
        accuracy: 0.000_001
    )

    let checkpointDelta = Array(checkpoint.deltas[6..<12])
    assertFloatElementsEqual(
        movable.translationDelta + movable.rotationDelta,
        checkpointDelta,
        accuracy: 0.000_001
    )
    assertFloatElementsEqual(
        movable.correctedCameraToWorld.elements,
        correctedIdentityFixturePose(delta: checkpointDelta),
        accuracy: 0.000_01
    )
}

private func correctedIdentityFixturePose(delta: [Float]) -> [Float] {
    XCTAssertEqual(delta.count, 6)
    guard delta.count == 6 else { return [] }

    // Both fixture cameras are identity at the origin, so auto-centering has
    // zero translation and unit scale. With F = diag(1, -1, -1), a left-view
    // correction [R, t] produces OpenGL C2W rotation F * R^T * F and position
    // -F * R^T * t.
    let correction = so3Rotation(axisAngle: Array(delta[3..<6]))
    let axisFlip: [Float] = [1, -1, -1]
    var cameraToWorld = [Float](repeating: 0, count: 16)
    for row in 0..<3 {
        for column in 0..<3 {
            let transposedCorrection = correction[column * 3 + row]
            cameraToWorld[row * 4 + column] =
                axisFlip[row] * transposedCorrection * axisFlip[column]
            cameraToWorld[row * 4 + 3] -=
                axisFlip[row] * transposedCorrection * delta[column]
        }
    }
    cameraToWorld[15] = 1
    return cameraToWorld
}

private func so3Rotation(axisAngle: [Float]) -> [Float] {
    XCTAssertEqual(axisAngle.count, 3)
    guard axisAngle.count == 3 else { return [] }

    let x = axisAngle[0]
    let y = axisAngle[1]
    let z = axisAngle[2]
    let thetaSquared = x * x + y * y + z * z
    let a: Float
    let b: Float
    if thetaSquared < 0.000_000_01 {
        let thetaFourth = thetaSquared * thetaSquared
        a = 1 - thetaSquared / 6 + thetaFourth / 120
        b = 0.5 - thetaSquared / 24 + thetaFourth / 720
    } else {
        let theta = thetaSquared.squareRoot()
        a = sin(theta) / theta
        b = (1 - cos(theta)) / thetaSquared
    }

    return [
        1 - b * (y * y + z * z), b * x * y - a * z, b * x * z + a * y,
        b * x * y + a * z, 1 - b * (x * x + z * z), b * y * z - a * x,
        b * x * z - a * y, b * y * z + a * x, 1 - b * (x * x + y * y),
    ]
}

private func poseFixtureIdentity() -> [Float] {
    [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
}

private func assertFloatElementsEqual(
    _ actual: [Float],
    _ expected: [Float],
    accuracy: Float
) {
    XCTAssertEqual(actual.count, expected.count)
    guard actual.count == expected.count else { return }
    for index in actual.indices {
        XCTAssertEqual(
            actual[index],
            expected[index],
            accuracy: accuracy,
            "Element \(index) differs"
        )
    }
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
        let version = try readUInt32()
        guard version == 3 || version == 4 else {
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
        let conditioning: CameraPoseConditioning?
        let preconditionerReady: [UInt8]
        let preconditioners: [Float]
        let preconditionerReadyDataOffset: Int?
        let preconditionerDataOffset: Int?
        if version >= 4 {
            conditioning = CameraPoseConditioning(rawValue: try readUInt32())
            guard conditioning != nil else {
                throw PoseFixtureError.invalidCheckpoint
            }
            preconditionerReadyDataOffset = cursor
            preconditionerReady = try (0..<cameraCount).map { _ in
                try readUInt8()
            }
            let preconditionerTensor = try readTensorWithDataOffset(
                expectedShape: [cameraCount, 36]
            )
            preconditioners = preconditionerTensor.values
            preconditionerDataOffset = preconditionerTensor.dataOffset
        } else {
            conditioning = nil
            preconditionerReady = []
            preconditioners = []
            preconditionerReadyDataOffset = nil
            preconditionerDataOffset = nil
        }
        let deltas = try readTensor(expectedShape: [cameraCount, 6])
        let firstMoments = try readTensor(expectedShape: [cameraCount, 6])
        let secondMoments = try readTensor(expectedShape: [cameraCount, 6])
        guard cursor == data.count else {
            throw PoseFixtureError.invalidCheckpoint
        }

        return PoseCheckpointSnapshot(
            version: version,
            iteration: iteration,
            adamStep: adamStep,
            frameIDs: frameIDs,
            anchorIndex: anchorIndex,
            stepCounts: stepCounts,
            basePoses: basePoses,
            conditioning: conditioning,
            preconditionerReady: preconditionerReady,
            preconditioners: preconditioners,
            preconditionerReadyDataOffset: preconditionerReadyDataOffset,
            preconditionerDataOffset: preconditionerDataOffset,
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
        try readTensorWithDataOffset(expectedShape: expectedShape).values
    }

    private mutating func readTensorWithDataOffset(
        expectedShape: [Int]
    ) throws -> (values: [Float], dataOffset: Int) {
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
        let dataOffset = cursor
        let values = try (0..<elementCount).map { _ in
            Float(bitPattern: try readUInt32())
        }
        return (values, dataOffset)
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

    private mutating func readUInt8() throws -> UInt8 {
        let range = try takeRange(byteCount: MemoryLayout<UInt8>.size)
        guard let value = data[range].first else {
            throw PoseFixtureError.invalidCheckpoint
        }
        return value
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

private enum PreviewFixtureError: LocalizedError, Sendable {
    case inaccessibleReadbackStorage
    case readbackFailed(String)

    var errorDescription: String? {
        switch self {
        case .inaccessibleReadbackStorage:
            "Could not access the preview readback storage"
        case .readbackFailed(let reason):
            "Preview readback failed: \(reason)"
        }
    }
}
