import Foundation
import MsplatCore
@testable import Msplat
import XCTest

final class DatasetDescriptorTests: XCTestCase {
    func testPublicDescriptorValuesAreSendableAndEquatable() throws {
        let descriptor = try makeDescriptor()
        requireSendable(descriptor)
        XCTAssertEqual(descriptor, descriptor)
        XCTAssertEqual(DatasetRasterOrientation.encodedPixels.rawValue, 0)
        XCTAssertEqual(DatasetRasterOrientation.exifNormalized.rawValue, 1)
        let mask = try DatasetTrainingMask(
            url: URL(fileURLWithPath: "/tmp/mask.png"),
            coverageChannel: .alpha
        )
        requireSendable(mask)
        XCTAssertEqual(mask, mask)
        XCTAssertEqual(DatasetMaskCoverageChannel.luminance.rawValue, 0)
        XCTAssertEqual(DatasetMaskCoverageChannel.alpha.rawValue, 1)
    }

    func testValueValidationRejectsMalformedInputs() throws {
        XCTAssertThrowsError(try DatasetCalibration(
            width: 0, height: 3, fx: 2, fy: 2, cx: 2, cy: 1.5
        ))
        XCTAssertThrowsError(try DatasetCalibration(
            width: 4, height: 3, fx: .nan, fy: 2, cx: 2, cy: 1.5
        ))

        let calibration = try makeCalibration()
        let pose = try identityPose()
        let remoteURL = try XCTUnwrap(URL(string: "https://example.com/image.png"))
        XCTAssertThrowsError(try DatasetFrame(
            id: "frame",
            calibrationID: "camera",
            imageURL: remoteURL,
            rasterOrientation: .encodedPixels,
            calibration: calibration,
            cameraToWorld: pose
        ))
        XCTAssertThrowsError(try DatasetTrainingMask(
            url: remoteURL,
            coverageChannel: .luminance
        ))
        XCTAssertThrowsError(try DatasetSparsePointSet(
            xyz: [0, 1], rgb: [0, 0]
        ))
        XCTAssertThrowsError(try DatasetSparsePointSet(
            xyz: [0, 1, 2], rgb: [0, 0, 0], reprojectionErrors: [-1]
        ))
        XCTAssertThrowsError(try DatasetObservation(
            frameIndex: -1,
            frameObservationIndex: 0,
            pointIndex: nil,
            x: 0,
            y: 0
        ))
    }

    func testDescriptorRejectsInvalidObservationOrderAndIndexes() throws {
        let first = try makeFrame(id: "first", filename: "a.png")
        let points = try makePoints()
        let second = try makeFrame(id: "second", filename: "b.png")
        let later = try DatasetObservation(
            frameIndex: 1,
            frameObservationIndex: 0,
            pointIndex: 0,
            x: 1,
            y: 2
        )
        let earlier = try DatasetObservation(
            frameIndex: 0,
            frameObservationIndex: 0,
            pointIndex: nil,
            x: 3,
            y: 4
        )
        XCTAssertThrowsError(try DatasetDescriptor(
            frames: [first, second],
            points: points,
            observations: [later, earlier]
        ))

        let outOfRange = try DatasetObservation(
            frameIndex: 0,
            frameObservationIndex: 0,
            pointIndex: points.count,
            x: 0,
            y: 0
        )
        XCTAssertThrowsError(try DatasetDescriptor(
            frames: [first], points: points, observations: [outOfRange]
        ))
    }

    func testDescriptorCapsRejectWithoutLargeFixtures() throws {
        XCTAssertEqual(DatasetDescriptorNativeLimits.maximumStringBytes, 1_048_576)
        XCTAssertEqual(DatasetDescriptorNativeLimits.maximumFrames, 1_000_000)
        XCTAssertEqual(DatasetDescriptorNativeLimits.maximumPoints, 100_000_000)
        XCTAssertEqual(
            DatasetDescriptorNativeLimits.maximumObservations,
            100_000_000
        )

        XCTAssertNoThrow(try validateDatasetDescriptorCounts(
            frameCount: DatasetDescriptorNativeLimits.maximumFrames,
            pointCount: DatasetDescriptorNativeLimits.maximumPoints,
            observationCount: DatasetDescriptorNativeLimits.maximumObservations
        ))
        XCTAssertThrowsError(try validateDatasetDescriptorCounts(
            frameCount: DatasetDescriptorNativeLimits.maximumFrames + 1,
            pointCount: 1,
            observationCount: 0
        ))
        XCTAssertThrowsError(try validateDatasetDescriptorCounts(
            frameCount: 1,
            pointCount: DatasetDescriptorNativeLimits.maximumPoints + 1,
            observationCount: 0
        ))
        XCTAssertThrowsError(try validateDatasetDescriptorCounts(
            frameCount: 1,
            pointCount: 1,
            observationCount:
                DatasetDescriptorNativeLimits.maximumObservations + 1
        ))

        var byteTotal = Int.max
        XCTAssertThrowsError(try addDatasetStringByteCount(1, to: &byteTotal))
        XCTAssertEqual(byteTotal, Int.max)
    }

    func testStringLimitCountsUTF8Bytes() throws {
        let byteLimit = DatasetDescriptorNativeLimits.maximumStringBytes
        let exactLimit = String(repeating: "é", count: byteLimit / 2)
        XCTAssertEqual(exactLimit.utf8.count, byteLimit)
        XCTAssertNoThrow(try DatasetProvenance(
            adapter: exactLimit,
            source: "project"
        ))

        let tooLong = exactLimit + "é"
        XCTAssertEqual(tooLong.utf8.count, byteLimit + 2)
        XCTAssertThrowsError(try DatasetProvenance(
            adapter: tooLong,
            source: "project"
        ))
        XCTAssertThrowsError(try DatasetFrame(
            id: tooLong,
            calibrationID: "camera",
            imageURL: URL(fileURLWithPath: "/tmp/image.png"),
            rasterOrientation: .encodedPixels,
            calibration: makeCalibration(),
            cameraToWorld: identityPose()
        ))
    }

    func testNativeMarshallingPreservesValuesAndDefaultProvenance() throws {
        let descriptor = try makeDescriptor(provenance: nil)

        try withUnsafeNativeDatasetDescriptorV6(descriptor) {
            pointer, masks, maskCount in
            let native = pointer.pointee
            XCTAssertEqual(native.frameCount, 2)
            XCTAssertEqual(native.pointXYZCount, 6)
            XCTAssertEqual(native.pointRGBCount, 6)
            XCTAssertEqual(native.pointSourceIdCount, 2)
            XCTAssertEqual(native.pointReprojectionErrorCount, 2)
            XCTAssertEqual(native.observationCount, 3)
            XCTAssertEqual(decode(native.provenanceAdapter), "swift")
            XCTAssertEqual(decode(native.provenanceSource), "descriptor")

            let frames = try XCTUnwrap(native.frames)
            XCTAssertEqual(decode(frames[0].id), "frame-a")
            XCTAssertEqual(decode(frames[0].calibrationId), "camera-a")
            XCTAssertEqual(decode(frames[0].imagePath), "/tmp/a.png")
            XCTAssertEqual(
                frames[0].rasterOrientation,
                DatasetRasterOrientation.encodedPixels.rawValue
            )
            XCTAssertEqual(frames[0].calibration.width, 4)
            XCTAssertEqual(frames[0].calibration.fy, 2.5)
            XCTAssertEqual(cameraPose(from: frames[0]), try identityPose().elements)

            let sourceIDs = try XCTUnwrap(native.pointSourceIds)
            XCTAssertEqual(Array(UnsafeBufferPointer(start: sourceIDs, count: 2)), [42, 84])
            let errors = try XCTUnwrap(native.pointReprojectionErrors)
            XCTAssertEqual(Array(UnsafeBufferPointer(start: errors, count: 2)), [0.25, 0.5])

            let observations = try XCTUnwrap(native.observations)
            XCTAssertEqual(observations[0].frameIndex, 0)
            XCTAssertEqual(observations[0].frameObservationIndex, 0)
            XCTAssertEqual(observations[0].pointIndex, 0)
            XCTAssertEqual(observations[0].reserved, 0)
            XCTAssertEqual(observations[1].pointIndex, -1)
            XCTAssertEqual(observations[2].frameIndex, 1)
            XCTAssertEqual(observations[2].pointIndex, 1)
            XCTAssertEqual(maskCount, 2)
            XCTAssertNil(masks[0].maskPath.data)
            XCTAssertEqual(masks[0].maskPath.length, 0)
            XCTAssertEqual(masks[0].coverageChannel, 0)
            XCTAssertNil(masks[1].maskPath.data)
        }
    }

    func testNativeMarshallingPreservesMaskSidecars() throws {
        let mask = try DatasetTrainingMask(
            url: URL(fileURLWithPath: "/tmp/soft-mask.png"),
            coverageChannel: .alpha
        )
        let frame = try DatasetFrame(
            id: "frame",
            calibrationID: "camera",
            imageURL: URL(fileURLWithPath: "/tmp/image.png"),
            rasterOrientation: .exifNormalized,
            calibration: makeCalibration(),
            cameraToWorld: identityPose(),
            trainingMask: mask
        )
        let descriptor = try DatasetDescriptor(
            frames: [frame], points: makePoints()
        )

        try withUnsafeNativeDatasetDescriptorV6(descriptor) {
            _, masks, maskCount in
            XCTAssertEqual(maskCount, 1)
            XCTAssertEqual(decode(masks[0].maskPath), "/tmp/soft-mask.png")
            XCTAssertEqual(
                masks[0].coverageChannel,
                DatasetMaskCoverageChannel.alpha.rawValue
            )
            XCTAssertEqual(masks[0].reserved, 0)
            XCTAssertEqual(masks[0].reserved2.0, 0)
            XCTAssertEqual(masks[0].reserved2.1, 0)
        }
    }

    func testObservationMarshallingBorrowsDescriptorStorage() throws {
        let descriptor = try makeDescriptor()

        try descriptor.observations.withUnsafeBufferPointer { source in
            try withUnsafeNativeDatasetDescriptor(descriptor) { pointer in
                XCTAssertEqual(
                    UnsafeRawPointer(pointer.pointee.observations),
                    UnsafeRawPointer(source.baseAddress)
                )
            }
        }
    }

    func testV5MarshallingRejectsTrainingMasks() throws {
        let mask = try DatasetTrainingMask(
            url: URL(fileURLWithPath: "/tmp/mask.png")
        )
        let frame = try DatasetFrame(
            id: "frame",
            calibrationID: "camera",
            imageURL: URL(fileURLWithPath: "/tmp/image.png"),
            rasterOrientation: .encodedPixels,
            calibration: makeCalibration(),
            cameraToWorld: identityPose(),
            trainingMask: mask
        )
        let descriptor = try DatasetDescriptor(
            frames: [frame], points: makePoints()
        )

        XCTAssertThrowsError(
            try withUnsafeNativeDatasetDescriptor(descriptor) { _ in () }
        )
    }

    func testNativeMarshallingUsesNilForAbsentPointMetadata() throws {
        let frame = try makeFrame(id: "frame", filename: "image.png")
        let points = try DatasetSparsePointSet(
            xyz: [1, 2, 3], rgb: [4, 5, 6],
            sourceIDs: [], reprojectionErrors: []
        )
        let provenance = try DatasetProvenance(adapter: "plinth", source: "project")
        let descriptor = try DatasetDescriptor(
            frames: [frame], points: points, provenance: provenance
        )

        try withUnsafeNativeDatasetDescriptor(descriptor) { pointer in
            let native = pointer.pointee
            XCTAssertNil(native.pointSourceIds)
            XCTAssertEqual(native.pointSourceIdCount, 0)
            XCTAssertNil(native.pointReprojectionErrors)
            XCTAssertEqual(native.pointReprojectionErrorCount, 0)
            XCTAssertNil(native.observations)
            XCTAssertEqual(native.observationCount, 0)
            XCTAssertEqual(decode(native.provenanceAdapter), "plinth")
            XCTAssertEqual(decode(native.provenanceSource), "project")
        }
    }

    func testSwiftImportedDescriptorLayoutsMatchTheCABI() {
        XCTAssertEqual(MemoryLayout<MsplatStringViewV5>.size, 16)
        XCTAssertEqual(MemoryLayout<MsplatCameraCalibrationV5>.size, 44)
        XCTAssertEqual(MemoryLayout<MsplatDatasetFrameV5>.size, 168)
        XCTAssertEqual(MemoryLayout<MsplatSparseObservationV5>.size, 24)
        XCTAssertEqual(MemoryLayout<MsplatDatasetDescriptorV5>.size, 144)
        XCTAssertEqual(MemoryLayout<MsplatFrameMaskV6>.size, 40)
        XCTAssertEqual(MemoryLayout<MsplatFrameMaskV6>.stride, 40)
        XCTAssertEqual(MemoryLayout<MsplatFrameMaskV6>.alignment, 8)

        XCTAssertEqual(MemoryLayout<DatasetObservation>.size, 24)
        XCTAssertEqual(MemoryLayout<DatasetObservation>.stride, 24)
        XCTAssertEqual(MemoryLayout<DatasetObservation>.alignment, 4)
        XCTAssertEqual(
            MemoryLayout<DatasetObservation>.offset(
                of: \DatasetObservation.nativeFrameIndex
            ),
            MemoryLayout<MsplatSparseObservationV5>.offset(
                of: \MsplatSparseObservationV5.frameIndex
            )
        )
        XCTAssertEqual(
            MemoryLayout<DatasetObservation>.offset(
                of: \DatasetObservation.nativeFrameObservationIndex
            ),
            MemoryLayout<MsplatSparseObservationV5>.offset(
                of: \MsplatSparseObservationV5.frameObservationIndex
            )
        )
        XCTAssertEqual(
            MemoryLayout<DatasetObservation>.offset(
                of: \DatasetObservation.nativePointIndex
            ),
            MemoryLayout<MsplatSparseObservationV5>.offset(
                of: \MsplatSparseObservationV5.pointIndex
            )
        )
        XCTAssertEqual(
            MemoryLayout<DatasetObservation>.offset(
                of: \DatasetObservation.nativeReserved
            ),
            MemoryLayout<MsplatSparseObservationV5>.offset(
                of: \MsplatSparseObservationV5.reserved
            )
        )
        XCTAssertEqual(
            MemoryLayout<DatasetObservation>.offset(of: \DatasetObservation.x),
            MemoryLayout<MsplatSparseObservationV5>.offset(
                of: \MsplatSparseObservationV5.x
            )
        )
        XCTAssertEqual(
            MemoryLayout<DatasetObservation>.offset(of: \DatasetObservation.y),
            MemoryLayout<MsplatSparseObservationV5>.offset(
                of: \MsplatSparseObservationV5.y
            )
        )
    }

    private func makeDescriptor(
        provenance: DatasetProvenance? = try? DatasetProvenance(
            adapter: "plinth",
            source: "project"
        )
    ) throws -> DatasetDescriptor {
        let observations = [
            try DatasetObservation(
                frameIndex: 0, frameObservationIndex: 0,
                pointIndex: 0, x: 10, y: 20
            ),
            try DatasetObservation(
                frameIndex: 0, frameObservationIndex: 1,
                pointIndex: nil, x: 30, y: 40
            ),
            try DatasetObservation(
                frameIndex: 1, frameObservationIndex: 0,
                pointIndex: 1, x: 50, y: 60
            ),
        ]
        return try DatasetDescriptor(
            frames: [
                makeFrame(id: "frame-a", filename: "a.png"),
                makeFrame(
                    id: "frame-b",
                    filename: "b.png",
                    orientation: .exifNormalized
                ),
            ],
            points: makePoints(),
            observations: observations,
            provenance: provenance
        )
    }

    private func makeCalibration() throws -> DatasetCalibration {
        try DatasetCalibration(
            width: 4, height: 3, fx: 2, fy: 2.5, cx: 2, cy: 1.5
        )
    }

    private func makeFrame(
        id: String,
        filename: String,
        orientation: DatasetRasterOrientation = .encodedPixels
    ) throws -> DatasetFrame {
        try DatasetFrame(
            id: id,
            calibrationID: id.replacingOccurrences(of: "frame", with: "camera"),
            imageURL: URL(fileURLWithPath: "/tmp/\(filename)"),
            rasterOrientation: orientation,
            calibration: makeCalibration(),
            cameraToWorld: identityPose()
        )
    }

    private func makePoints() throws -> DatasetSparsePointSet {
        try DatasetSparsePointSet(
            xyz: [1, 2, 3, -1, 0, 2],
            rgb: [255, 0, 0, 0, 255, 0],
            sourceIDs: [42, 84],
            reprojectionErrors: [0.25, 0.5]
        )
    }

    private func identityPose() throws -> CameraPose {
        try CameraPose(elements: [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ])
    }

    private func decode(_ view: MsplatStringViewV5) -> String {
        guard let data = view.data else { return "" }
        let bytes = UnsafeRawPointer(data).assumingMemoryBound(to: UInt8.self)
        return String(
            decoding: UnsafeBufferPointer(start: bytes, count: view.length),
            as: UTF8.self
        )
    }

    private func cameraPose(from frame: MsplatDatasetFrameV5) -> [Float] {
        withUnsafeBytes(of: frame.cameraToWorld) {
            Array($0.bindMemory(to: Float.self))
        }
    }

    private func requireSendable<Value: Sendable>(_ value: Value) {
        _ = value
    }
}
