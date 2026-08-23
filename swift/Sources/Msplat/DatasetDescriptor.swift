import Foundation
import MsplatCore

enum DatasetDescriptorNativeLimits {
    static let maximumStringBytes = Int(MSPLAT_DATASET_V5_MAX_STRING_BYTES)
    static let maximumFrames = Int(MSPLAT_DATASET_V5_MAX_FRAMES)
    static let maximumPoints = Int(MSPLAT_DATASET_V5_MAX_POINTS)
    static let maximumObservations = Int(
        MSPLAT_DATASET_V5_MAX_OBSERVATIONS
    )
}

/// The pixel orientation used by a frame's calibration and observations.
public enum DatasetRasterOrientation: UInt32, Sendable, Equatable {
    /// Calibration coordinates address pixels exactly as encoded in the file.
    case encodedPixels = 0
    /// ImageIO applies EXIF orientation after the caller has transformed the
    /// calibration, observations, and pose into that pixel frame. Mirrored
    /// EXIF tags are rejected because the camera model is right-handed.
    case exifNormalized = 1
}

/// Selects the soft-coverage channel extracted from a training-mask image.
public enum DatasetMaskCoverageChannel: UInt32, Sendable, Equatable {
    /// Rec. 709 luminance of the mask's premultiplied RGB values.
    case luminance = 0
    /// The mask image's alpha channel. Images without alpha are rejected.
    case alpha = 1
}

/// Optional soft per-pixel training coverage for one dataset frame.
public struct DatasetTrainingMask: Sendable, Equatable {
    public let url: URL
    public let coverageChannel: DatasetMaskCoverageChannel

    public init(
        url: URL,
        coverageChannel: DatasetMaskCoverageChannel = .luminance
    ) throws {
        guard url.isFileURL else {
            throw MsplatError.invalidArgument(
                "Dataset training-mask URLs must be file URLs"
            )
        }
        try validateDatasetString(
            url.path,
            name: "Dataset training-mask paths"
        )
        self.url = url
        self.coverageChannel = coverageChannel
    }
}

/// Immutable pinhole calibration and optional lens-distortion coefficients.
public struct DatasetCalibration: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let fx: Float
    public let fy: Float
    public let cx: Float
    public let cy: Float
    public let k1: Float
    public let k2: Float
    public let k3: Float
    public let p1: Float
    public let p2: Float

    public init(
        width: Int,
        height: Int,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        k1: Float = 0,
        k2: Float = 0,
        k3: Float = 0,
        p1: Float = 0,
        p2: Float = 0
    ) throws {
        guard width > 0, Int32(exactly: width) != nil,
              height > 0, Int32(exactly: height) != nil else {
            throw MsplatError.invalidArgument(
                "Dataset calibration dimensions must be in 1...2147483647"
            )
        }
        let values = [fx, fy, cx, cy, k1, k2, k3, p1, p2]
        guard values.allSatisfy(\.isFinite) else {
            throw MsplatError.invalidArgument(
                "Dataset calibration values must be finite"
            )
        }
        guard fx > 0, fy > 0 else {
            throw MsplatError.invalidArgument(
                "Dataset calibration focal lengths must be greater than zero"
            )
        }

        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2
    }
}

/// One image and camera in a canonical dataset.
public struct DatasetFrame: Sendable, Equatable {
    public let id: String
    public let calibrationID: String
    public let imageURL: URL
    public let rasterOrientation: DatasetRasterOrientation
    public let calibration: DatasetCalibration
    /// Row-major OpenGL camera-to-world transform (Y-up, Z-back).
    public let cameraToWorld: CameraPose
    /// Soft coverage in the same oriented source-pixel frame as the image.
    public let trainingMask: DatasetTrainingMask?

    public init(
        id: String,
        calibrationID: String,
        imageURL: URL,
        rasterOrientation: DatasetRasterOrientation,
        calibration: DatasetCalibration,
        cameraToWorld: CameraPose,
        trainingMask: DatasetTrainingMask? = nil
    ) throws {
        try validateDatasetString(id, name: "Dataset frame IDs")
        try validateDatasetString(
            calibrationID,
            name: "Dataset calibration IDs"
        )
        guard imageURL.isFileURL else {
            throw MsplatError.invalidArgument(
                "Dataset image URLs must be file URLs"
            )
        }
        try validateDatasetString(
            imageURL.path,
            name: "Dataset image paths"
        )

        self.id = id
        self.calibrationID = calibrationID
        self.imageURL = imageURL
        self.rasterOrientation = rasterOrientation
        self.calibration = calibration
        self.cameraToWorld = cameraToWorld
        self.trainingMask = trainingMask
    }
}

/// Structure-of-arrays sparse point storage matching the native descriptor.
public struct DatasetSparsePointSet: Sendable, Equatable {
    /// Flattened XYZ triplets.
    public let xyz: [Float]
    /// Flattened RGB8 triplets with the same cardinality as ``xyz``.
    public let rgb: [UInt8]
    /// Optional stable source IDs, one per point.
    public let sourceIDs: [UInt64]?
    /// Optional reprojection errors, one per point.
    public let reprojectionErrors: [Float]?

    public var count: Int { xyz.count / 3 }

    public init(
        xyz: [Float],
        rgb: [UInt8],
        sourceIDs: [UInt64]? = nil,
        reprojectionErrors: [Float]? = nil
    ) throws {
        let sourceIDs = sourceIDs?.isEmpty == true ? nil : sourceIDs
        let reprojectionErrors = reprojectionErrors?.isEmpty == true
            ? nil : reprojectionErrors
        guard !xyz.isEmpty, xyz.count.isMultiple(of: 3) else {
            throw MsplatError.invalidArgument(
                "Dataset XYZ storage must contain one or more complete triplets"
            )
        }
        guard xyz.count == rgb.count else {
            throw MsplatError.invalidArgument(
                "Dataset RGB storage must match the XYZ storage length"
            )
        }
        let pointCount = xyz.count / 3
        try validateDatasetPointCount(pointCount)
        guard xyz.allSatisfy(\.isFinite) else {
            throw MsplatError.invalidArgument("Dataset XYZ values must be finite")
        }
        if let sourceIDs {
            guard sourceIDs.count == pointCount else {
                throw MsplatError.invalidArgument(
                    "Dataset point source IDs must match the point count"
                )
            }
        }
        if let reprojectionErrors {
            guard reprojectionErrors.count == pointCount else {
                throw MsplatError.invalidArgument(
                    "Dataset reprojection errors must match the point count"
                )
            }
            guard reprojectionErrors.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                throw MsplatError.invalidArgument(
                    "Dataset reprojection errors must be finite and non-negative"
                )
            }
        }

        self.xyz = xyz
        self.rgb = rgb
        self.sourceIDs = sourceIDs
        self.reprojectionErrors = reprojectionErrors
    }
}

/// One source image feature, optionally linked to a sparse point.
///
/// `x` and `y` are image-edge coordinates in the frame's declared source
/// raster; the upper-left pixel center is `(0.5, 0.5)`. Native decode scaling,
/// rectification, and cropping do not mutate them.
///
/// Its frozen stored representation matches `MsplatSparseObservationV5` so a
/// descriptor can lend its observation storage directly to one synchronous C
/// call without allocating a second observation array.
@frozen
public struct DatasetObservation: Sendable, Equatable {
    @usableFromInline internal let nativeFrameIndex: UInt32
    @usableFromInline internal let nativeFrameObservationIndex: UInt32
    @usableFromInline internal let nativePointIndex: Int32
    @usableFromInline internal let nativeReserved: UInt32
    public let x: Float
    public let y: Float

    public var frameIndex: Int { Int(nativeFrameIndex) }
    public var frameObservationIndex: Int {
        Int(nativeFrameObservationIndex)
    }
    public var pointIndex: Int? {
        nativePointIndex == -1 ? nil : Int(nativePointIndex)
    }

    public init(
        frameIndex: Int,
        frameObservationIndex: Int,
        pointIndex: Int?,
        x: Float,
        y: Float
    ) throws {
        guard let nativeFrameIndex = UInt32(exactly: frameIndex) else {
            throw MsplatError.invalidArgument(
                "Dataset observation frame indexes must fit UInt32"
            )
        }
        guard let nativeFrameObservationIndex = UInt32(
            exactly: frameObservationIndex
        ) else {
            throw MsplatError.invalidArgument(
                "Dataset frame-observation indexes must fit UInt32"
            )
        }
        let nativePointIndex: Int32
        if let pointIndex {
            guard pointIndex >= 0,
                  let value = Int32(exactly: pointIndex) else {
                throw MsplatError.invalidArgument(
                    "Dataset observation point indexes must fit non-negative Int32"
                )
            }
            nativePointIndex = value
        } else {
            nativePointIndex = -1
        }
        guard x.isFinite, y.isFinite else {
            throw MsplatError.invalidArgument(
                "Dataset observation coordinates must be finite"
            )
        }

        self.nativeFrameIndex = nativeFrameIndex
        self.nativeFrameObservationIndex = nativeFrameObservationIndex
        self.nativePointIndex = nativePointIndex
        self.nativeReserved = 0
        self.x = x
        self.y = y
    }
}

/// Identifies the adapter and source that produced a descriptor.
public struct DatasetProvenance: Sendable, Equatable {
    public let adapter: String
    public let source: String

    public init(adapter: String, source: String) throws {
        try validateDatasetString(adapter, name: "Dataset provenance adapter")
        try validateDatasetString(source, name: "Dataset provenance source")
        self.adapter = adapter
        self.source = source
    }
}

/// An immutable canonical dataset that can be copied into the native runtime.
public struct DatasetDescriptor: Sendable, Equatable {
    public let frames: [DatasetFrame]
    public let points: DatasetSparsePointSet
    /// Strictly ordered by `(frameIndex, frameObservationIndex)` when present.
    public let observations: [DatasetObservation]
    public let provenance: DatasetProvenance?

    public init(
        frames: [DatasetFrame],
        points: DatasetSparsePointSet,
        observations: [DatasetObservation] = [],
        provenance: DatasetProvenance? = nil
    ) throws {
        try validateDatasetDescriptorCounts(
            frameCount: frames.count,
            pointCount: points.count,
            observationCount: observations.count
        )
        var previousKey: (frame: Int, observation: Int)?
        for observation in observations {
            guard observation.frameIndex < frames.count else {
                throw MsplatError.invalidArgument(
                    "Dataset observation frame index is out of range"
                )
            }
            if let pointIndex = observation.pointIndex {
                guard pointIndex < points.count else {
                    throw MsplatError.invalidArgument(
                        "Dataset observation point index is out of range"
                    )
                }
            }
            if let previousKey {
                let isStrictlyLater = observation.frameIndex > previousKey.frame ||
                    (observation.frameIndex == previousKey.frame &&
                     observation.frameObservationIndex > previousKey.observation)
                guard isStrictlyLater else {
                    throw MsplatError.invalidArgument(
                        "Dataset observations must be strictly ordered by frame and source index"
                    )
                }
            }
            previousKey = (
                observation.frameIndex,
                observation.frameObservationIndex
            )
        }

        self.frames = frames
        self.points = points
        self.observations = observations
        self.provenance = provenance
    }
}

func validateDatasetDescriptorCounts(
    frameCount: Int,
    pointCount: Int,
    observationCount: Int
) throws {
    guard (1...DatasetDescriptorNativeLimits.maximumFrames).contains(
        frameCount
    ) else {
        throw MsplatError.invalidArgument(
            "Dataset frame count exceeds the native descriptor limit"
        )
    }
    try validateDatasetPointCount(pointCount)
    guard (0...DatasetDescriptorNativeLimits.maximumObservations).contains(
        observationCount
    ) else {
        throw MsplatError.invalidArgument(
            "Dataset observation count exceeds the native descriptor limit"
        )
    }
}

func addDatasetStringByteCount(
    _ byteCount: Int,
    to total: inout Int
) throws {
    guard total >= 0,
          (0...DatasetDescriptorNativeLimits.maximumStringBytes).contains(
            byteCount
          ) else {
        throw MsplatError.invalidArgument(
            "Dataset string length exceeds the native descriptor limit"
        )
    }
    let (newTotal, overflow) = total.addingReportingOverflow(byteCount)
    guard !overflow else {
        throw MsplatError.invalidArgument(
            "Dataset string storage exceeds the Swift addressable range"
        )
    }
    total = newTotal
}

private func validateDatasetPointCount(_ pointCount: Int) throws {
    guard (1...DatasetDescriptorNativeLimits.maximumPoints).contains(
        pointCount
    ) else {
        throw MsplatError.invalidArgument(
            "Dataset point count exceeds the native descriptor limit"
        )
    }
}

private func validateDatasetString(_ value: String, name: String) throws {
    guard !value.isEmpty, !value.contains("\0") else {
        throw MsplatError.invalidArgument(
            "\(name) must be non-empty and contain no NUL bytes"
        )
    }
    guard value.utf8.count <= DatasetDescriptorNativeLimits.maximumStringBytes else {
        throw MsplatError.invalidArgument(
            "\(name) must not exceed the native descriptor UTF-8 byte limit"
        )
    }
}

private func datasetStringTableByteCount(
    descriptor: DatasetDescriptor,
    provenance: DatasetProvenance
) throws -> Int {
    var total = 0
    for frame in descriptor.frames {
        try addDatasetStringByteCount(frame.id.utf8.count, to: &total)
        try addDatasetStringByteCount(
            frame.calibrationID.utf8.count,
            to: &total
        )
        try addDatasetStringByteCount(frame.imageURL.path.utf8.count, to: &total)
        if let trainingMask = frame.trainingMask {
            try addDatasetStringByteCount(
                trainingMask.url.path.utf8.count,
                to: &total
            )
        }
    }
    try addDatasetStringByteCount(provenance.adapter.utf8.count, to: &total)
    try addDatasetStringByteCount(provenance.source.utf8.count, to: &total)
    return total
}

private struct NativeStringSlice {
    let offset: Int
    let length: Int
}

private struct NativeFrameStrings {
    let id: NativeStringSlice
    let calibrationID: NativeStringSlice
    let imagePath: NativeStringSlice
}

private struct NativeStringTable {
    var bytes: [UInt8] = []

    mutating func append(_ value: String) -> NativeStringSlice {
        let offset = bytes.count
        bytes.append(contentsOf: value.utf8)
        return NativeStringSlice(offset: offset, length: bytes.count - offset)
    }
}

/// Pins every caller-owned buffer for exactly one synchronous C invocation.
/// Native descriptor entry points deep-copy all views before returning; no
/// pointer escapes this body.
private func withUnsafeNativeDatasetDescriptorStorage<Result>(
    _ descriptor: DatasetDescriptor,
    includeFrameMasks: Bool,
    _ body: (
        UnsafePointer<MsplatDatasetDescriptorV5>,
        UnsafePointer<MsplatFrameMaskV6>?,
        Int
    ) throws -> Result
) throws -> Result {
    let provenance = try descriptor.provenance ?? DatasetProvenance(
        adapter: "swift",
        source: "descriptor"
    )
    let stringByteCount = try datasetStringTableByteCount(
        descriptor: descriptor,
        provenance: provenance
    )
    var strings = NativeStringTable()
    strings.bytes.reserveCapacity(stringByteCount)
    var frameStrings: [NativeFrameStrings] = []
    frameStrings.reserveCapacity(descriptor.frames.count)
    var maskPaths: [NativeStringSlice?] = []
    if includeFrameMasks {
        maskPaths.reserveCapacity(descriptor.frames.count)
    }
    for frame in descriptor.frames {
        frameStrings.append(NativeFrameStrings(
            id: strings.append(frame.id),
            calibrationID: strings.append(frame.calibrationID),
            imagePath: strings.append(frame.imageURL.path)
        ))
        if includeFrameMasks {
            maskPaths.append(
                frame.trainingMask.map { strings.append($0.url.path) }
            )
        }
    }
    let provenanceAdapter = strings.append(provenance.adapter)
    let provenanceSource = strings.append(provenance.source)

    return try strings.bytes.withUnsafeBufferPointer { stringBytes in
        guard let stringBase = stringBytes.baseAddress else {
            throw MsplatError.internalFailure(
                "Could not create native dataset string storage"
            )
        }

        func stringView(_ slice: NativeStringSlice) -> MsplatStringViewV5 {
            var view = MsplatStringViewV5()
            view.data = UnsafeRawPointer(
                stringBase.advanced(by: slice.offset)
            ).assumingMemoryBound(to: CChar.self)
            view.length = slice.length
            return view
        }

        var nativeFrames: [MsplatDatasetFrameV5] = []
        nativeFrames.reserveCapacity(descriptor.frames.count)
        var nativeMasks: [MsplatFrameMaskV6] = []
        if includeFrameMasks {
            nativeMasks.reserveCapacity(descriptor.frames.count)
        }
        for (index, frame) in descriptor.frames.enumerated() {
            var native = MsplatDatasetFrameV5()
            native.id = stringView(frameStrings[index].id)
            native.calibrationId = stringView(frameStrings[index].calibrationID)
            native.imagePath = stringView(frameStrings[index].imagePath)
            native.rasterOrientation = frame.rasterOrientation.rawValue
            native.reserved = 0

            guard let width = Int32(exactly: frame.calibration.width),
                  let height = Int32(exactly: frame.calibration.height) else {
                throw MsplatError.internalFailure(
                    "Validated dataset dimensions could not be represented natively"
                )
            }
            var calibration = MsplatCameraCalibrationV5()
            calibration.width = width
            calibration.height = height
            calibration.fx = frame.calibration.fx
            calibration.fy = frame.calibration.fy
            calibration.cx = frame.calibration.cx
            calibration.cy = frame.calibration.cy
            calibration.k1 = frame.calibration.k1
            calibration.k2 = frame.calibration.k2
            calibration.k3 = frame.calibration.k3
            calibration.p1 = frame.calibration.p1
            calibration.p2 = frame.calibration.p2
            native.calibration = calibration

            try withUnsafeMutableBytes(of: &native.cameraToWorld) { destination in
                try frame.cameraToWorld.elements.withUnsafeBytes { source in
                    guard destination.count == source.count else {
                        throw MsplatError.internalFailure(
                            "Native camera-pose layout does not contain 16 floats"
                        )
                    }
                    destination.copyBytes(from: source)
                }
            }
            nativeFrames.append(native)

            if includeFrameMasks {
                var nativeMask = MsplatFrameMaskV6()
                if let mask = frame.trainingMask,
                   let maskPath = maskPaths[index] {
                    nativeMask.maskPath = stringView(maskPath)
                    nativeMask.coverageChannel = mask.coverageChannel.rawValue
                }
                nativeMasks.append(nativeMask)
            }
        }

        return try nativeFrames.withUnsafeBufferPointer { frames in
            func invokeBody(
                maskBase: UnsafePointer<MsplatFrameMaskV6>?,
                maskCount: Int
            ) throws -> Result {
                try descriptor.points.xyz.withUnsafeBufferPointer {
                    pointXYZ in
                    try descriptor.points.rgb.withUnsafeBufferPointer {
                        pointRGB in
                        try withOptionalBufferPointer(
                            descriptor.points.sourceIDs
                        ) { pointSourceIDs, pointSourceIDCount in
                            try withOptionalBufferPointer(
                                descriptor.points.reprojectionErrors
                            ) { reprojectionErrors, reprojectionErrorCount in
                                try withUnsafeNativeObservations(
                                    descriptor.observations
                                ) { observations, observationCount in
                                    var native = MsplatDatasetDescriptorV5()
                                    native.frames = frames.baseAddress
                                    native.frameCount = frames.count
                                    native.pointXYZ = pointXYZ.baseAddress
                                    native.pointXYZCount = pointXYZ.count
                                    native.pointRGB = pointRGB.baseAddress
                                    native.pointRGBCount = pointRGB.count
                                    native.pointSourceIds = pointSourceIDs
                                    native.pointSourceIdCount = pointSourceIDCount
                                    native.pointReprojectionErrors =
                                        reprojectionErrors
                                    native.pointReprojectionErrorCount =
                                        reprojectionErrorCount
                                    native.observations = observations
                                    native.observationCount = observationCount
                                    native.provenanceAdapter = stringView(
                                        provenanceAdapter
                                    )
                                    native.provenanceSource = stringView(
                                        provenanceSource
                                    )
                                    native.reserved = (0, 0)
                                    return try withUnsafePointer(to: &native) {
                                        try body($0, maskBase, maskCount)
                                    }
                                }
                            }
                        }
                    }
                }
            }

            guard includeFrameMasks else {
                return try invokeBody(maskBase: nil, maskCount: 0)
            }
            return try nativeMasks.withUnsafeBufferPointer { masks in
                guard let maskBase = masks.baseAddress else {
                    throw MsplatError.internalFailure(
                        "Could not create native frame-mask storage"
                    )
                }
                return try invokeBody(
                    maskBase: maskBase,
                    maskCount: masks.count
                )
            }
        }
    }
}

func withUnsafeNativeDatasetDescriptor<Result>(
    _ descriptor: DatasetDescriptor,
    _ body: (UnsafePointer<MsplatDatasetDescriptorV5>) throws -> Result
) throws -> Result {
    guard descriptor.frames.allSatisfy({ $0.trainingMask == nil }) else {
        throw MsplatError.invalidArgument(
            "ABI v5 cannot represent dataset training masks"
        )
    }
    return try withUnsafeNativeDatasetDescriptorStorage(
        descriptor,
        includeFrameMasks: false
    ) { native, _, _ in
        try body(native)
    }
}

func withUnsafeNativeDatasetDescriptorV6<Result>(
    _ descriptor: DatasetDescriptor,
    _ body: (
        UnsafePointer<MsplatDatasetDescriptorV5>,
        UnsafePointer<MsplatFrameMaskV6>,
        Int
    ) throws -> Result
) throws -> Result {
    try withUnsafeNativeDatasetDescriptorStorage(
        descriptor,
        includeFrameMasks: true
    ) { native, masks, maskCount in
        guard let masks else {
            throw MsplatError.internalFailure(
                "Could not create native frame-mask storage"
            )
        }
        return try body(native, masks, maskCount)
    }
}

private func withUnsafeNativeObservations<Result>(
    _ observations: [DatasetObservation],
    _ body: (UnsafePointer<MsplatSparseObservationV5>?, Int) throws -> Result
) throws -> Result {
    try validateNativeObservationLayout()
    guard !observations.isEmpty else { return try body(nil, 0) }

    return try observations.withUnsafeBufferPointer { buffer in
        guard let baseAddress = buffer.baseAddress else {
            throw MsplatError.internalFailure(
                "Could not access non-empty dataset observation storage"
            )
        }
        return try baseAddress.withMemoryRebound(
            to: MsplatSparseObservationV5.self,
            capacity: buffer.count
        ) { native in
            try body(native, buffer.count)
        }
    }
}

private func validateNativeObservationLayout() throws {
    let compatible =
        MemoryLayout<DatasetObservation>.size ==
            MemoryLayout<MsplatSparseObservationV5>.size &&
        MemoryLayout<DatasetObservation>.stride ==
            MemoryLayout<MsplatSparseObservationV5>.stride &&
        MemoryLayout<DatasetObservation>.alignment ==
            MemoryLayout<MsplatSparseObservationV5>.alignment &&
        MemoryLayout<DatasetObservation>.offset(
            of: \DatasetObservation.nativeFrameIndex
        ) == MemoryLayout<MsplatSparseObservationV5>.offset(
            of: \MsplatSparseObservationV5.frameIndex
        ) &&
        MemoryLayout<DatasetObservation>.offset(
            of: \DatasetObservation.nativeFrameObservationIndex
        ) == MemoryLayout<MsplatSparseObservationV5>.offset(
            of: \MsplatSparseObservationV5.frameObservationIndex
        ) &&
        MemoryLayout<DatasetObservation>.offset(
            of: \DatasetObservation.nativePointIndex
        ) == MemoryLayout<MsplatSparseObservationV5>.offset(
            of: \MsplatSparseObservationV5.pointIndex
        ) &&
        MemoryLayout<DatasetObservation>.offset(
            of: \DatasetObservation.nativeReserved
        ) == MemoryLayout<MsplatSparseObservationV5>.offset(
            of: \MsplatSparseObservationV5.reserved
        ) &&
        MemoryLayout<DatasetObservation>.offset(of: \DatasetObservation.x) ==
            MemoryLayout<MsplatSparseObservationV5>.offset(
                of: \MsplatSparseObservationV5.x
            ) &&
        MemoryLayout<DatasetObservation>.offset(of: \DatasetObservation.y) ==
            MemoryLayout<MsplatSparseObservationV5>.offset(
                of: \MsplatSparseObservationV5.y
            )
    guard compatible else {
        throw MsplatError.internalFailure(
            "Dataset observation storage is incompatible with native ABI v5"
        )
    }
}

private func withOptionalBufferPointer<Element, Result>(
    _ values: [Element]?,
    _ body: (UnsafePointer<Element>?, Int) throws -> Result
) rethrows -> Result {
    guard let values, !values.isEmpty else { return try body(nil, 0) }
    return try values.withUnsafeBufferPointer {
        try body($0.baseAddress, $0.count)
    }
}
