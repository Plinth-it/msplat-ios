import MsplatCore

/// Aggregate pixel-error statistics for one non-empty sample population.
public struct DatasetReprojectionStatistics: Sendable, Equatable {
    public let sampleCount: UInt64
    public let meanPixels: Double
    public let rootMeanSquarePixels: Double
    public let maximumPixels: Double

    init(_ native: MsplatReprojectionErrorStatisticsV7) {
        sampleCount = native.sampleCount
        meanPixels = native.meanPixels
        rootMeanSquarePixels = native.rootMeanSquarePixels
        maximumPixels = native.maximumPixels
    }
}

/// Calibration and sparse-observation checks for one capture frame.
public struct DatasetFrameCaptureDiagnostics: Sendable, Equatable {
    public let frameIndex: Int
    public let frameID: String
    public let calibrationID: String
    public let width: Int
    public let height: Int
    public let hasTrainingMask: Bool
    public let observationCount: UInt64
    public let linkedObservationCount: UInt64
    /// Source observations outside `[0, width) × [0, height)`.
    public let observedOutsideFrameCount: UInt64
    public let reprojectedObservationCount: UInt64
    public let behindCameraObservationCount: UInt64
    public let nonFiniteProjectionCount: UInt64
    public let projectedOutsideFrameCount: UInt64
    /// `nil` when no linked observation could be reprojected.
    public let reprojectionError: DatasetReprojectionStatistics?

    public var unlinkedObservationCount: UInt64 {
        observationCount - linkedObservationCount
    }

    public var observedInsideFrameCount: UInt64 {
        observationCount - observedOutsideFrameCount
    }

    public var unprojectableObservationCount: UInt64 {
        behindCameraObservationCount + nonFiniteProjectionCount
    }

    public var projectedInsideFrameCount: UInt64 {
        reprojectedObservationCount - projectedOutsideFrameCount
    }

    init(
        native: MsplatFrameCaptureDiagnosticsV7,
        frame: DatasetFrame
    ) {
        frameIndex = Int(native.frameIndex)
        frameID = frame.id
        calibrationID = frame.calibrationID
        width = frame.calibration.width
        height = frame.calibration.height
        hasTrainingMask = frame.trainingMask != nil
        observationCount = native.observationCount
        linkedObservationCount = native.linkedObservationCount
        observedOutsideFrameCount = native.observedOutsideFrameCount
        reprojectedObservationCount = native.reprojectedObservationCount
        behindCameraObservationCount = native.behindCameraObservationCount
        nonFiniteProjectionCount = native.nonFiniteProjectionCount
        projectedOutsideFrameCount = native.projectedOutsideFrameCount
        reprojectionError = native.reprojectionError.sampleCount > 0
            ? DatasetReprojectionStatistics(native.reprojectionError) : nil
    }
}

/// Dataset-wide capture checks computed from canonical calibrations, poses,
/// sparse points, and observations without decoding images or starting Metal.
public struct DatasetCaptureDiagnostics: Sendable, Equatable {
    public let frameCount: UInt64
    public let maskedFrameCount: UInt64
    public let pointCount: UInt64
    public let observationCount: UInt64
    public let linkedObservationCount: UInt64
    /// Source observations outside their frame's declared half-open bounds.
    public let observedOutsideFrameCount: UInt64
    public let reprojectedObservationCount: UInt64
    public let behindCameraObservationCount: UInt64
    public let nonFiniteProjectionCount: UInt64
    public let projectedOutsideFrameCount: UInt64
    /// Points referenced by at least one linked observation.
    public let observedPointCount: UInt64
    /// Points linked from at least two distinct frames.
    public let multiViewPointCount: UInt64
    /// Largest number of distinct observing frames for one point.
    public let maximumTrackLength: UInt32
    /// Mean distinct-frame track length over observed points only.
    public let meanTrackLength: Double
    /// `nil` when no linked observation could be reprojected.
    public let reprojectionError: DatasetReprojectionStatistics?
    /// Statistics over optional source-reported per-point errors; `nil` when
    /// the descriptor contains no such metadata.
    public let sourcePointReprojectionError: DatasetReprojectionStatistics?
    public let frames: [DatasetFrameCaptureDiagnostics]

    public var unmaskedFrameCount: UInt64 {
        frameCount - maskedFrameCount
    }

    public var unlinkedObservationCount: UInt64 {
        observationCount - linkedObservationCount
    }

    public var observedInsideFrameCount: UInt64 {
        observationCount - observedOutsideFrameCount
    }

    public var unprojectableObservationCount: UInt64 {
        behindCameraObservationCount + nonFiniteProjectionCount
    }

    public var projectedInsideFrameCount: UInt64 {
        reprojectedObservationCount - projectedOutsideFrameCount
    }

    public var unobservedPointCount: UInt64 {
        pointCount - observedPointCount
    }

    public var singleViewPointCount: UInt64 {
        observedPointCount - multiViewPointCount
    }

    init(
        native: MsplatDatasetCaptureDiagnosticsV7,
        maskedFrameCount: UInt64,
        frames: [DatasetFrameCaptureDiagnostics]
    ) {
        frameCount = native.frameCount
        self.maskedFrameCount = maskedFrameCount
        pointCount = native.pointCount
        observationCount = native.observationCount
        linkedObservationCount = native.linkedObservationCount
        observedOutsideFrameCount = native.observedOutsideFrameCount
        reprojectedObservationCount = native.reprojectedObservationCount
        behindCameraObservationCount = native.behindCameraObservationCount
        nonFiniteProjectionCount = native.nonFiniteProjectionCount
        projectedOutsideFrameCount = native.projectedOutsideFrameCount
        observedPointCount = native.observedPointCount
        multiViewPointCount = native.multiViewPointCount
        maximumTrackLength = native.maximumTrackLength
        meanTrackLength = native.meanTrackLength
        reprojectionError = native.reprojectionError.sampleCount > 0
            ? DatasetReprojectionStatistics(native.reprojectionError) : nil
        sourcePointReprojectionError =
            native.sourcePointReprojectionError.sampleCount > 0
                ? DatasetReprojectionStatistics(
                    native.sourcePointReprojectionError
                ) : nil
        self.frames = frames
    }
}

public extension DatasetDescriptor {
    /// Checks capture geometry synchronously without reading image assets or
    /// reserving the process-global training engine.
    func captureDiagnostics() throws -> DatasetCaptureDiagnostics {
        let expectedABI = UInt32(MSPLAT_ABI_VERSION)
        let actualABI = msplat_abi_version()
        guard actualABI == expectedABI else {
            throw MsplatError.incompatibleABI(
                expected: expectedABI,
                actual: actualABI
            )
        }

        // The V6 marshaller accepts both masked and unmasked descriptors. The
        // sidecar is intentionally ignored because diagnostics use only the
        // canonical V5 geometry and observation payload.
        return try withUnsafeNativeDatasetDescriptorV6(self) {
            descriptor, _, _ in
            var nativeSummary = MsplatDatasetCaptureDiagnosticsV7()
            var nativeFrames = Array(
                repeating: MsplatFrameCaptureDiagnosticsV7(),
                count: frames.count
            )
            var nativeError = MsplatErrorInfo()

            try nativeFrames.withUnsafeMutableBufferPointer { frameBuffer in
                let status = msplat_dataset_capture_diagnostics_v7(
                    descriptor,
                    MemoryLayout<MsplatDatasetDescriptorV5>.size,
                    &nativeSummary,
                    MemoryLayout<MsplatDatasetCaptureDiagnosticsV7>.size,
                    frameBuffer.baseAddress,
                    frameBuffer.count,
                    MemoryLayout<MsplatFrameCaptureDiagnosticsV7>.stride,
                    &nativeError
                )
                try checkNativeStatus(status, error: &nativeError)
            }

            guard nativeSummary.frameCount == UInt64(frames.count) else {
                throw MsplatError.internalFailure(
                    "Native capture diagnostics returned an unexpected frame count"
                )
            }
            let frameDiagnostics = try nativeFrames.enumerated().map {
                index, native -> DatasetFrameCaptureDiagnostics in
                guard native.frameIndex == UInt32(index) else {
                    throw MsplatError.internalFailure(
                        "Native capture diagnostics returned an unexpected frame index"
                    )
                }
                return DatasetFrameCaptureDiagnostics(
                    native: native,
                    frame: frames[index]
                )
            }
            return DatasetCaptureDiagnostics(
                native: nativeSummary,
                maskedFrameCount: UInt64(
                    frames.lazy.filter { $0.trainingMask != nil }.count
                ),
                frames: frameDiagnostics
            )
        }
    }
}
