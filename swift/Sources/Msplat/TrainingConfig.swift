import MsplatCore

/// How a per-frame training mask participates in the objective.
public enum TrainingMaskMode: UInt32, CaseIterable, Sendable {
    /// Mask values weight RGB loss; masked-out pixels provide no supervision.
    case coverage = 0
    /// Mask values are target alpha; RGB is composited over the background and
    /// rendered alpha is supervised across the complete frame.
    case transparent = 1
}

/// Configuration for Gaussian splatting training.
public struct TrainingConfig: Sendable {
    public var iterations: Int32 = 30_000
    public var shDegree: Int32 = 3
    public var shDegreeInterval: Int32 = 1_000
    public var ssimWeight: Float = 0.2
    public var numDownscales: Int32 = 2
    public var resolutionSchedule: Int32 = 3_000
    public var refineEvery: Int32 = 100
    public var warmupLength: Int32 = 500
    public var resetAlphaEvery: Int32 = 30
    public var densifyGradThresh: Float = 0.0002
    public var densifySizeThresh: Float = 0.01
    public var stopScreenSizeAt: Int32 = 4_000
    /// Stop topology growth after this step. -1 uses half of the iteration budget.
    public var stopDensifyAt: Int32 = -1
    public var splitScreenSize: Float = 0.05
    public var keepCrs: Bool = false
    /// Learn bounded per-camera RGB gains while evaluating the training loss.
    /// The source images are sRGB encoded, so this is a photometric correction,
    /// not a physical linear-light exposure model. Canonical renders are unchanged.
    public var refinePhotometricGains: Bool = false
    /// Learn small, regularized camera-space pose corrections after warm-up.
    /// Imported geometry and canonical render, evaluation, and export stay unchanged.
    public var refineCameraPoses: Bool = false
    /// Treatment for frames that have a training mask. Frames without a mask
    /// remain ordinary opaque RGB targets in either mode. Transparent mode
    /// cannot be combined with `refinePhotometricGains`.
    public var trainingMaskMode: TrainingMaskMode = .coverage
    /// Weight of the full-frame L1 alpha term used by transparent mask mode.
    public var transparentAlphaLossWeight: Float = 0.1
    /// Legacy ABI field. Use `DatasetOptions.downscaleFactor`; this value does
    /// not affect training resolution.
    public var downscaleFactor: Float = 1.0
    /// Background color as (R, G, B) in [0, 1]. Default magenta — high contrast
    /// against typical scenes, makes under-reconstructed regions obvious.
    public var bgColor: (Float, Float, Float) = (0.6130, 0.0101, 0.3984)

    public init() {}

    /// Validate values before they reach native division, modulo, shifts, or allocations.
    public func validate() throws {
        guard (1...1_000_000).contains(iterations) else {
            throw MsplatError.invalidArgument("iterations must be in 1...1000000")
        }
        guard (0...4).contains(shDegree) else {
            throw MsplatError.invalidArgument("shDegree must be in 0...4")
        }
        guard shDegreeInterval > 0 else {
            throw MsplatError.invalidArgument("shDegreeInterval must be greater than zero")
        }
        guard ssimWeight.isFinite, (0...1).contains(ssimWeight) else {
            throw MsplatError.invalidArgument("ssimWeight must be finite and in 0...1")
        }
        guard (0...30).contains(numDownscales) else {
            throw MsplatError.invalidArgument("numDownscales must be in 0...30")
        }
        guard resolutionSchedule > 0 else {
            throw MsplatError.invalidArgument("resolutionSchedule must be greater than zero")
        }
        guard refineEvery > 0 else {
            throw MsplatError.invalidArgument("refineEvery must be greater than zero")
        }
        guard warmupLength >= 0 else {
            throw MsplatError.invalidArgument("warmupLength must not be negative")
        }
        guard resetAlphaEvery > 0 else {
            throw MsplatError.invalidArgument("resetAlphaEvery must be greater than zero")
        }
        let (_, resetOverflow) = resetAlphaEvery.multipliedReportingOverflow(by: refineEvery)
        guard !resetOverflow else {
            throw MsplatError.invalidArgument("resetAlphaEvery * refineEvery is too large")
        }
        guard densifyGradThresh.isFinite, densifyGradThresh >= 0 else {
            throw MsplatError.invalidArgument("densifyGradThresh must be finite and non-negative")
        }
        guard densifySizeThresh.isFinite, densifySizeThresh >= 0 else {
            throw MsplatError.invalidArgument("densifySizeThresh must be finite and non-negative")
        }
        guard stopScreenSizeAt >= 0 else {
            throw MsplatError.invalidArgument("stopScreenSizeAt must not be negative")
        }
        guard stopDensifyAt >= -1 else {
            throw MsplatError.invalidArgument("stopDensifyAt must be -1 or non-negative")
        }
        guard splitScreenSize.isFinite, splitScreenSize >= 0 else {
            throw MsplatError.invalidArgument("splitScreenSize must be finite and non-negative")
        }
        guard transparentAlphaLossWeight.isFinite,
              transparentAlphaLossWeight >= 0 else {
            throw MsplatError.invalidArgument(
                "transparentAlphaLossWeight must be finite and non-negative"
            )
        }
        guard trainingMaskMode != .transparent || !refinePhotometricGains else {
            throw MsplatError.invalidArgument(
                "Transparent training masks cannot be combined with photometric gain refinement"
            )
        }
        guard downscaleFactor.isFinite, (1...32).contains(downscaleFactor) else {
            throw MsplatError.invalidArgument("downscaleFactor must be finite and in 1...32")
        }
        let components = [bgColor.0, bgColor.1, bgColor.2]
        guard components.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
            throw MsplatError.invalidArgument("bgColor components must be finite and in 0...1")
        }
    }

    func toC() -> MsplatConfig {
        var c = msplat_default_config()
        c.iterations = iterations
        c.shDegree = shDegree
        c.shDegreeInterval = shDegreeInterval
        c.ssimWeight = ssimWeight
        c.numDownscales = numDownscales
        c.resolutionSchedule = resolutionSchedule
        c.refineEvery = refineEvery
        c.warmupLength = warmupLength
        c.resetAlphaEvery = resetAlphaEvery
        c.densifyGradThresh = densifyGradThresh
        c.densifySizeThresh = densifySizeThresh
        c.stopScreenSizeAt = stopScreenSizeAt
        c.stopDensifyAt = stopDensifyAt
        c.splitScreenSize = splitScreenSize
        c.keepCrs = keepCrs
        c.downscaleFactor = downscaleFactor
        c.bgColor = (bgColor.0, bgColor.1, bgColor.2)
        return c
    }

    func toRefinementOptionsV8() -> MsplatRefinementOptionsV8 {
        var options = msplat_default_refinement_options_v8()
        if refinePhotometricGains {
            options.flags |= UInt32(MSPLAT_REFINEMENT_PHOTOMETRIC_RGB_GAINS)
        }
        if refineCameraPoses {
            options.flags |= UInt32(MSPLAT_REFINEMENT_CAMERA_POSE_DELTAS)
        }
        return options
    }

    func toTrainingMaskOptionsV11() -> MsplatTrainingMaskOptionsV11 {
        var options = msplat_default_training_mask_options_v11()
        options.mode = trainingMaskMode.rawValue
        options.alphaLossWeight = transparentAlphaLossWeight
        return options
    }
}
