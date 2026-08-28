import Foundation

/// Pixel dimensions of the largest source image before input decoding.
///
/// Use conservative maximum dimensions when dataset images are not uniform.
public struct TrainingImageDimensions: Sendable, Equatable {
    // Current image conversion and loss code still uses signed 32-bit flattened
    // indexes in a few places. This is far above a viable device resolution and
    // keeps every four-component pixel offset representable until those paths
    // are migrated to size_t/UInt64 throughout.
    private static let maximumNativeElementCount = Int64(Int32.max) / 4

    public let width: Int
    public let height: Int

    public init(width: Int, height: Int) throws {
        guard width > 0, height > 0 else {
            throw MsplatError.invalidArgument(
                "Input image dimensions must be greater than zero"
            )
        }
        guard width <= Int(Self.maximumNativeElementCount),
              height <= Int(Self.maximumNativeElementCount),
              Int64(width) * Int64(height) <= Self.maximumNativeElementCount else {
            throw MsplatError.invalidArgument(
                "Input image dimensions exceed the current native image index range"
            )
        }
        self.width = width
        self.height = height
    }
}

/// A requested training-resolution interval using one-based iteration numbers.
///
/// `downscaleFactor` is applied after ``TrainingPlan/inputDecodeScale`` and
/// must be a power of two. For example, a factor of 2 trains at half the
/// decoded width and height during the specified iterations.
public struct TrainingResolutionStage: Sendable, Equatable {
    public let iterations: ClosedRange<Int32>
    public let downscaleFactor: Int32

    public init(
        iterations: ClosedRange<Int32>,
        downscaleFactor: Int32
    ) throws {
        guard iterations.lowerBound >= 1 else {
            throw MsplatError.invalidArgument(
                "Resolution-stage iterations are one-based and must start at 1 or later"
            )
        }
        guard downscaleFactor > 0,
              downscaleFactor <= 1_073_741_824,
              downscaleFactor & (downscaleFactor - 1) == 0 else {
            throw MsplatError.invalidArgument(
                "Resolution-stage downscaleFactor must be a power of two in 1...1073741824"
            )
        }
        self.iterations = iterations
        self.downscaleFactor = downscaleFactor
    }
}

/// Concrete dimensions produced by a validated resolution stage.
public struct ResolvedTrainingResolutionStage: Sendable, Equatable {
    public let iterations: ClosedRange<Int32>
    public let downscaleFactor: Int32
    public let dimensions: TrainingImageDimensions

    public init(
        iterations: ClosedRange<Int32>,
        downscaleFactor: Int32,
        dimensions: TrainingImageDimensions
    ) {
        self.iterations = iterations
        self.downscaleFactor = downscaleFactor
        self.dimensions = dimensions
    }
}

/// Code-derived allocation terms for one training-resolution stage.
public struct TrainingStageMemoryEstimate: Sendable, Equatable {
    public let stage: ResolvedTrainingResolutionStage
    public let pixelCount: Int64
    public let tileCount: Int64
    /// Planning estimate for the frame's exact intersection count.
    /// Runtime correctness does not depend on this heuristic.
    public let estimatedIntersectionCount: Int64
    @available(*, deprecated, renamed: "estimatedIntersectionCount")
    public var hardIntersectionCapacity: Int64 {
        estimatedIntersectionCount
    }
    /// Estimated arena capacity, including runtime-equivalent growth slack.
    /// The native arena may grow beyond this value for a denser frame.
    public let intersectionCapacity: Int64
    public let chunkCount: Int64
    public let trainingCacheBytes: Int64
}

/// Intersection-attribute representation budgeted by ``TrainingPlan``.
///
/// The configured default mirrors `MSPLAT_INTERSECTION_ATTRIBUTES`; only the
/// exact value `packed` selects the larger fallback representation.
public enum TrainingIntersectionAttributeStorage: Sendable, Equatable {
    case gather
    case packed

    public static var configured: Self {
        ProcessInfo.processInfo.environment["MSPLAT_INTERSECTION_ATTRIBUTES"] == "packed"
            ? .packed
            : .gather
    }

    fileprivate var arenaBytesPerSlot: Int64 {
        switch self {
        case .gather: 16
        case .packed: 52
        }
    }
}

/// A checked peak-memory estimate derived from the current native allocations.
///
/// This is a planning estimate, not a runtime memory measurement. The
/// recommended headroom allows for allocations outside the modeled native
/// buffers, but cannot guarantee survival under changing system pressure.
public struct TrainingMemoryEstimate: Sendable, Equatable {
    public static let defaultIOSImageCacheBudgetBytes: Int64 = 512 * 1_024 * 1_024
    public static let defaultMacOSImageCacheBudgetBytes: Int64 = 2 * 1_024 * 1_024 * 1_024
#if os(iOS)
    public static let defaultNativeImageCacheBudgetBytes = defaultIOSImageCacheBudgetBytes
#else
    public static let defaultNativeImageCacheBudgetBytes = defaultMacOSImageCacheBudgetBytes
#endif
    /// Native cache budget after applying a valid `MSPLAT_IMAGE_CACHE_MB`
    /// process-environment override.
    public static let configuredNativeImageCacheBudgetBytes: Int64 = {
        let megabyte: Int64 = 1_024 * 1_024
        let rawValue = ProcessInfo.processInfo.environment["MSPLAT_IMAGE_CACHE_MB"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let rawValue,
           let megabytes = Int64(rawValue),
           megabytes > 0,
           megabytes <= Int64(Int32.max),
           megabytes <= Int64.max / megabyte {
            return megabytes * megabyte
        }
        return defaultNativeImageCacheBudgetBytes
    }()

    public let imageCacheBudgetBytes: Int64
    public let intersectionAttributeStorage: TrainingIntersectionAttributeStorage
    public let modelStorageBytes: Int64
    public let modelLifecycleBytes: Int64
    public let stages: [TrainingStageMemoryEstimate]
    public let peakTrainingCacheBytes: Int64
    public let largestImageCacheEntryBytes: Int64
    public let imageDecodeTransientBytes: Int64
    public let imageInsertionPeakBytes: Int64
    public let codeDerivedBytes: Int64
    public let recommendedHeadroomBytes: Int64
    public let estimatedPeakMemory: Int64
}

/// An explicit, validated description of input decoding and training resolution.
///
/// The current native scheduler changes resolution at fixed iteration
/// intervals. Initializing a plan rejects stage boundaries that this scheduler
/// cannot represent exactly, including its one-based first training step.
public struct TrainingPlan: Sendable, Equatable {
    /// Maximum source dimensions before `inputDecodeScale` is applied.
    /// Use conservative dimensions when dataset images are not uniform.
    public let inputDimensions: TrainingImageDimensions
    /// Dataset decode divisor. A value of 2 decodes at half width and height.
    public let inputDecodeScale: Float
    /// Total number of one-based training iterations covered by `stages`.
    public let iterationBudget: Int32
    /// Resolution stages covering every iteration in `1...iterationBudget`.
    public let stages: [TrainingResolutionStage]
    /// Highest spherical-harmonics degree that training will reach.
    public let targetSHDegree: Int32
    /// Hard Gaussian population limit enforced when the plan is used with
    /// ``MsplatSession``.
    public let maximumGaussianCount: Int
    /// Whether path-based plan sessions discover mask sidecars and whether
    /// decode-transient estimates include their per-frame UInt8 storage.
    public let includesTrainingMasks: Bool
    /// Training-only appearance compensation included in the configuration
    /// mapping and peak training-memory estimate.
    public let appearanceMode: AppearanceMode
    /// Intersection representation included in ``memoryEstimate``.
    public let intersectionAttributeStorage: TrainingIntersectionAttributeStorage
    /// Source dimensions after `inputDecodeScale` is applied.
    public let decodedInputDimensions: TrainingImageDimensions
    /// Effective pixel dimensions for every stage.
    public let resolvedStages: [ResolvedTrainingResolutionStage]
    /// Peak-memory estimate using the native cache default for this platform.
    public let memoryEstimate: TrainingMemoryEstimate
    /// Estimated peak bytes including recommended headroom.
    public var estimatedPeakMemory: Int64 {
        memoryEstimate.estimatedPeakMemory
    }

    public init(
        inputDimensions: TrainingImageDimensions,
        inputDecodeScale: Float,
        iterationBudget: Int32,
        stages: [TrainingResolutionStage],
        targetSHDegree: Int32,
        maximumGaussianCount: Int,
        includesTrainingMasks: Bool = false,
        appearanceMode: AppearanceMode = .none,
        intersectionAttributeStorage: TrainingIntersectionAttributeStorage = .configured
    ) throws {
        guard inputDecodeScale.isFinite, (1...32).contains(inputDecodeScale) else {
            throw MsplatError.invalidArgument(
                "inputDecodeScale must be finite and in 1...32"
            )
        }
        guard (1...1_000_000).contains(iterationBudget) else {
            throw MsplatError.invalidArgument(
                "iterationBudget must be in 1...1000000"
            )
        }
        guard (0...4).contains(targetSHDegree) else {
            throw MsplatError.invalidArgument("targetSHDegree must be in 0...4")
        }
        guard iterationBudget > targetSHDegree else {
            throw MsplatError.invalidArgument(
                "iterationBudget must be greater than targetSHDegree so the target degree is reached"
            )
        }
        guard (1...Int(Int32.max)).contains(maximumGaussianCount) else {
            throw MsplatError.invalidArgument(
                "maximumGaussianCount must be in 1...2147483647"
            )
        }

        try Self.validateCoverageAndNativeMapping(
            stages: stages,
            iterationBudget: iterationBudget
        )
        let decodedInputDimensions = try Self.decodedDimensions(
            inputDimensions,
            inputDecodeScale: inputDecodeScale
        )
        let resolvedStages = try Self.resolveStages(
            stages,
            decodedInputDimensions: decodedInputDimensions
        )
        let memoryEstimate = try Self.calculateMemoryEstimate(
            inputDimensions: inputDimensions,
            decodedInputDimensions: decodedInputDimensions,
            resolvedStages: resolvedStages,
            targetSHDegree: targetSHDegree,
            maximumGaussianCount: maximumGaussianCount,
            includesTrainingMasks: includesTrainingMasks,
            appearanceMode: appearanceMode,
            intersectionAttributeStorage: intersectionAttributeStorage,
            imageCacheBudgetBytes: TrainingMemoryEstimate.configuredNativeImageCacheBudgetBytes
        )

        self.inputDimensions = inputDimensions
        self.inputDecodeScale = inputDecodeScale
        self.iterationBudget = iterationBudget
        self.stages = stages
        self.targetSHDegree = targetSHDegree
        self.maximumGaussianCount = maximumGaussianCount
        self.includesTrainingMasks = includesTrainingMasks
        self.appearanceMode = appearanceMode
        self.intersectionAttributeStorage = intersectionAttributeStorage
        self.decodedInputDimensions = decodedInputDimensions
        self.resolvedStages = resolvedStages
        self.memoryEstimate = memoryEstimate
    }

    /// Produces dataset options controlled by this plan.
    public func makeDatasetOptions(
        evalMode: Bool = false,
        testEvery: Int32 = 8,
        prefetchTrainingTargets: Bool = false
    ) throws -> DatasetOptions {
        guard testEvery > 0 else {
            throw MsplatError.invalidArgument("testEvery must be greater than zero")
        }
        return DatasetOptions(
            downscaleFactor: inputDecodeScale,
            evalMode: evalMode,
            testEvery: testEvery,
            discoverTrainingMasks: includesTrainingMasks,
            prefetchTrainingTargets: prefetchTrainingTargets
        )
    }

    /// Maps the plan exactly onto the fields supported by `TrainingConfig`.
    ///
    /// Other training knobs are retained from `base`. The legacy, unused
    /// `TrainingConfig.downscaleFactor` is reset to 1 because input decoding is
    /// represented solely by ``inputDecodeScale``.
    public func makeTrainingConfig(
        startingFrom base: TrainingConfig = TrainingConfig()
    ) throws -> TrainingConfig {
        guard let firstStage = stages.first else {
            throw MsplatError.invalidArgument(
                "A training plan must contain at least one resolution stage"
            )
        }

        var config = base
        config.iterations = iterationBudget
        config.numDownscales = Int32(firstStage.downscaleFactor.trailingZeroBitCount)
        config.resolutionSchedule = Self.nativeResolutionSchedule(
            stages: stages,
            iterationBudget: iterationBudget
        )
        config.shDegree = targetSHDegree
        config.shDegreeInterval = max(1, iterationBudget / (targetSHDegree + 1))
        config.appearanceMode = appearanceMode
        config.downscaleFactor = 1
        try config.validate()
        return config
    }

    /// Recomputes the code-derived estimate for a custom native image-cache
    /// budget. Pass the matching byte count when `MSPLAT_IMAGE_CACHE_MB`
    /// overrides the native default.
    public func memoryEstimate(
        imageCacheBudgetBytes: Int64
    ) throws -> TrainingMemoryEstimate {
        try Self.calculateMemoryEstimate(
            inputDimensions: inputDimensions,
            decodedInputDimensions: decodedInputDimensions,
            resolvedStages: resolvedStages,
            targetSHDegree: targetSHDegree,
            maximumGaussianCount: maximumGaussianCount,
            includesTrainingMasks: includesTrainingMasks,
            appearanceMode: appearanceMode,
            intersectionAttributeStorage: intersectionAttributeStorage,
            imageCacheBudgetBytes: imageCacheBudgetBytes
        )
    }

    private static func validateCoverageAndNativeMapping(
        stages: [TrainingResolutionStage],
        iterationBudget: Int32
    ) throws {
        guard let firstStage = stages.first else {
            throw MsplatError.invalidArgument(
                "A training plan must contain at least one resolution stage"
            )
        }

        var expectedStart: Int32 = 1
        var previousFactor: Int32?
        for stage in stages {
            guard stage.iterations.lowerBound == expectedStart else {
                throw MsplatError.invalidArgument(
                    "Resolution stages must be ordered and cover every iteration without gaps or overlaps"
                )
            }
            guard stage.iterations.upperBound <= iterationBudget else {
                throw MsplatError.invalidArgument(
                    "Resolution stages must not extend beyond iterationBudget"
                )
            }
            if let previousFactor {
                guard stage.downscaleFactor != previousFactor else {
                    throw MsplatError.invalidArgument(
                        "Adjacent resolution stages must describe an actual resolution change"
                    )
                }
            }
            previousFactor = stage.downscaleFactor
            expectedStart = stage.iterations.upperBound + 1
        }

        guard expectedStart == iterationBudget + 1 else {
            throw MsplatError.invalidArgument(
                "Resolution stages must cover iterations 1 through iterationBudget"
            )
        }

        let schedule = nativeResolutionSchedule(
            stages: stages,
            iterationBudget: iterationBudget
        )
        let initialLevel = firstStage.downscaleFactor.trailingZeroBitCount

        for stage in stages {
            let lowerFactor = nativeDownscaleFactor(
                initialLevel: initialLevel,
                schedule: schedule,
                iteration: stage.iterations.lowerBound
            )
            let upperFactor = nativeDownscaleFactor(
                initialLevel: initialLevel,
                schedule: schedule,
                iteration: stage.iterations.upperBound
            )
            guard lowerFactor == stage.downscaleFactor,
                  upperFactor == stage.downscaleFactor else {
                throw MsplatError.invalidArgument(
                    "Resolution stages cannot be represented exactly by the current fixed-interval native scheduler"
                )
            }
        }
    }

    private static func nativeResolutionSchedule(
        stages: [TrainingResolutionStage],
        iterationBudget: Int32
    ) -> Int32 {
        if stages.count > 1 {
            return stages[1].iterations.lowerBound
        }
        return iterationBudget + 1
    }

    private static func nativeDownscaleFactor(
        initialLevel: Int,
        schedule: Int32,
        iteration: Int32
    ) -> Int32 {
        let transitions = Int(iteration / schedule)
        let remainingLevel = max(initialLevel - transitions, 0)
        return Int32(1) << remainingLevel
    }

    private static func decodedDimensions(
        _ inputDimensions: TrainingImageDimensions,
        inputDecodeScale: Float
    ) throws -> TrainingImageDimensions {
        // Match the native loader's Float division followed by truncation.
        let decodedWidth = Int(
            (Float(inputDimensions.width) / inputDecodeScale).rounded(.towardZero)
        )
        let decodedHeight = Int(
            (Float(inputDimensions.height) / inputDecodeScale).rounded(.towardZero)
        )
        guard decodedWidth > 0, decodedHeight > 0 else {
            throw MsplatError.invalidArgument(
                "inputDecodeScale produces a zero-sized decoded image"
            )
        }
        return try TrainingImageDimensions(width: decodedWidth, height: decodedHeight)
    }

    private static func resolveStages(
        _ stages: [TrainingResolutionStage],
        decodedInputDimensions: TrainingImageDimensions
    ) throws -> [ResolvedTrainingResolutionStage] {
        return try stages.map { stage in
            let factor = Int(stage.downscaleFactor)
            let width = decodedInputDimensions.width / factor
            let height = decodedInputDimensions.height / factor
            guard width > 0, height > 0 else {
                throw MsplatError.invalidArgument(
                    "A resolution stage produces a zero-sized training image"
                )
            }
            return ResolvedTrainingResolutionStage(
                iterations: stage.iterations,
                downscaleFactor: stage.downscaleFactor,
                dimensions: try TrainingImageDimensions(width: width, height: height)
            )
        }
    }

    private static func calculateMemoryEstimate(
        inputDimensions: TrainingImageDimensions,
        decodedInputDimensions: TrainingImageDimensions,
        resolvedStages: [ResolvedTrainingResolutionStage],
        targetSHDegree: Int32,
        maximumGaussianCount: Int,
        includesTrainingMasks: Bool,
        appearanceMode: AppearanceMode,
        intersectionAttributeStorage: TrainingIntersectionAttributeStorage,
        imageCacheBudgetBytes: Int64
    ) throws -> TrainingMemoryEstimate {
        guard imageCacheBudgetBytes > 0 else {
            throw MsplatError.invalidArgument(
                "imageCacheBudgetBytes must be greater than zero"
            )
        }

        let gaussianCount = Int64(maximumGaussianCount)
        let degree = Int64(targetSHDegree)
        let bases = try checkedProduct(
            [degree + 1, degree + 1],
            component: "spherical-harmonics bases"
        )
        let featureStride = try checkedSum(
            [try checkedProduct([3, bases], component: "feature stride"), 11],
            component: "feature stride"
        )
        let compactStride = max(
            try checkedProduct([3, bases - 1], component: "compact stride"),
            4
        )
        let compactElements = try checkedProduct(
            [gaussianCount, compactStride],
            component: "densification compact elements"
        )
        guard compactElements <= Int64(UInt32.max) else {
            throw MsplatError.invalidArgument(
                "maximumGaussianCount and targetSHDegree exceed the native densification index range"
            )
        }

        let gaussianBlocks = try checkedCeilDivide(
            gaussianCount,
            by: 1_024,
            component: "Gaussian blocks"
        )
        let modelStorageBytes = try checkedSum([
            try checkedProduct(
                [12, gaussianCount, featureStride],
                component: "model storage"
            ),
            try checkedProduct([36, gaussianCount], component: "model storage"),
            try checkedProduct(
                [4, gaussianCount, compactStride],
                component: "model storage"
            ),
            try checkedProduct([4, gaussianBlocks], component: "model storage"),
        ], component: "model storage")
        let modelLifecycleBytes = try checkedSum([
            modelStorageBytes,
            try checkedProduct([12, gaussianCount], component: "model lifecycle"),
            12,
            try checkedProduct(
                [4, gaussianCount, featureStride],
                component: "model lifecycle"
            ),
        ], component: "model lifecycle")

        var stageEstimates: [TrainingStageMemoryEstimate] = []
        stageEstimates.reserveCapacity(resolvedStages.count)
        var peakTrainingCacheBytes: Int64 = 0
        var largestImageCacheEntryBytes: Int64 = 0
        let decodedPixelCount = try checkedProduct([
            Int64(decodedInputDimensions.width),
            Int64(decodedInputDimensions.height),
        ], component: "decoded input pixels")
        let sourcePixelCount = try checkedProduct([
            Int64(inputDimensions.width),
            Int64(inputDimensions.height),
        ], component: "source input pixels")

        for stage in resolvedStages {
            let width = Int64(stage.dimensions.width)
            let height = Int64(stage.dimensions.height)
            let pixelCount = try checkedProduct(
                [width, height],
                component: "stage pixels"
            )
            guard pixelCount <= Int64(UInt32.max) else {
                throw MsplatError.invalidArgument(
                    "A resolution stage exceeds the native pixel index range"
                )
            }
            let tilesWide = try checkedCeilDivide(
                width,
                by: 16,
                component: "tile width"
            )
            let tilesHigh = try checkedCeilDivide(
                height,
                by: 16,
                component: "tile height"
            )
            let tileCount = try checkedProduct(
                [tilesWide, tilesHigh],
                component: "tile count"
            )
            // Exact runtime counts depend on camera and model geometry. An
            // iPhone 15 Pro scene measured 10.4 intersections/Gaussian at
            // 960x720 and 43.5 at 1920x1440. Use a padded 16-at-960x720
            // planning baseline and scale it by tile count (64 at 1920x1440),
            // bounded by the number of tiles a single Gaussian could hit.
            // Runtime correctness never depends on this estimate: the GPU
            // count pass sizes each frame before scatter.
            let scaledIntersectionsPerGaussian = try checkedCeilDivide(
                try checkedProduct(
                    [16, tileCount],
                    component: "resolution-scaled intersections"
                ),
                by: 2_700,
                component: "resolution-scaled intersections"
            )
            let estimatedIntersectionsPerGaussian = min(
                tileCount, max(16, scaledIntersectionsPerGaussian))
            let estimatedIntersectionCount = try checkedProduct(
                [estimatedIntersectionsPerGaussian, gaussianCount],
                component: "estimated exact intersections"
            )
            let arenaSlack = max(estimatedIntersectionCount / 4, 4_096)
            let capacityWithSlack = try checkedSum(
                [estimatedIntersectionCount, arenaSlack],
                component: "exact intersection arena headroom"
            )
            let intersectionCapacity = min(
                capacityWithSlack, Int64(Int32.max))

            let chunkCount: Int64
            if tileCount >= 400 {
                chunkCount = 1
            } else {
                let averageIntersectionsPerTile =
                    estimatedIntersectionCount / tileCount
                let conservativeChunks = try checkedCeilDivide(
                    try checkedProduct(
                        [6, averageIntersectionsPerTile],
                        component: "conservative chunk capacity"
                    ),
                    by: 512,
                    component: "conservative chunk count"
                )
                let absoluteChunks = try checkedCeilDivide(
                    estimatedIntersectionCount,
                    by: 512,
                    component: "absolute chunk count"
                )
                chunkCount = min(max(1, conservativeChunks), absoluteChunks)
            }

            var trainingCacheBytes = try checkedSum([
                // 52 bytes of projected forward state plus 36 bytes of
                // raster-backward gradients per Gaussian. Geometry gradients
                // remain register-local in the terminal update kernel.
                try checkedProduct(
                    [88, gaussianCount],
                    component: "training Gaussian cache"
                ),
                try checkedProduct(
                    // Gather retains two key arenas; packed additionally owns
                    // three float3 attribute arrays. Runtime omits the second
                    // key arena until a tile exceeds 2,048 entries.
                    [intersectionAttributeStorage.arenaBytesPerSlot, intersectionCapacity],
                    component: "exact intersection arenas"
                ),
                try checkedProduct([116, pixelCount], component: "training pixel cache"),
                // Counts, inclusive offsets, int2 ranges, and compact sortable IDs.
                try checkedProduct([20, tileCount], component: "training tile metadata"),
                8,
            ], component: "training cache")
            if chunkCount > 1 {
                trainingCacheBytes = try checkedSum([
                    trainingCacheBytes,
                    try checkedProduct(
                        [36, chunkCount, pixelCount],
                        component: "chunked training cache"
                    ),
                ], component: "training cache")
            }
            if appearanceMode == .ppisp {
                trainingCacheBytes = try checkedSum([
                    trainingCacheBytes,
                    try checkedProduct(
                        [12, pixelCount],
                        component: "PPISP full-resolution RGB scratch"
                    ),
                ], component: "training cache")
            }

            // The native cache releases decoded and pyramid pixels after a
            // successful upload. Its retained stage target is one UInt8 RGBA
            // buffer. Masked targets also retain one UInt8 activity byte per
            // 16x16 render tile; coverage itself stays packed in RGBA alpha.
            let imageCacheEntryBytes = try checkedSum([
                try checkedProduct(
                    [4, pixelCount],
                    component: "compact stage image cache entry"
                ),
                includesTrainingMasks ? tileCount : 0,
            ], component: "compact stage image cache entry")

            peakTrainingCacheBytes = max(peakTrainingCacheBytes, trainingCacheBytes)
            largestImageCacheEntryBytes = max(
                largestImageCacheEntryBytes,
                imageCacheEntryBytes
            )
            stageEstimates.append(TrainingStageMemoryEstimate(
                stage: stage,
                pixelCount: pixelCount,
                tileCount: tileCount,
                estimatedIntersectionCount: estimatedIntersectionCount,
                intersectionCapacity: intersectionCapacity,
                chunkCount: chunkCount,
                trainingCacheBytes: trainingCacheBytes
            ))
        }

        // RGB remains compact RGBA8 throughout decode and calibrated image
        // transforms. Masks decode at source resolution before exact area
        // reduction because ImageIO does not promise a coverage-preserving
        // thumbnail filter. During that phase the retained target is 4P and
        // the source mask's explicit/decode-backed storage is 8S.
        let targetResolutionDecodeBytes = try checkedProduct(
            [includesTrainingMasks ? 29 : 24, decodedPixelCount],
            component: "target-resolution image decode transient"
        )
        let imageDecodeTransientBytes: Int64
        if includesTrainingMasks {
            let sourceMaskDecodeBytes = try checkedSum([
                try checkedProduct(
                    [4, decodedPixelCount],
                    component: "decoded RGBA8 retained during mask decode"
                ),
                try checkedProduct(
                    [8, sourcePixelCount],
                    component: "source-resolution mask decode"
                ),
            ], component: "source-resolution mask decode transient")
            imageDecodeTransientBytes = max(
                targetResolutionDecodeBytes, sourceMaskDecodeBytes)
        } else {
            imageDecodeTransientBytes = targetResolutionDecodeBytes
        }
        let imageInsertionPeakBytes = try checkedSum([
            max(imageCacheBudgetBytes, largestImageCacheEntryBytes),
            max(largestImageCacheEntryBytes, imageDecodeTransientBytes),
        ], component: "image insertion peak")
        let codeDerivedBytes = try checkedSum([
            modelLifecycleBytes,
            peakTrainingCacheBytes,
            imageInsertionPeakBytes,
        ], component: "code-derived peak")
        let twentyPercentHeadroom = try checkedCeilDivide(
            codeDerivedBytes,
            by: 5,
            component: "recommended headroom"
        )
        let recommendedHeadroomBytes = max(
            128 * 1_024 * 1_024,
            twentyPercentHeadroom
        )
        let estimatedPeakMemory = try checkedSum(
            [codeDerivedBytes, recommendedHeadroomBytes],
            component: "estimated peak memory"
        )

        return TrainingMemoryEstimate(
            imageCacheBudgetBytes: imageCacheBudgetBytes,
            intersectionAttributeStorage: intersectionAttributeStorage,
            modelStorageBytes: modelStorageBytes,
            modelLifecycleBytes: modelLifecycleBytes,
            stages: stageEstimates,
            peakTrainingCacheBytes: peakTrainingCacheBytes,
            largestImageCacheEntryBytes: largestImageCacheEntryBytes,
            imageDecodeTransientBytes: imageDecodeTransientBytes,
            imageInsertionPeakBytes: imageInsertionPeakBytes,
            codeDerivedBytes: codeDerivedBytes,
            recommendedHeadroomBytes: recommendedHeadroomBytes,
            estimatedPeakMemory: estimatedPeakMemory
        )
    }

    private static func checkedProduct(
        _ values: [Int64],
        component: String
    ) throws -> Int64 {
        var result: Int64 = 1
        for value in values {
            let (product, overflow) = result.multipliedReportingOverflow(by: value)
            guard !overflow else {
                throw MsplatError.invalidArgument(
                    "Memory estimate overflow while calculating \(component)"
                )
            }
            result = product
        }
        return result
    }

    private static func checkedSum(
        _ values: [Int64],
        component: String
    ) throws -> Int64 {
        var result: Int64 = 0
        for value in values {
            let (sum, overflow) = result.addingReportingOverflow(value)
            guard !overflow else {
                throw MsplatError.invalidArgument(
                    "Memory estimate overflow while calculating \(component)"
                )
            }
            result = sum
        }
        return result
    }

    private static func checkedCeilDivide(
        _ value: Int64,
        by divisor: Int64,
        component: String
    ) throws -> Int64 {
        guard value >= 0, divisor > 0 else {
            throw MsplatError.invalidArgument(
                "Invalid values while calculating \(component)"
            )
        }
        let quotient = value / divisor
        return quotient + (value % divisor == 0 ? 0 : 1)
    }
}
