import MsplatCore

/// Statistics from a single training step.
public struct TrainingStats: Sendable {
    public let iteration: Int
    public let splatCount: Int
    /// CPU encoding and command-submission time, excluding synchronous GPU waits.
    public let cpuSubmitMs: Float

    @available(*, deprecated, renamed: "cpuSubmitMs", message: "This value measures CPU submission, not completed GPU step time.")
    public var msPerStep: Float { cpuSubmitMs }

    init(from c: MsplatStats) {
        self.iteration = Int(c.iteration)
        self.splatCount = Int(c.splatCount)
        self.cpuSubmitMs = c.msPerStep
    }
}

/// Defensive rasterizer invariant flags. The exact intersection pipeline is
/// expected to complete with this set empty.
public struct RasterizerOverflowKinds: OptionSet, Sendable, Equatable {
    public let rawValue: UInt32

    public init(rawValue: UInt32) {
        self.rawValue = rawValue
    }

    /// Legacy ABI bit from the removed fixed per-tile sorting path.
    public static let tileCapacity = RasterizerOverflowKinds(
        rawValue: UInt32(MSPLAT_RASTER_OVERFLOW_TILE_CAP)
    )
    /// Exact offsets and the allocated arena disagreed unexpectedly.
    public static let packedCapacity = RasterizerOverflowKinds(
        rawValue: UInt32(MSPLAT_RASTER_OVERFLOW_PACKED_CAPACITY)
    )
}

/// The most recently submitted training step. Submission can be ahead of GPU
/// completion, so this descriptor intentionally carries no GPU duration.
public struct SubmittedTrainingStep: Sendable, Equatable {
    public let iteration: Int
    public let splatCount: Int
    public let modelCapacity: Int
    public let effectiveWidth: Int
    public let effectiveHeight: Int
    public let activeSHDegree: Int
    public let cpuSubmitMs: Float

    init(from c: MsplatSubmittedTrainingStep) {
        iteration = Int(c.iteration)
        splatCount = Int(c.splatCount)
        modelCapacity = Int(c.modelCapacity)
        effectiveWidth = Int(c.effectiveWidth)
        effectiveHeight = Int(c.effectiveHeight)
        activeSHDegree = Int(c.activeSHDegree)
        cpuSubmitMs = c.cpuSubmitMs
    }
}

/// Metrics from the latest logical step whose complete Metal command-buffer
/// chain finished successfully.
public struct CompletedTrainingStep: Sendable, Equatable {
    public let iteration: Int
    public let splatCount: Int
    public let modelCapacity: Int
    public let effectiveWidth: Int
    public let effectiveHeight: Int
    public let activeSHDegree: Int
    public let cpuSubmitMs: Float
    /// Sum of positive Metal GPU intervals across this step's command buffers.
    public let gpuExecutionMs: Float?
    /// Wall-clock time from step start until GPU completion is observed. This
    /// includes queueing and completion-handler scheduling.
    public let endToEndMs: Float
    /// Mean training loss for this completed step.
    public let loss: Float?
    public let overflowKinds: RasterizerOverflowKinds
    /// Exact live intersection count for the completed frame.
    public let retainedPackedIntersectionCount: UInt64?
    /// Clearer alias for `retainedPackedIntersectionCount`.
    public var exactIntersectionCount: UInt64? {
        retainedPackedIntersectionCount
    }
    public let packedIntersectionCapacity: UInt64?
    /// CPU-side image loading, decode, resize, and upload before GPU encoding.
    public let imagePrepareMs: Float
    /// GPU duration of the exact-count command buffer when Metal timestamps
    /// are available.
    public let countGpuMs: Float?
    /// Wall time spent waiting for the exact-count command buffer. This also
    /// exposes queue drain latency ahead of the count pass.
    public let countWaitWallMs: Float
    /// Uncovered time between this logical step's Metal command-buffer GPU
    /// intervals. This excludes backlog before the first interval and latency
    /// after the final interval.
    public let queueIdleMs: Float?
    /// CPU time used to encode the stages after exact-count readback.
    public let postCountEncodeMs: Float
    /// CPU time spent growing the exact-intersection arena, or zero when reused.
    public let intersectionArenaGrowMs: Float
    public let maximumTileCount: Int
    public let activeTileCount: Int
    /// Tiles containing zero or one intersection.
    public let trivialTileCount: Int
    /// Tiles containing 2...32 intersections.
    public let smallTileCount: Int
    /// Tiles containing 33...2,048 intersections.
    public let mediumTileCount: Int
    /// Tiles containing more than 2,048 intersections.
    public let largeTileCount: Int

    init(from c: MsplatCompletedTrainingStepV12, flags: UInt32) {
        iteration = Int(c.iteration)
        splatCount = Int(c.splatCount)
        modelCapacity = Int(c.modelCapacity)
        effectiveWidth = Int(c.effectiveWidth)
        effectiveHeight = Int(c.effectiveHeight)
        activeSHDegree = Int(c.activeSHDegree)
        cpuSubmitMs = c.cpuSubmitMs
        gpuExecutionMs = flags & UInt32(MSPLAT_TRAINING_METRICS_GPU_TIME_VALID) != 0
            ? c.gpuExecutionMs : nil
        endToEndMs = c.endToEndMs
        loss = flags & UInt32(MSPLAT_TRAINING_METRICS_LOSS_VALID) != 0
            ? c.loss : nil
        overflowKinds = RasterizerOverflowKinds(rawValue: c.overflowKinds)
        let hasIntersections =
            flags & UInt32(MSPLAT_TRAINING_METRICS_INTERSECTIONS_VALID) != 0
        retainedPackedIntersectionCount = hasIntersections
            ? c.retainedPackedIntersectionCount : nil
        packedIntersectionCapacity = hasIntersections
            ? c.packedIntersectionCapacity : nil
        imagePrepareMs = c.imagePrepareMs
        countGpuMs =
            flags & UInt32(MSPLAT_TRAINING_METRICS_COUNT_GPU_TIME_VALID) != 0
            ? c.countGpuMs : nil
        countWaitWallMs = c.countWaitWallMs
        queueIdleMs =
            flags & UInt32(MSPLAT_TRAINING_METRICS_QUEUE_IDLE_TIME_VALID) != 0
            ? c.queueIdleMs : nil
        postCountEncodeMs = c.postCountEncodeMs
        intersectionArenaGrowMs = c.intersectionArenaGrowMs
        maximumTileCount = Int(c.maximumTileCount)
        activeTileCount = Int(c.activeTileCount)
        trivialTileCount = Int(c.trivialTileCount)
        smallTileCount = Int(c.smallTileCount)
        mediumTileCount = Int(c.mediumTileCount)
        largeTileCount = Int(c.largeTileCount)
    }
}

/// Query-only training progress. `completed` can legitimately lag `submitted`.
public struct TrainingTelemetry: Sendable, Equatable {
    public let submitted: SubmittedTrainingStep?
    public let completed: CompletedTrainingStep?
    public let overflowedCompletedSteps: UInt64
    public let tileCapacityOverflowedSteps: UInt64
    public let packedCapacityOverflowedSteps: UInt64
    public let lastOverflowIteration: Int?
    public let lastFailedIteration: Int?

    init(from c: MsplatTrainingMetricsV12) {
        let flags = c.flags
        submitted = flags & UInt32(MSPLAT_TRAINING_METRICS_HAS_SUBMITTED_STEP) != 0
            ? SubmittedTrainingStep(from: c.submitted) : nil
        completed = flags & UInt32(MSPLAT_TRAINING_METRICS_HAS_COMPLETED_STEP) != 0
            ? CompletedTrainingStep(from: c.completed, flags: flags) : nil
        overflowedCompletedSteps = c.overflowedCompletedSteps
        tileCapacityOverflowedSteps = c.tileCapOverflowedSteps
        packedCapacityOverflowedSteps = c.packedCapacityOverflowedSteps
        lastOverflowIteration = c.lastOverflowIteration > 0
            ? Int(c.lastOverflowIteration) : nil
        lastFailedIteration = flags & UInt32(MSPLAT_TRAINING_METRICS_HAS_FAILED_STEP) != 0
            ? Int(c.lastFailedIteration) : nil
    }
}

/// Live native buffer accounting and process-memory measurements.
public struct TrainingMemorySnapshot: Sendable, Equatable {
    public let trainerModelBufferBytes: UInt64
    public let engineSharedTransientBufferBytes: UInt64
    public let engineTrainingTransientBufferBytes: UInt64
    public let trainerTelemetryReadbackBytes: UInt64
    public let trainerImageCacheCpuBytes: UInt64
    public let trainerImageCacheGpuBytes: UInt64
    public let trainerImageCacheBudgetBytes: UInt64
    public let processPhysicalFootprintBytes: UInt64?
    public let processAvailableBytes: UInt64?
    public let trainingGpuImageCacheHits: UInt64
    public let trainingGpuImageCacheMisses: UInt64

    /// Sum of the native buffers whose ownership is represented above.
    public var trackedNativeBufferBytes: UInt64 {
        trainerModelBufferBytes
            + engineSharedTransientBufferBytes
            + engineTrainingTransientBufferBytes
            + trainerTelemetryReadbackBytes
            + trainerImageCacheGpuBytes
    }

    public var trainingGpuImageCacheHitRate: Double? {
        let accesses = trainingGpuImageCacheHits + trainingGpuImageCacheMisses
        guard accesses > 0 else { return nil }
        return Double(trainingGpuImageCacheHits) / Double(accesses)
    }

    init(from c: MsplatTrainingMemoryMetrics) {
        trainerModelBufferBytes = c.trainerModelBufferBytes
        engineSharedTransientBufferBytes = c.engineSharedTransientBufferBytes
        engineTrainingTransientBufferBytes = c.engineTrainingTransientBufferBytes
        trainerTelemetryReadbackBytes = c.trainerTelemetryReadbackBytes
        trainerImageCacheCpuBytes = c.trainerImageCacheCpuBytes
        trainerImageCacheGpuBytes = c.trainerImageCacheGpuBytes
        trainerImageCacheBudgetBytes = c.trainerImageCacheBudgetBytes
        processPhysicalFootprintBytes =
            c.flags & UInt32(MSPLAT_MEMORY_METRICS_PHYS_FOOTPRINT_VALID) != 0
            ? c.processPhysFootprintBytes : nil
        processAvailableBytes =
            c.flags & UInt32(MSPLAT_MEMORY_METRICS_AVAILABLE_VALID) != 0
            ? c.processAvailableBytes : nil
        trainingGpuImageCacheHits = c.trainingGpuImageCacheHits
        trainingGpuImageCacheMisses = c.trainingGpuImageCacheMisses
    }
}

/// Evaluation metrics from held-out test views.
public struct EvalMetrics: Sendable {
    public let psnr: Float
    public let ssim: Float
    public let l1: Float
    public let numTest: Int
    public let numGaussians: Int

    init(from c: MsplatEvalMetrics) {
        self.psnr = c.psnr
        self.ssim = c.ssim
        self.l1 = c.l1
        self.numTest = Int(c.numTest)
        self.numGaussians = Int(c.numGaussians)
    }
}
