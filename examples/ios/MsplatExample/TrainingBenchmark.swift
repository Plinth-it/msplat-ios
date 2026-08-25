import Darwin
import Foundation
import Msplat

struct TrainingBenchmarkConfiguration: Sendable {
    static let enabledKey = "MSPLAT_BENCHMARK"
    static let labelKey = "MSPLAT_BENCHMARK_LABEL"
    static let warmupKey = "MSPLAT_BENCHMARK_WARMUP"
    static let measuredKey = "MSPLAT_BENCHMARK_MEASURED"

    let label: String
    let warmupIterations: Int
    let measuredIterations: Int

    var totalIterations: Int {
        warmupIterations + measuredIterations
    }

    static func requested(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Self? {
        guard environment[enabledKey] == "1" else { return nil }

        let requestedLabel = environment[labelKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let label = requestedLabel.flatMap { $0.isEmpty ? nil : $0 } ?? "baseline"
        let warmup = validIterationCount(
            environment[warmupKey],
            defaultValue: 50,
            allowsZero: true
        )
        let measured = validIterationCount(
            environment[measuredKey],
            defaultValue: 300,
            allowsZero: false
        )
        let (total, overflowed) = warmup.addingReportingOverflow(measured)

        guard !overflowed, total >= 2, total <= 1_000_000 else {
            return Self(label: label, warmupIterations: 50, measuredIterations: 300)
        }
        return Self(
            label: label,
            warmupIterations: warmup,
            measuredIterations: measured
        )
    }

    private static func validIterationCount(
        _ rawValue: String?,
        defaultValue: Int,
        allowsZero: Bool
    ) -> Int {
        guard let rawValue,
              let value = Int(rawValue),
              value <= 1_000_000,
              allowsZero ? value >= 0 : value > 0 else {
            return defaultValue
        }
        return value
    }
}

struct TrainingBenchmarkDevice: Codable, Sendable {
    let machine: String
    let operatingSystem: String

    static func current() -> Self {
        Self(
            machine: machineIdentifier(),
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString
        )
    }

    private static func machineIdentifier() -> String {
        var byteCount = 0
        guard sysctlbyname("hw.machine", nil, &byteCount, nil, 0) == 0,
              byteCount > 1 else {
            return "unknown"
        }

        var bytes = [CChar](repeating: 0, count: byteCount)
        let status = bytes.withUnsafeMutableBufferPointer { buffer in
            sysctlbyname("hw.machine", buffer.baseAddress, &byteCount, nil, 0)
        }
        guard status == 0 else { return "unknown" }

        let utf8 = bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        let value = String(decoding: utf8, as: UTF8.self)
        return value.isEmpty ? "unknown" : value
    }
}

struct TrainingBenchmarkMemorySample: Codable, Sendable {
    let trainerModelBufferBytes: UInt64
    let engineSharedTransientBufferBytes: UInt64
    let engineTrainingTransientBufferBytes: UInt64
    let trainerTelemetryReadbackBytes: UInt64
    let trainerImageCacheCPUBytes: UInt64
    let trainerImageCacheGPUBytes: UInt64
    let trainerImageCacheBudgetBytes: UInt64
    let trackedNativeBufferBytes: UInt64
    let processPhysicalFootprintBytes: UInt64?
    let processAvailableBytes: UInt64?
    let trainingGPUImageCacheHits: UInt64
    let trainingGPUImageCacheMisses: UInt64

    init(_ memory: TrainingMemorySnapshot) {
        trainerModelBufferBytes = memory.trainerModelBufferBytes
        engineSharedTransientBufferBytes = memory.engineSharedTransientBufferBytes
        engineTrainingTransientBufferBytes = memory.engineTrainingTransientBufferBytes
        trainerTelemetryReadbackBytes = memory.trainerTelemetryReadbackBytes
        trainerImageCacheCPUBytes = memory.trainerImageCacheCpuBytes
        trainerImageCacheGPUBytes = memory.trainerImageCacheGpuBytes
        trainerImageCacheBudgetBytes = memory.trainerImageCacheBudgetBytes
        trackedNativeBufferBytes = memory.trackedNativeBufferBytes
        processPhysicalFootprintBytes = memory.processPhysicalFootprintBytes
        processAvailableBytes = memory.processAvailableBytes
        trainingGPUImageCacheHits = memory.trainingGpuImageCacheHits
        trainingGPUImageCacheMisses = memory.trainingGpuImageCacheMisses
    }
}

struct TrainingBenchmarkSample: Codable, Sendable {
    let iteration: Int
    let population: Int
    let modelCapacity: Int
    let effectiveWidth: Int
    let effectiveHeight: Int
    let activeSHDegree: Int
    let cpuSubmitMs: Float
    let imagePrepareMs: Float
    let gpuExecutionMs: Float?
    let endToEndMs: Float
    let countGpuMs: Float?
    let countWaitWallMs: Float
    let postCountEncodeMs: Float
    let intersectionArenaGrowMs: Float
    let loss: Float?
    let exactIntersectionCount: UInt64?
    let packedIntersectionCapacity: UInt64?
    let maximumTileCount: Int
    let activeTileCount: Int
    let trivialTileCount: Int
    let smallTileCount: Int
    let mediumTileCount: Int
    let largeTileCount: Int
    let overflowKinds: UInt32
    let overflowedCompletedSteps: UInt64
    let tileCapacityOverflowedSteps: UInt64
    let packedCapacityOverflowedSteps: UInt64
    let lastOverflowIteration: Int?
    let lastFailedIteration: Int?
    let memory: TrainingBenchmarkMemorySample
    let thermalState: String

    init(
        completed: CompletedTrainingStep,
        telemetry: TrainingTelemetry,
        memory: TrainingMemorySnapshot,
        thermalState: String
    ) {
        iteration = completed.iteration
        population = completed.splatCount
        modelCapacity = completed.modelCapacity
        effectiveWidth = completed.effectiveWidth
        effectiveHeight = completed.effectiveHeight
        activeSHDegree = completed.activeSHDegree
        cpuSubmitMs = completed.cpuSubmitMs
        imagePrepareMs = completed.imagePrepareMs
        gpuExecutionMs = completed.gpuExecutionMs
        endToEndMs = completed.endToEndMs
        countGpuMs = completed.countGpuMs
        countWaitWallMs = completed.countWaitWallMs
        postCountEncodeMs = completed.postCountEncodeMs
        intersectionArenaGrowMs = completed.intersectionArenaGrowMs
        loss = completed.loss
        exactIntersectionCount = completed.exactIntersectionCount
        packedIntersectionCapacity = completed.packedIntersectionCapacity
        maximumTileCount = completed.maximumTileCount
        activeTileCount = completed.activeTileCount
        trivialTileCount = completed.trivialTileCount
        smallTileCount = completed.smallTileCount
        mediumTileCount = completed.mediumTileCount
        largeTileCount = completed.largeTileCount
        overflowKinds = completed.overflowKinds.rawValue
        overflowedCompletedSteps = telemetry.overflowedCompletedSteps
        tileCapacityOverflowedSteps = telemetry.tileCapacityOverflowedSteps
        packedCapacityOverflowedSteps = telemetry.packedCapacityOverflowedSteps
        lastOverflowIteration = telemetry.lastOverflowIteration
        lastFailedIteration = telemetry.lastFailedIteration
        self.memory = TrainingBenchmarkMemorySample(memory)
        self.thermalState = thermalState
    }
}

struct TrainingBenchmarkDistribution: Codable, Sendable {
    let count: Int
    let median: Double
    let p90: Double

    static func make(_ values: [Double]) -> Self? {
        let sorted = values.filter(\.isFinite).sorted()
        guard !sorted.isEmpty else { return nil }

        let middle = sorted.count / 2
        let median = sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle]
        let nearestRank = max(1, Int(ceil(Double(sorted.count) * 0.9)))
        return Self(
            count: sorted.count,
            median: median,
            p90: sorted[nearestRank - 1]
        )
    }
}

struct TrainingBenchmarkDistributions: Codable, Sendable {
    let cpuSubmitMs: TrainingBenchmarkDistribution?
    let imagePrepareMs: TrainingBenchmarkDistribution?
    let gpuExecutionMs: TrainingBenchmarkDistribution?
    let endToEndMs: TrainingBenchmarkDistribution?
    let countGpuMs: TrainingBenchmarkDistribution?
    let countWaitWallMs: TrainingBenchmarkDistribution?
    let postCountEncodeMs: TrainingBenchmarkDistribution?
    let intersectionArenaGrowMs: TrainingBenchmarkDistribution?
    let loss: TrainingBenchmarkDistribution?
    let population: TrainingBenchmarkDistribution?
    let exactIntersectionCount: TrainingBenchmarkDistribution?
    let maximumTileCount: TrainingBenchmarkDistribution?
    let activeTileCount: TrainingBenchmarkDistribution?

    init(samples: [TrainingBenchmarkSample]) {
        cpuSubmitMs = Self.distribution(samples.map { Double($0.cpuSubmitMs) })
        imagePrepareMs = Self.distribution(samples.map { Double($0.imagePrepareMs) })
        gpuExecutionMs = Self.distribution(samples.compactMap { $0.gpuExecutionMs.map(Double.init) })
        endToEndMs = Self.distribution(samples.map { Double($0.endToEndMs) })
        countGpuMs = Self.distribution(samples.compactMap { $0.countGpuMs.map(Double.init) })
        countWaitWallMs = Self.distribution(samples.map { Double($0.countWaitWallMs) })
        postCountEncodeMs = Self.distribution(samples.map { Double($0.postCountEncodeMs) })
        intersectionArenaGrowMs = Self.distribution(
            samples.map { Double($0.intersectionArenaGrowMs) }
        )
        loss = Self.distribution(samples.compactMap { $0.loss.map(Double.init) })
        population = Self.distribution(samples.map { Double($0.population) })
        exactIntersectionCount = Self.distribution(
            samples.compactMap { sample in
                sample.exactIntersectionCount.map { Double($0) }
            }
        )
        maximumTileCount = Self.distribution(samples.map { Double($0.maximumTileCount) })
        activeTileCount = Self.distribution(samples.map { Double($0.activeTileCount) })
    }

    private static func distribution(_ values: [Double]) -> TrainingBenchmarkDistribution? {
        TrainingBenchmarkDistribution.make(values)
    }
}

struct TrainingBenchmarkSummary: Codable, Sendable {
    let label: String
    let resultFile: String
    let warmupIterations: Int
    let requestedMeasuredIterations: Int
    let capturedMeasuredIterations: Int
    let firstMeasuredIteration: Int?
    let lastMeasuredIteration: Int?
    let finalCompletedIteration: Int
    let elapsedSeconds: Double
    let measuredElapsedSeconds: Double
    let measuredIterationsPerSecond: Double
    let device: TrainingBenchmarkDevice
    let nativeModeEnvironment: [String: String]
    let distributions: TrainingBenchmarkDistributions
    let peakProcessPhysicalFootprintBytes: UInt64?
    let minimumProcessAvailableBytes: UInt64?
    let peakTrackedNativeBufferBytes: UInt64?
    let highestThermalState: String
    let overflowedCompletedSteps: UInt64
    let tileCapacityOverflowedSteps: UInt64
    let packedCapacityOverflowedSteps: UInt64
    let missingMeasuredIterations: [Int]
}

struct TrainingBenchmarkReport: Codable, Sendable {
    let schemaVersion: Int
    let startedAt: Date
    let finishedAt: Date
    let datasetName: String
    let datasetKind: String
    let trainingMaskCandidateCount: Int?
    let trainingMasksEnabled: Bool
    let trainingMaskMode: String
    let profile: String
    let fixedPopulation: Bool
    let summary: TrainingBenchmarkSummary
    let finalDescriptor: TrainingBenchmarkSample
    let samples: [TrainingBenchmarkSample]
}

enum TrainingBenchmarkError: LocalizedError, Sendable {
    case missingFinalDescriptor
    case incompleteMeasurements(captured: Int, requested: Int, finalIteration: Int)
    case couldNotEncodeSummary

    var errorDescription: String? {
        switch self {
        case .missingFinalDescriptor:
            "Benchmark completed without final training telemetry."
        case let .incompleteMeasurements(captured, requested, finalIteration):
            "Benchmark captured \(captured) of \(requested) measured steps " +
                "and completed iteration \(finalIteration)."
        case .couldNotEncodeSummary:
            "The benchmark summary could not be encoded as UTF-8."
        }
    }
}

struct TrainingBenchmarkRecorder {
    let configuration: TrainingBenchmarkConfiguration
    let datasetName: String
    let datasetKind: String
    let trainingMaskCandidateCount: Int?
    let trainingMasksEnabled: Bool
    let trainingMaskMode: String
    let startedAt: Date
    let device: TrainingBenchmarkDevice
    let nativeModeEnvironment: [String: String]

    private(set) var samples: [TrainingBenchmarkSample] = []
    private(set) var finalDescriptor: TrainingBenchmarkSample?
    private var capturedIterations: Set<Int> = []

    init(
        configuration: TrainingBenchmarkConfiguration,
        folder: DatasetFolder,
        trainingMaskCandidateCount: Int?,
        trainingMasksEnabled: Bool,
        trainingMaskMode: TrainingMaskMode
    ) {
        self.configuration = configuration
        datasetName = folder.name
        datasetKind = folder.kind.rawValue
        self.trainingMaskCandidateCount = trainingMaskCandidateCount
        self.trainingMasksEnabled = trainingMasksEnabled
        self.trainingMaskMode = trainingMaskMode == .transparent
            ? "transparent" : "coverage"
        startedAt = Date()
        device = .current()
        nativeModeEnvironment = Self.nativeModes(
            environment: ProcessInfo.processInfo.environment
        )
        samples.reserveCapacity(configuration.totalIterations)
    }

    mutating func record(
        telemetry: TrainingTelemetry,
        memory: TrainingMemorySnapshot,
        thermalState: String,
        isFinalDescriptor: Bool = false
    ) {
        guard let completed = telemetry.completed else { return }
        let sample = TrainingBenchmarkSample(
            completed: completed,
            telemetry: telemetry,
            memory: memory,
            thermalState: thermalState
        )
        if isFinalDescriptor {
            finalDescriptor = sample
        }
        guard capturedIterations.insert(completed.iteration).inserted else { return }
        samples.append(sample)
    }

    mutating func finish(
        measuredElapsedSeconds: Double
    ) throws -> (url: URL, summaryLine: String) {
        guard let finalDescriptor else {
            throw TrainingBenchmarkError.missingFinalDescriptor
        }

        let finishedAt = Date()
        let filename = Self.resultFilename(
            label: configuration.label,
            timestamp: finishedAt
        )
        let url = URL.documentsDirectory.appending(path: filename)
        let measuredSamples = samples.filter {
            $0.iteration > configuration.warmupIterations &&
            $0.iteration <= configuration.totalIterations
        }
        let measuredIterations = Set(measuredSamples.map(\.iteration))
        let expectedRange = (configuration.warmupIterations + 1)...configuration.totalIterations
        let missingIterations = expectedRange.filter { !measuredIterations.contains($0) }
        guard missingIterations.isEmpty,
              measuredSamples.count == configuration.measuredIterations,
              finalDescriptor.iteration == configuration.totalIterations else {
            throw TrainingBenchmarkError.incompleteMeasurements(
                captured: measuredSamples.count,
                requested: configuration.measuredIterations,
                finalIteration: finalDescriptor.iteration
            )
        }
        let peakFootprint = samples.compactMap { $0.memory.processPhysicalFootprintBytes }.max()
        let minimumAvailable = samples.compactMap { $0.memory.processAvailableBytes }.min()
        let peakTracked = samples.map { $0.memory.trackedNativeBufferBytes }.max()
        let summary = TrainingBenchmarkSummary(
            label: configuration.label,
            resultFile: filename,
            warmupIterations: configuration.warmupIterations,
            requestedMeasuredIterations: configuration.measuredIterations,
            capturedMeasuredIterations: measuredSamples.count,
            firstMeasuredIteration: measuredSamples.first?.iteration,
            lastMeasuredIteration: measuredSamples.last?.iteration,
            finalCompletedIteration: finalDescriptor.iteration,
            elapsedSeconds: finishedAt.timeIntervalSince(startedAt),
            measuredElapsedSeconds: measuredElapsedSeconds,
            measuredIterationsPerSecond: measuredElapsedSeconds > 0
                ? Double(configuration.measuredIterations) / measuredElapsedSeconds
                : 0,
            device: device,
            nativeModeEnvironment: nativeModeEnvironment,
            distributions: TrainingBenchmarkDistributions(samples: measuredSamples),
            peakProcessPhysicalFootprintBytes: peakFootprint,
            minimumProcessAvailableBytes: minimumAvailable,
            peakTrackedNativeBufferBytes: peakTracked,
            highestThermalState: Self.highestThermalState(in: samples),
            overflowedCompletedSteps: finalDescriptor.overflowedCompletedSteps,
            tileCapacityOverflowedSteps: finalDescriptor.tileCapacityOverflowedSteps,
            packedCapacityOverflowedSteps: finalDescriptor.packedCapacityOverflowedSteps,
            missingMeasuredIterations: Array(missingIterations)
        )
        let report = TrainingBenchmarkReport(
            schemaVersion: 2,
            startedAt: startedAt,
            finishedAt: finishedAt,
            datasetName: datasetName,
            datasetKind: datasetKind,
            trainingMaskCandidateCount: trainingMaskCandidateCount,
            trainingMasksEnabled: trainingMasksEnabled,
            trainingMaskMode: trainingMaskMode,
            profile: "Preview",
            fixedPopulation: true,
            summary: summary,
            finalDescriptor: finalDescriptor,
            samples: samples
        )

        let reportData = try Self.encoder(prettyPrinted: true).encode(report)
        try reportData.write(to: url, options: .atomic)
        let summaryData = try Self.encoder(prettyPrinted: false).encode(summary)
        guard let summaryLine = String(data: summaryData, encoding: .utf8) else {
            throw TrainingBenchmarkError.couldNotEncodeSummary
        }
        return (url, summaryLine)
    }

    private static func nativeModes(environment: [String: String]) -> [String: String] {
        [
            "MSPLAT_RASTER_VARIANT": environment["MSPLAT_RASTER_VARIANT"] ?? "8x8",
            "MSPLAT_INTERSECTION_ATTRIBUTES":
                environment["MSPLAT_INTERSECTION_ATTRIBUTES"] ?? "packed",
            "MSPLAT_TILE_COUNT_MODE": environment["MSPLAT_TILE_COUNT_MODE"] ?? "enumerated",
            "MSPLAT_TILE_LAYOUT_MODE": environment["MSPLAT_TILE_LAYOUT_MODE"] ?? "cpu",
            "MSPLAT_TRAINING_ARENA_MODE":
                environment["MSPLAT_TRAINING_ARENA_MODE"] ?? "exact",
            "MSPLAT_SSIM_MODE": environment["MSPLAT_SSIM_MODE"] ?? "staged",
            "MSPLAT_DENSIFY_RANDOM_MODE":
                environment["MSPLAT_DENSIFY_RANDOM_MODE"] ?? "cpu",
            "MSPLAT_IMAGE_CACHE_MB": environment["MSPLAT_IMAGE_CACHE_MB"] ?? "512",
            "MSPLAT_CAMERA_PREFETCH": "1 (enabled by MsplatExample)",
            "PROFILE_STAGES": environment["PROFILE_STAGES"] == nil
                ? "disabled" : "enabled",
        ]
    }

    private static func resultFilename(label: String, timestamp: Date) -> String {
        let safeLabel = label.unicodeScalars.map { scalar -> Character in
            CharacterSet.alphanumerics.contains(scalar) || scalar == "-" || scalar == "_"
                ? Character(String(scalar)) : "-"
        }
        let compactLabel = String(safeLabel).prefix(64)
        let nonemptyLabel = compactLabel.isEmpty ? "benchmark" : String(compactLabel)
        let milliseconds = Int64(timestamp.timeIntervalSince1970 * 1_000)
        return "msplat-benchmark-\(nonemptyLabel)-\(milliseconds).json"
    }

    private static func encoder(prettyPrinted: Bool) -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "Infinity",
            negativeInfinity: "-Infinity",
            nan: "NaN"
        )
        encoder.outputFormatting = prettyPrinted
            ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            : [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }

    private static func highestThermalState(in samples: [TrainingBenchmarkSample]) -> String {
        let severity = [
            "Unknown": -1,
            "Nominal": 0,
            "Fair": 1,
            "Serious": 2,
            "Critical": 3,
        ]
        return samples.max {
            severity[$0.thermalState, default: -1] < severity[$1.thermalState, default: -1]
        }?.thermalState ?? "Unknown"
    }
}
