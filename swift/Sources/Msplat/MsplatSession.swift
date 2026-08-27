import Foundation
import MsplatCore
import Synchronization

/// Serializes access to msplat's process-global Metal state.
@globalActor
public actor MsplatRuntimeActor {
    public static let shared = MsplatRuntimeActor()
}

/// Dataset loading and evaluation-split options.
public struct DatasetOptions: Sendable, Equatable {
    public var downscaleFactor: Float
    public var evalMode: Bool
    public var testEvery: Int32
    /// Whether path-based dataset loaders discover optional mask sidecars.
    public var discoverTrainingMasks: Bool
    /// Whether the dataset prepares the next training target on a depth-one
    /// worker while the current Metal step runs. `false` leaves the native
    /// default unchanged, including its environment-variable opt-in.
    public var prefetchTrainingTargets: Bool

    public init(
        downscaleFactor: Float = 1.0,
        evalMode: Bool = false,
        testEvery: Int32 = 8,
        discoverTrainingMasks: Bool = false,
        prefetchTrainingTargets: Bool = false
    ) {
        self.downscaleFactor = downscaleFactor
        self.evalMode = evalMode
        self.testEvery = testEvery
        self.discoverTrainingMasks = discoverTrainingMasks
        self.prefetchTrainingTargets = prefetchTrainingTargets
    }
}

/// A validated 4-by-4, row-major camera-to-world transform.
public struct CameraPose: Sendable, Equatable {
    public static let elementCount = 16
    public let elements: [Float]

    public init(elements: [Float]) throws {
        guard elements.count == Self.elementCount else {
            throw MsplatError.invalidArgument(
                "A camera pose must contain exactly \(Self.elementCount) values"
            )
        }
        guard elements.allSatisfy(\.isFinite) else {
            throw MsplatError.invalidArgument("A camera pose must contain only finite values")
        }
        self.elements = elements
    }
}

/// An immutable, tightly packed RGBA8 render.
public struct RGBAFrame: Sendable, Equatable {
    public let data: Data
    public let width: Int
    public let height: Int
    public let bytesPerRow: Int

    fileprivate init(data: Data, width: Int, height: Int, bytesPerRow: Int) {
        self.data = data
        self.width = width
        self.height = height
        self.bytesPerRow = bytesPerRow
    }
}

/// An ownership-safe facade over one native dataset and trainer.
///
/// Only one session may be open because the native Metal resources are still
/// process-global. Legacy `GaussianDataset` and `GaussianTrainer` remain
/// source-compatible, but must not be used concurrently with this facade.
@MsplatRuntimeActor
public final class MsplatSession {
    private let resources: SessionResources

    /// Creates a session from one explicit resolution, SH, and topology plan.
    public convenience init(
        datasetURL: URL,
        trainingPlan plan: TrainingPlan,
        baseConfig: TrainingConfig = TrainingConfig(),
        evalMode: Bool = false,
        testEvery: Int32 = 8,
        prefetchTrainingTargets: Bool = false
    ) throws {
        try self.init(
            datasetURL: datasetURL,
            options: plan.makeDatasetOptions(
                evalMode: evalMode,
                testEvery: testEvery,
                prefetchTrainingTargets: prefetchTrainingTargets
            ),
            config: plan.makeTrainingConfig(startingFrom: baseConfig),
            maximumGaussianCount: plan.maximumGaussianCount
        )
    }

    /// Creates the native dataset and trainer and retains the dataset URL's
    /// security scope until ``close()`` completes.
    public init(
        datasetURL: URL,
        options: DatasetOptions = DatasetOptions(),
        config: TrainingConfig = TrainingConfig(),
        maximumGaussianCount: Int? = nil
    ) throws {
        try Self.validateCreation(
            config: config,
            maximumGaussianCount: maximumGaussianCount
        )
        guard datasetURL.isFileURL, !datasetURL.path.isEmpty else {
            throw MsplatError.invalidArgument(
                "The dataset URL must be a file URL with a non-empty path"
            )
        }
        try reserveNativeSession()
        var securityScopes = SecurityScopedURLLeases(urls: [datasetURL])

        do {
            let handles = try Self.createHandles(
                path: datasetURL.path,
                options: options,
                config: config,
                maximumGaussianCount: maximumGaussianCount
            )
            resources = SessionResources(
                dataset: handles.dataset,
                trainer: handles.trainer,
                securityScopes: securityScopes
            )
        } catch {
            securityScopes.close()
            releaseNativeSession()
            throw error
        }
    }

    /// Creates a session from caller-owned canonical data. Every descriptor
    /// buffer is copied synchronously by the checked native ABI v5 or v6 path
    /// before this returns.
    ///
    /// Pass selected folder or resolved-bookmark roots in
    /// `securityScopedResourceURLs` when frame or mask URLs are derived
    /// children. The session balances successful leases in ``close()`` after
    /// the trainer and dataset have been destroyed. It does not attempt one
    /// lease per asset.
    public init(
        dataset descriptor: DatasetDescriptor,
        securityScopedResourceURLs: [URL] = [],
        options: DatasetOptions = DatasetOptions(),
        config: TrainingConfig = TrainingConfig(),
        maximumGaussianCount: Int? = nil
    ) throws {
        try Self.validateCreation(
            config: config,
            maximumGaussianCount: maximumGaussianCount
        )
        for url in securityScopedResourceURLs {
            guard url.isFileURL, !url.path.isEmpty else {
                throw MsplatError.invalidArgument(
                    "Security-scoped resources must be non-empty file URLs"
                )
            }
        }

        try reserveNativeSession()
        var securityScopes = SecurityScopedURLLeases(
            urls: securityScopedResourceURLs
        )
        do {
            let handles = try Self.createHandles(
                descriptor: descriptor,
                options: options,
                config: config,
                maximumGaussianCount: maximumGaussianCount
            )
            resources = SessionResources(
                dataset: handles.dataset,
                trainer: handles.trainer,
                securityScopes: securityScopes
            )
        } catch {
            securityScopes.close()
            releaseNativeSession()
            throw error
        }
    }

    @discardableResult
    public func step() throws -> TrainingStats {
        try withTrainer { trainer in
            var stats = MsplatStats()
            var nativeError = MsplatErrorInfo()
            let status = msplat_trainer_step_v2(trainer, &stats, &nativeError)
            try checkNativeStatus(status, error: &nativeError)
            return TrainingStats(from: stats)
        }
    }

    /// Polls submission and GPU-completion progress without submitting work.
    public func trainingMetrics() throws -> TrainingTelemetry {
        try withTrainer { trainer in
            var metrics = MsplatTrainingMetricsV12()
            var nativeError = MsplatErrorInfo()
            let status = msplat_trainer_metrics_v12(
                trainer,
                &metrics,
                MemoryLayout<MsplatTrainingMetricsV12>.size,
                &nativeError
            )
            try checkNativeStatus(status, error: &nativeError)
            return TrainingTelemetry(from: metrics)
        }
    }

    /// Returns live categorized native-buffer and process-memory measurements.
    public func memoryMetrics() throws -> TrainingMemorySnapshot {
        try withTrainer { trainer in
            var metrics = MsplatTrainingMemoryMetricsV17()
            var nativeError = MsplatErrorInfo()
            let status = msplat_trainer_memory_metrics_v17(
                trainer,
                &metrics,
                MemoryLayout<MsplatTrainingMemoryMetricsV17>.size,
                &nativeError
            )
            try checkNativeStatus(status, error: &nativeError)
            return TrainingMemorySnapshot(from: metrics)
        }
    }

    /// Returns one read-only pose-refinement snapshot per canonical dataset
    /// camera, or an empty array when camera-pose refinement is disabled.
    ///
    /// This synchronizes pending trainer GPU work before reading pose tensors.
    /// Frame IDs are copied before the native trainer lock is released.
    public func cameraPoseRefinementStates() throws -> [CameraPoseRefinementState] {
        try withTrainer { trainer in
            var count: UInt32 = 0
            var nativeError = MsplatErrorInfo()
            let countStatus = msplat_trainer_pose_refinement_count_v15(
                trainer,
                &count,
                &nativeError
            )
            try checkNativeStatus(countStatus, error: &nativeError)

            var states: [CameraPoseRefinementState] = []
            states.reserveCapacity(Int(count))
            for cameraIndex in 0..<count {
                var native = MsplatPoseRefinementStateV15()
                var stateError = MsplatErrorInfo()
                let stateStatus = msplat_trainer_pose_refinement_state_v15(
                    trainer,
                    cameraIndex,
                    &native,
                    MemoryLayout<MsplatPoseRefinementStateV15>.size,
                    &stateError
                )
                try checkNativeStatus(stateStatus, error: &stateError)
                states.append(try CameraPoseRefinementState(from: native))
            }
            return states
        }
    }

    public func evaluate() throws -> EvalMetrics {
        try withTrainer { trainer in
            var metrics = MsplatEvalMetrics()
            var nativeError = MsplatErrorInfo()
            let status = msplat_trainer_evaluate_v2(trainer, &metrics, &nativeError)
            try checkNativeStatus(status, error: &nativeError)
            return EvalMetrics(from: metrics)
        }
    }

    /// Renders a train or test camera as RGB float32 data.
    public func render(camera index: Int, useTest: Bool = false) throws -> PixelData {
        let index = try nativeIndex(index)
        return try withTrainer { trainer in
            var buffer = MsplatPixelBuffer()
            var nativeError = MsplatErrorInfo()
            let status = msplat_trainer_render_v2(
                trainer, index, useTest, &buffer, &nativeError
            )
            try checkNativeStatus(status, error: &nativeError)
            return try Self.takePixelData(&buffer)
        }
    }

    /// Renders from an arbitrary pose as RGB float32 data.
    public func render(
        pose: CameraPose,
        referenceCamera: Int = 0
    ) throws -> PixelData {
        let referenceCamera = try nativeIndex(referenceCamera)
        return try withTrainer { trainer in
            var buffer = MsplatPixelBuffer()
            var nativeError = MsplatErrorInfo()
            let status = pose.elements.withUnsafeBufferPointer { elements in
                msplat_trainer_render_pose_v2(
                    trainer,
                    elements.baseAddress,
                    referenceCamera,
                    &buffer,
                    &nativeError
                )
            }
            try checkNativeStatus(status, error: &nativeError)
            return try Self.takePixelData(&buffer)
        }
    }

    /// Renders from an arbitrary pose as tightly packed RGBA8 data.
    public func renderRGBA(
        pose: CameraPose,
        referenceCamera: Int = 0
    ) throws -> RGBAFrame {
        let referenceCamera = try nativeIndex(referenceCamera)
        return try withTrainer { trainer in
            var width: Int32 = 0
            var height: Int32 = 0
            var nativeError = MsplatErrorInfo()
            let queryStatus = pose.elements.withUnsafeBufferPointer { elements in
                msplat_trainer_render_pose_to_buffer_v2(
                    trainer,
                    elements.baseAddress,
                    referenceCamera,
                    nil,
                    0,
                    &width,
                    &height,
                    &nativeError
                )
            }
            try checkNativeStatus(queryStatus, error: &nativeError)

            let frameWidth = Int(width)
            let frameHeight = Int(height)
            let layout = try checkedImageLayout(
                width: frameWidth,
                height: frameHeight,
                components: 4
            )
            var data = Data(count: layout.elementCount)
            var renderedWidth: Int32 = 0
            var renderedHeight: Int32 = 0

            try data.withUnsafeMutableBytes { bytes in
                guard bytes.count >= layout.elementCount,
                      let destination = bytes.baseAddress else {
                    throw MsplatError.internalFailure("Could not allocate RGBA storage")
                }
                var renderError = MsplatErrorInfo()
                let renderStatus = pose.elements.withUnsafeBufferPointer { elements in
                    msplat_trainer_render_pose_to_buffer_v2(
                        trainer,
                        elements.baseAddress,
                        referenceCamera,
                        destination.assumingMemoryBound(to: UInt8.self),
                        bytes.count,
                        &renderedWidth,
                        &renderedHeight,
                        &renderError
                    )
                }
                try checkNativeStatus(renderStatus, error: &renderError)
            }

            guard renderedWidth == width, renderedHeight == height else {
                throw MsplatError.internalFailure(
                    "Render dimensions changed while filling the RGBA buffer"
                )
            }
            return RGBAFrame(
                data: data,
                width: frameWidth,
                height: frameHeight,
                bytesPerRow: layout.elementsPerRow
            )
        }
    }

    /// Submits a fixed-pose preview into a separately owned BGRA8 Metal
    /// texture. The returned submission can be awaited without holding this
    /// runtime actor; keep displaying the latest completed surface meanwhile.
    ///
    /// The current renderer still performs its exact-count synchronization
    /// during submission. Final raster completion and texture conversion do
    /// not perform a CPU readback or block this method.
    public func submitPreview(
        pose: CameraPose,
        referenceCamera: Int = 0
    ) throws -> MetalPreviewSubmission {
        let referenceCamera = try nativeIndex(referenceCamera)
        return try withTrainer { trainer in
            var frame: MsplatPreviewFrame?
            var nativeError = MsplatErrorInfo()
            let status = pose.elements.withUnsafeBufferPointer { elements in
                msplat_trainer_render_pose_preview_v13(
                    trainer,
                    elements.baseAddress,
                    referenceCamera,
                    &frame,
                    &nativeError
                )
            }
            try checkNativeStatus(status, error: &nativeError)
            guard let frame else {
                throw MsplatError.internalFailure(
                    "Native preview submission returned no frame handle"
                )
            }
            return MetalPreviewSubmission(handle: frame)
        }
    }

    public func cameraPose(at index: Int) throws -> CameraPose {
        let index = try nativeIndex(index)
        return try withDataset { dataset in
            var elements = [Float](repeating: 0, count: CameraPose.elementCount)
            var nativeError = MsplatErrorInfo()
            let status = elements.withUnsafeMutableBufferPointer { values in
                msplat_dataset_camera_pose_v2(
                    dataset, index, values.baseAddress, &nativeError
                )
            }
            try checkNativeStatus(status, error: &nativeError)
            return try CameraPose(elements: elements)
        }
    }

    public func exportPLY(to url: URL) throws {
        let path = try filePath(url)
        try withTrainer { trainer in
            var nativeError = MsplatErrorInfo()
            let status = msplat_trainer_export_ply_v2(trainer, path, &nativeError)
            try checkNativeStatus(status, error: &nativeError)
        }
    }

    public func saveCheckpoint(to url: URL) throws {
        let path = try filePath(url)
        try withTrainer { trainer in
            var nativeError = MsplatErrorInfo()
            let status = msplat_trainer_save_checkpoint_v2(trainer, path, &nativeError)
            try checkNativeStatus(status, error: &nativeError)
        }
    }

    @discardableResult
    public func loadCheckpoint(from url: URL) throws -> Int {
        let path = try filePath(url)
        return try withTrainer { trainer in
            var iteration: Int32 = 0
            var nativeError = MsplatErrorInfo()
            let status = msplat_trainer_load_checkpoint_v2(
                trainer, path, &iteration, &nativeError
            )
            try checkNativeStatus(status, error: &nativeError)
            return Int(iteration)
        }
    }

    public var numTrain: Int {
        get throws {
            try withDataset { dataset in
                var count: Int32 = 0
                var nativeError = MsplatErrorInfo()
                let status = msplat_dataset_num_train_v2(dataset, &count, &nativeError)
                try checkNativeStatus(status, error: &nativeError)
                return Int(count)
            }
        }
    }

    public var splatCount: Int {
        get throws {
            try withTrainer { trainer in
                var count: Int32 = 0
                var nativeError = MsplatErrorInfo()
                let status = msplat_trainer_splat_count_v2(trainer, &count, &nativeError)
                try checkNativeStatus(status, error: &nativeError)
                return Int(count)
            }
        }
    }

    public var iteration: Int {
        get throws {
            try withTrainer { trainer in
                var value: Int32 = 0
                var nativeError = MsplatErrorInfo()
                let status = msplat_trainer_iteration_v2(trainer, &value, &nativeError)
                try checkNativeStatus(status, error: &nativeError)
                return Int(value)
            }
        }
    }

    /// Idempotently destroys the trainer before the dataset.
    public func close() throws {
        try resources.close()
    }

    private static func createHandles(
        path: String,
        options: DatasetOptions,
        config: TrainingConfig,
        maximumGaussianCount: Int?
    ) throws -> (dataset: MsplatDataset, trainer: MsplatTrainer) {
        try createHandles(
            config: config,
            maximumGaussianCount: maximumGaussianCount,
            prefetchTrainingTargets: options.prefetchTrainingTargets
        ) {
            var dataset: MsplatDataset?
            var nativeError = MsplatErrorInfo()
            let status = msplat_dataset_create_v10(
                path,
                options.downscaleFactor,
                options.evalMode,
                options.testEvery,
                options.discoverTrainingMasks,
                &dataset,
                &nativeError
            )
            try checkNativeStatus(status, error: &nativeError)
            guard let dataset else {
                throw MsplatError.internalFailure(
                    "Native dataset creation returned no handle"
                )
            }
            return dataset
        }
    }

    private static func createHandles(
        descriptor: DatasetDescriptor,
        options: DatasetOptions,
        config: TrainingConfig,
        maximumGaussianCount: Int?
    ) throws -> (dataset: MsplatDataset, trainer: MsplatTrainer) {
        try createHandles(
            config: config,
            maximumGaussianCount: maximumGaussianCount,
            prefetchTrainingTargets: options.prefetchTrainingTargets
        ) {
            if descriptor.frames.contains(where: { $0.trainingMask != nil }) {
                return try withUnsafeNativeDatasetDescriptorV6(descriptor) {
                    nativeDescriptor, frameMasks, frameMaskCount in
                    var dataset: MsplatDataset?
                    var nativeError = MsplatErrorInfo()
                    let status = msplat_dataset_create_from_descriptor_v6(
                        nativeDescriptor,
                        MemoryLayout<MsplatDatasetDescriptorV5>.size,
                        frameMasks,
                        frameMaskCount,
                        MemoryLayout<MsplatFrameMaskV6>.stride,
                        options.downscaleFactor,
                        options.evalMode,
                        options.testEvery,
                        &dataset,
                        &nativeError
                    )
                    try checkNativeStatus(status, error: &nativeError)
                    guard let dataset else {
                        throw MsplatError.internalFailure(
                            "Native descriptor creation returned no dataset handle"
                        )
                    }
                    return dataset
                }
            }

            return try withUnsafeNativeDatasetDescriptor(descriptor) {
                nativeDescriptor in
                var dataset: MsplatDataset?
                var nativeError = MsplatErrorInfo()
                let status = msplat_dataset_create_from_descriptor_v5(
                    nativeDescriptor,
                    MemoryLayout<MsplatDatasetDescriptorV5>.size,
                    options.downscaleFactor,
                    options.evalMode,
                    options.testEvery,
                    &dataset,
                    &nativeError
                )
                try checkNativeStatus(status, error: &nativeError)
                guard let dataset else {
                    throw MsplatError.internalFailure(
                        "Native descriptor creation returned no dataset handle"
                    )
                }
                return dataset
            }
        }
    }

    private static func createHandles(
        config: TrainingConfig,
        maximumGaussianCount: Int?,
        prefetchTrainingTargets: Bool,
        createDataset: () throws -> MsplatDataset
    ) throws -> (dataset: MsplatDataset, trainer: MsplatTrainer) {
        try withConfiguredNativeEngine(metallibResourceName) {
            let dataset = try createDataset()

            do {
                if prefetchTrainingTargets {
                    var prefetchError = MsplatErrorInfo()
                    let prefetchStatus =
                        msplat_dataset_enable_training_target_prefetch_v14(
                            dataset,
                            &prefetchError
                        )
                    try checkNativeStatus(prefetchStatus, error: &prefetchError)
                }
                var nativeConfig = config.toC()
                var limits = msplat_default_training_limits()
                var refinementOptions = config.toRefinementOptionsV8()
                var maskOptions = config.toTrainingMaskOptionsV11()
                if let maximumGaussianCount {
                    guard let nativeLimit = Int32(exactly: maximumGaussianCount) else {
                        throw MsplatError.invalidArgument(
                            "maximumGaussianCount is outside the native range"
                        )
                    }
                    limits.maxGaussians = nativeLimit
                }
                var trainer: MsplatTrainer?
                var nativeError = MsplatErrorInfo()
                let trainerStatus = msplat_trainer_create_v11(
                    dataset,
                    &nativeConfig,
                    MemoryLayout<MsplatConfig>.size,
                    &limits,
                    MemoryLayout<MsplatTrainingLimits>.size,
                    &refinementOptions,
                    MemoryLayout<MsplatRefinementOptionsV8>.size,
                    &maskOptions,
                    MemoryLayout<MsplatTrainingMaskOptionsV11>.size,
                    &trainer,
                    &nativeError
                )
                try checkNativeStatus(trainerStatus, error: &nativeError)
                guard let trainer else {
                    throw MsplatError.internalFailure(
                        "Native trainer creation returned no handle"
                    )
                }
                return (dataset, trainer)
            } catch {
                var destroyError = MsplatErrorInfo()
                _ = msplat_dataset_destroy_v2(dataset, &destroyError)
                throw error
            }
        }
    }

    private static func validateCreation(
        config: TrainingConfig,
        maximumGaussianCount: Int?
    ) throws {
        try config.validate()
        if let maximumGaussianCount {
            guard (1...Int(Int32.max)).contains(maximumGaussianCount) else {
                throw MsplatError.invalidArgument(
                    "maximumGaussianCount must be in 1...2147483647"
                )
            }
        }
    }

    private static func takePixelData(_ buffer: inout MsplatPixelBuffer) throws -> PixelData {
        defer { msplat_pixel_buffer_free(&buffer) }
        let width = Int(buffer.width)
        let height = Int(buffer.height)
        let layout = try checkedImageLayout(width: width, height: height, components: 3)
        guard let data = buffer.data else {
            throw MsplatError.internalFailure("Native rendering returned no pixel data")
        }
        return PixelData(
            pixels: Array(UnsafeBufferPointer(start: data, count: layout.elementCount)),
            width: width,
            height: height
        )
    }

    private func withDataset<Result>(
        _ operation: (MsplatDataset) throws -> Result
    ) throws -> Result {
        try resources.withDataset(operation)
    }

    private func withTrainer<Result>(
        _ operation: (MsplatTrainer) throws -> Result
    ) throws -> Result {
        try resources.withTrainer(operation)
    }
}

/// Owns native handles independently of global-actor deinitializer support.
private final class SessionResources: @unchecked Sendable {
    private struct DatasetHandle: @unchecked Sendable {
        let rawValue: MsplatDataset
    }

    private struct TrainerHandle: @unchecked Sendable {
        let rawValue: MsplatTrainer
    }

    private struct State: @unchecked Sendable {
        var dataset: DatasetHandle?
        var trainer: TrainerHandle?
        var securityScopes: SecurityScopedURLLeases?
        var ownsSessionReservation = true
    }

    private struct DetachedResources: @unchecked Sendable {
        let dataset: DatasetHandle?
        let trainer: TrainerHandle?
        let securityScopes: SecurityScopedURLLeases?
        let ownsSessionReservation: Bool
    }

    private let state: Mutex<State>

    init(
        dataset: MsplatDataset,
        trainer: MsplatTrainer,
        securityScopes: SecurityScopedURLLeases
    ) {
        state = Mutex(State(
            dataset: DatasetHandle(rawValue: dataset),
            trainer: TrainerHandle(rawValue: trainer),
            securityScopes: securityScopes
        ))
    }

    func withDataset<Result>(
        _ operation: (MsplatDataset) throws -> Result
    ) throws -> Result {
        try withNativeEngineLock {
            let handle = try state.withLock { state in
                guard let dataset = state.dataset else {
                    throw MsplatError.invalidArgument("MsplatSession is closed")
                }
                return dataset
            }
            return try operation(handle.rawValue)
        }
    }

    func withTrainer<Result>(
        _ operation: (MsplatTrainer) throws -> Result
    ) throws -> Result {
        try withNativeEngineLock {
            let handle = try state.withLock { state in
                guard let trainer = state.trainer else {
                    throw MsplatError.invalidArgument("MsplatSession is closed")
                }
                return trainer
            }
            return try operation(handle.rawValue)
        }
    }

    func close() throws {
        var firstError: Error?
        let detached = state.withLock { state in
            let resources = DetachedResources(
                dataset: state.dataset,
                trainer: state.trainer,
                securityScopes: state.securityScopes,
                ownsSessionReservation: state.ownsSessionReservation
            )
            state.dataset = nil
            state.trainer = nil
            state.securityScopes = nil
            state.ownsSessionReservation = false
            return resources
        }

        if let trainer = detached.trainer {
            do {
                try withNativeEngineLock {
                    var nativeError = MsplatErrorInfo()
                    let status = msplat_trainer_destroy_v2(trainer.rawValue, &nativeError)
                    try checkNativeStatus(status, error: &nativeError)
                }
            } catch {
                firstError = error
            }
        }

        if let dataset = detached.dataset {
            do {
                try withNativeEngineLock {
                    var nativeError = MsplatErrorInfo()
                    let status = msplat_dataset_destroy_v2(dataset.rawValue, &nativeError)
                    try checkNativeStatus(status, error: &nativeError)
                }
            } catch {
                if firstError == nil { firstError = error }
            }
        }

        var securityScopes = detached.securityScopes
        securityScopes?.close()
        if detached.ownsSessionReservation { releaseNativeSession() }

        if let firstError { throw firstError }
    }

    deinit {
        try? close()
    }
}

private struct SecurityScopedURLLease: Sendable {
    let url: URL
    private var isActive: Bool

    init(url: URL) {
        self.url = url
        isActive = url.startAccessingSecurityScopedResource()
    }

    mutating func close() {
        guard isActive else { return }
        isActive = false
        url.stopAccessingSecurityScopedResource()
    }
}

private struct SecurityScopedURLLeases: Sendable {
    private var leases: [SecurityScopedURLLease]

    init(urls: [URL]) {
        var seen = Set<URL>()
        var leases: [SecurityScopedURLLease] = []
        leases.reserveCapacity(urls.count)
        for url in urls {
            let key = url.standardizedFileURL
            guard seen.insert(key).inserted else { continue }
            leases.append(SecurityScopedURLLease(url: url))
        }
        self.leases = leases
    }

    mutating func close() {
        for index in leases.indices {
            leases[index].close()
        }
        leases.removeAll(keepingCapacity: false)
    }
}

private var metallibResourceName: String {
    #if os(macOS)
    return "default-macos"
    #elseif targetEnvironment(simulator)
    return "default-iossimulator"
    #else
    return "default-ios"
    #endif
}

private func nativeIndex(_ value: Int) throws -> Int32 {
    guard let value = Int32(exactly: value) else {
        throw MsplatError.invalidArgument("Camera index is outside the supported range")
    }
    return value
}

private func filePath(_ url: URL) throws -> String {
    guard url.isFileURL, !url.path.isEmpty else {
        throw MsplatError.invalidArgument("Output must be a file URL with a non-empty path")
    }
    return url.path
}

private func checkedImageLayout(
    width: Int,
    height: Int,
    components: Int
) throws -> (elementsPerRow: Int, elementCount: Int) {
    guard width > 0, height > 0 else {
        throw MsplatError.internalFailure("Native rendering returned invalid dimensions")
    }
    let (elementsPerRow, rowOverflow) = width.multipliedReportingOverflow(by: components)
    let (elementCount, countOverflow) = elementsPerRow.multipliedReportingOverflow(by: height)
    guard !rowOverflow, !countOverflow else {
        throw MsplatError.outOfMemory("Render dimensions exceed Swift buffer limits")
    }
    return (elementsPerRow, elementCount)
}
