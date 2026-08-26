import Metal
import MsplatCore

/// A pending GPU-native preview submission.
///
/// At most one submission should be in flight per display consumer. Waiting is
/// cancellation-aware and does not hold the msplat runtime actor.
public final class MetalPreviewSubmission: @unchecked Sendable {
    private let storage: PreviewFrameStorage

    init(handle: MsplatPreviewFrame) {
        storage = PreviewFrameStorage(handle: handle)
    }

    /// Polls the native completion state without waiting.
    public func poll() throws -> Bool {
        try storage.poll()
    }

    /// Suspends until the separately owned preview texture is immutable and
    /// safe to sample from another command queue.
    public func waitUntilReady() async throws -> MetalPreviewSurface {
        while try !storage.poll() {
            try Task.checkCancellation()
            try await Task.sleep(for: .milliseconds(2))
        }
        try Task.checkCancellation()
        return try storage.surface()
    }
}

/// A completed, immutable GPU-native preview.
///
/// The texture uses non-sRGB `bgra8Unorm`, has opaque alpha, and remains valid
/// until every surface/submission sharing its native frame storage is released.
/// The unchecked Sendable conformance is intentionally narrow: Metal protocol
/// objects do not declare Sendable, but this completed texture is immutable.
public final class MetalPreviewSurface: @unchecked Sendable {
    public let texture: any MTLTexture
    public let width: Int
    public let height: Int

    private let storage: PreviewFrameStorage

    fileprivate init(
        texture: any MTLTexture,
        width: Int,
        height: Int,
        storage: PreviewFrameStorage
    ) {
        self.texture = texture
        self.width = width
        self.height = height
        self.storage = storage
    }
}

private final class PreviewFrameStorage: @unchecked Sendable {
    private let handle: MsplatPreviewFrame

    init(handle: MsplatPreviewFrame) {
        self.handle = handle
    }

    deinit {
        _ = msplat_preview_frame_destroy_v13(handle, nil)
    }

    func poll() throws -> Bool {
        var ready = false
        var nativeError = MsplatErrorInfo()
        let status = msplat_preview_frame_poll_v13(
            handle,
            &ready,
            &nativeError
        )
        try checkNativeStatus(status, error: &nativeError)
        return ready
    }

    func surface() throws -> MetalPreviewSurface {
        var texture: (any MTLTexture)?
        var width: Int32 = 0
        var height: Int32 = 0
        var nativeError = MsplatErrorInfo()
        let status = msplat_preview_frame_texture_v13(
            handle,
            &texture,
            &width,
            &height,
            &nativeError
        )
        try checkNativeStatus(status, error: &nativeError)

        guard let texture else {
            throw MsplatError.internalFailure(
                "Completed native preview returned no Metal texture"
            )
        }
        guard width > 0, height > 0,
              texture.width == Int(width), texture.height == Int(height),
              texture.pixelFormat == .bgra8Unorm,
              texture.usage.contains(.shaderRead) else {
            throw MsplatError.internalFailure(
                "Completed native preview returned an invalid texture contract"
            )
        }
        return MetalPreviewSurface(
            texture: texture,
            width: Int(width),
            height: Int(height),
            storage: self
        )
    }
}
