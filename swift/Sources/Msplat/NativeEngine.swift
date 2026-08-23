import MsplatCore
import Foundation
import Synchronization

private struct NativeEngineState {
    var metallibConfigured = false
    var hasActiveSession = false
}

private let nativeEngineState = Mutex(NativeEngineState())

/// The native command buffer and transient caches are process-global. Keep all
/// Swift entry points serialized until those resources become instance-owned.
func withNativeEngineLock<Result>(_ operation: () throws -> Result) rethrows -> Result {
    try nativeEngineState.withLock { _ in
        try operation()
    }
}

/// Reserve the process-global engine for one ownership-safe session.
func reserveNativeSession() throws {
    try nativeEngineState.withLock { state in
        guard !state.hasActiveSession else {
            throw MsplatError.invalidArgument("Another MsplatSession is already active")
        }
        state.hasActiveSession = true
    }
}

/// Release a session reservation. Idempotent so fallback teardown is safe.
func releaseNativeSession() {
    nativeEngineState.withLock { state in
        state.hasActiveSession = false
    }
}

/// Configure the platform-specific shader library exactly once, marking it
/// complete only after the resource and native call both succeed.
func withConfiguredNativeEngine<Result>(_ metallibResourceName: String,
                                        _ operation: () throws -> Result) throws -> Result {
    try nativeEngineState.withLock { state in
        if !state.metallibConfigured {
            let expectedABI = UInt32(MSPLAT_ABI_VERSION)
            let actualABI = msplat_abi_version()
            guard actualABI == expectedABI else {
                throw MsplatError.incompatibleABI(expected: expectedABI, actual: actualABI)
            }

            guard let path = Bundle.module.path(
                forResource: metallibResourceName,
                ofType: "metallib"
            ) else {
                throw MsplatError.metallibNotFound(
                    "Missing \(metallibResourceName).metallib in the Msplat package"
                )
            }

            var nativeError = MsplatErrorInfo()
            let status = msplat_set_metallib_path_v2(path, &nativeError)
            try checkNativeStatus(status, error: &nativeError)
            state.metallibConfigured = true
        }

        return try operation()
    }
}
