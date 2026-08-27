import MsplatCore
import Foundation

/// Recoverable errors reported by the native msplat engine.
public enum MsplatError: LocalizedError, Sendable, Equatable {
    case invalidArgument(String)
    case invalidDataset(String)
    case outOfMemory(String)
    case gpuFailure(String)
    case ioFailure(String)
    case cancelled(String)
    case internalFailure(String)
    case incompatibleABI(expected: UInt32, actual: UInt32)
    case metallibNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .invalidArgument(let message),
             .invalidDataset(let message),
             .outOfMemory(let message),
             .gpuFailure(let message),
             .ioFailure(let message),
             .cancelled(let message),
             .internalFailure(let message),
             .metallibNotFound(let message):
            return message
        case .incompatibleABI(let expected, let actual):
            return "Incompatible msplat ABI: Swift expects \(expected), native library provides \(actual)"
        }
    }
}

/// Errors raised before a render writes into caller-owned storage.
public enum MsplatRenderBufferError: LocalizedError, Sendable, Equatable {
    case insufficientCapacity(required: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .insufficientCapacity(let required, let actual):
            return "Buffer capacity is \(actual) bytes; \(required) bytes are required"
        }
    }
}

func checkNativeStatus(_ status: MsplatStatus,
                       error nativeError: inout MsplatErrorInfo) throws {
    guard status.rawValue != 0 else { return }

    let message = withUnsafeBytes(of: &nativeError.message) { bytes -> String in
        guard let baseAddress = bytes.baseAddress else {
            return "The native msplat operation failed"
        }
        let value = String(cString: baseAddress.assumingMemoryBound(to: CChar.self))
        return value.isEmpty ? "The native msplat operation failed" : value
    }

    switch status.rawValue {
    case 1: throw MsplatError.invalidArgument(message)
    case 2: throw MsplatError.invalidDataset(message)
    case 3: throw MsplatError.outOfMemory(message)
    case 4: throw MsplatError.gpuFailure(message)
    case 5: throw MsplatError.ioFailure(message)
    case 6: throw MsplatError.cancelled(message)
    default: throw MsplatError.internalFailure(message)
    }
}
