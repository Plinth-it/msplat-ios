import CoreGraphics
import Foundation
import simd

enum CaptureGeometry {
    static func orientedGeometry(
        calibration: CaptureCalibrationRecord,
        cameraToWorld: simd_float4x4,
        orientation: CaptureDisplayOrientation
    ) -> (
        calibration: CaptureCalibrationRecord,
        cameraToWorld: simd_float4x4
    ) {
        let orientedCalibration: CaptureCalibrationRecord
        let sourceCameraFromOrientedCamera: simd_float4x4

        switch orientation {
        case .up:
            orientedCalibration = calibration
            sourceCameraFromOrientedCamera = matrix_identity_float4x4
        case .right:
            orientedCalibration = CaptureCalibrationRecord(
                width: calibration.height,
                height: calibration.width,
                fx: calibration.fy,
                fy: calibration.fx,
                cx: Float(calibration.height) - calibration.cy,
                cy: calibration.cx
            )
            sourceCameraFromOrientedCamera = simd_float4x4(
                SIMD4<Float>(0, 1, 0, 0),
                SIMD4<Float>(-1, 0, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            )
        case .down:
            orientedCalibration = CaptureCalibrationRecord(
                width: calibration.width,
                height: calibration.height,
                fx: calibration.fx,
                fy: calibration.fy,
                cx: Float(calibration.width) - calibration.cx,
                cy: Float(calibration.height) - calibration.cy
            )
            sourceCameraFromOrientedCamera = simd_float4x4(
                SIMD4<Float>(-1, 0, 0, 0),
                SIMD4<Float>(0, -1, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            )
        case .left:
            orientedCalibration = CaptureCalibrationRecord(
                width: calibration.height,
                height: calibration.width,
                fx: calibration.fy,
                fy: calibration.fx,
                cx: calibration.cy,
                cy: Float(calibration.width) - calibration.cx
            )
            sourceCameraFromOrientedCamera = simd_float4x4(
                SIMD4<Float>(0, -1, 0, 0),
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            )
        }

        return (
            calibration: orientedCalibration,
            cameraToWorld: cameraToWorld * sourceCameraFromOrientedCamera
        )
    }

    static func orientedInterleavedBytes(
        _ source: [UInt8],
        width: Int,
        height: Int,
        components: Int,
        orientation: CaptureDisplayOrientation
    ) throws -> (bytes: [UInt8], width: Int, height: Int) {
        guard width > 0, height > 0, components > 0 else {
            throw CaptureFailure.invalidFrame(
                "capture raster dimensions and component count must be positive"
            )
        }
        let (pixelCount, pixelCountOverflow) = width.multipliedReportingOverflow(
            by: height
        )
        let (expectedCount, byteCountOverflow) = pixelCount
            .multipliedReportingOverflow(by: components)
        guard !pixelCountOverflow, !byteCountOverflow,
              source.count == expectedCount else {
            throw CaptureFailure.invalidFrame(
                "capture raster storage does not match its dimensions"
            )
        }

        let destinationWidth: Int
        let destinationHeight: Int
        switch orientation {
        case .up, .down:
            destinationWidth = width
            destinationHeight = height
        case .right, .left:
            destinationWidth = height
            destinationHeight = width
        }
        guard orientation != .up else {
            return (source, destinationWidth, destinationHeight)
        }

        var destination = [UInt8](repeating: 0, count: expectedCount)
        for sourceY in 0..<height {
            for sourceX in 0..<width {
                let destinationX: Int
                let destinationY: Int
                switch orientation {
                case .right:
                    destinationX = height - 1 - sourceY
                    destinationY = sourceX
                case .down:
                    destinationX = width - 1 - sourceX
                    destinationY = height - 1 - sourceY
                case .left:
                    destinationX = sourceY
                    destinationY = width - 1 - sourceX
                case .up:
                    destinationX = sourceX
                    destinationY = sourceY
                }

                let sourceOffset = (sourceY * width + sourceX) * components
                let destinationOffset = (
                    destinationY * destinationWidth + destinationX
                ) * components
                for component in 0..<components {
                    destination[destinationOffset + component] =
                        source[sourceOffset + component]
                }
            }
        }
        return (destination, destinationWidth, destinationHeight)
    }

    static func rowMajorCameraToWorld(_ matrix: simd_float4x4) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(16)
        for row in 0..<4 {
            for column in 0..<4 {
                values.append(matrix[column][row])
            }
        }
        return values
    }

    static func cameraToWorld(fromRowMajor values: [Float]) throws -> simd_float4x4 {
        guard values.count == 16 else {
            throw CaptureFailure.invalidFrame(
                "camera transform must contain exactly 16 values"
            )
        }
        var matrix = matrix_identity_float4x4
        for row in 0..<4 {
            for column in 0..<4 {
                matrix[column][row] = values[row * 4 + column]
            }
        }
        return matrix
    }

    static func translationDistance(
        from first: simd_float4x4,
        to second: simd_float4x4
    ) -> Float {
        simd_distance(first.columns.3.xyz, second.columns.3.xyz)
    }

    static func rotationAngle(
        from first: simd_float4x4,
        to second: simd_float4x4
    ) -> Float {
        let firstRotation = simd_float3x3(
            first.columns.0.xyz,
            first.columns.1.xyz,
            first.columns.2.xyz
        )
        let secondRotation = simd_float3x3(
            second.columns.0.xyz,
            second.columns.1.xyz,
            second.columns.2.xyz
        )
        let relative = simd_transpose(firstRotation) * secondRotation
        let cosine = max(-1, min(1, (relative.trace - 1) * 0.5))
        return acos(cosine)
    }

    /// Back-projects a depth-map pixel into ARKit world coordinates. Image
    /// coordinates are top-left origin; ARKit camera space is Y-up, Z-back.
    static func worldPoint(
        depth: Float,
        depthPixel: SIMD2<Float>,
        depthWidth: Int,
        depthHeight: Int,
        calibration: CaptureCalibrationRecord,
        cameraToWorld: simd_float4x4
    ) -> SIMD3<Float>? {
        guard depth.isFinite, depth > 0,
              depthWidth > 0, depthHeight > 0 else {
            return nil
        }
        let scaleX = Float(depthWidth) / Float(calibration.width)
        let scaleY = Float(depthHeight) / Float(calibration.height)
        let fx = calibration.fx * scaleX
        let fy = calibration.fy * scaleY
        let cx = calibration.cx * scaleX
        let cy = calibration.cy * scaleY
        guard fx > 0, fy > 0 else { return nil }

        let cameraPoint = SIMD4<Float>(
            (depthPixel.x - cx) * depth / fx,
            -(depthPixel.y - cy) * depth / fy,
            -depth,
            1
        )
        let world = cameraToWorld * cameraPoint
        guard world.x.isFinite, world.y.isFinite, world.z.isFinite else {
            return nil
        }
        return world.xyz
    }

    /// Projects a world point into native encoded-image coordinates with a
    /// top-left origin. Returns nil for points behind the camera.
    static func normalizedImagePoint(
        worldPoint: SIMD3<Float>,
        calibration: CaptureCalibrationRecord,
        cameraToWorld: simd_float4x4
    ) -> CGPoint? {
        let camera = simd_inverse(cameraToWorld) * SIMD4<Float>(worldPoint, 1)
        let forwardDepth = -camera.z
        guard forwardDepth.isFinite, forwardDepth > 0 else { return nil }

        let x = calibration.fx * camera.x / forwardDepth + calibration.cx
        let y = calibration.cy - calibration.fy * camera.y / forwardDepth
        guard x.isFinite, y.isFinite else { return nil }
        return CGPoint(
            x: CGFloat(x / Float(calibration.width)),
            y: CGFloat(y / Float(calibration.height))
        )
    }
}

private extension SIMD4 where Scalar == Float {
    var xyz: SIMD3<Float> { SIMD3(x, y, z) }
}

private extension simd_float3x3 {
    var trace: Float {
        columns.0.x + columns.1.y + columns.2.z
    }
}
