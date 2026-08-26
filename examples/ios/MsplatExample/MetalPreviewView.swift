import CoreImage
import MetalKit
import Msplat
import SwiftUI

/// Displays a completed native preview texture without copying it through a
/// Swift `Data`, `CGImage`, or `UIImage`.
@MainActor
struct MetalPreviewView: UIViewRepresentable {
    let surface: MetalPreviewSurface

    /// The legacy `UIImage` path converted the renderer's scanline order when
    /// displaying it. Restore that display-only flip for the native texture.
    nonisolated static func textureToCoreImageTransform(
        height: Int
    ) -> CGAffineTransform {
        CGAffineTransform(
            a: 1,
            b: 0,
            c: 0,
            d: -1,
            tx: 0,
            ty: CGFloat(height)
        )
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: surface.texture.device)
        view.autoResizeDrawable = false
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = false
        view.enableSetNeedsDisplay = true
        view.isPaused = true
        view.clearColor = MTLClearColorMake(0, 0, 0, 1)
        view.delegate = context.coordinator
        context.coordinator.update(surface: surface, in: view)
        return view
    }

    func updateUIView(_ view: MTKView, context: Context) {
        context.coordinator.update(surface: surface, in: view)
    }

    static func dismantleUIView(_ view: MTKView, coordinator: Coordinator) {
        view.delegate = nil
        coordinator.clear()
    }

    @MainActor
    final class Coordinator: NSObject, MTKViewDelegate {
        private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
            ?? CGColorSpaceCreateDeviceRGB()
        private var surface: MetalPreviewSurface?
        private var context: CIContext?
        private var commandQueue: MTLCommandQueue?
        private var deviceRegistryID: UInt64?

        func update(surface: MetalPreviewSurface, in view: MTKView) {
            let device = surface.texture.device
            if deviceRegistryID != device.registryID {
                deviceRegistryID = device.registryID
                context = CIContext(mtlDevice: device)
                commandQueue = device.makeCommandQueue()
                view.device = device
            }

            self.surface = surface
            view.drawableSize = CGSize(
                width: CGFloat(surface.width),
                height: CGFloat(surface.height)
            )
            view.setNeedsDisplay()
        }

        func clear() {
            surface = nil
            context = nil
            commandQueue = nil
            deviceRegistryID = nil
        }

        func draw(in view: MTKView) {
            guard let surface,
                  let context,
                  let commandBuffer = commandQueue?.makeCommandBuffer(),
                  let drawable = view.currentDrawable,
                  let sourceImage = CIImage(
                    mtlTexture: surface.texture,
                    options: [.colorSpace: colorSpace]
                  ) else {
                return
            }

            let image = sourceImage.transformed(
                by: MetalPreviewView.textureToCoreImageTransform(
                    height: surface.height
                )
            )

            let bounds = CGRect(
                x: 0,
                y: 0,
                width: CGFloat(surface.width),
                height: CGFloat(surface.height)
            )
            context.render(
                image,
                to: drawable.texture,
                commandBuffer: commandBuffer,
                bounds: bounds,
                colorSpace: colorSpace
            )
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    }
}
