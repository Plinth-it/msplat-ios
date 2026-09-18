// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Msplat",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "Msplat", targets: ["Msplat"]),
    ],
    targets: [
        .binaryTarget(
            name: "MsplatCore",
            url: "https://github.com/Plinth-it/msplat-ios/releases/download/binary-v2.1.2/MsplatCore.xcframework.zip",
            checksum: "9f8b93e11498f263c1af81c3416b131c86c03f1da702f6b3dcb374e1ed0e881b"
        ),
        .target(
            name: "Msplat",
            dependencies: ["MsplatCore"],
            path: "swift/Sources/Msplat",
            resources: [
                .copy("Resources/default-macos.metallib"),
                .copy("Resources/default-ios.metallib"),
                .copy("Resources/default-iossimulator.metallib"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("Foundation"),
                .linkedFramework("ImageIO"),
                .linkedFramework("CoreGraphics"),
                .linkedLibrary("z"),
                .linkedLibrary("c++"),
            ]
        ),
        .testTarget(
            name: "MsplatTests",
            dependencies: ["Msplat", "MsplatCore"],
            path: "swift/Tests"
        ),
    ]
)
