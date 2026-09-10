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
            url: "https://github.com/Plinth-it/msplat-ios/releases/download/binary-v2.1.1/MsplatCore.xcframework.zip",
            checksum: "a3dac0f7c7905e72874fd603b98d7f6b9af294584df1007872efd02a4611fbd6"
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
