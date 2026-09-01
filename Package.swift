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
            url: "https://github.com/Plinth-it/msplat-ios/releases/download/binary-v2.0.0/MsplatCore.xcframework.zip",
            checksum: "4726a0f4b37262c1807355007d39229da3d3a3d46c4e1ad5d72f61450b35d164"
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
