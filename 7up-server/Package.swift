// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "7up-server",
    platforms: [
        .macOS(.v14),
    ],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
        .package(name: "ml-stable-diffusion", path: "../ml-stable-diffusion"),
    ],
    targets: [
        .executableTarget(
            name: "7up-server",
            dependencies: [
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "StableDiffusion", package: "ml-stable-diffusion"),
            ],
            path: "Sources"
        ),
    ]
)
