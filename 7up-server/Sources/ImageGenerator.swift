import CoreGraphics
import CoreML
import Foundation
import ImageIO
import StableDiffusion
import UniformTypeIdentifiers

/// Thread-safe wrapper around StableDiffusionXLPipeline.
/// Uses an actor to serialize access to the non-Sendable pipeline.
actor ImageGenerator {
    private let pipeline: StableDiffusionXLPipeline

    init(modelPath: String) throws {
        let resourceURL = URL(fileURLWithPath: modelPath)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        self.pipeline = try StableDiffusionXLPipeline(
            resourcesAt: resourceURL,
            configuration: config,
            reduceMemory: false
        )
        try self.pipeline.loadResources()
    }

    struct Request: Sendable {
        var prompt: String
        var negativePrompt: String
        var seed: UInt32
        var width: Int
        var height: Int
        var steps: Int
        var guidanceScale: Float
        var schedulerName: String
    }

    struct Result: Sendable {
        var pngData: Data
        var elapsedSeconds: Double
    }

    private func schedulerType(from name: String) -> StableDiffusionScheduler {
        switch name {
        case "pndm": return .pndmScheduler
        case "dpmSolverMultistep": return .dpmSolverMultistepScheduler
        case "discreteFlow": return .discreteFlowScheduler
        case "eulerAncestral": return .eulerAncestralDiscreteScheduler
        default: return .eulerAncestralDiscreteScheduler
        }
    }

    func generate(_ request: Request) throws -> Result {
        let start = CFAbsoluteTimeGetCurrent()

        var pipelineConfig = StableDiffusionXLPipeline.Configuration(prompt: request.prompt)
        pipelineConfig.negativePrompt = request.negativePrompt
        pipelineConfig.seed = request.seed
        pipelineConfig.stepCount = request.steps
        pipelineConfig.guidanceScale = request.guidanceScale
        pipelineConfig.schedulerType = schedulerType(from: request.schedulerName)
        pipelineConfig.imageCount = 1

        let images = try pipeline.generateImages(configuration: pipelineConfig) { progress in
            return true  // continue generation
        }

        guard let cgImage = images.first, let image = cgImage else {
            throw GenerationError.noImageProduced
        }

        // Resize if requested dimensions differ from native 512x512
        let finalImage: CGImage
        if request.width != image.width || request.height != image.height {
            finalImage = try resizeImage(image, to: (request.width, request.height))
        } else {
            finalImage = image
        }

        let pngData = try encodePNG(finalImage)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        return Result(pngData: pngData, elapsedSeconds: elapsed)
    }

    private func resizeImage(_ image: CGImage, to size: (width: Int, height: Int)) throws -> CGImage {
        guard let ctx = CGContext(
            data: nil,
            width: size.width,
            height: size.height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw GenerationError.resizeFailed
        }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        guard let resized = ctx.makeImage() else {
            throw GenerationError.resizeFailed
        }
        return resized
    }

    private func encodePNG(_ image: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            data as CFMutableData,
            "public.png" as CFString,
            1,
            nil
        ) else {
            throw GenerationError.pngEncodingFailed
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw GenerationError.pngEncodingFailed
        }
        return data as Data
    }
}

enum GenerationError: Error, CustomStringConvertible {
    case noImageProduced
    case pngEncodingFailed
    case resizeFailed

    var description: String {
        switch self {
        case .noImageProduced: return "Pipeline returned no image"
        case .pngEncodingFailed: return "Failed to encode CGImage as PNG"
        case .resizeFailed: return "Failed to resize image"
        }
    }
}
