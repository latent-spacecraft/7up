import CoreGraphics
import CoreImage
import CoreML
import Foundation
import ImageIO
import StableDiffusion
import Vision

/// Thread-safe wrapper around StableDiffusionXLPipeline.
/// Uses an actor to serialize access to the non-Sendable pipeline.
actor ImageGenerator {
    private let pipeline: StableDiffusionXLPipeline
    private let ciContext = CIContext()

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
        var removeBackground: Bool  // Use Vision framework to create alpha mask
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
            return true
        }

        guard let cgImage = images.first, let image = cgImage else {
            throw GenerationError.noImageProduced
        }

        // Resize if needed
        var finalImage = image
        if request.width != image.width || request.height != image.height {
            finalImage = try resizeImage(image, to: (request.width, request.height))
        }

        // Remove background if requested
        if request.removeBackground {
            finalImage = try removeBackground(from: finalImage)
        }

        let pngData = try encodePNG(finalImage)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        return Result(pngData: pngData, elapsedSeconds: elapsed)
    }

    // MARK: - Sprite Extraction via Apple Vision Instance Segmentation

    struct Sprite: Sendable {
        var pngData: Data
        var x: Int          // Bounding box in source image
        var y: Int
        var width: Int
        var height: Int
    }

    struct SpritesResult: Sendable {
        var sprites: [Sprite]
        var elapsedSeconds: Double
    }

    /// Generate an image, then use Vision to isolate each individual object as a separate sprite.
    /// This is the same tech behind "Copy Subject" in Photos/Messages.
    func generateSprites(_ request: Request) throws -> SpritesResult {
        let start = CFAbsoluteTimeGetCurrent()

        // Generate the image
        var pipelineConfig = StableDiffusionXLPipeline.Configuration(prompt: request.prompt)
        pipelineConfig.negativePrompt = request.negativePrompt
        pipelineConfig.seed = request.seed
        pipelineConfig.stepCount = request.steps
        pipelineConfig.guidanceScale = request.guidanceScale
        pipelineConfig.schedulerType = schedulerType(from: request.schedulerName)
        pipelineConfig.imageCount = 1

        let images = try pipeline.generateImages(configuration: pipelineConfig) { _ in true }
        guard let cgImage = images.first, let image = cgImage else {
            throw GenerationError.noImageProduced
        }

        // Run instance segmentation
        let instanceRequest = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([instanceRequest])

        guard let observation = instanceRequest.results?.first else {
            // No instances found — return the whole image as one sprite
            let pngData = try encodePNG(image)
            let sprite = Sprite(pngData: pngData, x: 0, y: 0, width: image.width, height: image.height)
            return SpritesResult(sprites: [sprite], elapsedSeconds: CFAbsoluteTimeGetCurrent() - start)
        }

        let allInstances = observation.allInstances
        var sprites: [Sprite] = []

        // Extract each instance individually
        for instanceIdx in allInstances {
            let singleInstance = IndexSet([instanceIdx])

            // Get mask for just this instance
            let maskBuffer = try observation.generateScaledMaskForImage(
                forInstances: singleInstance,
                from: handler
            )

            let maskCI = CIImage(cvPixelBuffer: maskBuffer)
            let originalCI = CIImage(cgImage: image)

            // Apply mask
            guard let blendFilter = CIFilter(name: "CIBlendWithMask") else { continue }
            blendFilter.setValue(originalCI, forKey: kCIInputImageKey)
            blendFilter.setValue(CIImage.empty(), forKey: kCIInputBackgroundImageKey)
            blendFilter.setValue(maskCI, forKey: "inputMaskImage")

            guard let maskedCI = blendFilter.outputImage else { continue }

            // Find the bounding box of non-transparent pixels
            // Use the mask to determine bounds
            let maskedCG = ciContext.createCGImage(
                maskedCI, from: originalCI.extent,
                format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB()
            )
            guard let fullImage = maskedCG else { continue }

            // Compute tight bounding box from the mask
            let bbox = computeAlphaBoundingBox(fullImage)
            guard bbox.width > 4 && bbox.height > 4 else { continue } // Skip tiny fragments

            // Crop to bounding box with small padding
            let pad = 4
            let cropX = max(0, bbox.minX - pad)
            let cropY = max(0, bbox.minY - pad)
            let cropW = min(fullImage.width - cropX, bbox.width + pad * 2)
            let cropH = min(fullImage.height - cropY, bbox.height + pad * 2)
            let cropRect = CGRect(x: cropX, y: cropY, width: cropW, height: cropH)

            guard let croppedCG = fullImage.cropping(to: cropRect) else { continue }

            let pngData = try encodePNG(croppedCG)
            sprites.append(Sprite(
                pngData: pngData,
                x: cropX, y: cropY,
                width: cropW, height: cropH
            ))
        }

        // Sort by x position (left to right)
        sprites.sort { $0.x < $1.x }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return SpritesResult(sprites: sprites, elapsedSeconds: elapsed)
    }

    /// Compute the tight bounding box of non-transparent pixels in a RGBA image.
    private func computeAlphaBoundingBox(_ image: CGImage) -> (minX: Int, minY: Int, width: Int, height: Int) {
        guard let data = image.dataProvider?.data,
              let ptr = CFDataGetBytePtr(data) else {
            return (0, 0, image.width, image.height)
        }

        let w = image.width
        let h = image.height
        let bytesPerRow = image.bytesPerRow
        var minX = w, minY = h, maxX = 0, maxY = 0

        for y in 0..<h {
            for x in 0..<w {
                let offset = y * bytesPerRow + x * 4
                let alpha = ptr[offset + 3]
                if alpha > 20 { // Threshold to ignore near-transparent pixels
                    minX = min(minX, x)
                    minY = min(minY, y)
                    maxX = max(maxX, x)
                    maxY = max(maxY, y)
                }
            }
        }

        if maxX < minX { return (0, 0, w, h) } // No opaque pixels found
        return (minX, minY, maxX - minX + 1, maxY - minY + 1)
    }

    // MARK: - Background Removal via Apple Vision

    /// Uses VNGenerateForegroundInstanceMaskRequest to isolate the subject
    /// and produce a CGImage with proper alpha transparency.
    private func removeBackground(from image: CGImage) throws -> CGImage {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])

        guard let result = request.results?.first else {
            // No foreground detected — return original with full alpha
            return image
        }

        // Get the mask as a pixel buffer
        let maskPixelBuffer = try result.generateScaledMaskForImage(
            forInstances: result.allInstances,
            from: handler
        )

        // Convert mask to CIImage
        let maskCI = CIImage(cvPixelBuffer: maskPixelBuffer)

        // Original image as CIImage
        let originalCI = CIImage(cgImage: image)

        // Apply mask as alpha channel
        // The mask is grayscale — white = foreground, black = background
        guard let filter = CIFilter(name: "CIBlendWithMask") else {
            throw GenerationError.backgroundRemovalFailed
        }
        filter.setValue(originalCI, forKey: kCIInputImageKey)
        filter.setValue(CIImage.empty(), forKey: kCIInputBackgroundImageKey)
        filter.setValue(maskCI, forKey: "inputMaskImage")

        guard let outputCI = filter.outputImage else {
            throw GenerationError.backgroundRemovalFailed
        }

        // Render to CGImage with alpha
        guard let outputCG = ciContext.createCGImage(
            outputCI,
            from: originalCI.extent,
            format: .RGBA8,
            colorSpace: CGColorSpaceCreateDeviceRGB()
        ) else {
            throw GenerationError.backgroundRemovalFailed
        }

        return outputCG
    }

    // MARK: - Resize

    private func resizeImage(_ image: CGImage, to size: (width: Int, height: Int)) throws -> CGImage {
        guard let ctx = CGContext(
            data: nil,
            width: size.width,
            height: size.height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
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

    // MARK: - PNG Encoding

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
    case backgroundRemovalFailed

    var description: String {
        switch self {
        case .noImageProduced: return "Pipeline returned no image"
        case .pngEncodingFailed: return "Failed to encode CGImage as PNG"
        case .resizeFailed: return "Failed to resize image"
        case .backgroundRemovalFailed: return "Vision framework background removal failed"
        }
    }
}
