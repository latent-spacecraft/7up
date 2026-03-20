import Foundation
import Hummingbird
import Logging

// ---------------------------------------------------------------------------
// CLI argument parsing (lightweight, no ArgumentParser dependency)
// ---------------------------------------------------------------------------
func parseArg(_ flag: String, default defaultVal: String) -> String {
    let args = CommandLine.arguments
    if let idx = args.firstIndex(of: flag), idx + 1 < args.count {
        return args[idx + 1]
    }
    return defaultVal
}

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------
struct GenerateRequest: Decodable, Sendable {
    var prompt: String
    var negative_prompt: String?
    var seed: UInt32?
    var width: Int?
    var height: Int?
    var steps: Int?
    var guidance_scale: Float?
    var scheduler: String?
    var remove_background: Bool?
    var model: String?  // "sdxl" or "pixel" — selects which pipeline
}

struct HealthResponse: Encodable, Sendable {
    var status: String
    var model: String
}

struct ErrorBody: Encodable, Sendable {
    var error: String
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
let logger = Logger(label: "7up-server")

// Model paths — can be overridden via CLI
let sdxlPath = parseArg("--sdxl-path", default: "../models/sdxl-turbo-coreml/apple/Resources")
let pixelPath = parseArg("--pixel-path", default: "../models/pixnite15-coreml/apple/Resources")
// Legacy single --model-path sets the default
let legacyPath = parseArg("--model-path", default: "")

// Load available models (written once at boot, read-only after)
nonisolated(unsafe) var generators: [String: ImageGenerator] = [:]

func tryLoad(_ name: String, _ path: String) {
    guard FileManager.default.fileExists(atPath: path) else {
        logger.info("Skipping \(name): \(path) not found")
        return
    }
    logger.info("Loading \(name) from: \(path)")
    do {
        generators[name] = try ImageGenerator(modelPath: path)
        logger.info("✓ \(name) loaded")
    } catch {
        logger.error("✗ \(name) failed: \(error)")
    }
}

if !legacyPath.isEmpty {
    tryLoad("default", legacyPath)
} else {
    tryLoad("sdxl", sdxlPath)
    tryLoad("pixel", pixelPath)
}

// Convenience: pick generator by request model field
func generator(for modelName: String?) -> ImageGenerator? {
    if let name = modelName, let gen = generators[name] { return gen }
    // Fallback order: requested → sdxl → pixel → default → first available
    return generators["sdxl"] ?? generators["pixel"] ?? generators["default"] ?? generators.values.first
}

let defaultGenerator = generator(for: nil)
logger.info("Models loaded: \(generators.keys.sorted().joined(separator: ", "))")

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------
let router = Router()

// CORS middleware on all routes
router.middlewares.add(CORSMiddleware())

// GET /health — lists all loaded models
router.get("health") { _, _ -> Response in
    let models = generators.keys.sorted().joined(separator: ",")
    let body = try JSONEncoder().encode(HealthResponse(status: "ok", model: models))
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: .init(data: body))
    )
}

// POST /generate
router.post("generate") { request, _ -> Response in
    // Peek at model field from body (we'll fully decode later)
    let bodyData = try await request.body.collect(upTo: 1_048_576)
    let payload: GenerateRequest
    do {
        payload = try JSONDecoder().decode(GenerateRequest.self, from: bodyData)
    } catch {
        let body = try JSONEncoder().encode(ErrorBody(error: "Invalid JSON body: \(error)"))
        return Response(
            status: .badRequest,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(data: body))
        )
    }

    guard let gen = generator(for: payload.model) else {
        let body = try JSONEncoder().encode(
            ErrorBody(error: "Model not loaded. Start server with --model-path pointing to SDXL Turbo Core ML models.")
        )
        return Response(
            status: .serviceUnavailable,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(data: body))
        )
    }

    let genRequest = ImageGenerator.Request(
        prompt: payload.prompt,
        negativePrompt: payload.negative_prompt ?? "",
        seed: payload.seed ?? UInt32.random(in: 0...UInt32.max),
        width: payload.width ?? 512,
        height: payload.height ?? 512,
        steps: payload.steps ?? 1,
        guidanceScale: payload.guidance_scale ?? 0.0,
        schedulerName: payload.scheduler ?? "dpmSolverMultistep",
        removeBackground: payload.remove_background ?? false
    )

    logger.info("Generating: \"\(genRequest.prompt)\" seed=\(genRequest.seed) \(genRequest.width)x\(genRequest.height)")

    let result: ImageGenerator.Result
    do {
        result = try await gen.generate(genRequest)
    } catch {
        logger.error("Generation failed: \(error)")
        let body = try JSONEncoder().encode(ErrorBody(error: "Generation failed: \(error)"))
        return Response(
            status: .internalServerError,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(data: body))
        )
    }

    logger.info("Generated image in \(String(format: "%.2f", result.elapsedSeconds))s (\(result.pngData.count) bytes)")

    return Response(
        status: .ok,
        headers: [.contentType: "image/png"],
        body: .init(byteBuffer: .init(data: result.pngData))
    )
}

// POST /generate-sprites — generates image then extracts individual objects as separate RGBA PNGs
router.post("generate-sprites") { request, _ -> Response in
    let payload: GenerateRequest
    do {
        let collected = try await request.body.collect(upTo: 1_048_576)
        payload = try JSONDecoder().decode(GenerateRequest.self, from: collected)
    } catch {
        let body = try JSONEncoder().encode(ErrorBody(error: "Invalid JSON: \(error)"))
        return Response(status: .badRequest, headers: [.contentType: "application/json"],
                        body: .init(byteBuffer: .init(data: body)))
    }

    guard let gen = generator(for: payload.model) else {
        let body = try JSONEncoder().encode(ErrorBody(error: "No model loaded."))
        return Response(status: .serviceUnavailable, headers: [.contentType: "application/json"],
                        body: .init(byteBuffer: .init(data: body)))
    }

    let genRequest = ImageGenerator.Request(
        prompt: payload.prompt,
        negativePrompt: payload.negative_prompt ?? "",
        seed: payload.seed ?? UInt32.random(in: 0...UInt32.max),
        width: payload.width ?? 512,
        height: payload.height ?? 512,
        steps: payload.steps ?? 2,
        guidanceScale: payload.guidance_scale ?? 1.0,
        schedulerName: payload.scheduler ?? "eulerAncestral",
        removeBackground: false
    )

    logger.info("Generating sprites: \"\(genRequest.prompt.prefix(50))\" seed=\(genRequest.seed)")

    let result: ImageGenerator.SpritesResult
    do {
        result = try await gen.generateSprites(genRequest)
    } catch {
        logger.error("Sprite generation failed: \(error)")
        let body = try JSONEncoder().encode(ErrorBody(error: "Failed: \(error)"))
        return Response(status: .internalServerError, headers: [.contentType: "application/json"],
                        body: .init(byteBuffer: .init(data: body)))
    }

    // Return JSON with base64-encoded sprites
    struct SpriteResponse: Encodable {
        var count: Int
        var elapsed: Double
        var sprites: [SpriteEntry]
    }
    struct SpriteEntry: Encodable {
        var png: String     // base64
        var x: Int
        var y: Int
        var width: Int
        var height: Int
    }

    let entries = result.sprites.map { sprite in
        SpriteEntry(
            png: sprite.pngData.base64EncodedString(),
            x: sprite.x, y: sprite.y,
            width: sprite.width, height: sprite.height
        )
    }
    let spriteResponse = SpriteResponse(
        count: entries.count,
        elapsed: result.elapsedSeconds,
        sprites: entries
    )

    logger.info("Extracted \(entries.count) sprites in \(String(format: "%.2f", result.elapsedSeconds))s")

    let body = try JSONEncoder().encode(spriteResponse)
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: .init(data: body))
    )
}

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------
let app = Application(router: router, configuration: .init(address: .hostname("127.0.0.1", port: 8090)))

logger.info("Starting 7up-server on http://localhost:8090")
try await app.runService()
