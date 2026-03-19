import Foundation
import Hummingbird
import Logging

// ---------------------------------------------------------------------------
// CLI argument parsing (lightweight, no ArgumentParser dependency)
// ---------------------------------------------------------------------------
func parseModelPath() -> String {
    let args = CommandLine.arguments
    if let idx = args.firstIndex(of: "--model-path"), idx + 1 < args.count {
        return args[idx + 1]
    }
    // Default relative to where the server sits in the repo layout
    return "../models/sdxl-turbo-coreml/resources"
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
let modelPath = parseModelPath()

logger.info("Loading SDXL Turbo pipeline from: \(modelPath)")

// The generator is optional: nil while the model hasn't loaded (or failed).
// We load eagerly on the main path, but wrap in do/catch so the server still
// boots and can return 503 on /generate if the model isn't available.
let generator: ImageGenerator?
do {
    generator = try ImageGenerator(modelPath: modelPath)
    logger.info("Pipeline loaded successfully")
} catch {
    logger.error("Failed to load pipeline: \(error)")
    logger.warning("Server will start but /generate will return 503")
    generator = nil
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------
let router = Router()

// CORS middleware on all routes
router.middlewares.add(CORSMiddleware())

// GET /health
router.get("health") { _, _ -> Response in
    let body = try JSONEncoder().encode(HealthResponse(status: "ok", model: "sdxl-turbo"))
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: .init(data: body))
    )
}

// POST /generate
router.post("generate") { request, _ -> Response in
    guard let gen = generator else {
        let body = try JSONEncoder().encode(
            ErrorBody(error: "Model not loaded. Start server with --model-path pointing to SDXL Turbo Core ML models.")
        )
        return Response(
            status: .serviceUnavailable,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(data: body))
        )
    }

    // Decode request body
    let payload: GenerateRequest
    do {
        let collected = try await request.body.collect(upTo: 1_048_576)  // 1 MB max
        payload = try JSONDecoder().decode(GenerateRequest.self, from: collected)
    } catch {
        let body = try JSONEncoder().encode(ErrorBody(error: "Invalid JSON body: \(error)"))
        return Response(
            status: .badRequest,
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
    guard let gen = generator else {
        let body = try JSONEncoder().encode(ErrorBody(error: "Model not loaded."))
        return Response(status: .serviceUnavailable, headers: [.contentType: "application/json"],
                        body: .init(byteBuffer: .init(data: body)))
    }

    let payload: GenerateRequest
    do {
        let collected = try await request.body.collect(upTo: 1_048_576)
        payload = try JSONDecoder().decode(GenerateRequest.self, from: collected)
    } catch {
        let body = try JSONEncoder().encode(ErrorBody(error: "Invalid JSON: \(error)"))
        return Response(status: .badRequest, headers: [.contentType: "application/json"],
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
