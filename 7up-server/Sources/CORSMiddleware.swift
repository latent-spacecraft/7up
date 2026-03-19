import HTTPTypes
import Hummingbird

/// Middleware that adds CORS headers to every response and handles preflight OPTIONS requests.
struct CORSMiddleware<Context: RequestContext>: RouterMiddleware {
    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        // Handle preflight
        if request.method == .options {
            var response = Response(status: .noContent)
            addCORSHeaders(&response)
            response.headers[.allow] = "GET, POST, OPTIONS"
            return response
        }

        var response = try await next(request, context)
        addCORSHeaders(&response)
        return response
    }

    private func addCORSHeaders(_ response: inout Response) {
        response.headers[HTTPField.Name("Access-Control-Allow-Origin")!] = "*"
        response.headers[HTTPField.Name("Access-Control-Allow-Methods")!] = "GET, POST, OPTIONS"
        response.headers[HTTPField.Name("Access-Control-Allow-Headers")!] =
            "Content-Type, Accept"
    }
}
