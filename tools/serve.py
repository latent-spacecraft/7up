#!/usr/bin/env python3
"""
Dev server for SDXL Turbo WebGPU inference.
Serves the project directory with required COOP/COEP headers
for SharedArrayBuffer (needed by ORT WASM multithreading).
Handles Range requests for large model files.
"""

import http.server
import os
import sys

PORT = 8080
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Required for SharedArrayBuffer (ORT WASM threads)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        # Cache model files aggressively
        if self.path.endswith((".onnx", ".onnx_data")):
            self.send_header("Cache-Control", "public, max-age=86400")
        super().end_headers()

    def do_GET(self):
        # Handle Range requests for large files
        path = self.translate_path(self.path)
        if os.path.isfile(path) and "Range" in self.headers:
            self.send_range_response(path)
            return
        super().do_GET()

    def send_range_response(self, path):
        file_size = os.path.getsize(path)
        range_header = self.headers["Range"]
        # Parse "bytes=start-end"
        range_spec = range_header.replace("bytes=", "")
        parts = range_spec.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(remaining, 1024 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f"Serving {DIRECTORY} on http://localhost:{port}")
    print(f"Open http://localhost:{port}/demo/sdxl-turbo.html")
    print("COOP/COEP headers enabled for SharedArrayBuffer")
    httpd = http.server.HTTPServer(("", port), CORSHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()
