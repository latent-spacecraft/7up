# 7UP Engine

**Seven-layer parallax · GenAI-native · Apple Silicon**

A sprite-based 2D game engine built around a seven-layer compositor with real-time AI texture generation via SDXL Turbo on Core ML. The name is the architecture: seven layers up from flat sprites.

## Demo

Open `7up-demo-2.html` in a browser. The engine runs standalone with procedural generation. When the SDXL server is running, AI-generated textures fill the terrain and enemy starfighters populate the world.

**Share levels by seed:** `7up-demo-2.html#seed=42`

## Architecture

```
Browser (Canvas2D)                Swift Server (localhost:8090)
┌──────────────────────┐          ┌──────────────────────────────┐
│ 7UP Engine           │   HTTP   │ SDXL Turbo                   │
│ ├─ 7-layer parallax  │◄────────►│ ├─ Core ML (int8, 3.4GB)     │
│ ├─ Procedural gen    │ /generate│ ├─ Metal / ANE compute       │
│ ├─ AI terrain fill   │  → PNG   │ ├─ Euler Ancestral scheduler │
│ ├─ Lissajous enemies │          │ ├─ Vision instance segment.  │
│ ├─ AI sprite extract │ /sprites │ │   └─ Per-object alpha mask  │
│ └─ Seed-based levels │  → JSON  │ └─ Hummingbird HTTP          │
└──────────────────────┘          └──────────────────────────────┘
Engine works without server (graceful degradation)
```

### Layers (bottom → top)

| # | Name | Parallax | AI Texture | Role |
|---|------|----------|-----------|------|
| 0 | Deep Space | 0.02 | Starfield photo | Screen-blended sky background |
| 1 | Nebula | 0.06 | Nebula clouds | Screen-blended atmospheric |
| 2 | Far Terrain | 0.12 | Rock cliff fill | Clipped to terrain silhouette |
| 3 | Mid Terrain | 0.25 | Crystal formations | Clipped to terrain silhouette |
| 4 | Structures | 0.40 | Procedural | Buildings, spires |
| 5 | Near Terrain | 0.65 | Mossy rock fill | Clipped to terrain silhouette |
| 6 | Foreground | 0.90 | Ground texture | Clipped to terrain silhouette |

### Key Concepts

- **Latent Parallax**: Each layer evolves through latent space at its own rate. Deep space barely changes while foreground morphs readily.
- **Dual-World Crossfade**: Two complete seed states maintained simultaneously with quintic smoothstep interpolation.
- **Seed-as-Level**: A single seed number deterministically generates all 7 AI textures, procedural terrain, and entity placement. Share the seed, share the world.

### Entities

Enemies follow **Lissajous curves** — parametric paths that produce circles, figure-8s, trefoils, and complex knots from simple sine functions:

```
x(t) = A · sin(a · t + δ)
y(t) = B · sin(b · t)
```

Each entity's curve parameters are derived from its seed. Sprite size drives a **mass/momentum simulation**: large ships carve wide lazy arcs with long engine trails, while small interceptors zip through tight aggressive patterns.

| Ship Size | Amplitude | Speed | Trail | Behavior |
|-----------|-----------|-------|-------|----------|
| 12px (interceptor) | 15-30px | 1.8-3.0x | 6 frames | Tight, aggressive |
| 24px (fighter) | 33-48px | 1.0-2.2x | 12 frames | Balanced |
| 36px (capital) | 50-65px | 0.3-1.5x | 18 frames | Wide, sweeping |

### AI Sprite Pipeline

Enemy sprites are generated and isolated automatically:

1. **SDXL Turbo** generates a 512×512 image of multiple starfighters
2. **Apple Vision** (`VNGenerateForegroundInstanceMaskRequest`) isolates each individual ship — the same tech behind "Copy Subject" in Messages
3. Each instance gets its own **alpha mask**, **bounding box crop**, and **RGBA PNG**
4. Sprites are assigned to Lissajous entities and **rotated to match heading**

One generation (~4s) produces 8-12 unique sprites with clean transparent edges.

## Quick Start

### 1. Engine Only (no AI)

```bash
# Any static file server works
python3 -m http.server 8080
open http://localhost:8080/7up-demo-2.html
```

### 2. With AI Textures + Sprites

```bash
# Download Core ML models from HuggingFace
mkdir -p models/sdxl-turbo-coreml/apple
cd models/sdxl-turbo-coreml/apple
git clone https://huggingface.co/latentspacecraft/sdxl-turbo-coreml Resources

# Build and run the Swift server
cd ../../../7up-server
swift build
.build/debug/7up-server --model-path ../models/sdxl-turbo-coreml/apple/Resources

# In another terminal, serve the HTML
python3 tools/serve.py 8080
open http://localhost:8080/7up-demo-2.html
```

### 3. SDXL Turbo Studio

Visual parameter tuning UI:

```bash
open http://localhost:8080/demo/sdxl-turbo.html
```

## Server API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status |
| `/generate` | POST | Generate image (returns PNG, optionally RGBA with `remove_background`) |
| `/generate-sprites` | POST | Generate + Vision instance segmentation → JSON array of cropped RGBA sprites |

### POST /generate-sprites

```json
{
  "prompt": "pixel art enemy starfighters, solid black background",
  "seed": 42,
  "steps": 2,
  "guidance_scale": 1.0,
  "scheduler": "eulerAncestral"
}
```

Returns:
```json
{
  "count": 10,
  "elapsed": 4.66,
  "sprites": [
    { "png": "<base64>", "x": 0, "y": 104, "width": 117, "height": 189 },
    ...
  ]
}
```

## SDXL Turbo Settings

| Parameter | Value |
|-----------|-------|
| Steps | 2 |
| CFG Scale | 1.0 |
| Scheduler | Euler Ancestral (custom, added to ml-stable-diffusion fork) |
| Resolution | 512×512 |
| Quantization | int8 palettization |
| Model size | 3.4 GB |
| Inference | ~3.7s on M4 |

## Project Structure

```
7up/
├── 7up-demo-2.html              # Main engine demo
├── 7UP-ENGINE.md                 # Architecture deep-dive
├── README.md
├── .gitignore
├── demo/
│   └── sdxl-turbo.html           # SDXL Turbo Studio (parameter tuning UI)
├── tools/
│   ├── serve.py                  # Dev server (COOP/COEP headers)
│   ├── export_onnx.py            # SDXL → ONNX (reference)
│   ├── convert_coreml.py         # SDXL → Core ML (reference)
│   └── convert_fp16.py           # ONNX fp16 conversion (reference)
├── 7up-server/                   # Swift SDXL Turbo server
│   ├── Package.swift
│   └── Sources/
│       ├── main.swift            # Hummingbird HTTP routes
│       ├── ImageGenerator.swift  # Core ML pipeline + Vision sprite extraction
│       └── CORSMiddleware.swift
└── ml-stable-diffusion/          # Fork of Apple's library (submodule)
    └── swift/StableDiffusion/pipeline/
        └── EulerAncestralDiscreteScheduler.swift  # Added
```

## Models

Core ML models (int8 quantized) hosted on HuggingFace:

**[latentspacecraft/sdxl-turbo-coreml](https://huggingface.co/latentspacecraft/sdxl-turbo-coreml)**

| Model | Size |
|-------|------|
| TextEncoder.mlmodelc | 154 MB |
| TextEncoder2.mlmodelc | 724 MB |
| Unet.mlmodelc | 2.4 GB |
| VAEDecoder.mlmodelc | 189 MB |
| **Total** | **3.4 GB** |

## Requirements

- **Engine**: Any modern browser
- **AI Server**: macOS 14+, Apple Silicon (M1+), Swift 6.0, ~8GB RAM
- **Optimal**: M4 with 32GB unified memory (~3.7s/image)
- **iOS Target**: iPhone 15 Pro+ (A17 Pro, 8GB RAM)

## Controls

| Key | Action |
|-----|--------|
| ← → | Move |
| ↑ ↓ | Altitude |
| Space | Boost |
| 1-7 | Solo layer |

## License

Engine: MIT. Model weights: [Stability AI Community License](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md).
