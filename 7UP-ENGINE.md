# 7UP Engine

**Seven-layer parallax · GenAI-native · Cross-platform**

A sprite-based 2D/pseudo-3D game engine built around a seven-layer compositor with first-class support for procedural generation and on-device generative AI asset synthesis. The name is the architecture: seven layers up from flat sprites.

---

## Status

**v0.2.0** — Proof of concept running as a single-file Canvas2D demo.

Working: 7-layer parallax, seeded procedural generation, dual-world crossfade with smooth seed interpolation, per-layer blend rates, numeric HSL palette lerping, value noise with multi-octave fractal stacking, fixed-timestep game loop, SVG overlay layer.

Next milestone: WebGPU renderer + SDXL Turbo integration for real-time generative asset synthesis.

---

## Architecture

The engine is organized as seven distinct layers (bottom-up), mirroring the seven parallax layers it composites.

```
┌─────────────────────────────────────────────────────────┐
│ L7  APPLICATION                                         │
│     Game logic · ECS · Scene DSL · Lua/WASM scripting   │
├─────────────────────────────────────────────────────────┤
│ L6  GENAI PIPELINE                                      │
│     SDXL Turbo · SVG gen · Noise fields · Seed slerp    │
├─────────────────────────────────────────────────────────┤
│ L5  ASSET CACHE & ATLAS                                 │
│     Texture atlas packing · LRU eviction · Mipmaps      │
├─────────────────────────────────────────────────────────┤
│ L4  COMPOSITOR                                          │
│     7-layer parallax sort · Z-buffer · Blend modes      │
├─────────────────────────────────────────────────────────┤
│ L3  RENDERER                                            │
│     WebGPU → WebGL2 → Canvas2D fallback chain           │
├─────────────────────────────────────────────────────────┤
│ L2  INTERPOLATION & PHYSICS                             │
│     Sub-pixel motion · Easing · Verlet · AABB collision │
├─────────────────────────────────────────────────────────┤
│ L1  PLATFORM ABSTRACTION                                │
│     Input · Audio (WebAudio) · Storage · Timing (RAF)   │
└─────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Latent Parallax

The central insight: each of the seven parallax layers can evolve through latent space at its own rate. As `cameraX` advances, seeds interpolate (slerp) at layer-proportional speeds. Deep space barely changes while the foreground morphs readily. The result is infinite non-repeating worlds with a dreamlike quality — the universe transforms at different rates depending on how far away you're looking.

```
seed(layer, x) = slerp(seedA, seedB, smootherstep(x · rate[layer]))
```

### Dual-World Crossfade

Rather than snapping between seed states, the engine maintains two complete `WorldSnapshot` objects simultaneously. Each snapshot encapsulates a full palette (numeric HSL), noise field set, and structure PRNG. During traversal, all rendering queries blend between the two worlds, with the blend factor passed through Perlin's smootherstep (`6t⁵ − 15t⁴ + 10t³`) for organic feel.

When a seed boundary is crossed, the window slides: A ← B, B ← new(nextSeed).

### Per-Layer Seed Rates

Each layer has a `seedRate` (0.0–1.0) controlling how quickly it responds to the global blend factor:

| Layer | Name | Parallax Speed | Seed Rate | Behavior |
|-------|------|---------------|-----------|----------|
| 0 | Deep Space | 0.02 | 0.15 | Barely changes — cosmic permanence |
| 1 | Nebula | 0.06 | 0.25 | Slow color drift |
| 2 | Far Terrain | 0.12 | 0.40 | Gradual terrain evolution |
| 3 | Mid Terrain | 0.25 | 0.60 | Moderate morphing |
| 4 | Structures | 0.40 | 0.75 | Buildings fade in/out |
| 5 | Near Terrain | 0.65 | 0.90 | Responsive ground changes |
| 6 | Foreground | 0.90 | 1.00 | Immediate, tracks global blend |

### Numeric HSL Color System

All palette colors are stored as `{h, s, l, a}` numeric objects (not CSS strings) to enable smooth interpolation. Hue interpolation uses shortest-arc logic to avoid rainbow artifacts when crossing the 360°/0° boundary.

### BlendedNoise

A lightweight wrapper that lerps the output of two `NoiseField` instances at query time. Terrain height, fog density, nebula shape — everything reads through this blend layer and gets continuous transitions for free.

---

## File Structure (Target)

```
7up/
├── README.md
├── package.json
├── tsconfig.json
├── src/
│   ├── engine/
│   │   ├── core.ts              # Engine init, game loop, config
│   │   ├── ecs.ts               # Entity-Component-System
│   │   ├── scene.ts             # Scene graph / DSL parser
│   │   └── types.ts             # Shared type definitions
│   ├── platform/
│   │   ├── input.ts             # Keyboard, gamepad, touch
│   │   ├── audio.ts             # WebAudio wrapper
│   │   ├── storage.ts           # IndexedDB asset cache
│   │   └── timing.ts            # RAF + fixed timestep
│   ├── renderer/
│   │   ├── webgpu/
│   │   │   ├── device.ts        # Adapter/device negotiation
│   │   │   ├── pipeline.ts      # Render + compute pipelines
│   │   │   ├── shaders/         # WGSL shaders
│   │   │   │   ├── sprite.wgsl
│   │   │   │   ├── composite.wgsl
│   │   │   │   ├── noise.wgsl
│   │   │   │   └── diffusion.wgsl  # SD inference compute
│   │   │   └── textures.ts      # GPU texture management
│   │   ├── webgl2.ts            # Fallback renderer
│   │   ├── canvas2d.ts          # Minimal fallback
│   │   └── svg.ts               # SVG overlay layer
│   ├── compositor/
│   │   ├── parallax.ts          # 7-layer parallax stack
│   │   ├── blend.ts             # Blend modes (normal, add, multiply, screen)
│   │   └── depth.ts             # Z-sorting / pseudo-3D
│   ├── interpolation/
│   │   ├── tween.ts             # Easing library
│   │   ├── smoothstep.ts        # Smoothstep / smootherstep
│   │   └── verlet.ts            # Verlet integration for particles
│   ├── generation/
│   │   ├── rng.ts               # SeededRNG (xorshift128)
│   │   ├── noise.ts             # NoiseField + BlendedNoise
│   │   ├── palette.ts           # Numeric HSL palette gen + lerp
│   │   ├── world.ts             # WorldSnapshot + dual-world manager
│   │   └── ai/
│   │       ├── pipeline.ts      # GenAI asset pipeline orchestrator
│   │       ├── sdxl.ts          # SDXL Turbo ONNX integration
│   │       ├── scheduler.ts     # Asset generation queue + priority
│   │       └── cache.ts         # Generated asset LRU cache
│   └── assets/
│       ├── atlas.ts             # Texture atlas packer
│       ├── spritesheet.ts       # Animation frame management
│       └── loader.ts            # Asset loading / streaming
├── demo/
│   ├── latent-sidescroller/     # The flagship demo
│   │   ├── index.html
│   │   ├── game.ts
│   │   ├── layers/              # Per-layer renderers
│   │   └── ship.ts              # Player entity
│   └── test-harness/            # Layer isolation, perf profiling
├── models/                      # ONNX model files (gitignored, downloaded)
│   └── sdxl-turbo-fp16/
└── tools/
    ├── model-prep/              # Olive/ONNX optimization scripts
    └── atlas-packer/            # Offline atlas generation
```

---

## Next Steps: WebGPU Renderer

### Phase 1 — WebGPU Device Abstraction

Replace Canvas2D with a WebGPU render pipeline while maintaining the Canvas2D path as fallback.

```typescript
// src/renderer/webgpu/device.ts
export async function initWebGPU(): Promise<GPUDevice | null> {
  if (!navigator.gpu) return null;

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter) return null;

  // Request max limits for buffer sizes (critical for large models)
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    },
    requiredFeatures: adapter.features.has('shader-f16')
      ? ['shader-f16']
      : [],
  });

  return device;
}
```

**Key decisions:**
- Request `high-performance` adapter to prefer discrete GPU (avoids the laptop integrated GPU bug documented in Chromium).
- Request `shader-f16` feature when available — required for fp16 SDXL inference and cuts VRAM usage in half.
- Fallback chain: WebGPU → WebGL2 → Canvas2D. Detection at boot, not per-frame.

### Phase 2 — Sprite Render Pipeline

A minimal WGSL sprite batcher that replaces the Canvas2D `fillRect` calls:

```wgsl
// src/renderer/webgpu/shaders/sprite.wgsl
struct Sprite {
  pos: vec2f,       // screen position
  size: vec2f,      // width, height
  uv_offset: vec2f, // atlas UV origin
  uv_size: vec2f,   // atlas UV extent
  color: vec4f,     // tint + alpha
  layer: f32,       // parallax layer (for depth sort)
  _pad: f32,
};

@group(0) @binding(0) var<storage, read> sprites: array<Sprite>;
@group(0) @binding(1) var atlas_texture: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;

struct VertexOutput {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) tint: vec4f,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
  let sprite = sprites[iid];
  // Quad vertices: 0-1-2, 2-1-3
  let corner = vec2f(f32(vid & 1u), f32((vid >> 1u) & 1u));
  var out: VertexOutput;
  out.pos = vec4f(
    (sprite.pos + corner * sprite.size) * vec2f(2.0 / 480.0, -2.0 / 270.0) + vec2f(-1.0, 1.0),
    sprite.layer * 0.001,
    1.0
  );
  out.uv = sprite.uv_offset + corner * sprite.uv_size;
  out.tint = sprite.color;
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let tex = textureSample(atlas_texture, atlas_sampler, in.uv);
  return tex * in.tint;
}
```

This gives us instanced sprite rendering — one draw call for potentially thousands of sprites, sorted by layer. The CPU fills a storage buffer with sprite data, the GPU does the rest.

### Phase 3 — Compute-Shader Noise

Move noise generation to GPU compute shaders. Currently noise is evaluated per-pixel on CPU (expensive for terrain). A compute shader can generate an entire layer's heightmap in one dispatch:

```wgsl
// src/renderer/webgpu/shaders/noise.wgsl
@group(0) @binding(0) var<storage, read_write> heightmap: array<f32>;
@group(0) @binding(1) var<uniform> params: NoiseParams;

struct NoiseParams {
  offset_x: f32,
  scale: f32,
  octaves: u32,
  width: u32,
  perm: array<u32, 256>,
};

@compute @workgroup_size(256)
fn generate_heightmap(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.width) { return; }
  let x = (f32(id.x) + params.offset_x) * params.scale;
  // Multi-octave value noise (same algorithm as CPU, but parallel)
  var val: f32 = 0.0;
  var amp: f32 = 1.0;
  var freq: f32 = params.scale;
  var total_amp: f32 = 0.0;
  for (var o: u32 = 0u; o < params.octaves; o++) {
    val += noise_1d(x * freq, params.perm) * amp;
    total_amp += amp;
    amp *= 0.5;
    freq *= 2.0;
  }
  heightmap[id.x] = val / total_amp;
}
```

---

## Next Steps: SDXL Turbo Integration

This is the big one — generating actual sprite textures on-device via diffusion.

### The Stack

```
User's GPU
  ├── ONNX Runtime Web (ort npm package)
  │     └── WebGPU Execution Provider
  │           └── SDXL Turbo (fp16 quantized ONNX)
  │                 ├── Text Encoder (CLIP)
  │                 ├── UNet (1-step denoiser)
  │                 └── VAE Decoder
  └── 7UP Render Pipeline (also WebGPU)
        └── Compositor reads generated textures directly from GPU
```

### Why SDXL Turbo

SDXL Turbo generates viable 512×512 images in a single denoising step. This is critical for a game engine — we can't wait 20+ steps per asset. On an RTX 4090, ONNX Runtime Web + WebGPU produces an image in under 1 second. On more typical hardware (RTX 3060, M1), expect 2–4 seconds per image at fp16. That's too slow for frame-by-frame generation but perfect for a lookahead cache that pre-generates tiles as the camera approaches.

### Model Preparation

Use Microsoft Olive to optimize the SDXL Turbo ONNX graph for WebGPU:

```bash
# Install Olive
pip install olive-ai

# Export SDXL Turbo to ONNX with fp16
python -m olive run \
  --model stabilityai/sdxl-turbo \
  --output models/sdxl-turbo-fp16 \
  --target webgpu \
  --precision fp16 \
  --optimize

# Expected output: 3 ONNX files
#   text_encoder.onnx   (~250 MB)
#   unet.onnx           (~1.5 GB fp16, ~800 MB int8)
#   vae_decoder.onnx    (~80 MB)
```

For smaller downloads, further quantize with mixed int8/fp16:
- UNet weights: int8 (biggest model, biggest savings)
- UNet activations: fp16 (quantized activations hurt SD quality badly)
- VAE: fp16 (small enough to keep full precision)
- Text encoder: int8 (robust to quantization)

The LeeButtonman approach (6-bit weights, fp16 activations) gets the full SD pipeline under 250 MB compressed — viable for a browser download.

### Integration Architecture

```typescript
// src/generation/ai/pipeline.ts

import * as ort from 'onnxruntime-web';

interface AssetRequest {
  prompt: string;
  seed: number;
  width: number;    // 512 for SDXL Turbo
  height: number;
  priority: number; // Higher = generate sooner
  layerIdx: number; // Which parallax layer wants this
}

interface GeneratedAsset {
  texture: GPUTexture;   // Stays on GPU — no readback!
  seed: number;
  prompt: string;
  timestamp: number;
}

export class GenAIPipeline {
  private sessions: {
    textEncoder: ort.InferenceSession;
    unet: ort.InferenceSession;
    vaeDecoder: ort.InferenceSession;
  } | null = null;

  private queue: AssetRequest[] = [];
  private cache: Map<string, GeneratedAsset> = new Map();
  private generating = false;
  private device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async init(): Promise<boolean> {
    if (!navigator.gpu) return false;

    try {
      // Configure ORT to use WebGPU
      ort.env.wasm.numThreads = 1;

      const opts: ort.InferenceSession.SessionOptions = {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      };

      // Load models (cached in IndexedDB after first download)
      this.sessions = {
        textEncoder: await ort.InferenceSession.create('/models/text_encoder.onnx', opts),
        unet: await ort.InferenceSession.create('/models/unet.onnx', opts),
        vaeDecoder: await ort.InferenceSession.create('/models/vae_decoder.onnx', opts),
      };
      return true;
    } catch (e) {
      console.warn('GenAI pipeline unavailable:', e);
      return false;
    }
  }

  /**
   * Request an asset. Non-blocking — asset is generated async
   * and inserted into the texture cache when ready.
   */
  request(req: AssetRequest): void {
    const key = `${req.prompt}:${req.seed}`;
    if (this.cache.has(key)) return; // Already have it
    this.queue.push(req);
    this.queue.sort((a, b) => b.priority - a.priority);
    this.processQueue();
  }

  private async processQueue(): Promise<void> {
    if (this.generating || !this.sessions || this.queue.length === 0) return;
    this.generating = true;

    const req = this.queue.shift()!;
    try {
      const texture = await this.generate(req);
      const key = `${req.prompt}:${req.seed}`;
      this.cache.set(key, {
        texture,
        seed: req.seed,
        prompt: req.prompt,
        timestamp: performance.now(),
      });
    } catch (e) {
      console.error('Generation failed:', e);
    }

    this.generating = false;
    // Continue processing
    if (this.queue.length > 0) {
      requestIdleCallback(() => this.processQueue());
    }
  }

  private async generate(req: AssetRequest): Promise<GPUTexture> {
    const { textEncoder, unet, vaeDecoder } = this.sessions!;

    // 1. Encode prompt → CLIP embeddings
    //    (tokenization handled by a lightweight JS tokenizer)
    const tokens = tokenize(req.prompt); // BPE tokenizer
    const textOutput = await textEncoder.run({
      input_ids: new ort.Tensor('int32', tokens, [1, 77]),
    });

    // 2. Single-step UNet denoise
    //    SDXL Turbo needs only 1 step with guidance_scale=0
    const latentNoise = generateLatentNoise(req.seed, 1, 4, 64, 64);
    const unetOutput = await unet.run({
      sample: latentNoise,
      timestep: new ort.Tensor('int64', [1n], [1]),
      encoder_hidden_states: textOutput.last_hidden_state,
    });

    // 3. VAE decode → pixel space
    const decoded = await vaeDecoder.run({
      latent_sample: unetOutput.out_sample,
    });

    // 4. Write directly to GPU texture (IO binding — no CPU readback!)
    //    decoded.sample is already a GPU tensor if IO binding is active
    const texture = this.device.createTexture({
      size: [req.width, req.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING |
             GPUTextureUsage.COPY_DST |
             GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Copy decoded output to texture
    // (exact mechanism depends on ORT IO binding setup)
    this.copyToTexture(decoded, texture);

    return texture;
  }

  /**
   * Get a cached texture for a given prompt+seed, or null.
   */
  getTexture(prompt: string, seed: number): GPUTexture | null {
    return this.cache.get(`${prompt}:${seed}`)?.texture ?? null;
  }

  /**
   * Evict oldest entries when cache exceeds size limit.
   */
  evict(maxEntries: number = 64): void {
    if (this.cache.size <= maxEntries) return;
    const sorted = [...this.cache.entries()]
      .sort((a, b) => a[1].timestamp - b[1].timestamp);
    while (this.cache.size > maxEntries) {
      const [key, asset] = sorted.shift()!;
      asset.texture.destroy();
      this.cache.delete(key);
    }
  }
}
```

### Lookahead Strategy

The engine doesn't generate textures on-demand per frame. Instead, it uses the camera velocity to predict which world regions are approaching and pre-generates assets:

```
Camera at X=5000, velocity=1.2 px/frame
├── Currently visible: X 4880–5360
├── Lookahead zone:    X 5360–6560 (1 second ahead)
│   ├── Check cache for tiles in this zone
│   └── Queue generation for any misses, priority = 1/distance
└── Dispose zone:      X < 3880 (1 second behind)
    └── Evict textures from LRU cache
```

Prompts are composed procedurally based on position + layer:
```
layer 2 (far terrain):  "alien mountain range, purple sky, pixel art, 16-bit"
layer 4 (structures):   "crystalline monolith, bioluminescent, sci-fi, sprite"
layer 6 (foreground):   "rocky ground with glowing moss, side view, tileable"
```

As `cameraX` crosses seed boundaries, the prompt suffixes shift to reflect the new latent region — "frozen tundra" → "volcanic plains" → "bioluminescent deep sea" — while the seed ensures deterministic noise for each tile.

### Graceful Degradation

Not every device can run SDXL. The engine provides a tiered experience:

| Tier | GPU | Capability | Asset Source |
|------|-----|-----------|-------------|
| S | RTX 4080+, M2 Pro+ | Full SDXL Turbo @ fp16 | Real-time gen (< 1s/image) |
| A | RTX 3060, M1 | SDXL Turbo @ int8 | Lookahead gen (2–4s/image) |
| B | Integrated GPU | No diffusion | Procedural noise + SVG |
| C | No WebGPU | Canvas2D only | Procedural noise + SVG |

Tier detection happens at boot:
```typescript
async function detectTier(device: GPUDevice): Promise<'S' | 'A' | 'B' | 'C'> {
  if (!device) return 'C';

  const maxBuffer = device.limits.maxBufferSize;
  const hasF16 = device.features.has('shader-f16');

  // UNet fp16 needs ~1.5 GB buffer space
  if (hasF16 && maxBuffer >= 2_147_483_648) return 'S';
  if (maxBuffer >= 1_073_741_824) return 'A';
  return 'B';
}
```

---

## Seed Interpolation (Current Implementation)

The v0.2 demo implements the dual-world crossfade system described above. Key parameters:

```javascript
const CFG = {
  SEED_PERIOD: 2000,        // pixels between seed waypoints
  TRANSITION_ZONE: 0.7,     // (reserved for future partial-window blending)
};
```

The blend curve uses Perlin's improved smootherstep:

```
smootherstep(t) = t³(t(6t − 15) + 10)
```

Per-layer blend is computed as:

```
raw = globalT × seedRate + (1 − seedRate) × globalT³
layerBlend = smootherstep(clamp(raw, 0, 1))
```

This gives deep layers (low seedRate) a delayed, cubic-onset response — they "resist" the transition and then catch up — while foreground layers track the global blend closely.

---

## Rendering Model

### Current (v0.2): Canvas2D

Every layer is drawn procedurally per-frame using `ctx.fillRect`, `ctx.beginPath/fill`, and basic gradient fills. Stars, terrain, structures, fog, and particles are all immediate-mode rendered.

### Target (v1.0): WebGPU Hybrid

```
Frame N:
 ├── Compute pass: generate heightmaps (WGSL noise shaders)
 ├── Compute pass: run queued SDXL inference step (if any)
 ├── Render pass:
 │    ├── Clear framebuffer
 │    ├── For each layer 0..6:
 │    │    ├── Bind layer's texture atlas (procedural + AI-gen tiles)
 │    │    ├── Set parallax offset uniform
 │    │    ├── Draw instanced sprite batch
 │    │    └── Apply blend mode
 │    ├── Draw player entity
 │    └── Draw HUD (SVG overlay composited last)
 └── Present
```

The critical optimization: SDXL output stays on the GPU as a texture. No CPU readback. The ORT WebGPU execution provider writes to a GPU buffer, we create a texture view from that buffer, and the compositor binds it directly in the render pass. This is the IO binding pattern documented in the ORT WebGPU guide.

---

## Dependencies

### Runtime
- `onnxruntime-web` — ONNX inference with WebGPU EP
- Lightweight BPE tokenizer (CLIP-compatible, ~50 KB)
- No framework dependency — vanilla TypeScript

### Build
- TypeScript 5.x
- Vite (for dev server + WASM/ONNX asset serving)
- `@aspect-build/rules_js` or similar for monorepo (optional)

### Model Preparation (Offline)
- Python 3.10+
- `olive-ai` — Microsoft's ONNX optimization toolkit
- `optimum` — Hugging Face ONNX export
- `onnxruntime` — validation

---

## Browser Support

| Browser | WebGPU | GenAI (SDXL) | Fallback |
|---------|--------|-------------|----------|
| Chrome 121+ | ✅ | ✅ (fp16) | — |
| Edge 121+ | ✅ | ✅ (fp16) | — |
| Firefox 141+ | ✅ (Windows) | ⚠️ (stability issues) | WebGL2 |
| Safari 26+ | ✅ (macOS/iOS) | ⚠️ (buffer limits) | WebGL2 |
| Older | ❌ | ❌ | Canvas2D |

The engine auto-detects capability at boot and selects the best available renderer + asset strategy.

---

## Dev Setup (Planned)

```bash
git clone https://github.com/latentspacecraft/7up.git
cd 7up
npm install
npm run dev          # Vite dev server at localhost:5173
npm run build        # Production build
npm run demo         # Launch latent sidescroller demo
npm run model:prep   # Download + optimize SDXL Turbo ONNX models
```

---

## Design Philosophy

**"Do everything with nothing."** The engine should produce stunning results on modest hardware. Procedural generation and noise-based rendering (Tier B/C) should still look beautiful — AI generation (Tier S/A) should make it extraordinary.

**GenAI is a material, not a crutch.** AI-generated textures are one input to the compositor, blended with procedural noise, SVG vector effects, and hand-authored sprites. The engine is useful and complete without any AI model loaded.

**Seven layers deep, not seven frameworks wide.** Vanilla TypeScript, minimal dependencies, no React/Vue/Angular. The engine is a library, not an application.

---

## Research Threads

- **Latent space navigation as game mechanic** — seed interpolation already creates "biomes" naturally. Can we give the player a latent space compass?
- **Compartmental automata terrain** — replace value noise with CA-generated heightmaps for more organic, emergent terrain features.
- **LoRA hot-swapping** — load different LoRA adapters per layer for style variation (pixel art LoRA for foreground, painterly LoRA for distant mountains).
- **Sprite decomposition** — generate a full scene with SDXL, then use SAM (Segment Anything) to decompose it into individual sprite layers with transparency.
- **WGSL noise as learned function** — train a tiny MLP to approximate multi-octave noise and compile it to WGSL. Fewer ALU ops, same visual quality.

---

## License

TBD — likely MIT for the engine, with model weights under their respective licenses (SDXL Turbo: Stability AI Community License).
