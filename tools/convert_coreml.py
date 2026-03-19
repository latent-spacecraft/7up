#!/usr/bin/env python3
"""
Convert SDXL Turbo from safetensors -> Core ML (.mlpackage) for Apple Silicon (M4).

Produces 4 Core ML submodels:
  - text_encoder/TextEncoder.mlpackage
  - text_encoder_2/TextEncoder2.mlpackage
  - unet/Unet.mlpackage
  - vae_decoder/VaeDecoder.mlpackage

Uses coremltools ct.convert() with fp16 compute precision for ANE/GPU on M4.
"""

import gc
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

CHECKPOINT = Path("/Users/gtaghon/LocalCompute/ComfyUI/models/checkpoints/sd_xl_turbo_1.0_fp16.safetensors")
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "sdxl-turbo-coreml"


def load_pipeline():
    """Load SDXL Turbo pipeline from single safetensors checkpoint."""
    from diffusers import StableDiffusionXLPipeline

    print(f"Loading SDXL Turbo from {CHECKPOINT}...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(CHECKPOINT),
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipe = pipe.to("cpu")
    print("Pipeline loaded successfully.")
    return pipe


def convert_text_encoder(pipe, name, encoder, output_name):
    """Convert a CLIP text encoder to Core ML."""
    import coremltools as ct

    print(f"\n{'='*60}")
    print(f"  Converting {name}")
    print(f"{'='*60}")

    encoder.eval()

    # Wrapper to handle the output format
    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids):
            outputs = self.encoder(input_ids, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            # TE2 (CLIPTextModelWithProjection) has text_embeds; TE1 doesn't
            if hasattr(outputs, 'text_embeds') and outputs.text_embeds is not None:
                return last_hidden, outputs.text_embeds
            return (last_hidden,)

    wrapper = TextEncoderWrapper(encoder).eval()

    # Determine output count
    dummy_input = torch.zeros(1, 77, dtype=torch.int32)
    with torch.no_grad():
        test_out = wrapper(dummy_input)
    has_embeds = len(test_out) == 2

    print(f"  Tracing {name} ({'with text_embeds' if has_embeds else 'hidden state only'})...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input)

    print(f"  Converting to Core ML...")
    out_path = OUTPUT_DIR / name / f"{output_name}.mlpackage"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_types = [ct.TensorType(name="last_hidden_state")]
    if has_embeds:
        output_types.append(ct.TensorType(name="text_embeds"))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 77), dtype=np.int32),
        ],
        outputs=output_types,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    mlmodel.save(str(out_path))
    print(f"  Saved -> {out_path}")
    del traced, wrapper, mlmodel
    gc.collect()


def convert_unet(pipe):
    """Convert UNet to Core ML with fp16 precision."""
    import coremltools as ct

    print(f"\n{'='*60}")
    print(f"  Converting UNet (this is the largest model, be patient)")
    print(f"{'='*60}")

    unet = pipe.unet
    unet.eval()

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            out = self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                return_dict=False,
            )[0]
            return out

    wrapper = UNetWrapper(unet).eval()

    # Dummy inputs
    dummy_sample = torch.randn(1, 4, 64, 64)
    dummy_timestep = torch.tensor([1.0])
    dummy_encoder_hidden_states = torch.randn(1, 77, 2048)
    dummy_text_embeds = torch.randn(1, 1280)
    dummy_time_ids = torch.randn(1, 6)

    print("  Tracing UNet...")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (dummy_sample, dummy_timestep, dummy_encoder_hidden_states,
             dummy_text_embeds, dummy_time_ids),
        )
    print(f"  Tracing took {time.time() - t0:.1f}s")

    print("  Converting UNet to Core ML (fp16)...")
    t0 = time.time()
    out_path = OUTPUT_DIR / "unet" / "Unet.mlpackage"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sample", shape=(1, 4, 64, 64)),
            ct.TensorType(name="timestep", shape=(1,)),
            ct.TensorType(name="encoder_hidden_states", shape=(1, 77, 2048)),
            ct.TensorType(name="text_embeds", shape=(1, 1280)),
            ct.TensorType(name="time_ids", shape=(1, 6)),
        ],
        outputs=[
            ct.TensorType(name="out_sample"),
        ],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"  Conversion took {time.time() - t0:.1f}s")

    mlmodel.save(str(out_path))
    print(f"  Saved -> {out_path}")
    del traced, wrapper, mlmodel, unet
    gc.collect()


def convert_vae_decoder(pipe):
    """Convert VAE decoder to Core ML."""
    import coremltools as ct

    print(f"\n{'='*60}")
    print(f"  Converting VAE Decoder")
    print(f"{'='*60}")

    vae = pipe.vae
    vae.eval()

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent_sample):
            latent_sample = latent_sample / self.vae.config.scaling_factor
            return self.vae.decode(latent_sample, return_dict=False)[0]

    wrapper = VAEDecoderWrapper(vae).eval()

    dummy_latent = torch.randn(1, 4, 64, 64)
    print("  Tracing VAE Decoder...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_latent)

    print("  Converting to Core ML...")
    out_path = OUTPUT_DIR / "vae_decoder" / "VaeDecoder.mlpackage"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="latent_sample", shape=(1, 4, 64, 64)),
        ],
        outputs=[
            ct.TensorType(name="sample"),
        ],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    mlmodel.save(str(out_path))
    print(f"  Saved -> {out_path}")
    del traced, wrapper, mlmodel
    gc.collect()


def print_summary():
    print(f"\n{'='*60}")
    print(f"  Conversion complete! Core ML models:")
    print(f"{'='*60}")
    total = 0
    for f in sorted(OUTPUT_DIR.rglob("*.mlpackage")):
        # mlpackage is a directory; get total size
        size = sum(p.stat().st_size for p in f.rglob("*") if p.is_file())
        size_mb = size / 1024**2
        total += size_mb
        print(f"  {f.relative_to(OUTPUT_DIR)} -- {size_mb:.1f} MB")
    print(f"  TOTAL: {total:.1f} MB")


if __name__ == "__main__":
    print("=" * 60)
    print("  SDXL Turbo -> Core ML (Apple Silicon M4 target)")
    print("=" * 60)

    if not CHECKPOINT.exists():
        print(f"ERROR: Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    pipe = load_pipeline()

    # Convert each submodel
    convert_text_encoder(pipe, "text_encoder", pipe.text_encoder, "TextEncoder")
    convert_text_encoder(pipe, "text_encoder_2", pipe.text_encoder_2, "TextEncoder2")

    # Free text encoders before UNet (save memory)
    del pipe.text_encoder, pipe.text_encoder_2
    gc.collect()

    convert_unet(pipe)

    # Free UNet before VAE
    del pipe.unet
    gc.collect()

    convert_vae_decoder(pipe)

    print_summary()
