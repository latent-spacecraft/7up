#!/usr/bin/env python3
"""
Export SDXL Turbo from safetensors → ONNX (WebGPU-compatible).

Produces 3-4 ONNX submodels:
  - text_encoder/model.onnx
  - text_encoder_2/model.onnx
  - unet/model.onnx
  - vae_decoder/model.onnx

Exported fp32, opset 17 — avoids CUDA-specific ops for WebGPU EP compatibility.
"""

import os
import sys
import shutil
from pathlib import Path

import torch

CHECKPOINT = Path("/Users/gtaghon/LocalCompute/ComfyUI/models/checkpoints/sd_xl_turbo_1.0_fp16.safetensors")
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "sdxl-turbo-onnx"


def export_with_optimum():
    """Use optimum to export the HF pipeline to ONNX."""
    from optimum.exporters.onnx import main_export
    from diffusers import StableDiffusionXLPipeline

    print(f"[1/4] Loading SDXL Turbo from {CHECKPOINT}...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(CHECKPOINT),
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    staging_dir = OUTPUT_DIR.parent / "sdxl-turbo-staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/4] Saving HF pipeline layout to {staging_dir}...")
    pipe.save_pretrained(str(staging_dir))
    del pipe

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] Exporting to ONNX → {OUTPUT_DIR}...")
    main_export(
        model_name_or_path=str(staging_dir),
        output=str(OUTPUT_DIR),
        task="stable-diffusion-xl",
        opset=17,
        fp16=False,
        device="cpu",
    )

    print(f"[4/4] Cleaning up staging directory...")
    shutil.rmtree(staging_dir, ignore_errors=True)

    print_summary()


def export_manual():
    """Manual torch.onnx.export — fallback with more control."""
    from diffusers import StableDiffusionXLPipeline

    print(f"[1/5] Loading SDXL Turbo from {CHECKPOINT}...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(CHECKPOINT),
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipe = pipe.to("cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Text Encoder 1 ---
    print("[2/5] Exporting text_encoder/model.onnx...")
    te = pipe.text_encoder
    te.eval()
    te_path = OUTPUT_DIR / "text_encoder" / "model.onnx"
    te_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        te,
        (torch.zeros(1, 77, dtype=torch.long),),
        str(te_path),
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={"input_ids": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"   → {te_path.stat().st_size / 1024**2:.1f} MB")
    del te

    # --- Text Encoder 2 ---
    print("[3/5] Exporting text_encoder_2/model.onnx...")
    te2 = pipe.text_encoder_2
    te2.eval()
    te2_path = OUTPUT_DIR / "text_encoder_2" / "model.onnx"
    te2_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        te2,
        (torch.zeros(1, 77, dtype=torch.long),),
        str(te2_path),
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={"input_ids": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"   → {te2_path.stat().st_size / 1024**2:.1f} MB")
    del te2

    # --- UNet ---
    print("[4/5] Exporting unet/model.onnx (largest model)...")
    unet = pipe.unet
    unet.eval()

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            return self.unet(
                sample, timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                return_dict=False,
            )[0]

    unet_path = OUTPUT_DIR / "unet" / "model.onnx"
    unet_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        UNetWrapper(unet).eval(),
        (
            torch.randn(1, 4, 64, 64),
            torch.tensor([1.0]),
            torch.randn(1, 77, 2048),
            torch.randn(1, 1280),
            torch.randn(1, 6),
        ),
        str(unet_path),
        input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        output_names=["out_sample"],
        dynamic_axes={
            "sample": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "text_embeds": {0: "batch"},
            "time_ids": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"   → {unet_path.stat().st_size / 1024**2:.1f} MB")
    del unet

    # --- VAE Decoder ---
    print("[5/5] Exporting vae_decoder/model.onnx...")
    vae = pipe.vae
    vae.eval()

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent_sample):
            latent_sample = latent_sample / self.vae.config.scaling_factor
            return self.vae.decode(latent_sample, return_dict=False)[0]

    vae_path = OUTPUT_DIR / "vae_decoder" / "model.onnx"
    vae_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        VAEDecoderWrapper(vae).eval(),
        (torch.randn(1, 4, 64, 64),),
        str(vae_path),
        input_names=["latent_sample"],
        output_names=["sample"],
        dynamic_axes={"latent_sample": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"   → {vae_path.stat().st_size / 1024**2:.1f} MB")

    print_summary()


def print_summary():
    print("\n✅ Export complete! ONNX models:")
    total = 0
    for f in sorted(OUTPUT_DIR.rglob("*.onnx")):
        size_mb = f.stat().st_size / 1024**2
        total += size_mb
        print(f"   {f.relative_to(OUTPUT_DIR)} — {size_mb:.1f} MB")
    print(f"   TOTAL: {total:.1f} MB")


if __name__ == "__main__":
    print("=" * 60)
    print("  SDXL Turbo → ONNX Export (WebGPU target)")
    print("=" * 60)

    if not CHECKPOINT.exists():
        print(f"❌ Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    try:
        export_with_optimum()
    except Exception as e:
        print(f"\n⚠️  optimum export failed: {e}")
        print("   Falling back to manual torch.onnx.export...\n")
        export_manual()
