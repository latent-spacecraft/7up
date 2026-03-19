#!/usr/bin/env python3
"""
Convert ONNX models from fp32 → fp16 (handles >2GB models).

Uses onnxconverter_common with shape inference disabled to avoid
the protobuf 2GB serialization limit. Properly rewrites the graph
types, not just the weight tensors.
"""

from pathlib import Path
import onnx
from onnxconverter_common import float16

MODEL_DIR = Path(__file__).parent.parent / "models" / "sdxl-turbo-onnx"


def convert_model(subdir):
    model_path = MODEL_DIR / subdir / "model.onnx"
    out_path = MODEL_DIR / subdir / "model_fp16.onnx"
    out_data = "model_fp16.onnx_data"

    print(f"\n{'='*60}")
    print(f"  Converting {subdir} to fp16")
    print(f"{'='*60}")

    print(f"  Loading model...", end=" ", flush=True)
    model = onnx.load(str(model_path), load_external_data=True)
    print("done.")

    print(f"  Converting graph + weights to fp16...", end=" ", flush=True)
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,          # Keep model inputs/outputs as fp32
        disable_shape_infer=True,    # Skip shape inference (avoids protobuf 2GB limit)
        op_block_list=[              # Keep normalization ops in fp32 for stability
            "LayerNormalization",
            "GroupNorm",
            "BatchNormalization",
            "InstanceNormalization",
        ],
    )
    print("done.")

    # Remove old fp16 files if they exist
    for old in (out_path, out_path.parent / out_data):
        if old.exists():
            old.unlink()

    print(f"  Saving → {out_path.name}...", end=" ", flush=True)
    onnx.save(
        model_fp16,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out_data,
        size_threshold=1024,
    )
    print("done.")

    # Report
    data_file = out_path.parent / out_data
    orig_data = model_path.parent / "model.onnx_data"
    orig_size = orig_data.stat().st_size if orig_data.exists() else model_path.stat().st_size
    new_size = data_file.stat().st_size if data_file.exists() else out_path.stat().st_size
    print(f"  {orig_size/1024**3:.2f} GB → {new_size/1024**3:.2f} GB ({new_size/orig_size*100:.0f}%)")

    del model, model_fp16


if __name__ == "__main__":
    for target in ["text_encoder_2", "unet"]:
        convert_model(target)
    print("\n✅ Conversion complete.")
