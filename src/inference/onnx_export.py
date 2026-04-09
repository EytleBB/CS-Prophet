"""Export a trained BombSiteTransformer checkpoint to ONNX format.

Usage:
    python -m src.inference.onnx_export
    python -m src.inference.onnx_export --checkpoint checkpoints/best.pt --output model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.model.transformer import BombSiteTransformer


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    seq_len: int = 720,
    opset: int = 17,
) -> Path:
    """Load a checkpoint and export the model to ONNX.

    Args:
        checkpoint_path: Path to a .pt checkpoint saved by train.py.
        output_path: Destination .onnx file.
        seq_len: Fixed sequence length (must match training config).
        opset: ONNX opset version.

    Returns:
        Resolved Path to the written .onnx file.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = BombSiteTransformer(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Dummy inputs — batch=1, fixed seq_len, 275 features
    dummy_x = torch.randn(1, seq_len, 275)
    dummy_mask = torch.zeros(1, seq_len, dtype=torch.bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_x, dummy_mask),
        str(output_path),
        opset_version=opset,
        input_names=["features", "src_key_padding_mask"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch"},
            "src_key_padding_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    print(f"Exported ONNX model → {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    _verify(output_path, dummy_x.numpy(), dummy_mask.numpy(), model)
    return output_path.resolve()


def _verify(
    onnx_path: Path,
    sample_x: np.ndarray,
    sample_mask: np.ndarray,
    torch_model: BombSiteTransformer,
) -> None:
    """Run onnxruntime inference and compare against PyTorch output."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    ort_out = sess.run(None, {
        "features": sample_x,
        "src_key_padding_mask": sample_mask,
    })[0]

    with torch.no_grad():
        pt_out = torch_model(
            torch.tensor(sample_x),
            src_key_padding_mask=torch.tensor(sample_mask),
        ).numpy()

    if np.allclose(pt_out, ort_out, atol=1e-5):
        print("  Verification PASSED — ONNX output matches PyTorch.")
    else:
        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        print(f"  Verification WARNING — max diff {max_diff:.6f} (atol=1e-5)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export BombSiteTransformer to ONNX")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt",
                    help="Path to .pt checkpoint")
    ap.add_argument("--output", default="model.onnx",
                    help="Output .onnx file path")
    ap.add_argument("--seq-len", type=int, default=720,
                    help="Fixed sequence length (default 720)")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version (default 17)")
    args = ap.parse_args()

    export_onnx(args.checkpoint, args.output, args.seq_len, args.opset)


if __name__ == "__main__":
    main()
