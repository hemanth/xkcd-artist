#!/usr/bin/env python3
"""
Export XkcdArtistForGeneration to ONNX opset 17 (FP32) and quantize to Dynamic QUInt8
for Browser WASM SIMD / WebGPU inference.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from modeling_xkcd_artist import XkcdArtistForGeneration

ARTIFACT_DIR = ROOT / "artifacts" / "xkcd-artist"
ONNX_DIR = ARTIFACT_DIR / "onnx"


class XkcdArtistOnnxWrapper(nn.Module):
    """Pure tensor-in, tensor-tuple-out wrapper for ONNX export."""

    def __init__(self, model: XkcdArtistForGeneration) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        token_ids: torch.Tensor,
        cast_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        out = self.model(token_ids=token_ids, cast_ids=cast_ids)
        return (
            out["layout_logits"],
            out["prop_logits"],
            out["poses"],
            out["bubbles"],
            out["style_vec"],
            out["ink_patch"],
            out["invalid_logit"],
        )


def main() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    model = XkcdArtistForGeneration.from_pretrained(ARTIFACT_DIR).cpu().eval()
    wrapper = XkcdArtistOnnxWrapper(model).eval()

    dummy_tokens = torch.zeros((1, model.config.max_seq_len), dtype=torch.long)
    dummy_cast = torch.tensor([[0, 1]], dtype=torch.long)

    fp32_path = ONNX_DIR / "model.onnx"
    int8_path = ONNX_DIR / "model_quantized.onnx"

    torch.onnx.export(
        wrapper,
        (dummy_tokens, dummy_cast),
        str(fp32_path),
        input_names=["token_ids", "cast_ids"],
        output_names=[
            "layout_logits",
            "prop_logits",
            "poses",
            "bubbles",
            "style_vec",
            "ink_patch",
            "invalid_logit",
        ],
        dynamic_axes={
            "token_ids": {0: "batch"},
            "cast_ids": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(fp32_path), load_external_data=True)
    onnx.save_model(onnx_model, str(fp32_path), save_as_external_data=False)

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,
    )

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(int8_path, docs_dir / "model_quantized.onnx")

    fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
    int8_mb = int8_path.stat().st_size / (1024 * 1024)
    print(f"[export_onnx] FP32: {fp32_mb:.2f} MB -> Int8 Quantized: {int8_mb:.2f} MB ({fp32_mb/max(0.01, int8_mb):.2f}x smaller)")


if __name__ == "__main__":
    main()
