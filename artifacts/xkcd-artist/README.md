---
language:
- en
license: mit
library_name: transformers
tags:
- xkcd
- comics
- diffusion-lora
- onnx
- stick-figure
- scene-graph
- pytorch
datasets:
- olivierdehaene/xkcd
pipeline_tag: text-to-image
---

# xkcd-artist (`gnumanth/xkcd-artist`)

**Hybrid Fine-Tuned Diffusion UNet LoRA (`sd_xkcd_lora`) + 5-Head ExplainXKCD Scene, Pose & Speech-Bubble Planner (`XkcdArtistForGeneration`)**, trained on **GCP NVIDIA L4 GPU (`24 GB VRAM`)** over **891 ExplainXKCD stage directions** and **226 speech-bubble-masked real XKCD comic panels**.

- **GitHub Repository**: [https://github.com/hemanth/xkcd-artist](https://github.com/hemanth/xkcd-artist)
- **Live Interactive Studio**: [https://hemanth.github.io/xkcd-artist/](https://hemanth.github.io/xkcd-artist/)

## Model Contents

| File | Size | Description |
| :--- | :---: | :--- |
| `model.safetensors` | `6.16 MB` | `XkcdArtistForGeneration` FP32 weights (`1,586,603` params) |
| `lora_weights.safetensors` | `376 KB` | Internal LoRA adapter weights (`93,568` params) |
| `onnx/model_quantized.onnx` | `1.72 MB` | Dynamic Int8 quantized ONNX Runtime graph (`3.58x` compression) |
| `sd_xkcd_lora/adapter_model.safetensors` | `6.60 MB` | Stable Diffusion (`segmind/tiny-sd`) UNet LoRA (`r=16`, `1,726,464` trainable params) trained on speech-bubble-masked XKCD panels |
| `sd_xkcd_lora/adapter_config.json` | `1 KB` | PEFT / Diffusers LoRA config (`attn1`/`attn2` `to_q`, `to_k`, `to_v`, `to_out.0`) |
| `modeling_xkcd_artist.py` | `6 KB` | Self-contained PyTorch model definition (`PreTrainedModel`) |

## Empirical Benchmark (`B = 1,000` Bootstrap on Held-Out XKCD + ExplainXKCD)

| Runtime | Artifact Size | Layout Acc (`%`) | Stage Prop Acc (`%`) | Kinematic Pose MAE | Bubble Clearance (`%`) | Abstention AUROC | Latency `p50` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch FP32 + Diffusion LoRA** | `6.16 MB` (`1.59M`) | **100.0%** `[100.0, 100.0]` | **100.0%** `[100.0, 100.0]` | **0.0051** | **100.0%** | **1.0** | `0.629 ms` |
| **Browser ONNX Int8 (WASM SIMD)** | `1.72 MB` (`3.58x` smaller) | **100.0%** `[100.0, 100.0]` | **100.0%** `[100.0, 100.0]` | **0.0053** | **100.0%** | **1.0** | `0.621 ms` |

## Usage (PyTorch / ONNX)

```python
from xkcd_artist import xkcd_artist

comic = xkcd_artist(
    "Machine learning pile of linear algebra stir the pile",
    backend="onnx",  # 'onnx' | 'pytorch'
)

svg_markup = comic.to_svg()
png_bytes = comic.to_png("machine_learning_xkcd.png")
```

## License

MIT © [Hemanth HM](https://h3manth.com)
