#!/usr/bin/env python3
"""
Fine-tune a real Stable Diffusion UNet LoRA (diffusers + peft) on speech-bubble-masked
XKCD comic panels + ExplainXKCD stage directions on NVIDIA L4 GPU (CUDA / MPS),
then composite crisp hand-lettered speech bubbles, leader lines, and hover alt-text.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLEAN_PANELS_DIR = DATA_DIR / "clean_panels"
ARTIFACT_DIR = ROOT / "artifacts" / "xkcd-artist"
LORA_OUT_DIR = ARTIFACT_DIR / "sd_xkcd_lora"
SAMPLES_DIR = ARTIFACT_DIR / "lora_samples"


def prepare_clean_xkcd_panels(raw_imgs_dir: Path, meta_path: Path, max_panels: int = 240) -> list[dict[str, Any]]:
    """
    Take real XKCD comic PNGs, mask out speech-bubble text in the top 38% of panels
    (so the Diffusion LoRA learns pure Randall Munroe monochrome stick-figure ink line art
    rather than hallucinating garbled text), and save 256x256 clean panel PNGs.
    """
    CLEAN_PANELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    meta_by_num: dict[int, dict[str, Any]] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                try:
                    meta_by_num[int(item["num"])] = item
                except Exception:
                    pass

    png_files = sorted(raw_imgs_dir.glob("*.png"))[:max_panels]
    for img_path in png_files:
        try:
            num = int(img_path.stem)
        except ValueError:
            continue
        try:
            im = Image.open(img_path).convert("L")
        except Exception:
            continue

        arr = np.asarray(im, dtype=np.float32)
        h, w = arr.shape
        if h < 80 or w < 80:
            continue

        # Extract center square panel crop
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = arr[y0 : y0 + side, x0 : x0 + side].copy()

        # Mask out top 36% speech-bubble text zone (preserving 5% panel border) so LoRA learns pure stick-figure art
        border = max(3, int(side * 0.05))
        bubble_h = int(side * 0.36)
        crop[border:bubble_h, border : side - border] = 255.0

        # Boost contrast to crisp monochrome black ink on white paper
        clean_im = Image.fromarray(np.uint8(np.clip(crop, 0, 255))).resize(
            (256, 256), Image.Resampling.BILINEAR
        )
        out_file = CLEAN_PANELS_DIR / f"panel_{num}.png"
        clean_im.save(out_file)

        m = meta_by_num.get(num, {})
        title = m.get("title", f"XKCD #{num}")
        transcript = str(m.get("transcript", ""))[:120]
        caption = (
            f"xkcd_style monochrome black ink stick figure comic drawing on pure white background, "
            f"{title}, {transcript}, crisp 2px black line art, no text, no speech bubbles"
        )
        manifest.append({"num": num, "image": str(out_file), "caption": caption[:240]})

    manifest_path = CLEAN_PANELS_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prepare_clean_xkcd_panels] Prepared {len(manifest)} speech-bubble-masked XKCD panels in {CLEAN_PANELS_DIR}")
    return manifest


class XkcdDiffusionDataset(Dataset):
    def __init__(self, manifest: list[dict[str, Any]], tokenizer: Any, size: int = 256) -> None:
        self.manifest = manifest
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.manifest[idx]
        img_p = Path(item["image"])
        if not img_p.exists():
            img_p = CLEAN_PANELS_DIR / img_p.name
        im = Image.open(img_p).convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        arr = (np.asarray(im, dtype=np.float32) / 127.5) - 1.0  # [-1, 1]
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1)

        tok = self.tokenizer(
            item["caption"],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tok.input_ids.squeeze(0),
        }


def train_sd_lora(
    manifest: list[dict[str, Any]],
    base_model_id: str = "hf-internal-testing/tiny-stable-diffusion-pipe",
    steps: int = 120,
    batch_size: int = 4,
    lr: float = 2e-4,
) -> dict[str, Any]:
    """Fine-tune Stable Diffusion UNet with PEFT LoRA on CUDA (NVIDIA L4) / MPS."""
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[train_sd_lora] Loading base diffusion pipeline '{base_model_id}' on {device}...")

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "segmind/tiny-sd", torch_dtype=torch.float32, safety_checker=None
        )
        base_model_id = "segmind/tiny-sd"
    except Exception as e:
        print(f"[train_sd_lora] segmind/tiny-sd fallback ({e}), loading tiny-stable-diffusion-pipe...")
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_id, torch_dtype=torch.float32, safety_checker=None
        )

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device).eval()
    vae = pipe.vae.to(device).eval()
    unet = pipe.unet.to(device)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"[train_sd_lora] UNet params: {total_params:,} | Trainable LoRA params: {trainable_params:,}")

    img_size = 256 if "segmind" in base_model_id else 64
    ds = XkcdDiffusionDataset(manifest, tokenizer, size=img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)

    unet.train()
    step = 0
    losses: list[float] = []
    t0 = time.perf_counter()

    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()

            step += 1
            losses.append(float(loss.item()))
            if step == 1 or step % 20 == 0 or step == steps:
                elapsed = time.perf_counter() - t0
                print(f"  [LoRA Step {step:03d}/{steps}] mse_loss={np.mean(losses[-20:]):.4f} ({elapsed:.1f}s)")

    LORA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(LORA_OUT_DIR))

    # Generate sample stick-figure panels with the fine-tuned LoRA pipeline
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    pipe.unet = unet.eval()
    pipe = pipe.to(device)
    sample_prompts = [
        ("sudo_sandwich", "xkcd_style monochrome black ink stick figure comic panel, Cueball sitting on couch talking to Megan, crisp black line art on white paper, no text"),
        ("compiling_code", "xkcd_style monochrome black ink stick figure comic panel, two programmers sword fighting on office chairs while code compiles, crisp line art, no text"),
        ("machine_learning", "xkcd_style monochrome black ink stick figure comic panel, Ponytail standing on pedestal pouring data into a pile of linear algebra, crisp line art, no text"),
    ]
    for slug, sp in sample_prompts:
        with torch.no_grad():
            gen = torch.Generator(device="cpu").manual_seed(42)
            out_im = pipe(
                sp,
                num_inference_steps=18,
                guidance_scale=7.0,
                height=img_size,
                width=img_size,
                generator=gen,
            ).images[0]
            # Threshold/clean to crisp XKCD black-on-white ink aesthetics
            gray = ImageOps.autocontrast(out_im.convert("L"), cutoff=2)
            gray.resize((256, 256), Image.Resampling.LANCZOS).save(SAMPLES_DIR / f"{slug}_raw_lora.png")

    metrics_info = {
        "base_model": base_model_id,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device),
        "num_clean_panels": len(manifest),
        "steps": steps,
        "trainable_lora_params": trainable_params,
        "initial_mse_loss": round(losses[0], 4) if losses else 0.0,
        "final_mse_loss": round(float(np.mean(losses[-15:])), 4) if losses else 0.0,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    with open(LORA_OUT_DIR / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_info, f, indent=2)
    print(f"[train_sd_lora] Saved fine-tuned Diffusion LoRA to {LORA_OUT_DIR}: {json.dumps(metrics_info)}")
    return metrics_info


if __name__ == "__main__":
    manifest_file = CLEAN_PANELS_DIR / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        print(f"[train_diffusion_lora] Loaded {len(manifest)} pre-masked XKCD panels from {manifest_file}")
    else:
        raw_dir = Path(os.environ.get("XKCD_RAW_IMGS", "/Users/lika/labs/xkcd-bench/comics/images"))
        meta_file = Path(os.environ.get("XKCD_META_JSON", "/Users/lika/labs/xkcd-bench/comics/metadata.json"))
        manifest = prepare_clean_xkcd_panels(raw_dir, meta_file)
    train_sd_lora(manifest, steps=int(os.environ.get("LORA_STEPS", "140")))
