#!/usr/bin/env python3
"""
Train, calibrate, and export XkcdArtistForGeneration + standalone LoRA adapter weights
on Apple Silicon MPS / CUDA / CPU over the real XKCD + ExplainXKCD dataset.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from generate_dataset import DATA_DIR, VOCAB_WORDS, build_datasets
from modeling_xkcd_artist import XkcdArtistConfig, XkcdArtistForGeneration

ARTIFACT_DIR = ROOT / "artifacts" / "xkcd-artist"


class XkcdArtistDataset(Dataset):
    def __init__(self, jsonl_path: Path) -> None:
        self.rows: list[dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.rows[idx]
        return {
            "token_ids": torch.tensor(r["token_ids"], dtype=torch.long),
            "cast_ids": torch.tensor(r["cast_ids"], dtype=torch.long),
            "layout_id": torch.tensor(r["layout_id"], dtype=torch.long),
            "prop_id": torch.tensor(r["prop_id"], dtype=torch.long),
            "poses": torch.tensor(r["poses"], dtype=torch.float32),
            "bubbles": torch.tensor(r["bubbles"], dtype=torch.float32),
            "style_vec": torch.tensor(r["style_vec"], dtype=torch.float32),
            "ink_patch": torch.tensor(r["ink_patch"], dtype=torch.float32),
            "is_invalid": torch.tensor(float(r["is_invalid"]), dtype=torch.float32),
        }


def evaluate(model: XkcdArtistForGeneration, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    layout_correct = 0
    prop_correct = 0
    valid_total = 0
    pose_mae_sum = 0.0
    ink_mae_sum = 0.0
    abstention_correct = 0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            cast_ids = batch["cast_ids"].to(device)
            layout_id = batch["layout_id"].to(device)
            prop_id = batch["prop_id"].to(device)
            poses = batch["poses"].to(device)
            ink_patch = batch["ink_patch"].to(device)
            is_invalid = batch["is_invalid"].to(device)

            out = model(token_ids=token_ids, cast_ids=cast_ids)
            pred_invalid = (torch.sigmoid(out["invalid_logit"]) > 0.5).float()
            abstention_correct += int((pred_invalid == is_invalid).sum().item())
            total_items += token_ids.size(0)

            valid_mask = is_invalid < 0.5
            n_val = int(valid_mask.sum().item())
            if n_val > 0:
                valid_total += n_val
                layout_pred = out["layout_logits"][valid_mask].argmax(dim=-1)
                prop_pred = out["prop_logits"][valid_mask].argmax(dim=-1)
                layout_correct += int((layout_pred == layout_id[valid_mask]).sum().item())
                prop_correct += int((prop_pred == prop_id[valid_mask]).sum().item())
                pose_mae_sum += float(
                    F.l1_loss(out["poses"][valid_mask], poses[valid_mask], reduction="sum").item()
                ) / 12.0
                ink_mae_sum += float(
                    F.l1_loss(out["ink_patch"][valid_mask], ink_patch[valid_mask], reduction="sum").item()
                ) / 256.0

    return {
        "layout_acc": round(layout_correct / max(1, valid_total), 4),
        "prop_acc": round(prop_correct / max(1, valid_total), 4),
        "pose_mae": round(pose_mae_sum / max(1, valid_total), 4),
        "ink_mae": round(ink_mae_sum / max(1, valid_total), 4),
        "abstention_acc": round(abstention_correct / max(1, total_items), 4),
    }


def main() -> None:
    if not (DATA_DIR / "train.jsonl").exists():
        build_datasets()

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[train_gcp] Training XkcdArtistForGeneration + LoRA on device={device}")

    train_ds = XkcdArtistDataset(DATA_DIR / "train.jsonl")
    val_ds = XkcdArtistDataset(DATA_DIR / "val.jsonl")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    cfg = XkcdArtistConfig(vocab_size=1200, max_seq_len=36)
    model = XkcdArtistForGeneration(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    print(f"[train_gcp] Total params: {total_params:,} | LoRA adapter params: {lora_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=0.01)
    epochs = 18
    total_steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(4, int(total_steps * 0.08)), num_training_steps=total_steps
    )

    best_score = -1.0
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        t0 = time.perf_counter()

        for batch in train_loader:
            token_ids = batch["token_ids"].to(device)
            cast_ids = batch["cast_ids"].to(device)
            layout_id = batch["layout_id"].to(device)
            prop_id = batch["prop_id"].to(device)
            poses = batch["poses"].to(device)
            bubbles = batch["bubbles"].to(device)
            style_vec = batch["style_vec"].to(device)
            ink_patch = batch["ink_patch"].to(device)
            is_invalid = batch["is_invalid"].to(device)

            out = model(token_ids=token_ids, cast_ids=cast_ids)
            loss_abst = F.binary_cross_entropy_with_logits(out["invalid_logit"], is_invalid)

            valid_mask = is_invalid < 0.5
            if valid_mask.any():
                loss_layout = F.cross_entropy(out["layout_logits"][valid_mask], layout_id[valid_mask])
                loss_prop = F.cross_entropy(out["prop_logits"][valid_mask], prop_id[valid_mask])
                loss_pose = F.smooth_l1_loss(out["poses"][valid_mask], poses[valid_mask]) * 4.0
                loss_bubble = F.smooth_l1_loss(out["bubbles"][valid_mask], bubbles[valid_mask]) * 4.0
                loss_style = F.smooth_l1_loss(out["style_vec"][valid_mask], style_vec[valid_mask]) * 2.0
                loss_ink = F.smooth_l1_loss(out["ink_patch"][valid_mask], ink_patch[valid_mask]) * 5.0
                loss = loss_abst + loss_layout + loss_prop + loss_pose + loss_bubble + loss_style + loss_ink
            else:
                loss = loss_abst

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.item())

        metrics = evaluate(model, val_loader, device)
        composite = metrics["layout_acc"] + metrics["prop_acc"] + metrics["abstention_acc"] - metrics["pose_mae"]
        dt = (time.perf_counter() - t0) * 1000.0
        if epoch in (1, 5, 10, 15, epochs) or composite > best_score:
            print(
                f"  Epoch {epoch:02d}/{epochs} ({dt:.0f}ms) | loss={loss_sum/len(train_loader):.4f} | "
                f"layout={metrics['layout_acc']:.3f} prop={metrics['prop_acc']:.3f} "
                f"pose_mae={metrics['pose_mae']:.4f} ink_mae={metrics['ink_mae']:.4f} abst={metrics['abstention_acc']:.3f}"
            )
        if composite >= best_score:
            best_score = composite
            model.save_pretrained(ARTIFACT_DIR)
            # Also save standalone LoRA adapter weights
            lora_state = {
                k: v.detach().cpu().contiguous()
                for k, v in model.state_dict().items()
                if "lora_" in k or "ink_patch_head" in k
            }
            save_file(lora_state, str(ARTIFACT_DIR / "lora_weights.safetensors"))

    print(f"[train_gcp] Saved model.safetensors and lora_weights.safetensors to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
