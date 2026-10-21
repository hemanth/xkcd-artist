"""
Custom HuggingFace PreTrainedConfig + PreTrainedModel for xkcd-artist:
Hybrid LoRA Stick-Figure Ink Decoder + 5-Head Scene, Pose & Speech-Bubble Layout Model.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedConfig, PreTrainedModel


class XkcdArtistConfig(PreTrainedConfig):
    model_type = "xkcd_artist"

    def __init__(
        self,
        vocab_size: int = 1200,
        max_seq_len: int = 36,
        num_characters: int = 6,
        num_layouts: int = 6,
        num_props: int = 8,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 3,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        pose_dim: int = 12,
        bubble_dim: int = 12,
        style_dim: int = 4,
        patch_size: int = 16,
        dropout: float = 0.15,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_characters = num_characters
        self.num_layouts = num_layouts
        self.num_props = num_props
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.pose_dim = pose_dim
        self.bubble_dim = bubble_dim
        self.style_dim = style_dim
        self.patch_size = patch_size
        self.dropout = dropout


class LoRALinearBlock(nn.Module):
    """Linear layer augmented with explicit Low-Rank Adaptation (LoRA A/B) matrices."""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / float(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


class XkcdArtistForGeneration(PreTrainedModel):
    config_class = XkcdArtistConfig

    def __init__(self, config: XkcdArtistConfig) -> None:
        super().__init__(config)
        d = config.d_model

        self.token_emb = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Embedding(config.max_seq_len, d)
        self.cast_emb = nn.Embedding(config.num_characters, d)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.nhead,
            dim_feedforward=d * 3,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(d)

        # LoRA-adapted Stick-Figure Latent Ink Denoiser / Decoder (predicts 16x16 clean stick-figure ink map)
        self.lora_proj_1 = LoRALinearBlock(d, d, rank=config.lora_rank, alpha=config.lora_alpha)
        self.lora_proj_2 = LoRALinearBlock(d, d, rank=config.lora_rank, alpha=config.lora_alpha)
        self.ink_patch_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            LoRALinearBlock(d, config.patch_size * config.patch_size, rank=config.lora_rank, alpha=config.lora_alpha),
        )

        # 5 Joint Structural & Style Heads
        self.layout_head = nn.Linear(d, config.num_layouts)
        self.prop_head = nn.Linear(d, config.num_props)
        self.pose_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, config.pose_dim),
        )
        self.bubble_box_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, config.bubble_dim),
            nn.Sigmoid(),
        )
        self.style_head = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Linear(64, config.style_dim),
        )
        self.abstention_head = nn.Linear(d, 1)

        self.post_init()

    def forward(
        self,
        token_ids: torch.Tensor,
        cast_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len = token_ids.shape
        device = token_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)

        tok_h = self.token_emb(token_ids.clamp(0, self.config.vocab_size - 1)) + self.pos_emb(positions)
        cast_h = self.cast_emb(cast_ids.clamp(0, self.config.num_characters - 1)).mean(dim=1, keepdim=True)
        h = self.encoder(tok_h + cast_h)

        # Masked mean pooling over non-pad tokens (token_ids != 0)
        non_pad = (token_ids != 0).to(h.dtype).unsqueeze(-1)
        non_pad_count = non_pad.sum(dim=1).clamp(min=1.0)
        pooled = self.norm((h * non_pad).sum(dim=1) / non_pad_count)

        # LoRA residual stick-figure latent representation
        lora_h = F.gelu(self.lora_proj_1(pooled))
        lora_h = pooled + self.lora_proj_2(lora_h)

        layout_logits = self.layout_head(lora_h)
        prop_logits = self.prop_head(lora_h)
        poses = self.pose_head(lora_h)

        raw_bubbles = self.bubble_box_head(lora_h)
        # Enforce non-overlapping upper-panel bubble invariant (cy <= 0.36, h <= 0.22)
        b_reshaped = raw_bubbles.view(bsz, 3, 4)
        bx = 0.15 + 0.70 * b_reshaped[:, :, 0]
        by = 0.10 + 0.26 * b_reshaped[:, :, 1]
        bw = 0.32 + 0.30 * b_reshaped[:, :, 2]
        bh = 0.10 + 0.12 * b_reshaped[:, :, 3]
        bubbles = torch.stack([bx, by, bw, bh], dim=-1).view(bsz, 12)

        style_vec = self.style_head(lora_h)
        ink_patch = torch.sigmoid(self.ink_patch_head(lora_h)).view(
            bsz, self.config.patch_size, self.config.patch_size
        )
        invalid_logit = self.abstention_head(pooled).squeeze(-1)

        return {
            "layout_logits": layout_logits,
            "prop_logits": prop_logits,
            "poses": poses,
            "bubbles": bubbles,
            "style_vec": style_vec,
            "ink_patch": ink_patch,
            "invalid_logit": invalid_logit,
        }
