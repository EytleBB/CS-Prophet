"""Attention modules for BombSiteTransformer."""

from __future__ import annotations
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sinusoidal encoding; got {d_model}")
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding and apply dropout.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]  # type: ignore[index]
        return self.dropout(x)


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer: query attends to key_value with residual + LayerNorm."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute cross-attention with residual connection.

        Args:
            query: (batch, seq_len, d_model) — T-side representation.
            key_value: (batch, seq_len, d_model) — CT-side representation.
            key_padding_mask: (batch, seq_len) bool mask; True = ignore position.

        Returns:
            (batch, seq_len, d_model) — query enriched with CT context.
        """
        attn_out, _ = self.attn(query, key_value, key_value, key_padding_mask=key_padding_mask)
        return self.norm(query + self.dropout(attn_out))
