"""BombSiteTransformer — predicts bomb plant site (A / B / other) from round sequences."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.attention import CrossAttentionLayer, PositionalEncoding

# Indices into the 74-dim feature vector (see state_vector.py):
#   [0:35]   T-player features (5 × 7)
#   [35:70]  CT-player features (5 × 7)
#   [70:74]  map_zone one-hot
_T_SLICE = (slice(None), slice(None), slice(0, 35))       # T-player features
_ZONE_SLICE = (slice(None), slice(None), slice(70, 74))   # zone one-hot
_CT_SLICE = (slice(None), slice(None), slice(35, 70))     # CT-player features

_T_IN = 39   # 35 T-player features + 4 zone features → projected to d_model
_CT_IN = 35  # 35 CT-player features → projected to d_model


class BombSiteTransformer(nn.Module):
    """Sequence-to-label Transformer for bomb-site prediction.

    Architecture:
        1. Split input into T-side (player + zone, 39-dim) and CT-side (35-dim)
        2. Project each to d_model with learned linear layers
        3. Add sinusoidal positional encoding to both
        4. Cross-attention: T queries CT to model adversarial interaction
        5. Self-attention encoder stack on T-side representation
        6. Linear classifier on last-token output → (num_classes,) logits
    """

    def __init__(
        self,
        input_dim: int = 74,   # kept for config compatibility; actual split is fixed
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if input_dim != 74:
            raise ValueError(
                f"BombSiteTransformer requires input_dim=74 (35 T + 35 CT + 4 zone); got {input_dim}"
            )
        self.t_proj = nn.Linear(_T_IN, d_model)
        self.ct_proj = nn.Linear(_CT_IN, d_model)
        self.t_pos_enc = PositionalEncoding(d_model, dropout)
        self.ct_pos_enc = PositionalEncoding(d_model, dropout)
        self.cross_attn = CrossAttentionLayer(d_model, nhead, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict bomb plant site from a batch of round sequences.

        Args:
            x: float32 of shape (batch, seq_len, 74).
            src_key_padding_mask: optional bool tensor of shape (batch, seq_len),
                True at positions that should be ignored (padding). Forwarded to
                cross-attention and the self-attention encoder.

        Returns:
            (batch, num_classes) logits — pass through softmax for probabilities.
        """
        # Split feature vector
        t_feats = torch.cat([x[_T_SLICE], x[_ZONE_SLICE]], dim=-1)  # (B, T, 39)
        ct_feats = x[_CT_SLICE]                                       # (B, T, 35)

        # Project + positional encoding
        t_emb = self.t_pos_enc(self.t_proj(t_feats))
        ct_emb = self.ct_pos_enc(self.ct_proj(ct_feats))

        # Cross-attention: T side enriched with CT context
        t_emb = self.cross_attn(t_emb, ct_emb, key_padding_mask=src_key_padding_mask)

        # Self-attention encoder
        out = self.encoder(t_emb, src_key_padding_mask=src_key_padding_mask)

        # Classify from last timestep
        return self.classifier(out[:, -1, :])
