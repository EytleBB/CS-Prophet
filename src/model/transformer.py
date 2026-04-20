"""BombSiteTransformer for bomb-site prediction from round sequences."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.attention import CrossAttentionLayer, PositionalEncoding

# weapon_id offset within each player's stride (last field in PLAYER_BASE_FIELDS)
_V2_WEAPON_ID_OFFSET: int = 17
_V2_PLAYER_STRIDE: int = 18
_V2_NUM_WEAPONS: int = 35
_V2_WEAPON_EMBED_DIM: int = 8

_SCHEMA_SPECS: dict[int, dict] = {
    # v1: 10 x 27 player features + 5 global = 275
    275: {
        "t_players": slice(0, 135),
        "ct_players": slice(135, 270),
        "global": slice(270, 275),
        "t_in": 140,
        "ct_in": 135,
        "has_weapon_embed": False,
    },
    # v2: 10 x 18 stored + 38 shared = 218 stored
    # after weapon embedding: per-player 17 cont + 8 embed = 25
    # t_in = 5*25 + 38 = 163, ct_in = 5*25 = 125
    218: {
        "t_players": slice(0, 90),
        "ct_players": slice(90, 180),
        "global": slice(180, 218),
        "t_in": 163,
        "ct_in": 125,
        "has_weapon_embed": True,
    },
}


def _embed_players(
    player_block: torch.Tensor,
    weapon_embedding: nn.Embedding,
    stride: int,
    weapon_offset: int,
) -> torch.Tensor:
    """Replace weapon_id with learned embedding in a player feature block.

    Args:
        player_block: (B, T, 5*stride) — 5 players concatenated.
        weapon_embedding: Embedding(num_weapons, embed_dim).
        stride: fields per player in stored representation.
        weapon_offset: index of weapon_id within each player's stride.

    Returns:
        (B, T, 5*(stride - 1 + embed_dim)) — weapon_id replaced by embedding.
    """
    B, T, _ = player_block.shape
    embed_dim = weapon_embedding.embedding_dim
    parts = []
    for i in range(5):
        start = i * stride
        end = start + stride
        player = player_block[:, :, start:end]  # (B, T, stride)

        # Split: continuous features before weapon_id, weapon_id, after weapon_id
        cont_before = player[:, :, :weapon_offset]
        wid = player[:, :, weapon_offset].long().clamp(0, weapon_embedding.num_embeddings - 1)
        cont_after = player[:, :, weapon_offset + 1:]        # (B, T, 0) if weapon_id is last

        w_emb = weapon_embedding(wid)  # (B, T, embed_dim)
        parts.append(torch.cat([cont_before, w_emb, cont_after], dim=-1))

    return torch.cat(parts, dim=-1)


class BombSiteTransformer(nn.Module):
    """Sequence-to-label Transformer for bomb-site prediction.

    Architecture:
    1. Split the input into T-side player features + shared context and CT-side
       player features.
        2. For v2 (218-dim): replace per-player weapon_id with learned embedding.
    3. Project each stream to ``d_model`` with learned linear layers.
    4. Add sinusoidal positional encoding to both.
    5. Cross-attention: T queries CT to model adversarial interaction.
    6. Self-attention encoder stack on the T-side representation.
    7. Linear classifier on the last real (non-padded) timestep.
    """

    def __init__(
        self,
        input_dim: int = 275,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_dim not in _SCHEMA_SPECS:
            raise ValueError(
                "BombSiteTransformer supports input_dim in "
                f"{sorted(_SCHEMA_SPECS)}; got {input_dim}"
            )

        spec = _SCHEMA_SPECS[input_dim]
        self._t_player_slice = spec["t_players"]
        self._ct_player_slice = spec["ct_players"]
        self._global_slice = spec["global"]
        self._has_weapon_embed = spec["has_weapon_embed"]

        if self._has_weapon_embed:
            self.weapon_embedding = nn.Embedding(
                _V2_NUM_WEAPONS, _V2_WEAPON_EMBED_DIM,
            )

        self.t_proj = nn.Linear(int(spec["t_in"]), d_model)
        self.ct_proj = nn.Linear(int(spec["ct_in"]), d_model)
        self.t_pos_enc = PositionalEncoding(d_model, dropout)
        self.ct_pos_enc = PositionalEncoding(d_model, dropout)
        self.cross_attn = CrossAttentionLayer(d_model, nhead, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict bomb plant site from a batch of round sequences."""
        t_players = x[:, :, self._t_player_slice]
        ct_players = x[:, :, self._ct_player_slice]
        global_feats = x[:, :, self._global_slice]

        if self._has_weapon_embed:
            t_players = _embed_players(
                t_players, self.weapon_embedding,
                _V2_PLAYER_STRIDE, _V2_WEAPON_ID_OFFSET,
            )
            ct_players = _embed_players(
                ct_players, self.weapon_embedding,
                _V2_PLAYER_STRIDE, _V2_WEAPON_ID_OFFSET,
            )

        t_feats = torch.cat([t_players, global_feats], dim=-1)
        ct_feats = ct_players

        t_emb = self.t_pos_enc(self.t_proj(t_feats))
        ct_emb = self.ct_pos_enc(self.ct_proj(ct_feats))

        t_emb = self.cross_attn(t_emb, ct_emb, key_padding_mask=src_key_padding_mask)
        out = self.encoder(t_emb, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            seq_lens = (~src_key_padding_mask).sum(dim=1) - 1
            last_real = out[torch.arange(out.size(0), device=out.device), seq_lens, :]
        else:
            last_real = out[:, -1, :]
        return self.classifier(last_real)
