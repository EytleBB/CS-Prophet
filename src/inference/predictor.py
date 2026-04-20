"""Real-time inference helper that loads a trained model and scores sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.model.transformer import BombSiteTransformer


class RoundPredictor:
    """Load a trained BombSiteTransformer checkpoint and predict A/B probabilities."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        temperature: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.temperature = temperature
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_config = checkpoint["model_config"]
        self.model = BombSiteTransformer(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict(self, features: np.ndarray) -> dict[str, float]:
        """Return bomb-site probabilities for one round sequence.

        Args:
            features: Float32 array of shape ``(seq_len, input_dim)``.

        Returns:
            ``{"A": p_a, "B": p_b}`` where probabilities sum to 1.
        """
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        src_key_padding_mask = (x.abs().sum(dim=-1) == 0)
        with torch.no_grad():
            logits = self.model(x, src_key_padding_mask=src_key_padding_mask)
            probs = torch.softmax(logits / self.temperature, dim=-1).squeeze().tolist()
        return {"A": probs[0], "B": probs[1]}
