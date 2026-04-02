"""Real-time inference module — loads a trained model and scores live round state."""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np

from src.model.transformer import BombSiteTransformer


class RoundPredictor:
    """Load a trained BombSiteTransformer checkpoint and predict bomb plant site probabilities."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_config = checkpoint["model_config"]
        self.model = BombSiteTransformer(**model_config)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict(self, features: np.ndarray) -> dict[str, float]:
        """Return bomb plant site probabilities.

        Args:
            features: float32 array of shape (seq_len, 74) — one round's state sequence.

        Returns:
            dict with keys 'A', 'B', 'other' — probabilities summing to 1.0.
        """
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        return {"A": probs[0], "B": probs[1], "other": probs[2]}
