"""Real-time inference module — loads a trained model and scores live round state."""

import torch
import numpy as np
from pathlib import Path

from src.model.transformer import RoundTransformer


class RoundPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = RoundTransformer(**checkpoint["model_kwargs"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, features: np.ndarray) -> dict:
        """Return win probabilities for CT and T sides."""
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        return {"CT": probs[0], "T": probs[1]}
