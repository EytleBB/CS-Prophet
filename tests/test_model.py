"""Basic smoke tests for the Transformer model."""

import torch
import pytest
from src.model.transformer import RoundTransformer


def test_forward_shape():
    model = RoundTransformer(input_dim=64)
    x = torch.randn(4, 128, 64)  # (batch, seq_len, input_dim)
    out = model(x)
    assert out.shape == (4, 2), f"Unexpected output shape: {out.shape}"


def test_forward_no_nan():
    model = RoundTransformer(input_dim=64)
    x = torch.randn(2, 32, 64)
    out = model(x)
    assert not torch.isnan(out).any()
