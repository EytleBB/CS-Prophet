"""Tests for BombSiteTransformer."""

import torch
import pytest
from src.model.transformer import BombSiteTransformer


class TestBombSiteTransformer:
    def _model(self, **kwargs) -> BombSiteTransformer:
        defaults = dict(input_dim=74, d_model=64, nhead=4,
                        num_layers=2, dropout=0.0, num_classes=3)
        defaults.update(kwargs)
        return BombSiteTransformer(**defaults)

    def test_output_shape(self):
        model = self._model()
        x = torch.randn(4, 240, 74)
        out = model(x)
        assert out.shape == (4, 3)

    def test_output_shape_short_sequence(self):
        model = self._model()
        x = torch.randn(2, 10, 74)
        out = model(x)
        assert out.shape == (2, 3)

    def test_no_nan(self):
        model = self._model()
        x = torch.randn(2, 50, 74)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_batch_size_1(self):
        model = self._model()
        x = torch.randn(1, 240, 74)
        out = model(x)
        assert out.shape == (1, 3)

    def test_num_classes_respected(self):
        model = self._model(num_classes=3)
        x = torch.randn(2, 20, 74)
        out = model(x)
        assert out.shape[-1] == 3

    def test_gradients_flow(self):
        model = self._model()
        x = torch.randn(2, 30, 74, requires_grad=False)
        out = model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_eval_mode_no_dropout_effect(self):
        model = self._model(dropout=0.5)
        model.eval()
        x = torch.randn(1, 20, 74)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)
