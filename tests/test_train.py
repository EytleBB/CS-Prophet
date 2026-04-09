"""Tests for FocalLoss and evaluate."""

import torch
import torch.nn.functional as F
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.model.train import FocalLoss, evaluate
from src.model.transformer import BombSiteTransformer


def _tiny_model() -> BombSiteTransformer:
    return BombSiteTransformer(
        input_dim=275, d_model=32, nhead=4, num_layers=1, dropout=0.0, num_classes=3
    )


def _tiny_loader(n: int = 8, seq_len: int = 10) -> DataLoader:
    x = torch.randn(n, seq_len, 275)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


# ── FocalLoss ─────────────────────────────────────────────────────────────

class TestFocalLoss:
    def test_returns_scalar(self):
        criterion = FocalLoss(gamma=2.0)
        loss = criterion(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))
        assert loss.shape == ()

    def test_positive_loss(self):
        criterion = FocalLoss(gamma=2.0)
        loss = criterion(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))
        assert loss.item() > 0

    def test_no_nan(self):
        criterion = FocalLoss(gamma=2.0)
        loss = criterion(torch.randn(8, 3), torch.randint(0, 3, (8,)))
        assert not torch.isnan(loss)

    def test_gamma_0_equals_cross_entropy(self):
        """With γ=0, focal loss reduces to standard cross-entropy."""
        torch.manual_seed(42)
        criterion = FocalLoss(gamma=0.0)
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        focal = criterion(logits, targets)
        ce = F.cross_entropy(logits, targets)
        assert torch.allclose(focal, ce, atol=1e-5)

    def test_gamma_2_leq_gamma_0_for_easy_examples(self):
        """Higher γ down-weights easy (high-confidence correct) examples."""
        logits = torch.zeros(4, 3)
        logits[0, 0] = 10.0; logits[1, 1] = 10.0
        logits[2, 2] = 10.0; logits[3, 0] = 10.0
        targets = torch.tensor([0, 1, 2, 0])
        loss_g0 = FocalLoss(gamma=0.0)(logits, targets)
        loss_g2 = FocalLoss(gamma=2.0)(logits, targets)
        assert loss_g2.item() < loss_g0.item()

    def test_differentiable(self):
        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 3, requires_grad=True)
        loss = criterion(logits, torch.tensor([0, 1, 2, 1]))
        loss.backward()
        assert logits.grad is not None


# ── evaluate ──────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_float_tuple(self):
        model = _tiny_model()
        criterion = FocalLoss(gamma=2.0)
        loss, acc = evaluate(model, _tiny_loader(), criterion,
                             torch.device("cpu"), use_amp=False)
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_accuracy_in_unit_range(self):
        model = _tiny_model()
        criterion = FocalLoss(gamma=2.0)
        _, acc = evaluate(model, _tiny_loader(), criterion,
                          torch.device("cpu"), use_amp=False)
        assert 0.0 <= acc <= 1.0

    def test_loss_positive(self):
        model = _tiny_model()
        criterion = FocalLoss(gamma=2.0)
        loss, _ = evaluate(model, _tiny_loader(), criterion,
                           torch.device("cpu"), use_amp=False)
        assert loss > 0.0

    def test_perfect_model_accuracy_1(self):
        """A model that always outputs the correct class should get 1.0 accuracy."""
        class PerfectModel(torch.nn.Module):
            def forward(self, x, src_key_padding_mask=None):
                # Always predict class 0 for a batch of all-0 labels
                return torch.tensor([[10.0, 0.0, 0.0]]).expand(x.size(0), -1)

        x = torch.randn(8, 10, 275)
        y = torch.zeros(8, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        criterion = FocalLoss(gamma=2.0)
        _, acc = evaluate(PerfectModel(), loader, criterion,
                          torch.device("cpu"), use_amp=False)
        assert acc == pytest.approx(1.0)
