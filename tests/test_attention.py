"""Tests for PositionalEncoding and CrossAttentionLayer."""

import torch
import pytest
from src.model.attention import CrossAttentionLayer, PositionalEncoding


class TestPositionalEncoding:
    def test_output_shape_preserved(self):
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        x = torch.randn(4, 30, 64)
        out = pe(x)
        assert out.shape == x.shape

    def test_no_nan(self):
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        x = torch.zeros(2, 50, 64)
        out = pe(x)
        assert not torch.isnan(out).any()

    def test_adds_different_values_per_position(self):
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        x = torch.zeros(1, 5, 64)
        out = pe(x)
        for i in range(4):
            assert not torch.allclose(out[0, i], out[0, i + 1])

    def test_long_sequence_within_max_len(self):
        pe = PositionalEncoding(d_model=32, dropout=0.0, max_len=300)
        x = torch.randn(1, 240, 32)
        out = pe(x)
        assert out.shape == (1, 240, 32)


class TestCrossAttentionLayer:
    def test_output_shape_matches_query(self):
        layer = CrossAttentionLayer(d_model=64, nhead=4, dropout=0.0)
        q = torch.randn(2, 30, 64)
        kv = torch.randn(2, 30, 64)
        out = layer(q, kv)
        assert out.shape == q.shape

    def test_no_nan(self):
        layer = CrossAttentionLayer(d_model=64, nhead=4, dropout=0.0)
        q = torch.randn(2, 20, 64)
        kv = torch.randn(2, 20, 64)
        out = layer(q, kv)
        assert not torch.isnan(out).any()

    def test_different_kv_produces_different_output(self):
        torch.manual_seed(0)
        layer = CrossAttentionLayer(d_model=32, nhead=4, dropout=0.0)
        layer.eval()
        q = torch.randn(1, 10, 32)
        kv1 = torch.randn(1, 10, 32)
        kv2 = torch.randn(1, 10, 32)
        out1 = layer(q, kv1)
        out2 = layer(q, kv2)
        assert not torch.allclose(out1, out2)

    def test_batch_size_1(self):
        layer = CrossAttentionLayer(d_model=64, nhead=8, dropout=0.0)
        q = torch.randn(1, 240, 64)
        kv = torch.randn(1, 240, 64)
        out = layer(q, kv)
        assert out.shape == (1, 240, 64)
