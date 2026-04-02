"""Tests for state_vector — single-row feature vector builder."""

import numpy as np
import pandas as pd
import pytest
from src.features.state_vector import FEATURE_DIM, build_state_vector


def _make_row(map_zone: str = "A", **overrides) -> pd.Series:
    data: dict = {"map_zone": map_zone}
    for side in ("t", "ct"):
        for i in range(5):
            data[f"{side}{i}_x"] = 0.5
            data[f"{side}{i}_y"] = 0.5
            data[f"{side}{i}_z"] = 0.5
            data[f"{side}{i}_hp"] = 100
            data[f"{side}{i}_armor"] = 100
            data[f"{side}{i}_helmet"] = True
            data[f"{side}{i}_alive"] = True
    data.update(overrides)
    return pd.Series(data)


class TestFeatureDim:
    def test_constant_is_74(self):
        assert FEATURE_DIM == 74


class TestBuildStateVector:
    def test_output_shape(self):
        vec = build_state_vector(_make_row())
        assert vec.shape == (74,)

    def test_dtype_is_float32(self):
        vec = build_state_vector(_make_row())
        assert vec.dtype == np.float32

    def test_hp_normalised_to_1(self):
        vec = build_state_vector(_make_row(t0_hp=100))
        assert vec[3] == pytest.approx(1.0)

    def test_hp_normalised_to_half(self):
        vec = build_state_vector(_make_row(t0_hp=50))
        assert vec[3] == pytest.approx(0.5)

    def test_armor_normalised(self):
        vec = build_state_vector(_make_row(t0_armor=60))
        assert vec[4] == pytest.approx(0.6)

    def test_ct_player_offset(self):
        vec = build_state_vector(_make_row(ct0_hp=80))
        assert vec[38] == pytest.approx(0.8)

    def test_zone_one_hot_A(self):
        vec = build_state_vector(_make_row(map_zone="A"))
        assert vec[70] == 1.0
        assert vec[71] == 0.0
        assert vec[72] == 0.0
        assert vec[73] == 0.0

    def test_zone_one_hot_B(self):
        vec = build_state_vector(_make_row(map_zone="B"))
        assert vec[70] == 0.0
        assert vec[71] == 1.0

    def test_zone_one_hot_mid(self):
        vec = build_state_vector(_make_row(map_zone="mid"))
        assert vec[72] == 1.0

    def test_zone_one_hot_other(self):
        vec = build_state_vector(_make_row(map_zone="other"))
        assert vec[73] == 1.0

    def test_all_values_in_unit_range(self):
        vec = build_state_vector(_make_row())
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)

    def test_dead_player_alive_false(self):
        vec = build_state_vector(_make_row(t0_alive=False))
        assert vec[6] == 0.0

    def test_alive_player_alive_true(self):
        vec = build_state_vector(_make_row(t0_alive=True))
        assert vec[6] == 1.0
