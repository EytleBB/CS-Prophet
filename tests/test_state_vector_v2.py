"""Tests for the current 218-dim realtime-aligned state_vector_v2 schema."""

import numpy as np
import pandas as pd
import pytest

from src.features.state_vector_v2 import (
    BOMB_BASE,
    CT_BASE,
    FEATURE_DIM,
    FEATURE_IDX,
    FEATURE_NAMES,
    GLOBAL_BASE,
    MAPS,
    MOLOTOV_BASE,
    PLAYER_HP_MAX,
    SMOKE_BASE,
    TEAM_SCORE_MAX,
    TIME_IN_ROUND_MAX,
    WEAPON_ID_MAP,
    build_state_matrix,
    build_state_vector,
    flatten_feature_dict,
    normalize_feature_row,
)
from src.utils.map_utils import normalize_coords


def _make_flat_row(**overrides) -> pd.Series:
    data = {name: 0.0 for name in FEATURE_NAMES}
    data["map_de_dust2"] = 1.0
    data.update(overrides)
    return pd.Series(data)


def _make_preview_feature(**overrides) -> dict:
    feature: dict = {
        "ct_score": 3,
        "t_score": 1,
        "round_num": 5,
        "time_in_round": 12.5,
        "bomb_dropped": 0.0,
        "bomb_x": 0.0,
        "bomb_y": 0.0,
        "map_onehot": [1.0 if m == "de_dust2" else 0.0 for m in MAPS],
    }
    for side in ("t", "ct"):
        for idx in range(5):
            prefix = f"{side}{idx}"
            for field in (
                "x",
                "y",
                "z",
                "yaw",
                "in_bomb_zone",
                "hp",
                "armor",
                "helmet",
                "alive",
                "has_smoke",
                "has_flash",
                "has_he",
                "has_molotov",
                "has_c4",
                "balance",
                "equip_value",
                "score",
                "weapon_id",
            ):
                feature[f"{prefix}_{field}"] = 0.0
    feature.update(overrides)
    return feature


class TestSchema:
    def test_feature_dim_is_218(self):
        assert FEATURE_DIM == 218

    def test_feature_names_len_matches_dim(self):
        assert len(FEATURE_NAMES) == FEATURE_DIM

    def test_feature_names_are_unique(self):
        assert len(set(FEATURE_NAMES)) == FEATURE_DIM

    def test_anchor_indices_are_stable(self):
        assert FEATURE_NAMES[0] == "t0_x"
        assert FEATURE_NAMES[17] == "t0_weapon_id"
        assert FEATURE_NAMES[CT_BASE] == "ct0_x"
        assert FEATURE_NAMES[GLOBAL_BASE] == "ct_score"
        assert FEATURE_NAMES[BOMB_BASE] == "bomb_dropped"
        assert FEATURE_NAMES[SMOKE_BASE] == "smoke0_x"
        assert FEATURE_NAMES[MOLOTOV_BASE] == "molotov0_x"
        assert FEATURE_NAMES[-1] == "map_de_anubis"


class TestBuildStateVector:
    def test_output_shape_and_dtype(self):
        vec = build_state_vector(_make_flat_row())
        assert vec.shape == (FEATURE_DIM,)
        assert vec.dtype == np.float32

    def test_flat_row_values_are_normalized_at_expected_indices(self):
        row = _make_flat_row(
            t0_x=1000.0,
            t0_y=2000.0,
            t0_z=50.0,
            t0_yaw=90.0,
            t0_in_bomb_zone=1.0,
            t0_hp=80.0,
            t0_armor=50.0,
            t0_helmet=1.0,
            t0_has_smoke=1.0,
            t0_balance=8000.0,
            t0_equip_value=6200.0,
            t0_score=25.0,
            t0_weapon_id=float(WEAPON_ID_MAP["ak_47"]),
            ct0_alive=1.0,
            ct_score=7.0,
            time_in_round=44.25,
            bomb_dropped=1.0,
            bomb_x=900.0,
            bomb_y=2200.0,
            smoke0_remain=0.75,
            map_de_dust2=1.0,
        )
        vec = build_state_vector(row)
        exp_x, exp_y, exp_z = normalize_coords(1000.0, 2000.0, 50.0, "de_dust2")
        exp_bx, exp_by, _ = normalize_coords(900.0, 2200.0, 0.0, "de_dust2")

        assert vec[FEATURE_IDX["t0_x"]] == pytest.approx(exp_x)
        assert vec[FEATURE_IDX["t0_y"]] == pytest.approx(exp_y)
        assert vec[FEATURE_IDX["t0_z"]] == pytest.approx(exp_z)
        assert vec[FEATURE_IDX["t0_yaw"]] == pytest.approx(0.5)
        assert vec[FEATURE_IDX["t0_in_bomb_zone"]] == pytest.approx(1.0)
        assert vec[FEATURE_IDX["t0_hp"]] == pytest.approx(80.0 / PLAYER_HP_MAX)
        assert vec[FEATURE_IDX["t0_armor"]] == pytest.approx(0.5)
        assert vec[FEATURE_IDX["t0_helmet"]] == pytest.approx(1.0)
        assert vec[FEATURE_IDX["t0_has_smoke"]] == pytest.approx(1.0)
        assert vec[FEATURE_IDX["t0_balance"]] == pytest.approx(0.5)
        assert vec[FEATURE_IDX["t0_score"]] == pytest.approx(0.5)
        assert vec[FEATURE_IDX["t0_weapon_id"]] == pytest.approx(float(WEAPON_ID_MAP["ak_47"]))
        assert vec[FEATURE_IDX["ct0_alive"]] == pytest.approx(1.0)
        assert vec[FEATURE_IDX["ct_score"]] == pytest.approx(7.0 / TEAM_SCORE_MAX)
        assert vec[FEATURE_IDX["time_in_round"]] == pytest.approx(44.25 / TIME_IN_ROUND_MAX)
        assert vec[FEATURE_IDX["bomb_dropped"]] == pytest.approx(1.0)
        assert vec[FEATURE_IDX["bomb_x"]] == pytest.approx(exp_bx)
        assert vec[FEATURE_IDX["bomb_y"]] == pytest.approx(exp_by)
        assert vec[FEATURE_IDX["smoke0_remain"]] == pytest.approx(0.75)
        assert vec[FEATURE_IDX["map_de_dust2"]] == pytest.approx(1.0)

    def test_missing_fields_default_to_zero(self):
        vec = build_state_vector(pd.Series({"map_de_dust2": 1.0}))
        assert vec[FEATURE_IDX["t0_x"]] == 0.0
        assert vec[FEATURE_IDX["ct_score"]] == 0.0
        assert vec[FEATURE_IDX["map_de_dust2"]] == 1.0

    def test_preview_style_dict_is_flattened(self):
        feat = _make_preview_feature(
            t0_x=1000.0,
            t0_y=2000.0,
            t0_hp=55.0,
            t0_weapon_id=float(WEAPON_ID_MAP["ak_47"]),
            bomb_dropped=1.0,
            bomb_x=12.0,
            bomb_y=34.0,
        )
        vec = build_state_vector(feat)
        exp_x, exp_y, _ = normalize_coords(1000.0, 2000.0, 0.0, "de_dust2")
        exp_bx, exp_by, _ = normalize_coords(12.0, 34.0, 0.0, "de_dust2")

        assert vec[FEATURE_IDX["t0_x"]] == pytest.approx(exp_x)
        assert vec[FEATURE_IDX["t0_y"]] == pytest.approx(exp_y)
        assert vec[FEATURE_IDX["t0_hp"]] == pytest.approx(55.0 / PLAYER_HP_MAX)
        assert vec[FEATURE_IDX["t0_weapon_id"]] == pytest.approx(float(WEAPON_ID_MAP["ak_47"]))
        assert vec[FEATURE_IDX["bomb_dropped"]] == pytest.approx(1.0)
        assert vec[FEATURE_IDX["bomb_x"]] == pytest.approx(exp_bx)
        assert vec[FEATURE_IDX["bomb_y"]] == pytest.approx(exp_by)
        assert vec[FEATURE_IDX["map_de_dust2"]] == pytest.approx(1.0)

    def test_normalize_feature_row_exposes_current_policy(self):
        row = _make_flat_row(
            t0_x=1000.0,
            t0_y=2000.0,
            t0_z=50.0,
            t0_yaw=-180.0,
            t0_hp=100.0,
            smoke0_x=900.0,
            smoke0_y=2200.0,
            smoke0_remain=0.5,
            map_de_dust2=1.0,
        )
        normalized = normalize_feature_row(row)
        exp_x, exp_y, exp_z = normalize_coords(1000.0, 2000.0, 50.0, "de_dust2")
        exp_sx, exp_sy, _ = normalize_coords(900.0, 2200.0, 0.0, "de_dust2")

        assert normalized["t0_x"] == pytest.approx(exp_x)
        assert normalized["t0_y"] == pytest.approx(exp_y)
        assert normalized["t0_z"] == pytest.approx(exp_z)
        assert normalized["t0_yaw"] == pytest.approx(-1.0)
        assert normalized["t0_hp"] == pytest.approx(1.0)
        assert normalized["smoke0_x"] == pytest.approx(exp_sx)
        assert normalized["smoke0_y"] == pytest.approx(exp_sy)
        assert normalized["smoke0_remain"] == pytest.approx(0.5)


class TestFlattenFeatureDict:
    def test_flattens_weapon_id_and_map_vector(self):
        feat = _make_preview_feature(
            t0_weapon_id=float(WEAPON_ID_MAP["awp"]),
            map_onehot=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        flat = flatten_feature_dict(feat)

        assert flat["t0_weapon_id"] == pytest.approx(float(WEAPON_ID_MAP["awp"]))
        assert flat["map_de_dust2"] == pytest.approx(1.0)
        assert flat["map_de_mirage"] == pytest.approx(0.0)


class TestBuildStateMatrix:
    def test_matrix_shape_and_dtype(self):
        rows = [
            _make_flat_row(t0_x=1000.0, map_de_dust2=1.0),
            _make_flat_row(t0_x=1200.0, map_de_dust2=1.0),
        ]
        df = pd.DataFrame(rows)
        mat = build_state_matrix(df)
        ref = np.vstack([build_state_vector(row) for _, row in df.iterrows()])

        assert mat.shape == (2, FEATURE_DIM)
        assert mat.dtype == np.float32
        assert np.allclose(mat, ref)

    def test_empty_matrix(self):
        mat = build_state_matrix(pd.DataFrame(columns=FEATURE_NAMES))
        assert mat.shape == (0, FEATURE_DIM)
        assert mat.dtype == np.float32
