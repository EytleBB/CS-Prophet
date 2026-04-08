"""Tests for state_vector — single-row feature vector builder."""

import numpy as np
import pandas as pd
import pytest
from src.features.state_vector import (
    FEATURE_DIM, ROLE_IDX, WEAPON_CAT_IDX, build_state_vector,
    _PLAYER_STRIDE, _CT_BASE, _ZONE_BASE, _GLOBAL_BASE,
)


def _make_row(map_zone: str = "A", **overrides) -> pd.Series:
    data: dict = {
        "map_zone":         map_zone,
        "ct_score":         0,
        "t_score":          0,
        "round_num":        1,
        "ct_losing_streak": 0,
        "t_losing_streak":  0,
    }
    for side in ("t", "ct"):
        for i in range(5):
            data[f"{side}{i}_x"]             = 0.5
            data[f"{side}{i}_y"]             = 0.5
            data[f"{side}{i}_z"]             = 0.5
            data[f"{side}{i}_hp"]            = 100
            data[f"{side}{i}_armor"]         = 100
            data[f"{side}{i}_helmet"]        = True
            data[f"{side}{i}_alive"]         = True
            data[f"{side}{i}_role"]          = ""
            data[f"{side}{i}_weapon"]        = "other"
            data[f"{side}{i}_has_smoke"]     = False
            data[f"{side}{i}_has_flash"]     = False
            data[f"{side}{i}_has_he"]        = False
            data[f"{side}{i}_has_molotov"]   = False
            data[f"{side}{i}_flash_duration"] = 0.0
            data[f"{side}{i}_equip_value"]   = 0
            data[f"{side}{i}_is_scoped"]     = False
            data[f"{side}{i}_is_defusing"]   = False
    data.update(overrides)
    return pd.Series(data)


class TestFeatureDim:
    def test_constant_is_279(self):
        assert FEATURE_DIM == 279


class TestBuildStateVector:
    def test_output_shape(self):
        assert build_state_vector(_make_row()).shape == (279,)

    def test_dtype_is_float32(self):
        assert build_state_vector(_make_row()).dtype == np.float32

    def test_hp_normalised_to_1(self):
        assert build_state_vector(_make_row(t0_hp=100))[3] == pytest.approx(1.0)

    def test_hp_normalised_to_half(self):
        assert build_state_vector(_make_row(t0_hp=50))[3] == pytest.approx(0.5)

    def test_armor_normalised(self):
        assert build_state_vector(_make_row(t0_armor=60))[4] == pytest.approx(0.6)

    def test_ct_player_offset(self):
        # CT base=135, player 0, hp is field index 3 → vec[138]
        assert build_state_vector(_make_row(ct0_hp=80))[138] == pytest.approx(0.8)

    def test_dead_player_alive_false(self):
        assert build_state_vector(_make_row(t0_alive=False))[6] == 0.0

    def test_alive_player_alive_true(self):
        assert build_state_vector(_make_row(t0_alive=True))[6] == 1.0

    def test_all_values_in_unit_range(self):
        vec = build_state_vector(_make_row())
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)


class TestZoneOneHot:
    def test_a(self):
        vec = build_state_vector(_make_row(map_zone="A"))
        assert vec[_ZONE_BASE + 0] == 1.0
        assert vec[_ZONE_BASE + 1] == 0.0
        assert vec[_ZONE_BASE + 2] == 0.0
        assert vec[_ZONE_BASE + 3] == 0.0

    def test_b(self):
        assert build_state_vector(_make_row(map_zone="B"))[_ZONE_BASE + 1] == 1.0

    def test_mid(self):
        assert build_state_vector(_make_row(map_zone="mid"))[_ZONE_BASE + 2] == 1.0

    def test_other(self):
        assert build_state_vector(_make_row(map_zone="other"))[_ZONE_BASE + 3] == 1.0


class TestRoleOneHot:
    def test_known_role_sets_correct_bit(self):
        for role, expected_idx in ROLE_IDX.items():
            vec = build_state_vector(_make_row(t0_role=role))
            role_start = 7
            for k in range(5):
                expected = 1.0 if k == expected_idx else 0.0
                assert vec[role_start + k] == pytest.approx(expected), (
                    f"role={role!r}: expected vec[{role_start + k}]={expected}"
                )

    def test_unknown_role_is_all_zeros(self):
        assert np.all(build_state_vector(_make_row(t0_role="Rifler"))[7:12] == 0.0)

    def test_empty_role_is_all_zeros(self):
        assert np.all(build_state_vector(_make_row(t0_role=""))[7:12] == 0.0)

    def test_missing_role_column_is_all_zeros(self):
        data = {k: v for k, v in _make_row().items() if not k.endswith("_role")}
        assert np.all(build_state_vector(pd.Series(data))[7:12] == 0.0)

    def test_ct_player_role_offset(self):
        vec = build_state_vector(_make_row(ct0_role="AWPer"))
        awper_idx = ROLE_IDX["AWPer"]  # 1
        assert vec[_CT_BASE + 7 + awper_idx] == 1.0
        for k in range(5):
            if k != awper_idx:
                assert vec[_CT_BASE + 7 + k] == 0.0

    def test_role_bits_isolated_between_players(self):
        vec = build_state_vector(_make_row(t0_role="IGL"))
        t1_role_start = _PLAYER_STRIDE + 7  # player 1 base=27, role offset=7
        assert np.all(vec[t1_role_start:t1_role_start + 5] == 0.0)


class TestWeaponOneHot:
    def test_rifle_sets_correct_bit(self):
        vec = build_state_vector(_make_row(t0_weapon="rifle"))
        weapon_start = 12  # within player 0 block
        assert vec[weapon_start + WEAPON_CAT_IDX["rifle"]] == 1.0

    def test_sniper_sets_correct_bit(self):
        vec = build_state_vector(_make_row(t0_weapon="sniper"))
        assert vec[12 + WEAPON_CAT_IDX["sniper"]] == 1.0

    def test_unknown_weapon_maps_to_other(self):
        vec = build_state_vector(_make_row(t0_weapon="unknown_gun"))
        assert vec[12 + WEAPON_CAT_IDX["other"]] == 1.0

    def test_missing_weapon_column_maps_to_other(self):
        data = {k: v for k, v in _make_row().items() if k != "t0_weapon"}
        vec = build_state_vector(pd.Series(data))
        assert vec[12 + WEAPON_CAT_IDX["other"]] == 1.0

    def test_only_one_weapon_bit_set(self):
        vec = build_state_vector(_make_row(t0_weapon="pistol"))
        assert vec[12:19].sum() == pytest.approx(1.0)

    def test_ct_player_weapon_offset(self):
        vec = build_state_vector(_make_row(ct0_weapon="sniper"))
        assert vec[_CT_BASE + 12 + WEAPON_CAT_IDX["sniper"]] == 1.0


class TestGrenadeInventory:
    def test_has_smoke(self):
        vec = build_state_vector(_make_row(t0_has_smoke=True))
        assert vec[19] == 1.0

    def test_has_flash(self):
        assert build_state_vector(_make_row(t0_has_flash=True))[20] == 1.0

    def test_has_he(self):
        assert build_state_vector(_make_row(t0_has_he=True))[21] == 1.0

    def test_has_molotov(self):
        assert build_state_vector(_make_row(t0_has_molotov=True))[22] == 1.0

    def test_no_nades_all_zero(self):
        vec = build_state_vector(_make_row())
        assert np.all(vec[19:23] == 0.0)

    def test_ct_nade_offset(self):
        vec = build_state_vector(_make_row(ct0_has_smoke=True))
        assert vec[_CT_BASE + 19] == 1.0


class TestScalarFeatures:
    def test_flash_duration_normalised(self):
        vec = build_state_vector(_make_row(t0_flash_duration=1.5))
        assert vec[23] == pytest.approx(0.5)

    def test_flash_duration_capped_at_1(self):
        assert build_state_vector(_make_row(t0_flash_duration=9.0))[23] == pytest.approx(1.0)

    def test_equip_value_normalised(self):
        vec = build_state_vector(_make_row(t0_equip_value=10000))
        assert vec[24] == pytest.approx(0.5)

    def test_is_scoped(self):
        assert build_state_vector(_make_row(t0_is_scoped=True))[25] == 1.0

    def test_is_defusing(self):
        assert build_state_vector(_make_row(t0_is_defusing=True))[26] == 1.0


class TestGlobalFeatures:
    def test_ct_score_normalised(self):
        vec = build_state_vector(_make_row(ct_score=15))
        assert vec[_GLOBAL_BASE + 0] == pytest.approx(0.5)

    def test_t_score_normalised(self):
        vec = build_state_vector(_make_row(t_score=10))
        assert vec[_GLOBAL_BASE + 1] == pytest.approx(10 / 30)

    def test_round_num_normalised(self):
        vec = build_state_vector(_make_row(round_num=6))
        assert vec[_GLOBAL_BASE + 2] == pytest.approx(6 / 30)

    def test_ct_losing_streak(self):
        vec = build_state_vector(_make_row(ct_losing_streak=5))
        assert vec[_GLOBAL_BASE + 3] == pytest.approx(1.0)

    def test_t_losing_streak(self):
        vec = build_state_vector(_make_row(t_losing_streak=2))
        assert vec[_GLOBAL_BASE + 4] == pytest.approx(2 / 5)

    def test_missing_global_cols_default_zero(self):
        data = {k: v for k, v in _make_row().items()
                if k not in ("ct_score", "t_score", "ct_losing_streak", "t_losing_streak")}
        vec = build_state_vector(pd.Series(data))
        assert vec[_GLOBAL_BASE + 0] == 0.0
        assert vec[_GLOBAL_BASE + 1] == 0.0
