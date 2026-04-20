"""Tests for the current canonical v2 feature row builder."""

import pandas as pd
import pytest

from src.features.feature_builder_v2 import build_feature_row_v2, build_round_label_map
from src.features.state_vector_v2 import FEATURE_NAMES, WEAPON_ID_MAP


def _player_row(name: str, team_name: str, inventory: list[str], **overrides) -> dict:
    row = {
        "round_num": 1,
        "tick": 250,
        "name": name,
        "team_name": team_name,
        "X": 1000.0 if team_name == "TERRORIST" else -1000.0,
        "Y": 2200.0 if team_name == "TERRORIST" else 2300.0,
        "Z": 10.0,
        "yaw": 90.0,
        "in_bomb_zone": team_name == "TERRORIST",
        "health": 100,
        "armor_value": 100,
        "has_helmet": True,
        "is_alive": True,
        "inventory": inventory,
        "balance": 3700,
        "current_equip_value": 6200,
        "score": 12,
    }
    row.update(overrides)
    return row


def _tick_df() -> pd.DataFrame:
    t_rows = [
        _player_row("alpha", "TERRORIST", ["AK-47", "Glock-18", "C4 Explosive", "Smoke Grenade"]),
        _player_row("bravo", "TERRORIST", ["Galil AR", "Flashbang"], X=1100.0),
        _player_row("charlie", "TERRORIST", ["Desert Eagle", "Molotov"], X=1200.0),
        _player_row("delta", "TERRORIST", ["Tec-9"], X=1300.0),
        _player_row("echo", "TERRORIST", [], X=1400.0, is_alive=False, yaw=0.0),
    ]
    ct_rows = [
        _player_row("foxtrot", "CT", ["M4A1-S"], X=-1200.0),
        _player_row("golf", "CT", ["AWP"], X=-1300.0),
        _player_row("hotel", "CT", ["MP9"], X=-1400.0),
        _player_row("india", "CT", ["USP-S"], X=-1500.0),
        _player_row("juliet", "CT", [], X=-1600.0),
    ]
    return pd.DataFrame(t_rows + ct_rows)


def _events() -> dict:
    return {
        "bomb_dropped": pd.DataFrame(
            [{"tick": 240, "X": 900.0, "Y": 2200.0, "user_name": "alpha", "round_num": 1}]
        ),
        "bomb_pickup": pd.DataFrame(columns=["tick", "X", "Y", "user_name", "round_num"]),
        "smokegrenade_detonate": pd.DataFrame(
            [{"tick": 230, "x": 800.0, "y": 2100.0, "round_num": 1}]
        ),
        "inferno_startburn": pd.DataFrame(
            [{"tick": 220, "x": -1000.0, "y": 2400.0, "round_num": 1}]
        ),
        "bomb_planted": pd.DataFrame(
            [{"tick": 260, "site": 504, "X": 1300.0, "Y": 2200.0, "Z": 0.0, "round_num": 1}]
        ),
    }


def _round_info() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "round_num": 1,
                "freeze_tick": 100,
                "plant_tick": 260,
                "end_tick": 300,
                "bomb_site": pd.NA,
                "ct_score": 3,
                "t_score": 1,
            }
        ]
    )


class TestBuildRoundLabelMap:
    def test_extracts_A_B_from_normalized_xyz(self):
        labels = build_round_label_map(_events(), map_name="de_dust2")
        assert labels == {1: "A"}


class TestBuildFeatureRowV2:
    def test_returns_current_canonical_feature_row(self):
        tick_df = _tick_df()
        row = build_feature_row_v2(
            tick_df=tick_df,
            tick_slice=tick_df,
            events=_events(),
            tick=250,
            round_num=1,
            map_name="de_dust2",
            round_info=_round_info(),
        )

        assert set(FEATURE_NAMES).issubset(row.keys())
        assert len([key for key in FEATURE_NAMES if key in row]) == len(FEATURE_NAMES)
        assert row["t0_x"] == pytest.approx(1000.0)
        assert row["t0_hp"] == pytest.approx(100.0)
        assert row["t0_has_c4"] == pytest.approx(1.0)
        assert row["t0_has_smoke"] == pytest.approx(1.0)
        assert row["t0_weapon_id"] == pytest.approx(float(WEAPON_ID_MAP["ak_47"]))
        assert row["ct1_weapon_id"] == pytest.approx(float(WEAPON_ID_MAP["awp"]))
        assert row["round_num"] == pytest.approx(1.0)
        assert row["time_in_round"] == pytest.approx((250 - 100) / 64.0)
        assert row["ct_score"] == pytest.approx(3.0)
        assert row["t_score"] == pytest.approx(1.0)
        assert row["bomb_dropped"] == pytest.approx(1.0)
        assert row["bomb_x"] == pytest.approx(900.0)
        assert row["bomb_y"] == pytest.approx(2200.0)
        assert row["smoke0_x"] == pytest.approx(800.0)
        assert row["smoke0_y"] == pytest.approx(2100.0)
        assert row["smoke0_remain"] == pytest.approx(1.0 - (250 - 230) / (18 * 64))
        assert row["molotov0_x"] == pytest.approx(-1000.0)
        assert row["molotov0_y"] == pytest.approx(2400.0)
        assert row["molotov0_remain"] == pytest.approx(1.0 - (250 - 220) / (7 * 64))
        assert row["map_de_dust2"] == pytest.approx(1.0)
        assert row["map_de_mirage"] == pytest.approx(0.0)
        assert "t0_vx" not in row
        assert "ct1_is_scoped" not in row
        assert "t0_flash_dur" not in row
