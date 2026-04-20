"""Tests for realtime GSI -> v2 feature-row conversion."""

import pytest

from src.features.state_vector_v2 import WEAPON_ID_MAP
from src.inference.gsi_state_builder import build_row_from_gsi


def _payload() -> dict:
    return {
        "map": {
            "name": "de_dust2",
            "phase": "live",
            "round": 7,
            "team_ct": {"score": 3},
            "team_t": {"score": 5},
        },
        "round": {"phase": "live"},
        "phase_countdowns": {"phase": "live", "phase_ends_in": "100.0"},
        "bomb": {"state": "dropped", "position": "900.0, 2200.0, 0.0"},
        "grenades": {
            "1": {
                "type": "smoke",
                "position": "800.0, 2100.0, 0.0",
                "effecttime": "9.0",
                "lifetime": "15.0",
                "velocity": "0.0, 0.0, 0.0",
            },
            "2": {
                "type": "inferno",
                "lifetime": "3.5",
                "flames": {
                    "a": "1000.0, 2000.0, 0.0",
                    "b": "1010.0, 2020.0, 0.0",
                },
            },
        },
        "allplayers": {
            "steam_t": {
                "team": "T",
                "name": "alpha",
                "position": "1000.0, 2200.0, 10.0",
                "forward": "0.0, 1.0, 0.0",
                "state": {
                    "health": 100,
                    "armor": 50,
                    "helmet": True,
                    "money": 3700,
                    "equip_value": 6200,
                },
                "match_stats": {"score": 12},
                "weapons": {
                    "weapon_0": {"name": "weapon_knife"},
                    "weapon_1": {"name": "weapon_ak47"},
                    "weapon_2": {"name": "weapon_smokegrenade"},
                    "weapon_3": {"name": "weapon_c4"},
                },
            },
            "steam_ct": {
                "team": "CT",
                "name": "bravo",
                "position": "-1000.0, 2300.0, 10.0",
                "forward": "1.0, 0.0, 0.0",
                "state": {
                    "health": 95,
                    "armor": 100,
                    "helmet": True,
                    "money": 5100,
                    "equip_value": 5800,
                },
                "match_stats": {"score": 15},
                "weapons": {
                    "weapon_0": {"name": "weapon_knife"},
                    "weapon_1": {"name": "weapon_m4a1_silencer"},
                    "weapon_2": {"name": "weapon_flashbang"},
                },
            },
        },
    }


def test_build_row_from_gsi_maps_live_payload():
    row = build_row_from_gsi(_payload(), step=3, round_num=7, map_name="de_dust2")

    assert row is not None
    assert row["t0_weapon_id"] == pytest.approx(float(WEAPON_ID_MAP["ak_47"]))
    assert row["ct0_weapon_id"] == pytest.approx(float(WEAPON_ID_MAP["m4a1_s"]))
    assert row["t0_has_smoke"] == pytest.approx(1.0)
    assert row["t0_has_c4"] == pytest.approx(1.0)
    assert row["ct0_has_flash"] == pytest.approx(1.0)
    assert row["t0_in_bomb_zone"] == pytest.approx(1.0)
    assert row["t0_yaw"] == pytest.approx(90.0)
    assert row["ct0_yaw"] == pytest.approx(0.0)
    assert row["time_in_round"] == pytest.approx(15.0)
    assert row["bomb_dropped"] == pytest.approx(1.0)
    assert row["bomb_x"] == pytest.approx(900.0)
    assert row["bomb_y"] == pytest.approx(2200.0)
    assert row["smoke0_x"] == pytest.approx(800.0)
    assert row["smoke0_y"] == pytest.approx(2100.0)
    assert row["smoke0_remain"] == pytest.approx(0.5)
    assert row["molotov0_x"] == pytest.approx(1005.0)
    assert row["molotov0_y"] == pytest.approx(2010.0)
    assert row["molotov0_remain"] == pytest.approx(0.5)
    assert row["ct_score"] == pytest.approx(3.0)
    assert row["t_score"] == pytest.approx(5.0)
    assert row["round_num"] == pytest.approx(7.0)
    assert row["map_de_dust2"] == pytest.approx(1.0)
