"""Tests for memory-reader -> raw v2 feature-row conversion."""

from __future__ import annotations

import pytest

from src.features.state_vector_v2 import FEATURE_NAMES, WEAPON_ID_MAP
from src.inference.memory_state_builder import build_row_from_memory


def _player(
    *,
    name: str,
    team: str,
    alive: bool = True,
    hp: int = 100,
    armor: int = 50,
    money: int = 3000,
    helmet: bool = False,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
    weapons: list[str] | None = None,
    active_weapon_class: str | None = None,
) -> dict:
    return {
        "steamid": hash((team, name)) & 0xFFFFFFFF,
        "name": name,
        "team": team,
        "alive": alive,
        "hp": hp,
        "armor": armor,
        "money": money,
        "helmet": helmet,
        "x": x,
        "y": y,
        "z": z,
        "yaw": yaw,
        "weapons": list(weapons or []),
        "active_weapon_class": active_weapon_class,
    }


def test_build_row_from_memory_maps_players_to_raw_schema():
    players = [
        {
            "steamid": 2,
            "name": "zeta",
            "team": "T",
            "alive": False,
            "hp": 0,
            "armor": 80,
            "money": 2400,
            "helmet": True,
            "x": 650.0,
            "y": 2200.0,
            "z": 0.0,
            "yaw": 45.0,
        },
        {
            "steamid": 1,
            "name": "alpha",
            "team": "T",
            "alive": True,
            "hp": 100,
            "armor": 50,
            "money": 3700,
            "helmet": True,
            "x": 900.0,
            "y": 2300.0,
            "z": 10.0,
            "yaw": 90.0,
        },
        {
            "steamid": 3,
            "name": "bravo",
            "team": "CT",
            "alive": True,
            "hp": 95,
            "armor": 100,
            "money": 5100,
            "helmet": True,
            "x": -1000.0,
            "y": 2300.0,
            "z": 5.0,
            "yaw": 180.0,
        },
    ]
    map_state = {
        "ct_score": 3,
        "t_score": 5,
        "time_in_round": 42.5,
        "map_phase": "live",
        "round_phase": "live",
        "bomb_state": "",
    }

    row = build_row_from_memory(
        players=players,
        map_state=map_state,
        round_num=7,
        map_name="de_dust2",
    )

    assert row is not None
    assert set(row) == set(FEATURE_NAMES)
    assert len(row) == 218

    assert row["t0_x"] == pytest.approx(900.0)
    assert row["t0_y"] == pytest.approx(2300.0)
    assert row["t0_z"] == pytest.approx(10.0)
    assert row["t0_yaw"] == pytest.approx(90.0)
    assert row["t0_in_bomb_zone"] == pytest.approx(1.0)
    assert row["t0_hp"] == pytest.approx(100.0)
    assert row["t0_armor"] == pytest.approx(50.0)
    assert row["t0_helmet"] == pytest.approx(1.0)
    assert row["t0_alive"] == pytest.approx(1.0)
    assert row["t0_balance"] == pytest.approx(3700.0)

    assert row["t1_yaw"] == pytest.approx(0.0)
    assert row["t1_in_bomb_zone"] == pytest.approx(0.0)
    assert row["t1_alive"] == pytest.approx(0.0)
    assert row["t1_balance"] == pytest.approx(2400.0)

    assert row["ct0_x"] == pytest.approx(-1000.0)
    assert row["ct0_yaw"] == pytest.approx(180.0)
    assert row["ct0_hp"] == pytest.approx(95.0)
    assert row["ct0_balance"] == pytest.approx(5100.0)

    assert row["t2_hp"] == pytest.approx(0.0)
    assert row["ct1_hp"] == pytest.approx(0.0)
    assert row["ct_score"] == pytest.approx(3.0)
    assert row["t_score"] == pytest.approx(5.0)
    assert row["round_num"] == pytest.approx(7.0)
    assert row["time_in_round"] == pytest.approx(42.5)
    assert row["bomb_dropped"] == pytest.approx(0.0)
    assert row["bomb_x"] == pytest.approx(0.0)
    assert row["bomb_y"] == pytest.approx(0.0)
    assert row["map_de_dust2"] == pytest.approx(1.0)
    assert sum(row[name] for name in FEATURE_NAMES if name.startswith("map_")) == pytest.approx(1.0)


def test_build_row_from_memory_rejects_missing_or_unknown_map():
    players = [{"name": "alpha", "team": "T", "alive": True, "hp": 100, "x": 0.0, "y": 0.0, "z": 0.0}]

    assert build_row_from_memory(players, {}, round_num=1, map_name="") is None
    assert build_row_from_memory(players, {}, round_num=1, map_name="de_cache") is None
    assert build_row_from_memory([], {}, round_num=1, map_name="de_dust2") is None


def test_build_row_from_memory_populates_inventory_flags_weapon_ids_and_projectiles():
    players = [
        _player(
            name="alpha",
            team="T",
            armor=100,
            money=5400,
            helmet=True,
            x=100.0,
            y=200.0,
            z=5.0,
            yaw=90.0,
            weapons=[
                "weapon_ak47",
                "weapon_glock",
                "weapon_smokegrenade",
                "weapon_flashbang",
                "weapon_hegrenade",
                "weapon_molotov",
                "weapon_c4",
            ],
            active_weapon_class="weapon_ak47",
        ),
        _player(
            name="bravo",
            team="CT",
            armor=100,
            money=4800,
            helmet=True,
            x=-100.0,
            y=50.0,
            z=2.0,
            yaw=180.0,
            weapons=[
                "weapon_m4a1_silencer",
                "weapon_hkp2000",
                "weapon_incgrenade",
                "weapon_flashbang",
            ],
            active_weapon_class="weapon_m4a1_silencer",
        ),
    ]
    map_state = {
        "ct_score": 8,
        "t_score": 9,
        "time_in_round": 30.0,
        "bomb": {"planted": False, "dropped": True, "x": 120.0, "y": 240.0, "site": ""},
        "projectiles": {
            "smokes": [
                (100.0, 100.0, 0.25),
                (200.0, 200.0, 0.90),
                (150.0, 150.0, 0.50),
                (50.0, 50.0, 0.10),
                (60.0, 60.0, 0.75),
                (70.0, 70.0, 0.80),
            ],
            "molotovs": [
                (10.0, 10.0, 0.20),
                (30.0, 30.0, 0.90),
                (20.0, 20.0, 0.40),
                (40.0, 40.0, 0.10),
            ],
        },
    }

    row = build_row_from_memory(
        players=players,
        map_state=map_state,
        round_num=12,
        map_name="de_dust2",
    )

    assert row is not None
    assert set(row) == set(FEATURE_NAMES)
    assert len(row) == 218

    assert row["t0_has_smoke"] == pytest.approx(1.0)
    assert row["t0_has_flash"] == pytest.approx(1.0)
    assert row["t0_has_he"] == pytest.approx(1.0)
    assert row["t0_has_molotov"] == pytest.approx(1.0)
    assert row["t0_has_c4"] == pytest.approx(1.0)
    assert row["t0_weapon_id"] == pytest.approx(float(WEAPON_ID_MAP["ak_47"]))
    assert row["t0_equip_value"] == pytest.approx(4100.0)

    assert row["ct0_has_smoke"] == pytest.approx(0.0)
    assert row["ct0_has_flash"] == pytest.approx(1.0)
    assert row["ct0_has_he"] == pytest.approx(0.0)
    assert row["ct0_has_molotov"] == pytest.approx(1.0)
    assert row["ct0_has_c4"] == pytest.approx(0.0)
    assert row["ct0_weapon_id"] == pytest.approx(float(WEAPON_ID_MAP["m4a1_s"]))
    assert row["ct0_equip_value"] == pytest.approx(3900.0)

    assert row["bomb_dropped"] == pytest.approx(1.0)
    assert row["bomb_x"] == pytest.approx(120.0)
    assert row["bomb_y"] == pytest.approx(240.0)

    assert row["smoke0_x"] == pytest.approx(200.0)
    assert row["smoke0_y"] == pytest.approx(200.0)
    assert row["smoke0_remain"] == pytest.approx(0.90)
    assert row["smoke1_remain"] == pytest.approx(0.80)
    assert row["smoke2_remain"] == pytest.approx(0.75)
    assert row["smoke3_remain"] == pytest.approx(0.50)
    assert row["smoke4_remain"] == pytest.approx(0.25)

    assert row["molotov0_x"] == pytest.approx(30.0)
    assert row["molotov0_y"] == pytest.approx(30.0)
    assert row["molotov0_remain"] == pytest.approx(0.90)
    assert row["molotov1_remain"] == pytest.approx(0.40)
    assert row["molotov2_remain"] == pytest.approx(0.20)

    assert row["map_de_dust2"] == pytest.approx(1.0)
    assert sum(row[name] for name in FEATURE_NAMES if name.startswith("map_")) == pytest.approx(1.0)


def test_build_row_from_memory_keeps_planted_bomb_coords_without_setting_dropped():
    players = [
        _player(name="alpha", team="T", x=1.0, y=2.0, z=0.0),
        _player(name="bravo", team="CT", x=-1.0, y=-2.0, z=0.0),
    ]
    map_state = {
        "ct_score": 2,
        "t_score": 3,
        "time_in_round": 60.0,
        "bomb": {"planted": True, "dropped": False, "x": 111.0, "y": 222.0, "site": "A"},
        "projectiles": {"smokes": [], "molotovs": []},
    }

    row = build_row_from_memory(
        players=players,
        map_state=map_state,
        round_num=6,
        map_name="de_dust2",
    )

    assert row is not None
    assert row["bomb_dropped"] == pytest.approx(0.0)
    assert row["bomb_x"] == pytest.approx(111.0)
    assert row["bomb_y"] == pytest.approx(222.0)
