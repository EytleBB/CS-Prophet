"""Tests for processed_v2 export and dataset loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features.dataset_v2 import RoundSequenceDatasetV2
from src.features.processed_v2 import (
    METADATA_COLUMNS,
    build_processed_frame_v2,
    export_full_pkl_to_processed_v2,
)
from src.features.state_vector_v2 import FEATURE_DIM, FEATURE_NAMES, WEAPON_ID_MAP


def _player_row(
    name: str,
    team_name: str,
    tick: int,
    round_num: int,
    inventory: list[str],
    **overrides,
) -> dict:
    row = {
        "round_num": round_num,
        "tick": tick,
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
    rows: list[dict] = []
    for tick in (200, 208):
        t_rows = [
            _player_row("alpha", "TERRORIST", tick, 1, ["AK-47", "C4 Explosive", "Smoke Grenade"]),
            _player_row("bravo", "TERRORIST", tick, 1, ["Galil AR", "Flashbang"], X=1100.0),
            _player_row("charlie", "TERRORIST", tick, 1, ["Desert Eagle", "Molotov"], X=1200.0),
            _player_row("delta", "TERRORIST", tick, 1, ["Tec-9"], X=1300.0),
            _player_row("echo", "TERRORIST", tick, 1, [], X=1400.0),
        ]
        ct_rows = [
            _player_row("foxtrot", "CT", tick, 1, ["M4A1-S"], X=-1200.0),
            _player_row("golf", "CT", tick, 1, ["AWP"], X=-1300.0),
            _player_row("hotel", "CT", tick, 1, ["MP9"], X=-1400.0),
            _player_row("india", "CT", tick, 1, ["USP-S"], X=-1500.0),
            _player_row("juliet", "CT", tick, 1, [], X=-1600.0),
        ]
        rows.extend(t_rows + ct_rows)
    return pd.DataFrame(rows)


def _events() -> dict:
    return {
        "bomb_dropped": pd.DataFrame(
            [{"tick": 198, "X": 900.0, "Y": 2200.0, "user_name": "alpha", "round_num": 1}]
        ),
        "bomb_pickup": pd.DataFrame(columns=["tick", "X", "Y", "user_name", "round_num"]),
        "smokegrenade_detonate": pd.DataFrame(
            [{"tick": 196, "x": 800.0, "y": 2100.0, "round_num": 1}]
        ),
        "inferno_startburn": pd.DataFrame(
            [{"tick": 190, "x": -1000.0, "y": 2400.0, "round_num": 1}]
        ),
        "bomb_planted": pd.DataFrame(
            [{"tick": 220, "site": 504, "X": 1300.0, "Y": 2200.0, "Z": 0.0, "round_num": 1}]
        ),
    }


def _round_info() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "round_num": 1,
                "freeze_tick": 160,
                "plant_tick": 220,
                "end_tick": 240,
                "bomb_site": pd.NA,
                "ct_score": 3,
                "t_score": 1,
            }
        ]
    )


def _payload() -> dict:
    return {
        "header": {"map_name": "de_dust2"},
        "tick_df": _tick_df(),
        "events": _events(),
        "round_info": _round_info(),
    }


def test_build_processed_frame_v2_returns_expected_columns_and_rows():
    df = build_processed_frame_v2(_payload(), demo_name="demo_a")

    assert list(df.columns[: len(METADATA_COLUMNS)]) == list(METADATA_COLUMNS)
    assert all(name in df.columns for name in FEATURE_NAMES)
    assert "t0_vx" not in df.columns
    assert len(df) == 2
    assert df["demo_name"].unique().tolist() == ["demo_a"]
    assert df["bomb_site"].unique().tolist() == ["A"]
    assert df["step"].tolist() == [0, 1]
    assert df["tick"].tolist() == [200, 208]
    assert df["round_num"].tolist() == [1, 1]
    assert df["t0_weapon_id"].iloc[0] == pytest.approx(float(WEAPON_ID_MAP["ak_47"]))


def test_export_full_pkl_to_processed_v2_writes_parquet(monkeypatch: pytest.MonkeyPatch):
    written: dict[str, object] = {}

    def fake_load_full_payload(_: Path) -> dict:
        return _payload()

    def fake_mkdir(self, parents=False, exist_ok=False):  # noqa: ANN001
        return None

    def fake_to_parquet(self, path, index=False):  # noqa: ANN001
        written["path"] = path
        written["index"] = index
        written["df"] = self.copy()

    monkeypatch.setattr("src.features.processed_v2.load_full_payload", fake_load_full_payload)
    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    out_path = export_full_pkl_to_processed_v2(Path("demo_a_full.pkl"), Path("processed_v2"))

    assert out_path == Path("processed_v2") / "demo_a.parquet"
    assert written["path"] == out_path
    assert written["index"] is False
    assert len(written["df"]) == 2
    assert written["df"]["demo_name"].iloc[0] == "demo_a"


def test_round_sequence_dataset_v2_loads_and_pads(monkeypatch: pytest.MonkeyPatch):
    rows: list[dict[str, object]] = []
    for round_num, label in ((1, "A"), (2, "B")):
        for step in range(3):
            row = {name: 0.0 for name in FEATURE_NAMES}
            row["map_de_dust2"] = 1.0
            row["round_num"] = round_num
            row["time_in_round"] = float(step)
            row["t0_hp"] = 100.0 - step
            row["t0_weapon_id"] = float(WEAPON_ID_MAP["ak_47"])
            rows.append(
                {
                    "demo_name": "demo_a",
                    "map_name": "de_dust2",
                    "bomb_site": label,
                    "step": step,
                    "tick": 200 + step,
                    **row,
                }
            )

    df = pd.DataFrame(rows)

    def fake_read_parquet(*args, **kwargs) -> pd.DataFrame:  # noqa: ANN002, ANN003
        columns = kwargs.get("columns")
        if columns:
            return df.loc[:, columns].copy()
        return df.copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    ds = RoundSequenceDatasetV2([Path("demo_a.parquet")], sequence_length=5, training=False)
    assert len(ds) == 2

    x0, y0 = ds[0]
    x1, y1 = ds[1]
    assert tuple(x0.shape) == (5, FEATURE_DIM)
    assert tuple(x1.shape) == (5, FEATURE_DIM)
    assert int(y0.item()) == 0
    assert int(y1.item()) == 1
    assert x0[0].dtype.is_floating_point
    assert x0[3:].abs().sum().item() == pytest.approx(0.0)
