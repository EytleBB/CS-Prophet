"""Unit and integration tests for demo_parser — demoparser2 is fully mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.parser.demo_parser import (
    DOWNSAMPLE,
    MAX_STEPS,
    POST_START_SECS,
    TARGET_RATE,
    TICK_RATE,
    _build_state_row,
    _extract_sequence,
    parse_demo,
)


# ── Test fixtures ─────────────────────────────────────────────────────────

def _make_tick_df(tick: int, n_t: int = 5, n_ct: int = 5) -> pd.DataFrame:
    """Minimal one-tick DataFrame matching demoparser2 column names."""
    rows = []
    for i in range(n_t):
        rows.append({
            "tick": tick, "name": f"t_p{i}", "team_name": "TERRORIST",
            "X": float(1000 + i * 10), "Y": float(500 + i * 5), "Z": 0.0,
            "health": 100, "armor_value": 100,
            "has_helmet": True, "is_alive": True,
        })
    for i in range(n_ct):
        rows.append({
            "tick": tick, "name": f"ct_p{i}", "team_name": "CT",
            "X": float(-500 + i * 10), "Y": float(200 + i * 5), "Z": 0.0,
            "health": 100, "armor_value": 100,
            "has_helmet": True, "is_alive": True,
        })
    return pd.DataFrame(rows)


def _make_multi_tick_df(ticks: list[int]) -> pd.DataFrame:
    return pd.concat([_make_tick_df(t) for t in ticks], ignore_index=True)


def _expected_ticks(round_start_tick: int, plant_tick: int) -> list[int]:
    """Compute expected tick list for the post-start window."""
    end_tick = min(round_start_tick + POST_START_SECS * TICK_RATE, plant_tick)
    return list(range(round_start_tick, end_tick + 1, DOWNSAMPLE))


def _make_mock_parser(
    plant_tick: int = 5120,
    round_start_tick: int = 3200,
    site: int = 0,
) -> MagicMock:
    """DemoParser mock for a single-round demo with one bomb plant."""
    ticks = _expected_ticks(round_start_tick, plant_tick)

    mock = MagicMock()

    def _parse_event(event_name, **kwargs):
        if event_name == "bomb_planted":
            return pd.DataFrame({"site": [site], "tick": [plant_tick]})
        if event_name == "round_end":
            return pd.DataFrame({"winner": pd.Series([], dtype=str), "tick": pd.Series([], dtype=int)})
        if event_name == "round_freeze_end":
            return pd.DataFrame({"tick": [round_start_tick]})
        return pd.DataFrame()

    mock.parse_event.side_effect = _parse_event
    mock.parse_ticks.return_value = _make_multi_tick_df(ticks)
    mock.parse_header.return_value = {"map_name": "de_mirage"}
    return mock


# ── Constants ─────────────────────────────────────────────────────────────

def test_constants_match_spec():
    assert TICK_RATE == 64
    assert TARGET_RATE == 8
    assert DOWNSAMPLE == 8          # 64 // 8
    assert POST_START_SECS == 90
    assert MAX_STEPS == 720         # 90 * 8


# ── _build_state_row ──────────────────────────────────────────────────────

class TestBuildStateRow:
    def _row(self, tick: int = 100, bomb_site: str = "A"):
        return _build_state_row(
            _make_tick_df(tick), step=0, tick=tick,
            round_num=1, bomb_site=bomb_site, map_name="de_mirage",
        )

    def test_all_player_keys_present(self):
        row = self._row()
        for side in ("t", "ct"):
            for i in range(5):
                for suffix in ("_x", "_y", "_z", "_hp", "_armor", "_helmet", "_alive", "_role"):
                    assert f"{side}{i}{suffix}" in row

    def test_bomb_site_stored(self):
        assert self._row(bomb_site="B")["bomb_site"] == "B"

    def test_step_and_tick_stored(self):
        row = _build_state_row(
            _make_tick_df(500), step=7, tick=500,
            round_num=3, bomb_site="A", map_name="de_mirage",
        )
        assert row["step"] == 7
        assert row["tick"] == 500
        assert row["round_num"] == 3

    def test_normalized_coords_in_unit_cube(self):
        row = self._row()
        for side in ("t", "ct"):
            for i in range(5):
                for axis in ("_x", "_y", "_z"):
                    val = row[f"{side}{i}{axis}"]
                    assert 0.0 <= val <= 1.0, f"{side}{i}{axis} = {val}"

    def test_missing_players_zero_padded(self):
        slim = _make_tick_df(100, n_t=3, n_ct=5)
        row = _build_state_row(slim, step=0, tick=100,
                               round_num=1, bomb_site="A", map_name="de_mirage")
        assert row["t3_hp"] == 0
        assert row["t4_x"] == 0.0
        assert row["t4_alive"] is False
        assert row["t4_role"] == ""

    def test_role_written_when_player_roles_provided(self):
        tick_df = _make_tick_df(100)
        roles = {"t_p0": "AWPer", "ct_p1": "IGL"}
        row = _build_state_row(tick_df, step=0, tick=100,
                               round_num=1, bomb_site="A", map_name="de_mirage",
                               player_roles=roles)
        assert row["t0_role"] == "AWPer"
        assert row["ct1_role"] == "IGL"
        assert row["t1_role"] == ""   # not in roles dict

    def test_role_empty_when_no_player_roles(self):
        row = _build_state_row(
            _make_tick_df(100), step=0, tick=100,
            round_num=1, bomb_site="A", map_name="de_mirage",
        )
        for side in ("t", "ct"):
            for i in range(5):
                assert row[f"{side}{i}_role"] == ""

    def test_map_zone_key_absent(self):
        assert "map_zone" not in self._row()


# ── _extract_sequence ─────────────────────────────────────────────────────

class TestExtractSequence:
    def _mock_parser(self, round_start_tick: int = 0, plant_tick: int = 5120) -> MagicMock:
        ticks = _expected_ticks(round_start_tick, plant_tick)
        mock = MagicMock()
        mock.parse_ticks.return_value = _make_multi_tick_df(ticks)
        return mock

    def test_returns_dataframe(self):
        mock = self._mock_parser()
        result = _extract_sequence(mock, round_num=1, round_start_tick=0,
                                   plant_tick=5120,
                                   bomb_site="A", map_name="de_mirage")
        assert isinstance(result, pd.DataFrame)

    def test_step_count_at_most_max_steps_plus_one(self):
        mock = self._mock_parser()
        result = _extract_sequence(mock, round_num=1, round_start_tick=0,
                                   plant_tick=5120,
                                   bomb_site="A", map_name="de_mirage")
        assert len(result) <= MAX_STEPS + 1

    def test_steps_are_sequential_from_zero(self):
        mock = self._mock_parser()
        result = _extract_sequence(mock, round_num=1, round_start_tick=0,
                                   plant_tick=5120,
                                   bomb_site="A", map_name="de_mirage")
        steps = result["step"].tolist()
        assert steps == list(range(len(steps)))

    def test_bomb_site_column_is_consistent(self):
        mock = self._mock_parser()
        result = _extract_sequence(mock, round_num=1, round_start_tick=0,
                                   plant_tick=5120,
                                   bomb_site="B", map_name="de_mirage")
        assert (result["bomb_site"] == "B").all()

    def test_returns_none_on_empty_tick_df(self):
        mock = MagicMock()
        mock.parse_ticks.return_value = pd.DataFrame()
        result = _extract_sequence(mock, round_num=1, round_start_tick=0,
                                   plant_tick=1280,
                                   bomb_site="A", map_name="de_mirage")
        assert result is None

    def test_parse_ticks_called_with_correct_ticks(self):
        round_start_tick = 0
        plant_tick = 5120
        expected = _expected_ticks(round_start_tick, plant_tick)

        mock = self._mock_parser(round_start_tick, plant_tick)
        _extract_sequence(mock, round_num=1, round_start_tick=round_start_tick,
                          plant_tick=plant_tick,
                          bomb_site="A", map_name="de_mirage")

        called_ticks = mock.parse_ticks.call_args.kwargs["ticks"]
        assert called_ticks == expected

    def test_truncated_when_plant_before_30s(self):
        # Plant at round_start + 10s → only ~80 steps, not 240
        rs = 1000
        plant = rs + 10 * TICK_RATE  # 10 seconds after start
        ticks = _expected_ticks(rs, plant)
        mock = MagicMock()
        mock.parse_ticks.return_value = _make_multi_tick_df(ticks)
        result = _extract_sequence(mock, round_num=1, round_start_tick=rs,
                                   plant_tick=plant,
                                   bomb_site="A", map_name="de_mirage")
        assert result is not None
        assert len(result) < MAX_STEPS


# ── parse_demo integration ─────────────────────────────────────────────────

class TestParseDemoIntegration:
    def test_writes_parquet_file(self, tmp_path: Path):
        mock = _make_mock_parser()
        dem = tmp_path / "match.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        assert out is not None
        assert out.exists()
        assert out.suffix == ".parquet"

    def test_output_contains_demo_name_column(self, tmp_path: Path):
        mock = _make_mock_parser()
        dem = tmp_path / "match_xyz.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        df = pd.read_parquet(out)
        assert "demo_name" in df.columns
        assert (df["demo_name"] == "match_xyz").all()

    def test_site_0_produces_A_labels(self, tmp_path: Path):
        mock = _make_mock_parser(site=0)
        dem = tmp_path / "a_site.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        df = pd.read_parquet(out)
        assert (df["bomb_site"] == "A").all()

    def test_site_1_produces_B_labels(self, tmp_path: Path):
        rs = 3200
        plant_tick = 5120
        ticks = _expected_ticks(rs, plant_tick)
        mock = MagicMock()

        def _parse_event(e, **kw):
            if e == "bomb_planted":
                return pd.DataFrame({"site": [1], "tick": [plant_tick]})
            if e == "round_freeze_end":
                return pd.DataFrame({"tick": [rs]})
            return pd.DataFrame({"winner": pd.Series([], dtype=str), "tick": pd.Series([], dtype=int)})

        mock.parse_event.side_effect = _parse_event
        mock.parse_ticks.return_value = _make_multi_tick_df(ticks)
        mock.parse_header.return_value = {"map_name": "de_inferno"}
        dem = tmp_path / "b_site.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        df = pd.read_parquet(out)
        assert (df["bomb_site"] == "B").all()

    def test_returns_none_when_no_plant_events(self, tmp_path: Path):
        mock = MagicMock()
        mock.parse_event.return_value = pd.DataFrame()  # empty for any event
        mock.parse_header.return_value = {"map_name": "de_mirage"}
        dem = tmp_path / "no_plants.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        assert out is None

    def test_output_dir_created_if_missing(self, tmp_path: Path):
        mock = _make_mock_parser()
        dem = tmp_path / "match.dem"
        dem.touch()
        new_dir = tmp_path / "brand_new_subdir"

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, new_dir)

        assert new_dir.exists()
        assert out is not None

    def test_output_has_expected_columns(self, tmp_path: Path):
        mock = _make_mock_parser()
        dem = tmp_path / "match.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        df = pd.read_parquet(out)
        for col in ("demo_name", "round_num", "step", "tick", "bomb_site"):
            assert col in df.columns, f"Missing column: {col}"
        for side in ("t", "ct"):
            for i in range(5):
                assert f"{side}{i}_x" in df.columns
                assert f"{side}{i}_hp" in df.columns
                assert f"{side}{i}_role" in df.columns

    def test_player_roles_written_to_parquet(self, tmp_path: Path):
        mock = _make_mock_parser()
        dem = tmp_path / "match.dem"
        dem.touch()
        roles = {"t_p0": "AWPer", "ct_p0": "IGL"}

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out", player_roles=roles)

        df = pd.read_parquet(out)
        assert (df["t0_role"] == "AWPer").all()
        assert (df["ct0_role"] == "IGL").all()
        assert (df["t1_role"] == "").all()

    def test_multi_round_demo(self, tmp_path: Path):
        rs_1, plant_1 = 1000, 5120
        rs_2, plant_2 = 6000, 10000
        ticks_1 = _expected_ticks(rs_1, plant_1)
        ticks_2 = _expected_ticks(rs_2, plant_2)

        mock = MagicMock()

        def _parse_event(e, **kw):
            if e == "bomb_planted":
                return pd.DataFrame({"site": [0, 1], "tick": [plant_1, plant_2]})
            if e == "round_freeze_end":
                return pd.DataFrame({"tick": [rs_1, rs_2]})
            return pd.DataFrame({"winner": pd.Series([], dtype=str), "tick": pd.Series([], dtype=int)})

        mock.parse_event.side_effect = _parse_event
        mock.parse_ticks.side_effect = [
            _make_multi_tick_df(ticks_1),
            _make_multi_tick_df(ticks_2),
        ]
        mock.parse_header.return_value = {"map_name": "de_mirage"}
        dem = tmp_path / "two_rounds.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        df = pd.read_parquet(out)
        assert df["round_num"].nunique() == 2
        assert set(df["bomb_site"].unique()) == {"A", "B"}
