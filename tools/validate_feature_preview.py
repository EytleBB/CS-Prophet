"""Validate the legacy 348-dim exploratory feature schema against extracted demo data.

This validator is retained for historical comparison only. The active v2
training/inference pipeline now uses the 218-dim realtime-aligned schema.

Usage:
    python tools/validate_feature_preview.py viz/2389983_de_dust2_full.pkl --all
    python tools/validate_feature_preview.py viz/2389983_de_dust2_full.pkl --sample 2048
"""
from __future__ import annotations

import argparse
import math
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.demo_feature_preview import (  # noqa: E402
    MAPS,
    MOLOTOV_DURATION_TICKS,
    MOLOTOV_SLOTS,
    SMOKE_DURATION_TICKS,
    SMOKE_SLOTS,
    WEAPON_CATS,
    best_weapon,
    build_feature_vector,
    has_c4,
    map_onehot,
    weapon_onehot,
)
from src.utils.paths import resolve_path_input  # noqa: E402

EXPECTED_DIM = 348


class CheckRecorder:
    """Track validation pass/fail counts with a few examples per check."""

    def __init__(self, max_examples: int = 5) -> None:
        self.max_examples = max_examples
        self.totals: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.examples: dict[str, list[dict]] = defaultdict(list)

    def check(self, name: str, ok: bool, context: dict | None = None) -> None:
        self.totals[name] += 1
        if ok:
            return
        self.failures[name] += 1
        if context is not None and len(self.examples[name]) < self.max_examples:
            self.examples[name].append(context)

    @property
    def failed(self) -> bool:
        return any(self.failures.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate extracted feature vectors.")
    parser.add_argument("pkl_path", help="Path to extracted *_full.pkl file")
    parser.add_argument("--sample", type=int, default=2048, help="Number of unique ticks to validate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--all", action="store_true", help="Validate every unique tick instead of sampling")
    return parser.parse_args()


def _inventory_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    return []


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def _safe_float(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _safe_int(value, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(value)


def _feature_dim(feat: dict) -> int:
    dim = 0
    for side in ("t", "ct"):
        for i in range(5):
            dim += 25 + len(feat[f"{side}{i}_weapon_cat"])
    dim += 4
    dim += 3
    dim += SMOKE_SLOTS * 3
    dim += MOLOTOV_SLOTS * 3
    dim += len(feat["map_onehot"])
    return dim


def _expected_bomb_state(
    events: dict,
    round_info: pd.DataFrame,
    current_tick: int,
    current_round: int,
) -> tuple[bool, float, float]:
    drop_df = events.get("bomb_dropped", pd.DataFrame())
    pick_df = events.get("bomb_pickup", pd.DataFrame())
    ri = round_info[round_info["round_num"] == current_round]
    round_start = int(ri["freeze_tick"].iloc[0]) if not ri.empty else 0
    round_end = int(ri["end_tick"].iloc[0]) if not ri.empty and pd.notna(ri["end_tick"].iloc[0]) else current_tick

    timeline: list[tuple[int, int, str, pd.Series]] = []
    if not drop_df.empty and "tick" in drop_df.columns:
        rnd = drop_df[(drop_df["tick"] >= round_start) & (drop_df["tick"] <= round_end)]
        for _, row in rnd.iterrows():
            tick = int(row["tick"])
            if tick <= current_tick:
                timeline.append((tick, 0, "drop", row))
    if not pick_df.empty and "tick" in pick_df.columns:
        rnd = pick_df[(pick_df["tick"] >= round_start) & (pick_df["tick"] <= round_end)]
        for _, row in rnd.iterrows():
            tick = int(row["tick"])
            if tick <= current_tick:
                timeline.append((tick, 1, "pick", row))

    if not timeline:
        return False, 0.0, 0.0

    timeline.sort(key=lambda x: (x[0], x[1]))
    tick, _, event_type, row = timeline[-1]
    if event_type == "pick":
        return False, 0.0, 0.0

    x = _safe_float(row["X"]) if "X" in row.index and pd.notna(row["X"]) else 0.0
    y = _safe_float(row["Y"]) if "Y" in row.index and pd.notna(row["Y"]) else 0.0
    return True, x, y


def _expected_active_utils(
    events: dict,
    current_tick: int,
    current_round: int,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    smokes: list[tuple[float, float, float]] = []
    molotovs: list[tuple[float, float, float]] = []

    smoke_df = events.get("smokegrenade_detonate", pd.DataFrame())
    if not smoke_df.empty and "round_num" in smoke_df.columns:
        rnd = smoke_df[smoke_df["round_num"] == current_round]
        for _, row in rnd.iterrows():
            det_tick = int(row["tick"])
            elapsed = current_tick - det_tick
            if 0 <= elapsed < SMOKE_DURATION_TICKS:
                remain = 1.0 - elapsed / SMOKE_DURATION_TICKS
                smokes.append((float(row["x"]), float(row["y"]), remain))

    inferno_df = events.get("inferno_startburn", pd.DataFrame())
    if not inferno_df.empty and "round_num" in inferno_df.columns:
        rnd = inferno_df[inferno_df["round_num"] == current_round]
        for _, row in rnd.iterrows():
            det_tick = int(row["tick"])
            elapsed = current_tick - det_tick
            if 0 <= elapsed < MOLOTOV_DURATION_TICKS:
                remain = 1.0 - elapsed / MOLOTOV_DURATION_TICKS
                molotovs.append((float(row["x"]), float(row["y"]), remain))

    return smokes, molotovs


def _validate_dataset_structure(
    tick_df: pd.DataFrame,
    events: dict,
    round_info: pd.DataFrame,
    map_name: str,
    recorder: CheckRecorder,
) -> None:
    recorder.check("map_supported", map_name in MAPS, {"map_name": map_name})
    recorder.check("round_info_non_empty", not round_info.empty, {"round_rows": len(round_info)})
    recorder.check(
        "round_info_monotonic",
        round_info["freeze_tick"].dropna().is_monotonic_increasing,
        {"freeze_ticks_head": round_info["freeze_tick"].head(10).tolist()},
    )

    group_sizes = tick_df.groupby(["round_num", "tick"]).size()
    bad_group_sizes = group_sizes[group_sizes != 10]
    recorder.check(
        "tick_slice_has_10_players",
        bad_group_sizes.empty,
        {"bad_groups": bad_group_sizes.head(5).to_dict(), "count": int(len(bad_group_sizes))},
    )

    side_counts = tick_df.groupby(["round_num", "tick", "team_name"]).size().unstack(fill_value=0)
    bad_sides = side_counts[(side_counts.get("TERRORIST", 0) != 5) | (side_counts.get("CT", 0) != 5)]
    recorder.check(
        "tick_slice_has_5v5_split",
        bad_sides.empty,
        {"bad_groups": bad_sides.head(5).to_dict("index"), "count": int(len(bad_sides))},
    )

    for event_name in ("bomb_dropped", "bomb_pickup", "bomb_planted"):
        df = events.get(event_name, pd.DataFrame())
        recorder.check(
            f"{event_name}_has_xyz_columns",
            {"X", "Y", "Z"}.issubset(df.columns),
            {"columns": list(df.columns)},
        )
        if not df.empty and {"X", "Y", "Z"}.issubset(df.columns):
            missing_xyz = df[["X", "Y", "Z"]].isna().any(axis=1)
            recorder.check(
                f"{event_name}_xyz_complete",
                not missing_xyz.any(),
                {"missing_rows": df.loc[missing_xyz, ["tick", "user_name"]].head(5).to_dict("records")},
            )


def _validate_player_slot(
    feat: dict,
    prefix: str,
    player_row: pd.Series,
    tick: int,
    round_num: int,
    recorder: CheckRecorder,
) -> None:
    inv = _inventory_list(player_row.get("inventory", []))
    alive = bool(player_row.get("is_alive", False))
    expected_best_weapon = best_weapon(inv)
    expected_weapon_cat = weapon_onehot(expected_best_weapon)

    expected_scalars = {
        "name": str(player_row.get("name", "")),
        "x": _safe_float(player_row.get("X", 0.0)),
        "y": _safe_float(player_row.get("Y", 0.0)),
        "z": _safe_float(player_row.get("Z", 0.0)),
        "vx": _safe_float(player_row.get("velocity_X", 0.0)) if alive else 0.0,
        "vy": _safe_float(player_row.get("velocity_Y", 0.0)) if alive else 0.0,
        "vz": _safe_float(player_row.get("velocity_Z", 0.0)) if alive else 0.0,
        "yaw": _safe_float(player_row.get("yaw", 0.0)) if alive else 0.0,
        "is_walking": bool(player_row.get("is_walking", False)) if alive else False,
        "ducking": bool(player_row.get("ducking", False)) if alive else False,
        "in_bomb_zone": bool(player_row.get("in_bomb_zone", False)) if alive else False,
        "hp": _safe_int(player_row.get("health", 0)),
        "armor": _safe_int(player_row.get("armor_value", 0)),
        "helmet": bool(player_row.get("has_helmet", False)),
        "alive": alive,
        "spotted": bool(player_row.get("spotted", False)) if alive else False,
        "is_scoped": bool(player_row.get("is_scoped", False)) if alive else False,
        "flash_dur": _safe_float(player_row.get("flash_duration", 0.0)),
        "has_smoke": "Smoke Grenade" in inv,
        "has_flash": "Flashbang" in inv,
        "has_he": any(x in inv for x in ("HE Grenade", "High Explosive Grenade")),
        "has_molotov": any(x in inv for x in ("Molotov", "Incendiary Grenade")),
        "has_c4": has_c4(inv),
        "balance": _safe_int(player_row.get("balance", 0)),
        "equip_value": _safe_int(player_row.get("current_equip_value", 0)),
        "score": _safe_int(player_row.get("score", 0)),
    }

    recorder.check(
        "player_best_weapon",
        feat[f"{prefix}_best_weapon"] == expected_best_weapon,
        {
            "tick": tick,
            "round": round_num,
            "player": expected_scalars["name"],
            "got": feat[f"{prefix}_best_weapon"],
            "expected": expected_best_weapon,
            "inventory": inv,
        },
    )
    recorder.check(
        "player_weapon_onehot",
        feat[f"{prefix}_weapon_cat"] == expected_weapon_cat,
        {
            "tick": tick,
            "round": round_num,
            "player": expected_scalars["name"],
            "got": feat[f"{prefix}_weapon_cat"],
            "expected": expected_weapon_cat,
        },
    )
    recorder.check(
        "player_weapon_onehot_sum",
        sum(feat[f"{prefix}_weapon_cat"]) == 1.0,
        {
            "tick": tick,
            "round": round_num,
            "player": expected_scalars["name"],
            "vector": feat[f"{prefix}_weapon_cat"],
        },
    )

    for field, expected in expected_scalars.items():
        actual = feat[f"{prefix}_{field}"]
        if isinstance(expected, float):
            ok = _close(actual, expected)
        else:
            ok = actual == expected
        recorder.check(
            f"player_{field}",
            ok,
            {
                "tick": tick,
                "round": round_num,
                "player": expected_scalars["name"],
                "field": field,
                "got": actual,
                "expected": expected,
            },
        )


def _validate_feature_tick(
    tick_df: pd.DataFrame,
    tick_slice: pd.DataFrame,
    events: dict,
    round_info: pd.DataFrame,
    map_name: str,
    round_num: int,
    tick: int,
    recorder: CheckRecorder,
) -> None:
    feat = build_feature_vector(tick_df, tick_slice, events, tick, round_num, map_name, round_info)

    recorder.check(
        "feature_dim",
        _feature_dim(feat) == EXPECTED_DIM,
        {"tick": tick, "round": round_num, "dim": _feature_dim(feat)},
    )
    recorder.check(
        "map_onehot_value",
        feat["map_onehot"] == map_onehot(map_name),
        {"tick": tick, "round": round_num, "got": feat["map_onehot"], "expected": map_onehot(map_name)},
    )
    recorder.check(
        "map_onehot_sum",
        sum(feat["map_onehot"]) == 1.0,
        {"tick": tick, "round": round_num, "vector": feat["map_onehot"]},
    )

    ri = round_info[round_info["round_num"] == round_num].iloc[0]
    expected_time_in_round = round((tick - _safe_int(ri["freeze_tick"])) / 64.0, 2)
    recorder.check("global_ct_score", feat["ct_score"] == _safe_int(ri["ct_score"]), {"tick": tick, "round": round_num})
    recorder.check("global_t_score", feat["t_score"] == _safe_int(ri["t_score"]), {"tick": tick, "round": round_num})
    recorder.check("global_round_num", feat["round_num"] == round_num, {"tick": tick, "round": round_num})
    recorder.check(
        "global_time_in_round",
        _close(feat["time_in_round"], expected_time_in_round),
        {"tick": tick, "round": round_num, "got": feat["time_in_round"], "expected": expected_time_in_round},
    )

    expected_bomb = _expected_bomb_state(events, round_info, tick, round_num)
    recorder.check(
        "bomb_dropped",
        feat["bomb_dropped"] == expected_bomb[0],
        {"tick": tick, "round": round_num, "got": feat["bomb_dropped"], "expected": expected_bomb[0]},
    )
    recorder.check(
        "bomb_x",
        _close(feat["bomb_x"], expected_bomb[1]),
        {"tick": tick, "round": round_num, "got": feat["bomb_x"], "expected": expected_bomb[1]},
    )
    recorder.check(
        "bomb_y",
        _close(feat["bomb_y"], expected_bomb[2]),
        {"tick": tick, "round": round_num, "got": feat["bomb_y"], "expected": expected_bomb[2]},
    )

    expected_smokes, expected_molotovs = _expected_active_utils(events, tick, round_num)
    for slot in range(SMOKE_SLOTS):
        exp = expected_smokes[slot] if slot < len(expected_smokes) else (0.0, 0.0, 0.0)
        recorder.check(
            "smoke_slot_x",
            _close(feat[f"smoke{slot}_x"], exp[0]),
            {"tick": tick, "round": round_num, "slot": slot, "got": feat[f"smoke{slot}_x"], "expected": exp[0]},
        )
        recorder.check(
            "smoke_slot_y",
            _close(feat[f"smoke{slot}_y"], exp[1]),
            {"tick": tick, "round": round_num, "slot": slot, "got": feat[f"smoke{slot}_y"], "expected": exp[1]},
        )
        recorder.check(
            "smoke_slot_remain",
            _close(feat[f"smoke{slot}_remain"], exp[2]),
            {
                "tick": tick,
                "round": round_num,
                "slot": slot,
                "got": feat[f"smoke{slot}_remain"],
                "expected": exp[2],
            },
        )
        recorder.check(
            "smoke_slot_range",
            0.0 <= feat[f"smoke{slot}_remain"] <= 1.0,
            {"tick": tick, "round": round_num, "slot": slot, "remain": feat[f"smoke{slot}_remain"]},
        )

    for slot in range(MOLOTOV_SLOTS):
        exp = expected_molotovs[slot] if slot < len(expected_molotovs) else (0.0, 0.0, 0.0)
        recorder.check(
            "molotov_slot_x",
            _close(feat[f"molotov{slot}_x"], exp[0]),
            {"tick": tick, "round": round_num, "slot": slot, "got": feat[f"molotov{slot}_x"], "expected": exp[0]},
        )
        recorder.check(
            "molotov_slot_y",
            _close(feat[f"molotov{slot}_y"], exp[1]),
            {"tick": tick, "round": round_num, "slot": slot, "got": feat[f"molotov{slot}_y"], "expected": exp[1]},
        )
        recorder.check(
            "molotov_slot_remain",
            _close(feat[f"molotov{slot}_remain"], exp[2]),
            {
                "tick": tick,
                "round": round_num,
                "slot": slot,
                "got": feat[f"molotov{slot}_remain"],
                "expected": exp[2],
            },
        )
        recorder.check(
            "molotov_slot_range",
            0.0 <= feat[f"molotov{slot}_remain"] <= 1.0,
            {"tick": tick, "round": round_num, "slot": slot, "remain": feat[f"molotov{slot}_remain"]},
        )

    for side_name, team_name in (("t", "TERRORIST"), ("ct", "CT")):
        side_rows = tick_slice[tick_slice["team_name"] == team_name].sort_values("name").reset_index(drop=True)
        for idx in range(5):
            prefix = f"{side_name}{idx}"
            if idx < len(side_rows):
                _validate_player_slot(feat, prefix, side_rows.iloc[idx], tick, round_num, recorder)
            else:
                recorder.check(
                    "no_empty_slots_expected",
                    False,
                    {"tick": tick, "round": round_num, "missing_slot": prefix, "team": team_name},
                )


def _select_ticks(tick_df: pd.DataFrame, sample: int, seed: int, validate_all: bool) -> pd.DataFrame:
    unique_ticks = tick_df[tick_df["round_num"].notna()][["round_num", "tick"]].drop_duplicates()
    if validate_all or sample >= len(unique_ticks):
        return unique_ticks.sort_values(["round_num", "tick"]).reset_index(drop=True)
    return unique_ticks.sample(sample, random_state=seed).sort_values(["round_num", "tick"]).reset_index(drop=True)


def _print_summary(recorder: CheckRecorder, validated_ticks: int, total_ticks: int, elapsed_s: float) -> None:
    print(f"\nValidated {validated_ticks}/{total_ticks} unique ticks in {elapsed_s:.2f}s")
    print(f"Checks run: {sum(recorder.totals.values())}")
    print(f"Failed checks: {sum(recorder.failures.values())}")

    if not recorder.failed:
        print("All validation checks passed.")
        return

    print("\nFailures by check:")
    for name, failed in recorder.failures.most_common():
        total = recorder.totals[name]
        print(f"  - {name}: {failed}/{total}")
        for example in recorder.examples[name]:
            print(f"      example: {example}")


def main() -> int:
    args = parse_args()
    pkl_path = resolve_path_input(args.pkl_path)
    start = time.perf_counter()

    with pkl_path.open("rb") as handle:
        data = pickle.load(handle)

    tick_df = data["tick_df"]
    events = data["events"]
    round_info = data["round_info"]
    header = data["header"]
    map_name = header.get("map_name", "unknown")

    recorder = CheckRecorder()
    _validate_dataset_structure(tick_df, events, round_info, map_name, recorder)

    selected = _select_ticks(tick_df, args.sample, args.seed, args.all)
    total_unique_ticks = len(tick_df[["round_num", "tick"]].drop_duplicates())

    for _, row in selected.iterrows():
        round_num = int(row["round_num"])
        tick = int(row["tick"])
        tick_slice = tick_df[(tick_df["round_num"] == round_num) & (tick_df["tick"] == tick)]
        _validate_feature_tick(tick_df, tick_slice, events, round_info, map_name, round_num, tick, recorder)

    elapsed = time.perf_counter() - start
    _print_summary(recorder, len(selected), total_unique_ticks, elapsed)
    return 1 if recorder.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
