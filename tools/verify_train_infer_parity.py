#!/usr/bin/env python3
"""Compare offline/training and online/GSI feature extraction on the same ticks."""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.feature_builder_v2 import WEAPON_NAME_TO_ID, build_feature_row_v2, get_active_utils
from src.features.processed_v2 import load_full_payload
from src.features.state_vector_v2 import FEATURE_DIM, FEATURE_NAMES, MAPS, build_state_vector
from src.inference.gsi_state_builder import _GSI_TO_CANONICAL_WEAPON, build_row_from_gsi
from src.utils.paths import data_root, resolve_path_input

TOL = 1e-6
LIVE_ROUND_SECONDS = 115.0
SMOKE_DURATION_SECONDS = 18.0
MOLOTOV_DURATION_SECONDS = 7.0
GRENADE_NAME_MAP = {
    "Smoke Grenade": "weapon_smokegrenade",
    "Flashbang": "weapon_flashbang",
    "HE Grenade": "weapon_hegrenade",
    "High Explosive Grenade": "weapon_hegrenade",
    "Molotov": "weapon_molotov",
    "Incendiary Grenade": "weapon_incgrenade",
    "C4 Explosive": "weapon_c4",
}
CANONICAL_TO_GSI: dict[str, str] = {}
for gsi_name, canonical in _GSI_TO_CANONICAL_WEAPON.items():
    CANONICAL_TO_GSI.setdefault(canonical, gsi_name)


def _fmt(value: float) -> str: return f"{value:.6f}"
def _vec3(x: float, y: float, z: float) -> str: return f"{x:.6f}, {y:.6f}, {z:.6f}"
def _safe_float(value, default: float = 0.0) -> float: return default if pd.isna(value) else float(value)
def _safe_int(value, default: int = 0) -> int: return default if pd.isna(value) else int(value)
def _safe_bool(value) -> bool: return False if pd.isna(value) else bool(value)


def _inventory_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    return []


def _resolve_pkl(user_path: str | None) -> Path:
    if user_path:
        path = resolve_path_input(user_path)
        if not path.exists():
            raise FileNotFoundError(f"pkl not found: {path}")
        return path
    preferred = data_root() / "viz" / "2389983_de_dust2_full.pkl"
    if preferred.exists():
        return preferred
    candidates = sorted((data_root() / "viz").glob("*_full.pkl"))
    if not candidates:
        raise FileNotFoundError(f"no *_full.pkl found under {(data_root() / 'viz')}")
    return candidates[0]


def _infer_map_name(pkl_path: Path, override: str | None, header: dict) -> str:
    if override:
        return override
    stem = pkl_path.stem[:-5] if pkl_path.stem.endswith("_full") else pkl_path.stem
    for candidate in MAPS:
        if stem.endswith(candidate):
            return candidate
    return str(header.get("map_name", "unknown"))


def _round_row(round_info: pd.DataFrame, round_num: int) -> pd.Series | None:
    rows = round_info[round_info["round_num"] == round_num]
    return None if rows.empty else rows.iloc[0]


def _item_to_gsi_name(item: str) -> str | None:
    if item in GRENADE_NAME_MAP:
        return GRENADE_NAME_MAP[item]
    canonical = WEAPON_NAME_TO_ID.get(item)
    return None if canonical is None else CANONICAL_TO_GSI.get(canonical)


def _weapon_payload(inventory: list[str], active_weapon_name: str | None) -> dict[str, dict[str, str]]:
    active_gsi = _item_to_gsi_name(str(active_weapon_name)) if active_weapon_name else None
    weapons, active_assigned = {}, False
    for idx, item in enumerate(inventory):
        gsi_name = _item_to_gsi_name(item)
        if gsi_name is None:
            continue
        state = "active" if active_gsi and not active_assigned and gsi_name == active_gsi else "holstered"
        active_assigned = active_assigned or state == "active"
        weapons[f"weapon_{idx}"] = {"name": gsi_name, "state": state}
    return weapons


def _bomb_payload(events: dict, round_info: pd.DataFrame, tick: int, round_num: int) -> dict[str, str]:
    ri = _round_row(round_info, round_num)
    plant_tick = None if ri is None or pd.isna(ri.get("plant_tick")) else _safe_int(ri.get("plant_tick"))
    if plant_tick is not None and tick >= plant_tick:
        planted_df = events.get("bomb_planted", pd.DataFrame())
        if isinstance(planted_df, pd.DataFrame) and not planted_df.empty and {"tick", "round_num"} <= set(planted_df.columns):
            planted = planted_df[(planted_df["round_num"] == round_num) & (planted_df["tick"] <= tick)]
            if not planted.empty:
                row = planted.sort_values("tick").iloc[-1]
                return {"state": "planted", "position": _vec3(_safe_float(row.get("X", 0.0)), _safe_float(row.get("Y", 0.0)), _safe_float(row.get("Z", 0.0)))}
        return {"state": "planted", "position": _vec3(0.0, 0.0, 0.0)}
    timeline: list[tuple[int, int, str, pd.Series]] = []
    for order, key, state in ((0, "bomb_dropped", "dropped"), (1, "bomb_pickup", "carried")):
        df = events.get(key, pd.DataFrame())
        if not isinstance(df, pd.DataFrame) or df.empty or "tick" not in df.columns:
            continue
        if "round_num" in df.columns:
            df = df[df["round_num"] == round_num]
        df = df[df["tick"] <= tick]
        for _, row in df.iterrows():
            timeline.append((_safe_int(row["tick"]), order, state, row))
    if not timeline:
        return {"state": "carried", "position": _vec3(0.0, 0.0, 0.0)}
    _, _, state, row = sorted(timeline, key=lambda item: (item[0], item[1]))[-1]
    return {"state": state, "position": _vec3(_safe_float(row.get("X", 0.0)), _safe_float(row.get("Y", 0.0)), _safe_float(row.get("Z", 0.0)))}


def _grenade_payload(events: dict, tick: int, round_num: int, gaps: set[str]) -> dict[str, dict[str, object]]:
    if not isinstance(events.get("smokegrenade_detonate"), pd.DataFrame):
        gaps.add("No `events[\"smokegrenade_detonate\"]` table; smoke entries were left unset.")
    inferno_df = events.get("inferno_startburn")
    if not isinstance(inferno_df, pd.DataFrame):
        gaps.add("No `events[\"inferno_startburn\"]` table; molotov entries were left unset.")
    elif not inferno_df.empty:
        gaps.add("Molotov flame geometry is not present in `*_full.pkl`; synthesized GSI uses a single flame point from `inferno_startburn`.")
    smokes, molotovs = get_active_utils(events, tick, round_num)
    out: dict[str, dict[str, object]] = {}
    for idx, (x, y, remain) in enumerate(smokes):
        out[f"smoke_{idx}"] = {"type": "smoke", "position": _vec3(x, y, 0.0), "effecttime": SMOKE_DURATION_SECONDS * (1.0 - remain)}
    for idx, (x, y, remain) in enumerate(molotovs):
        out[f"inferno_{idx}"] = {"type": "inferno", "flames": {"0": _vec3(x, y, 0.0)}, "lifetime": MOLOTOV_DURATION_SECONDS * (1.0 - remain)}
    return out


def _build_gsi(tick_slice: pd.DataFrame, events: dict, round_info: pd.DataFrame, tick: int, round_num: int, map_name: str, gaps: set[str]) -> dict[str, object]:
    ri = _round_row(round_info, round_num)
    ct_score = _safe_int(ri.get("ct_score", 0)) if ri is not None else 0
    t_score = _safe_int(ri.get("t_score", 0)) if ri is not None else 0
    freeze_tick = _safe_int(ri.get("freeze_tick", tick)) if ri is not None else tick
    allplayers: dict[str, dict[str, object]] = {}
    for _, player in tick_slice.iterrows():
        team_name = str(player.get("team_name", ""))
        team = "T" if team_name == "TERRORIST" else "CT" if team_name == "CT" else ""
        if not team:
            continue
        inventory = _inventory_list(player.get("inventory", []))
        hp = _safe_int(player.get("health", 0))
        allplayers[str(player.get("steamid", player.get("player_steamid", "")))] = {
            "name": str(player.get("name", "")),
            "team": team,
            "position": _vec3(_safe_float(player.get("X", 0.0)), _safe_float(player.get("Y", 0.0)), _safe_float(player.get("Z", 0.0))),
            "forward": _vec3(math.cos(math.radians(_safe_float(player.get("yaw", 0.0)))), math.sin(math.radians(_safe_float(player.get("yaw", 0.0)))), 0.0),
            "state": {
                "health": hp,
                "armor": _safe_int(player.get("armor_value", 0)),
                "helmet": _safe_bool(player.get("has_helmet", False)),
                "money": _safe_int(player.get("balance", 0)),
                "equip_value": _safe_int(player.get("current_equip_value", 0)),
                "flashed": _safe_bool(_safe_float(player.get("flash_duration", 0.0)) > 0.0),
                "defusekit": _safe_bool(player.get("has_defuser", False)),
            },
            "match_stats": {"score": _safe_int(player.get("score", 0))},
            "weapons": _weapon_payload(inventory, player.get("active_weapon_name")),
        }
    time_in_round = (tick - freeze_tick) / 64.0
    return {
        "map": {"name": map_name, "phase": "live", "round": round_num, "team_ct": {"score": ct_score}, "team_t": {"score": t_score}},
        "round": {"phase": "live"},
        "phase_countdowns": {"phase_ends_in": LIVE_ROUND_SECONDS - time_in_round},
        "bomb": _bomb_payload(events, round_info, tick, round_num),
        "grenades": _grenade_payload(events, tick, round_num, gaps),
        "allplayers": allplayers,
    }


def _stats_table(stats: pd.DataFrame, limit: int | None = None) -> str:
    if limit is not None:
        stats = stats.head(limit)
    lines = ["| feature | max_abs_diff | mean_abs_diff | fraction_nonzero_diff |", "| --- | ---: | ---: | ---: |"]
    for row in stats.itertuples(index=False):
        lines.append(f"| {row.feature} | {_fmt(row.max_abs_diff)} | {_fmt(row.mean_abs_diff)} | {_fmt(row.fraction_nonzero_diff)} |")
    return "\n".join(lines)


def _player_ordering_signal(systematic: pd.DataFrame) -> tuple[bool, str]:
    slots: dict[str, set[str]] = {}
    for name in systematic["feature"]:
        if name.startswith(("t", "ct")) and "_" in name:
            slot, field = name.split("_", 1)
            slots.setdefault(slot, set()).add(field)
    detail = ", ".join(f"{slot}:{len(fields)}" for slot, fields in sorted(slots.items())) or "none"
    return sum(len(fields) >= 6 for fields in slots.values()) >= 4, detail


def _likely_causes(nonzero: pd.DataFrame, systematic: pd.DataFrame) -> list[str]:
    if nonzero.empty:
        return ["No feature diffs above tolerance.", "Velocity-missing cannot be tested directly here because `state_vector_v2` exposes no velocity features."]
    causes: list[str] = []

    def add_if(label: str, predicate) -> None:
        subset = nonzero[nonzero["feature"].map(predicate)]
        if subset.empty:
            return
        top = subset.sort_values(["max_abs_diff", "mean_abs_diff"], ascending=False).iloc[0]
        causes.append(f"{label}: {len(subset)} features, max_abs_diff={_fmt(subset['max_abs_diff'].max())}, top={top['feature']}")

    add_if("Yaw convention / forward-vector round-trip", lambda name: name.endswith("_yaw"))
    add_if("Time-in-round / phase-countdown mapping", lambda name: name == "time_in_round")
    add_if("Coordinate normalization / bomb-zone classification", lambda name: name.endswith(("_x", "_y", "_z", "_in_bomb_zone")) or name.startswith(("bomb_", "smoke", "molotov")))
    add_if("Weapons / grenade inventory synthesis", lambda name: name.endswith(("_weapon_id", "_has_smoke", "_has_flash", "_has_he", "_has_molotov", "_has_c4")))
    add_if("Scores / economy state mapping", lambda name: name in {"ct_score", "t_score"} or name.endswith(("_balance", "_equip_value", "_score")))
    player_ordering, detail = _player_ordering_signal(systematic)
    causes.append(f"{'Player ordering / slot assignment may be drifting' if player_ordering else 'Player ordering / slot assignment is not strongly indicated'} ({detail}).")
    causes.append("Velocity-missing cannot be tested directly here because `state_vector_v2` exposes no velocity features.")
    return causes


def _build_report(pkl_path: Path, map_name: str, n_ticks: int, n_rows_offline: int, n_rows_online: int, stats: pd.DataFrame, row_mismatch_rate: float, gaps: set[str]) -> str:
    nonzero = stats[stats["max_abs_diff"] > TOL].copy()
    systematic = nonzero[nonzero["fraction_nonzero_diff"] > 0.5].sort_values(["fraction_nonzero_diff", "max_abs_diff"], ascending=False)
    lines = [
        "# Train/Infer Feature Parity Report",
        "",
        f"- pkl: `{pkl_path}`",
        f"- map_name: `{map_name}`",
        f"- total_ticks_compared: `{n_ticks}`",
        f"- sanity: `n_ticks={n_ticks} n_rows_offline={n_rows_offline} n_rows_online={n_rows_online}`",
        f"- row_mismatch_rate_any_feature_gt_{TOL:.0e}: `{_fmt(row_mismatch_rate)}`",
        f"- nonzero_diff_features: `{len(nonzero)}`",
        "",
        "## Top 40 Mismatched Features",
        "",
        _stats_table(nonzero, 40) if not nonzero.empty else "_No feature diffs above tolerance._",
        "",
        "## Systematic Features (>50% of compared ticks)",
        "",
        _stats_table(systematic) if not systematic.empty else "_No features exceeded the >50% systematic threshold._",
        "",
        "## Likely Causes",
        "",
        *[f"- {cause}" for cause in _likely_causes(nonzero, systematic)],
        "",
        "## Synthesis Gaps",
        "",
        *([f"- {gap}" for gap in sorted(gaps)] if gaps else ["- None detected for this payload."]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", help="Path to a *_full.pkl")
    parser.add_argument("--map-name", help="Override inferred map name")
    parser.add_argument("--max-ticks", type=int, default=500, help="Cap unique ticks compared")
    parser.add_argument("--out", help="Write markdown report instead of stdout")
    args = parser.parse_args()

    pkl_path = _resolve_pkl(args.pkl)
    payload = load_full_payload(pkl_path)
    tick_df, events, round_info = payload["tick_df"], payload["events"], payload["round_info"]
    map_name = _infer_map_name(pkl_path, args.map_name, payload.get("header", {}) or {})
    unique_ticks = tick_df[tick_df["round_num"].notna()][["round_num", "tick"]].drop_duplicates().sort_values(["round_num", "tick"]).head(args.max_ticks)
    if unique_ticks.empty:
        raise RuntimeError("no comparable ticks found in payload")
    tick_groups = tick_df.groupby(["round_num", "tick"], sort=False)
    gaps: set[str] = set()
    offline_vecs, online_vecs, round_steps = [], [], Counter()

    for item in unique_ticks.itertuples(index=False):
        round_num, tick = _safe_int(item.round_num), _safe_int(item.tick)
        tick_slice = tick_groups.get_group((item.round_num, item.tick))
        row_offline = build_feature_row_v2(tick_df=tick_df, tick_slice=tick_slice, events=events, tick=tick, round_num=round_num, map_name=map_name, round_info=round_info)
        row_online = build_row_from_gsi(_build_gsi(tick_slice, events, round_info, tick, round_num, map_name, gaps), round_steps[round_num], round_num, map_name)
        round_steps[round_num] += 1
        if row_online is None:
            gaps.add(f"Online row builder returned None for round={round_num} tick={tick}.")
            continue
        vec_offline, vec_online = build_state_vector(row_offline), build_state_vector(row_online)
        if vec_offline.shape != (FEATURE_DIM,) or vec_online.shape != (FEATURE_DIM,):
            raise RuntimeError(f"unexpected vector shapes at round={round_num} tick={tick}: {vec_offline.shape=} {vec_online.shape=}")
        offline_vecs.append(vec_offline)
        online_vecs.append(vec_online)

    if not offline_vecs:
        raise RuntimeError("no vectors produced for comparison")
    offline_mat, online_mat = np.stack(offline_vecs), np.stack(online_vecs)
    print(f"SANITY n_ticks={len(offline_vecs)} n_rows_offline={len(offline_vecs)} n_rows_online={len(online_vecs)}")
    diffs = np.abs(offline_mat - online_mat)
    stats = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "max_abs_diff": diffs.max(axis=0),
        "mean_abs_diff": diffs.mean(axis=0),
        "fraction_nonzero_diff": (diffs > TOL).mean(axis=0),
    }).sort_values(["max_abs_diff", "mean_abs_diff", "fraction_nonzero_diff"], ascending=False)
    report = _build_report(pkl_path, map_name, len(offline_vecs), len(offline_vecs), len(online_vecs), stats, float((diffs > TOL).any(axis=1).mean()), gaps)
    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Report written to {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
