from __future__ import annotations

import argparse
import os
import pickle
from bisect import bisect_right
from pathlib import Path
import sys
from typing import Any, Callable

import pandas as pd
from demoparser2 import DemoParser

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_path, resolve_path_input

TICK_RATE = 64
DOWNSAMPLE = 8

ALL_PROPS = [
    "X",
    "Y",
    "Z",
    "health",
    "armor_value",
    "has_helmet",
    "is_alive",
    "team_name",
    "name",
    "weapon_name",
    "inventory",
    "flash_duration",
    "equipment_value_this_round",
    "is_scoped",
    "is_defusing",
    "pitch",
    "yaw",
    "velocity_X",
    "velocity_Y",
    "velocity_Z",
    "is_walking",
    "ducking",
    "spotted",
    "has_defuser",
    "last_place_name",
    "in_bomb_zone",
    "in_buy_zone",
    "is_airborne",
    "balance",
    "current_equip_value",
    "round_start_equip_value",
    "cash_spent_this_round",
    "kills_total",
    "deaths_total",
    "assists_total",
    "damage_total",
    "active_weapon_name",
    "game_time",
    # economy system
    "ct_losing_streak",
    "t_losing_streak",
    "total_cash_spent",
    "total_rounds_played",
    # cumulative stats (extra)
    "headshot_kills_total",
    "enemies_flashed_total",
    "utility_damage_total",
    "mvps",
    "score",
    # identity (kept for optional player embedding)
    "player_steamid",
    # real team score (entity field)
    "CCSTeam.m_iScore",
]

EVENT_NAMES = [
    "player_death",
    "player_hurt",
    "weapon_fire",
    "grenade_thrown",
    "flashbang_detonate",
    "smokegrenade_detonate",
    "hegrenade_detonate",
    "inferno_startburn",
    "bomb_dropped",
    "bomb_pickup",
    "bomb_planted",
    "bomb_defused",
    "round_freeze_end",
    "round_end",
]

EVENT_PARSE_KWARGS = {
    # Keep precise actor position for bomb handoff events so dropped C4 can use
    # the actual event coordinates instead of inferring from later tick snapshots.
    "bomb_dropped": {"player": ["X", "Y", "Z"]},
    "bomb_pickup": {"player": ["X", "Y", "Z"]},
    "bomb_planted": {"player": ["X", "Y", "Z"]},
}

ROUND_INFO_COLUMNS = [
    "round_num",
    "freeze_tick",
    "plant_tick",
    "end_tick",
    "bomb_site",
    "ct_score",
    "t_score",
]

SITE_MAP = {
    0: "A",
    1: "B",
    "0": "A",
    "1": "B",
    "A": "A",
    "B": "B",
    "BOMBSITE_A": "A",
    "BOMBSITE_B": "B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract full CS2 demo data with demoparser2.",
    )
    parser.add_argument("demo_path", help="Path to a .dem file")
    parser.add_argument(
        "--downsample",
        type=int,
        default=DOWNSAMPLE,
        help=f"Tick stride (default {DOWNSAMPLE}=8Hz; use 32 for 2Hz).",
    )
    parser.add_argument(
        "--output-dir",
        default="viz",
        help="Data-root-relative output directory for *_full.pkl (default: viz).",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _empty_round_info() -> pd.DataFrame:
    return pd.DataFrame(columns=ROUND_INFO_COLUMNS)


def _prepare_dataframe(df: Any) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame):
        out = df.copy()
    else:
        out = pd.DataFrame(df)

    if "tick" in out.columns:
        out["tick"] = pd.to_numeric(out["tick"], errors="coerce")
        out = out[out["tick"].notna()].copy()
        out["tick"] = out["tick"].astype(int)
        out = out.sort_values("tick").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def _parse_event(parser: DemoParser, event_name: str) -> pd.DataFrame:
    print(f"[events] parsing {event_name}")
    kwargs = EVENT_PARSE_KWARGS.get(event_name, {})
    try:
        out = _prepare_dataframe(parser.parse_event(event_name, **kwargs))
    except Exception as exc:
        if kwargs:
            print(f"[events] {event_name} extra fields unavailable, retrying default parse: {exc}")
            try:
                out = _prepare_dataframe(parser.parse_event(event_name))
            except Exception as fallback_exc:
                print(f"[events] {event_name} unavailable: {fallback_exc}")
                return pd.DataFrame()
        else:
            print(f"[events] {event_name} unavailable: {exc}")
            return pd.DataFrame()

    # demoparser2 returns player-requested event fields as user_X/user_Y/user_Z.
    # Normalize bomb events so downstream feature code can read stable X/Y/Z names.
    if event_name in {"bomb_dropped", "bomb_pickup", "bomb_planted"}:
        rename_map = {}
        for src, dst in (("user_X", "X"), ("user_Y", "Y"), ("user_Z", "Z")):
            if src in out.columns and dst not in out.columns:
                rename_map[src] = dst
        if rename_map:
            out = out.rename(columns=rename_map)

    return out


def _normalize_bomb_site(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if value in SITE_MAP:
        return SITE_MAP[value]

    text = str(value).strip().upper()
    if text in SITE_MAP:
        return SITE_MAP[text]
    if text.endswith("_A") or text.endswith("SITEA") or text == "A":
        return "A"
    if text.endswith("_B") or text.endswith("SITEB") or text == "B":
        return "B"
    return value


def _normalize_winner(value: Any) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip().upper()
    if text in {"CT", "COUNTERTERRORIST", "COUNTER-TERRORIST", "COUNTER_TERRORIST"}:
        return "CT"
    if text in {"T", "TERRORIST", "TERRORISTS"}:
        return "T"
    return None


def _score_before_tick(round_end_df: pd.DataFrame, tick: int) -> tuple[int, int]:
    if round_end_df.empty or "tick" not in round_end_df.columns:
        return 0, 0

    prior = round_end_df[round_end_df["tick"] < tick]
    if prior.empty:
        return 0, 0

    if {"ct_score", "t_score"}.issubset(prior.columns):
        last_row = prior.iloc[-1]
        try:
            return int(last_row["ct_score"]), int(last_row["t_score"])
        except (TypeError, ValueError):
            pass

    if "winner" not in prior.columns:
        return 0, 0

    winners = prior["winner"].map(_normalize_winner)
    ct_score = int((winners == "CT").sum())
    t_score = int((winners == "T").sum())
    return ct_score, t_score


def _pick_first_in_window(
    df: pd.DataFrame,
    start_tick: int,
    stop_tick: int | None,
    *,
    stop_inclusive: bool,
) -> pd.Series | None:
    if df.empty or "tick" not in df.columns:
        return None

    mask = df["tick"] >= start_tick
    if stop_tick is not None:
        if stop_inclusive:
            mask &= df["tick"] <= stop_tick
        else:
            mask &= df["tick"] < stop_tick

    candidates = df.loc[mask]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def build_round_info(events: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[int]]:
    freeze_df = events["round_freeze_end"]
    plant_df = events["bomb_planted"]
    round_end_df = events["round_end"]

    if freeze_df.empty or "tick" not in freeze_df.columns:
        print("[rounds] no round_freeze_end ticks found; round mapping will be empty")
        return _empty_round_info(), []

    freeze_ticks = sorted(pd.unique(freeze_df["tick"]).tolist())
    rows: list[dict[str, Any]] = []
    interval_ends: list[int] = []

    print(f"[rounds] building round windows from {len(freeze_ticks)} freeze_end ticks")
    for idx, freeze_tick in enumerate(freeze_ticks, start=1):
        next_freeze = freeze_ticks[idx] if idx < len(freeze_ticks) else None
        end_row = _pick_first_in_window(
            round_end_df,
            freeze_tick,
            next_freeze,
            stop_inclusive=False,
        )
        end_tick = int(end_row["tick"]) if end_row is not None else pd.NA

        plant_stop = int(end_tick) if pd.notna(end_tick) else next_freeze
        plant_row = _pick_first_in_window(
            plant_df,
            freeze_tick,
            plant_stop,
            stop_inclusive=pd.notna(end_tick),
        )
        plant_tick = int(plant_row["tick"]) if plant_row is not None else pd.NA
        bomb_site = (
            _normalize_bomb_site(plant_row.get("site", pd.NA))
            if plant_row is not None
            else pd.NA
        )

        ct_score, t_score = _score_before_tick(round_end_df, freeze_tick)

        interval_end: int
        if pd.notna(end_tick):
            interval_end = int(end_tick)
        elif pd.notna(plant_tick):
            interval_end = int(plant_tick)
        elif next_freeze is not None:
            interval_end = int(next_freeze) - 1
        else:
            interval_end = int(freeze_tick)

        rows.append(
            {
                "round_num": idx,
                "freeze_tick": int(freeze_tick),
                "plant_tick": plant_tick,
                "end_tick": end_tick,
                "bomb_site": bomb_site,
                "ct_score": ct_score,
                "t_score": t_score,
            }
        )
        interval_ends.append(interval_end)

    round_info = pd.DataFrame(rows, columns=ROUND_INFO_COLUMNS)
    for column in ("round_num", "freeze_tick", "plant_tick", "end_tick", "ct_score", "t_score"):
        if column in round_info.columns:
            round_info[column] = round_info[column].astype("Int64")
    return round_info, interval_ends


def backfill_team_scores(round_info: pd.DataFrame, tick_df: pd.DataFrame) -> pd.DataFrame:
    """Replace round_info ct_score/t_score with team-indexed scores from tick_df.

    The original build_round_info tallies winners by current side name, which
    is wrong after halftime when sides swap. CCSTeam.m_iScore tracks cumulative
    score by team identity (matches GSI and scoreboard semantics).
    """
    if round_info.empty or tick_df.empty or "CCSTeam.m_iScore" not in tick_df.columns:
        return round_info

    out = round_info.copy()
    ct_scores: list[int] = []
    t_scores: list[int] = []
    for freeze_tick in out["freeze_tick"].astype(int).tolist():
        slc = tick_df[tick_df["tick"] == freeze_tick]
        if slc.empty:
            ct_scores.append(0)
            t_scores.append(0)
            continue
        ct_rows = slc[slc["team_name"] == "CT"]
        t_rows = slc[slc["team_name"] == "TERRORIST"]
        ct = int(ct_rows["CCSTeam.m_iScore"].iloc[0]) if not ct_rows.empty else 0
        tt = int(t_rows["CCSTeam.m_iScore"].iloc[0]) if not t_rows.empty else 0
        ct_scores.append(ct)
        t_scores.append(tt)
    out["ct_score"] = pd.array(ct_scores, dtype="Int64")
    out["t_score"] = pd.array(t_scores, dtype="Int64")
    return out


def build_round_mapper(
    round_info: pd.DataFrame,
    interval_ends: list[int],
) -> Callable[[Any], Any]:
    if round_info.empty:
        return lambda tick: pd.NA

    start_ticks = round_info["freeze_tick"].astype(int).tolist()
    round_nums = round_info["round_num"].astype(int).tolist()

    def map_tick(tick: Any) -> Any:
        if pd.isna(tick):
            return pd.NA

        tick_int = int(tick)
        idx = bisect_right(start_ticks, tick_int) - 1
        if idx < 0:
            return pd.NA
        if tick_int <= interval_ends[idx]:
            return round_nums[idx]
        return pd.NA

    return map_tick


def add_round_num(df: pd.DataFrame, round_mapper: Callable[[Any], Any]) -> pd.DataFrame:
    out = df.copy()
    if "tick" not in out.columns:
        out["round_num"] = pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"))
        return out

    mapped = out["tick"].map(round_mapper)
    out["round_num"] = pd.Series(mapped, index=out.index, dtype="Int64")
    return out


def extract_tick_data(
    parser: DemoParser,
    round_info: pd.DataFrame,
    downsample: int = DOWNSAMPLE,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for row in round_info.itertuples(index=False):
        start_tick = int(row.freeze_tick)
        stop_value = row.plant_tick if pd.notna(row.plant_tick) else row.end_tick
        if pd.isna(stop_value):
            print(f"[ticks] round {row.round_num}: skipped (no plant_tick or end_tick)")
            continue

        stop_tick = int(stop_value)
        if stop_tick < start_tick:
            print(f"[ticks] round {row.round_num}: skipped ({stop_tick} < {start_tick})")
            continue

        wanted_ticks = list(range(start_tick, stop_tick + 1, downsample))
        print(
            f"[ticks] round {row.round_num}: parsing {len(wanted_ticks)} sampled ticks "
            f"from {start_tick} to {stop_tick}"
        )
        try:
            tick_df = _prepare_dataframe(parser.parse_ticks(ALL_PROPS, ticks=wanted_ticks))
        except Exception as exc:
            print(f"[ticks] round {row.round_num}: parse_ticks failed: {exc}")
            continue

        if tick_df.empty:
            print(f"[ticks] round {row.round_num}: no tick rows returned")
            continue
        if "tick" not in tick_df.columns:
            print(f"[ticks] round {row.round_num}: missing tick column")
            continue

        parts.append(tick_df)

    if not parts:
        return pd.DataFrame(columns=["tick", *ALL_PROPS, "round_num"])

    tick_df = pd.concat(parts, ignore_index=True, sort=False)
    sort_cols = ["tick"]
    if "name" in tick_df.columns:
        sort_cols.append("name")
    return tick_df.sort_values(sort_cols).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    demo_path = resolve_path_input(args.demo_path)
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        return 1

    if demo_path.suffix.lower() != ".dem":
        print(f"Input is not a .dem file: {demo_path}")
        return 1

    output_dir = data_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"{demo_path.stem}_full.pkl"

    downsample = int(args.downsample)
    if downsample < 1:
        print(f"[start] invalid --downsample {downsample}; must be >= 1")
        return 1

    print(f"[start] demo={demo_path}")
    print(f"[start] tick_rate={TICK_RATE}, downsample_every={downsample}, output_dir={output_dir}")
    parser = DemoParser(str(demo_path))

    print("[header] parsing header")
    try:
        header = parser.parse_header() or {}
    except Exception as exc:
        print(f"[header] parse_header failed: {exc}")
        header = {}

    events = {event_name: _parse_event(parser, event_name) for event_name in EVENT_NAMES}
    round_info, interval_ends = build_round_info(events)
    round_mapper = build_round_mapper(round_info, interval_ends)

    tick_df = extract_tick_data(parser, round_info, downsample=downsample)
    tick_df = add_round_num(tick_df, round_mapper)
    if not tick_df.empty:
        tick_df = tick_df[tick_df["round_num"].notna()].reset_index(drop=True)

    round_info = backfill_team_scores(round_info, tick_df)

    for event_name, event_df in list(events.items()):
        events[event_name] = add_round_num(event_df, round_mapper)

    payload = {
        "header": header,
        "tick_df": tick_df,
        "events": events,
        "round_info": round_info,
    }

    print(f"[save] writing {output_path}")
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        "[done] "
        f"rounds={len(round_info)}, "
        f"tick_rows={len(tick_df)}, "
        f"output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
