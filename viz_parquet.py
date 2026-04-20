"""
Visualize parquet content for a manual QA sample.

Default behavior:
- pick 2 demos per map from processed/
- require matching raw demo in raw/demos/
- write HTML report to the active data root

Usage:
    python viz_parquet.py
    python viz_parquet.py --per-map 2 --open
    python viz_parquet.py --schema v2 --per-map 2 --open
"""

from __future__ import annotations

import argparse
import html
import json
import webbrowser
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.utils.paths import data_path


ROOT = Path(__file__).parent
PROCESSED = data_path("processed")
PROCESSED_V2 = data_path("processed_v2")
RAW_DEMOS = data_path("raw", "demos")
MANIFEST = data_path("raw", "manifest.jsonl")
OUT_HTML = data_path("viz_report.html")
OUT_HTML_V2 = data_path("viz_report_v2.html")

MAP_ORDER = [
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
]

SITE_COLOR = {"A": "#4ade80", "B": "#60a5fa", "other": "#f87171", "?": "#e5e7eb"}

# --- v2 schema constants (imported lazily to avoid hard dep for v1 mode) ---
V2_WEAPONS = (
    "Glock-18", "USP-S", "P2000", "P250", "Five-SeveN", "Tec-9",
    "CZ75-Auto", "Desert Eagle", "R8 Revolver", "Dual Berettas",
    "AK-47", "M4A4", "M4A1-S", "FAMAS", "Galil AR", "SG 553", "AUG",
    "AWP", "SSG 08", "SCAR-20", "G3SG1",
    "MP9", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "MAC-10", "MP7",
    "Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev",
    "None",
)
V2_MAPS = (
    "de_mirage",
    "de_inferno",
    "de_dust2",
    "de_nuke",
    "de_ancient",
    "de_overpass",
    "de_anubis",
)


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read parquet via pyarrow directly to bypass pandas/pyarrow version issues."""
    return pq.read_table(path).to_pandas()


def load_manifest_index() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    if not MANIFEST.exists():
        return index

    with MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            demo_file = row.get("demo_file")
            if not demo_file:
                continue
            stem = Path(demo_file).stem
            index[stem] = {
                "event": row.get("event", ""),
                "date": row.get("date", ""),
                "map": row.get("map", ""),
            }
    return index


# ─── Demo selection ──────────────────────────────────────────────────────────


def select_demo_stems(per_map: int) -> list[str]:
    by_map: dict[str, list[str]] = defaultdict(list)

    for parquet_path in PROCESSED.glob("*.parquet"):
        stem = parquet_path.stem
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue

        match_id, map_name = parts
        raw_demo = RAW_DEMOS / f"{stem}.dem"
        if not raw_demo.exists():
            continue

        if map_name not in MAP_ORDER:
            continue

        try:
            sort_key = int(match_id)
        except ValueError:
            continue
        by_map[map_name].append((sort_key, stem))

    selected: list[str] = []
    missing_maps: list[str] = []
    for map_name in MAP_ORDER:
        candidates = sorted(by_map[map_name], reverse=True)
        picked = [stem for _, stem in candidates[:per_map]]
        if len(picked) < per_map:
            missing_maps.append(f"{map_name} ({len(picked)}/{per_map})")
        selected.extend(picked)

    if missing_maps:
        raise RuntimeError(
            "Not enough demos for manual QA sample: " + ", ".join(missing_maps)
        )

    return selected


def select_demo_stems_v2(per_map: int) -> list[str]:
    """Select demos from processed_v2/ directory.

    v2 parquets have a map_name metadata column, so we read the first row
    to determine the map.  No raw demo requirement for v2.
    """
    by_map: dict[str, list[str]] = defaultdict(list)

    for parquet_path in PROCESSED_V2.glob("*.parquet"):
        stem = parquet_path.stem
        # Try to get map_name from the parquet metadata columns
        try:
            df_head = pq.read_table(parquet_path, columns=["map_name"]).to_pandas()
            if df_head.empty:
                continue
            map_name = str(df_head["map_name"].iloc[0])
        except Exception:
            # Fallback: infer from filename
            parts = stem.split("_", 1)
            if len(parts) != 2:
                continue
            map_name = parts[1]

        if map_name not in MAP_ORDER:
            continue

        parts = stem.split("_", 1)
        try:
            sort_key = int(parts[0])
        except (ValueError, IndexError):
            sort_key = 0
        by_map[map_name].append((sort_key, stem))

    selected: list[str] = []
    missing_maps: list[str] = []
    for map_name in MAP_ORDER:
        candidates = sorted(by_map[map_name], reverse=True)
        picked = [stem for _, stem in candidates[:per_map]]
        if len(picked) < per_map:
            missing_maps.append(f"{map_name} ({len(picked)}/{per_map})")
        selected.extend(picked)

    if missing_maps:
        raise RuntimeError(
            "Not enough v2 demos for manual QA sample: " + ", ".join(missing_maps)
        )

    return selected


# ─── v2 helpers ──────────────────────────────────────────────────────────────


def _v2_decode_weapon(row: pd.Series, prefix: str) -> str:
    """Decode weapon_id integer back to weapon display name."""
    wid = int(round(float(row.get(f"{prefix}_weapon_id", len(V2_WEAPONS) - 1))))
    if 0 <= wid < len(V2_WEAPONS):
        return V2_WEAPONS[wid]
    return "None"


def _v2_decode_map(row: pd.Series) -> str:
    """Decode map one-hot back to map name."""
    best_map = "unknown"
    best_val = -1.0
    for m in V2_MAPS:
        val = float(row.get(f"map_{m}", 0.0))
        if val > best_val:
            best_val = val
            best_map = m
    return best_map


def _v2_infer_round_num(df: pd.DataFrame) -> pd.DataFrame:
    """Add an integer round_num column.

    processed_v2 parquet stores RAW values (not normalized), so round_num
    is already the real round number — just cast to int.
    """
    if "round_num" in df.columns:
        df = df.copy()
        df["_round_num"] = df["round_num"].astype(float).round().astype(int)
    return df


# ─── v1 round table / player snapshot ───────────────────────────────────────


def round_table(df: pd.DataFrame) -> str:
    rounds = (
        df.groupby("round_num", sort=True)
        .agg(
            bomb_site=("bomb_site", "first"),
            steps=("step", "count"),
            ct_score=("ct_score", "first"),
            t_score=("t_score", "first"),
        )
        .reset_index()
    )

    rows_html = []
    for _, row in rounds.iterrows():
        site = str(row["bomb_site"]) if pd.notna(row["bomb_site"]) else "?"
        color = SITE_COLOR.get(site, "#e5e7eb")
        rows_html.append(
            "<tr>"
            f"<td>{int(row['round_num'])}</td>"
            f"<td style='background:{color};font-weight:bold;text-align:center'>{html.escape(site)}</td>"
            f"<td>{int(row['ct_score'])}-{int(row['t_score'])}</td>"
            f"<td>{int(row['steps'])}</td>"
            "</tr>"
        )

    return (
        "<table border='1' cellpadding='4' cellspacing='0' "
        "style='border-collapse:collapse;font-size:13px'>"
        "<thead><tr>"
        "<th>Round</th><th>Bomb Site</th><th>Score CT-T</th><th>Steps</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows_html) + "</tbody>"
        "</table>"
    )


def round_table_v2(df: pd.DataFrame) -> str:
    """Build per-round summary table for v2 schema."""
    df = _v2_infer_round_num(df)
    rounds = (
        df.groupby("_round_num", sort=True)
        .agg(
            bomb_site=("bomb_site", "first"),
            steps=("step", "count"),
            ct_score=("ct_score", "first"),
            t_score=("t_score", "first"),
            time_in_round_max=("time_in_round", "max"),
        )
        .reset_index()
    )

    rows_html = []
    for _, row in rounds.iterrows():
        site = str(row["bomb_site"]) if pd.notna(row["bomb_site"]) else "?"
        color = SITE_COLOR.get(site, "#e5e7eb")
        ct_s = int(round(float(row["ct_score"])))
        t_s = int(round(float(row["t_score"])))
        max_time = float(row["time_in_round_max"])
        rows_html.append(
            "<tr>"
            f"<td>{int(row['_round_num'])}</td>"
            f"<td style='background:{color};font-weight:bold;text-align:center'>{html.escape(site)}</td>"
            f"<td>{ct_s}-{t_s}</td>"
            f"<td>{int(row['steps'])}</td>"
            f"<td>{max_time:.1f}s</td>"
            "</tr>"
        )

    return (
        "<table border='1' cellpadding='4' cellspacing='0' "
        "style='border-collapse:collapse;font-size:13px'>"
        "<thead><tr>"
        "<th>Round</th><th>Bomb Site</th><th>Score CT-T</th><th>Steps</th><th>Max Time</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows_html) + "</tbody>"
        "</table>"
    )


def player_snapshot(df: pd.DataFrame, round_num: int, step: int = 0) -> str:
    row = df[(df["round_num"] == round_num) & (df["step"] == step)]
    if row.empty:
        return "<p><em>No data</em></p>"
    row = row.iloc[0]

    cells = []
    for side in ("t", "ct"):
        for i in range(5):
            hp = int(row.get(f"{side}{i}_hp", 0))
            alive = bool(row.get(f"{side}{i}_alive", False))
            weapon = html.escape(str(row.get(f"{side}{i}_weapon", "?")))
            style = "color:#9ca3af" if not alive else ""
            cells.append(
                f"<td style='{style}'>{side.upper()}{i} hp={hp} {weapon}</td>"
            )

    return (
        "<table border='1' cellpadding='3' cellspacing='0' "
        "style='border-collapse:collapse;font-size:12px'>"
        f"<tr>{''.join(cells)}</tr>"
        "</table>"
    )


def player_snapshot_v2(df: pd.DataFrame, round_num_int: int, step: int = 0) -> str:
    """Player snapshot for v2 schema with denormalized values."""
    df = _v2_infer_round_num(df)
    row = df[(df["_round_num"] == round_num_int) & (df["step"] == step)]
    if row.empty:
        return "<p><em>No data</em></p>"
    row = row.iloc[0]

    cells = []
    for side in ("t", "ct"):
        for i in range(5):
            prefix = f"{side}{i}"
            hp = int(round(float(row.get(f"{prefix}_hp", 0))))
            armor = int(round(float(row.get(f"{prefix}_armor", 0))))
            alive = float(row.get(f"{prefix}_alive", 0)) > 0.5
            weapon = _v2_decode_weapon(row, prefix)
            balance = int(round(float(row.get(f"{prefix}_balance", 0))))
            equip = int(round(float(row.get(f"{prefix}_equip_value", 0))))
            in_bz = float(row.get(f"{prefix}_in_bomb_zone", 0)) > 0.5
            has_c4 = float(row.get(f"{prefix}_has_c4", 0)) > 0.5
            yaw = float(row.get(f"{prefix}_yaw", 0))

            style = "color:#9ca3af" if not alive else ""
            extra_tags = []
            if in_bz:
                extra_tags.append("BZ")
            if has_c4:
                extra_tags.append("C4")
            extras = f" [{'/'.join(extra_tags)}]" if extra_tags else ""

            cells.append(
                f"<td style='{style}'>"
                f"{side.upper()}{i} hp={hp} arm={armor} {weapon} ${balance} eq={equip} yaw={yaw:.0f}{extras}"
                f"</td>"
            )

    return (
        "<table border='1' cellpadding='3' cellspacing='0' "
        "style='border-collapse:collapse;font-size:12px'>"
        f"<tr>{''.join(cells)}</tr>"
        "</table>"
    )


def schema_section(df: pd.DataFrame) -> str:
    sample = df.iloc[0]
    rows = []
    for col in df.columns:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(col))}</td>"
            f"<td>{html.escape(str(df[col].dtype))}</td>"
            f"<td>{html.escape(str(sample[col]))}</td>"
            "</tr>"
        )

    return (
        "<details><summary style='cursor:pointer;font-weight:bold'>"
        "Schema and first-row sample"
        "</summary>"
        "<table border='1' cellpadding='3' cellspacing='0' "
        "style='border-collapse:collapse;font-size:11px;margin-top:6px'>"
        "<thead><tr><th>Column</th><th>Dtype</th><th>Sample value</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table></details>"
    )


def bomb_site_bar(df: pd.DataFrame) -> str:
    counts = df.groupby("round_num")["bomb_site"].first().value_counts()
    total = counts.sum()
    bars = []
    for site in ["A", "B", "other"]:
        n = counts.get(site, 0)
        pct = n / total * 100 if total else 0
        color = SITE_COLOR[site]
        bars.append(
            "<div style='margin:2px 0'>"
            f"<span style='display:inline-block;width:60px'>{site}</span>"
            f"<span style='display:inline-block;background:{color};"
            f"width:{int(pct * 3)}px;height:16px;vertical-align:middle'></span>"
            f"<span style='margin-left:6px'>{n} rounds ({pct:.0f}%)</span>"
            "</div>"
        )
    return "".join(bars)


def bomb_site_bar_v2(df: pd.DataFrame) -> str:
    """Bomb site distribution bar for v2 schema."""
    df = _v2_infer_round_num(df)
    counts = df.groupby("_round_num")["bomb_site"].first().value_counts()
    total = counts.sum()
    bars = []
    for site in ["A", "B", "other"]:
        n = counts.get(site, 0)
        pct = n / total * 100 if total else 0
        color = SITE_COLOR[site]
        bars.append(
            "<div style='margin:2px 0'>"
            f"<span style='display:inline-block;width:60px'>{site}</span>"
            f"<span style='display:inline-block;background:{color};"
            f"width:{int(pct * 3)}px;height:16px;vertical-align:middle'></span>"
            f"<span style='margin-left:6px'>{n} rounds ({pct:.0f}%)</span>"
            "</div>"
        )
    return "".join(bars)


# ─── v2 extra sections ──────────────────────────────────────────────────────


def tick_sample_v2(df: pd.DataFrame) -> str:
    """Show 2 sampled ticks (early + late) with full player/bomb detail."""
    df = _v2_infer_round_num(df)
    rounds = sorted(df["_round_num"].unique())
    if not rounds:
        return "<p><em>No rounds</em></p>"

    r1 = rounds[0]
    r2 = rounds[-1]
    r1_df = df[df["_round_num"] == r1].sort_values("step")
    r2_df = df[df["_round_num"] == r2].sort_values("step")

    samples = [
        ("Early", r1, r1_df.iloc[len(r1_df) // 4]),
        ("Late", r2, r2_df.iloc[3 * len(r2_df) // 4]),
    ]

    sections = []
    for label, rn, row in samples:
        step = int(row["step"])
        tick = int(row["tick"])
        site = str(row.get("bomb_site", "?"))
        ct_s = int(round(float(row.get("ct_score", 0))))
        t_s = int(round(float(row.get("t_score", 0))))
        time_r = float(row.get("time_in_round", 0))

        bomb_dropped = float(row.get("bomb_dropped", 0)) > 0.5
        bomb_x = float(row.get("bomb_x", 0))
        bomb_y = float(row.get("bomb_y", 0))
        bomb_txt = f"DROPPED ({bomb_x:.0f}, {bomb_y:.0f})" if bomb_dropped else "carried"

        # Active utility
        util_parts = []
        for s in range(5):
            rem = float(row.get(f"smoke{s}_remain", 0))
            if rem > 0:
                sx, sy = float(row.get(f"smoke{s}_x", 0)), float(row.get(f"smoke{s}_y", 0))
                util_parts.append(f"smoke({sx:.0f},{sy:.0f} {rem:.0%})")
        for m in range(3):
            rem = float(row.get(f"molotov{m}_remain", 0))
            if rem > 0:
                mx, my = float(row.get(f"molotov{m}_x", 0)), float(row.get(f"molotov{m}_y", 0))
                util_parts.append(f"molo({mx:.0f},{my:.0f} {rem:.0%})")
        util_txt = ", ".join(util_parts) if util_parts else "none"

        # Player rows
        player_rows = []
        for side in ("t", "ct"):
            for i in range(5):
                p = f"{side}{i}"
                alive = float(row.get(f"{p}_alive", 0)) > 0.5
                hp = int(round(float(row.get(f"{p}_hp", 0))))
                armor = int(round(float(row.get(f"{p}_armor", 0))))
                helmet = float(row.get(f"{p}_helmet", 0)) > 0.5

                wid = int(round(float(row.get(f"{p}_weapon_id", len(V2_WEAPONS) - 1))))
                wep = V2_WEAPONS[wid] if 0 <= wid < len(V2_WEAPONS) else "None"

                bal = int(round(float(row.get(f"{p}_balance", 0))))
                eq = int(round(float(row.get(f"{p}_equip_value", 0))))
                score = int(round(float(row.get(f"{p}_score", 0))))
                yaw = float(row.get(f"{p}_yaw", 0))
                x = float(row.get(f"{p}_x", 0))
                y = float(row.get(f"{p}_y", 0))

                flags = []
                if float(row.get(f"{p}_in_bomb_zone", 0)) > 0.5:
                    flags.append("BZ")
                if float(row.get(f"{p}_has_c4", 0)) > 0.5:
                    flags.append("C4")

                nades = []
                for g in ("smoke", "flash", "he", "molotov"):
                    if float(row.get(f"{p}_has_{g}", 0)) > 0.5:
                        nades.append(g[0].upper())

                style = " style='color:#9ca3af'" if not alive else ""
                player_rows.append(
                    f"<tr{style}>"
                    f"<td>{side.upper()}{i}</td>"
                    f"<td>{'Y' if alive else '-'}</td>"
                    f"<td>{hp}</td><td>{armor}{'H' if helmet else ''}</td>"
                    f"<td>{wep}</td>"
                    f"<td>${bal:,}</td><td>{eq:,}</td>"
                    f"<td>{score}</td><td>{yaw:.0f}</td>"
                    f"<td>({x:.0f}, {y:.0f})</td>"
                    f"<td>{''.join(nades) if nades else '-'}</td>"
                    f"<td>{' '.join(flags) if flags else '-'}</td>"
                    "</tr>"
                )

        sections.append(f"""
<div style='margin-bottom:12px'>
  <strong>{html.escape(label)} sample</strong> — Round {rn}, step {step}, tick {tick}<br>
  Score: CT {ct_s} - {t_s} T &nbsp;|&nbsp; Site: {html.escape(site)} &nbsp;|&nbsp;
  Time: {time_r:.1f}s &nbsp;|&nbsp; Bomb: {html.escape(bomb_txt)}<br>
  Utility: {html.escape(util_txt)}
  <table border='1' cellpadding='3' cellspacing='0'
    style='border-collapse:collapse;font-size:11px;margin-top:4px'>
    <thead><tr>
      <th>Slot</th><th>Alive</th><th>HP</th><th>Armor</th><th>Weapon</th>
      <th>$</th><th>Equip</th><th>Score</th><th>Yaw</th><th>Pos</th><th>Nades</th><th>Flags</th>
    </tr></thead>
    <tbody>{"".join(player_rows)}</tbody>
  </table>
</div>""")

    return "".join(sections)


def bomb_state_section_v2(df: pd.DataFrame, round_num_int: int, step: int = 0) -> str:
    """Show bomb/utility state for a v2 snapshot."""
    df = _v2_infer_round_num(df)
    row = df[(df["_round_num"] == round_num_int) & (df["step"] == step)]
    if row.empty:
        return ""
    row = row.iloc[0]

    bomb_dropped = float(row.get("bomb_dropped", 0)) > 0.5
    bomb_x = float(row.get("bomb_x", 0))
    bomb_y = float(row.get("bomb_y", 0))

    parts = [f"<strong>Bomb:</strong> {'DROPPED' if bomb_dropped else 'carried'} ({bomb_x:.3f}, {bomb_y:.3f})"]

    # Active smokes
    smokes = []
    for s in range(5):
        remain = float(row.get(f"smoke{s}_remain", 0))
        if remain > 0:
            sx = float(row.get(f"smoke{s}_x", 0))
            sy = float(row.get(f"smoke{s}_y", 0))
            smokes.append(f"({sx:.3f},{sy:.3f} rem={remain:.2f})")
    if smokes:
        parts.append(f"<strong>Smokes:</strong> {' '.join(smokes)}")

    # Active molotovs
    molos = []
    for m in range(3):
        remain = float(row.get(f"molotov{m}_remain", 0))
        if remain > 0:
            mx = float(row.get(f"molotov{m}_x", 0))
            my = float(row.get(f"molotov{m}_y", 0))
            molos.append(f"({mx:.3f},{my:.3f} rem={remain:.2f})")
    if molos:
        parts.append(f"<strong>Molotovs:</strong> {' '.join(molos)}")

    return "<p style='font-size:12px'>" + "<br>".join(parts) + "</p>"


# ─── Summary tables ─────────────────────────────────────────────────────────


def summary_table(stems: list[str], manifest_index: dict[str, dict[str, str]], schema: str = "v1") -> str:
    processed_dir = PROCESSED_V2 if schema == "v2" else PROCESSED
    rows = []
    for stem in stems:
        meta = manifest_index.get(stem, {})
        map_name = stem.split("_", 1)[1] if "_" in stem else "unknown"
        rows.append(
            "<tr>"
            f"<td>{html.escape(map_name)}</td>"
            f"<td>{html.escape(stem)}</td>"
            f"<td>{html.escape(meta.get('date', ''))}</td>"
            f"<td>{html.escape(meta.get('event', ''))}</td>"
            f"<td>{html.escape(str(RAW_DEMOS / f'{stem}.dem'))}</td>"
            f"<td>{html.escape(str(processed_dir / f'{stem}.parquet'))}</td>"
            "</tr>"
        )

    return (
        "<table border='1' cellpadding='4' cellspacing='0' "
        "style='border-collapse:collapse;font-size:12px;width:100%'>"
        "<thead><tr>"
        "<th>Map</th><th>Demo</th><th>Date</th><th>Event</th>"
        "<th>Raw DEM</th><th>Parquet</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


# ─── Per-demo section builders ───────────────────────────────────────────────


def demo_section(stem: str, manifest_index: dict[str, dict[str, str]]) -> str:
    path = PROCESSED / f"{stem}.parquet"
    raw_demo_path = RAW_DEMOS / f"{stem}.dem"
    df = _read_parquet(path)

    total_rounds = df["round_num"].nunique()
    total_rows = len(df)
    planted_rounds = df[df["bomb_site"].isin(["A", "B"])]["round_num"].unique()
    round_nums = sorted(df["round_num"].unique())
    snap_round = int(planted_rounds[0]) if len(planted_rounds) else int(round_nums[0])
    meta = manifest_index.get(stem, {})

    return f"""
<div style='border:1px solid #ccc;border-radius:6px;padding:16px;margin-bottom:24px;font-family:monospace'>
  <h2 style='margin-top:0'>{html.escape(stem)}</h2>
  <p>
    <strong>Map:</strong> {html.escape(meta.get("map") or stem.split("_", 1)[1])}<br>
    <strong>Date:</strong> {html.escape(meta.get("date", ""))}<br>
    <strong>Event:</strong> {html.escape(meta.get("event", ""))}<br>
    <strong>Raw DEM:</strong> {html.escape(str(raw_demo_path))}<br>
    <strong>Parquet:</strong> {html.escape(str(path))}
  </p>
  <p>
    <strong>{total_rounds}</strong> rounds &nbsp;|&nbsp;
    <strong>{total_rows:,}</strong> total rows &nbsp;|&nbsp;
    <strong>{df.shape[1]}</strong> columns
  </p>
  <h3>Bomb site distribution by round</h3>
  {bomb_site_bar(df)}
  <h3>Per-round summary</h3>
  {round_table(df)}
  <h3>Player snapshot: Round {snap_round}, Step 0</h3>
  {player_snapshot(df, snap_round, step=0)}
  <br>
  {schema_section(df)}
</div>
"""


def demo_section_v2(stem: str, manifest_index: dict[str, dict[str, str]]) -> str:
    """Build a per-demo HTML section for v2 schema."""
    path = PROCESSED_V2 / f"{stem}.parquet"
    raw_demo_path = RAW_DEMOS / f"{stem}.dem"
    df = _read_parquet(path)

    df = _v2_infer_round_num(df)

    # Determine map from metadata column or one-hot
    if "map_name" in df.columns and pd.notna(df["map_name"].iloc[0]):
        map_display = str(df["map_name"].iloc[0])
    else:
        map_display = _v2_decode_map(df.iloc[0])

    total_rounds = df["_round_num"].nunique()
    total_rows = len(df)
    planted_rounds = df[df["bomb_site"].isin(["A", "B"])]["_round_num"].unique()
    round_nums = sorted(df["_round_num"].unique())
    snap_round = int(planted_rounds[0]) if len(planted_rounds) else int(round_nums[0])
    meta = manifest_index.get(stem, {})

    return f"""
<div style='border:1px solid #ccc;border-radius:6px;padding:16px;margin-bottom:24px;font-family:monospace'>
  <h2 style='margin-top:0'>{html.escape(stem)}</h2>
  <p>
    <strong>Map:</strong> {html.escape(map_display)}<br>
    <strong>Schema:</strong> v2 (218-dim realtime-aligned)<br>
    <strong>Date:</strong> {html.escape(meta.get("date", ""))}<br>
    <strong>Event:</strong> {html.escape(meta.get("event", ""))}<br>
    <strong>Raw DEM:</strong> {html.escape(str(raw_demo_path))}<br>
    <strong>Parquet:</strong> {html.escape(str(path))}
  </p>
  <p>
    <strong>{total_rounds}</strong> rounds &nbsp;|&nbsp;
    <strong>{total_rows:,}</strong> total rows &nbsp;|&nbsp;
    <strong>{df.shape[1]}</strong> columns (5 meta + {df.shape[1] - 5} features)
  </p>
  <h3>Bomb site distribution by round</h3>
  {bomb_site_bar_v2(df)}
  <h3>Per-round summary</h3>
  {round_table_v2(df)}
  <h3>Player snapshot: Round {snap_round}, Step 0</h3>
  {player_snapshot_v2(df, snap_round, step=0)}
  <h3>Bomb / Utility state: Round {snap_round}, Step 0</h3>
  {bomb_state_section_v2(df, snap_round, step=0)}
  <h3>Tick samples (early + late)</h3>
  {tick_sample_v2(df)}
  <br>
  {schema_section(df)}
</div>
"""


# ─── Report builders ─────────────────────────────────────────────────────────


def build_report(stems: list[str], manifest_index: dict[str, dict[str, str]]) -> str:
    sections = []
    for stem in stems:
        print(f"Processing {stem}...")
        sections.append(demo_section(stem, manifest_index))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CS Prophet Parquet Viewer</title>
<style>
  body {{ font-family: monospace; max-width: 1400px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 6px; }}
  h2, h3 {{ margin-bottom: 8px; }}
  summary {{ font-size: 13px; }}
</style>
</head>
<body>
<h1>CS Prophet Manual QA Report</h1>
<p style="color:#666">
Selected 2 demos per map from <code>processed/</code> when a matching raw demo exists in
<code>raw/demos/</code> under the active data root. Use the raw demo path to cross-check the parquet labels manually.
</p>
<h2>Selected demos</h2>
{summary_table(stems, manifest_index)}
{"".join(sections)}
</body>
</html>"""


def build_report_v2(stems: list[str], manifest_index: dict[str, dict[str, str]]) -> str:
    sections = []
    for stem in stems:
        print(f"Processing (v2) {stem}...")
        sections.append(demo_section_v2(stem, manifest_index))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CS Prophet Parquet Viewer (v2 218-dim)</title>
<style>
  body {{ font-family: monospace; max-width: 1400px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 6px; }}
  h2, h3 {{ margin-bottom: 8px; }}
  summary {{ font-size: 13px; }}
</style>
</head>
<body>
<h1>CS Prophet Manual QA Report (v2 218-dim Schema)</h1>
<p style="color:#666">
Selected demos from <code>processed_v2/</code> under the active data root.
Feature values are raw (pre-normalization); normalization happens at model input time.
</p>
<h2>Selected demos</h2>
{summary_table(stems, manifest_index, schema="v2")}
{"".join(sections)}
</body>
</html>"""


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manual QA report for parquet data.")
    parser.add_argument("--per-map", type=int, default=2, help="How many demos to sample per map.")
    parser.add_argument("--open", action="store_true", help="Open the generated HTML report in a browser.")
    parser.add_argument(
        "--schema",
        choices=["v1", "v2"],
        default="v1",
        help="Which parquet schema to visualize (default: v1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_index = load_manifest_index()

    if args.schema == "v2":
        stems = select_demo_stems_v2(args.per_map)
        print(f"Selected {len(stems)} v2 demos for QA.")
        html_doc = build_report_v2(stems, manifest_index)
        OUT_HTML_V2.parent.mkdir(parents=True, exist_ok=True)
        OUT_HTML_V2.write_text(html_doc, encoding="utf-8")
        print(f"\nSaved v2 report: {OUT_HTML_V2}")
        if args.open:
            webbrowser.open(OUT_HTML_V2.as_uri())
    else:
        stems = select_demo_stems(args.per_map)
        print(f"Selected {len(stems)} demos for QA.")
        html_doc = build_report(stems, manifest_index)
        OUT_HTML.write_text(html_doc, encoding="utf-8")
        print(f"\nSaved report: {OUT_HTML}")
        if args.open:
            webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()
