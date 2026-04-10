"""
Visualize parquet content for a manual QA sample.

Default behavior:
- pick 2 demos per map from data/processed
- require matching raw demo in data/raw/demos
- write HTML report to data/viz_report.html

Usage:
    python viz_parquet.py
    python viz_parquet.py --per-map 2 --open
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


ROOT = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"
RAW_DEMOS = ROOT / "data" / "raw" / "demos"
MANIFEST = ROOT / "data" / "raw" / "manifest.jsonl"
OUT_HTML = ROOT / "data" / "viz_report.html"

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


def summary_table(stems: list[str], manifest_index: dict[str, dict[str, str]]) -> str:
    rows = []
    for stem in stems:
        meta = manifest_index.get(stem, {})
        map_name = stem.split("_", 1)[1]
        rows.append(
            "<tr>"
            f"<td>{html.escape(map_name)}</td>"
            f"<td>{html.escape(stem)}</td>"
            f"<td>{html.escape(meta.get('date', ''))}</td>"
            f"<td>{html.escape(meta.get('event', ''))}</td>"
            f"<td>{html.escape(str(RAW_DEMOS / f'{stem}.dem'))}</td>"
            f"<td>{html.escape(str(PROCESSED / f'{stem}.parquet'))}</td>"
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
Selected 2 demos per map from <code>data/processed</code> when a matching raw demo exists in
<code>data/raw/demos</code>. Use the raw demo path to cross-check the parquet labels manually.
</p>
<h2>Selected demos</h2>
{summary_table(stems, manifest_index)}
{"".join(sections)}
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manual QA report for parquet data.")
    parser.add_argument("--per-map", type=int, default=2, help="How many demos to sample per map.")
    parser.add_argument("--open", action="store_true", help="Open the generated HTML report in a browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_index = load_manifest_index()
    stems = select_demo_stems(args.per_map)
    print(f"Selected {len(stems)} demos for QA.")

    html_doc = build_report(stems, manifest_index)
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    print(f"\nSaved report: {OUT_HTML}")

    if args.open:
        webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()
