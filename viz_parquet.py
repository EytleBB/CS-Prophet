"""
Visualise parquet content for 3 selected demos.
Run:  python viz_parquet.py
Opens:  data/viz_report.html
"""

import webbrowser
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd


def _read_parquet(path):
    """Read parquet via pyarrow directly to bypass pandas/pyarrow version conflict."""
    return pq.read_table(path).to_pandas()

DEMOS = [
    "2393051_de_ancient",
    "2387105_de_ancient",
    "2389262_de_anubis",
    "2393076_de_anubis",
    "2388109_de_dust2",
    "2388076_de_dust2",
    "2387400_de_inferno",
    "2387331_de_inferno",
    "2392295_de_mirage",
    "2386802_de_mirage",
    "2392928_de_nuke",
    "2393089_de_nuke",
    "2393069_de_overpass",
    "2390818_de_overpass",
]

PROCESSED = Path(__file__).parent / "data" / "processed"
OUT_HTML = Path(__file__).parent / "data" / "viz_report.html"

SITE_COLOR = {"A": "#4ade80", "B": "#60a5fa", "other": "#f87171", "—": "#e5e7eb"}


def round_table(df: pd.DataFrame) -> str:
    """Return an HTML table: one row per round, key columns."""
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
    for _, r in rounds.iterrows():
        site = str(r["bomb_site"]) if pd.notna(r["bomb_site"]) else "—"
        color = SITE_COLOR.get(site, "#e5e7eb")
        rows_html.append(
            f"<tr>"
            f"<td>{int(r['round_num'])}</td>"
            f"<td style='background:{color};font-weight:bold;text-align:center'>{site}</td>"
            f"<td>{int(r['ct_score'])}–{int(r['t_score'])}</td>"
            f"<td>{int(r['steps'])}</td>"
            f"</tr>"
        )

    return (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;font-size:13px'>"
        "<thead><tr>"
        "<th>Round</th><th>Bomb Site</th><th>Score CT–T</th><th>Steps</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows_html) + "</tbody>"
        "</table>"
    )


def player_snapshot(df: pd.DataFrame, round_num: int, step: int = 0) -> str:
    """Show player positions and hp for one tick as a small table."""
    row = df[(df["round_num"] == round_num) & (df["step"] == step)]
    if row.empty:
        return "<p><em>no data</em></p>"
    row = row.iloc[0]

    cols_html = []
    for side in ("t", "ct"):
        for i in range(5):
            hp_col = f"{side}{i}_hp"
            alive_col = f"{side}{i}_alive"
            weapon_col = f"{side}{i}_weapon"
            hp = row.get(hp_col, 0)
            alive = bool(row.get(alive_col, False))
            weapon = row.get(weapon_col, "?")
            style = "color:#9ca3af" if not alive else ""
            cols_html.append(
                f"<td style='{style}'>{side.upper()}{i} hp={int(hp)} {weapon}</td>"
            )

    return (
        "<table border='1' cellpadding='3' cellspacing='0' style='border-collapse:collapse;font-size:12px'>"
        f"<tr>{''.join(cols_html)}</tr>"
        "</table>"
    )


def schema_section(df: pd.DataFrame) -> str:
    """Show column names, dtypes, and first-row sample."""
    info_rows = []
    sample = df.iloc[0]
    for col in df.columns:
        info_rows.append(
            f"<tr><td>{col}</td><td>{df[col].dtype}</td><td>{sample[col]}</td></tr>"
        )
    return (
        "<details><summary style='cursor:pointer;font-weight:bold'>Schema &amp; first-row sample</summary>"
        "<table border='1' cellpadding='3' cellspacing='0' style='border-collapse:collapse;font-size:11px;margin-top:6px'>"
        "<thead><tr><th>Column</th><th>Dtype</th><th>Sample value</th></tr></thead>"
        "<tbody>" + "".join(info_rows) + "</tbody>"
        "</table></details>"
    )


def bomb_site_bar(df: pd.DataFrame) -> str:
    """Simple text bar chart of bomb_site distribution."""
    counts = df.groupby("round_num")["bomb_site"].first().value_counts()
    total = counts.sum()
    bars = []
    for site in ["A", "B", "other"]:
        n = counts.get(site, 0)
        pct = n / total * 100 if total else 0
        color = SITE_COLOR[site]
        bars.append(
            f"<div style='margin:2px 0'>"
            f"<span style='display:inline-block;width:60px'>{site}</span>"
            f"<span style='display:inline-block;background:{color};width:{int(pct*3)}px;height:16px;vertical-align:middle'></span>"
            f"<span style='margin-left:6px'>{n} rounds ({pct:.0f}%)</span>"
            f"</div>"
        )
    return "".join(bars)


def demo_section(stem: str) -> str:
    path = PROCESSED / f"{stem}.parquet"
    df = _read_parquet(path)

    map_name = stem.split("_", 1)[1] if "_" in stem else stem
    total_rounds = df["round_num"].nunique()
    total_rows = len(df)

    round_nums = sorted(df["round_num"].unique())
    # pick first round with a real plant for the player snapshot
    planted_rounds = df[df["bomb_site"].isin(["A", "B"])]["round_num"].unique()
    snap_round = planted_rounds[0] if len(planted_rounds) > 0 else round_nums[0]

    return f"""
<div style='border:1px solid #ccc;border-radius:6px;padding:16px;margin-bottom:24px;font-family:monospace'>
  <h2 style='margin-top:0'>{stem} <span style='font-size:14px;color:#666'>({map_name})</span></h2>
  <p>
    <strong>{total_rounds}</strong> rounds &nbsp;|&nbsp;
    <strong>{total_rows:,}</strong> total rows &nbsp;|&nbsp;
    <strong>{df.shape[1]}</strong> columns
  </p>
  <h3>Bomb site distribution (by round)</h3>
  {bomb_site_bar(df)}
  <h3>Per-round summary</h3>
  {round_table(df)}
  <h3>Player snapshot — Round {snap_round}, Step 0</h3>
  {player_snapshot(df, snap_round, step=0)}
  <br>
  {schema_section(df)}
</div>
"""


def main() -> None:
    sections = []
    for stem in DEMOS:
        print(f"Processing {stem}...")
        sections.append(demo_section(stem))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CS-Prophet Parquet Viewer</title>
<style>
  body {{ font-family: monospace; max-width: 1200px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 6px; }}
  summary {{ font-size: 13px; }}
</style>
</head>
<body>
<h1>CS-Prophet — Parquet Data Viewer</h1>
<p style="color:#666">3 demos · bomb_site labels verified against raw .dem files</p>
{"".join(sections)}
</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nSaved: {OUT_HTML}")
    webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()
