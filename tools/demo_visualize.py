"""Generate interactive HTML visualization report from extracted demo data.

Usage:
    python tools/demo_visualize.py viz/2389471_de_mirage_full.pkl
    python tools/demo_visualize.py viz/2389471_de_mirage_full.pkl --rounds 1,3,5
"""
from __future__ import annotations

import argparse
import base64
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_path, resolve_path_input

# ── Styles ────────────────────────────────────────────────────────────────
T_COLOR = "#e74c3c"   # red
CT_COLOR = "#3498db"   # blue
BG_COLOR = "#1a1a2e"
PAPER_COLOR = "#16213e"
GRID_COLOR = "#2a2a4a"

# ── Radar image metadata (CS2 standard) ───────────────────────────────────
# pos_x = game X at left edge, pos_y = game Y at top edge, scale = game units/pixel
RADAR_META = {
    "de_mirage": {"pos_x": -3230, "pos_y": 1713, "scale": 5.0},
    "de_inferno": {"pos_x": -2087, "pos_y": 3870, "scale": 4.9},
    "de_dust2": {"pos_x": -2476, "pos_y": 3239, "scale": 4.4},
    "de_nuke": {"pos_x": -3453, "pos_y": 2887, "scale": 7.0},
    "de_ancient": {"pos_x": -2953, "pos_y": 2164, "scale": 5.0},
    "de_overpass": {"pos_x": -4831, "pos_y": 1781, "scale": 5.2},
    "de_anubis": {"pos_x": -2796, "pos_y": 3328, "scale": 5.22},
}

ASSETS_DIR = data_path("viz", "assets", prefer_existing=True)


def _load_radar_b64(map_name: str) -> str | None:
    """Load radar PNG as base64 data URI."""
    img_path = ASSETS_DIR / f"{map_name}.png"
    if not img_path.exists():
        return None
    with open(img_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# ── Team identity (handle side swap at halftime) ─────────────────────────

def detect_teams(tick_df: pd.DataFrame) -> dict:
    """Detect actual team rosters from round 1 and return mapping.

    Returns dict with:
        "team_a_players": set of player names (CT in round 1)
        "team_b_players": set of player names (T in round 1)
        "team_a_name": str (clan name or "Team A")
        "team_b_name": str (clan name or "Team B")
    """
    r1 = tick_df[tick_df["round_num"] == 1]
    if r1.empty:
        return {"team_a_players": set(), "team_b_players": set(),
                "team_a_name": "Team A", "team_b_name": "Team B"}

    first = r1.drop_duplicates("name")
    ct_players = set(first[first["team_name"] == "CT"]["name"])
    t_players = set(first[first["team_name"] == "TERRORIST"]["name"])

    # Try to get clan names
    a_name = "Team A"
    b_name = "Team B"
    if "clan_name" in first.columns:
        ct_clans = first[first["team_name"] == "CT"]["clan_name"].dropna().unique()
        t_clans = first[first["team_name"] == "TERRORIST"]["clan_name"].dropna().unique()
        if len(ct_clans) == 1 and ct_clans[0]:
            a_name = ct_clans[0]
        if len(t_clans) == 1 and t_clans[0]:
            b_name = t_clans[0]

    return {
        "team_a_players": ct_players,   # CT side in round 1
        "team_b_players": t_players,    # T side in round 1
        "team_a_name": a_name,
        "team_b_name": b_name,
    }


TEAM_A_COLOR = "#3498db"  # blue — CT side in round 1
TEAM_B_COLOR = "#e74c3c"  # red  — T side in round 1


def get_team_label(name: str, teams: dict) -> str:
    """Get 'A' or 'B' team label for a player."""
    if name in teams["team_a_players"]:
        return "A"
    return "B"


def get_team_color(name: str, teams: dict) -> str:
    if name in teams["team_a_players"]:
        return TEAM_A_COLOR
    return TEAM_B_COLOR


def get_side(name: str, round_num: int, tick_df: pd.DataFrame) -> str:
    """Get current side (CT/TERRORIST) for a player in a given round."""
    rdf = tick_df[(tick_df["round_num"] == round_num) & (tick_df["name"] == name)]
    if rdf.empty:
        return "?"
    return rdf["team_name"].iloc[0]


# ── Weapon classification for "best weapon" feature ─────────────────────
_PRIMARY_WEAPONS = {
    # Rifles
    "AK-47", "M4A4", "M4A1-S", "FAMAS", "Galil AR", "SG 553", "AUG",
    # Snipers
    "AWP", "SSG 08", "SCAR-20", "G3SG1",
    # SMGs
    "MP9", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "MAC-10", "MP7",
    # Heavy
    "Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev",
}

_SECONDARY_WEAPONS = {
    "Glock-18", "USP-S", "P2000", "P250", "Five-SeveN", "Tec-9",
    "CZ75-Auto", "Desert Eagle", "R8 Revolver", "Dual Berettas",
}

_IGNORE_ITEMS = {
    "C4 Explosive",
    "Smoke Grenade", "Flashbang", "High Explosive Grenade",
    "HE Grenade", "Incendiary Grenade", "Molotov", "Decoy Grenade",
}


def _is_knife(name: str) -> bool:
    n = name.lower()
    return "knife" in n or "bayonet" in n


def best_weapon(inventory) -> str:
    """Pick best weapon from inventory: primary > secondary. Never knife/grenade/C4.

    Returns weapon name string or "none".
    """
    if not isinstance(inventory, (list, np.ndarray)):
        return "none"
    primary = None
    secondary = None
    for item in inventory:
        s = str(item)
        if s in _PRIMARY_WEAPONS:
            primary = s
        elif s in _SECONDARY_WEAPONS:
            secondary = s
    return primary or secondary or "none"


GRENADE_COLORS = {
    "flashbang_detonate": "#f1c40f",
    "smokegrenade_detonate": "#95a5a6",
    "hegrenade_detonate": "#e74c3c",
    "inferno_startburn": "#e67e22",
}
GRENADE_SYMBOLS = {
    "flashbang_detonate": "star",
    "smokegrenade_detonate": "circle",
    "hegrenade_detonate": "diamond",
    "inferno_startburn": "triangle-up",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color="#ecf0f1", size=12),
    margin=dict(l=60, r=30, t=50, b=40),
)


def styled_fig(fig: go.Figure, title: str = "", height: int = 500) -> go.Figure:
    fig.update_layout(**LAYOUT_DEFAULTS, title=title, height=height)
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def fig_to_html(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def _team_iter(tick_df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return [(team_name, color), ...] for the two teams in tick_df."""
    teams = tick_df["team_name"].unique()
    # Assign colors: first team gets TEAM_A_COLOR, second gets TEAM_B_COLOR
    result = []
    for i, t in enumerate(sorted(teams)):
        result.append((t, TEAM_A_COLOR if i == 0 else TEAM_B_COLOR))
    return result


# ── Section builders ──────────────────────────────────────────────────────

def section_round_overview(round_info: pd.DataFrame, tick_df: pd.DataFrame) -> str:
    """Section 1: Round overview table + economy bar chart."""
    html_parts = ['<h2>1. Round Overview</h2>']

    # Derive real scores from CCSTeam.m_iScore using original side (CT/T)
    score_col = "CCSTeam.m_iScore"
    side_col = "side" if "side" in tick_df.columns else "team_name"
    teams_list = _team_iter(tick_df)
    team_a_name = teams_list[0][0] if teams_list else "Team A"
    team_b_name = teams_list[1][0] if len(teams_list) > 1 else "Team B"

    if not tick_df.empty and score_col in tick_df.columns:
        ri = round_info.copy()
        # Rename score columns to actual team names
        ri[team_a_name] = 0
        ri[team_b_name] = 0
        for _, rrow in ri.iterrows():
            rn = rrow["round_num"]
            rdf = tick_df[tick_df["round_num"] == rn]
            if rdf.empty:
                continue
            # Use team_name (now = actual team) to group
            first_tick = rdf.sort_values("tick").groupby("team_name").first().reset_index()
            for _, prow in first_tick.iterrows():
                tname = prow["team_name"]
                sc = prow.get(score_col, None)
                if sc is not None:
                    ri.loc[ri["round_num"] == rn, tname] = int(sc)
        # Drop old ct_score/t_score, use actual team names
        ri = ri.drop(columns=["ct_score", "t_score"], errors="ignore")
        round_info = ri

    # Table
    cols_show = [c for c in ["round_num", team_a_name, team_b_name, "site", "freeze_tick", "plant_tick", "end_tick"] if c in round_info.columns]
    if "freeze_tick" in round_info.columns and "end_tick" in round_info.columns:
        round_info = round_info.copy()
        round_info["duration_s"] = ((round_info["end_tick"] - round_info["freeze_tick"]) / 64).round(1)
        cols_show.append("duration_s")

    table_html = round_info[cols_show].to_html(index=False, classes="styled-table", border=0)
    html_parts.append(table_html)

    # Economy per round
    if not tick_df.empty and "round_num" in tick_df.columns and "round_start_equip_value" in tick_df.columns:
        first_ticks = tick_df.groupby(["round_num", "team_name"]).first().reset_index()
        econ = first_ticks.groupby(["round_num", "team_name"])["round_start_equip_value"].sum().unstack(fill_value=0)

        fig = go.Figure()
        for team, color in _team_iter(tick_df):
            if team in econ.columns:
                fig.add_trace(go.Bar(x=econ.index, y=econ[team], name=team, marker_color=color))
        styled_fig(fig, "Team Equipment Value per Round", 400)
        fig.update_layout(barmode="group", xaxis_title="Round", yaxis_title="Total Equip Value")
        html_parts.append(fig_to_html(fig))

    return "\n".join(html_parts)


def section_positions(tick_df: pd.DataFrame, events: dict, rn: int, map_name: str = "") -> str:
    """Section 2: 2D position map for a specific round."""
    html_parts = [f'<h3>Round {rn} — Player Trajectories</h3>']
    rdf = tick_df[tick_df["round_num"] == rn]
    if rdf.empty:
        html_parts.append("<p>No data for this round.</p>")
        return "\n".join(html_parts)

    fig = go.Figure()

    # Add radar background image
    radar_b64 = _load_radar_b64(map_name) if map_name else None
    meta = RADAR_META.get(map_name)
    if radar_b64 and meta:
        img_size = 1024
        x0 = meta["pos_x"]
        y0 = meta["pos_y"] - img_size * meta["scale"]  # bottom (game Y)
        x1 = meta["pos_x"] + img_size * meta["scale"]
        y1 = meta["pos_y"]  # top (game Y)
        fig.add_layout_image(
            source=radar_b64,
            x=x0, y=y1,
            sizex=x1 - x0, sizey=y1 - y0,
            xref="x", yref="y",
            sizing="stretch",
            opacity=0.6,
            layer="below",
        )

    # Player trajectories
    for team, color in _team_iter(tick_df):
        team_df = rdf[rdf["team_name"] == team]
        for name, pdf in team_df.groupby("name"):
            fig.add_trace(go.Scatter(
                x=pdf["X"], y=pdf["Y"],
                mode="lines+markers",
                marker=dict(size=2, color=color, opacity=0.5),
                line=dict(color=color, width=1),
                name=f"{name} ({team[0]})",
                hovertext=pdf.get("last_place_name", ""),
            ))

    # Grenade detonations
    for ev_name, color in GRENADE_COLORS.items():
        ev_df = events.get(ev_name, pd.DataFrame())
        if not ev_df.empty and "round_num" in ev_df.columns:
            gdf = ev_df[ev_df["round_num"] == rn]
            if not gdf.empty and "x" in gdf.columns:
                fig.add_trace(go.Scatter(
                    x=gdf["x"], y=gdf["y"],
                    mode="markers",
                    marker=dict(size=12, color=color, symbol=GRENADE_SYMBOLS[ev_name], line=dict(width=1, color="white")),
                    name=ev_name.replace("_", " ").title(),
                ))

    # Kill locations
    death_df = events.get("player_death", pd.DataFrame())
    if not death_df.empty and "round_num" in death_df.columns:
        kill_df = death_df[death_df["round_num"] == rn]
        if not kill_df.empty:
            # Find victim positions from tick_df at closest tick
            kill_markers_x, kill_markers_y, kill_labels = [], [], []
            for _, krow in kill_df.iterrows():
                ktick = int(krow["tick"])
                victim = krow.get("user_name", "?")
                # find closest tick in rdf for this player
                vdf = rdf[(rdf["name"] == victim) & (rdf["tick"] <= ktick)]
                if not vdf.empty:
                    last = vdf.iloc[-1]
                    kill_markers_x.append(last["X"])
                    kill_markers_y.append(last["Y"])
                    hs = " (HS)" if krow.get("headshot", False) else ""
                    kill_labels.append(f"{krow.get('attacker_name', '?')} → {victim}{hs}")
            if kill_markers_x:
                fig.add_trace(go.Scatter(
                    x=kill_markers_x, y=kill_markers_y,
                    mode="markers+text",
                    marker=dict(size=14, color="white", symbol="x", line=dict(width=2, color="red")),
                    text=kill_labels, textposition="top center", textfont=dict(size=9, color="white"),
                    name="Kills",
                ))

    styled_fig(fig, f"Round {rn} — Map View (X vs Y)", 700)
    fig.update_layout(xaxis_title="X", yaxis_title="Y", yaxis_scaleanchor="x")
    html_parts.append(fig_to_html(fig))
    return "\n".join(html_parts)


def section_view_angles(tick_df: pd.DataFrame, rn: int) -> str:
    """Section 3: View angles (yaw) over time."""
    html_parts = [f'<h3>Round {rn} — View Angles</h3>']
    rdf = tick_df[tick_df["round_num"] == rn]
    if rdf.empty or "yaw" not in rdf.columns:
        html_parts.append("<p>No yaw data.</p>")
        return "\n".join(html_parts)

    fig = go.Figure()
    for team, color in _team_iter(tick_df):
        team_df = rdf[rdf["team_name"] == team]
        for name, pdf in team_df.groupby("name"):
            fig.add_trace(go.Scatter(
                x=pdf["tick"], y=pdf["yaw"],
                mode="lines", line=dict(width=1),
                name=f"{name} ({team[0]})",
                opacity=0.7,
            ))
    styled_fig(fig, f"Round {rn} — Yaw over Time", 400)
    fig.update_layout(xaxis_title="Tick", yaxis_title="Yaw (degrees)")
    html_parts.append(fig_to_html(fig))

    return "\n".join(html_parts)


def section_velocity(tick_df: pd.DataFrame, rn: int) -> str:
    """Section 4: Velocity & movement states."""
    html_parts = [f'<h3>Round {rn} — Velocity & Movement</h3>']
    rdf = tick_df[tick_df["round_num"] == rn].copy()
    if rdf.empty:
        html_parts.append("<p>No data.</p>")
        return "\n".join(html_parts)

    # Velocity magnitude
    has_vel = all(c in rdf.columns for c in ["velocity_X", "velocity_Y", "velocity_Z"])
    if has_vel:
        rdf["speed"] = np.sqrt(
            rdf["velocity_X"].fillna(0)**2 +
            rdf["velocity_Y"].fillna(0)**2 +
            rdf["velocity_Z"].fillna(0)**2
        )
        fig = go.Figure()
        for team, color in _team_iter(tick_df):
            team_df = rdf[rdf["team_name"] == team]
            for name, pdf in team_df.groupby("name"):
                fig.add_trace(go.Scatter(
                    x=pdf["tick"], y=pdf["speed"],
                    mode="lines", line=dict(width=1),
                    name=f"{name} ({team[0]})", opacity=0.7,
                ))
        styled_fig(fig, f"Round {rn} — Player Speed", 400)
        fig.update_layout(xaxis_title="Tick", yaxis_title="Speed (units/tick)")
        html_parts.append(fig_to_html(fig))

    # Movement states heatmap
    state_cols = [c for c in ["is_walking", "ducking"] if c in rdf.columns]
    if state_cols:
        # Aggregate: fraction of players in each state per tick
        state_agg = rdf.groupby("tick")[state_cols].mean()
        fig2 = go.Figure()
        colors = ["#2ecc71", "#9b59b6", "#e67e22"]
        for i, col in enumerate(state_cols):
            fig2.add_trace(go.Scatter(
                x=state_agg.index, y=state_agg[col],
                mode="lines", fill="tozeroy",
                name=col, line=dict(color=colors[i % len(colors)]),
                opacity=0.6,
            ))
        styled_fig(fig2, f"Round {rn} — Movement States (fraction of players)", 350)
        fig2.update_layout(xaxis_title="Tick", yaxis_title="Fraction")
        html_parts.append(fig_to_html(fig2))

    return "\n".join(html_parts)


def section_health_combat(tick_df: pd.DataFrame, events: dict, rn: int) -> str:
    """Section 5: HP curves + hurt/death events."""
    html_parts = [f'<h3>Round {rn} — Health & Combat</h3>']
    rdf = tick_df[tick_df["round_num"] == rn]
    if rdf.empty:
        html_parts.append("<p>No data.</p>")
        return "\n".join(html_parts)

    fig = go.Figure()
    for team, base_color in _team_iter(tick_df):
        team_df = rdf[rdf["team_name"] == team]
        for name, pdf in team_df.groupby("name"):
            fig.add_trace(go.Scatter(
                x=pdf["tick"], y=pdf["health"],
                mode="lines", line=dict(width=2),
                name=f"{name} ({team[0]})",
            ))

    # Hurt events
    hurt_df = events.get("player_hurt", pd.DataFrame())
    if not hurt_df.empty and "round_num" in hurt_df.columns:
        hdf = hurt_df[hurt_df["round_num"] == rn]
        if not hdf.empty and "health" in hdf.columns:
            fig.add_trace(go.Scatter(
                x=hdf["tick"], y=hdf["health"],
                mode="markers",
                marker=dict(size=hdf.get("dmg_health", pd.Series([10]*len(hdf))).clip(5, 30), color="orange", opacity=0.6),
                name="Damage Taken",
                hovertext=hdf.apply(lambda r: f"{r.get('attacker_name','?')}→{r.get('user_name','?')} ({r.get('dmg_health',0)} dmg)", axis=1),
            ))

    # Death events
    death_df = events.get("player_death", pd.DataFrame())
    if not death_df.empty and "round_num" in death_df.columns:
        ddf = death_df[death_df["round_num"] == rn]
        if not ddf.empty:
            fig.add_trace(go.Scatter(
                x=ddf["tick"], y=[0] * len(ddf),
                mode="markers+text",
                marker=dict(size=14, color="white", symbol="x"),
                text=ddf.apply(lambda r: f"{r.get('attacker_name','?')}→{r.get('user_name','?')}", axis=1),
                textposition="top center", textfont=dict(size=8),
                name="Deaths",
            ))

    styled_fig(fig, f"Round {rn} — Health over Time", 500)
    fig.update_layout(xaxis_title="Tick", yaxis_title="HP")
    html_parts.append(fig_to_html(fig))
    return "\n".join(html_parts)


def section_economy(round_info: pd.DataFrame, tick_df: pd.DataFrame) -> str:
    """Section 6: Economy across all rounds."""
    html_parts = ['<h2>6. Economy</h2>']
    if tick_df.empty or "balance" not in tick_df.columns:
        html_parts.append("<p>No economy data.</p>")
        return "\n".join(html_parts)

    # Average balance per team at round start (first tick of each round)
    first = tick_df.sort_values("tick").groupby(["round_num", "team_name", "name"]).first().reset_index()
    bal = first.groupby(["round_num", "team_name"])["balance"].mean().unstack(fill_value=0)

    fig = go.Figure()
    for team, color in _team_iter(tick_df):
        if team in bal.columns:
            fig.add_trace(go.Scatter(x=bal.index, y=bal[team], mode="lines+markers",
                                     name=f"{team} avg balance", line=dict(color=color)))
    styled_fig(fig, "Average Player Balance per Round", 400)
    fig.update_layout(xaxis_title="Round", yaxis_title="Balance ($)")
    html_parts.append(fig_to_html(fig))

    # Current equip value
    if "current_equip_value" in first.columns:
        equip = first.groupby(["round_num", "team_name"])["current_equip_value"].sum().unstack(fill_value=0)
        fig2 = go.Figure()
        for team, color in _team_iter(tick_df):
            if team in equip.columns:
                fig2.add_trace(go.Bar(x=equip.index, y=equip[team], name=team, marker_color=color))
        styled_fig(fig2, "Team Loadout Value per Round", 400)
        fig2.update_layout(barmode="group", xaxis_title="Round", yaxis_title="Current Equip Value ($)")
        html_parts.append(fig_to_html(fig2))

    return "\n".join(html_parts)


def section_weapons(tick_df: pd.DataFrame, events: dict) -> str:
    """Section 7: Weapons & utility."""
    html_parts = ['<h2>7. Weapons & Utility</h2>']

    # Best weapon distribution per round (primary > secondary, no knives)
    wpn_col = "best_weapon" if "best_weapon" in tick_df.columns else "weapon_name"
    if not tick_df.empty and wpn_col in tick_df.columns:
        first = tick_df.sort_values("tick").groupby(["round_num", "name"]).first().reset_index()
        wpn_counts = first.groupby(["round_num", wpn_col]).size().unstack(fill_value=0)
        # Drop "none" from chart
        if "none" in wpn_counts.columns:
            wpn_counts = wpn_counts.drop(columns=["none"])
        fig = go.Figure()
        for wpn in wpn_counts.columns:
            fig.add_trace(go.Bar(x=wpn_counts.index, y=wpn_counts[wpn], name=str(wpn)))
        styled_fig(fig, "Best Weapon per Round (primary > secondary, no knife)", 450)
        fig.update_layout(barmode="stack", xaxis_title="Round", yaxis_title="# Players")
        html_parts.append(fig_to_html(fig))

    # Utility & C4 distribution per round (from inventory)
    if not tick_df.empty and "inventory" in tick_df.columns:
        _UTIL_ITEMS = {"Smoke Grenade", "Flashbang", "High Explosive Grenade",
                       "HE Grenade", "Incendiary Grenade", "Molotov",
                       "Decoy Grenade", "C4 Explosive"}
        first = tick_df.sort_values("tick").groupby(["round_num", "name"]).first().reset_index()
        util_rows = []
        for _, row in first.iterrows():
            inv = row.get("inventory", [])
            if not isinstance(inv, (list, np.ndarray)):
                continue
            for item in inv:
                if str(item) in _UTIL_ITEMS:
                    util_rows.append({"round_num": row["round_num"], "item": str(item)})
        if util_rows:
            util_df = pd.DataFrame(util_rows)
            util_counts = util_df.groupby(["round_num", "item"]).size().unstack(fill_value=0)
            fig_u = go.Figure()
            util_colors = {"Smoke Grenade": "#95a5a6", "Flashbang": "#f1c40f",
                           "High Explosive Grenade": "#e74c3c", "HE Grenade": "#e74c3c",
                           "Incendiary Grenade": "#e67e22", "Molotov": "#e67e22",
                           "Decoy Grenade": "#8e44ad", "C4 Explosive": "#f39c12"}
            for item in util_counts.columns:
                fig_u.add_trace(go.Bar(
                    x=util_counts.index, y=util_counts[item], name=str(item),
                    marker_color=util_colors.get(str(item), "#bdc3c7"),
                ))
            styled_fig(fig_u, "Utility & C4 Distribution per Round", 400)
            fig_u.update_layout(barmode="stack", xaxis_title="Round", yaxis_title="Count")
            html_parts.append(fig_to_html(fig_u))

    # Grenade timeline
    thrown_df = events.get("grenade_thrown", pd.DataFrame())
    if not thrown_df.empty and "round_num" in thrown_df.columns:
        fig2 = go.Figure()
        nade_colors = {"smokegrenade": "#95a5a6", "flashbang": "#f1c40f",
                       "hegrenade": "#e74c3c", "incgrenade": "#e67e22",
                       "molotov": "#e67e22", "decoy": "#8e44ad"}
        for wpn, gdf in thrown_df.groupby("weapon"):
            c = nade_colors.get(str(wpn), "#bdc3c7")
            fig2.add_trace(go.Scatter(
                x=gdf["tick"], y=gdf.get("user_name", range(len(gdf))),
                mode="markers",
                marker=dict(size=8, color=c),
                name=str(wpn),
            ))
        styled_fig(fig2, "Grenade Thrown Timeline", 400)
        fig2.update_layout(xaxis_title="Tick", yaxis_title="Player")
        html_parts.append(fig_to_html(fig2))

    return "\n".join(html_parts)


def section_info_status(tick_df: pd.DataFrame, rn: int) -> str:
    """Section 8: Spotted & defuser status."""
    html_parts = [f'<h3>Round {rn} — Information & Status</h3>']
    rdf = tick_df[tick_df["round_num"] == rn]
    if rdf.empty:
        html_parts.append("<p>No data.</p>")
        return "\n".join(html_parts)

    # Spotted heatmap
    if "spotted" in rdf.columns:
        pivot = rdf.pivot_table(index="name", columns="tick", values="spotted", aggfunc="first")
        fig = go.Figure(go.Heatmap(
            z=pivot.values.astype(float),
            x=pivot.columns, y=pivot.index,
            colorscale=[[0, "#1a1a2e"], [1, "#e74c3c"]],
            showscale=False,
        ))
        styled_fig(fig, f"Round {rn} — Spotted Status", 400)
        fig.update_layout(xaxis_title="Tick", yaxis_title="Player")
        html_parts.append(fig_to_html(fig))

    return "\n".join(html_parts)


def section_named_locations(tick_df: pd.DataFrame, rn: int) -> str:
    """Section 9: Named location analysis."""
    html_parts = [f'<h3>Round {rn} — Named Locations</h3>']
    rdf = tick_df[tick_df["round_num"] == rn]
    if rdf.empty or "last_place_name" not in rdf.columns:
        html_parts.append("<p>No location data.</p>")
        return "\n".join(html_parts)

    # Time-series: each player's location over time (categorical color)
    players = rdf["name"].unique()
    locations = rdf["last_place_name"].dropna().unique()
    loc_to_num = {loc: i for i, loc in enumerate(sorted(locations))}

    fig = go.Figure()
    for name in players:
        pdf = rdf[rdf["name"] == name]
        team = pdf["team_name"].iloc[0] if not pdf.empty else "?"
        loc_nums = pdf["last_place_name"].map(loc_to_num)
        fig.add_trace(go.Scatter(
            x=pdf["tick"], y=loc_nums,
            mode="lines+markers", marker=dict(size=3),
            name=f"{name} ({team})",
            hovertext=pdf["last_place_name"],
        ))
    styled_fig(fig, f"Round {rn} — Player Locations over Time", 500)
    fig.update_layout(
        xaxis_title="Tick",
        yaxis=dict(
            tickmode="array",
            tickvals=list(loc_to_num.values()),
            ticktext=list(loc_to_num.keys()),
        ),
    )
    html_parts.append(fig_to_html(fig))

    # Global location time distribution
    all_locs = tick_df["last_place_name"].value_counts().head(20)
    fig2 = go.Figure(go.Bar(x=all_locs.values, y=all_locs.index, orientation="h", marker_color="#2ecc71"))
    styled_fig(fig2, "Top 20 Locations by Tick Count (All Rounds)", 500)
    fig2.update_layout(xaxis_title="Tick Count", yaxis_title="Location")
    html_parts.append(fig_to_html(fig2))

    return "\n".join(html_parts)


def section_player_snapshot(tick_df: pd.DataFrame, rn: int) -> str:
    """Single-player state snapshot from the middle of a round."""
    html_parts = [f'<h3>Round {rn} — Player State Snapshot</h3>']
    rdf = tick_df[tick_df["round_num"] == rn].sort_values("tick")
    if rdf.empty:
        html_parts.append("<p>No data.</p>")
        return "\n".join(html_parts)

    # Pick middle tick of the round
    ticks = rdf["tick"].unique()
    mid_tick = ticks[len(ticks) // 2]
    snap = rdf[rdf["tick"] == mid_tick]
    if snap.empty:
        html_parts.append("<p>No data at mid-tick.</p>")
        return "\n".join(html_parts)

    # Pick one player (first alive, prefer an interesting one)
    alive = snap[snap["is_alive"] == True]
    if alive.empty:
        alive = snap
    player_row = alive.iloc[0]
    player_name = player_row["name"]

    # Categorize fields
    categories = {
        "Identity": ["name", "team_name", "side", "player_steamid", "steamid"],
        "Position & Movement": ["X", "Y", "Z", "velocity_X", "velocity_Y", "velocity_Z",
                                "yaw", "is_walking", "ducking",
                                "in_bomb_zone"],
        "Vitals & Status": ["health", "armor_value", "has_helmet", "is_alive",
                           "spotted", "is_scoped", "flash_duration"],
        "Weapons & Utility": ["best_weapon", "weapon_name", "inventory"],
        "Economy": ["balance", "current_equip_value"],
        "Stats": ["score"],
        "Round Context": ["round_num", "tick", "game_time", "total_rounds_played"],
    }

    time_s = (mid_tick - ticks[0]) / 64
    html_parts.append(f'<p style="color:#95a5a6">Player: <b style="color:#f39c12">{player_name}</b> '
                      f'| Tick: {mid_tick} ({time_s:.1f}s into round) '
                      f'| {len(ticks)} ticks in round</p>')

    html_parts.append('<div style="display:flex; flex-wrap:wrap; gap:20px">')
    for cat_name, fields in categories.items():
        present = [f for f in fields if f in snap.columns]
        if not present:
            continue
        html_parts.append(f'<div style="flex:1; min-width:280px; background:#1a1a2e; '
                          f'padding:12px; border-radius:6px; border:1px solid #2a2a4a">')
        html_parts.append(f'<h4 style="color:#f39c12; margin:0 0 8px 0">{cat_name}</h4>')
        html_parts.append('<table style="width:100%; font-size:12px">')
        for f in present:
            val = player_row[f]
            # Format value
            if isinstance(val, float) and not np.isnan(val):
                val_str = f"{val:.2f}"
            elif isinstance(val, list):
                val_str = ", ".join(str(v) for v in val)
            elif isinstance(val, bool) or (hasattr(val, 'item') and isinstance(val.item(), bool)):
                val_str = f'<span style="color:{"#2ecc71" if val else "#e74c3c"}">{val}</span>'
            else:
                val_str = str(val)
            html_parts.append(f'<tr><td style="color:#7f8c8d; padding:2px 6px">{f}</td>'
                              f'<td style="padding:2px 6px">{val_str}</td></tr>')
        html_parts.append('</table></div>')
    html_parts.append('</div>')

    return "\n".join(html_parts)


# ── Main assembly ─────────────────────────────────────────────────────────

CSS = """
<style>
body {
    background: #0f0f23;
    color: #ecf0f1;
    font-family: 'Segoe UI', 'Noto Sans SC', sans-serif;
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}
h1 { color: #f39c12; border-bottom: 2px solid #f39c12; padding-bottom: 10px; }
h2 { color: #3498db; margin-top: 40px; border-bottom: 1px solid #3498db; padding-bottom: 5px; }
h3 { color: #2ecc71; margin-top: 30px; }
.styled-table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 13px;
}
.styled-table th, .styled-table td {
    padding: 8px 12px;
    text-align: center;
    border: 1px solid #2a2a4a;
}
.styled-table th { background: #16213e; color: #f39c12; }
.styled-table tr:nth-child(even) { background: #1a1a2e; }
.styled-table tr:nth-child(odd) { background: #16213e; }
.round-section { margin: 20px 0; padding: 15px; background: #16213e; border-radius: 8px; }
</style>
"""


def build_report(data: dict, rounds: list[int] | None = None) -> str:
    tick_df = data["tick_df"].copy()
    events = data["events"]
    round_info = data["round_info"]
    header = data["header"]

    map_name = header.get("map_name", "unknown")
    all_rounds = sorted(round_info["round_num"].unique())

    # Detect actual team rosters from round 1, add stable "team" column
    teams = detect_teams(tick_df)
    team_a = teams["team_a_name"]
    team_b = teams["team_b_name"]
    tick_df["side"] = tick_df["team_name"]  # preserve original CT/T
    tick_df["team_name"] = tick_df["name"].apply(
        lambda n: team_a if n in teams["team_a_players"] else team_b
    )
    # Now team_name = stable team identity (e.g. "Team A" / "Team B")
    # All sections that group by team_name will use stable teams

    # Normalize knife variants → "Knife"
    if "weapon_name" in tick_df.columns:
        tick_df["weapon_name"] = tick_df["weapon_name"].apply(
            lambda w: "Knife" if isinstance(w, str) and ("knife" in w.lower() or "bayonet" in w.lower()) else w
        )

    # Compute "best weapon" from inventory: primary > secondary, never knife/grenade
    if "inventory" in tick_df.columns:
        tick_df["best_weapon"] = tick_df["inventory"].apply(best_weapon)

    if rounds is None:
        # Default: first 3 rounds
        rounds = all_rounds[:3]
    else:
        rounds = [r for r in rounds if r in all_rounds]

    parts = []
    parts.append(f"<html><head><meta charset='utf-8'><title>Demo Report — {map_name}</title>")
    parts.append('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>')
    parts.append(CSS)
    parts.append("</head><body>")
    # Detect halftime: find round where team_a switches side
    r1_side = "CT"  # team_a is CT in round 1 by definition
    halftime_round = None
    for rn in all_rounds[1:]:
        rdf = tick_df[(tick_df["round_num"] == rn)]
        sample = rdf[rdf["name"].isin(teams["team_a_players"])]
        if not sample.empty and sample["side"].iloc[0] != r1_side:
            halftime_round = rn
            break

    parts.append(f"<h1>CS2 Demo Analysis — {map_name}</h1>")
    ht_info = f" | Halftime after round {halftime_round - 1} (sides swap)" if halftime_round else ""
    parts.append(f"<p>Server: {header.get('server_name', '?')} | "
                 f"<span style='color:{TEAM_A_COLOR}'>{team_a}</span> vs "
                 f"<span style='color:{TEAM_B_COLOR}'>{team_b}</span> | "
                 f"Rounds: {len(all_rounds)}{ht_info} | "
                 f"Visualizing: {rounds}</p>")

    # Section 1: Round overview (all rounds)
    print("  Building section 1: Round Overview...")
    parts.append(section_round_overview(round_info, tick_df))

    # Section 6: Economy (all rounds)
    print("  Building section 6: Economy...")
    parts.append(section_economy(round_info, tick_df))

    # Section 7: Weapons (all rounds)
    print("  Building section 7: Weapons...")
    parts.append(section_weapons(tick_df, events))

    # Per-round sections
    for rn in rounds:
        parts.append(f'<h2>Round {rn} Detail</h2>')
        parts.append('<div class="round-section">')

        print(f"  Building round {rn} sections...")
        parts.append(section_player_snapshot(tick_df, rn))
        parts.append(section_positions(tick_df, events, rn, map_name))
        parts.append(section_view_angles(tick_df, rn))
        parts.append(section_velocity(tick_df, rn))
        parts.append(section_health_combat(tick_df, events, rn))
        parts.append(section_info_status(tick_df, rn))

        parts.append('</div>')

    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Visualize extracted demo data")
    ap.add_argument("pkl_path", help="Path to extracted .pkl file")
    ap.add_argument("--rounds", default=None, help="Comma-separated round numbers (default: first 3)")
    args = ap.parse_args()

    pkl_path = resolve_path_input(args.pkl_path)
    if not pkl_path.exists():
        print(f"Error: {pkl_path} not found")
        sys.exit(1)

    print(f"Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    rounds = None
    if args.rounds:
        rounds = [int(r) for r in args.rounds.split(",")]

    print("Building report...")
    html = build_report(data, rounds)

    out_path = pkl_path.with_name(pkl_path.stem.replace("_full", "") + "_report.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved → {out_path} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
