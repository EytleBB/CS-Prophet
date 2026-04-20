"""Build and export processed_v2 parquet files for the v2 218-dim schema."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.features.feature_builder_v2 import build_feature_row_v2, build_round_label_map
from src.features.state_vector_v2 import FEATURE_NAMES

METADATA_COLUMNS: tuple[str, ...] = (
    "demo_name",
    "map_name",
    "bomb_site",
    "step",
    "tick",
)
PROCESSED_V2_COLUMNS: tuple[str, ...] = METADATA_COLUMNS + FEATURE_NAMES


def infer_demo_name(full_pkl_path: Path) -> str:
    """Infer demo stem from a ``*_full.pkl`` path."""
    stem = full_pkl_path.stem
    if stem.endswith("_full"):
        return stem[:-5]
    return stem


def load_full_payload(path: Path) -> dict:
    """Load an extracted ``*_full.pkl`` payload."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_processed_frame_v2(data: dict, demo_name: str) -> pd.DataFrame:
    """Convert one extracted full payload into processed_v2 parquet rows."""
    tick_df = data["tick_df"]
    events = data["events"]
    round_info = data["round_info"]
    header = data.get("header", {}) or {}
    map_name = header.get("map_name", "unknown")
    label_map = build_round_label_map(events, map_name)

    rows: list[dict[str, object]] = []
    for round_num in sorted(label_map):
        bomb_site = label_map[round_num]
        if bomb_site not in {"A", "B"}:
            continue

        round_ticks = (
            tick_df[tick_df["round_num"] == round_num][["tick"]]
            .drop_duplicates()
            .sort_values("tick")
        )
        if round_ticks.empty:
            continue

        for step, tick_value in enumerate(round_ticks["tick"].tolist()):
            tick = int(tick_value)
            tick_slice = tick_df[
                (tick_df["round_num"] == round_num) & (tick_df["tick"] == tick)
            ]
            feature_row = build_feature_row_v2(
                tick_df=tick_df,
                tick_slice=tick_slice,
                events=events,
                tick=tick,
                round_num=round_num,
                map_name=map_name,
                round_info=round_info,
            )
            feature_row["round_num"] = int(round_num)

            row: dict[str, object] = {
                "demo_name": demo_name,
                "map_name": map_name,
                "bomb_site": bomb_site,
                "step": int(step),
                "tick": int(tick),
            }
            row.update(feature_row)
            rows.append(row)

    return pd.DataFrame(rows, columns=PROCESSED_V2_COLUMNS)


def export_full_pkl_to_processed_v2(full_pkl_path: Path, output_dir: Path) -> Path | None:
    """Export one ``*_full.pkl`` file to a processed_v2 parquet."""
    payload = load_full_payload(full_pkl_path)
    demo_name = infer_demo_name(full_pkl_path)
    frame = build_processed_frame_v2(payload, demo_name=demo_name)
    if frame.empty:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{demo_name}.parquet"
    frame.to_parquet(out_path, index=False)
    return out_path
