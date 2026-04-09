"""Extract bomb-site labels from demoparser2 bomb_planted events."""
from __future__ import annotations
import pandas as pd

from .map_utils import classify_zone

_SITE_INT_MAP: dict = {0: "A", 1: "B", "A": "A", "B": "B"}


def extract_bomb_site(bomb_planted_df: pd.DataFrame, map_name: str = "") -> pd.Series:
    if "site" not in bomb_planted_df.columns:
        raise ValueError("bomb_planted_df must contain a 'site' column")

    if map_name and {"user_X", "user_Y", "user_Z"}.issubset(bomb_planted_df.columns):
        def _pos_classify(row: pd.Series) -> str:
            zone = classify_zone(
                row["user_X"], row["user_Y"], map_name, z=row.get("user_Z")
            )
            return zone if zone in ("A", "B") else "other"
        return bomb_planted_df.apply(_pos_classify, axis=1)

    return bomb_planted_df["site"].map(lambda v: _SITE_INT_MAP.get(v, "other"))


def get_plant_ticks(bomb_planted_df: pd.DataFrame) -> pd.Series:
    if "tick" not in bomb_planted_df.columns:
        raise ValueError("bomb_planted_df must contain a 'tick' column")
    return bomb_planted_df["tick"].astype(int)
