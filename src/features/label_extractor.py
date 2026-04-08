"""Extract bomb-site labels from demoparser2 bomb_planted events."""

from __future__ import annotations
import pandas as pd

from src.utils.map_utils import classify_zone

# Fallback: demoparser2 may return 0/1 on some older demos
_SITE_INT_MAP: dict = {0: "A", 1: "B", "A": "A", "B": "B"}


def extract_bomb_site(
    bomb_planted_df: pd.DataFrame,
    map_name: str = "",
) -> pd.Series:
    """Map raw bomb_planted rows to 'A', 'B', or 'other'.

    Strategy (in order of preference):
    1. If planter X/Y/Z columns are present (user_X/user_Y/user_Z), classify
       by world position using map zone boxes — this handles CS2's numeric
       site entity handles correctly.
    2. If 'site' is already a string 'A'/'B', use it directly.
    3. If 'site' is 0/1 (older demos), map via lookup table.

    Args:
        bomb_planted_df: DataFrame from DemoParser.parse_event('bomb_planted').
                         Should be fetched with player=['X','Y','Z'] to enable
                         position-based classification.
        map_name: Map name (e.g. 'de_mirage').  Required for position strategy.

    Returns:
        pd.Series of str labels ('A', 'B', or 'other') aligned to input index.
    """
    if "site" not in bomb_planted_df.columns:
        raise ValueError("bomb_planted_df must contain a 'site' column")

    # Strategy 1: position-based (most reliable for CS2 demos)
    if map_name and {"user_X", "user_Y", "user_Z"}.issubset(bomb_planted_df.columns):
        def _pos_classify(row: pd.Series) -> str:
            zone = classify_zone(
                row["user_X"], row["user_Y"], map_name, z=row.get("user_Z")
            )
            return zone if zone in ("A", "B") else "other"

        return bomb_planted_df.apply(_pos_classify, axis=1)

    # Strategy 2/3: fall back to site value mapping
    return bomb_planted_df["site"].map(
        lambda v: _SITE_INT_MAP.get(v, "other")
    )


def get_plant_ticks(bomb_planted_df: pd.DataFrame) -> pd.Series:
    """Return the tick series from a bomb_planted event DataFrame.

    Args:
        bomb_planted_df: DataFrame from DemoParser.parse_event('bomb_planted').
                         Must contain a 'tick' column.

    Returns:
        pd.Series of int tick values.

    Raises:
        ValueError: If 'tick' column is missing.
    """
    if "tick" not in bomb_planted_df.columns:
        raise ValueError("bomb_planted_df must contain a 'tick' column")
    return bomb_planted_df["tick"].astype(int)
