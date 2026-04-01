"""Extract bomb-site labels from demoparser2 bomb_planted events."""

from __future__ import annotations
import pandas as pd

# demoparser2 returns site as int (0=A, 1=B) or sometimes string ('A'/'B')
_SITE_MAP: dict = {0: "A", 1: "B", "A": "A", "B": "B"}


def extract_bomb_site(bomb_planted_df: pd.DataFrame) -> pd.Series:
    """Map raw site values to 'A', 'B', or 'other'.

    Args:
        bomb_planted_df: DataFrame from ``DemoParser.parse_event('bomb_planted')``.
                         Must contain a 'site' column.

    Returns:
        pd.Series of str labels with the same index as the input.

    Raises:
        ValueError: If 'site' column is missing.
    """
    if "site" not in bomb_planted_df.columns:
        raise ValueError("bomb_planted_df must contain a 'site' column")
    return bomb_planted_df["site"].map(lambda v: _SITE_MAP.get(v, "other"))


def get_plant_ticks(bomb_planted_df: pd.DataFrame) -> pd.Series:
    """Return the tick series from a bomb_planted event DataFrame.

    Args:
        bomb_planted_df: DataFrame from ``DemoParser.parse_event('bomb_planted')``.
                         Must contain a 'tick' column.

    Returns:
        pd.Series of int tick values.

    Raises:
        ValueError: If 'tick' column is missing.
    """
    if "tick" not in bomb_planted_df.columns:
        raise ValueError("bomb_planted_df must contain a 'tick' column")
    return bomb_planted_df["tick"].astype(int)
