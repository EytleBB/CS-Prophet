"""Build fixed-size float32 feature vectors from parquet rows."""

from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_DIM: int = 74
# Layout:
#   [0:35]  T players 0–4: (x, y, z, hp/100, armor/100, helmet, alive) × 5
#   [35:70] CT players 0–4: same layout × 5
#   [70:74] map_zone one-hot: A=70, B=71, mid=72, other=73

ZONE_IDX: dict[str, int] = {"A": 0, "B": 1, "mid": 2, "other": 3}
PLAYER_FIELDS: tuple[str, ...] = ("x", "y", "z", "hp", "armor", "helmet", "alive")
NORMALISE: frozenset[str] = frozenset({"hp", "armor"})


def build_state_vector(row: pd.Series) -> np.ndarray:
    """Convert one parquet row to a float32 feature vector of shape (FEATURE_DIM,).

    hp and armor are normalised by dividing by 100.
    Coordinates and boolean flags are used as-is (already in [0, 1]).
    Missing columns default to 0.0.

    Args:
        row: One row from a parsed demo parquet (pd.Series with column-name index).

    Returns:
        np.ndarray of shape (74,) and dtype float32.
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    for side, base in (("t", 0), ("ct", 35)):
        for i in range(5):
            for j, field in enumerate(PLAYER_FIELDS):
                col = f"{side}{i}_{field}"
                val = float(row[col]) if col in row.index else 0.0
                if field in NORMALISE:
                    val /= 100.0
                vec[base + i * 7 + j] = val

    zone_idx = ZONE_IDX.get(str(row["map_zone"]) if "map_zone" in row.index else "other", 3)
    vec[70 + zone_idx] = 1.0

    return vec
