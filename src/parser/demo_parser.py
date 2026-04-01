"""Demo parser module — parses CS2 .dem files into structured DataFrames."""

import pandas as pd
from pathlib import Path


def parse_demo(demo_path: str | Path) -> dict[str, pd.DataFrame]:
    """Parse a CS2 demo file and return a dict of DataFrames keyed by event type."""
    raise NotImplementedError
