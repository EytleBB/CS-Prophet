"""Feature engineering module — builds model-ready feature tensors from parsed demos."""

import pandas as pd
import numpy as np


def build_features(events: dict[str, pd.DataFrame]) -> np.ndarray:
    """Transform parsed demo events into a feature matrix."""
    raise NotImplementedError
