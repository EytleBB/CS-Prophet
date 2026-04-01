"""Tests for label_extractor — bomb site label extraction."""

import pandas as pd
import pytest
from src.features.label_extractor import extract_bomb_site, get_plant_ticks


class TestExtractBombSite:
    def test_integer_0_maps_to_A(self):
        df = pd.DataFrame({"site": [0], "tick": [1000]})
        assert extract_bomb_site(df).iloc[0] == "A"

    def test_integer_1_maps_to_B(self):
        df = pd.DataFrame({"site": [1], "tick": [2000]})
        assert extract_bomb_site(df).iloc[0] == "B"

    def test_string_A_passthrough(self):
        df = pd.DataFrame({"site": ["A"], "tick": [1000]})
        assert extract_bomb_site(df).iloc[0] == "A"

    def test_string_B_passthrough(self):
        df = pd.DataFrame({"site": ["B"], "tick": [1000]})
        assert extract_bomb_site(df).iloc[0] == "B"

    def test_unknown_integer_returns_other(self):
        df = pd.DataFrame({"site": [99], "tick": [1000]})
        assert extract_bomb_site(df).iloc[0] == "other"

    def test_none_value_returns_other(self):
        df = pd.DataFrame({"site": [None], "tick": [1000]})
        assert extract_bomb_site(df).iloc[0] == "other"

    def test_missing_site_column_raises_value_error(self):
        df = pd.DataFrame({"tick": [1000]})
        with pytest.raises(ValueError, match="site"):
            extract_bomb_site(df)

    def test_multiple_rows_all_mapped(self):
        df = pd.DataFrame({"site": [0, 1, "A", 99], "tick": [100, 200, 300, 400]})
        result = extract_bomb_site(df).tolist()
        assert result == ["A", "B", "A", "other"]

    def test_returns_series(self):
        df = pd.DataFrame({"site": [0], "tick": [1000]})
        result = extract_bomb_site(df)
        assert isinstance(result, pd.Series)


class TestGetPlantTicks:
    def test_returns_int_value(self):
        df = pd.DataFrame({"site": [0], "tick": [1234]})
        assert get_plant_ticks(df).iloc[0] == 1234

    def test_missing_tick_column_raises_value_error(self):
        df = pd.DataFrame({"site": [0]})
        with pytest.raises(ValueError, match="tick"):
            get_plant_ticks(df)

    def test_coerces_float_tick_to_int(self):
        df = pd.DataFrame({"site": [0], "tick": [1234.0]})
        result = get_plant_ticks(df)
        assert result.dtype.kind == "i"  # integer kind

    def test_multiple_ticks(self):
        df = pd.DataFrame({"site": [0, 1], "tick": [100, 200]})
        result = get_plant_ticks(df).tolist()
        assert result == [100, 200]
