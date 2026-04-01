"""Tests for map zone classification and coordinate normalisation."""

import pytest
from src.utils.map_utils import classify_zone, normalize_coords


class TestClassifyZone:
    def test_a_site_mirage(self):
        # Centre of Mirage A site box
        assert classify_zone(1080.0, -400.0, "de_mirage") == "A"

    def test_b_site_mirage(self):
        assert classify_zone(-670.0, 60.0, "de_mirage") == "B"

    def test_mid_mirage(self):
        assert classify_zone(100.0, -200.0, "de_mirage") == "mid"

    def test_outside_all_zones_returns_other(self):
        assert classify_zone(9999.0, 9999.0, "de_mirage") == "other"

    def test_unknown_map_returns_other(self):
        assert classify_zone(0.0, 0.0, "de_unknown_map") == "other"

    def test_a_site_inferno(self):
        assert classify_zone(2000.0, 800.0, "de_inferno") == "A"

    def test_b_site_inferno(self):
        assert classify_zone(500.0, -900.0, "de_inferno") == "B"

    def test_a_site_dust2(self):
        assert classify_zone(1300.0, 2200.0, "de_dust2") == "A"

    def test_b_site_nuke(self):
        # Centre of nuke B — x in [-900,-280], y in [230,1130]
        assert classify_zone(-600.0, 680.0, "de_nuke") == "B"

    def test_mid_ancient(self):
        # Central ancient mid — x in [-180,930], y in [-690,370]
        assert classify_zone(375.0, -160.0, "de_ancient") == "mid"


class TestNormalizeCoords:
    def test_output_in_unit_cube_mirage(self):
        x_n, y_n, z_n = normalize_coords(1080.0, -400.0, 0.0, "de_mirage")
        assert 0.0 <= x_n <= 1.0
        assert 0.0 <= y_n <= 1.0
        assert 0.0 <= z_n <= 1.0

    def test_unknown_map_returns_midpoint(self):
        assert normalize_coords(0.0, 0.0, 0.0, "de_unknown") == (0.5, 0.5, 0.5)

    def test_z_clamped_below_floor(self):
        _, _, z_n = normalize_coords(0.0, 0.0, -9999.0, "de_mirage")
        assert z_n == 0.0

    def test_z_clamped_above_ceiling(self):
        _, _, z_n = normalize_coords(0.0, 0.0, 9999.0, "de_mirage")
        assert z_n == 1.0

    def test_returns_three_floats(self):
        result = normalize_coords(500.0, 200.0, 100.0, "de_inferno")
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)
