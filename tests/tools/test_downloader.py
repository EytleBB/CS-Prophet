import os
import zipfile
import pytest
from tools.hltv.downloader import extract_archive, get_dem_files


@pytest.fixture
def zip_with_two_dems(tmp_path):
    archive = tmp_path / "demo.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("navi-vs-faze-m1-de_mirage.dem", b"FAKE_DEM_CONTENT" * 10)
        zf.writestr("navi-vs-faze-m2-de_inferno.dem", b"FAKE_DEM_CONTENT" * 10)
    return str(archive)


@pytest.fixture
def zip_with_non_dem(tmp_path):
    archive = tmp_path / "demo.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("match-de_dust2.dem", b"FAKE")
        zf.writestr("README.txt", b"hello")
    return str(archive)


def test_extract_archive_returns_dem_paths(zip_with_two_dems, tmp_path):
    dest = str(tmp_path / "out")
    os.makedirs(dest)
    paths = extract_archive(zip_with_two_dems, dest)
    assert len(paths) == 2
    assert all(p.endswith(".dem") for p in paths)


def test_extract_archive_filters_non_dem(zip_with_non_dem, tmp_path):
    dest = str(tmp_path / "out")
    os.makedirs(dest)
    paths = extract_archive(zip_with_non_dem, dest)
    assert len(paths) == 1
    assert paths[0].endswith(".dem")


def test_get_dem_files_finds_all(tmp_path):
    (tmp_path / "a.dem").write_bytes(b"x")
    (tmp_path / "b.dem").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    result = get_dem_files(str(tmp_path))
    assert len(result) == 2
    assert all(f.endswith(".dem") for f in result)
