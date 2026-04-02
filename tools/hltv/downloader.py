from __future__ import annotations

import os
import zipfile


def extract_archive(archive_path: str, dest_dir: str) -> list[str]:
    """
    Extract .dem files from a .zip or .rar archive into dest_dir.
    Returns list of absolute paths to extracted .dem files.
    """
    archive_path = os.path.abspath(archive_path)
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    if archive_path.endswith(".zip"):
        return _extract_zip(archive_path, dest_dir)
    elif archive_path.endswith(".rar"):
        return _extract_rar(archive_path, dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def _extract_zip(archive_path: str, dest_dir: str) -> list[str]:
    extracted = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".dem"):
                zf.extract(name, dest_dir)
                extracted.append(os.path.join(dest_dir, name))
    return extracted


def _extract_rar(archive_path: str, dest_dir: str) -> list[str]:
    try:
        import rarfile
    except ImportError:
        raise RuntimeError("rarfile not installed. Run: pip install rarfile")
    extracted = []
    with rarfile.RarFile(archive_path) as rf:
        for name in rf.namelist():
            if name.endswith(".dem"):
                rf.extract(name, dest_dir)
                extracted.append(os.path.join(dest_dir, name))
    return extracted


def get_dem_files(directory: str) -> list[str]:
    """Return list of .dem file paths in directory (non-recursive)."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".dem")
    ]
