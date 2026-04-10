from __future__ import annotations

import glob
import os
import subprocess


_BZ_CANDIDATES = [
    r"C:\Program Files\Bandizip\bz.exe",
    r"C:\Program Files (x86)\Bandizip\bz.exe",
]


def _find_bz() -> str:
    bz = next((p for p in _BZ_CANDIDATES if os.path.isfile(p)), None)
    if bz is None:
        raise RuntimeError("Cannot find Bandizip bz.exe")
    return bz


def extract_archive(archive_path: str, dest_dir: str) -> list[str]:
    """
    Extract .dem files from any archive (zip/rar/7z) into dest_dir using Bandizip.
    Returns list of absolute paths to extracted .dem files.
    """
    archive_path = os.path.abspath(archive_path)
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    bz = _find_bz()
    result = subprocess.run(
        [bz, "x", f"-o:{dest_dir}", "-y", archive_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and result.returncode != 2:
        raise RuntimeError(
            f"Bandizip failed (rc={result.returncode}): "
            f"stderr={result.stderr.strip()!r} stdout={result.stdout.strip()!r}"
        )

    if result.returncode == 2:
        # CRC error on one or more files — partial extraction.
        # Keep files that are large enough to be valid demos (>= 1 MB).
        print(f"    [warn] CRC error during extraction, using partial results")

    return [
        os.path.abspath(p)
        for p in glob.glob(os.path.join(dest_dir, "**", "*.dem"), recursive=True)
        if os.path.getsize(p) >= 1024 * 1024
    ]


def get_dem_files(directory: str) -> list[str]:
    """Return list of .dem file paths in directory (non-recursive)."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".dem")
    ]
