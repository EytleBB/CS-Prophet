"""Portable archive extraction with fallback chain: 7-Zip -> Bandizip -> Python zipfile."""
from __future__ import annotations

import glob
import os
import subprocess
import zipfile


# ── Extractor discovery ───────────────────────────────────────────────────

_7Z_CANDIDATES = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    # Portable 7-Zip next to this script (user can drop 7z.exe on USB)
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "7z.exe"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "7z.exe"),
]

_BZ_CANDIDATES = [
    r"D:\EDGE\APP\Bandizip\bz.exe",
    r"C:\Program Files\Bandizip\bz.exe",
    r"C:\Program Files (x86)\Bandizip\bz.exe",
]


def _find_7z() -> str | None:
    return next((p for p in _7Z_CANDIDATES if os.path.isfile(p)), None)


def _find_bz() -> str | None:
    return next((p for p in _BZ_CANDIDATES if os.path.isfile(p)), None)


# ── Extraction backends ──────────────────────────────────────────────────

def _extract_7z(archive_path: str, dest_dir: str) -> None:
    exe = _find_7z()
    if exe is None:
        raise RuntimeError("7-Zip not found")
    result = subprocess.run(
        [exe, "x", f"-o{dest_dir}", "-y", archive_path],
        capture_output=True, text=True,
    )
    # 7z returns 0=ok, 1=warning, 2=fatal
    if result.returncode == 2:
        raise RuntimeError(
            f"7-Zip failed (rc=2): stderr={result.stderr.strip()!r}"
        )
    if result.returncode == 1:
        print(f"    [warn] 7-Zip warning during extraction, using partial results")


def _extract_bz(archive_path: str, dest_dir: str) -> None:
    exe = _find_bz()
    if exe is None:
        raise RuntimeError("Bandizip not found")
    result = subprocess.run(
        [exe, "x", f"-o:{dest_dir}", "-y", archive_path],
        capture_output=True, text=True,
    )
    if result.returncode not in (0, 2):
        raise RuntimeError(
            f"Bandizip failed (rc={result.returncode}): "
            f"stderr={result.stderr.strip()!r}"
        )
    if result.returncode == 2:
        print(f"    [warn] Bandizip CRC error, using partial results")


def _extract_python_zip(archive_path: str, dest_dir: str) -> None:
    if not zipfile.is_zipfile(archive_path):
        raise RuntimeError("Not a zip file — .rar requires 7-Zip or Bandizip")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(dest_dir)


# ── Public API ───────────────────────────────────────────────────────────

def extract_archive(archive_path: str, dest_dir: str) -> list[str]:
    """
    Extract .dem files from archive using best available extractor.
    Fallback chain: 7-Zip -> Bandizip -> Python zipfile (zip only).
    Returns list of absolute paths to extracted .dem files (>= 1 MB).
    """
    archive_path = os.path.abspath(archive_path)
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    is_zip = archive_path.lower().endswith(".zip")
    errors = []

    # Try 7-Zip first (handles zip/rar/7z)
    if _find_7z():
        try:
            _extract_7z(archive_path, dest_dir)
            return _collect_dems(dest_dir)
        except Exception as e:
            errors.append(f"7z: {e}")

    # Try Bandizip (handles zip/rar/7z)
    if _find_bz():
        try:
            _extract_bz(archive_path, dest_dir)
            return _collect_dems(dest_dir)
        except Exception as e:
            errors.append(f"bz: {e}")

    # Python zipfile fallback (zip only)
    if is_zip:
        try:
            _extract_python_zip(archive_path, dest_dir)
            return _collect_dems(dest_dir)
        except Exception as e:
            errors.append(f"zipfile: {e}")

    raise RuntimeError(
        f"No extractor could handle {os.path.basename(archive_path)}. "
        f"Install 7-Zip or place 7z.exe next to this tool.\n"
        f"Errors: {'; '.join(errors)}"
    )


def _collect_dems(dest_dir: str) -> list[str]:
    return [
        os.path.abspath(p)
        for p in glob.glob(os.path.join(dest_dir, "**", "*.dem"), recursive=True)
        if os.path.getsize(p) >= 1024 * 1024
    ]
