r"""Centralized path helpers for repo code and external data storage.

The project now prefers an external data root on Windows:
    H:\CS_Prophet\data

Selection priority:
1. ``CS_PROPHET_DATA_ROOT`` environment variable
2. Default external Windows data root if the drive/folder exists
3. Repo-local ``data/`` as a portable fallback
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_EXTERNAL_DATA_ROOT = Path(r"H:\CS_Prophet\data")
DATA_ROOT_PREFIXES = {"raw", "processed", "processed_v2", "splits", "viz", "tmp"}
# Any first-path-component starting with one of these, followed by "_", also
# routes to the data root (e.g. "processed_v2_2hz_preplant", "viz_2hz").
DATA_ROOT_PREFIX_BASES = ("raw", "processed", "splits", "viz", "tmp")


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def repo_data_root() -> Path:
    """Return the repo-local data directory."""
    return repo_root() / "data"


def _windows_external_root_available(path: Path) -> bool:
    if os.name != "nt":
        return False
    if path.exists() or path.parent.exists():
        return True
    drive = path.drive
    if not drive:
        return False
    return Path(f"{drive}\\").exists()


def data_root() -> Path:
    """Return the active data root."""
    env = os.environ.get("CS_PROPHET_DATA_ROOT")
    if env:
        return Path(env).expanduser()

    if _windows_external_root_available(DEFAULT_EXTERNAL_DATA_ROOT):
        return DEFAULT_EXTERNAL_DATA_ROOT

    return repo_data_root()


def data_path(*parts: str, prefer_existing: bool = False) -> Path:
    """Join *parts* under the active data root.

    When ``prefer_existing`` is true, repo-local data is used as a fallback
    if the external target does not exist yet. This is useful during
    migration or for lightweight assets that may still live in the repo.
    """

    preferred = data_root().joinpath(*parts)
    if prefer_existing and not preferred.exists():
        fallback = repo_data_root().joinpath(*parts)
        if fallback.exists():
            return fallback
    return preferred


def resolve_path_input(value: str | os.PathLike[str]) -> Path:
    """Resolve user/config path input to an absolute path.

    Rules:
    - absolute paths are returned unchanged
    - ``data/...`` paths are redirected to the active data root
    - ``raw/...``, ``processed/...`` etc. are treated as data-root relative
    - everything else is treated as repo-root relative
    """

    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    parts = path.parts
    if not parts:
        return data_root()

    if parts[0] == "data":
        active = data_root().joinpath(*parts[1:])
        fallback = repo_root().joinpath(*parts)
        if active.exists() or not fallback.exists():
            return active
        return fallback

    if parts[0] in DATA_ROOT_PREFIXES:
        return data_root().joinpath(*parts)

    if any(parts[0].startswith(f"{base}_") for base in DATA_ROOT_PREFIX_BASES):
        return data_root().joinpath(*parts)

    return repo_root().joinpath(*parts)


def ensure_data_layout(root: Path | None = None) -> dict[str, Path]:
    """Create and return the standard data directory layout."""

    base = root if root is not None else data_root()
    layout = {
        "root": base,
        "raw": base / "raw",
        "raw_demos": base / "raw" / "demos",
        "raw_manifest_dir": base / "raw" / "manifest",
        "processed": base / "processed",
        "processed_v2": base / "processed_v2",
        "splits": base / "splits",
        "viz": base / "viz",
        "viz_assets": base / "viz" / "assets",
        "tmp": base / "tmp",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout
