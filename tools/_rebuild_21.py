"""One-off: force re-extract + rebuild parquet for 21 demos (3 per map)."""
from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_path

MAPS = ("de_ancient", "de_anubis", "de_dust2", "de_inferno", "de_mirage", "de_nuke", "de_overpass")
PER_MAP = 3


def pick_demos() -> list[Path]:
    raw_dir = data_path("raw", "demos")
    by_map: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for dem in raw_dir.glob("*.dem"):
        m = re.match(r"(\d+)_(de_\w+)", dem.stem)
        if not m:
            continue
        match_id, map_name = int(m.group(1)), m.group(2)
        if map_name in MAPS:
            by_map[map_name].append((match_id, dem))

    picked: list[Path] = []
    for map_name in MAPS:
        for _, p in sorted(by_map[map_name], reverse=True)[:PER_MAP]:
            picked.append(p)
    return picked


def process_one(dem: Path, python: str) -> str:
    stem = dem.stem
    viz_dir = data_path("viz")
    pv2_dir = data_path("processed_v2")
    pkl_path = viz_dir / f"{stem}_full.pkl"
    pq_path = pv2_dir / f"{stem}.parquet"

    if pkl_path.exists():
        pkl_path.unlink()
    if pq_path.exists():
        pq_path.unlink()

    ret = subprocess.run(
        [python, "tools/demo_full_extract.py", str(dem)],
        cwd=str(REPO_ROOT), capture_output=True,
    )
    if ret.returncode != 0:
        return f"[FAIL extract] {stem}: {ret.stderr[-200:].decode('utf-8', 'replace') if ret.stderr else ''}"

    ret = subprocess.run(
        [python, "tools/build_processed_v2.py", str(pkl_path), "--out-dir", "processed_v2"],
        cwd=str(REPO_ROOT), capture_output=True,
    )
    if ret.returncode != 0:
        return f"[FAIL convert] {stem}: {ret.stderr[-200:].decode('utf-8', 'replace') if ret.stderr else ''}"
    return f"[ok] {stem}"


def main() -> int:
    demos = pick_demos()
    print(f"Rebuilding {len(demos)} demos with 8 workers\n", flush=True)

    python = sys.executable
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_one, d, python): d for d in demos}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            print(f"[{i}/{len(demos)}] {result}", flush=True)
            if result.startswith("[ok]"):
                ok += 1
            else:
                fail += 1
    print(f"\nDone. ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
