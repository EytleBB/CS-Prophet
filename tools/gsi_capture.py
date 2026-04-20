#!/usr/bin/env python3
"""Capture CS2 GSI payloads to a JSONL file for field analysis.

Usage:
    python tools/gsi_capture.py
    python tools/gsi_capture.py --port 3001 --out data/gsi_dump.jsonl

Then launch CS2, load a demo (e.g. GOTV), and play through a few rounds.
Press Ctrl+C to stop. The captured payloads are saved to the output file.
"""

import argparse
import json
import time
from pathlib import Path

from flask import Flask, request

app = Flask(__name__)

_payloads: list[dict] = []
_out_path: Path = Path("data/gsi_dump.jsonl")
_count = 0


@app.post("/gsi")
@app.post("/")
def recv():
    global _count
    data = request.get_json(silent=True, force=True)
    if not data:
        return "", 204

    _count += 1
    ts = time.time()
    data["_capture_ts"] = ts
    data["_capture_seq"] = _count
    _payloads.append(data)

    # Print summary
    map_info = data.get("map", {})
    round_info = data.get("round", {})
    bomb = data.get("bomb", {})
    # CS2 payloads observed in this project expose active grenades under
    # "grenades" (not "allgrenades"), but keep a fallback for older configs.
    grenades = data.get("grenades", data.get("allgrenades", {}))
    allplayers = data.get("allplayers", {})
    player = data.get("player", {})

    print(
        f"[{_count:4d}] "
        f"map={map_info.get('name', '?'):12s} "
        f"round={map_info.get('round', '?'):>2} "
        f"phase={round_info.get('phase', '?'):10s} "
        f"bomb={round_info.get('bomb', bomb.get('state', '?')):10s} "
        f"players={len(allplayers):>2} "
        f"grenades={len(grenades):>2} "
        f"keys={sorted(data.keys())}"
    )

    # Flush every 10 payloads
    if _count % 10 == 0:
        _flush()

    return "", 204


def _flush():
    if not _payloads:
        return
    _out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_out_path, "a", encoding="utf-8") as f:
        for p in _payloads:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    _payloads.clear()


def main():
    global _out_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--out", default="data/gsi_dump.jsonl")
    args = parser.parse_args()
    _out_path = Path(args.out)

    # Clear old dump
    if _out_path.exists():
        _out_path.unlink()

    print(f"GSI capture server on http://127.0.0.1:{args.port}")
    print(f"Output: {_out_path}")
    print(f"Start CS2, load a demo, play a few rounds. Ctrl+C to stop.\n")

    try:
        app.run(host="127.0.0.1", port=args.port, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        _flush()
        print(f"\nSaved {_count} payloads to {_out_path}")


if __name__ == "__main__":
    main()
