"""Real-time GSI receiver + bomb-site predictor + web dashboard.

Usage:
    python -m src.inference.realtime_engine
    python -m src.inference.realtime_engine --checkpoint checkpoints/best.pt --port 3000

Then open http://localhost:3000 in a browser, start CS2, and load a demo or GOTV.
"""

from __future__ import annotations

import argparse
import collections
import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from src.features.state_vector import FEATURE_DIM, build_state_vector
from src.inference.gsi_state_builder import build_row_from_gsi
from src.inference.predictor import RoundPredictor

logger = logging.getLogger(__name__)

_MAX_STEPS   = 720
_DASHBOARD   = Path(__file__).parent.parent.parent / "dashboard"
_LIVE_PHASES = {"live"}


class _GameState:
    """Thread-safe container for the current round state and latest prediction."""

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._window: collections.deque = collections.deque(maxlen=_MAX_STEPS)
        self._round  = -1
        self._step   = 0
        self.map_name  = ""
        self.round_num = 0
        self.ct_score  = 0
        self.t_score   = 0
        self.prediction: dict[str, float] = {"A": 0.5, "B": 0.5}

    def update(self, gsi: dict, predictor: RoundPredictor) -> None:
        map_info  = gsi.get("map", {})
        map_name  = map_info.get("name", "")
        round_num = int(map_info.get("round", 0))
        phase     = map_info.get("phase", "")

        if phase not in _LIVE_PHASES or not map_name:
            return
        if not gsi.get("allplayers"):
            return

        with self._lock:
            if round_num != self._round:
                self._window.clear()
                self._step  = 0
                self._round = round_num

            self.map_name  = map_name
            self.round_num = round_num
            self.ct_score  = int(map_info.get("team_ct", {}).get("score", 0))
            self.t_score   = int(map_info.get("team_t",  {}).get("score", 0))

            row = build_row_from_gsi(gsi, self._step, round_num, map_name)
            if row is None:
                return

            vec = build_state_vector(pd.Series(row))
            self._window.append(vec)
            self._step += 1

            mat = np.zeros((_MAX_STEPS, FEATURE_DIM), dtype=np.float32)
            for i, v in enumerate(self._window):
                mat[i] = v

            self.prediction = predictor.predict(mat)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "A":        self.prediction.get("A", 0.5),
                "B":        self.prediction.get("B", 0.5),
                "map":      self.map_name,
                "round":    self.round_num,
                "ct_score": self.ct_score,
                "t_score":  self.t_score,
                "steps":    len(self._window),
            }


def create_app(checkpoint: Path, device: str = "cpu") -> Flask:
    predictor = RoundPredictor(checkpoint, device=device)
    game      = _GameState()
    app       = Flask(__name__, static_folder=str(_DASHBOARD))

    @app.post("/gsi")
    def recv_gsi():
        data = request.get_json(silent=True, force=True)
        if data:
            game.update(data, predictor)
        return "", 204

    @app.get("/state")
    def state():
        return jsonify(game.snapshot())

    @app.get("/")
    def dashboard():
        return send_from_directory(str(_DASHBOARD), "index.html")

    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    ap = argparse.ArgumentParser(description="CS Prophet real-time engine")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--port",       type=int, default=3000)
    ap.add_argument("--device",     default="cpu")
    args = ap.parse_args()

    logger.info("Loading checkpoint: %s", args.checkpoint)
    app = create_app(Path(args.checkpoint), device=args.device)
    logger.info("CS Prophet  →  http://localhost:%d", args.port)
    logger.info("Open this URL in your browser, then start CS2 and load a demo.")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
