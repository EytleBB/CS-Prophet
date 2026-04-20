"""Real-time GSI receiver + bomb-site predictor + web dashboard."""

from __future__ import annotations

import argparse
import collections
import logging
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from src.features.state_vector_v2 import FEATURE_DIM, FEATURE_NAMES, build_state_vector
from src.inference.gsi_state_builder import build_row_from_gsi
from src.inference.predictor import RoundPredictor
from src.utils.map_utils import infer_map_from_positions, map_fit_fraction

logger = logging.getLogger(__name__)

_MAX_STEPS = 180
_STEP_INTERVAL_SEC = 0.5  # 2 Hz bucketing, matches training target_tick_rate
_MAP_SWITCH_CONFIRM_FRAMES = 3
_CACHED_MAP_KEEP_FRACTION = 0.6
_NEW_MAP_SWITCH_FRACTION = 0.8
_MIN_POSITIONS_FOR_COORD_SWITCH = 4
_DASHBOARD = Path(__file__).parent.parent.parent / "dashboard"


def _merge_payload(current: dict, incoming: dict) -> dict:
    """Merge incremental GSI payloads into the latest full state."""
    merged = dict(current)
    for key, value in incoming.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
            and key not in {"bomb", "grenades", "phase_countdowns"}
        ):
            merged[key] = _merge_payload(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_memory_map_name(
    *,
    map_override: str,
    advertised_map: str,
    player_positions: list[tuple[float, float, float]],
    cached_map: str,
    pending_map: str,
    pending_count: int,
) -> tuple[str, str, int]:
    """Resolve a stable live map name without letting coord inference flicker.

    Order:
    1. CLI override
    2. map name read directly from memory
    3. keep the cached map if current positions still fit it
    4. coord-based reverse lookup, but require repeated agreement before
       switching away from an existing cached map
    5. final fallback to cached map
    """
    if map_override:
        return map_override, "", 0

    if advertised_map:
        return advertised_map, "", 0

    nonzero_positions = [pos for pos in player_positions if pos != (0.0, 0.0, 0.0)]
    if cached_map and map_fit_fraction(player_positions, cached_map) >= _CACHED_MAP_KEEP_FRACTION:
        return cached_map, "", 0

    if len(nonzero_positions) < _MIN_POSITIONS_FOR_COORD_SWITCH:
        return cached_map, "", 0

    inferred_map = infer_map_from_positions(player_positions)
    if not inferred_map:
        return cached_map, "", 0

    if map_fit_fraction(player_positions, inferred_map) < _NEW_MAP_SWITCH_FRACTION:
        return cached_map, "", 0

    if not cached_map:
        return inferred_map, "", 0

    if inferred_map == cached_map:
        return cached_map, "", 0

    if inferred_map == pending_map:
        next_count = pending_count + 1
    else:
        pending_map = inferred_map
        next_count = 1

    if next_count >= _MAP_SWITCH_CONFIRM_FRAMES:
        logger.info(
            "Memory map switch confirmed: %s -> %s after %d consecutive frames",
            cached_map,
            inferred_map,
            next_count,
        )
        return inferred_map, "", 0

    logger.debug(
        "Memory map switch pending: cached=%s inferred=%s streak=%d/%d",
        cached_map,
        inferred_map,
        next_count,
        _MAP_SWITCH_CONFIRM_FRAMES,
    )
    return cached_map, pending_map, next_count


class _GameState:
    """Thread-safe container for the current round state and latest prediction."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._window: collections.deque = collections.deque(maxlen=_MAX_STEPS)
        self._round = -1
        self._step = 0
        self._prev_round_phase = ""
        self._frozen = False
        # Memory mode only: block ingestion during the post-round gap. Memory
        # sees round_num tick up as soon as a round is decided, but round_phase
        # can stay "live" until the freezetime flag flips. We only trust that
        # the action phase has truly begun after seeing freezetime -> live.
        self._awaiting_freeze_to_live = False
        self._merged_gsi: dict = {}
        self._last_map_state: dict = {}
        self._score_baseline: dict[str, int] = {}
        self._round_start_time: float | None = None
        self._last_vec: np.ndarray | None = None
        self._last_row: dict | None = None
        self.map_name = ""
        self.round_num = 0
        self.ct_score = 0
        self.t_score = 0
        self.prediction: dict[str, float] = {"A": 0.5, "B": 0.5}

    def update(self, gsi: dict, predictor: RoundPredictor) -> None:
        self._merged_gsi = _merge_payload(self._merged_gsi, gsi)
        self._advance(predictor)

    def tick(self, predictor: RoundPredictor) -> None:
        """Advance the 2 Hz window even when no new GSI event has arrived."""
        if not self._merged_gsi:
            return
        self._advance(predictor)

    def note_map_state(self, map_state: dict) -> None:
        with self._lock:
            self._last_map_state = dict(map_state)

    @staticmethod
    def _infer_map_name_from_row(row: dict[str, float]) -> str:
        for feature_name in FEATURE_NAMES:
            if feature_name.startswith("map_") and float(row.get(feature_name, 0.0)) > 0.5:
                return feature_name[4:]
        return ""

    def _reset_round_locked(self, round_num: int, await_freeze: bool = False) -> None:
        self._window.clear()
        self._step = 0
        self._round = round_num
        self._frozen = False
        self._score_baseline = {}
        # Do not start the wall-clock timer until we ingest the first true
        # live frame of the round. In memory mode, round_num can advance during
        # the post-round / freeze-time gap; starting the timer here would
        # wrongly accumulate steps before the round actually begins.
        self._round_start_time = None
        self._awaiting_freeze_to_live = await_freeze
        self.prediction = {"A": 0.5, "B": 0.5}

    def _ingest_row_locked(
        self,
        row: dict[str, float],
        expected_step: int,
        predictor: RoundPredictor,
    ) -> None:
        vec = build_state_vector(row)
        self._last_vec = vec
        self._last_row = row

        # Fill any missed buckets by repeating the latest vector. This keeps the
        # position-axis cadence aligned with training when updates stay silent.
        while self._step <= expected_step and self._step < _MAX_STEPS:
            self._window.append(vec)
            self._step += 1

        mat = np.zeros((_MAX_STEPS, FEATURE_DIM), dtype=np.float32)
        for idx, value in enumerate(self._window):
            mat[idx] = value

        self.prediction = predictor.predict(mat)

    def _advance(self, predictor: RoundPredictor) -> None:
        merged = self._merged_gsi

        map_info = merged.get("map", {})
        round_info = merged.get("round", {})
        bomb_info = merged.get("bomb", {})
        map_name = str(map_info.get("name", ""))
        round_num = int(map_info.get("round", 0))
        map_phase = str(map_info.get("phase", ""))
        round_phase = str(round_info.get("phase", ""))
        bomb_state = ""
        if isinstance(bomb_info, dict):
            bomb_state = str(bomb_info.get("state", round_info.get("bomb", "")))
        else:
            bomb_state = str(round_info.get("bomb", ""))

        logger.info(
            "GSI: map=%s round=%s map_phase=%s round_phase=%s bomb=%s",
            map_name,
            round_num,
            map_phase,
            round_phase,
            bomb_state,
        )

        if map_phase != "live" or not map_name:
            return
        if not merged.get("allplayers"):
            return

        with self._lock:
            new_round = False
            if round_phase == "live" and self._prev_round_phase == "freezetime":
                new_round = True
            elif round_num != self._round:
                new_round = True
            self._prev_round_phase = round_phase

            if new_round:
                self._reset_round_locked(round_num)
                # Clear stale bomb state left over from previous round because
                # GSI does not always re-send bomb state after round reset.
                self._merged_gsi.pop("bomb", None)
                if isinstance(self._merged_gsi.get("round"), dict):
                    self._merged_gsi["round"].pop("bomb", None)
                bomb_state = ""
                allplayers = merged.get("allplayers", {})
                if isinstance(allplayers, dict):
                    for pdata in allplayers.values():
                        if not isinstance(pdata, dict):
                            continue
                        player_name = str(pdata.get("name", ""))
                        stats = pdata.get("match_stats", {})
                        if player_name and isinstance(stats, dict):
                            self._score_baseline[player_name] = int(stats.get("score", 0) or 0)

            if round_phase != "live":
                return

            if bomb_state in ("planted", "planting") or round_phase == "over":
                if not self._frozen:
                    self._frozen = True
                    logger.info(
                        "Prediction frozen (bomb=%s phase=%s)",
                        bomb_state,
                        round_phase,
                    )
                return
            if self._frozen:
                return

            self.map_name = map_name
            self.round_num = round_num
            self.ct_score = int(map_info.get("team_ct", {}).get("score", 0))
            self.t_score = int(map_info.get("team_t", {}).get("score", 0))

            # Time-bucket to 2 Hz so step index matches training (target_tick_rate=2).
            if self._round_start_time is None:
                self._round_start_time = time.monotonic()
            elapsed = time.monotonic() - self._round_start_time
            expected_step = int(elapsed / _STEP_INTERVAL_SEC)
            if expected_step < self._step:
                return

            row = build_row_from_gsi(
                merged,
                self._step,
                round_num,
                map_name,
                score_baseline=self._score_baseline,
            )
            if row is None:
                return

            self._last_map_state = {
                "map_name": map_name,
                "round_num": round_num,
                "ct_score": self.ct_score,
                "t_score": self.t_score,
                "map_phase": map_phase,
                "round_phase": round_phase,
                "bomb_state": bomb_state,
            }
            self._ingest_row_locked(row, expected_step, predictor)

    def update_from_row(
        self,
        row: dict[str, float],
        map_state: dict,
        predictor: RoundPredictor,
    ) -> None:
        """Advance the live prediction window from a raw memory-derived row."""
        map_name = (
            str(map_state.get("map_name", ""))
            or self._infer_map_name_from_row(row)
            or self.map_name
        )
        round_num = int(map_state.get("round_num", 0) or 0)
        map_phase = str(map_state.get("map_phase", "")).lower()
        round_phase = str(map_state.get("round_phase", "")).lower()
        bomb_state = str(map_state.get("bomb_state", "")).lower()

        logger.info(
            "MEM: map=%s round=%s map_phase=%s round_phase=%s bomb=%s",
            map_name,
            round_num,
            map_phase,
            round_phase,
            bomb_state,
        )

        with self._lock:
            self._last_map_state = dict(map_state)
            if map_name:
                self._last_map_state["map_name"] = map_name

            if map_phase != "live" or not map_name:
                return

            if round_num != self._round:
                # Only wait for a freezetime->live transition if we've already
                # observed a prior round. Initial attach (self._round == -1)
                # might land mid-round, so begin ingesting immediately.
                first_seen = self._round == -1
                self._reset_round_locked(round_num, await_freeze=not first_seen)

            prev_phase = self._prev_round_phase
            self._prev_round_phase = round_phase

            # Freezetime -> live marks the true start of the action phase.
            if (
                self._awaiting_freeze_to_live
                and prev_phase == "freezetime"
                and round_phase == "live"
            ):
                self._awaiting_freeze_to_live = False

            # While awaiting freeze->live, the bomb/phase flags still reflect
            # the tail of the previous round and must not mutate state on the
            # fresh round — otherwise a lingering "planted" flag would re-freeze
            # the round we just reset and stall step advancement until the
            # round after that resets _frozen again.
            if self._awaiting_freeze_to_live:
                return

            if bomb_state in {"planting", "planted"} or round_phase == "over":
                if not self._frozen:
                    self._frozen = True
                    logger.info(
                        "Prediction frozen (bomb=%s phase=%s)",
                        bomb_state,
                        round_phase,
                    )
                return

            if round_phase != "live" or self._frozen:
                return

            self.map_name = map_name
            self.round_num = round_num
            self.ct_score = int(map_state.get("ct_score", 0) or 0)
            self.t_score = int(map_state.get("t_score", 0) or 0)

            if self._round_start_time is None:
                self._round_start_time = time.monotonic()
            elapsed = time.monotonic() - self._round_start_time
            expected_step = int(elapsed / _STEP_INTERVAL_SEC)
            if expected_step < self._step:
                return

            self._ingest_row_locked(row, expected_step, predictor)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "A": self.prediction.get("A", 0.5),
                "B": self.prediction.get("B", 0.5),
                "map": self.map_name,
                "round": self.round_num,
                "ct_score": self.ct_score,
                "t_score": self.t_score,
                "steps": len(self._window),
                "frozen": self._frozen,
            }

    def debug_vec(self) -> dict:
        """Return the latest feature row + normalized vec for inspection."""
        with self._lock:
            if self._last_vec is None:
                return {"ready": False}
            vec = self._last_vec
            nonzero = [
                (FEATURE_NAMES[i], float(vec[i]))
                for i in range(len(vec))
                if abs(float(vec[i])) > 1e-6
            ]
            return {
                "ready": True,
                "steps": len(self._window),
                "nonzero_count": len(nonzero),
                "total_features": len(vec),
                "nonzero": nonzero,
                "raw_row": {k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in (self._last_row or {}).items()
                            if isinstance(v, (int, float)) and abs(float(v)) > 1e-6},
            }

    def debug_raw(self) -> dict:
        with self._lock:
            if self._merged_gsi:
                return dict(self._merged_gsi)
            return dict(self._last_map_state)


def create_app(
    checkpoint: Path,
    device: str = "cpu",
    input_mode: str = "gsi",
    map_override: str = "",
) -> Flask:
    predictor = RoundPredictor(checkpoint, device=device)
    game = _GameState()
    if map_override:
        game.map_name = map_override
    app = Flask(__name__, static_folder=str(_DASHBOARD))

    if input_mode == "gsi":
        def _ticker() -> None:
            while True:
                time.sleep(_STEP_INTERVAL_SEC / 2)
                try:
                    game.tick(predictor)
                except Exception:
                    logger.exception("ticker iteration failed")

        threading.Thread(target=_ticker, daemon=True).start()
    else:
        from src.inference.memory_reader import CS2MemoryReader
        from src.inference.memory_state_builder import build_row_from_memory

        reader = CS2MemoryReader.attach()

        def _memory_loop() -> None:
            score_baseline: dict[int, int] = {}
            pending_map = ""
            pending_map_count = 0
            while True:
                try:
                    map_state = reader.read_map_state()
                    game.note_map_state(map_state)
                    players = reader.read_players()
                    if map_state.get("map_phase") == "live" and players:
                        player_positions = [
                            (
                                float(player.get("x", 0.0) or 0.0),
                                float(player.get("y", 0.0) or 0.0),
                                float(player.get("z", 0.0) or 0.0),
                            )
                            for player in players
                        ]
                        canonical_map, pending_map, pending_map_count = _resolve_memory_map_name(
                            map_override=map_override,
                            advertised_map=str(map_state.get("map_name") or ""),
                            player_positions=player_positions,
                            cached_map=game.map_name,
                            pending_map=pending_map,
                            pending_count=pending_map_count,
                        )
                        row = build_row_from_memory(
                            players=players,
                            map_state=map_state,
                            round_num=int(map_state.get("round_num", 0) or 0),
                            map_name=canonical_map,
                            score_baseline=score_baseline,
                        )
                        if row is not None:
                            game.update_from_row(row, map_state, predictor)
                except Exception:
                    logger.exception("memory loop iteration failed")
                time.sleep(_STEP_INTERVAL_SEC)

        threading.Thread(target=_memory_loop, daemon=True).start()

    @app.post("/gsi")
    @app.post("/")
    def recv_gsi():
        data = request.get_json(silent=True, force=True)
        if input_mode == "gsi" and data:
            game.update(data, predictor)
        return "", 204

    @app.get("/state")
    def state():
        return jsonify(game.snapshot())

    @app.get("/debug")
    @app.get("/debug/vec")
    def debug_vec():
        return jsonify(game.debug_vec())

    @app.get("/debug/raw")
    def debug_raw():
        return jsonify(game.debug_raw())

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
    ap.add_argument("--checkpoint", default="checkpoints/v2_2hz/best.pt")
    ap.add_argument("--port", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--input", choices=("gsi", "memory"), default="gsi")
    ap.add_argument("--map", default="")
    args = ap.parse_args()

    logger.info("Loading checkpoint: %s", args.checkpoint)
    app = create_app(
        Path(args.checkpoint),
        device=args.device,
        input_mode=args.input,
        map_override=args.map,
    )
    logger.info("CS Prophet -> http://localhost:%d", args.port)
    logger.info("Open this URL in your browser, then start CS2 and load a demo.")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
