from __future__ import annotations

import src.inference.realtime_engine as realtime_engine
from src.features.state_vector_v2 import FEATURE_NAMES


class _DummyPredictor:
    def predict(self, mat):  # noqa: ANN001
        return {"A": 0.6, "B": 0.4}


def _row(map_name: str = "de_mirage") -> dict[str, float]:
    row = {name: 0.0 for name in FEATURE_NAMES}
    row[f"map_{map_name}"] = 1.0
    row["t0_x"] = -500.0
    row["t0_y"] = -1500.0
    row["t0_z"] = 0.0
    row["t0_hp"] = 100.0
    row["t0_armor"] = 100.0
    row["t0_alive"] = 1.0
    return row


def test_memory_round_reset_does_not_accumulate_steps_during_freezetime(monkeypatch):
    clock_values = iter([100.0, 100.0, 170.0, 170.0])
    monkeypatch.setattr(realtime_engine.time, "monotonic", lambda: next(clock_values))

    game = realtime_engine._GameState()
    predictor = _DummyPredictor()
    row = _row()

    game.update_from_row(
        row,
        {
            "map_name": "de_mirage",
            "round_num": 1,
            "ct_score": 0,
            "t_score": 0,
            "map_phase": "live",
            "round_phase": "live",
            "bomb_state": "",
        },
        predictor,
    )
    assert game.snapshot()["steps"] == 1

    game.update_from_row(
        row,
        {
            "map_name": "de_mirage",
            "round_num": 2,
            "ct_score": 0,
            "t_score": 0,
            "map_phase": "live",
            "round_phase": "freezetime",
            "bomb_state": "",
        },
        predictor,
    )
    assert game.snapshot()["steps"] == 0
    assert game._round_start_time is None

    game.update_from_row(
        row,
        {
            "map_name": "de_mirage",
            "round_num": 2,
            "ct_score": 0,
            "t_score": 0,
            "map_phase": "live",
            "round_phase": "live",
            "bomb_state": "",
        },
        predictor,
    )
    assert game.snapshot()["steps"] == 1
    assert game.round_num == 2


def test_memory_blocks_ingestion_during_postround_gap(monkeypatch):
    """Between round win and next freezetime, round_num increments but
    round_phase can still read 'live'. Ingestion must pause until we've
    observed freezetime -> live for the new round.
    """
    # Only frames that actually ingest consume monotonic() calls (2 per ingest:
    # one to seed _round_start_time, one to compute elapsed). Blocked frames
    # short-circuit before touching the clock.
    clock = iter([0.0, 0.0, 100.0, 100.0])
    monkeypatch.setattr(realtime_engine.time, "monotonic", lambda: next(clock))

    game = realtime_engine._GameState()
    predictor = _DummyPredictor()
    row = _row()

    # Round 1: first ingest (fresh attach, no await needed).
    game.update_from_row(
        row,
        {"map_name": "de_mirage", "round_num": 1, "ct_score": 0, "t_score": 0,
         "map_phase": "live", "round_phase": "live", "bomb_state": ""},
        predictor,
    )
    assert game.snapshot()["steps"] == 1

    # Round 1 ends, round 2 announced while round_phase still reads "live"
    # (post-round gap). This frame MUST NOT start the clock for round 2.
    game.update_from_row(
        row,
        {"map_name": "de_mirage", "round_num": 2, "ct_score": 0, "t_score": 1,
         "map_phase": "live", "round_phase": "live", "bomb_state": ""},
        predictor,
    )
    assert game.snapshot()["steps"] == 0
    assert game._round_start_time is None
    assert game._awaiting_freeze_to_live is True

    # Still post-round, still live — block.
    game.update_from_row(
        row,
        {"map_name": "de_mirage", "round_num": 2, "ct_score": 0, "t_score": 1,
         "map_phase": "live", "round_phase": "live", "bomb_state": ""},
        predictor,
    )
    assert game.snapshot()["steps"] == 0

    # Freezetime begins — still block.
    game.update_from_row(
        row,
        {"map_name": "de_mirage", "round_num": 2, "ct_score": 0, "t_score": 1,
         "map_phase": "live", "round_phase": "freezetime", "bomb_state": ""},
        predictor,
    )
    assert game.snapshot()["steps"] == 0

    # Freezetime -> live: flag clears, step 0 gets ingested fresh.
    game.update_from_row(
        row,
        {"map_name": "de_mirage", "round_num": 2, "ct_score": 0, "t_score": 1,
         "map_phase": "live", "round_phase": "live", "bomb_state": ""},
        predictor,
    )
    assert game._awaiting_freeze_to_live is False
    assert game.snapshot()["steps"] == 1
    assert game.round_num == 2


def test_memory_stale_planted_bomb_does_not_stall_next_round(monkeypatch):
    """After a bomb-ended round, m_bBombPlanted can stay True for a frame or
    two after round_num bumps. That stale flag must NOT re-freeze the fresh
    round — otherwise step advancement stalls at 0 until the round after that
    resets _frozen again (what the user reported as 'stuck at step 0 until
    the next round').
    """
    # Only ingesting frames consume monotonic() (2 calls each). Blocked frames
    # short-circuit before touching the clock.
    clock = iter([0.0, 0.0, 100.0, 100.0, 200.0, 200.0])
    monkeypatch.setattr(realtime_engine.time, "monotonic", lambda: next(clock))

    game = realtime_engine._GameState()
    predictor = _DummyPredictor()
    row = _row()

    # Round 1 live: ingests step 0.
    game.update_from_row(
        row,
        {"map_name": "de_mirage", "round_num": 1, "ct_score": 0, "t_score": 0,
         "map_phase": "live", "round_phase": "live", "bomb_state": ""},
        predictor,
    )
    assert game.snapshot()["steps"] == 1

    # Round 1 ends: bomb planted, round_phase still live. _frozen locks.
    game.update_from_row(
        row,
        {"map_name": "de_mirage", "round_num": 1, "ct_score": 0, "t_score": 0,
         "map_phase": "live", "round_phase": "live", "bomb_state": "planted"},
        predictor,
    )
    assert game._frozen is True

    # Round 2 announced, but memory still reports bomb_state=planted for a
    # frame. Reset MUST clear _frozen AND the stale planted flag must not
    # re-freeze the new round.
    game.update_from_row(
        row,
        {"map_name": "de_mirage", "round_num": 2, "ct_score": 1, "t_score": 0,
         "map_phase": "live", "round_phase": "live", "bomb_state": "planted"},
        predictor,
    )
    assert game._awaiting_freeze_to_live is True
    assert game._frozen is False  # reset cleared it; stale bomb must not re-set

    # Freezetime starts, bomb clears. Still awaiting freeze->live.
    game.update_from_row(
        row,
        {"map_name": "de_mirage", "round_num": 2, "ct_score": 1, "t_score": 0,
         "map_phase": "live", "round_phase": "freezetime", "bomb_state": ""},
        predictor,
    )
    assert game.snapshot()["steps"] == 0
    assert game._frozen is False

    # Freezetime -> live: flag clears, fresh round ingests.
    game.update_from_row(
        row,
        {"map_name": "de_mirage", "round_num": 2, "ct_score": 1, "t_score": 0,
         "map_phase": "live", "round_phase": "live", "bomb_state": ""},
        predictor,
    )
    assert game._awaiting_freeze_to_live is False
    assert game._frozen is False
    assert game.snapshot()["steps"] == 1
    assert game.round_num == 2


def test_resolve_memory_map_name_keeps_cached_map_when_positions_still_fit(monkeypatch):
    monkeypatch.setattr(realtime_engine, "infer_map_from_positions", lambda positions: "de_overpass")

    resolved, pending, count = realtime_engine._resolve_memory_map_name(
        map_override="",
        advertised_map="",
        player_positions=[(-500.0, -1500.0, 0.0)],
        cached_map="de_mirage",
        pending_map="de_overpass",
        pending_count=2,
    )

    assert resolved == "de_mirage"
    assert pending == ""
    assert count == 0


def test_resolve_memory_map_name_requires_consistent_frames_before_switch(monkeypatch):
    monkeypatch.setattr(realtime_engine, "infer_map_from_positions", lambda positions: "de_inferno")

    args = dict(
        map_override="",
        advertised_map="",
        player_positions=[
            (500.0, 2800.0, 0.0),
            (700.0, 2900.0, 0.0),
            (900.0, 2600.0, 0.0),
            (1100.0, 2500.0, 0.0),
        ],
        cached_map="de_mirage",
    )

    resolved, pending, count = realtime_engine._resolve_memory_map_name(
        **args,
        pending_map="",
        pending_count=0,
    )
    assert resolved == "de_mirage"
    assert pending == "de_inferno"
    assert count == 1

    resolved, pending, count = realtime_engine._resolve_memory_map_name(
        **args,
        pending_map=pending,
        pending_count=count,
    )
    assert resolved == "de_mirage"
    assert pending == "de_inferno"
    assert count == 2

    resolved, pending, count = realtime_engine._resolve_memory_map_name(
        **args,
        pending_map=pending,
        pending_count=count,
    )
    assert resolved == "de_inferno"
    assert pending == ""
    assert count == 0


def test_resolve_memory_map_name_tolerates_single_outlier_when_cached_map_majority_fits(monkeypatch):
    monkeypatch.setattr(realtime_engine, "infer_map_from_positions", lambda positions: "de_inferno")

    resolved, pending, count = realtime_engine._resolve_memory_map_name(
        map_override="",
        advertised_map="",
        player_positions=[
            (-500.0, -1500.0, 0.0),
            (-700.0, -1200.0, 0.0),
            (-300.0, -1700.0, 0.0),
            (900.0, 2600.0, 0.0),
        ],
        cached_map="de_mirage",
        pending_map="",
        pending_count=0,
    )

    assert resolved == "de_mirage"
    assert pending == ""
    assert count == 0
