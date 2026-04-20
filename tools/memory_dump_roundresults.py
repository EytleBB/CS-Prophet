"""Dump C_CSGameRules.m_iMatchStats_RoundResults to verify the round-winner array.

We already know no C_CSTeam entity lives in the entity list in demo playback,
so cumulative CT/T scores must come from this per-round array on the gamerules.

Each slot is an int32 encoding the round-end reason (RoundEndReason_t):
  target_bombed=1, bomb_defused=7, ctwin=8, twin=9, ...

Usage:
    python tools/memory_dump_roundresults.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.memory_reader import CS2MemoryReader  # noqa: E402


_ROUND_RESULTS_OFFSET = 2488
_ROUND_RESULTS_SLOTS = 32  # 30 + padding; safe to read
_TOTAL_ROUNDS_OFFSET = 136
_ROUND_WIN_STATUS_OFFSET = 2476
_ROUND_END_WINNER_OFFSET = 3848

# Mapping derived from Source 2 RoundEndReason_t.
_CT_WIN = {5, 6, 7, 8, 11, 12, 14, 17}
_T_WIN = {1, 3, 4, 9, 13, 15, 18}


def main() -> int:
    try:
        reader = CS2MemoryReader.attach()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    pm = reader.pm
    rules_ptr = reader._resolve_game_rules_ptr()
    if rules_ptr == 0:
        print("ERROR: could not resolve C_CSGameRules pointer")
        return 2
    print(f"gamerules_ptr = {hex(rules_ptr)}")

    total = pm.read_int(rules_ptr + _TOTAL_ROUNDS_OFFSET)
    win_status = pm.read_int(rules_ptr + _ROUND_WIN_STATUS_OFFSET)
    winner_team = pm.read_int(rules_ptr + _ROUND_END_WINNER_OFFSET)
    print(f"m_totalRoundsPlayed  = {total}")
    print(f"m_iRoundWinStatus    = {win_status}")
    print(f"m_iRoundEndWinnerTeam= {winner_team}")
    print()

    raw = pm.read_bytes(rules_ptr + _ROUND_RESULTS_OFFSET, _ROUND_RESULTS_SLOTS * 4)
    import struct
    values = struct.unpack(f"<{_ROUND_RESULTS_SLOTS}i", raw)

    print("round_results array (raw int values):")
    for i, v in enumerate(values):
        team = "CT" if v in _CT_WIN else "T" if v in _T_WIN else "?"
        marker = "  <-- BEYOND total" if i >= total else ""
        print(f"  [{i:>2}] = {v:>3}  ({team}){marker}")

    ct_score = sum(1 for v in values[: max(0, total)] if v in _CT_WIN)
    t_score = sum(1 for v in values[: max(0, total)] if v in _T_WIN)
    print()
    print(f"Derived scores: CT={ct_score}  T={t_score}  (over first {total} slots)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
