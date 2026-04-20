#!/usr/bin/env python3
"""Compare captured CS2 GSI payloads vs pkl-synthesized GSI payloads."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.processed_v2 import load_full_payload
from src.utils.paths import data_root
from tools.verify_train_infer_parity import _build_gsi, _infer_map_name, _resolve_pkl, _safe_int

FIELD_PATHS = """
provider map map.name map.phase map.round map.team_ct.score map.team_t.score
round round.phase round.bomb round.win_team player allplayers
allplayers.<pid>.name allplayers.<pid>.team allplayers.<pid>.position allplayers.<pid>.forward
allplayers.<pid>.state.health allplayers.<pid>.state.armor allplayers.<pid>.state.money
allplayers.<pid>.state.equip_value allplayers.<pid>.state.helmet allplayers.<pid>.weapons
allplayers.<pid>.match_stats.score bomb bomb.state bomb.position grenades
grenades.<id>.type grenades.<id>.position grenades.<id>.effecttime grenades.<id>.lifetime
grenades.<id>.flames phase_countdowns phase_countdowns.phase phase_countdowns.phase_ends_in
""".split()

BUILDER_PATHS = """
allplayers allplayers.<pid>.name allplayers.<pid>.team allplayers.<pid>.position
allplayers.<pid>.forward allplayers.<pid>.state.health allplayers.<pid>.state.armor
allplayers.<pid>.state.helmet allplayers.<pid>.state.money allplayers.<pid>.state.equip_value
allplayers.<pid>.match_stats.score allplayers.<pid>.weapons bomb.state bomb.position
grenades.<id>.type grenades.<id>.position grenades.<id>.effecttime grenades.<id>.lifetime
grenades.<id>.flames map.team_ct.score map.team_t.score phase_countdowns.phase_ends_in round.phase
""".split()

BUCKETS = [("<50", 0.0, 50.0), ("50-100", 50.0, 100.0), ("100-150", 100.0, 150.0), ("150-250", 150.0, 250.0), ("250-500", 250.0, 500.0), ("500-1000", 500.0, 1000.0), (">1000", 1000.0, math.inf)]


def _fmt(value: float | None, digits: int = 1) -> str: return "n/a" if value is None else f"{value:.{digits}f}"
def _pct(count: int, total: int) -> float: return 0.0 if total <= 0 else 100.0 * count / total


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _to_float(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _timestamp_ms(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) * 1000.0 if float(value) < 1e12 else float(value)
    text = str(value).strip()
    if not text:
        return None
    numeric = _to_float(text)
    if numeric is not None:
        return numeric * 1000.0 if numeric < 1e12 else numeric
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000.0
    except ValueError:
        return None


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, str):
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        return None
    nums = [_to_float(part) for part in parts]
    return None if any(num is None for num in nums) else (nums[0], nums[1], nums[2])


def _iter_values(obj: Any, parts: list[str]):
    if not parts:
        yield obj
        return
    head, rest = parts[0], parts[1:]
    if head in {"<pid>", "<id>"}:
        if isinstance(obj, dict):
            for value in obj.values():
                yield from _iter_values(value, rest)
        return
    if isinstance(obj, dict) and head in obj:
        yield from _iter_values(obj[head], rest)


def _present(payload: dict[str, Any], path: str) -> bool:
    return any(True for _ in _iter_values(payload, path.split(".")))


def _presence_pct(payloads: list[dict[str, Any]], path: str) -> float:
    return _pct(sum(1 for payload in payloads if _present(payload, path)), len(payloads))


def _values(payloads: list[dict[str, Any]], path: str) -> list[Any]:
    out, parts = [], path.split(".")
    for payload in payloads:
        out.extend(_iter_values(payload, parts))
    return out


def _nums(payloads: list[dict[str, Any]], path: str) -> list[float]:
    out = []
    for value in _values(payloads, path):
        num = _to_float(value)
        if num is not None:
            out.append(num)
    return out


def _vec_axis(payloads: list[dict[str, Any]], path: str, axis: int) -> list[float]:
    out = []
    for value in _values(payloads, path):
        parsed = _vec3(value)
        if parsed is not None:
            out.append(parsed[axis])
    return out


def _summary(values: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    return (
        min(values) if values else None,
        max(values) if values else None,
        _percentile(values, 0.50),
        _percentile(values, 0.95),
    )


def _range_flag(real_vals: list[float], synth_vals: list[float]) -> str:
    if not real_vals or not synth_vals:
        return ""
    r_span, s_span = max(real_vals) - min(real_vals), max(synth_vals) - min(synth_vals)
    nonzero = [span for span in (r_span, s_span) if span > 0.0]
    if not nonzero:
        return ""
    ratio = max(r_span, s_span) / min(nonzero)
    if ratio >= 4.0:
        return f"width x{ratio:.1f}"
    if min(real_vals) > max(synth_vals) or max(real_vals) < min(synth_vals):
        return "no overlap"
    return ""


def _timestamp_from_payload(payload: dict[str, Any]) -> tuple[str | None, float | None]:
    capture = _timestamp_ms(payload.get("_capture_ts"))
    if capture is not None:
        return "_capture_ts", capture
    provider = payload.get("provider", {})
    if isinstance(provider, dict):
        ts = _timestamp_ms(provider.get("timestamp"))
        if ts is not None:
            return "provider.timestamp", ts
    return None, None


def _interval_stats(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    source, stamps = None, []
    for payload in payloads:
        this_source, stamp = _timestamp_from_payload(payload)
        if stamp is None:
            continue
        source = source or this_source
        stamps.append(stamp)
    intervals = [b - a for a, b in zip(stamps, stamps[1:]) if b >= a]
    histogram = Counter()
    for value in intervals:
        for label, lo, hi in BUCKETS:
            if lo <= value < hi:
                histogram[label] += 1
                break
    return {
        "source": source,
        "count": len(intervals),
        "mean": statistics.mean(intervals) if intervals else None,
        "median": statistics.median(intervals) if intervals else None,
        "p95": _percentile(intervals, 0.95),
        "min": min(intervals) if intervals else None,
        "max": max(intervals) if intervals else None,
        "hist": histogram,
    }


def _load_real(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"Real GSI dump not found: {path}")
        return []
    payloads = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed JSONL line {line_no}: {exc}")
                continue
            if isinstance(obj, dict):
                payloads.append(obj)
    if not payloads:
        print(f"Real GSI dump is empty or unreadable: {path}")
    return payloads


def _real_map_name(payloads: list[dict[str, Any]]) -> str | None:
    counts = Counter()
    for payload in payloads:
        map_info = payload.get("map", {})
        if isinstance(map_info, dict) and map_info.get("name"):
            counts[str(map_info["name"])] += 1
    return counts.most_common(1)[0][0] if counts else None


def _resolve_pkl_for_map(real_map_name: str | None, explicit: str | None) -> Path | None:
    if explicit:
        return _resolve_pkl(explicit)
    if real_map_name:
        matches = sorted((data_root() / "viz").glob(f"*_{real_map_name}_full.pkl"))
        if matches:
            return matches[0]
    try:
        return _resolve_pkl(None)
    except FileNotFoundError:
        return None


def _fallback_synth(real_map_name: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample = {
        "name": "T_Player",
        "team": "T",
        "position": "0, 0, 0",
        "forward": "1, 0, 0",
        "state": {"health": 100, "armor": 0, "helmet": False, "money": 800, "equip_value": 0},
        "match_stats": {"score": 0},
        "weapons": {},
    }
    synth = [{
        "map": {"name": real_map_name or "unknown", "phase": "live", "round": 1, "team_ct": {"score": 0}, "team_t": {"score": 0}},
        "round": {"phase": "live"},
        "phase_countdowns": {"phase_ends_in": 115.0},
        "bomb": {"state": "carried", "position": "0, 0, 0"},
        "grenades": {},
        "allplayers": {"t_0": sample, "ct_0": dict(sample | {"name": "CT_Player", "team": "CT"})},
    }]
    return synth, {"path": None, "map_name": real_map_name or "unknown", "gaps": [], "cadence_ms": []}


def _build_synth(pkl_path: Path | None, real_map_name: str | None, max_ticks: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if pkl_path is None:
        return _fallback_synth(real_map_name)
    payload = load_full_payload(pkl_path)
    tick_df, events, round_info = payload["tick_df"], payload["events"], payload["round_info"]
    map_name = _infer_map_name(pkl_path, real_map_name, payload.get("header", {}) or {})
    unique_ticks = tick_df[tick_df["round_num"].notna()][["round_num", "tick"]].drop_duplicates().sort_values(["round_num", "tick"]).head(max_ticks)
    if unique_ticks.empty:
        return _fallback_synth(real_map_name)
    tick_groups, gaps, synth, cadence_ms = tick_df.groupby(["round_num", "tick"], sort=False), set(), [], []
    prev_round, prev_tick = None, None
    for item in unique_ticks.itertuples(index=False):
        round_num, tick = _safe_int(item.round_num), _safe_int(item.tick)
        tick_slice = tick_groups.get_group((item.round_num, item.tick))
        synth.append(_build_gsi(tick_slice, events, round_info, tick, round_num, map_name, gaps))
        if prev_round == round_num and prev_tick is not None and tick >= prev_tick:
            cadence_ms.append((tick - prev_tick) * 1000.0 / 64.0)
        prev_round, prev_tick = round_num, tick
    return synth, {"path": pkl_path, "map_name": map_name, "gaps": sorted(gaps), "cadence_ms": cadence_ms}


def _presence_rows(real_payloads: list[dict[str, Any]], synth_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for path in FIELD_PATHS:
        real_pct = _presence_pct(real_payloads, path)
        synth_present = any(_present(payload, path) for payload in synth_payloads)
        rows.append({"path": path, "real_pct": real_pct, "synth": "yes" if synth_present else "no", "gap": abs(real_pct - (100.0 if synth_present else 0.0))})
    rows.sort(key=lambda row: (-row["gap"], row["path"]))
    return rows


def _score_behavior(payloads: list[dict[str, Any]]) -> tuple[str, str]:
    per_player: dict[str, list[tuple[int, int]]] = defaultdict(list)
    max_score, max_key, max_round = -1, None, 0
    for payload in payloads:
        map_info = payload.get("map", {})
        round_num = _safe_int(map_info.get("round", 0), 0) if isinstance(map_info, dict) else 0
        allplayers = payload.get("allplayers", {})
        if not isinstance(allplayers, dict):
            continue
        for key, pdata in allplayers.items():
            if not isinstance(pdata, dict):
                continue
            stats = pdata.get("match_stats", {})
            score = _to_float(stats.get("score")) if isinstance(stats, dict) else None
            if score is None:
                continue
            score_int = int(score)
            per_player[str(key)].append((round_num, score_int))
            if score_int > max_score:
                max_score, max_key, max_round = score_int, str(key), round_num
    transitions = nondecreasing = resets = 0
    example = None
    for key, entries in per_player.items():
        rounds: dict[int, list[int]] = defaultdict(list)
        for round_num, score in entries:
            rounds[round_num].append(score)
        items = sorted((round_num, values[0], values[-1]) for round_num, values in rounds.items() if round_num > 0)
        if len(items) >= 2 and example is None:
            example = (key, items[:5])
        for (_, _, prev_last), (_, next_first, _) in zip(items, items[1:]):
            transitions += 1
            if next_first >= prev_last - 1:
                nondecreasing += 1
            if prev_last >= 5 and next_first <= 2:
                resets += 1
    if transitions:
        label = "accumulated" if nondecreasing / transitions >= 0.8 and resets / transitions <= 0.2 else "per-round"
        if example:
            key, items = example
            detail = "; ".join(f"R{round_num}:{first}->{last}" for round_num, first, last in items)
            return label, f"{key}: {detail}; transitions_nonreset={nondecreasing}/{transitions}"
        return label, f"transitions_nonreset={nondecreasing}/{transitions}; resets={resets}/{transitions}"
    if max_score >= 6:
        return "accumulated", f"single-round max score {max_score} for {max_key} in round {max_round} exceeds plausible per-round kills"
    if 0 <= max_score <= 5:
        return "per-round", f"single-round observed max score only {max_score}; no cross-round carry visible"
    return "unknown", "No usable score observations found."


def _player_report(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    key_kinds, examples, key_to_names, round_keysets, non_ascii = Counter(), [], defaultdict(set), defaultdict(list), set()
    for payload in payloads:
        map_info, allplayers = payload.get("map", {}), payload.get("allplayers", {})
        round_num = _safe_int(map_info.get("round", 0), 0) if isinstance(map_info, dict) else 0
        if not isinstance(allplayers, dict) or not allplayers:
            continue
        round_keysets[round_num].append(tuple(sorted(str(key) for key in allplayers)))
        for key, pdata in allplayers.items():
            text = str(key)
            if len(examples) < 2 and text not in examples:
                examples.append(text)
            key_kinds["steamid64" if re.fullmatch(r"\d{15,20}", text) else "player_N" if re.fullmatch(r"player_\d+", text) else "other"] += 1
            if isinstance(pdata, dict):
                name = str(pdata.get("name", ""))
                if name:
                    key_to_names[text].add(name)
                    if any(ord(ch) > 127 for ch in name):
                        non_ascii.add(name)
    stable_rounds = total_rounds = 0
    for keysets in round_keysets.values():
        if not keysets:
            continue
        total_rounds += 1
        if Counter(keysets).most_common(1)[0][1] == len(keysets):
            stable_rounds += 1
    unstable = [key for key, names in key_to_names.items() if len(names) > 1][:5]
    return {
        "kind": key_kinds.most_common(1)[0][0] if key_kinds else "unknown",
        "examples": examples,
        "stability": f"{stable_rounds}/{total_rounds} rounds fully stable" if total_rounds else "n/a",
        "unstable": unstable,
        "non_ascii": sorted(non_ascii)[:10],
    }


def _builder_missing(real_payloads: list[dict[str, Any]], synth_payloads: list[dict[str, Any]], builder_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    for path in BUILDER_PATHS:
        real_pct = _presence_pct(real_payloads, path)
        if real_pct < 95.0:
            rows.append({"path": path, "real_pct": real_pct, "synth": "yes" if any(_present(payload, path) for payload in synth_payloads) else "no"})
    rows.sort(key=lambda row: (row["real_pct"], row["path"]))
    grepped = sorted(set(re.findall(r'get\("([^"]+)"', builder_text)))
    return rows, grepped


def _auto_verdict(rows: list[dict[str, Any]], intervals: dict[str, Any], score_label: str) -> tuple[str, list[str]]:
    row_map = {row["path"]: row for row in rows}
    if row_map.get("allplayers", {}).get("real_pct", 100.0) < 80.0 or (row_map.get("grenades.<id>.type", {}).get("real_pct", 100.0) < 5.0 and row_map.get("grenades.<id>.type", {}).get("synth") == "yes"):
        return "allplayers/grenades missing", [
            "Capture merged payloads as well as raw payloads during diagnostics.",
            "Validate the realtime merge path for `bomb`, `grenades`, and `phase_countdowns` because those branches replace wholesale.",
            "Keep `allplayers_*` and grenade payloads enabled in the CS2 cfg during GOTV/demo playback.",
        ]
    if (intervals["mean"] or 0.0) >= 100.0 or (intervals["p95"] or 0.0) >= 250.0:
        return "cadence too slow/bursty", [
            "Reduce GSI `throttle`/`buffer` or retrain with sparse windows that match live cadence.",
            "Use `_capture_ts` for timing diagnostics because `provider.timestamp` is only second-level here.",
            "Inspect duplicate-vector suppression in realtime inference; bursty snapshots plus dedupe shrink the live window further.",
        ]
    if score_label == "accumulated":
        return "score accumulates", [
            "Normalize or delta-encode `match_stats.score` if live and offline semantics diverge.",
            "Check the offline pipeline uses the same scoreboard meaning as live GSI.",
            "Reset score-derived features at round start if the model only needs round-local state.",
        ]
    return "shapes match — gap elsewhere", [
        "Compare post-merge live payloads against synth, not only raw fragments.",
        "Instrument `build_row_from_gsi` output during live runs to catch ordering or normalization drift.",
        "Verify predictor windowing and round-reset logic with a live dump replay.",
    ]


def _report(real_payloads: list[dict[str, Any]], real_path: Path, synth_payloads: list[dict[str, Any]], synth_meta: dict[str, Any]) -> str:
    key_counts = Counter()
    keysets = Counter(", ".join(sorted(payload.keys())) for payload in real_payloads)
    for payload in real_payloads:
        key_counts.update(payload.keys())
    real_map = _real_map_name(real_payloads)
    incremental = sum(1 for payload in real_payloads if "added" in payload or "previously" in payload)
    intervals = _interval_stats(real_payloads)
    presence = _presence_rows(real_payloads, synth_payloads)
    score_label, score_evidence = _score_behavior(real_payloads)
    player_info = _player_report(real_payloads)
    builder_text = (REPO_ROOT / "src" / "inference" / "gsi_state_builder.py").read_text(encoding="utf-8")
    builder_rows, grepped = _builder_missing(real_payloads, synth_payloads, builder_text)
    verdict, fixes = _auto_verdict(presence, intervals, score_label)
    synth_mean = statistics.mean(synth_meta["cadence_ms"]) if synth_meta["cadence_ms"] else None
    synth_p95 = _percentile(synth_meta["cadence_ms"], 0.95)
    lines = [
        "# Real vs Synth GSI Report",
        "",
        f"- real_dump: `{real_path}`",
        f"- real_payloads: `{len(real_payloads)}`",
        f"- real_map_name: `{real_map or 'unknown'}`",
        f"- synth_source: `{synth_meta.get('path') or 'fallback synthetic tick'}`",
        f"- synth_payloads: `{len(synth_payloads)}`",
        f"- raw_payload_shape: `incremental_markers={incremental} ({_fmt(_pct(incremental, len(real_payloads)))}%) full_like={len(real_payloads) - incremental} ({_fmt(_pct(len(real_payloads) - incremental, len(real_payloads)))}%)`",
        "- note: `src/inference/realtime_engine.py` merges incremental payloads, but replaces `bomb`, `grenades`, and `phase_countdowns` wholesale.",
        "",
        "## Top-Level Keys",
        "",
        "| key | real % present |",
        "| --- | ---: |",
        *[f"| {key} | {_fmt(_pct(count, len(real_payloads)))} |" for key, count in sorted(key_counts.items())],
        "",
        f"- most_common_keyset: `{keysets.most_common(1)[0][0] if keysets else 'n/a'}`",
        "",
        "## A. Payload Cadence",
        "",
        f"- timestamp_source: `{intervals['source'] or 'absent'}`",
        f"- real_intervals_ms: `count={intervals['count']} mean={_fmt(intervals['mean'])} median={_fmt(intervals['median'])} p95={_fmt(intervals['p95'])} min={_fmt(intervals['min'])} max={_fmt(intervals['max'])}`",
        f"- synth_tick_dense_rate_ms: `samples={len(synth_meta['cadence_ms'])} mean={_fmt(synth_mean)} p95={_fmt(synth_p95)}`",
        "- commentary: real capture is far sparser than synth when the mean real interval sits well above the synth tick interval.",
        "",
        "| bucket_ms | count |",
        "| --- | ---: |",
        *[f"| {label} | {intervals['hist'].get(label, 0)} |" for label, _, _ in BUCKETS],
        "",
        "## B. Field Presence Matrix",
        "",
        "| path | real % present | synth | abs gap |",
        "| --- | ---: | --- | ---: |",
        *[f"| {row['path']} | {_fmt(row['real_pct'])} | {row['synth']} | {_fmt(row['gap'])} |" for row in presence],
        "",
        "## C. Value Distributions",
        "",
        "| field | side | min | max | p50 | p95 | flag |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, real_vals, synth_vals in [
        ("allplayers.<pid>.position.x", _vec_axis(real_payloads, "allplayers.<pid>.position", 0), _vec_axis(synth_payloads, "allplayers.<pid>.position", 0)),
        ("allplayers.<pid>.position.y", _vec_axis(real_payloads, "allplayers.<pid>.position", 1), _vec_axis(synth_payloads, "allplayers.<pid>.position", 1)),
        ("allplayers.<pid>.position.z", _vec_axis(real_payloads, "allplayers.<pid>.position", 2), _vec_axis(synth_payloads, "allplayers.<pid>.position", 2)),
        ("allplayers.<pid>.state.money", _nums(real_payloads, "allplayers.<pid>.state.money"), _nums(synth_payloads, "allplayers.<pid>.state.money")),
        ("allplayers.<pid>.state.equip_value", _nums(real_payloads, "allplayers.<pid>.state.equip_value"), _nums(synth_payloads, "allplayers.<pid>.state.equip_value")),
        ("allplayers.<pid>.match_stats.score", _nums(real_payloads, "allplayers.<pid>.match_stats.score"), _nums(synth_payloads, "allplayers.<pid>.match_stats.score")),
    ]:
        flag = _range_flag(real_vals, synth_vals)
        for side, values in (("real", real_vals), ("synth", synth_vals)):
            low, high, p50, p95 = _summary(values)
            lines.append(f"| {label} | {side} | {_fmt(low)} | {_fmt(high)} | {_fmt(p50)} | {_fmt(p95)} | {flag if side == 'real' else ''} |")
    lines.extend([
        "",
        "| enum field | real distinct | synth distinct |",
        "| --- | --- | --- |",
        *[f"| {field} | {', '.join(sorted({str(v) for v in _values(real_payloads, field) if v not in (None, '')})) or '_none_'} | {', '.join(sorted({str(v) for v in _values(synth_payloads, field) if v not in (None, '')})) or '_none_'} |" for field in ("bomb.state", "round.phase", "map.phase", "grenades.<id>.type")],
        "",
        f"- match_stats.score_behavior: `{score_label}`",
        f"- evidence: `{score_evidence}`",
        "",
        "## D. Player Keying & Ordering",
        "",
        f"- key_format: `{player_info['kind']}`",
        f"- example_keys: `{', '.join(player_info['examples']) or 'n/a'}`",
        f"- keyset_stability_by_round: `{player_info['stability']}`",
        f"- keys_with_name_changes: `{', '.join(player_info['unstable']) or 'none'}`",
        f"- non_ascii_names: `{', '.join(player_info['non_ascii']) or 'none'}`",
        "- note: `gsi_state_builder.py` sorts T/CT players alphabetically by `name`, so unstable or non-ASCII names can perturb slot assignment.",
        "",
        "## E. Missing/Rare Builder Inputs",
        "",
        f"- grepped_get_tokens: `{', '.join(grepped)}`",
        "",
        "| path | real % present | synth |",
        "| --- | ---: | --- |",
        *[f"| {row['path']} | {_fmt(row['real_pct'])} | {row['synth']} |" for row in builder_rows],
        "",
        "## F. Auto-Verdict",
        "",
        f"- verdict: `{verdict}`",
        "1. " + fixes[0],
        "2. " + fixes[1],
        "3. " + fixes[2],
    ])
    if synth_meta.get("gaps"):
        lines.extend(["", "## Synth Gaps", "", *[f"- {gap}" for gap in synth_meta["gaps"]]])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", default="data/gsi_dump.jsonl")
    parser.add_argument("--pkl")
    parser.add_argument("--max-synth", type=int, default=500)
    parser.add_argument("--out", default="tools/real_vs_synth_report.md")
    args = parser.parse_args()
    real_path = Path(args.real).expanduser()
    real_payloads = _load_real(real_path)
    if not real_payloads:
        return
    synth_payloads, synth_meta = _build_synth(_resolve_pkl_for_map(_real_map_name(real_payloads), args.pkl), _real_map_name(real_payloads), args.max_synth)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = _report(real_payloads, real_path, synth_payloads, synth_meta)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path} ({len(report)} chars)")


if __name__ == "__main__":
    main()
