#!/usr/bin/env python3
"""Compare offline parquet vs realtime GSI predictions on the same round."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.processed_v2 import load_full_payload
from src.features.state_vector_v2 import FEATURE_DIM, FEATURE_NAMES, build_state_vector
from src.inference.gsi_state_builder import build_row_from_gsi
from src.inference.predictor import RoundPredictor
from src.inference.realtime_engine import _GameState, _MAX_STEPS, _merge_payload
# Reuse the existing pkl->GSI payload synthesizer from the parity tool.
from tools.verify_train_infer_parity import _build_gsi, _infer_map_name

TIME_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-20", 0.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-80", 60.0, 80.0),
    ("80-115", 80.0, 115.0),
)
PROGRESS_EVERY = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", required=True, help="Path to *_full.pkl")
    parser.add_argument("--parquet", required=True, help="Path to processed_v2 parquet")
    parser.add_argument("--round-num", type=int, required=True, help="Target round number")
    parser.add_argument("--checkpoint", default="checkpoints/v2/best.pt")
    parser.add_argument("--map-name", help="Optional map override")
    parser.add_argument("--out-dir", default="tools/compare_output")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return cleaned.strip("_") or "round"


def fmt_float(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.4f}"


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def round_row(round_info: pd.DataFrame, round_num: int) -> pd.Series:
    rows = round_info[round_info["round_num"] == round_num]
    if rows.empty:
        raise KeyError(f"round {round_num} not found in round_info")
    return rows.iloc[0]


def infer_demo_name(parquet_path: Path, round_df: pd.DataFrame) -> str:
    if "demo_name" in round_df.columns:
        values = sorted({str(v) for v in round_df["demo_name"].dropna().astype(str)})
        if values:
            return values[0]
    return parquet_path.stem


def infer_label(round_df: pd.DataFrame, round_num: int) -> str:
    labels = sorted({str(v) for v in round_df["bomb_site"].dropna().astype(str) if str(v) in {"A", "B"}})
    if len(labels) != 1:
        raise ValueError(f"round {round_num} expected one A/B label, got {labels}")
    return labels[0]


def print_progress(prefix: str, idx: int, total: int, tick: int | None = None) -> None:
    if idx == 1 or idx == total or idx % PROGRESS_EVERY == 0:
        tick_part = f" tick={tick}" if tick is not None else ""
        print(f"[{prefix}] {idx}/{total}{tick_part}", flush=True)


def build_offline_records(round_df: pd.DataFrame, predictor: RoundPredictor) -> tuple[pd.DataFrame, str]:
    missing = [name for name in FEATURE_NAMES if name not in round_df.columns]
    if missing:
        raise KeyError(f"parquet missing feature columns: {missing[:10]}")

    label = infer_label(round_df, int(round_df["round_num"].iloc[0]))
    if len(round_df) > _MAX_STEPS:
        raise ValueError(f"round has {len(round_df)} steps, exceeds {_MAX_STEPS}")

    round_df = round_df.sort_values(["step", "tick"]).reset_index(drop=True)
    vectors = np.stack(
        [
            build_state_vector(round_df.loc[idx, FEATURE_NAMES])
            for idx in range(len(round_df))
        ],
        axis=0,
    )

    padded = np.zeros((_MAX_STEPS, FEATURE_DIM), dtype=np.float32)
    rows: list[dict[str, object]] = []
    total = len(round_df)
    for idx in range(total):
        padded[idx] = vectors[idx]
        probs = predictor.predict(padded)
        tick = int(round_df.loc[idx, "tick"])
        rows.append(
            {
                "tick": tick,
                "step_offline": int(round_df.loc[idx, "step"]),
                "time_in_round": float(round_df.loc[idx, "time_in_round"]),
                "p_A_offline": float(probs["A"]),
                "p_B_offline": float(probs["B"]),
                "label": label,
            }
        )
        print_progress("offline", idx + 1, total, tick=tick)
    return pd.DataFrame(rows), label


def build_realtime_records(
    tick_df: pd.DataFrame,
    events: dict,
    round_info: pd.DataFrame,
    round_num: int,
    map_name: str,
    predictor: RoundPredictor,
) -> tuple[pd.DataFrame, int, int]:
    round_tick_df = tick_df[tick_df["round_num"] == round_num].copy()
    if round_tick_df.empty:
        raise ValueError(f"round {round_num} missing from pkl tick_df")

    unique_ticks = round_tick_df[["tick"]].drop_duplicates().sort_values("tick")
    tick_groups = round_tick_df.groupby("tick", sort=False)
    game = _GameState()
    merged_debug: dict[str, object] = {}
    last_window_vec: np.ndarray | None = None
    dedup_skipped = 0
    no_growth_unclassified = 0
    prev_steps = 0
    freeze_tick = int(round_row(round_info, round_num)["freeze_tick"])

    rows: list[dict[str, object]] = []
    total = len(unique_ticks)
    for idx, tick_value in enumerate(unique_ticks["tick"].tolist(), start=1):
        tick = int(tick_value)
        tick_slice = tick_groups.get_group(tick)
        live_payload = _build_gsi(tick_slice, events, round_info, tick, round_num, map_name, set())

        if idx == 1:
            freezetime_payload = copy.deepcopy(live_payload)
            freezetime_payload.setdefault("round", {})
            freezetime_payload["round"]["phase"] = "freezetime"
            merged_debug = _merge_payload(merged_debug, freezetime_payload)
            game.update(freezetime_payload, predictor)

        merged_debug = _merge_payload(merged_debug, live_payload)
        game.update(live_payload, predictor)
        snapshot = game.snapshot()

        current_vec = None
        row = build_row_from_gsi(merged_debug, snapshot["steps"], round_num, map_name)
        if row is not None:
            current_vec = build_state_vector(row)

        step_grew = int(snapshot["steps"]) > prev_steps
        dedup_now = False
        if (
            not bool(snapshot["frozen"])
            and not step_grew
            and current_vec is not None
            and last_window_vec is not None
            and np.array_equal(last_window_vec, current_vec)
        ):
            dedup_skipped += 1
            dedup_now = True
        elif not bool(snapshot["frozen"]) and not step_grew:
            no_growth_unclassified += 1

        if step_grew and current_vec is not None:
            last_window_vec = current_vec
        prev_steps = int(snapshot["steps"])

        rows.append(
            {
                "tick": tick,
                "step_realtime": int(snapshot["steps"]),
                "time_in_round_rt": max(0.0, (tick - freeze_tick) / 64.0),
                "p_A_realtime": float(snapshot["A"]),
                "p_B_realtime": float(snapshot["B"]),
                "realtime_frozen": bool(snapshot["frozen"]),
                "dedup_skipped": dedup_now,
            }
        )
        print_progress("realtime", idx, total, tick=tick)

        if bool(snapshot["frozen"]):
            break

    realtime_df = pd.DataFrame(rows)
    if no_growth_unclassified:
        print(
            f"[realtime] warning: {no_growth_unclassified} live ticks had no window growth but were not classified as np.array_equal dedup",
            flush=True,
        )
    return realtime_df, dedup_skipped, no_growth_unclassified


def bucket_mask(series: pd.Series, start: float, stop: float, is_last: bool) -> pd.Series:
    if is_last:
        return (series >= start) & (series <= stop)
    return (series >= start) & (series < stop)


def path_accuracy(df: pd.DataFrame, prob_col: str, label: str, time_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, (bucket, start, stop) in enumerate(TIME_BUCKETS):
        mask = bucket_mask(df[time_col], start, stop, idx == len(TIME_BUCKETS) - 1)
        bucket_df = df.loc[mask]
        acc = math.nan
        if not bucket_df.empty:
            preds = (bucket_df[prob_col] >= 0.5).map({True: "A", False: "B"})
            acc = float((preds == label).mean())
        rows.append({"bucket": bucket, "accuracy": acc, "count": int(len(bucket_df))})
    return pd.DataFrame(rows)


def compare_buckets(offline_df: pd.DataFrame, realtime_df: pd.DataFrame, label: str) -> pd.DataFrame:
    left = path_accuracy(offline_df, "p_A_offline", label, "time_in_round").rename(
        columns={"accuracy": "offline_acc", "count": "offline_n"}
    )
    right = path_accuracy(realtime_df, "p_A_realtime", label, "time_in_round_rt").rename(
        columns={"accuracy": "realtime_acc", "count": "realtime_n"}
    )
    out = left.merge(right, on="bucket", how="inner")
    out["delta"] = out["realtime_acc"] - out["offline_acc"]
    return out


def detect_flatline_tick(
    realtime_df: pd.DataFrame,
    offline_ticks: list[int],
    min_future_ticks: int = 3,
) -> int | None:
    if realtime_df.empty:
        return None
    offline_sorted = sorted(int(tick) for tick in offline_ticks)
    offline_last_tick = offline_sorted[-1]
    probs = realtime_df["p_A_realtime"].to_numpy(dtype=np.float64)
    steps = realtime_df["step_realtime"].to_numpy(dtype=np.int64)
    ticks = realtime_df["tick"].to_numpy(dtype=np.int64)
    final_prob = probs[-1]
    final_step = steps[-1]

    for idx in range(len(realtime_df)):
        if ticks[idx] >= offline_last_tick:
            return None
        future_ticks = sum(1 for tick in offline_sorted if tick > ticks[idx])
        if future_ticks < min_future_ticks:
            continue
        if np.allclose(probs[idx:], final_prob, atol=1e-12) and np.all(steps[idx:] == final_step):
            return int(ticks[idx])
    return None


def choose_verdict(
    bucket_df: pd.DataFrame,
    rms_diff: float,
    window_divergence: float,
    flatline_tick: int | None,
) -> str:
    comparable = bucket_df.dropna(subset=["offline_acc", "realtime_acc"])
    bucket_gap = comparable["offline_acc"] - comparable["realtime_acc"] if not comparable.empty else pd.Series(dtype=float)
    acc_matches = bool(not comparable.empty and (bucket_gap.abs() <= 0.10).all())

    if not comparable.empty and (bucket_gap > 0.10).any() and window_divergence > 0.20:
        return "step-rate mismatch is the main cause"
    if acc_matches and rms_diff > 0.10:
        return "feature-level divergence still present beyond in_bomb_zone"
    if flatline_tick is not None:
        return "realtime freeze logic firing too early"
    return "realtime path roughly matches offline on this round"


def plot_curves(
    offline_df: pd.DataFrame,
    realtime_df: pd.DataFrame,
    plant_time: float,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(offline_df["time_in_round"], offline_df["p_A_offline"], color="tab:blue", linewidth=1.8, label="offline")
    ax.plot(realtime_df["time_in_round_rt"], realtime_df["p_A_realtime"], color="tab:red", linewidth=1.8, label="realtime")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.0)
    ax.axvline(plant_time, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("time_in_round (s)")
    ax.set_ylabel("P(A)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_diff(aligned_df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(aligned_df["time_in_round"], aligned_df["diff"], color="tab:purple", linewidth=1.8)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_xlabel("time_in_round (s)")
    ax.set_ylabel("P(A) offline - realtime")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def summarize_ticks(ticks: list[int], limit: int = 10) -> str:
    if not ticks:
        return "none"
    preview = ", ".join(str(tick) for tick in ticks[:limit])
    if len(ticks) > limit:
        preview += ", ..."
    return preview


def write_report(
    out_path: Path,
    demo_name: str,
    round_num: int,
    label: str,
    map_name: str,
    checkpoint: Path,
    parquet_path: Path,
    pkl_path: Path,
    offline_df: pd.DataFrame,
    realtime_df: pd.DataFrame,
    aligned_df: pd.DataFrame,
    path_a_only: list[int],
    path_b_only: list[int],
    dedup_skipped: int,
    no_growth_unclassified: int,
    window_divergence: float,
    mean_abs_diff: float,
    rms_diff: float,
    max_abs_diff: float,
    max_diff_tick: int | None,
    bucket_df: pd.DataFrame,
    verdict: str,
    flatline_tick: int | None,
) -> None:
    lines = [
        "# Offline vs Realtime Comparison",
        "",
        f"- demo: `{demo_name}`",
        f"- round_num: `{round_num}`",
        f"- label: `{label}`",
        f"- map_name: `{map_name}`",
        f"- checkpoint: `{checkpoint}`",
        f"- parquet: `{parquet_path}`",
        f"- pkl: `{pkl_path}`",
        "",
        "## Tick Alignment",
        "",
        f"- common_ticks: `{len(aligned_df)}`",
        f"- path_a_only_count: `{len(path_a_only)}`",
        f"- path_a_only_ticks: `{summarize_ticks(path_a_only)}`",
        f"- path_b_only_count: `{len(path_b_only)}`",
        f"- path_b_only_ticks: `{summarize_ticks(path_b_only)}`",
        "",
        "## Window Growth",
        "",
        f"- offline_steps: `{len(offline_df)}`",
        f"- realtime_steps_final: `{int(realtime_df['step_realtime'].max()) if not realtime_df.empty else 0}`",
        f"- window_divergence: `{window_divergence:.2%}`",
        f"- dedup_skipped_ticks: `{dedup_skipped}`",
        f"- no_growth_unclassified: `{no_growth_unclassified}`",
        f"- realtime_frozen_seen: `{bool(realtime_df['realtime_frozen'].any()) if not realtime_df.empty else False}`",
        f"- flatline_tick: `{flatline_tick if flatline_tick is not None else 'none'}`",
        "",
        "## Probability Diffs",
        "",
        f"- mean_abs_diff: `{mean_abs_diff:.6f}`",
        f"- rms_diff: `{rms_diff:.6f}`",
        f"- max_abs_diff: `{max_abs_diff:.6f}`",
        f"- max_abs_diff_tick: `{max_diff_tick if max_diff_tick is not None else 'none'}`",
        "",
        "## Bucket Accuracy",
        "",
        "| bucket | offline_acc | realtime_acc | delta | offline_n | realtime_n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bucket_df.itertuples(index=False):
        lines.append(
            f"| {row.bucket} | {fmt_float(float(row.offline_acc))} | {fmt_float(float(row.realtime_acc))} | "
            f"{fmt_float(float(row.delta))} | {int(row.offline_n)} | {int(row.realtime_n)} |"
        )

    lines.extend(["", "## Auto Verdict", "", verdict, ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    pkl_path = resolve_path(args.pkl)
    parquet_path = resolve_path(args.parquet)
    checkpoint = resolve_path(args.checkpoint)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] reading parquet", flush=True)
    parquet_df = pd.read_parquet(parquet_path)
    round_df = parquet_df[parquet_df["round_num"] == args.round_num].copy()
    if round_df.empty:
        raise ValueError(f"round {args.round_num} missing from parquet")
    round_df = round_df.sort_values(["step", "tick"]).reset_index(drop=True)

    payload = load_full_payload(pkl_path)
    tick_df = payload["tick_df"]
    events = payload["events"]
    round_info = payload["round_info"]
    map_name = _infer_map_name(pkl_path, args.map_name, payload.get("header", {}) or {})
    rr = round_row(round_info, args.round_num)
    freeze_tick = int(rr["freeze_tick"])
    plant_tick = int(rr["plant_tick"]) if pd.notna(rr["plant_tick"]) else int(round_df["tick"].iloc[-1])
    plant_time = max(0.0, (plant_tick - freeze_tick) / 64.0)

    demo_name = infer_demo_name(parquet_path, round_df)
    print(
        f"[round] demo={demo_name} round={args.round_num} label={infer_label(round_df, args.round_num)} steps={len(round_df)} map={map_name}",
        flush=True,
    )
    predictor = RoundPredictor(checkpoint, device=args.device)

    offline_df, label = build_offline_records(round_df, predictor)
    realtime_df, dedup_skipped, no_growth_unclassified = build_realtime_records(
        tick_df=tick_df,
        events=events,
        round_info=round_info,
        round_num=args.round_num,
        map_name=map_name,
        predictor=predictor,
    )

    path_a_ticks = sorted(int(v) for v in offline_df["tick"].tolist())
    path_b_ticks = sorted(int(v) for v in realtime_df["tick"].tolist())
    path_a_only = sorted(set(path_a_ticks) - set(path_b_ticks))
    path_b_only = sorted(set(path_b_ticks) - set(path_a_ticks))

    aligned_df = offline_df.merge(realtime_df, on="tick", how="inner")
    if aligned_df.empty:
        raise RuntimeError("no common ticks between offline and realtime paths")
    aligned_df["diff"] = aligned_df["p_A_offline"] - aligned_df["p_A_realtime"]
    aligned_df = aligned_df.sort_values("tick").reset_index(drop=True)

    csv_name = f"per_step_{sanitize_name(demo_name)}_{args.round_num}.csv"
    aligned_df.loc[
        :,
        [
            "tick",
            "step_offline",
            "step_realtime",
            "time_in_round",
            "p_A_offline",
            "p_A_realtime",
            "diff",
            "realtime_frozen",
            "label",
        ],
    ].to_csv(out_dir / csv_name, index=False)

    mean_abs_diff = float(aligned_df["diff"].abs().mean())
    rms_diff = float(np.sqrt(np.mean(np.square(aligned_df["diff"].to_numpy(dtype=np.float64)))))
    max_idx = aligned_df["diff"].abs().idxmax()
    max_abs_diff = float(abs(aligned_df.loc[max_idx, "diff"]))
    max_diff_tick = int(aligned_df.loc[max_idx, "tick"])
    final_realtime_steps = int(realtime_df["step_realtime"].max()) if not realtime_df.empty else 0
    window_divergence = max(0.0, len(offline_df) - final_realtime_steps) / max(1, len(offline_df))
    bucket_df = compare_buckets(offline_df, realtime_df, label)
    flatline_tick = detect_flatline_tick(realtime_df, path_a_ticks)
    verdict = choose_verdict(bucket_df, rms_diff, window_divergence, flatline_tick)

    base = f"{sanitize_name(demo_name)}_{args.round_num}"
    plot_curves(
        offline_df=offline_df,
        realtime_df=realtime_df,
        plant_time=plant_time,
        title=f"{demo_name} round {args.round_num} label={label}",
        out_path=out_dir / f"curves_{base}.png",
    )
    plot_diff(
        aligned_df=aligned_df,
        title=f"{demo_name} round {args.round_num} diff",
        out_path=out_dir / f"diff_{base}.png",
    )
    write_report(
        out_path=out_dir / f"report_{base}.md",
        demo_name=demo_name,
        round_num=args.round_num,
        label=label,
        map_name=map_name,
        checkpoint=checkpoint,
        parquet_path=parquet_path,
        pkl_path=pkl_path,
        offline_df=offline_df,
        realtime_df=realtime_df,
        aligned_df=aligned_df,
        path_a_only=path_a_only,
        path_b_only=path_b_only,
        dedup_skipped=dedup_skipped,
        no_growth_unclassified=no_growth_unclassified,
        window_divergence=window_divergence,
        mean_abs_diff=mean_abs_diff,
        rms_diff=rms_diff,
        max_abs_diff=max_abs_diff,
        max_diff_tick=max_diff_tick,
        bucket_df=bucket_df,
        verdict=verdict,
        flatline_tick=flatline_tick,
    )

    print(f"[done] wrote outputs to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
