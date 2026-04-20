#!/usr/bin/env python3
"""Analyze how round-level A/B probabilities evolve over time."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.state_vector_v2 import FEATURE_DIM, FEATURE_NAMES, build_state_matrix
from src.inference.predictor import RoundPredictor
from src.utils.paths import data_path, repo_data_root, resolve_path_input

SEQ_LEN = 720
LABELS = ("A", "B")
TIME_BUCKETS = [
    ("0-20", 0.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-80", 60.0, 80.0),
    ("80-115", 80.0, 115.0),
]


@dataclass(frozen=True)
class RoundRef:
    path: Path
    demo: str
    round_num: int
    label: str
    steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-step bomb-site probabilities on processed_v2 rounds.",
    )
    parser.add_argument("--checkpoint", default="checkpoints/v2/best.pt")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Processed_v2 parquet dir. Default: active data_path('processed_v2'), "
        "falling back to repo-local data/processed_v2 if the active dir is empty.",
    )
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument("--out-dir", default="tools/analysis_output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_data_dir(arg_value: str | None) -> tuple[Path, list[Path], list[str]]:
    notes: list[str] = []
    if arg_value:
        chosen = resolve_path_input(arg_value)
        parquet_files = sorted(chosen.glob("*.parquet")) if chosen.exists() else []
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {chosen}")
        return chosen, parquet_files, notes

    primary = data_path("processed_v2")
    primary_files = sorted(primary.glob("*.parquet")) if primary.exists() else []
    if primary_files:
        return primary, primary_files, notes

    fallback = repo_data_root() / "processed_v2"
    fallback_files = sorted(fallback.glob("*.parquet")) if fallback.exists() else []
    if fallback_files:
        notes.append(f"Active data root was empty; fell back to repo-local {fallback}")
        return fallback, fallback_files, notes

    raise FileNotFoundError(
        f"No parquet files found in {primary} or repo-local fallback {fallback}"
    )


def index_rounds(parquet_files: list[Path]) -> tuple[list[RoundRef], dict[str, int]]:
    round_refs: list[RoundRef] = []
    stats = {"indexed": 0, "kept": 0, "gt720": 0}
    for path in parquet_files:
        df = pd.read_parquet(path, columns=["demo_name", "round_num", "bomb_site"])
        stats["indexed"] += int(df[["demo_name", "round_num"]].drop_duplicates().shape[0])
        for (demo_name, round_num), group in df.groupby(["demo_name", "round_num"], sort=False):
            labels = sorted({str(v) for v in group["bomb_site"].dropna().astype(str) if str(v) in LABELS})
            if len(labels) != 1:
                continue
            steps = int(len(group))
            if steps > SEQ_LEN:
                stats["gt720"] += 1
            round_refs.append(
                RoundRef(
                    path=path,
                    demo=str(demo_name),
                    round_num=int(round_num),
                    label=labels[0],
                    steps=steps,
                )
            )
    stats["kept"] = len(round_refs)
    return round_refs, stats


def sample_rounds(round_refs: list[RoundRef], num_rounds: int, seed: int) -> tuple[list[RoundRef], list[str]]:
    notes: list[str] = []
    rng = np.random.default_rng(seed)
    targets = {"A": num_rounds // 2, "B": num_rounds - (num_rounds // 2)}
    selected: list[RoundRef] = []

    for label in LABELS:
        label_rounds = [r for r in round_refs if r.label == label]
        short_rounds = [r for r in label_rounds if r.steps <= SEQ_LEN]
        long_rounds = [r for r in label_rounds if r.steps > SEQ_LEN]
        target = targets[label]

        rng.shuffle(short_rounds)
        rng.shuffle(long_rounds)

        chosen = short_rounds[:target]
        if len(chosen) < target and long_rounds:
            needed = target - len(chosen)
            extra = long_rounds[:needed]
            chosen.extend(extra)
            notes.append(
                f"Used {len(extra)} {label}-rounds longer than {SEQ_LEN} steps; "
                "their analysis is truncated to the model-visible prefix."
            )
        if len(chosen) < target:
            notes.append(f"Only found {len(chosen)}/{target} requested {label}-rounds.")
        selected.extend(chosen)

    rng.shuffle(selected)
    return selected, notes


def sanitize_name(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return out.strip("_") or "round"


def choose_verdict(bucket_df: pd.DataFrame) -> str:
    metrics = {
        row["bucket"]: (float(row["accuracy"]), int(row["count"]))
        for _, row in bucket_df.iterrows()
    }
    early_acc, early_count = metrics.get("0-20", (float("nan"), 0))
    late_acc, late_count = metrics.get("80-115", (float("nan"), 0))
    valid_accs = [acc for acc, count in metrics.values() if count > 0]

    if early_count > 0 and early_acc > 0.7:
        return "model learned too well (likely label leakage / data snooping)"
    if early_count > 0 and late_count > 0 and 0.45 <= early_acc <= 0.55 and late_acc > 0.75:
        return "task definition issue: model only converges near plant time, training every tick with same label creates noise"
    if len(valid_accs) == len(TIME_BUCKETS) and all(0.5 <= acc <= 0.6 for acc in valid_accs):
        return "model did not learn, suspect data volume / loss / features"
    return "mixed signal, see plots"


def compute_bucket_summary(step_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, start, stop in TIME_BUCKETS:
        if label == TIME_BUCKETS[-1][0]:
            mask = (step_df["time_in_round"] >= start) & (step_df["time_in_round"] <= stop)
        else:
            mask = (step_df["time_in_round"] >= start) & (step_df["time_in_round"] < stop)
        bucket = step_df.loc[mask]
        if bucket.empty:
            mean_p = float("nan")
            acc = float("nan")
            count = 0
        else:
            mean_p = float(bucket["p_correct"].mean())
            acc = float(bucket["is_correct"].mean())
            count = int(len(bucket))
        rows.append(
            {
                "bucket": label,
                "mean_p_correct": mean_p,
                "accuracy": acc,
                "count": count,
            }
        )
    return pd.DataFrame(rows)


def plot_round(ax: plt.Axes, round_df: pd.DataFrame, title: str, plant_time: float) -> None:
    ax.plot(round_df["time_in_round"], round_df["p_A"], color="tab:blue", linewidth=1.8)
    ax.axvline(plant_time, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time_in_round (s)")
    ax.set_ylabel("P(A)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)


def save_round_plot(round_df: pd.DataFrame, round_ref: RoundRef, out_dir: Path, plant_time: float) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    title = f"{round_ref.demo} round {round_ref.round_num} truth={round_ref.label}"
    plot_round(ax, round_df, title, plant_time)
    fig.tight_layout()
    out_path = out_dir / f"{sanitize_name(round_ref.demo)}_{round_ref.round_num}_{round_ref.label}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_overview(round_entries: list[dict[str, object]], label: str, out_path: Path) -> None:
    subset = [entry for entry in round_entries if entry["label"] == label][:10]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=True)
    flat_axes = list(axes.flat)
    for ax, entry in zip(flat_axes, subset, strict=False):
        round_ref: RoundRef = entry["round_ref"]  # type: ignore[assignment]
        round_df: pd.DataFrame = entry["pred_df"]  # type: ignore[assignment]
        plant_time = float(entry["plant_time"])
        title = f"{round_ref.demo} r{round_ref.round_num} truth={round_ref.label}"
        plot_round(ax, round_df, title, plant_time)
    for ax in flat_axes[len(subset):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def format_float(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.3f}"


def build_summary(
    out_path: Path,
    args: argparse.Namespace,
    data_dir: Path,
    notes: list[str],
    selected: list[RoundRef],
    bucket_df: pd.DataFrame,
    final_df: pd.DataFrame,
    verdict: str,
) -> None:
    lines = [
        "# Round Prediction Analysis",
        "",
        f"Total rounds analyzed: {len(selected)}",
        "",
        "## Configuration",
        f"- checkpoint: `{Path(args.checkpoint)}`",
        f"- data_dir: `{data_dir}`",
        f"- device: `{args.device}`",
        f"- seed: `{args.seed}`",
        "",
        "## Sampled Rounds",
    ]
    for ref in selected:
        lines.append(f"- {ref.demo} round {ref.round_num} label={ref.label} steps={ref.steps}")
    if notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in notes)

    lines.extend(
        [
            "",
            "## Bucket Metrics",
            "",
            "| bucket | mean P(correct_site) | accuracy | count |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in bucket_df.iterrows():
        lines.append(
            f"| {row['bucket']} | {format_float(float(row['mean_p_correct']))} | "
            f"{format_float(float(row['accuracy']))} | {int(row['count'])} |"
        )

    lines.extend(
        [
            "",
            "## Final-Step Predictions",
            "",
            "| demo | round_num | label | final_p_A | final_p_B | final_pred | correct |",
            "| --- | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for _, row in final_df.iterrows():
        lines.append(
            f"| {row['demo']} | {int(row['round_num'])} | {row['label']} | "
            f"{row['final_p_A']:.3f} | {row['final_p_B']:.3f} | {row['final_pred']} | "
            f"{'yes' if bool(row['correct']) else 'no'} |"
        )

    lines.extend(["", "## Judgment", "", f"Verdict: {verdict}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    checkpoint = resolve_path_input(args.checkpoint)
    out_dir = resolve_path_input(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir, parquet_files, notes = resolve_data_dir(args.data_dir)
    print(f"[data] using {data_dir} ({len(parquet_files)} parquet files)")

    round_refs, stats = index_rounds(parquet_files)
    notes.append(
        f"Indexed {stats['kept']} labeled A/B rounds from {stats['indexed']} total rounds; "
        f"{stats['gt720']} labeled rounds exceed {SEQ_LEN} steps."
    )
    if not round_refs:
        raise RuntimeError(f"No labeled A/B rounds found in {data_dir}")

    selected, sample_notes = sample_rounds(round_refs, args.num_rounds, args.seed)
    notes.extend(sample_notes)
    if not selected:
        raise RuntimeError("Sampling produced no rounds to analyze")

    print("[sample] selected rounds:")
    for ref in selected:
        print(f"  - {ref.demo} round {ref.round_num} label={ref.label} steps={ref.steps}")

    predictor = RoundPredictor(checkpoint, device=args.device)
    cache: dict[Path, pd.DataFrame] = {}
    step_rows: list[dict[str, object]] = []
    round_entries: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    for ref in selected:
        if ref.path not in cache:
            cache[ref.path] = pd.read_parquet(ref.path)
        df = cache[ref.path]
        round_df = df[(df["demo_name"] == ref.demo) & (df["round_num"] == ref.round_num)].copy()
        round_df = round_df.sort_values("step").reset_index(drop=True)
        missing = [col for col in FEATURE_NAMES if col not in round_df.columns]
        if missing:
            raise KeyError(f"{ref.demo} round {ref.round_num} missing feature columns: {missing[:5]}")

        mat = build_state_matrix(round_df.loc[:, FEATURE_NAMES])
        analyzed_steps = min(len(mat), SEQ_LEN)
        if len(mat) > SEQ_LEN:
            notes.append(
                f"{ref.demo} round {ref.round_num} has {len(mat)} steps; "
                f"analysis was truncated to the first {SEQ_LEN} steps."
            )
        plant_rows = round_df[round_df["bomb_site"].astype(str).isin(LABELS)]
        plant_time = float(plant_rows["time_in_round"].iloc[-1])

        padded = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        round_step_rows: list[dict[str, object]] = []
        for idx in range(analyzed_steps):
            padded[idx] = mat[idx]
            probs = predictor.predict(padded)
            row = {
                "demo": ref.demo,
                "round_num": ref.round_num,
                "step": int(round_df.loc[idx, "step"]),
                "time_in_round": float(round_df.loc[idx, "time_in_round"]),
                "p_A": float(probs["A"]),
                "p_B": float(probs["B"]),
                "label": ref.label,
            }
            step_rows.append(row)
            round_step_rows.append(row)

        pred_df = pd.DataFrame(round_step_rows)
        png_path = save_round_plot(pred_df, ref, out_dir, plant_time)
        round_entries.append(
            {
                "round_ref": ref,
                "label": ref.label,
                "pred_df": pred_df,
                "plant_time": plant_time,
                "png_path": png_path,
            }
        )

        last = pred_df.iloc[-1]
        final_pred = "A" if float(last["p_A"]) >= float(last["p_B"]) else "B"
        final_rows.append(
            {
                "demo": ref.demo,
                "round_num": ref.round_num,
                "label": ref.label,
                "final_p_A": float(last["p_A"]),
                "final_p_B": float(last["p_B"]),
                "final_pred": final_pred,
                "correct": final_pred == ref.label,
            }
        )

    step_df = pd.DataFrame(step_rows)
    step_df["p_correct"] = np.where(step_df["label"] == "A", step_df["p_A"], step_df["p_B"])
    step_df["pred"] = np.where(step_df["p_A"] >= step_df["p_B"], "A", "B")
    step_df["is_correct"] = (step_df["pred"] == step_df["label"]).astype(float)

    csv_path = out_dir / "per_step_predictions.csv"
    step_df.loc[:, ["demo", "round_num", "step", "time_in_round", "p_A", "p_B", "label"]].to_csv(
        csv_path, index=False
    )

    bucket_df = compute_bucket_summary(step_df)
    verdict = choose_verdict(bucket_df)
    final_df = pd.DataFrame(final_rows).sort_values(["demo", "round_num"]).reset_index(drop=True)

    save_overview(round_entries, "A", out_dir / "overview_A.png")
    save_overview(round_entries, "B", out_dir / "overview_B.png")
    build_summary(
        out_path=out_dir / "summary.md",
        args=args,
        data_dir=data_dir,
        notes=notes,
        selected=selected,
        bucket_df=bucket_df,
        final_df=final_df,
        verdict=verdict,
    )

    print(f"[done] wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
