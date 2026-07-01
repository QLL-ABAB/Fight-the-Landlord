from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

try:
    from config import REPO_ROOT
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[1]


POSITIONS = ("landlord", "landlord_up", "landlord_down")
DEFAULT_DIAG_CSV = (
    REPO_ROOT
    / "approx_qlearning_checkpoints"
    / "approx_qlearning"
    / "approxq_logadp_cmp_1m_history"
    / "feature_diagnostics.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualization" / "approxq_feature_diagnostics"


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values):
    values = list(values)
    return sum(values) / float(len(values)) if values else 0.0


def stdev(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def pearson(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return 0.0
    xs, ys = zip(*pairs)
    avg_x = mean(xs)
    avg_y = mean(ys)
    var_x = sum((x - avg_x) ** 2 for x in xs)
    var_y = sum((y - avg_y) ** 2 for y in ys)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return 0.0
    cov = sum((x - avg_x) * (y - avg_y) for x, y in pairs)
    return cov / math.sqrt(var_x * var_y)


def sign(value, eps=1e-12):
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def sign_changes(values):
    previous = 0
    changes = 0
    for value in values:
        current = sign(value)
        if current == 0:
            continue
        if previous and current != previous:
            changes += 1
        previous = current
    return changes


def normalize_summaries(summaries, key, output_key):
    values = [summary.get(key, 0.0) for summary in summaries]
    low = min(values) if values else 0.0
    high = max(values) if values else 0.0
    span = high - low
    for summary in summaries:
        if span <= 1e-12:
            summary[output_key] = 0.0
        else:
            summary[output_key] = (summary.get(key, 0.0) - low) / span


def read_diagnostics(path: Path, position: str):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if position != "all" and row.get("position") != position:
                continue
            parsed = {
                "episode": safe_int(row.get("episode")),
                "window_start_episode": safe_int(row.get("window_start_episode")),
                "window_end_episode": safe_int(row.get("window_end_episode")),
                "elapsed_sec": safe_float(row.get("elapsed_sec")),
                "total_steps": safe_int(row.get("total_steps")),
                "epsilon": safe_float(row.get("epsilon")),
                "landlord_wp": safe_float(row.get("landlord_wp")),
                "avg_steps": safe_float(row.get("avg_steps")),
                "avg_abs_td": safe_float(row.get("avg_abs_td")),
                "position": row.get("position", ""),
                "feature_index": safe_int(row.get("feature_index")),
                "feature": row.get("feature", ""),
                "updates": safe_int(row.get("updates")),
                "weight": safe_float(row.get("weight")),
                "window_weight_delta": safe_float(row.get("window_weight_delta")),
                "abs_delta_w_sum": safe_float(row.get("abs_delta_w_sum")),
                "signed_delta_w_sum": safe_float(row.get("signed_delta_w_sum")),
                "avg_abs_delta_w": safe_float(row.get("avg_abs_delta_w")),
                "mean_feature": safe_float(row.get("mean_feature")),
                "mean_abs_feature": safe_float(row.get("mean_abs_feature")),
                "activation_rate": safe_float(row.get("activation_rate")),
                "avg_abs_td_x_feature": safe_float(row.get("avg_abs_td_x_feature")),
                "avg_signed_td_x_feature": safe_float(row.get("avg_signed_td_x_feature")),
                "abs_contribution": safe_float(row.get("abs_contribution")),
                "signed_contribution": safe_float(row.get("signed_contribution")),
                "sign_flips": safe_int(row.get("sign_flips")),
                "avg_abs_raw_td": safe_float(row.get("avg_abs_raw_td")),
                "avg_abs_clipped_td": safe_float(row.get("avg_abs_clipped_td")),
            }
            if parsed["episode"] > 0 and parsed["feature"]:
                rows.append(parsed)
    if not rows:
        raise ValueError("No diagnostics rows found in {}".format(path))
    return rows


def read_eval_csv(path: Path, series_name: str):
    rows = []
    if path is None:
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if series_name and row.get("series") != series_name:
                continue
            if "episode" not in row or "win_rate" not in row:
                continue
            rows.append((
                safe_float(row.get("episode")),
                safe_float(row.get("win_rate")),
            ))
    rows = [(episode, value) for episode, value in rows if episode > 0]
    rows.sort()
    return rows


def nearest_eval_value(eval_rows, episode):
    if not eval_rows:
        return None
    best_episode, best_value = min(
        eval_rows,
        key=lambda item: abs(float(item[0]) - float(episode)),
    )
    return best_value


def performance_by_episode(rows, eval_rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode"]].append(row["landlord_wp"])
    performance = {}
    source = "train_landlord_wp"
    for episode, values in grouped.items():
        eval_value = nearest_eval_value(eval_rows, episode) if eval_rows else None
        if eval_value is None:
            performance[episode] = mean(values)
        else:
            source = "eval_win_rate"
            performance[episode] = eval_value
    return performance, source


def performance_deltas(performance):
    deltas = {}
    previous_episode = None
    for episode in sorted(performance):
        if previous_episode is None:
            deltas[episode] = 0.0
        else:
            deltas[episode] = performance[episode] - performance[previous_episode]
        previous_episode = episode
    return deltas


def summarize_features(rows, performance, min_windows):
    deltas = performance_deltas(performance)
    unique_episodes = sorted({row["episode"] for row in rows})
    total_windows = len(unique_episodes)
    grouped = defaultdict(list)
    for row in rows:
        key = (row["position"], row["feature_index"], row["feature"])
        grouped[key].append(row)

    summaries = []
    for (position, feature_index, feature), feature_rows in grouped.items():
        feature_rows.sort(key=lambda row: row["episode"])
        if len(feature_rows) < min_windows:
            continue

        weights = [row["weight"] for row in feature_rows]
        signed_delta = [row["signed_delta_w_sum"] for row in feature_rows]
        abs_delta = [row["abs_delta_w_sum"] for row in feature_rows]
        signals = [
            row["avg_abs_td_x_feature"] + row["abs_contribution"] + row["avg_abs_delta_w"]
            for row in feature_rows
        ]
        perf_values = [performance.get(row["episode"], 0.0) for row in feature_rows]
        perf_delta_values = [deltas.get(row["episode"], 0.0) for row in feature_rows]
        positive_improvement_signal = sum(
            max(0.0, delta) * signal
            for delta, signal in zip(perf_delta_values, signals)
        )
        negative_improvement_signal = sum(
            max(0.0, -delta) * signal
            for delta, signal in zip(perf_delta_values, signals)
        )
        total_abs_delta = sum(abs_delta)
        net_delta = sum(signed_delta)
        jitter = max(0.0, total_abs_delta - abs(net_delta))
        relative_jitter = jitter / total_abs_delta if total_abs_delta > 1e-12 else 0.0
        direction_changes = sign_changes(signed_delta)

        summaries.append({
            "position": position,
            "feature_index": feature_index,
            "feature": feature,
            "windows": len(feature_rows),
            "coverage_rate": len(feature_rows) / float(max(1, total_windows)),
            "first_episode": feature_rows[0]["episode"],
            "last_episode": feature_rows[-1]["episode"],
            "final_weight": weights[-1],
            "mean_abs_weight": mean(abs(value) for value in weights),
            "weight_std": stdev(weights),
            "total_abs_delta_w": total_abs_delta,
            "net_signed_delta_w": net_delta,
            "relative_jitter": relative_jitter,
            "direction_changes": direction_changes,
            "direction_changes_per_window": direction_changes / float(max(1, len(feature_rows) - 1)),
            "sign_flips": sum(row["sign_flips"] for row in feature_rows),
            "mean_abs_delta_w": mean(row["avg_abs_delta_w"] for row in feature_rows),
            "delta_w_std": stdev(row["signed_delta_w_sum"] for row in feature_rows),
            "mean_abs_td_x_feature": mean(row["avg_abs_td_x_feature"] for row in feature_rows),
            "mean_signed_td_x_feature": mean(row["avg_signed_td_x_feature"] for row in feature_rows),
            "mean_abs_contribution": mean(row["abs_contribution"] for row in feature_rows),
            "final_abs_contribution": feature_rows[-1]["abs_contribution"],
            "mean_signed_contribution": mean(row["signed_contribution"] for row in feature_rows),
            "mean_activation_rate": mean(row["activation_rate"] for row in feature_rows),
            "mean_abs_feature": mean(row["mean_abs_feature"] for row in feature_rows),
            "mean_abs_raw_td": mean(row["avg_abs_raw_td"] for row in feature_rows),
            "mean_abs_clipped_td": mean(row["avg_abs_clipped_td"] for row in feature_rows),
            "performance_corr": pearson(signals, perf_values),
            "improvement_corr": pearson(signals, perf_delta_values),
            "positive_improvement_signal": positive_improvement_signal,
            "negative_improvement_signal": negative_improvement_signal,
        })

    if not summaries:
        raise ValueError("No features had at least {} windows".format(min_windows))

    for key in (
        "mean_abs_contribution",
        "final_abs_contribution",
        "mean_abs_td_x_feature",
        "positive_improvement_signal",
        "total_abs_delta_w",
        "relative_jitter",
        "direction_changes_per_window",
        "sign_flips",
        "delta_w_std",
        "weight_std",
    ):
        normalize_summaries(summaries, key, "norm_" + key)

    for summary in summaries:
        positive_corr = max(0.0, summary["improvement_corr"])
        summary["impact_score"] = (
            0.30 * summary["norm_mean_abs_contribution"]
            + 0.20 * summary["norm_final_abs_contribution"]
            + 0.20 * summary["norm_mean_abs_td_x_feature"]
            + 0.20 * summary["norm_positive_improvement_signal"]
            + 0.10 * positive_corr
        )
        summary["instability_score"] = (
            0.25 * summary["norm_relative_jitter"]
            + 0.20 * summary["norm_direction_changes_per_window"]
            + 0.15 * summary["norm_sign_flips"]
            + 0.15 * summary["norm_delta_w_std"]
            + 0.15 * summary["norm_weight_std"]
            + 0.10 * summary["norm_total_abs_delta_w"]
        )
    return summaries


SUMMARY_FIELDS = (
    "position",
    "feature_index",
    "feature",
    "impact_score",
    "instability_score",
    "windows",
    "coverage_rate",
    "first_episode",
    "last_episode",
    "final_weight",
    "mean_abs_weight",
    "weight_std",
    "total_abs_delta_w",
    "net_signed_delta_w",
    "relative_jitter",
    "direction_changes",
    "direction_changes_per_window",
    "sign_flips",
    "mean_abs_delta_w",
    "delta_w_std",
    "mean_abs_td_x_feature",
    "mean_signed_td_x_feature",
    "mean_abs_contribution",
    "final_abs_contribution",
    "mean_signed_contribution",
    "mean_activation_rate",
    "mean_abs_feature",
    "mean_abs_raw_td",
    "mean_abs_clipped_td",
    "performance_corr",
    "improvement_corr",
    "positive_improvement_signal",
    "negative_improvement_signal",
)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def feature_label(row, max_len=42):
    label = "{}:{}".format(row["position"], row["feature"])
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def plot_bar(path: Path, rows, score_key: str, title: str, topk: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(rows[:topk])
    rows.reverse()
    fig_height = max(5.0, 0.36 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    labels = [feature_label(row) for row in rows]
    values = [row[score_key] for row in rows]
    colors = ["#2E86AB" if score_key == "impact_score" else "#C05746" for _ in rows]
    ax.barh(labels, values, color=colors, alpha=0.9)
    ax.set_xlabel(score_key)
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_training_signal(path: Path, rows, performance, source):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = sorted(performance)
    td_by_episode = defaultdict(list)
    for row in rows:
        td_by_episode[row["episode"]].append(row["avg_abs_td"])
    td_values = [mean(td_by_episode[episode]) for episode in episodes]
    perf_values = [performance[episode] for episode in episodes]

    fig, ax1 = plt.subplots(figsize=(11, 5.8))
    ax1.plot(episodes, perf_values, marker="o", color="#2E86AB", label=source)
    ax1.set_xlabel("episode")
    ax1.set_ylabel(source, color="#2E86AB")
    ax1.tick_params(axis="y", labelcolor="#2E86AB")
    ax1.grid(True, linestyle=":", alpha=0.35)
    ax2 = ax1.twinx()
    ax2.plot(episodes, td_values, marker=".", color="#C05746", label="avg_abs_td")
    ax2.set_ylabel("avg_abs_td", color="#C05746")
    ax2.tick_params(axis="y", labelcolor="#C05746")
    ax1.set_title("Training signal used by feature analysis")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_feature_lines(path: Path, rows, selected, value_key, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_keys = {
        (row["position"], row["feature_index"], row["feature"])
        for row in selected
    }
    grouped = defaultdict(list)
    for row in rows:
        key = (row["position"], row["feature_index"], row["feature"])
        if key in selected_keys:
            grouped[key].append(row)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for key, values in grouped.items():
        values.sort(key=lambda item: item["episode"])
        label = "{}:{}".format(key[0], key[2])
        if len(label) > 34:
            label = label[:31] + "..."
        ax.plot(
            [row["episode"] for row in values],
            [row[value_key] for row in values],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=label,
        )
    ax.set_xlabel("episode")
    ax.set_ylabel(value_key)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_markdown_summary(path: Path, impact_rows, unstable_rows, performance_source):
    lines = [
        "# ApproxQ feature diagnostics summary",
        "",
        "Performance source: `{}`".format(performance_source),
        "",
        "## Most impact-aligned features",
        "",
        "| rank | position | feature | impact_score | improvement_corr | mean_abs_contribution | mean_abs_td_x_feature |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(impact_rows[:10], start=1):
        lines.append(
            "| {} | {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                rank,
                row["position"],
                row["feature"],
                row["impact_score"],
                row["improvement_corr"],
                row["mean_abs_contribution"],
                row["mean_abs_td_x_feature"],
            )
        )
    lines.extend([
        "",
        "## Most unstable features",
        "",
        "| rank | position | feature | instability_score | relative_jitter | direction_changes | sign_flips | total_abs_delta_w |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for rank, row in enumerate(unstable_rows[:10], start=1):
        lines.append(
            "| {} | {} | {} | {:.6f} | {:.6f} | {} | {} | {:.6f} |".format(
                rank,
                row["position"],
                row["feature"],
                row["instability_score"],
                row["relative_jitter"],
                row["direction_changes"],
                row["sign_flips"],
                row["total_abs_delta_w"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ApproxQ per-feature diagnostics and rank features by "
            "impact-aligned signal and instability."
        )
    )
    parser.add_argument("--diag_csv", type=Path, default=DEFAULT_DIAG_CSV)
    parser.add_argument(
        "--eval_csv",
        type=Path,
        default=None,
        help="Optional plot_eval_trends CSV with episode/win_rate columns.",
    )
    parser.add_argument(
        "--eval_series",
        type=str,
        default="",
        help="Optional series name to select from --eval_csv.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--timeline_topk", type=int, default=6)
    parser.add_argument("--min_windows", type=int, default=2)
    parser.add_argument(
        "--position",
        choices=("all",) + POSITIONS,
        default="all",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    if not args.diag_csv.exists():
        raise FileNotFoundError(args.diag_csv)
    prefix = args.prefix or args.diag_csv.parent.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_diagnostics(args.diag_csv, args.position)
    eval_rows = read_eval_csv(args.eval_csv, args.eval_series) if args.eval_csv else []
    performance, performance_source = performance_by_episode(rows, eval_rows)
    summaries = summarize_features(rows, performance, args.min_windows)

    impact_rows = sorted(
        summaries,
        key=lambda row: (
            row["impact_score"],
            row["mean_abs_contribution"],
            row["mean_abs_td_x_feature"],
        ),
        reverse=True,
    )
    unstable_rows = sorted(
        summaries,
        key=lambda row: (
            row["instability_score"],
            row["relative_jitter"],
            row["direction_changes"],
        ),
        reverse=True,
    )

    importance_csv = args.output_dir / "{}_feature_importance.csv".format(prefix)
    instability_csv = args.output_dir / "{}_feature_instability.csv".format(prefix)
    write_csv(importance_csv, impact_rows)
    write_csv(instability_csv, unstable_rows)
    write_markdown_summary(
        args.output_dir / "{}_summary.md".format(prefix),
        impact_rows,
        unstable_rows,
        performance_source,
    )

    plot_training_signal(
        args.output_dir / "{}_training_signal.png".format(prefix),
        rows,
        performance,
        performance_source,
    )
    plot_bar(
        args.output_dir / "{}_impact_top{}.png".format(prefix, args.topk),
        impact_rows,
        "impact_score",
        "Features most aligned with stronger training signal",
        args.topk,
    )
    plot_bar(
        args.output_dir / "{}_unstable_top{}.png".format(prefix, args.topk),
        unstable_rows,
        "instability_score",
        "Features with repeated jitter or sign changes",
        args.topk,
    )
    plot_feature_lines(
        args.output_dir / "{}_impact_timeline.png".format(prefix),
        rows,
        impact_rows[: args.timeline_topk],
        "abs_contribution",
        "Top impact feature contribution over training",
    )
    plot_feature_lines(
        args.output_dir / "{}_unstable_delta_timeline.png".format(prefix),
        rows,
        unstable_rows[: args.timeline_topk],
        "signed_delta_w_sum",
        "Top unstable feature signed weight update over training",
    )

    print("Read diagnostics: {}".format(args.diag_csv))
    print("Performance source: {}".format(performance_source))
    print("Wrote importance ranking: {}".format(importance_csv))
    print("Wrote instability ranking: {}".format(instability_csv))
    print("Wrote plots under: {}".format(args.output_dir))


if __name__ == "__main__":
    main()
