from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

from analyze_approxq_feature_diagnostics import (
    POSITIONS,
    mean,
    pearson,
    performance_by_episode,
    read_diagnostics,
    summarize_features,
)


DEFAULT_OUTPUT_DIR = Path("visualization") / "feature_weight_jitter_correlation"


WEIGHT_RANK_METRICS = (
    "mean_abs_weight",
    "final_abs_weight",
    "weight_std",
    "mean_abs_contribution",
    "impact_score",
)

JITTER_RANK_METRICS = (
    "instability_score",
    "total_abs_delta_w",
    "relative_jitter",
    "direction_changes_per_window",
    "sign_flips",
    "delta_w_std",
    "weight_std",
)

WEIGHT_SERIES = (
    "abs_weight",
    "weight",
    "abs_contribution",
    "signed_contribution",
)

JITTER_SERIES = (
    "abs_delta_w_sum",
    "avg_abs_delta_w",
    "abs_signed_delta_w_sum",
    "signed_delta_w_sum",
    "abs_window_weight_delta",
    "sign_flips",
)


def feature_key(row):
    return (row["position"], row["feature_index"], row["feature"])


def feature_label(row, max_len=46):
    label = "{}:{}:{}".format(row["position"], row["feature_index"], row["feature"])
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def summary_metric(row, metric):
    if metric == "final_abs_weight":
        return abs(float(row.get("final_weight", 0.0)))
    return float(row.get(metric, 0.0))


def row_series_value(row, series_name):
    if series_name == "abs_weight":
        return abs(float(row["weight"]))
    if series_name == "weight":
        return float(row["weight"])
    if series_name == "abs_contribution":
        return abs(float(row["abs_contribution"]))
    if series_name == "signed_contribution":
        return float(row["signed_contribution"])
    if series_name == "abs_delta_w_sum":
        return float(row["abs_delta_w_sum"])
    if series_name == "avg_abs_delta_w":
        return float(row["avg_abs_delta_w"])
    if series_name == "abs_signed_delta_w_sum":
        return abs(float(row["signed_delta_w_sum"]))
    if series_name == "signed_delta_w_sum":
        return float(row["signed_delta_w_sum"])
    if series_name == "abs_window_weight_delta":
        return abs(float(row["window_weight_delta"]))
    if series_name == "sign_flips":
        return float(row["sign_flips"])
    raise ValueError("Unsupported series: {}".format(series_name))


def build_episode_series(rows, value_name):
    grouped = {}
    for row in rows:
        key = feature_key(row)
        grouped.setdefault(key, {}).setdefault(row["episode"], []).append(
            row_series_value(row, value_name)
        )
    return {
        key: {episode: mean(values) for episode, values in episode_values.items()}
        for key, episode_values in grouped.items()
    }


def select_top_features(summaries, metric, topk):
    rows = sorted(
        summaries,
        key=lambda row: (
            summary_metric(row, metric),
            summary_metric(row, "mean_abs_weight"),
            row.get("windows", 0),
        ),
        reverse=True,
    )
    return rows[: max(1, topk)]


def correlation_matrix(weight_rows, jitter_rows, weight_series, jitter_series, min_overlap):
    matrix = []
    overlap_matrix = []
    for weight_row in weight_rows:
        weight_key = feature_key(weight_row)
        weight_values = weight_series.get(weight_key, {})
        matrix_row = []
        overlap_row = []
        for jitter_row in jitter_rows:
            jitter_key = feature_key(jitter_row)
            jitter_values = jitter_series.get(jitter_key, {})
            episodes = sorted(set(weight_values).intersection(jitter_values))
            overlap_row.append(len(episodes))
            if len(episodes) < min_overlap:
                matrix_row.append(None)
                continue
            xs = [weight_values[episode] for episode in episodes]
            ys = [jitter_values[episode] for episode in episodes]
            matrix_row.append(pearson(xs, ys))
        matrix.append(matrix_row)
        overlap_matrix.append(overlap_row)
    return matrix, overlap_matrix


def write_matrix_csv(path, weight_rows, jitter_rows, matrix, overlap_matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [feature_label(row, max_len=80) for row in jitter_rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["weight_feature"] + labels)
        for weight_row, values in zip(weight_rows, matrix):
            writer.writerow([
                feature_label(weight_row, max_len=80),
                *["" if value is None else "{:.8f}".format(value) for value in values],
            ])

    overlap_path = path.with_name(path.stem + "_overlap.csv")
    with overlap_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["weight_feature"] + labels)
        for weight_row, values in zip(weight_rows, overlap_matrix):
            writer.writerow([feature_label(weight_row, max_len=80), *values])


def write_selected_features(path, weight_rows, jitter_rows, weight_metric, jitter_metric):
    fields = (
        "side",
        "rank",
        "position",
        "feature_index",
        "feature",
        "rank_metric",
        "rank_value",
        "windows",
        "final_weight",
        "mean_abs_weight",
        "weight_std",
        "impact_score",
        "instability_score",
        "total_abs_delta_w",
        "relative_jitter",
        "direction_changes_per_window",
        "sign_flips",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for side, rows, metric in (
            ("weight", weight_rows, weight_metric),
            ("jitter", jitter_rows, jitter_metric),
        ):
            for rank, row in enumerate(rows, start=1):
                out = {field: row.get(field, "") for field in fields}
                out.update({
                    "side": side,
                    "rank": rank,
                    "rank_metric": metric,
                    "rank_value": summary_metric(row, metric),
                })
                writer.writerow(out)


def plot_heatmap(
    path,
    weight_rows,
    jitter_rows,
    matrix,
    weight_rank_metric,
    jitter_rank_metric,
    weight_value,
    jitter_value,
    annotate,
    dpi,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.array(
        [[math.nan if value is None else value for value in row] for row in matrix],
        dtype=float,
    )
    masked = np.ma.masked_invalid(values)

    width = max(9.0, 0.58 * len(jitter_rows) + 5.0)
    height = max(6.5, 0.46 * len(weight_rows) + 3.2)
    fig, ax = plt.subplots(figsize=(width, height))

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#E6E6E6")
    image = ax.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(jitter_rows)))
    ax.set_xticklabels(
        [feature_label(row, max_len=34) for row in jitter_rows],
        rotation=48,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(weight_rows)))
    ax.set_yticklabels([feature_label(row, max_len=42) for row in weight_rows], fontsize=8)

    ax.set_xlabel("Top jitter features by {} ({})".format(jitter_rank_metric, jitter_value))
    ax.set_ylabel("Top weight features by {} ({})".format(weight_rank_metric, weight_value))
    ax.set_title(
        "Correlation between feature weight magnitude and jitter\n"
        "cell = Pearson corr(weight-series, jitter-series)"
    )

    ax.set_xticks([x - 0.5 for x in range(1, len(jitter_rows))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(weight_rows))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                value = values[y, x]
                if not math.isfinite(value):
                    continue
                color = "white" if abs(value) >= 0.55 else "black"
                ax.text(x, y, "{:.2f}".format(value), ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("Pearson correlation")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a compact correlation heatmap between top weight features "
            "and top jitter features from ApproxQ feature_diagnostics.csv."
        )
    )
    parser.add_argument("--diag_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--top_weight", type=int, default=16)
    parser.add_argument("--top_jitter", type=int, default=16)
    parser.add_argument("--min_windows", type=int, default=3)
    parser.add_argument("--min_overlap", type=int, default=3)
    parser.add_argument("--position", choices=("all",) + POSITIONS, default="all")
    parser.add_argument("--weight_rank_metric", choices=WEIGHT_RANK_METRICS, default="mean_abs_weight")
    parser.add_argument("--jitter_rank_metric", choices=JITTER_RANK_METRICS, default="instability_score")
    parser.add_argument("--weight_value", choices=WEIGHT_SERIES, default="abs_weight")
    parser.add_argument("--jitter_value", choices=JITTER_SERIES, default="abs_delta_w_sum")
    parser.add_argument("--no_annotate", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    if not args.diag_csv.exists():
        raise FileNotFoundError(args.diag_csv)

    prefix = args.prefix or args.diag_csv.parent.name
    rows = read_diagnostics(args.diag_csv, args.position)
    performance, _ = performance_by_episode(rows, [])
    summaries = summarize_features(rows, performance, args.min_windows)

    weight_rows = select_top_features(summaries, args.weight_rank_metric, args.top_weight)
    jitter_rows = select_top_features(summaries, args.jitter_rank_metric, args.top_jitter)

    weight_series = build_episode_series(rows, args.weight_value)
    jitter_series = build_episode_series(rows, args.jitter_value)
    matrix, overlap_matrix = correlation_matrix(
        weight_rows,
        jitter_rows,
        weight_series,
        jitter_series,
        args.min_overlap,
    )

    output_prefix = args.output_dir / prefix
    png_path = output_prefix.with_name(output_prefix.name + "_weight_jitter_corr_heatmap.png")
    matrix_csv = output_prefix.with_name(output_prefix.name + "_weight_jitter_corr_matrix.csv")
    selected_csv = output_prefix.with_name(output_prefix.name + "_selected_features.csv")

    write_matrix_csv(matrix_csv, weight_rows, jitter_rows, matrix, overlap_matrix)
    write_selected_features(
        selected_csv,
        weight_rows,
        jitter_rows,
        args.weight_rank_metric,
        args.jitter_rank_metric,
    )
    plot_heatmap(
        png_path,
        weight_rows,
        jitter_rows,
        matrix,
        args.weight_rank_metric,
        args.jitter_rank_metric,
        args.weight_value,
        args.jitter_value,
        not args.no_annotate and args.top_weight * args.top_jitter <= 400,
        args.dpi,
    )

    print("Read diagnostics: {}".format(args.diag_csv))
    print("Selected top weight features: {}".format(len(weight_rows)))
    print("Selected top jitter features: {}".format(len(jitter_rows)))
    print("Wrote heatmap: {}".format(png_path))
    print("Wrote matrix CSV: {}".format(matrix_csv))
    print("Wrote selected features CSV: {}".format(selected_csv))


if __name__ == "__main__":
    main()
