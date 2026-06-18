from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
import warnings
from pathlib import Path

from config import REPO_ROOT, with_prefix
from douzero.evaluation.rlcard_data import ensure_rlcard_doudizhu_jsondata
from douzero.evaluation.simulation import resolve_eval_data_path, simulate_one_assignment


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch.load` with `weights_only=False`.*",
)

DEFAULT_APPROXQ_DIR = (
    REPO_ROOT
    / "approx_qlearning_checkpoints"
    / "approx_qlearning"
    / "approxq_logadp_cmp_1m_history"
)
DEFAULT_DOUZERO_ROOT = REPO_ROOT / "base" / "douzero_checkpoints"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualization"
DEFAULT_EVAL_DATA = REPO_ROOT / "eval_data.pkl"
DEFAULT_OUTPUT_PREFIX = "approxq_douzero_vs_rlcard_fixed_trend"
DOUZERO_CHECKPOINT_RE = re.compile(r"landlord_weights_(\d+)\.ckpt$")
SERIES_ARG_RE = re.compile(r"^(?P<label>[^=]+)=(?P<path>.+)$")
SERIES_PREFIXES = {"approxq", "approx_doufeature", "approxdou", "approxdf"}


def numeric_approxq_checkpoints(directory: Path):
    checkpoints = []
    for path in directory.glob("*.pkl"):
        if path.stem.isdigit():
            checkpoints.append((int(path.stem), path))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def douzero_landlord_checkpoints(root: Path, avg_steps_per_episode: float):
    checkpoints = []
    for path in root.rglob("landlord_weights_*.ckpt"):
        match = DOUZERO_CHECKPOINT_RE.match(path.name)
        if not match:
            continue
        step = int(match.group(1))
        episode = step / avg_steps_per_episode
        checkpoints.append((episode, path))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def limited_points(points, max_points: int):
    if max_points <= 0 or len(points) <= max_points:
        return points
    return points[:max_points]


def parse_series_args(series_args, default_label, default_path):
    if not series_args:
        return [(default_label, "approxq", Path(default_path))]
    parsed = []
    for item in series_args:
        match = SERIES_ARG_RE.match(item)
        if not match:
            raise ValueError(
                "Invalid series spec '{}'. Use label=/path or "
                "label=agent_prefix:/path".format(item)
            )
        value = match.group("path")
        method_prefix = "approxq"
        for prefix in SERIES_PREFIXES:
            marker = "{}:".format(prefix)
            if value.startswith(marker):
                method_prefix = prefix
                value = value[len(marker):]
                break
        parsed.append((match.group("label"), method_prefix, Path(value)))
    return parsed


def evaluate_role_win_rate(
    card_play_data_list,
    test_role: str,
    tested_method: str,
    num_workers: int,
):
    if test_role == "landlord":
        role_to_method = {
            "landlord": tested_method,
            "landlord_up": "rlcard",
            "landlord_down": "rlcard",
        }
        metric = "landlord_win_rate"
    elif test_role == "farmer":
        role_to_method = {
            "landlord": "rlcard",
            "landlord_up": "rlcard",
            "landlord_down": tested_method,
        }
        metric = "farmer_win_rate"
    else:
        raise ValueError("Unknown test_role: {}".format(test_role))

    result = simulate_one_assignment(
        card_play_data_list,
        role_to_method,
        num_workers,
        show_progress=False,
    )
    return result[metric]


def append_series_rows(
    rows,
    series_name: str,
    checkpoints,
    method_prefix: str,
    card_play_data_list,
    num_workers,
    test_role: str,
):
    total = len(checkpoints)
    for index, (episode, checkpoint_path) in enumerate(checkpoints, start=1):
        print(
            "[{}/{}] Evaluating {} episode={} checkpoint={}".format(
                index, total, series_name, episode, checkpoint_path
            )
        )
        win_rate = evaluate_role_win_rate(
            card_play_data_list,
            test_role,
            with_prefix(method_prefix, checkpoint_path),
            num_workers,
        )
        rows.append(
            {
                "series": series_name,
                "episode": episode,
                "win_rate": win_rate,
            }
        )


def row_key(row):
    return (row["series"], row["episode"])


def write_plot_data(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["series", "episode", "win_rate"]
        )
        writer.writeheader()
        writer.writerows(rows)


def read_plot_data(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "series": row["series"],
                    "episode": float(row["episode"]),
                    "win_rate": float(row["win_rate"]),
                }
            )
    return rows


def plot_rows(path: Path, rows, baseline_win_rate: float, test_role: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {"rlcard_baseline": "#555555"}
    y_label = "Landlord win rate" if test_role == "landlord" else "Farmer win rate"
    title_role = "Landlord" if test_role == "landlord" else "Farmer"

    series_names = [
        name
        for name in dict.fromkeys(row["series"] for row in rows)
        if name != "rlcard_baseline"
    ]
    for index, series_name in enumerate(series_names):
        series_rows = [row for row in rows if row["series"] == series_name]
        if not series_rows:
            continue
        color = colors.get(series_name, color_cycle[index % len(color_cycle)])
        ax.plot(
            [row["episode"] for row in series_rows],
            [row["win_rate"] for row in series_rows],
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            label=series_name,
            color=color,
        )

    model_rows = [row for row in rows if row["series"] != "rlcard_baseline"]
    min_episode = min(row["episode"] for row in model_rows)
    max_episode = max(row["episode"] for row in model_rows)
    ax.hlines(
        baseline_win_rate,
        min_episode,
        max_episode,
        colors=colors["rlcard_baseline"],
        linestyles="--",
        linewidth=2,
        label="rlcard baseline",
    )

    ax.set_title("{} win-rate trend vs RLCard baseline".format(title_role))
    ax.set_xlabel("Training episodes")
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved ApproxQ/DouZero checkpoints as fixed landlord vs two "
            "RLCard farmers, then save minimal plot data and a win-rate trend chart."
        )
    )
    parser.add_argument("--approxq_dir", type=Path, default=DEFAULT_APPROXQ_DIR)
    parser.add_argument(
        "--approxq_series",
        action="append",
        default=[],
        help=(
            "Optional extra ApproxQ series in label=/path form. "
            "Repeatable; when omitted, --approxq_dir is used as the only series."
        ),
    )
    parser.add_argument("--douzero_root", type=Path, default=DEFAULT_DOUZERO_ROOT)
    parser.add_argument("--eval_data", type=Path, default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--reuse_plot_data",
        type=Path,
        default=None,
        help=(
            "Optional existing CSV produced by this script. "
            "douzero and rlcard_baseline rows will be reused instead of re-evaluated."
        ),
    )
    parser.add_argument(
        "--test_role",
        choices=["landlord", "farmer"],
        default="landlord",
        help=(
            "Choose which side to evaluate as the tested model. "
            "landlord uses landlord win rate; farmer uses farmer win rate."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument(
        "--avg_steps_per_episode",
        type=float,
        default=40.0,
        help="DouZero step to episode conversion: episode = step / avg_steps_per_episode",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Optional smoke-test limit per model series; 0 means evaluate all checkpoints.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    ensure_rlcard_doudizhu_jsondata()

    if args.output_prefix == DEFAULT_OUTPUT_PREFIX:
        args.output_prefix = "{}_{}".format(DEFAULT_OUTPUT_PREFIX, args.test_role)

    model_series = parse_series_args(
        args.approxq_series, "approxq", args.approxq_dir
    )

    reused_rows = read_plot_data(args.reuse_plot_data) if args.reuse_plot_data else []
    reused_douzero_rows = [row for row in reused_rows if row["series"] == "douzero"]
    reused_baseline_rows = [
        row for row in reused_rows if row["series"] == "rlcard_baseline"
    ]

    all_model_points = {}
    for label, _, directory in model_series:
        points = numeric_approxq_checkpoints(directory)
        points = limited_points(points, args.max_points)
        if not points:
            raise FileNotFoundError(
                "No numeric model .pkl checkpoints found under {}".format(
                    directory
                )
            )
        all_model_points[label] = points

    eval_data_path = resolve_eval_data_path(str(args.eval_data))
    with open(eval_data_path, "rb") as f:
        card_play_data_list = pickle.load(f)

    rows = []
    if reused_baseline_rows:
        baseline_win_rate = reused_baseline_rows[0]["win_rate"]
        print(
            "Reusing RLCard baseline for {} from {}".format(
                args.test_role, args.reuse_plot_data
            )
        )
    else:
        print("Evaluating RLCard baseline for {}".format(args.test_role))
        baseline_win_rate = evaluate_role_win_rate(
            card_play_data_list,
            args.test_role,
            "rlcard",
            args.num_workers,
        )

    baseline_series_name = "rlcard_baseline"

    for label, method_prefix, directory in model_series:
        append_series_rows(
            rows,
            label,
            all_model_points[label],
            method_prefix,
            card_play_data_list,
            args.num_workers,
            args.test_role,
        )

    if reused_douzero_rows:
        print(
            "Reusing {} DouZero rows from {}".format(
                len(reused_douzero_rows), args.reuse_plot_data
            )
        )
        rows.extend(reused_douzero_rows)
    else:
        douzero_points = douzero_landlord_checkpoints(
            args.douzero_root, args.avg_steps_per_episode
        )
        douzero_points = limited_points(douzero_points, args.max_points)
        if not douzero_points:
            raise FileNotFoundError(
                "No landlord_weights_*.ckpt checkpoints found under {}".format(
                    args.douzero_root
                )
            )
        append_series_rows(
            rows,
            "douzero",
            douzero_points,
            "douzero",
            card_play_data_list,
            args.num_workers,
            args.test_role,
        )

    min_episode = min(row["episode"] for row in rows)
    max_episode = max(row["episode"] for row in rows)
    rows.extend(
        [
            {
                "series": baseline_series_name,
                "episode": min_episode,
                "win_rate": baseline_win_rate,
            },
            {
                "series": baseline_series_name,
                "episode": max_episode,
                "win_rate": baseline_win_rate,
            },
        ]
    )
    rows.sort(key=row_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "{}.csv".format(args.output_prefix)
    png_path = args.output_dir / "{}.png".format(args.output_prefix)
    write_plot_data(csv_path, rows)
    plot_rows(png_path, rows, baseline_win_rate, args.test_role)

    print("Saved plot data to {}".format(csv_path))
    print("Saved trend chart to {}".format(png_path))


if __name__ == "__main__":
    main()
