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
SERIES_PREFIXES = {
    "approxq",
    "approx_qlearning",
    "better_approxq",
    "betterapproxq",
    "better_approx_qlearning",
    "approxq_precise",
    "precise_approxq",
    "preciseapproxq",
    "approx_doufeature",
    "approxdou",
    "approxdf",
    "attention_dou",
    "attentiondou",
    "attndou",
    "attn_dou",
}


def numeric_approxq_checkpoints(directory: Path):
    checkpoints = []
    for path in directory.glob("*.pkl"):
        if path.stem.isdigit():
            checkpoints.append((int(path.stem), path))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def checkpoint_metadata(path: Path):
    #TODO: 读取我们自定义 pkl checkpoint 的轻量 metadata，用于按训练时间对齐。
    if path.suffix != ".pkl":
        return {}
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def checkpoint_elapsed_sec(path: Path):
    #TODO: 优先从 checkpoint metadata 中取训练耗时。
    metadata = checkpoint_metadata(path)
    for key in ("elapsed_sec", "train_elapsed_sec", "training_elapsed_sec"):
        value = metadata.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def douzero_frame_from_checkpoint(path: Path):
    #TODO: DouZero checkpoint 文件名中的数字就是训练 frames。
    match = DOUZERO_CHECKPOINT_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def checkpoint_frames(path: Path):
    #TODO: 统一读取真实训练 frames；attention_dou 存在 metadata，DouZero 存在文件名。
    frame = douzero_frame_from_checkpoint(path)
    if frame is not None:
        return float(frame)
    metadata = checkpoint_metadata(path)
    value = metadata.get("frames")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return None


def time_unit_scale(unit: str):
    if unit == "seconds":
        return 1.0
    if unit == "minutes":
        return 60.0
    if unit == "hours":
        return 3600.0
    raise ValueError("Unknown time unit: {}".format(unit))


def douzero_log_time_lookup(root: Path):
    #TODO: 从 DouZero logs.csv 中建立 frame -> elapsed_sec 映射，比文件 mtime 更可靠。
    lookup = {}
    for path in root.rglob("logs.csv"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = []
                for row in reader:
                    if "frames" not in row or "_time" not in row:
                        continue
                    rows.append((int(float(row["frames"])), float(row["_time"])))
        except Exception:
            continue
        if not rows:
            continue
        start_time = rows[0][1]
        for frame, timestamp in rows:
            lookup[frame] = max(0.0, timestamp - start_time)
    return lookup


def nearest_douzero_elapsed_sec(path: Path, lookup):
    #TODO: 用 checkpoint 文件名中的 frame 找最接近的 logs.csv 时间。
    match = DOUZERO_CHECKPOINT_RE.match(path.name)
    if not match or not lookup:
        return None
    frame = int(match.group(1))
    if frame in lookup:
        return lookup[frame]
    nearest = min(lookup, key=lambda value: abs(value - frame))
    return lookup[nearest]


def checkpoint_x_value(episode, path: Path, x_axis: str, time_unit: str,
                       series_start_mtime=None, douzero_time_lookup_data=None):
    #TODO: 根据横轴模式返回 episode、真实 frames 或训练耗时。
    if x_axis == "episode":
        return float(episode)
    if x_axis == "frames":
        frames = checkpoint_frames(path)
        return frames if frames is not None else float(episode)
    elapsed_sec = checkpoint_elapsed_sec(path)
    if elapsed_sec is None:
        elapsed_sec = nearest_douzero_elapsed_sec(path, douzero_time_lookup_data or {})
    if elapsed_sec is None:
        start = series_start_mtime if series_start_mtime is not None else path.stat().st_mtime
        elapsed_sec = max(0.0, path.stat().st_mtime - start)
    return elapsed_sec / time_unit_scale(time_unit)


def douzero_landlord_checkpoints(root: Path, avg_steps_per_episode: float):
    checkpoints = []
    for path in root.rglob("landlord_weights_*.ckpt"):
        step = douzero_frame_from_checkpoint(path)
        if step is None:
            continue
        episode = step / avg_steps_per_episode
        checkpoints.append((episode, path))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def limited_points(points, max_points: int):
    if max_points <= 0 or len(points) <= max_points:
        return points
    return points[:max_points]


def infer_method_prefix(directory: Path):
    #TODO: 自动判断 series 目录属于旧 ApproxQ 还是 DouZero-feature ApproxQ，减少命令出错。
    parts = set(directory.parts)
    if "better_approxq" in parts or directory.name.startswith("better_approxq"):
        return "better_approxq"
    if "approxq_precise" in parts or directory.name.startswith("approxq_precise"):
        return "approxq_precise"
    if "approx_doufeature" in parts or directory.name.startswith("approx_doufeature"):
        return "approx_doufeature"
    if "attention_dou" in parts or directory.name.startswith("attention_dou"):
        return "attention_dou"
    if "approx_qlearning" in parts or directory.name.startswith("approxq"):
        return "approxq"
    for checkpoint in numeric_approxq_checkpoints(directory)[:1]:
        try:
            with open(checkpoint[1], "rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            continue
        algorithm = payload.get("algorithm") if isinstance(payload, dict) else ""
        if algorithm == "approx_doufeature":
            return "approx_doufeature"
        if algorithm == "attention_dou":
            return "attention_dou"
        if algorithm in ("better_approxq", "betterapproxq", "better_approx_qlearning"):
            return "better_approxq"
        if algorithm in ("approxq_precise", "precise_feature_based_approx_qlearning"):
            return "approxq_precise"
        if algorithm in ("approxq", "approx_qlearning", "approx_qlearning"):
            return "approxq"
    return "approxq"


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
        method_prefix = None
        for prefix in SERIES_PREFIXES:
            marker = "{}:".format(prefix)
            if value.startswith(marker):
                if prefix in ("approxdou", "approxdf"):
                    method_prefix = "approx_doufeature"
                elif prefix in ("attentiondou", "attndou", "attn_dou"):
                    method_prefix = "attention_dou"
                else:
                    method_prefix = prefix
                value = value[len(marker):]
                break
        directory = Path(value)
        if method_prefix is None:
            method_prefix = infer_method_prefix(directory)
        parsed.append((match.group("label"), method_prefix, directory))
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
    elif test_role == "two_farmers":
        role_to_method = {
            "landlord": "rlcard",
            "landlord_up": tested_method,
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
    x_axis: str,
    time_unit: str,
    douzero_time_lookup_data=None,
):
    total = len(checkpoints)
    series_start_mtime = min(path.stat().st_mtime for _, path in checkpoints)
    for index, (episode, checkpoint_path) in enumerate(checkpoints, start=1):
        x_value = checkpoint_x_value(
            episode,
            checkpoint_path,
            x_axis,
            time_unit,
            series_start_mtime,
            douzero_time_lookup_data,
        )
        print(
            "[{}/{}] Evaluating {} episode={} x_value={} checkpoint={}".format(
                index, total, series_name, episode, x_value, checkpoint_path
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
                "x_value": x_value,
                "win_rate": win_rate,
            }
        )


def row_key(row):
    return (row["series"], row.get("x_value", row["episode"]))


def write_plot_data(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["series", "episode", "x_value", "win_rate"]
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
                    "x_value": float(row.get("x_value") or row["episode"]),
                    "win_rate": float(row["win_rate"]),
                }
            )
    return rows


def align_reused_douzero_rows(rows, douzero_root: Path, avg_steps_per_episode: float,
                              max_points: int, x_axis: str, time_unit: str,
                              douzero_time_lookup_data):
    #TODO: 复用旧 DouZero 胜率时，用当前横轴设置重算 x_value。
    if not rows:
        return []
    rows = sorted(rows, key=row_key)
    if max_points > 0:
        rows = rows[:max_points]
    points = limited_points(
        douzero_landlord_checkpoints(douzero_root, avg_steps_per_episode),
        max_points,
    )
    if len(points) != len(rows):
        warnings.warn(
            "Cannot rescale reused DouZero rows: checkpoint count {} != row count {}. "
            "Keeping reused episodes unchanged.".format(len(points), len(rows))
        )
        return rows
    aligned = []
    for row, (episode, _) in zip(rows, points):
        path = _
        aligned.append({
            "series": row["series"],
            "episode": episode,
            "x_value": checkpoint_x_value(
                episode,
                path,
                x_axis,
                time_unit,
                None,
                douzero_time_lookup_data,
            ),
            "win_rate": row["win_rate"],
        })
    return aligned


def plot_rows(path: Path, rows, baseline_win_rate: float, test_role: str,
              avg_steps_per_episode: float, x_axis: str, time_unit: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {"baseline_50": "#555555"}
    y_label = "Landlord win rate" if test_role == "landlord" else "Farmer win rate"
    title_role = {
        "landlord": "Landlord",
        "farmer": "Single farmer",
        "two_farmers": "Two farmers",
    }.get(test_role, test_role)

    series_names = [
        name
        for name in dict.fromkeys(row["series"] for row in rows)
        if name != "baseline_50"
    ]
    for index, series_name in enumerate(series_names):
        series_rows = [row for row in rows if row["series"] == series_name]
        if not series_rows:
            continue
        color = colors.get(series_name, color_cycle[index % len(color_cycle)])
        ax.plot(
            [row.get("x_value", row["episode"]) for row in series_rows],
            [row["win_rate"] for row in series_rows],
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            label=series_name,
            color=color,
        )

    model_rows = [row for row in rows if row["series"] != "baseline_50"]
    min_x = min(row.get("x_value", row["episode"]) for row in model_rows)
    max_x = max(row.get("x_value", row["episode"]) for row in model_rows)
    ax.hlines(
        baseline_win_rate,
        min_x,
        max_x,
        colors=colors["baseline_50"],
        linestyles="--",
        linewidth=2,
        label="50% baseline",
    )

    ax.set_title("{} win-rate trend vs 50% baseline".format(title_role))
    if x_axis == "time":
        ax.set_xlabel("Training time ({})".format(time_unit))
    elif x_axis == "frames":
        ax.set_xlabel("Training frames")
    else:
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
            "Use label=approx_doufeature:/path or label=attention_dou:/path to force agent type. "
            "When omitted, --approxq_dir is used as the only series."
        ),
    )
    parser.add_argument("--douzero_root", type=Path, default=DEFAULT_DOUZERO_ROOT)
    parser.add_argument(
        "--skip_douzero",
        action="store_true",
        help="Do not evaluate or plot DouZero checkpoints.",
    )
    parser.add_argument("--eval_data", type=Path, default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--reuse_plot_data",
        type=Path,
        default=None,
        help=(
            "Optional existing CSV produced by this script. "
            "douzero rows will be reused instead of re-evaluated. "
            "Baseline rows are ignored because plots use a fixed 50% baseline."
        ),
    )
    parser.add_argument(
        "--test_role",
        choices=["landlord", "farmer", "two_farmers"],
        default="landlord",
        help=(
            "Choose which side to evaluate as the tested model. "
            "landlord uses landlord win rate; farmer tests one farmer seat; "
            "two_farmers tests both farmer seats with the same method."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument(
        "--num_games",
        type=int,
        default=500,
        help="Number of eval games to use from eval_data; 0 means use all games.",
    )
    parser.add_argument(
        "--avg_steps_per_episode",
        type=float,
        default=60.0,
        help=(
            "DouZero frame to project-episode conversion: episode = frame / "
            "avg_steps_per_episode. Our DouZero script used STEP_MULTIPLIER=60."
        ),
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Optional smoke-test limit per model series; 0 means evaluate all checkpoints.",
    )
    parser.add_argument(
        "--x_axis",
        choices=["episode", "frames", "time"],
        default="episode",
        help="Use episode-equivalent, true training frames, or training-time x-axis.",
    )
    parser.add_argument(
        "--time_unit",
        choices=["seconds", "minutes", "hours"],
        default="hours",
        help="Unit for --x_axis time.",
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
    douzero_time_lookup_data = douzero_log_time_lookup(args.douzero_root)
    reused_douzero_rows = align_reused_douzero_rows(
        [row for row in reused_rows if row["series"] == "douzero"],
        args.douzero_root,
        args.avg_steps_per_episode,
        args.max_points,
        args.x_axis,
        args.time_unit,
        douzero_time_lookup_data,
    )

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
    if args.num_games > 0:
        card_play_data_list = card_play_data_list[:args.num_games]
    print("Using {} eval games from {}".format(len(card_play_data_list), eval_data_path))

    rows = []
    baseline_win_rate = 0.5
    baseline_series_name = "baseline_50"
    print("Using fixed 50% baseline")

    for label, method_prefix, directory in model_series:
        append_series_rows(
            rows,
            label,
            all_model_points[label],
            method_prefix,
            card_play_data_list,
            args.num_workers,
            args.test_role,
            args.x_axis,
            args.time_unit,
        )

    if args.skip_douzero:
        print("Skipping DouZero series")
    elif reused_douzero_rows:
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
            args.x_axis,
            args.time_unit,
            douzero_time_lookup_data,
        )

    min_episode = min(row["episode"] for row in rows)
    max_episode = max(row["episode"] for row in rows)
    min_x = min(row.get("x_value", row["episode"]) for row in rows)
    max_x = max(row.get("x_value", row["episode"]) for row in rows)
    rows.extend(
        [
            {
                "series": baseline_series_name,
                "episode": min_episode,
                "x_value": min_x,
                "win_rate": baseline_win_rate,
            },
            {
                "series": baseline_series_name,
                "episode": max_episode,
                "x_value": max_x,
                "win_rate": baseline_win_rate,
            },
        ]
    )
    rows.sort(key=row_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "{}.csv".format(args.output_prefix)
    png_path = args.output_dir / "{}.png".format(args.output_prefix)
    write_plot_data(csv_path, rows)
    plot_rows(
        png_path,
        rows,
        baseline_win_rate,
        args.test_role,
        args.avg_steps_per_episode,
        args.x_axis,
        args.time_unit,
    )

    print("Saved plot data to {}".format(csv_path))
    print("Saved trend chart to {}".format(png_path))


if __name__ == "__main__":
    main()
