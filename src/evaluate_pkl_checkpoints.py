import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def checkpoint_episode(path):
    stem = Path(path).stem
    if stem.isdigit():
        return int(stem)
    match = re.search(r"(\d+)", stem)
    return int(match.group(1)) if match else None


def find_checkpoints(checkpoint_dir):
    checkpoints = []
    for path in Path(checkpoint_dir).glob("*.pkl"):
        episode = checkpoint_episode(path)
        if episode is not None:
            checkpoints.append((episode, path))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def run_evaluation(method, checkpoint_path, result_dir, eval_data, num_workers,
                   assignment_workers, evaluate_name, skip_existing):
    result_path = Path(result_dir) / "{}.json".format(evaluate_name)
    if skip_existing and result_path.exists():
        return result_path

    agent_method = "{}:{}".format(method, checkpoint_path.as_posix())
    cmd = [
        sys.executable,
        "src/evaluate.py",
        "--methods",
        agent_method,
        "rlcard",
        "rlcard",
        "--eval_mode",
        "rotate",
        "--evaluate_name",
        evaluate_name,
        "--result_dir",
        result_dir,
        "--eval_data",
        eval_data,
        "--num_workers",
        str(num_workers),
        "--assignment_workers",
        str(assignment_workers),
    ]
    print("Evaluating {}...".format(agent_method), flush=True)
    completed = subprocess.run(cmd, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Evaluation failed for {} with exit code {}".format(
                checkpoint_path,
                completed.returncode,
            )
        )
    if not result_path.exists():
        raise FileNotFoundError("Expected result JSON not found: {}".format(result_path))
    return result_path


def parse_result(result_path, method, checkpoint_path):
    agent_method = "{}:{}".format(method, checkpoint_path.as_posix())
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary", {})
    stats = summary.get(agent_method)
    if stats is None:
        raise KeyError("No summary entry for {} in {}".format(agent_method, result_path))
    return {
        "games": stats.get("games", 0),
        "overall_win_rate": stats.get("overall_win_rate"),
        "landlord_win_rate": stats.get("landlord_win_rate"),
        "farmer_win_rate": stats.get("farmer_win_rate"),
    }


def write_csv(rows, csv_path):
    fieldnames = [
        "episodes",
        "overall_win_rate",
        "landlord_win_rate",
        "farmer_win_rate",
        "games",
        "checkpoint",
        "result_json",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_curve(rows, output_path, title):
    episodes = [int(row["episodes"]) for row in rows]
    overall = [float(row["overall_win_rate"]) for row in rows]
    landlord = [float(row["landlord_win_rate"]) for row in rows]
    farmer = [float(row["farmer_win_rate"]) for row in rows]
    width = 960
    height = 560
    left = 80
    right = 30
    top = 55
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_ep = min(episodes)
    max_ep = max(episodes)

    def x_scale(ep):
        if max_ep == min_ep:
            return left + plot_w / 2.0
        return left + (ep - min_ep) * plot_w / float(max_ep - min_ep)

    def y_scale(value):
        return top + (1.0 - value) * plot_h

    def polyline(values, color):
        points = " ".join(
            "{:.2f},{:.2f}".format(x_scale(ep), y_scale(value))
            for ep, value in zip(episodes, values)
        )
        circles = "\n".join(
            '<circle cx="{:.2f}" cy="{:.2f}" r="3.5" fill="{}" />'.format(
                x_scale(ep), y_scale(value), color
            )
            for ep, value in zip(episodes, values)
        )
        return (
            '<polyline fill="none" stroke="{}" stroke-width="2.5" '
            'points="{}" />\n{}'
        ).format(color, points, circles)

    grid_lines = []
    for tick in range(0, 11):
        value = tick / 10.0
        y = y_scale(value)
        grid_lines.append(
            '<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" '
            'stroke="#ddd" stroke-width="1" />'.format(left, y, width - right, y)
        )
        grid_lines.append(
            '<text x="{}" y="{:.2f}" text-anchor="end" '
            'font-size="12">{:.1f}</text>'.format(left - 8, y + 4, value)
        )

    x_ticks = []
    for ep in episodes:
        x = x_scale(ep)
        x_ticks.append(
            '<line x1="{:.2f}" y1="{}" x2="{:.2f}" y2="{}" '
            'stroke="#bbb" stroke-width="1" />'.format(
                x, height - bottom, x, height - bottom + 5
            )
        )
        x_ticks.append(
            '<text x="{:.2f}" y="{}" text-anchor="middle" '
            'font-size="11">{}</text>'.format(x, height - bottom + 22, ep)
        )

    legend = (
        '<rect x="720" y="72" width="185" height="78" fill="white" '
        'stroke="#ccc" />'
        '<line x1="740" y1="94" x2="775" y2="94" stroke="#1f77b4" '
        'stroke-width="3" /><text x="785" y="98" font-size="13">Overall</text>'
        '<line x1="740" y1="118" x2="775" y2="118" stroke="#ff7f0e" '
        'stroke-width="3" /><text x="785" y="122" font-size="13">Landlord</text>'
        '<line x1="740" y1="142" x2="775" y2="142" stroke="#2ca02c" '
        'stroke-width="3" /><text x="785" y="146" font-size="13">Farmer</text>'
    )

    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{cx}" y="30" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>
{grid}
<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom_y}" stroke="#333" stroke-width="1.5" />
<line x1="{left}" y1="{bottom_y}" x2="{right_x}" y2="{bottom_y}" stroke="#333" stroke-width="1.5" />
{x_ticks}
<text x="{cx}" y="{xlabel_y}" text-anchor="middle" font-size="14" font-family="Arial">Training Episodes</text>
<text x="22" y="{cy}" text-anchor="middle" transform="rotate(-90 22 {cy})" font-size="14" font-family="Arial">Win Rate vs RLCard</text>
{overall}
{landlord}
{farmer}
{legend}
</svg>
""".format(
        width=width,
        height=height,
        cx=width / 2,
        cy=top + plot_h / 2,
        title=title,
        grid="\n".join(grid_lines),
        left=left,
        top=top,
        bottom_y=height - bottom,
        right_x=width - right,
        x_ticks="\n".join(x_ticks),
        xlabel_y=height - 18,
        overall=polyline(overall, "#1f77b4"),
        landlord=polyline(landlord, "#ff7f0e"),
        farmer=polyline(farmer, "#2ca02c"),
        legend=legend,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate .pkl checkpoints against RLCard and plot win-rate curves"
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--method", required=True, choices=["pg", "ac"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--eval-data", default="evaluate_results/pg_full_5k_eval_data_1000.pkl")
    parser.add_argument("--result-dir", default="evaluate_results")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--assignment-workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    checkpoints = find_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError("No .pkl checkpoints found in {}".format(
            args.checkpoint_dir
        ))

    rows = []
    for episode, checkpoint_path in checkpoints:
        evaluate_name = "{}_{}k_vs_native_rlcard_rotate_1000".format(
            args.name,
            episode // 1000,
        )
        result_path = run_evaluation(
            args.method,
            checkpoint_path,
            args.result_dir,
            args.eval_data,
            args.num_workers,
            args.assignment_workers,
            evaluate_name,
            args.skip_existing,
        )
        stats = parse_result(result_path, args.method, checkpoint_path)
        row = {
            "episodes": episode,
            "overall_win_rate": stats["overall_win_rate"],
            "landlord_win_rate": stats["landlord_win_rate"],
            "farmer_win_rate": stats["farmer_win_rate"],
            "games": stats["games"],
            "checkpoint": checkpoint_path.as_posix(),
            "result_json": result_path.as_posix(),
        }
        rows.append(row)
        print(
            "episode={} overall={:.4f} landlord={:.4f} farmer={:.4f}".format(
                episode,
                float(row["overall_win_rate"]),
                float(row["landlord_win_rate"]),
                float(row["farmer_win_rate"]),
            ),
            flush=True,
        )

    csv_path = Path(args.result_dir) / "{}_training_curve_vs_native_rlcard_1000.csv".format(
        args.name
    )
    svg_path = Path(args.result_dir) / "{}_training_curve_vs_native_rlcard_1000.svg".format(
        args.name
    )
    write_csv(rows, csv_path)
    plot_curve(rows, svg_path, "{} vs Native RLCard".format(args.name))
    print("saved CSV to {}".format(csv_path))
    print("saved curve to {}".format(svg_path))


if __name__ == "__main__":
    main()
