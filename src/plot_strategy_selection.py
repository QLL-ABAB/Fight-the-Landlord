from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import REPO_ROOT


TEST_CASES = ["landlord", "farmer1", "farmer2", "two_farmer"]
OPPONENTS = ["rlcard", "random"]
TEST_CASE_LABELS = {
    "landlord": "Landlord",
    "farmer1": "Farmer 1",
    "farmer2": "Farmer 2",
    "two_farmer": "Two Farmers",
}


#TODO: 读取策略选择评测结果，并按 strategy_group 聚合候选 checkpoint 的最高胜率。
def load_best_rows(path: Path):
    best = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                win_rate = float(row["tested_win_rate"])
            except (KeyError, TypeError, ValueError):
                continue
            method = row.get("strategy_group") or row.get("strategy") or row.get("method_prefix")
            key = (method, row["opponent"], row["test_case"])
            current = best.get(key)
            if current is None or win_rate > current["tested_win_rate"]:
                best[key] = {
                    "method": method,
                    "strategy": row.get("strategy", method),
                    "opponent": row["opponent"],
                    "test_case": row["test_case"],
                    "tested_win_rate": win_rate,
                    "checkpoint": row.get("checkpoint", ""),
                    "source": row.get("selected_csv", ""),
                    "selection_rank": row.get("selection_rank", ""),
                }
    return best


#TODO: 保存画图用的聚合数据，便于检查每个柱子来自哪个 checkpoint。
def write_summary(path: Path, rows):
    fields = [
        "method",
        "strategy",
        "opponent",
        "test_case",
        "best_win_rate",
        "checkpoint",
        "source",
        "selection_rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "method": row["method"],
                "strategy": row["strategy"],
                "opponent": row["opponent"],
                "test_case": row["test_case"],
                "best_win_rate": row["tested_win_rate"],
                "checkpoint": row["checkpoint"],
                "source": row["source"],
                "selection_rank": row["selection_rank"],
            })


#TODO: 画 4x2 子图：4 种测试位置 x 2 种对手，每个柱子是该方法三个候选里的最高胜率。
def plot_bars(path: Path, summary_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({row["method"] for row in summary_rows})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}

    fig, axes = plt.subplots(
        len(TEST_CASES),
        len(OPPONENTS),
        figsize=(15, 13),
        sharey=True,
    )
    for row_index, test_case in enumerate(TEST_CASES):
        for col_index, opponent in enumerate(OPPONENTS):
            ax = axes[row_index][col_index]
            values = []
            labels = []
            for method in methods:
                match = next(
                    (
                        row
                        for row in summary_rows
                        if row["method"] == method
                        and row["opponent"] == opponent
                        and row["test_case"] == test_case
                    ),
                    None,
                )
                if match is None:
                    continue
                labels.append(method)
                values.append(match["tested_win_rate"])
            ax.bar(
                labels,
                values,
                color=[color_map[label] for label in labels],
                edgecolor="#2f2f2f",
                linewidth=0.7,
            )
            ax.set_title(f"{TEST_CASE_LABELS[test_case]} vs {opponent}")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", linestyle=":", alpha=0.45)
            ax.tick_params(axis="x", rotation=25)
            for x, value in enumerate(values):
                ax.text(x, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
            if col_index == 0:
                ax.set_ylabel("Win rate")

    fig.suptitle(
        "Best win rate by method from top-3 selected checkpoints per CSV",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 4x2 bar charts from visualization/策略选择.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "visualization" / "策略选择.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "visualization" / "策略选择_4x2柱形图.png",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "visualization" / "策略选择_4x2柱形图数据.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    best = load_best_rows(args.input)
    rows = sorted(
        best.values(),
        key=lambda row: (row["opponent"], row["test_case"], row["method"]),
    )
    if not rows:
        raise RuntimeError(f"No valid rows found in {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_summary(args.summary, rows)
    plot_bars(args.output, rows)
    print(f"Saved summary data to {args.summary}")
    print(f"Saved 4x2 bar chart to {args.output}")


if __name__ == "__main__":
    main()
