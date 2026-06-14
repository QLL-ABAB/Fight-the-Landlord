"""
可视化 adversarial_agent 在不同参数下与 rlcard 对战的胜率
图中包含三条折线：landlord、landlord_up、landlord_down
50% 处画虚线作为基准线
"""
import argparse
import csv
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from douzero.evaluation.simulation import resolve_eval_data_path, simulate_one_assignment


def evaluate_adversarial_vs_rlcard(card_play_data_list, position, num_workers, adversarial_param=""):
    """
    评估 adversarial_agent 在指定位置与 rlcard 对战
    
    Args:
        card_play_data_list: 牌局数据列表
        position: 位置 ("landlord", "landlord_up", "landlord_down")
        num_workers: 并行工作数
        adversarial_param: adversarial 参数，如 "adv:800"
    
    Returns:
        win_rate: 胜率
    """
    method = f"adversarial{':' + adversarial_param if adversarial_param else ''}"
    
    role_to_method = {
        "landlord": "rlcard",
        "landlord_up": "rlcard", 
        "landlord_down": "rlcard",
    }
    role_to_method[position] = method
    
    result = simulate_one_assignment(
        card_play_data_list,
        role_to_method,
        num_workers,
        show_progress=False,
    )
    
    if position == "landlord":
        return result["landlord_win_rate"]
    else:
        return result["farmer_win_rate"]


def run_evaluation(params_list, eval_data_path, num_workers):
    """
    运行评估并收集数据
    
    Args:
        params_list: 参数列表，如 [800, 1000, 1200]
        eval_data_path: 评估数据路径
        num_workers: 并行工作数
    
    Returns:
        results: 结果字典
    """
    with open(eval_data_path, "rb") as f:
        card_play_data_list = pickle.load(f)
    
    positions = ["landlord", "landlord_up", "landlord_down"]
    results = {pos: [] for pos in positions}
    
    for param in params_list:
        print(f"Evaluating adversarial with param={param}")
        
        for position in positions:
            win_rate = evaluate_adversarial_vs_rlcard(
                card_play_data_list, 
                position, 
                num_workers,
                str(param)
            )
            results[position].append({
                "param": param,
                "win_rate": win_rate
            })
            print(f"  {position}: {win_rate:.2%}")
    
    return results


def plot_results(results, output_path):
    """
    绘制胜率折线图
    
    Args:
        results: 结果字典
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "landlord": "#1f77b4",      # 蓝色
        "landlord_up": "#2ca02c",   # 绿色
        "landlord_down": "#ff7f0e", # 橙色
    }
    
    labels = {
        "landlord": "Landlord",
        "landlord_up": "Landlord_up",
        "landlord_down": "Landlord_down",
    }
    
    # 绘制三条折线
    for position in ["landlord", "landlord_up", "landlord_down"]:
        data = results[position]
        params = [r["param"] for r in data]
        win_rates = [r["win_rate"] for r in data]
        
        ax.plot(
            params,
            win_rates,
            marker='o',
            linewidth=2.5,
            markersize=6,
            label=labels[position],
            color=colors[position],
        )
    
    # 绘制 50% 虚线基准线
    min_param = min(results["landlord"][0]["param"], results["landlord"][-1]["param"])
    max_param = max(results["landlord"][0]["param"], results["landlord"][-1]["param"])
    ax.hlines(
        0.5,
        min_param,
        max_param,
        color="#555555",
        linestyle="--",
        linewidth=2,
        label="50% :baseline",
        alpha=0.7,
    )
    
    ax.set_title("Adversarial Agent vs RLCard win rate", fontsize=14, pad=15)
    ax.set_xlabel("Adversarial agent param (num_samples)", fontsize=12)
    ax.set_ylabel("Win rate", fontsize=12)
    ax.set_ylim(0.3, 0.8)  # 设置 y 轴范围
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(fontsize=11)
    
    # 添加数据点标签
    for position in ["landlord", "landlord_up", "landlord_down"]:
        data = results[position]
        for r in data:
            ax.text(
                r["param"],
                r["win_rate"],
                f'{r["win_rate"]:.1%}',
                ha='center',
                va='bottom',
                fontsize=9,
                color=colors[position],
            )
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_results(results, csv_path):
    """
    保存结果到 CSV 文件
    
    Args:
        results: 结果字典
        csv_path: CSV 输出路径
    """
    rows = []
    positions = ["landlord", "landlord_up", "landlord_down"]
    
    for i in range(len(results["landlord"])):
        row = {
            "param": results["landlord"][i]["param"],
        }
        for pos in positions:
            row[f"{pos}_win_rate"] = results[pos][i]["win_rate"]
        rows.append(row)
    
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["param", "landlord_win_rate", "landlord_up_win_rate", "landlord_down_win_rate"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate adversarial agent win rate vs rlcard win rate"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="eval_data.pkl",
        help="评估数据路径",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="10,50,200,500,1000",
        help="参数列表，用逗号分隔",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并行工作数",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualization",
        help="输出目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    
    # 解析参数列表
    params_list = [int(p.strip()) for p in args.params.split(",")]
    
    # 解析评估数据路径
    eval_data_path = resolve_eval_data_path(args.eval_data)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行评估
    print(f"Starting evaluation with params: {params_list}")
    results = run_evaluation(params_list, eval_data_path, args.num_workers)
    
    # 保存结果
    csv_path = os.path.join(args.output_dir, "adversarial_vs_rlcard_results.csv")
    save_results(results, csv_path)
    print(f"Results saved to {csv_path}")
    
    # 绘制图表
    png_path = os.path.join(args.output_dir, "adversarial_vs_rlcard_plot.png")
    plot_results(results, png_path)
    print(f"Plot saved to {png_path}")


if __name__ == "__main__":
    main()