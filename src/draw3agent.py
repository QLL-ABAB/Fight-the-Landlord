"""
可视化三个 agent 在三个位置的胜率对比柱状图
- agents: adversarial, probability, value
- positions: landlord, landlord_up, landlord_down
- 总共 9 根柱子
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


def evaluate_agent_vs_rlcard(card_play_data_list, position, method, num_workers):
    """
    评估指定 agent 在指定位置与 rlcard 对战
    
    Args:
        card_play_data_list: 牌局数据列表
        position: 位置 ("landlord", "landlord_up", "landlord_down")
        method: agent 名称
        num_workers: 并行工作数
    
    Returns:
        win_rate: 胜率
    """
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


def run_evaluation(eval_data_path, num_workers):
    """
    运行所有评估
    
    Args:
        eval_data_path: 评估数据路径
        num_workers: 并行工作数
    
    Returns:
        results: 结果字典
    """
    with open(eval_data_path, "rb") as f:
        card_play_data_list = pickle.load(f)
    
    agents = ["adversarial", "probability", "value"]
    positions = ["landlord", "landlord_up", "landlord_down"]
    
    results = {}
    
    for agent in agents:
        results[agent] = {}
        print(f"Evaluating {agent}...")
        
        for position in positions:
            win_rate = evaluate_agent_vs_rlcard(
                card_play_data_list, 
                position, 
                agent,
                num_workers
            )
            results[agent][position] = win_rate
            print(f"  {position}: {win_rate:.2%}")
    
    return results


def plot_results(results, output_path):
    """
    绘制柱状图
    
    Args:
        results: 结果字典
        output_path: 输出路径
    """
    agents = ["adversarial", "probability", "value"]
    positions = ["landlord", "landlord_up", "landlord_down"]
    
    agent_labels = {
        "adversarial": "Adversarial",
        "probability": "Probability",
        "value": "Value Iteration",
    }
    
    position_labels = {
        "landlord": "landlord",
        "landlord_up": "landlord_up",
        "landlord_down": "landlord_down",
    }
    
    colors = {
        "adversarial": "#1f77b4",      # 蓝色
        "probability": "#2ca02c",      # 绿色
        "value": "#ff7f0e",            # 橙色
    }
    
    # 设置柱子宽度和间距
    bar_width = 0.28
    index = range(len(positions))
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # 绘制每个 agent 的柱子
    for i, agent in enumerate(agents):
        win_rates = [results[agent][pos] for pos in positions]
        ax.bar(
            [x + i * bar_width for x in index],
            win_rates,
            width=bar_width,
            label=agent_labels[agent],
            color=colors[agent],
            edgecolor='white',
            linewidth=1,
        )
    
    # 添加 50% 基准线
    ax.axhline(
        y=0.5,
        color='#555555',
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label='50% 基准线'
    )
    
    # 设置图表属性
    ax.set_title("Agent win rate vs RLCard", fontsize=16, pad=20)
    ax.set_xlabel("Position", fontsize=13)
    ax.set_ylabel("Win Rate", fontsize=13)
    ax.set_xticks([x + bar_width for x in index])
    ax.set_xticklabels([position_labels[pos] for pos in positions], fontsize=12)
    ax.set_ylim(0.3, 0.85)  # 设置 y 轴范围
    ax.grid(True, linestyle=":", alpha=0.45, axis='y')
    ax.legend(fontsize=12)
    
    # 添加数据标签
    for i, agent in enumerate(agents):
        for j, pos in enumerate(positions):
            win_rate = results[agent][pos]
            ax.text(
                j + i * bar_width,
                win_rate + 0.01,
                f'{win_rate:.1%}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color=colors[agent],
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
    agents = ["adversarial", "probability", "value"]
    positions = ["landlord", "landlord_up", "landlord_down"]
    
    for agent in agents:
        row = {"agent": agent}
        for pos in positions:
            row[pos] = results[agent][pos]
        rows.append(row)
    
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent", "landlord", "landlord_up", "landlord_down"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估三个 agent（adversarial、probability、value）在三个位置的胜率并绘制柱状图"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="eval_data.pkl",
        help="评估数据路径",
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
    
    # 解析评估数据路径
    eval_data_path = resolve_eval_data_path(args.eval_data)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行评估
    print("Starting evaluation...")
    results = run_evaluation(eval_data_path, args.num_workers)
    
    # 保存结果
    csv_path = os.path.join(args.output_dir, "agent_win_rates.csv")
    save_results(results, csv_path)
    print(f"Results saved to {csv_path}")
    
    # 绘制图表
    png_path = os.path.join(args.output_dir, "agent_win_rates_bar.png")
    plot_results(results, png_path)
    print(f"Plot saved to {png_path}")
    
    # 打印汇总结果
    print("\n=== 评估结果汇总 ===")
    print(f"{'Agent':<15} {'地主':<10} {'农民-上家':<12} {'农民-下家':<12}")
    print("-" * 50)
    for agent in ["adversarial", "probability", "value"]:
        print(f"{agent:<15} {results[agent]['landlord']:<10.1%} {results[agent]['landlord_up']:<12.1%} {results[agent]['landlord_down']:<12.1%}")


if __name__ == "__main__":
    main()