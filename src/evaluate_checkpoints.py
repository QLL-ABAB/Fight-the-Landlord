# ------------------------------------------------------------
# Evaluate checkpoints and plot win rate trends
# ------------------------------------------------------------

import argparse
import json
import os
import re
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


def find_checkpoints(checkpoint_dir, pattern=r"^weights\.ep(\d+)\.json$"):
    """Find role-policy checkpoint files matching weights.ep{episode}.json."""
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, f)
        if not os.path.isfile(path):
            continue
        match = re.match(pattern, f)
        if match:
            ep = int(match.group(1))
            checkpoints.append((ep, path))
    # Sort by episode number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def evaluate_checkpoint(weights_path, position, eval_games=100):
    """Evaluate a checkpoint at a specific position against rlcard."""
    # Build the command
    cmd = [
        sys.executable, "src/evaluate.py",
        "--eval_mode", "fixed",
        "--evaluate_name", f"eval_temp_{position}",
        "--result_dir", "evaluate_results_temp"
    ]
    
    # Set the position to the neural policy agent, others to rlcard
    if position == "landlord":
        cmd.extend(["--landlord", f"nnpolicy:{weights_path}"])
        cmd.extend(["--landlord_up", "rlcard"])
        cmd.extend(["--landlord_down", "rlcard"])
    elif position == "landlord_up":
        cmd.extend(["--landlord", "rlcard"])
        cmd.extend(["--landlord_up", f"nnpolicy:{weights_path}"])
        cmd.extend(["--landlord_down", "rlcard"])
    elif position == "landlord_down":
        cmd.extend(["--landlord", "rlcard"])
        cmd.extend(["--landlord_up", "rlcard"])
        cmd.extend(["--landlord_down", f"nnpolicy:{weights_path}"])
    
    print(f"Evaluating {weights_path} as {position}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error evaluating {weights_path}: {result.stderr}")
        return None
    
    # Parse the result
    result_file = os.path.join("evaluate_results_temp", f"eval_temp_{position}.json")
    if not os.path.exists(result_file):
        print(f"Result file not found: {result_file}")
        return None
    
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract win rate for the agent at the given position
    agent_key = f"nnpolicy:{weights_path}"
    for assignment in data.get("assignments", []):
        # Check if the agent is in this assignment
        roles = assignment.get("roles", {})
        if roles.get(position) == agent_key:
            # For fixed mode, get the win rate based on position
            player_stats = assignment.get("player_stats", {})
            if agent_key in player_stats:
                stats = player_stats[agent_key]
                return stats.get("overall_win_rate", None)
    
    return None


def plot_trends(results, output_dir, name_suffix="", plot_name="win_rate_trend"):
    """Plot win rate trends for all three positions."""
    positions = ["landlord", "landlord_up", "landlord_down"]
    labels = ["Landlord", "Landlord Up", "Landlord Down"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    # Sort by episode
    episodes = sorted(results.keys())
    
    plt.figure(figsize=(10, 6))
    for i, position in enumerate(positions):
        win_rates = [results[ep].get(position, None) for ep in episodes]
        plt.plot(episodes, win_rates, label=labels[i], color=colors[i], marker='o', markersize=4)
    
    plt.xlabel("Training Episodes")
    plt.ylabel("Win Rate")
    plt.title("Win Rate vs Training Episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{plot_name}_all{name_suffix}.png"), dpi=150)
    plt.close()
    
    # Individual plots
    for i, position in enumerate(positions):
        plt.figure(figsize=(10, 6))
        win_rates = [results[ep].get(position, None) for ep in episodes]
        plt.plot(episodes, win_rates, label=labels[i], color=colors[i], marker='o', markersize=4)
        plt.xlabel("Training Episodes")
        plt.ylabel("Win Rate")
        plt.title(f"Win Rate vs Training Episodes - {labels[i]}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{plot_name}_{position}{name_suffix}.png"), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser("Evaluate checkpoints and plot win rate trends")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory containing checkpoint files")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="Number of games to evaluate per checkpoint")
    parser.add_argument("--output-dir", type=str, default="visualization",
                        help="Directory to save plots")
    parser.add_argument("--name", type=str, default="",
                        help="Custom name suffix for output files (prevents overwriting)")
    parser.add_argument("--plot-name", type=str, default="win_rate_trend",
                        help="Base name for output plot PNG files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip evaluation if result file already exists")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints matching weights.ep<number>.json")
    
    # Results dictionary: {episode: {position: win_rate}}
    results = {}
    
    # Evaluate each checkpoint
    for ep, weights_path in checkpoints:
        results[ep] = {}
        for position in ["landlord", "landlord_up", "landlord_down"]:
            win_rate = evaluate_checkpoint(weights_path, position, args.eval_games)
            if win_rate is not None:
                results[ep][position] = win_rate
                print(f"Episode {ep}, {position}: {win_rate:.4f}")
            else:
                print(f"Episode {ep}, {position}: FAILED")
    
    # Generate name suffix to prevent overwriting
    name_suffix = ""
    if args.name:
        # Use custom name if provided
        name_suffix = f"_{args.name}"
    else:
        # Use timestamp as default suffix
        name_suffix = f"_{int(time.time())}"
    
    # Save results to JSON
    results_file = os.path.join(args.output_dir, f"checkpoint_eval_results{name_suffix}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # Plot trends
    plot_trends(results, args.output_dir, name_suffix, args.plot_name)
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
