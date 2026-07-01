from __future__ import annotations

import argparse
import csv
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

from config import REPO_ROOT, with_prefix
from douzero.evaluation.rlcard_data import ensure_rlcard_doudizhu_jsondata
from douzero.evaluation.simulation import resolve_eval_data_path, simulate_one_assignment
from train_config import TRAIN_CONFIGS


TEST_CASES = ("landlord", "farmer1", "farmer2", "two_farmer")
OPPONENTS = ("rlcard", "random")


@dataclass(frozen=True)
class StrategyFamily:
    group: str
    series: str
    prefix: str
    config_name: str
    checkpoint_dir: Path
    farmer_csv: Path
    landlord_csv: Path


@dataclass
class SelectedCheckpoint:
    strategy_group: str
    source_role: str
    rank: int
    series: str
    csv_path: Path
    selected_episode: int
    selected_x_value: float | None
    selected_win_rate: float
    checkpoint_path: Path
    checkpoint_step: int


FAMILIES = (
    StrategyFamily(
        group="approx_doufeature_buffer_mcadv",
        series="mc_adv_buffer",
        prefix="approx_doufeature",
        config_name="approx_doufeature_logadp_mc_adv_buffer_1m",
        checkpoint_dir=REPO_ROOT
        / "approx_qlearning_checkpoints"
        / "approx_doufeature"
        / "approx_doufeature_logadp_mc_adv_buffer_1m",
        farmer_csv=REPO_ROOT
        / "visualization"
        / "approx_doufeature_buffer_td_mcadv_vs_rlcard_douzero_farmer.csv",
        landlord_csv=REPO_ROOT
        / "visualization"
        / "approx_doufeature_buffer_td_mcadv_vs_rlcard_douzero_landlord.csv",
    ),
    StrategyFamily(
        group="approx_doufeature_td",
        series="td_buffer",
        prefix="approx_doufeature",
        config_name="approx_doufeature_logadp_td_buffer_1m",
        checkpoint_dir=REPO_ROOT
        / "approx_qlearning_checkpoints"
        / "approx_doufeature"
        / "approx_doufeature_logadp_td_buffer_1m",
        farmer_csv=REPO_ROOT
        / "visualization"
        / "approx_doufeature_buffer_td_mcadv_vs_rlcard_douzero_farmer.csv",
        landlord_csv=REPO_ROOT
        / "visualization"
        / "approx_doufeature_buffer_td_mcadv_vs_rlcard_douzero_landlord.csv",
    ),
    StrategyFamily(
        group="approxq_douzero",
        series="approxq",
        prefix="approxq",
        config_name="approxq_logadp_cmp_1m_history",
        checkpoint_dir=REPO_ROOT
        / "approx_qlearning_checkpoints"
        / "approx_qlearning"
        / "approxq_logadp_cmp_1m_history",
        farmer_csv=REPO_ROOT
        / "visualization"
        / "approxq_douzero_vs_rlcard_fixed_trend_farmer.csv",
        landlord_csv=REPO_ROOT
        / "visualization"
        / "approxq_douzero_vs_rlcard_fixed_trend_landlord.csv",
    ),
    StrategyFamily(
        group="attention_dou",
        series="attention_dou",
        prefix="attention_dou",
        config_name="attention_dou_logadp_mc_adv_buffer_1m",
        checkpoint_dir=REPO_ROOT
        / "attention_dou_checkpoints"
        / "attention_dou"
        / "attention_dou_logadp_mc_adv_buffer_1m_fast",
        farmer_csv=REPO_ROOT
        / "visualization"
        / "attention_dou_vs_rlcard_douzero_farmer_frames.csv",
        landlord_csv=REPO_ROOT
        / "visualization"
        / "attention_dou_vs_rlcard_douzero_landlord_frames.csv",
    ),
)


OUTPUT_FIELDS = [
    "selection_rank",
    "strategy_group",
    "source_role",
    "series",
    "csv_path",
    "selected_episode",
    "selected_x_value",
    "selected_win_rate",
    "checkpoint_step",
    "checkpoint_path",
    "opponent",
    "test_case",
    "metric",
    "tested_win_rate",
    "landlord_win_rate",
    "farmer_win_rate",
    "games",
    "landlord_wins",
    "farmer_wins",
    "landlord_method",
    "landlord_up_method",
    "landlord_down_method",
]


def role_csv_path(family: StrategyFamily, role: str) -> Path:
    if role == "farmer":
        return family.farmer_csv
    if role == "landlord":
        return family.landlord_csv
    raise ValueError(f"Unknown role: {role}")


def short_config_line(family: StrategyFamily) -> str:
    config = TRAIN_CONFIGS[family.config_name]
    if family.group.startswith("approx_doufeature"):
        return (
            f"update_mode={config.update_mode}, alpha={config.alpha}, gamma={config.gamma}, "
            f"buffer_size={config.buffer_size}, learn_batch_size={config.learn_batch_size}, "
            f"learn_steps={config.learn_steps}, num_workers={config.num_workers}, "
            f"save_interval={config.save_interval}"
        )
    if family.group == "approxq_douzero":
        return (
            f"alpha={config.alpha}, gamma={config.gamma}, reward_shaping={config.reward_shaping}, "
            f"feature_mode={config.feature_mode}, save_interval={config.save_interval}"
        )
    if family.group == "attention_dou":
        return (
            f"update_mode={config.update_mode}, learning_rate={config.learning_rate}, "
            f"gamma={config.gamma}, hidden_dim={config.hidden_dim}, num_heads={config.num_heads}, "
            f"num_layers={config.num_layers}, num_workers={config.num_workers}, "
            f"num_threads={config.num_threads}, unroll_length={config.unroll_length}, "
            f"num_buffers={config.num_buffers}, learn_batch_size={config.learn_batch_size}, "
            f"save_interval={config.save_interval}"
        )
    return ""


def read_candidates(family: StrategyFamily, role: str):
    csv_path = role_csv_path(family, role)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {"series", "episode", "win_rate"}.issubset(fields):
            raise ValueError(f"Unexpected CSV columns in {csv_path}: {reader.fieldnames}")
        for row in reader:
            if row.get("series") != family.series:
                continue
            try:
                episode = int(round(float(row["episode"])))
                win_rate = float(row["win_rate"])
                x_value = float(row["x_value"]) if "x_value" in row and row["x_value"] else None
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Bad row in {csv_path}: {row}") from exc
            rows.append(
                SelectedCheckpoint(
                    strategy_group=family.group,
                    source_role=role,
                    rank=0,
                    series=family.series,
                    csv_path=csv_path,
                    selected_episode=episode,
                    selected_x_value=x_value,
                    selected_win_rate=win_rate,
                    checkpoint_path=Path(),
                    checkpoint_step=episode,
                )
            )
    if not rows:
        raise ValueError(
            f"No rows matched series={family.series} in {csv_path}. "
            f"Available series: {sorted({row.get('series', '') for row in csv.DictReader(csv_path.open('r', encoding='utf-8'))})}"
        )
    rows.sort(key=lambda row: (-row.selected_win_rate, -row.selected_episode))
    return rows


def resolve_checkpoint_path(family: StrategyFamily, checkpoint_step: int) -> Path:
    exact = family.checkpoint_dir / f"{checkpoint_step}.pkl"
    if exact.exists():
        return exact
    candidates = []
    for path in family.checkpoint_dir.glob("*.pkl"):
        if not path.stem.isdigit():
            continue
        candidates.append((abs(int(path.stem) - checkpoint_step), int(path.stem), path))
    if not candidates:
        raise FileNotFoundError(f"No numeric pkl checkpoints under {family.checkpoint_dir}")
    candidates.sort()
    return candidates[0][2]


def select_top_rows(top_k: int):
    selected = []
    for family in FAMILIES:
        for role in ("landlord", "farmer"):
            candidates = read_candidates(family, role)[:top_k]
            for index, row in enumerate(candidates, start=1):
                selected.append(
                    SelectedCheckpoint(
                        strategy_group=row.strategy_group,
                        source_role=row.source_role,
                        rank=index,
                        series=row.series,
                        csv_path=row.csv_path,
                        selected_episode=row.selected_episode,
                        selected_x_value=row.selected_x_value,
                        selected_win_rate=row.selected_win_rate,
                        checkpoint_path=resolve_checkpoint_path(family, row.checkpoint_step),
                        checkpoint_step=row.checkpoint_step,
                    )
                )
    return selected


def assignment_for_case(method: str, opponent: str, test_case: str):
    if test_case == "landlord":
        return {"landlord": method, "landlord_up": opponent, "landlord_down": opponent}, "landlord_win_rate"
    if test_case == "farmer1":
        return {"landlord": opponent, "landlord_up": method, "landlord_down": opponent}, "farmer_win_rate"
    if test_case == "farmer2":
        return {"landlord": opponent, "landlord_up": opponent, "landlord_down": method}, "farmer_win_rate"
    if test_case in ("two_farmer", "two_framer"):
        return {"landlord": opponent, "landlord_up": method, "landlord_down": method}, "farmer_win_rate"
    raise ValueError(f"Unknown test case: {test_case}")


def evaluate_assignment(card_play_data_list, role_to_method, metric, num_workers):
    result = simulate_one_assignment(
        card_play_data_list,
        role_to_method,
        num_workers,
        show_progress=False,
    )
    return result, result[metric]


def write_selection_markdown(path: Path, selected_rows):
    rows_by_family_role = {}
    for row in selected_rows:
        rows_by_family_role.setdefault((row.strategy_group, row.source_role), []).append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 策略选择摘要\n\n")
        handle.write("## 选中的 checkpoint 数字\n\n")
        for family in FAMILIES:
            handle.write(f"### {family.group}\n\n")
            for role in ("landlord", "farmer"):
                rows = sorted(
                    rows_by_family_role.get((family.group, role), []),
                    key=lambda row: row.rank,
                )
                digits = ", ".join(str(row.checkpoint_step) for row in rows)
                handle.write(f"- {role}: {digits}\n")
            handle.write("\n")
        handle.write("## 训练配置\n\n")
        for family in FAMILIES:
            handle.write(f"### {family.group}\n\n")
            handle.write(f"- config: `{family.config_name}`\n")
            handle.write(f"- {short_config_line(family)}\n\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select top-3 checkpoints per role for the four target strategies and evaluate them."
    )
    parser.add_argument(
        "--eval_data",
        type=Path,
        default=REPO_ROOT / "eval_data.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "visualization" / "策略选择.csv",
    )
    parser.add_argument(
        "--summary_md",
        type=Path,
        default=REPO_ROOT / "visualization" / "策略选择.md",
    )
    parser.add_argument("--num_games", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["rlcard", "random"],
        choices=["rlcard", "random"],
    )
    parser.add_argument(
        "--test_cases",
        nargs="+",
        default=["landlord", "farmer1", "farmer2", "two_farmer"],
        choices=["landlord", "farmer1", "farmer2", "two_farmer", "two_framer"],
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    os.environ.setdefault("ATTENTION_DOU_EVAL_THREADS", "1")
    ensure_rlcard_doudizhu_jsondata()

    selected_rows = select_top_rows(args.top_k)
    if not selected_rows:
        raise RuntimeError("No checkpoints were selected.")

    print("Selected checkpoints:")
    for row in selected_rows:
        print(
            f"  {row.strategy_group}/{row.source_role}#{row.rank}: "
            f"episode={row.selected_episode} win_rate={row.selected_win_rate} "
            f"checkpoint={row.checkpoint_path}"
        )

    write_selection_markdown(args.summary_md, selected_rows)
    print(f"Saved summary markdown to {args.summary_md}")

    if args.dry_run:
        return

    if args.output.exists():
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    eval_data_path = resolve_eval_data_path(str(args.eval_data))
    with open(eval_data_path, "rb") as handle:
        card_play_data_list = pickle.load(handle)
    if args.num_games > 0:
        card_play_data_list = card_play_data_list[: args.num_games]
    print(f"Using {len(card_play_data_list)} eval games from {eval_data_path}")

    total = len(selected_rows) * len(args.opponents) * len(args.test_cases)
    index = 0
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in selected_rows:
            method = with_prefix(
                next(f.prefix for f in FAMILIES if f.group == row.strategy_group),
                row.checkpoint_path,
            )
            for opponent in args.opponents:
                for test_case in args.test_cases:
                    index += 1
                    print(
                        f"[{index}/{total}] eval {row.strategy_group}/{row.source_role}#{row.rank} "
                        f"vs {opponent} case={test_case}"
                    )
                    role_to_method, metric = assignment_for_case(method, opponent, test_case)
                    result, tested_win_rate = evaluate_assignment(
                        card_play_data_list,
                        role_to_method,
                        metric,
                        args.num_workers,
                    )
                    writer.writerow(
                        {
                            "selection_rank": f"{row.strategy_group}/{row.source_role}#{row.rank}",
                            "strategy_group": row.strategy_group,
                            "source_role": row.source_role,
                            "series": row.series,
                            "csv_path": str(row.csv_path),
                            "selected_episode": row.selected_episode,
                            "selected_x_value": row.selected_x_value if row.selected_x_value is not None else "",
                            "selected_win_rate": row.selected_win_rate,
                            "checkpoint_step": row.checkpoint_step,
                            "checkpoint_path": str(row.checkpoint_path),
                            "opponent": opponent,
                            "test_case": test_case,
                            "metric": metric,
                            "tested_win_rate": tested_win_rate,
                            "landlord_win_rate": result["landlord_win_rate"],
                            "farmer_win_rate": result["farmer_win_rate"],
                            "games": result["games"],
                            "landlord_wins": result["landlord_wins"],
                            "farmer_wins": result["farmer_wins"],
                            "landlord_method": role_to_method["landlord"],
                            "landlord_up_method": role_to_method["landlord_up"],
                            "landlord_down_method": role_to_method["landlord_down"],
                        }
                    )

    print(f"Saved strategy selection results to {args.output}")


if __name__ == "__main__":
    main()
