from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from config import REPO_ROOT, with_prefix
from douzero.evaluation.rlcard_data import ensure_rlcard_doudizhu_jsondata
from douzero.evaluation.simulation import resolve_eval_data_path, simulate_one_assignment


DOUZERO_RE = re.compile(r"landlord_weights_(\d+)\.ckpt$")
OUTPUT_FIELDS = [
    "selection_rank",
    "strategy_group",
    "strategy",
    "method_prefix",
    "checkpoint",
    "selected_episode",
    "selected_x_value",
    "selected_win_rate",
    "selected_csv",
    "selected_role",
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


@dataclass(frozen=True)
class StrategySpec:
    prefix: str
    directory: Path
    kind: str = "pkl"


@dataclass
class BestRow:
    rank: int
    strategy_group: str
    strategy: str
    episode: float
    x_value: float | None
    win_rate: float
    csv_path: Path
    selected_role: str


STRATEGY_SPECS = {
    "approxq": StrategySpec(
        "approxq",
        REPO_ROOT / "approx_qlearning_checkpoints" / "approx_qlearning" / "approxq_logadp_cmp_1m_history",
    ),
    "finetune": StrategySpec(
        "approxq",
        REPO_ROOT / "approx_qlearning_checkpoints" / "approx_qlearning" / "approxq_logadp_best_landlord_finetune_50k",
    ),
    "time_equal": StrategySpec(
        "approxq",
        REPO_ROOT / "approx_qlearning_checkpoints" / "approx_qlearning" / "approxq_logadp_best_landlord_finetune_50k_time_equal",
    ),
    "td_1m": StrategySpec(
        "approx_doufeature",
        REPO_ROOT / "approx_qlearning_checkpoints" / "approx_doufeature" / "approx_doufeature_logadp_td_1m",
    ),
    "mc_1m": StrategySpec(
        "approx_doufeature",
        REPO_ROOT / "approx_qlearning_checkpoints" / "approx_doufeature" / "approx_doufeature_logadp_mc_1m",
    ),
    "td_1m_gamma": StrategySpec(
        "approx_doufeature",
        REPO_ROOT / "approx_qlearning_checkpoints" / "approx_doufeature" / "approx_doufeature_logadp_td_1m_gamma",
    ),
    "td_buffer": StrategySpec(
        "approx_doufeature",
        REPO_ROOT / "approx_qlearning_checkpoints" / "approx_doufeature" / "approx_doufeature_logadp_td_buffer_1m",
    ),
    "mc_adv_buffer": StrategySpec(
        "approx_doufeature",
        REPO_ROOT / "approx_qlearning_checkpoints" / "approx_doufeature" / "approx_doufeature_logadp_mc_adv_buffer_1m",
    ),
    "attention_dou": StrategySpec(
        "attention_dou",
        REPO_ROOT / "attention_dou_checkpoints" / "attention_dou" / "attention_dou_logadp_mc_adv_buffer_1m_fast",
    ),
    "douzero": StrategySpec(
        "douzero",
        REPO_ROOT / "base" / "douzero_checkpoints" / "douzero_logadp_cmp_60000000",
        "douzero",
    ),
}


#TODO: 把趋势 CSV 的 series 映射到我们汇报用的四个策略类别。
def strategy_group_for_series(series: str):
    if series == "approxq":
        return "approxq_douzero"
    if series in ("td_1m", "td_1m_gamma", "td_buffer"):
        return "approx_doufeature_td"
    if series == "mc_adv_buffer":
        return "approx_doufeature_buffer_mcadv"
    if series == "attention_dou":
        return "attention_dou"
    return None


#TODO: 判断 CSV 文件名对应的原始趋势评测角色。
def role_from_csv_name(path: Path):
    name = path.stem.lower()
    if "landlord" in name:
        return "landlord"
    if "farmer" in name:
        return "farmer"
    return "unknown"


#TODO: 从 visualization CSV 中收集所有可映射到 checkpoint 的候选行。
def collect_candidate_rows(visualization_dir: Path):
    candidates = []
    for path in sorted(visualization_dir.glob("*.csv")):
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not {"series", "episode", "win_rate"}.issubset(fields):
                continue
            for row in reader:
                strategy = row.get("series", "")
                strategy_group = strategy_group_for_series(strategy)
                if strategy_group is None or strategy not in STRATEGY_SPECS:
                    continue
                try:
                    episode = float(row["episode"])
                    win_rate = float(row["win_rate"])
                    x_value = float(row["x_value"]) if row.get("x_value") else None
                except (TypeError, ValueError):
                    continue
                candidates.append(BestRow(
                    rank=0,
                    strategy_group=strategy_group,
                    strategy=strategy,
                    episode=episode,
                    x_value=x_value,
                    win_rate=win_rate,
                    csv_path=path,
                    selected_role=role_from_csv_name(path),
                ))
    return candidates


#TODO: 按 CSV 中已有 win_rate 选择要重测的 checkpoint，不再重测所有 checkpoint。
def select_rows(candidates, mode: str, top_k: int):
    if mode == "top_per_strategy_group":
        grouped = {}
        for candidate in candidates:
            grouped.setdefault(candidate.strategy_group, []).append(candidate)
        selected = []
        for strategy_group in sorted(grouped):
            rows = sorted(
                grouped[strategy_group],
                key=lambda row: (-row.win_rate, row.strategy, -row.episode),
            )
            deduped = []
            seen = set()
            for row in rows:
                key = (
                    row.strategy,
                    round(row.episode, 6),
                    round(row.x_value, 6) if row.x_value is not None else None,
                )
                if key in seen:
                    continue
                seen.add(key)
                row.rank = len(deduped) + 1
                deduped.append(row)
                if top_k > 0 and len(deduped) >= top_k:
                    break
            selected.extend(deduped)
        return selected

    if mode == "top_per_csv":
        grouped = {}
        for candidate in candidates:
            grouped.setdefault(candidate.csv_path, []).append(candidate)
        selected = []
        for csv_path in sorted(grouped):
            rows = sorted(
                grouped[csv_path],
                key=lambda row: (-row.win_rate, row.strategy, -row.episode),
            )
            deduped = []
            seen = set()
            for row in rows:
                key = (
                    row.strategy,
                    round(row.episode, 6),
                    round(row.x_value, 6) if row.x_value is not None else None,
                )
                if key in seen:
                    continue
                seen.add(key)
                row.rank = len(deduped) + 1
                deduped.append(row)
                if top_k > 0 and len(deduped) >= top_k:
                    break
            selected.extend(deduped)
        return selected

    if mode == "best_per_strategy":
        best = {}
        for candidate in candidates:
            current = best.get(candidate.strategy)
            if current is None or (candidate.win_rate, candidate.episode) > (
                current.win_rate,
                current.episode,
            ):
                best[candidate.strategy] = candidate
        selected = sorted(best.values(), key=lambda row: (-row.win_rate, row.strategy))
    elif mode == "top_global":
        selected = sorted(
            candidates,
            key=lambda row: (-row.win_rate, row.strategy, -row.episode),
        )
    else:
        raise ValueError(f"Unknown selection mode: {mode}")

    deduped = []
    seen = set()
    for row in selected:
        key = (
            row.strategy,
            round(row.episode, 6),
            round(row.x_value, 6) if row.x_value is not None else None,
        )
        if key in seen:
            continue
        seen.add(key)
        row.rank = len(deduped) + 1
        deduped.append(row)
        if top_k > 0 and len(deduped) >= top_k:
            break
    return deduped


#TODO: 在数字命名的 pkl checkpoint 中找最接近目标 episode 的文件。
def nearest_pkl(directory: Path, episode: float):
    candidates = []
    for path in directory.glob("*.pkl"):
        if path.stem.isdigit():
            candidates.append((abs(int(path.stem) - episode), int(path.stem), path))
    if not candidates:
        raise FileNotFoundError(f"No numeric .pkl checkpoints under {directory}")
    candidates.sort()
    return candidates[0][2]


#TODO: DouZero 的 checkpoint 数字是 frames；兼容旧 CSV 的 frame/40 和新 CSV 的 frame/60。
def nearest_douzero_checkpoint(directory: Path, selected: BestRow):
    candidates = []
    targets = []
    if selected.x_value and selected.x_value > 1000:
        targets.append(selected.x_value)
    targets.extend([selected.episode * 40.0, selected.episode * 60.0])
    for path in directory.glob("landlord_weights_*.ckpt"):
        match = DOUZERO_RE.match(path.name)
        if not match:
            continue
        frame = int(match.group(1))
        distance = min(abs(frame - target) for target in targets)
        candidates.append((distance, frame, path))
    if not candidates:
        raise FileNotFoundError(f"No landlord_weights_*.ckpt under {directory}")
    candidates.sort()
    return candidates[0][2]


#TODO: 把选出的 best row 解析成 evaluate.py 可识别的 method 字符串。
def method_for_selected(selected: BestRow):
    spec = STRATEGY_SPECS[selected.strategy]
    if spec.kind == "douzero":
        checkpoint = nearest_douzero_checkpoint(spec.directory, selected)
    else:
        checkpoint = nearest_pkl(spec.directory, selected.episode)
    return with_prefix(spec.prefix, checkpoint), checkpoint, spec.prefix


#TODO: 构造四种测试位置：地主、农民上家、农民下家、双农民。
def assignment_for_case(method: str, opponent: str, test_case: str):
    if test_case == "landlord":
        return {
            "landlord": method,
            "landlord_up": opponent,
            "landlord_down": opponent,
        }, "landlord_win_rate"
    if test_case == "farmer1":
        return {
            "landlord": opponent,
            "landlord_up": method,
            "landlord_down": opponent,
        }, "farmer_win_rate"
    if test_case == "farmer2":
        return {
            "landlord": opponent,
            "landlord_up": opponent,
            "landlord_down": method,
        }, "farmer_win_rate"
    if test_case in ("two_farmer", "two_framer"):
        return {
            "landlord": opponent,
            "landlord_up": method,
            "landlord_down": method,
        }, "farmer_win_rate"
    raise ValueError(f"Unknown test case: {test_case}")


#TODO: 已经存在的结果自动跳过，方便长时间评测断点续跑。
def existing_keys(output_path: Path):
    if not output_path.exists():
        return set()
    with output_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            (
                row.get("selection_rank", ""),
                row.get("strategy_group", ""),
                row["strategy"],
                row["checkpoint"],
                row["opponent"],
                row["test_case"],
            )
            for row in reader
        }


#TODO: 逐行追加结果，避免长任务中断后丢失。
def append_result(output_path: Path, row):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


#TODO: 运行一次固定位置评测，并抽取被测策略对应的胜率。
def evaluate_assignment(card_play_data_list, role_to_method, metric, num_workers):
    result = simulate_one_assignment(
        card_play_data_list,
        role_to_method,
        num_workers,
        show_progress=False,
    )
    return result, result[metric]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select best checkpoints from visualization CSVs and evaluate them."
    )
    parser.add_argument(
        "--visualization_dir",
        type=Path,
        default=REPO_ROOT / "visualization",
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
    parser.add_argument("--num_games", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--selection_mode",
        choices=["top_per_strategy_group", "top_per_csv", "top_global", "best_per_strategy"],
        default="top_per_strategy_group",
        help=(
            "top_per_strategy_group selects top-k rows inside every project strategy; "
            "top_per_csv selects top-k rows inside every CSV; "
            "top_global selects top-k rows overall; best_per_strategy keeps one row per strategy."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="How many selected CSV rows to evaluate; 0 means no limit.",
    )
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print selected checkpoints; do not evaluate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    os.environ.setdefault("ATTENTION_DOU_EVAL_THREADS", "1")
    ensure_rlcard_doudizhu_jsondata()

    candidates = collect_candidate_rows(args.visualization_dir)
    if not candidates:
        raise RuntimeError(f"No checkpoint curve CSVs found under {args.visualization_dir}")
    selected_rows = select_rows(candidates, args.selection_mode, args.top_k)

    selected_methods = []
    for selected in selected_rows:
        strategy = selected.strategy
        method, checkpoint, prefix = method_for_selected(selected)
        selected_methods.append((strategy, selected, method, checkpoint, prefix))

    print("Selected checkpoints from CSV rows:")
    for strategy, selected, _, checkpoint, prefix in selected_methods:
        print(
            f"  #{selected.rank} {selected.strategy_group}/{strategy}: "
            f"win_rate={selected.win_rate} episode={selected.episode} "
            f"prefix={prefix} checkpoint={checkpoint} source={selected.csv_path.name}"
        )

    if args.dry_run:
        return

    if args.overwrite and args.output.exists():
        args.output.unlink()
    done = existing_keys(args.output)

    eval_data_path = resolve_eval_data_path(str(args.eval_data))
    with open(eval_data_path, "rb") as handle:
        card_play_data_list = pickle.load(handle)
    if args.num_games > 0:
        card_play_data_list = card_play_data_list[: args.num_games]
    print(f"Using {len(card_play_data_list)} eval games from {eval_data_path}")

    total = len(selected_methods) * len(args.opponents) * len(args.test_cases)
    index = 0
    for strategy, selected, method, checkpoint, prefix in selected_methods:
        checkpoint_text = str(checkpoint)
        for opponent in args.opponents:
            for test_case in args.test_cases:
                index += 1
                rank_key = f"{selected.strategy_group}#{selected.rank}"
                key = (rank_key, selected.strategy_group, strategy, checkpoint_text, opponent, test_case)
                if key in done:
                    print(f"[{index}/{total}] skip {strategy} vs {opponent} {test_case}")
                    continue
                role_to_method, metric = assignment_for_case(method, opponent, test_case)
                print(f"[{index}/{total}] eval {strategy} vs {opponent} case={test_case}")
                result, tested_win_rate = evaluate_assignment(
                    card_play_data_list,
                    role_to_method,
                    metric,
                    args.num_workers,
                )
                append_result(
                    args.output,
                    {
                        "selection_rank": rank_key,
                        "strategy_group": selected.strategy_group,
                        "strategy": strategy,
                        "method_prefix": prefix,
                        "checkpoint": checkpoint_text,
                        "selected_episode": selected.episode,
                        "selected_x_value": selected.x_value if selected.x_value is not None else "",
                        "selected_win_rate": selected.win_rate,
                        "selected_csv": selected.csv_path.name,
                        "selected_role": selected.selected_role,
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
                    },
                )

    print(f"Saved strategy selection results to {args.output}")


if __name__ == "__main__":
    main()
