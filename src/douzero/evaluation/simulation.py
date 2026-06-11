import multiprocessing as mp
import json
import os
import pickle
import traceback
from pathlib import Path

from tqdm import tqdm

from douzero.env.game import GameEnv


POSITIONS = ["landlord", "landlord_up", "landlord_down"]
FARMER_POSITIONS = ["landlord_up", "landlord_down"]
ROLE_ZH = {
    "landlord": "地主",
    "landlord_up": "农民-地主上家",
    "landlord_down": "农民-地主下家",
}
STAT_KEYS = [
    "games",
    "wins",
    "landlord_games",
    "landlord_wins",
    "farmer_games",
    "farmer_wins",
]


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _checkpoint_step_from_name(path):
    stem = Path(path).stem
    marker = "_weights_"
    if marker not in stem:
        return ""
    return stem.rsplit(marker, 1)[1]


def _douzero_checkpoint_dir(path):
    candidate = Path(path)
    if candidate.is_dir():
        return candidate
    return candidate.parent


def _latest_position_checkpoint(directory, position):
    prefix = "{}_weights_".format(position)
    candidates = []
    for path in Path(directory).glob("{}*.ckpt".format(prefix)):
        step = _checkpoint_step_from_name(path)
        numeric_step = int(step) if step.isdigit() else -1
        candidates.append((numeric_step, path.stat().st_mtime, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][2]


def resolve_douzero_model_path(position, model_path):
    raw_path = model_path.split(":", 1)[1] if ":" in model_path else model_path
    path = Path(raw_path)
    if not path.is_absolute():
        path = _repo_root() / path

    directory = _douzero_checkpoint_dir(path)
    step = _checkpoint_step_from_name(path)
    if step:
        candidate = directory / "{}_weights_{}.ckpt".format(position, step)
        if candidate.exists():
            return str(candidate)

    latest = _latest_position_checkpoint(directory, position)
    if latest:
        return str(latest)

    if path.exists():
        return str(path)

    raise FileNotFoundError(
        "Cannot resolve DouZero checkpoint for position {} from {}".format(
            position, model_path
        )
    )


def model_label(method):
    if not isinstance(method, str):
        return method
    if method.startswith("douzero:"):
        return method
    if method.endswith(".ckpt"):
        return "douzero:{}".format(method)
    return method


def resolve_model_for_position(position, method):
    if not isinstance(method, str):
        return method
    if method.startswith("douzero:"):
        return resolve_douzero_model_path(position, method)
    if method.endswith(".ckpt"):
        return resolve_douzero_model_path(position, "douzero:{}".format(method))
    return method


def label_role_assignment(role_to_method):
    return {
        position: model_label(method)
        for position, method in role_to_method.items()
    }


def startswith_method(method, prefix):
    return isinstance(method, str) and method.startswith(prefix)


def load_card_play_models(card_play_model_path_dict):
    players = {}

    for position in POSITIONS:
        method = card_play_model_path_dict[position]
        if method == "rlcard":
            from .rlcard_agent import RLCardAgent

            players[position] = RLCardAgent(position)
        elif method == "random":
            from .random_agent import RandomAgent

            players[position] = RandomAgent()
        elif method == "mdp":
            from .mdp_agent import BayesianMDPAgent

            players[position] = BayesianMDPAgent(position)
        elif method == "adv":
            from .adversarial_agent import AdversarialSearchAgent

            players[position] = AdversarialSearchAgent(position)
        elif method == "qlearning":
            from .qlearning_agent import QLearningAgent

            players[position] = QLearningAgent(position)
        elif startswith_method(method, "qlearning:"):
            from .qlearning_agent import QLearningAgent

            model_path = method.split(":", 1)[1]
            players[position] = QLearningAgent(position, model_path)
        elif method in ["approxq", "approx_qlearning"]:
            from .approx_qlearning_agent import ApproxQLearningAgent

            players[position] = ApproxQLearningAgent(position)
        elif startswith_method(method, "approxq:") or \
                startswith_method(method, "approx_qlearning:"):
            from .approx_qlearning_agent import ApproxQLearningAgent

            model_path = method.split(":", 1)[1]
            players[position] = ApproxQLearningAgent(position, model_path)
        elif startswith_method(method, "douzero:"):
            from .deep_agent import DeepAgent

            players[position] = DeepAgent(
                position, resolve_douzero_model_path(position, method)
            )
        elif isinstance(method, str) and method.endswith(".ckpt"):
            from .deep_agent import DeepAgent

            players[position] = DeepAgent(
                position, resolve_douzero_model_path(
                    position, "douzero:{}".format(method)
                )
            )
        elif isinstance(method, str) and method.startswith('search'):
            from .search_agent import SearchAgent
            
            s = method
            parts = s.split(':')
            if len(parts) == 2 and parts[1].isdigit():
                players[position] = SearchAgent(int(parts[1]))
            else:
                players[position] = SearchAgent()
        elif isinstance(method, str) and method.startswith('expectimax'):
            from .expectimax_agent import ExpectimaxAgent

            s = method
            parts = s.split(':')
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                players[position] = ExpectimaxAgent(int(parts[1]), int(parts[2]))
            elif len(parts) == 2 and parts[1].isdigit():
                players[position] = ExpectimaxAgent(int(parts[1]))
            else:
                players[position] = ExpectimaxAgent()
        else:
            from .deep_agent import DeepAgent

            players[position] = DeepAgent(position, method)
    return players


def empty_player_stats(methods):
    return {
        method: {
            "games": 0,
            "wins": 0,
            "landlord_games": 0,
            "landlord_wins": 0,
            "farmer_games": 0,
            "farmer_wins": 0,
        }
        for method in methods
    }


def update_player_stats(stats, role_to_method, winner):
    for position in POSITIONS:
        method = role_to_method[position]
        if method not in stats:
            stats[method] = empty_player_stats([method])[method]

        stats[method]["games"] += 1
        if position == "landlord":
            stats[method]["landlord_games"] += 1
            if winner == "landlord":
                stats[method]["wins"] += 1
                stats[method]["landlord_wins"] += 1
        else:
            stats[method]["farmer_games"] += 1
            if winner == "farmer":
                stats[method]["wins"] += 1
                stats[method]["farmer_wins"] += 1


def merge_player_stats(target, source):
    for method, method_stats in source.items():
        if method not in target:
            target[method] = empty_player_stats([method])[method]
        for key in STAT_KEYS:
            value = method_stats.get(key, 0)
            target[method][key] += value


def add_win_rates(stats):
    result = {}
    for method, method_stats in stats.items():
        games = method_stats["games"]
        landlord_games = method_stats["landlord_games"]
        farmer_games = method_stats["farmer_games"]
        enriched = dict(method_stats)
        enriched["overall_win_rate"] = (
            method_stats["wins"] / games if games else None
        )
        enriched["landlord_win_rate"] = (
            method_stats["landlord_wins"] / landlord_games
            if landlord_games else None
        )
        enriched["farmer_win_rate"] = (
            method_stats["farmer_wins"] / farmer_games
            if farmer_games else None
        )
        result[method] = enriched
    return result


def mp_simulate(card_play_data_list, card_play_model_path_dict, role_to_method, q, show_progress=True):
    try:
        players = load_card_play_models(card_play_model_path_dict)
        labeled_role_to_method = label_role_assignment(role_to_method)
        methods = list(dict.fromkeys(labeled_role_to_method.values()))
        player_stats = empty_player_stats(methods)

        env = GameEnv(players)
        total_games = len(card_play_data_list)
        
        if show_progress:
            iterator = tqdm(card_play_data_list, desc=f"Simulating games", 
                            total=total_games, leave=True, ncols=80)
        else:
            iterator = card_play_data_list
        
        for idx, card_play_data in enumerate(iterator):
            env.card_play_init(card_play_data)
            while not env.game_over:
                env.step()
            update_player_stats(
                player_stats, labeled_role_to_method, env.get_winner()
            )
            env.reset()

        q.put(
            (
                "ok",
                env.num_wins["landlord"],
                env.num_wins["farmer"],
                env.num_scores["landlord"],
                env.num_scores["farmer"],
                player_stats,
            )
        )
    except Exception:
        q.put(("error", traceback.format_exc()))


def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker


def simulate_one_assignment(card_play_data_list, role_to_method, num_workers):
    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers
    )

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_landlord_scores = 0
    num_farmer_scores = 0
    labeled_role_to_method = label_role_assignment(role_to_method)
    player_stats = empty_player_stats(labeled_role_to_method.values())

    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    processes = []
    for card_play_data in card_play_data_list_each_worker:
        p = ctx.Process(
            target=mp_simulate,
            args=(card_play_data, role_to_method, role_to_method, q),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                "Evaluation worker exited with code {}".format(p.exitcode)
            )

    for i in range(num_workers):
        result = q.get()
        if result[0] == "error":
            raise RuntimeError(
                "Evaluation worker failed:\n{}".format(result[1])
            )
        num_landlord_wins += result[1]
        num_farmer_wins += result[2]
        num_landlord_scores += result[3]
        num_farmer_scores += result[4]
        merge_player_stats(player_stats, result[5])

    total_games = num_landlord_wins + num_farmer_wins
    return {
        "roles": dict(labeled_role_to_method),
        "resolved_roles": {
            position: resolve_model_for_position(position, method)
            for position, method in role_to_method.items()
        },
        "roles_zh": {
            ROLE_ZH[position]: labeled_role_to_method[position]
            for position in POSITIONS
        },
        "组合说明": (
            "地主={}; 农民-地主上家={}; 农民-地主下家={}".format(
                labeled_role_to_method["landlord"],
                labeled_role_to_method["landlord_up"],
                labeled_role_to_method["landlord_down"],
            )
        ),
        "games": total_games,
        "landlord_wins": num_landlord_wins,
        "farmer_wins": num_farmer_wins,
        "landlord_win_rate": num_landlord_wins / total_games if total_games else None,
        "farmer_win_rate": num_farmer_wins / total_games if total_games else None,
        "landlord_adp": num_landlord_scores / total_games if total_games else None,
        "farmer_adp": 2 * num_farmer_scores / total_games if total_games else None,
        "player_stats": add_win_rates(player_stats),
    }


def mp_assignment_simulate(card_play_data_list, role_to_method, num_workers, index, q):
    try:
        assignment_result = simulate_one_assignment(
            card_play_data_list, role_to_method, num_workers
        )
        q.put(("ok", index, assignment_result))
    except Exception:
        q.put(("error", index, traceback.format_exc()))


def simulate_assignments(card_play_data_list, assignments, num_workers,
                         assignment_workers):
    if assignment_workers <= 1 or len(assignments) <= 1:
        return [
            simulate_one_assignment(card_play_data_list, role_to_method, num_workers)
            for role_to_method in assignments
        ]

    print(
        "Running {} assignments in parallel with assignment_workers={} "
        "and num_workers={} per assignment".format(
            len(assignments), min(assignment_workers, len(assignments)), num_workers
        )
    )
    max_workers = max(1, min(assignment_workers, len(assignments)))
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    results = [None] * len(assignments)
    running = []
    next_index = 0

    def start_next():
        nonlocal next_index
        if next_index >= len(assignments):
            return
        p = ctx.Process(
            target=mp_assignment_simulate,
            args=(
                card_play_data_list,
                assignments[next_index],
                num_workers,
                next_index,
                q,
            ),
        )
        p.start()
        running.append(p)
        next_index += 1

    for _ in range(max_workers):
        start_next()

    completed = 0
    while completed < len(assignments):
        status, index, payload = q.get()
        completed += 1
        if status == "error":
            for p in running:
                if p.is_alive():
                    p.terminate()
            for p in running:
                p.join()
            raise RuntimeError(
                "Assignment {} failed:\n{}".format(index + 1, payload)
            )
        results[index] = payload

        still_running = []
        for p in running:
            p.join(timeout=0)
            if p.exitcode is None:
                still_running.append(p)
            elif p.exitcode != 0:
                raise RuntimeError(
                    "Assignment worker exited with code {}".format(p.exitcode)
                )
        running = still_running
        start_next()

    for p in running:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                "Assignment worker exited with code {}".format(p.exitcode)
            )

    return results


def build_role_assignments(methods, eval_mode):
    if eval_mode == "fixed":
        return [
            {
                "landlord": methods[0],
                "landlord_up": methods[1],
                "landlord_down": methods[2],
            }
        ]

    if eval_mode == "rotate":
        assignments = []
        for i in range(3):
            assignments.append(
                {
                    "landlord": methods[i],
                    "landlord_up": methods[(i + 1) % 3],
                    "landlord_down": methods[(i + 2) % 3],
                }
            )
        return assignments

    raise ValueError("Unknown eval_mode: {}".format(eval_mode))


def add_assignment_notes(assignment_result, index, total_assignments):
    return {
        "轮换编号": "{}/{}".format(index, total_assignments),
        "中文说明": (
            "第{}组：{}，{}，{}".format(
                index,
                assignment_result["roles_zh"]["地主"],
                "农民-地主上家={}".format(
                    assignment_result["roles_zh"]["农民-地主上家"]
                ),
                "农民-地主下家={}".format(
                    assignment_result["roles_zh"]["农民-地主下家"]
                ),
            )
        ),
        "胜率说明": "landlord_win_rate 是地主阵营胜率，farmer_win_rate 是两个农民阵营合计胜率。",
        "ADP说明": "landlord_adp 是地主阵营平均分差，farmer_adp 是两个农民合计口径的平均分差。",
        **assignment_result,
    }


def json_with_assignment_spacing(output):
    text = json.dumps(output, indent=2, ensure_ascii=False)
    return text.replace(
        "\n    },\n    {\n      \"轮换编号\"",
        "\n    },\n\n    {\n      \"轮换编号\"",
    )


def readable_assignment_summary(role_to_method, index):
    return (
        "第{}组：地主={}，农民-地主上家={}，农民-地主下家={}".format(
            index,
            role_to_method["landlord"],
            role_to_method["landlord_up"],
            role_to_method["landlord_down"],
        )
    )


def result_path(evaluate_name, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    return os.path.join(result_dir, "{}.json".format(evaluate_name))


def resolve_eval_data_path(eval_data):
    if os.path.exists(eval_data):
        return eval_data

    if os.path.isabs(eval_data):
        return eval_data

    src_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    src_candidate = os.path.join(src_root, eval_data)
    if os.path.exists(src_candidate):
        return src_candidate

    return eval_data


def evaluate(landlord, landlord_up, landlord_down, eval_data, num_workers,
             eval_mode="fixed", methods=None, evaluate_name="evaluate",
             result_dir="evaluate_results", assignment_workers=1):

    eval_data_path = resolve_eval_data_path(eval_data)
    with open(eval_data_path, "rb") as f:
        card_play_data_list = pickle.load(f)

    if methods is None:
        methods = [landlord, landlord_up, landlord_down]
    if len(methods) != 3:
        raise ValueError("Exactly three methods are required")

    labeled_methods = [model_label(method) for method in methods]
    assignments = build_role_assignments(methods, eval_mode)
    summary_stats = empty_player_stats(labeled_methods)
    assignment_results = []

    for idx, role_to_method in enumerate(assignments):
        print("Evaluating assignment {}: {}".format(idx + 1, role_to_method))
        print(readable_assignment_summary(role_to_method, idx + 1))

    raw_assignment_results = simulate_assignments(
        card_play_data_list, assignments, num_workers, assignment_workers
    )

    for idx, assignment_result in enumerate(raw_assignment_results):
        assignment_result = add_assignment_notes(
            assignment_result, idx + 1, len(assignments)
        )
        assignment_results.append(assignment_result)
        merge_player_stats(summary_stats, assignment_result["player_stats"])

        print("WP results:")
        print(
            "landlord : Farmers - {} : {}".format(
                assignment_result["landlord_win_rate"],
                assignment_result["farmer_win_rate"],
            )
        )
        print("ADP results:")
        print(
            "landlord : Farmers - {} : {}".format(
                assignment_result["landlord_adp"],
                assignment_result["farmer_adp"],
            )
        )

    summary = add_win_rates(summary_stats)
    output = {
        "说明": {
            "mode": "rotate 表示三个 method 轮流担任地主；fixed 表示座位固定。",
            "roles": "landlord=地主，landlord_up=农民-地主上家，landlord_down=农民-地主下家。",
            "assignments": "每一项是一组座位组合，文件中用空行分隔三组 rotate 结果。",
            "landlord_win_rate": "该组中地主阵营胜率。",
            "farmer_win_rate": "该组中两个农民阵营合计胜率。",
            "landlord_adp": "该组中地主阵营平均分差。",
            "farmer_adp": "该组中农民阵营平均分差，已按两个农民合计口径输出。",
            "summary": "按 method 汇总整体胜率、担任地主胜率、担任农民胜率。",
            "assignment_workers": "并行评测多少组座位组合；rotate 最多 3 组。",
            "num_workers": "每组座位组合内部并行评测多少份牌局。",
        },
        "evaluate_name": evaluate_name,
        "mode": eval_mode,
        "methods": labeled_methods,
        "raw_methods": methods,
        "eval_data": eval_data_path,
        "num_workers": num_workers,
        "assignment_workers": assignment_workers,
        "summary": summary,
        "assignments": assignment_results,
    }

    path = result_path(evaluate_name, result_dir)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_with_assignment_spacing(output))
        f.write("\n")

    print("Player summary:")
    for method, stats in summary.items():
        print(
            "{} overall={} landlord={} farmer={}".format(
                method,
                stats["overall_win_rate"],
                stats["landlord_win_rate"],
                stats["farmer_win_rate"],
            )
        )
    print("saved evaluation result to {}".format(path))
    return output