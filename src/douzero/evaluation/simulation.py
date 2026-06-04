import multiprocessing as mp
import json
import os
import pickle

from douzero.env.game import GameEnv


POSITIONS = ["landlord", "landlord_up", "landlord_down"]
FARMER_POSITIONS = ["landlord_up", "landlord_down"]
STAT_KEYS = [
    "games",
    "wins",
    "landlord_games",
    "landlord_wins",
    "farmer_games",
    "farmer_wins",
]


def load_card_play_models(card_play_model_path_dict):
    players = {}

    for position in POSITIONS:
        if card_play_model_path_dict[position] == "rlcard":
            from .rlcard_agent import RLCardAgent

            players[position] = RLCardAgent(position)
        elif card_play_model_path_dict[position] == "random":
            from .random_agent import RandomAgent

            players[position] = RandomAgent()
        elif card_play_model_path_dict[position] == "mdp":
            from .mdp_agent import BayesianMDPAgent

            players[position] = BayesianMDPAgent(position)
        elif card_play_model_path_dict[position] == "qlearning":
            from .qlearning_agent import QLearningAgent

            players[position] = QLearningAgent(position)
        elif card_play_model_path_dict[position].startswith("qlearning:"):
            from .qlearning_agent import QLearningAgent

            model_path = card_play_model_path_dict[position].split(":", 1)[1]
            players[position] = QLearningAgent(position, model_path)
        elif isinstance(card_play_model_path_dict[position], str) and card_play_model_path_dict[position].startswith('search'):
            from .search_agent import SearchAgent
            
            s = card_play_model_path_dict[position]
            parts = s.split(':')
            if len(parts) == 2 and parts[1].isdigit():
                players[position] = SearchAgent(int(parts[1]))
            else:
                players[position] = SearchAgent()
        elif isinstance(card_play_model_path_dict[position], str) and card_play_model_path_dict[position].startswith('expectimax'):
            from .expectimax_agent import ExpectimaxAgent

            s = card_play_model_path_dict[position]
            parts = s.split(':')
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                players[position] = ExpectimaxAgent(int(parts[1]), int(parts[2]))
            elif len(parts) == 2 and parts[1].isdigit():
                players[position] = ExpectimaxAgent(int(parts[1]))
            else:
                players[position] = ExpectimaxAgent()
        else:
            from .deep_agent import DeepAgent

            players[position] = DeepAgent(position, card_play_model_path_dict[position])
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


def mp_simulate(card_play_data_list, card_play_model_path_dict, role_to_method, q):

    players = load_card_play_models(card_play_model_path_dict)
    methods = list(dict.fromkeys(role_to_method.values()))
    player_stats = empty_player_stats(methods)

    env = GameEnv(players)
    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        update_player_stats(player_stats, role_to_method, env.get_winner())
        env.reset()

    q.put(
        (
            env.num_wins["landlord"],
            env.num_wins["farmer"],
            env.num_scores["landlord"],
            env.num_scores["farmer"],
            player_stats,
        )
    )


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
    player_stats = empty_player_stats(role_to_method.values())

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

    for i in range(num_workers):
        result = q.get()
        num_landlord_wins += result[0]
        num_farmer_wins += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores += result[3]
        merge_player_stats(player_stats, result[4])

    total_games = num_landlord_wins + num_farmer_wins
    return {
        "roles": dict(role_to_method),
        "games": total_games,
        "landlord_wins": num_landlord_wins,
        "farmer_wins": num_farmer_wins,
        "landlord_win_rate": num_landlord_wins / total_games if total_games else None,
        "farmer_win_rate": num_farmer_wins / total_games if total_games else None,
        "landlord_adp": num_landlord_scores / total_games if total_games else None,
        "farmer_adp": 2 * num_farmer_scores / total_games if total_games else None,
        "player_stats": add_win_rates(player_stats),
    }


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
             result_dir="evaluate_results"):

    eval_data_path = resolve_eval_data_path(eval_data)
    with open(eval_data_path, "rb") as f:
        card_play_data_list = pickle.load(f)

    if methods is None:
        methods = [landlord, landlord_up, landlord_down]
    if len(methods) != 3:
        raise ValueError("Exactly three methods are required")

    assignments = build_role_assignments(methods, eval_mode)
    summary_stats = empty_player_stats(methods)
    assignment_results = []

    for idx, role_to_method in enumerate(assignments):
        print("Evaluating assignment {}: {}".format(idx + 1, role_to_method))
        assignment_result = simulate_one_assignment(
            card_play_data_list, role_to_method, num_workers
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
        "evaluate_name": evaluate_name,
        "mode": eval_mode,
        "methods": methods,
        "eval_data": eval_data_path,
        "num_workers": num_workers,
        "summary": summary,
        "assignments": assignment_results,
    }

    path = result_path(evaluate_name, result_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True, ensure_ascii=False)

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
