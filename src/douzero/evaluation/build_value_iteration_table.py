# build_value_iteration_table.py
# ------------------------------------------------------------
# Offline builder for value tables.
#
# It can save either:
#   hand mode:    old reduced table, key = current hand string;
#   feature mode: observable feature-state table, key = f(infoset, hand).
#
# Feature mode does not use hidden opponent hands. It still uses the reduced
# hand-decomposition planner for Bellman values, then stores them under richer
# observable state keys so the online value agent no longer treats "my hand
# only" as the whole state.
#
#   value_iteration_tables/<run>/
#       config.json
#       metadata.json
#       values.json
#       policy.json
# ------------------------------------------------------------

import argparse
import json
import os
import random
import time

try:
    from douzero.evaluation.value_iteration_planner import ValueIterationPlanner, CARD_ORDER
    from douzero.evaluation.valueiteration_agent import ValueIterationAgent
except Exception:
    from value_iteration_planner import ValueIterationPlanner, CARD_ORDER
    from valueiteration_agent import ValueIterationAgent


ALL_POSITIONS = ["landlord", "landlord_down", "landlord_up"]


class FakeInfoSet(object):
    """Minimal observable infoset used to build feature-state keys offline."""

    def __init__(self, position, hand, num_cards_left, last_move=None,
                 last_two_moves=None, last_pid=None, played_cards=None):
        self.player_position = position
        self.player_hand_cards = list(hand)
        self.legal_actions = []
        self.last_move = list(last_move or [])
        self.last_two_moves = list(last_two_moves or [[], []])
        self.last_pid = last_pid
        self.num_cards_left_dict = dict(num_cards_left)
        self.num_cards_left = dict(num_cards_left)
        self.player_num_cards_left = dict(num_cards_left)
        self.played_cards = played_cards or {
            "landlord": [],
            "landlord_down": [],
            "landlord_up": [],
        }


def repo_root():
    here = os.path.abspath(__file__)
    cur = os.path.dirname(here)
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "src")) and os.path.isdir(os.path.join(cur, "src", "douzero")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.getcwd()


def resolve_path(path, root=None):
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(root or repo_root(), path))


def safe_name(text):
    out = []
    for ch in str(text or ""):
        out.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(out).strip("._") or "run"


def make_run_dir(root_dir, run_name=None, seed=None):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if run_name:
        base = safe_name(run_name)
    else:
        base = "run_{}{}".format(time.strftime("%Y%m%d_%H%M%S"), "_seed{}".format(seed))
    run_dir = os.path.join(root_dir, base)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        return run_dir
    for i in range(2, 1000):
        candidate = os.path.join(root_dir, "{}_{}".format(base, i))
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
    raise RuntimeError("Could not create unique run directory under {}".format(root_dir))


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def format_eta(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return "{}h{}m{}s".format(h, m, s)
    if m:
        return "{}m{}s".format(m, s)
    return "{}s".format(s)


def print_expand_progress(done, total, states, start_time):
    elapsed = time.time() - start_time
    rate = done / max(1e-9, elapsed)
    remaining = (total - done) / max(1e-9, rate)
    pct = 100.0 * done / max(1, total)
    print(
        "expand_roots {}/{} ({:.1f}%) states {} rate {:.2f}/s elapsed {:.1f}s eta {}".format(
            done, total, pct, states, rate, elapsed, format_eta(remaining)
        ),
        flush=True,
    )


def card_id_to_symbol(card):
    card = int(card)
    level = card // 4 + (1 if card == 53 else 0)
    if level <= 11:
        return CARD_ORDER[level]
    if level == 12:
        return "2"
    if level == 13:
        return "B"
    if level == 14:
        return "R"
    return "3"


def cards_to_hand(cards):
    symbols = [card_id_to_symbol(c) for c in cards]
    index = {c: i for i, c in enumerate(CARD_ORDER)}
    return "".join(sorted(symbols, key=lambda c: index[c]))


def sample_root_hands(deals, seed, include_landlord=True, include_farmers=True):
    rng = random.Random(seed)
    roots = []
    for _ in range(int(deals)):
        deck = list(range(54))
        rng.shuffle(deck)
        public = deck[:3]
        landlord = deck[3:20] + public
        farmer_down = deck[20:37]
        farmer_up = deck[37:54]
        if include_landlord:
            roots.append(cards_to_hand(landlord))
        if include_farmers:
            roots.append(cards_to_hand(farmer_down))
            roots.append(cards_to_hand(farmer_up))
    return roots


def sample_root_contexts(deals, seed, include_landlord=True, include_farmers=True):
    rng = random.Random(seed)
    roots = []
    for _ in range(int(deals)):
        deck = list(range(54))
        rng.shuffle(deck)
        public = deck[:3]
        landlord = cards_to_hand(deck[3:20] + public)
        farmer_down = cards_to_hand(deck[20:37])
        farmer_up = cards_to_hand(deck[37:54])
        num_left = {
            "landlord": len(landlord),
            "landlord_down": len(farmer_down),
            "landlord_up": len(farmer_up),
        }
        if include_landlord:
            roots.append({"position": "landlord", "hand": landlord, "num_left": dict(num_left)})
        if include_farmers:
            roots.append({"position": "landlord_down", "hand": farmer_down, "num_left": dict(num_left)})
            roots.append({"position": "landlord_up", "hand": farmer_up, "num_left": dict(num_left)})
    return roots


def build_feature_table(planner, root_contexts, all_states, max_states_per_root,
                        max_feature_states, progress_every):
    agents = {
        position: ValueIterationAgent(position, planner=planner, state_mode="feature")
        for position in ALL_POSITIONS
    }
    value_sums = {}
    value_counts = {}
    feature_policy = {}
    t0 = time.time()
    done = 0

    for i, root in enumerate(root_contexts, 1):
        local_states = planner.enumerate_states(root["hand"], max_states=max_states_per_root)
        for hand in local_states:
            if hand not in all_states:
                continue
            num_left = dict(root["num_left"])
            num_left[root["position"]] = len(hand)
            info = FakeInfoSet(
                position=root["position"],
                hand=hand,
                num_cards_left=num_left,
            )
            keys = agents[root["position"]].feature_state_keys(info, hand)
            for key in keys:
                value_sums[key] = value_sums.get(key, 0.0) + planner.values.get(hand, 0.0)
                value_counts[key] = value_counts.get(key, 0) + 1
                if hand in planner.policy:
                    feature_policy[key] = planner.policy[hand]
                if len(value_sums) >= max_feature_states:
                    print(
                        "feature_state_limit_reached max_feature_states {}".format(max_feature_states),
                        flush=True,
                    )
                    return (
                        {key: value_sums[key] / max(1, value_counts[key]) for key in value_sums},
                        feature_policy,
                    )
            done += 1

        if progress_every and (i % progress_every == 0 or i == len(root_contexts)):
            elapsed = time.time() - t0
            print(
                "feature_keys {}/{} values {} emitted {} elapsed {:.1f}s".format(
                    i, len(root_contexts), len(value_sums), done, elapsed
                ),
                flush=True,
            )

    return (
        {key: value_sums[key] / max(1, value_counts[key]) for key in value_sums},
        feature_policy,
    )


def main():
    ap = argparse.ArgumentParser("Build reduced-MDP value-iteration table")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default="value_iteration_tables")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--roles", choices=["all", "landlord", "farmers"], default="all")
    ap.add_argument("--state-mode", choices=["hand", "feature"], default="hand",
                    help="hand saves old my-hand keys; feature saves observable feature-state keys")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--step-reward", type=float, default=-1.0)
    ap.add_argument("--terminal-reward", type=float, default=100.0)
    ap.add_argument("--pass-reward", type=float, default=-1.0)
    ap.add_argument("--theta", type=float, default=1e-6)
    ap.add_argument("--max-iterations", type=int, default=200)
    ap.add_argument("--max-states", type=int, default=200000)
    ap.add_argument("--max-feature-states", type=int, default=300000,
                    help="Only used by --state-mode feature")
    ap.add_argument("--max-states-per-root", type=int, default=5000,
                    help="Limit expansion per sampled root hand so progress stays responsive")
    ap.add_argument("--max-actions-per-state", type=int, default=120)
    ap.add_argument("--max-wing-combinations", type=int, default=16)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--vi-progress-every", type=int, default=10)
    args = ap.parse_args()

    root = repo_root()
    out_root = resolve_path(args.out_dir, root)
    run_dir = make_run_dir(out_root, args.run_name, args.seed)

    planner = ValueIterationPlanner(
        gamma=args.gamma,
        step_reward=args.step_reward,
        terminal_reward=args.terminal_reward,
        pass_reward=args.pass_reward,
        theta=args.theta,
        max_iterations=args.max_iterations,
        max_states=args.max_states,
        max_actions_per_state=args.max_actions_per_state,
        max_wing_combinations=args.max_wing_combinations,
    )
    planner.progress_every = args.vi_progress_every

    include_landlord = args.roles in ("all", "landlord")
    include_farmers = args.roles in ("all", "farmers")
    if args.state_mode == "feature":
        roots = sample_root_contexts(args.deals, args.seed, include_landlord, include_farmers)
    else:
        roots = sample_root_hands(args.deals, args.seed, include_landlord, include_farmers)

    states = set([""])
    t0 = time.time()
    truncated = False
    total_roots = len(roots)
    print(
        "expanding_roots total_roots {} max_states {} run_dir {}".format(
            total_roots, args.max_states, run_dir
        ),
        flush=True,
    )
    for i, hand in enumerate(roots, 1):
        if isinstance(hand, dict):
            hand = hand["hand"]
        before = len(states)
        states.update(planner.enumerate_states(hand, max_states=args.max_states_per_root))
        if len(states) >= args.max_states:
            truncated = True
            states = set(list(states)[: args.max_states])
            print_expand_progress(i, total_roots, len(states), t0)
            print("state_limit_reached max_states {}".format(args.max_states), flush=True)
            break
        if args.progress_every and i % args.progress_every == 0:
            print_expand_progress(i, total_roots, len(states), t0)
        elif args.progress_every and time.time() - t0 > 10 and i == 1:
            print(
                "expanded_first_root added_states {} total_states {} elapsed {:.1f}s".format(
                    len(states) - before, len(states), time.time() - t0
                ),
                flush=True,
            )
    else:
        print_expand_progress(total_roots, total_roots, len(states), t0)

    print("running_value_iteration states {} max_iterations {}".format(len(states), args.max_iterations), flush=True)
    planner.run_value_iteration(states)
    planner.stats["truncated"] = truncated or planner.stats.get("truncated", False)
    planner.stats["sampled_roots"] = len(roots)
    planner.stats["expanded_roots"] = i if roots else 0
    planner.stats["deals"] = args.deals
    planner.stats["roles"] = args.roles
    planner.stats["state_mode"] = args.state_mode

    if args.state_mode == "feature":
        print("building_feature_keys roots {} max_feature_states {}".format(len(roots), args.max_feature_states), flush=True)
        feature_values, feature_policy = build_feature_table(
            planner=planner,
            root_contexts=roots,
            all_states=states,
            max_states_per_root=args.max_states_per_root,
            max_feature_states=args.max_feature_states,
            progress_every=args.progress_every,
        )
        planner.values = feature_values
        planner.policy = feature_policy
        planner.stats["feature_values"] = len(feature_values)
        planner.stats["feature_policy"] = len(feature_policy)

    planner.save_table(run_dir)

    write_json(os.path.join(run_dir, "config.json"), vars(args))
    index_path = os.path.join(out_root, "planning_runs.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {"runs": []}
    index["runs"].append({
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": run_dir,
        "config": vars(args),
        "stats": dict(planner.stats),
    })
    write_json(index_path, index)

    print("saved", run_dir)
    print("values", len(planner.values), "policy", len(planner.policy), "iterations", planner.stats["iterations"])


if __name__ == "__main__":
    main()
