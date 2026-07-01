# value_iteration_planner.py
# ------------------------------------------------------------
# Pure reduced-MDP value iteration for Dou Dizhu hand decomposition.
#
# This planner does not model opponents, hidden cards, teammates, bidding, or
# initiative. Its state is only the current player's hand string, and every
# non-pass action removes cards from that hand:
#
#     V(s) = max_a [ R(s, a, s') + gamma * V(s') ]
#
# The default reward is intentionally simple:
#     - one step penalty for every played combination;
#     - terminal reward when the hand becomes empty.
# ------------------------------------------------------------

import json
import os
import time
from collections import Counter
from itertools import combinations


CARD_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "B", "R"]
NORMAL_CHAIN_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
INDEX = {c: i for i, c in enumerate(CARD_ORDER)}


class ValueIterationPlanner(object):
    """Value iteration over a finite reduced hand-decomposition MDP."""

    def __init__(
        self,
        gamma=1.0,
        step_reward=-1.0,
        terminal_reward=100.0,
        pass_reward=-1.0,
        theta=1e-6,
        max_iterations=200,
        max_states=5000,
        max_actions_per_state=120,
        max_wing_combinations=16,
        save_dir=None,
        auto_save=False,
    ):
        self.gamma = float(gamma)
        self.step_reward = float(step_reward)
        self.terminal_reward = float(terminal_reward)
        self.pass_reward = float(pass_reward)
        self.theta = float(theta)
        self.max_iterations = int(max_iterations)
        self.max_states = int(max_states)
        self.max_actions_per_state = int(max_actions_per_state)
        self.max_wing_combinations = int(max_wing_combinations)
        self.save_dir = save_dir
        self.auto_save = bool(auto_save)

        self.root_hand = None
        self.global_table_loaded = False
        self.allow_online_fallback = True
        self.values = {"": 0.0}
        self.policy = {}
        self.action_cache = {}
        self.stats = {
            "plans": 0,
            "states": 0,
            "iterations": 0,
            "last_delta": 0.0,
            "truncated": False,
        }

    def plan(self, root_hand):
        """Enumerate reachable states from root_hand and run value iteration."""
        root_hand = self.sort_hand(root_hand)
        if self.global_table_loaded and root_hand in self.values:
            return
        if self.global_table_loaded:
            if self.allow_online_fallback:
                states = self.enumerate_states(root_hand)
                self.run_value_iteration(states)
            return
        if self.root_hand is not None and self.is_subhand(root_hand, self.root_hand):
            return

        self.root_hand = root_hand
        self.global_table_loaded = False
        self.values = {"": 0.0}
        self.policy = {}
        self.action_cache = {}

        states = self.enumerate_states(root_hand)
        self.run_value_iteration(states)
        if self.auto_save and self.save_dir:
            self.save_plan(self.save_dir)

    def enumerate_states(self, root_hand, max_states=None):
        root_hand = self.sort_hand(root_hand)
        states = set(["", root_hand])
        stack = [root_hand]
        truncated = False
        limit = int(max_states or self.max_states)

        while stack and len(states) < limit:
            hand = stack.pop()
            for action in self.actions(hand):
                next_hand = self.next_hand(hand, action)
                if next_hand not in states:
                    states.add(next_hand)
                    if next_hand:
                        stack.append(next_hand)
                if len(states) >= limit:
                    truncated = True
                    break

        self.stats["truncated"] = truncated
        return states

    def run_value_iteration(self, states):
        states = set(states)
        states.add("")
        for hand in states:
            self.values[hand] = 0.0

        ordered = sorted(states, key=lambda s: len(s))
        last_delta = 0.0

        progress_every = int(getattr(self, "progress_every", 0) or 0)
        t0 = time.time()
        for iteration in range(1, self.max_iterations + 1):
            delta = 0.0
            new_values = {"": 0.0}
            new_policy = {}

            for hand in ordered:
                if hand == "":
                    continue

                best_q = None
                best_action = None
                for action in self.actions(hand):
                    q = self.q_value_from_cache(hand, action)
                    if best_q is None or q > best_q:
                        best_q = q
                        best_action = action

                if best_q is None:
                    best_q = 0.0

                new_values[hand] = best_q
                new_policy[hand] = best_action
                delta = max(delta, abs(best_q - self.values.get(hand, 0.0)))

            self.values.update(new_values)
            self.policy.update(new_policy)
            last_delta = delta
            if progress_every and (iteration == 1 or iteration % progress_every == 0 or delta < self.theta):
                elapsed = time.time() - t0
                print(
                    "vi_iter {}/{} delta {:.6g} states {} elapsed {:.1f}s".format(
                        iteration, self.max_iterations, delta, len(states), elapsed
                    ),
                    flush=True,
                )
            if delta < self.theta:
                break

        self.stats["plans"] += 1
        self.stats["states"] = len(states)
        self.stats["iterations"] = iteration
        self.stats["last_delta"] = last_delta

    def value(self, hand):
        hand = self.sort_hand(hand)
        if not (self.global_table_loaded and not self.allow_online_fallback):
            self.plan(hand)
        return self.values.get(hand, 0.0)

    def q_value(self, hand, action):
        hand = self.sort_hand(hand)
        action = self.sort_hand(action)
        if not (self.global_table_loaded and not self.allow_online_fallback):
            self.plan(hand)
        return self.q_value_from_cache(hand, action)

    def pass_q_value(self, hand):
        hand = self.sort_hand(hand)
        if not (self.global_table_loaded and not self.allow_online_fallback):
            self.plan(hand)
        return self.pass_reward + self.gamma * self.values.get(hand, 0.0)

    def q_value_from_cache(self, hand, action):
        next_hand = self.next_hand(hand, action)
        return self.reward(hand, action, next_hand) + self.gamma * self.values.get(next_hand, 0.0)

    def reward(self, hand, action, next_hand):
        reward = self.step_reward
        if next_hand == "":
            reward += self.terminal_reward
        return reward

    def actions(self, hand):
        hand = self.sort_hand(hand)
        if hand in self.action_cache:
            return self.action_cache[hand]

        counts = Counter(hand)
        actions = set()

        for c in CARD_ORDER:
            n = counts.get(c, 0)
            if n >= 1:
                actions.add(c)
            if n >= 2:
                actions.add(c * 2)
            if n >= 3:
                actions.add(c * 3)
            if n >= 4 and c not in ("B", "R"):
                actions.add(c * 4)

        if counts.get("B", 0) and counts.get("R", 0):
            actions.add("BR")

        self.add_trio_with_wings(actions, counts)
        self.add_four_with_wings(actions, counts)
        self.add_chains(actions, counts, need=1, min_len=5)
        self.add_chains(actions, counts, need=2, min_len=3)
        self.add_chains(actions, counts, need=3, min_len=2)
        self.add_planes_with_wings(actions, counts)

        valid = []
        for action in actions:
            action = self.sort_hand(action)
            if action and self.can_remove(hand, action):
                valid.append(action)

        valid = sorted(set(valid), key=lambda a: (-len(a), self.main_rank(a), a))
        valid = valid[: self.max_actions_per_state]
        self.action_cache[hand] = valid
        return valid

    def add_trio_with_wings(self, actions, counts):
        trio_cards = [c for c in NORMAL_CHAIN_ORDER + ["2"] if counts.get(c, 0) >= 3]
        solo_cards = [c for c in CARD_ORDER if counts.get(c, 0) >= 1]
        pair_cards = [c for c in NORMAL_CHAIN_ORDER + ["2"] if counts.get(c, 0) >= 2]

        for t in trio_cards:
            base = t * 3
            for s in solo_cards:
                if s != t:
                    actions.add(base + s)
            for p in pair_cards:
                if p != t:
                    actions.add(base + p * 2)

    def add_four_with_wings(self, actions, counts):
        bomb_cards = [c for c in NORMAL_CHAIN_ORDER + ["2"] if counts.get(c, 0) >= 4]
        solo_cards = [c for c in CARD_ORDER if counts.get(c, 0) >= 1]
        pair_cards = [c for c in NORMAL_CHAIN_ORDER + ["2"] if counts.get(c, 0) >= 2]

        for b in bomb_cards:
            base = b * 4
            solos = [c for c in solo_cards if c != b]
            pairs = [c for c in pair_cards if c != b]
            for wings in list(combinations(solos, 2))[: self.max_wing_combinations]:
                actions.add(base + "".join(wings))
            for wings in list(combinations(pairs, 2))[: self.max_wing_combinations]:
                actions.add(base + "".join(w * 2 for w in wings))

    def add_chains(self, actions, counts, need, min_len):
        run = []
        for c in NORMAL_CHAIN_ORDER:
            if counts.get(c, 0) >= need:
                run.append(c)
            else:
                self.emit_chains(actions, run, need, min_len)
                run = []
        self.emit_chains(actions, run, need, min_len)

    def emit_chains(self, actions, run, need, min_len):
        if len(run) < min_len:
            return
        for length in range(min_len, len(run) + 1):
            for start in range(0, len(run) - length + 1):
                segment = run[start : start + length]
                actions.add("".join(c * need for c in segment))

    def add_planes_with_wings(self, actions, counts):
        runs = []
        run = []
        for c in NORMAL_CHAIN_ORDER:
            if counts.get(c, 0) >= 3:
                run.append(c)
            else:
                if len(run) >= 2:
                    runs.append(run)
                run = []
        if len(run) >= 2:
            runs.append(run)

        for run in runs:
            for length in range(2, len(run) + 1):
                for start in range(0, len(run) - length + 1):
                    trio_seq = run[start : start + length]
                    base = "".join(c * 3 for c in trio_seq)
                    base_set = set(trio_seq)
                    solos = [c for c in CARD_ORDER if c not in base_set and counts.get(c, 0) >= 1]
                    pairs = [c for c in NORMAL_CHAIN_ORDER + ["2"] if c not in base_set and counts.get(c, 0) >= 2]
                    for wings in list(combinations(solos, length))[: self.max_wing_combinations]:
                        actions.add(base + "".join(wings))
                    for wings in list(combinations(pairs, length))[: self.max_wing_combinations]:
                        actions.add(base + "".join(w * 2 for w in wings))

    def next_hand(self, hand, action):
        hand = self.sort_hand(hand)
        action = self.sort_hand(action)
        counter = Counter(hand)
        for c in action:
            counter[c] -= 1
            if counter[c] <= 0:
                del counter[c]
        return "".join(c * counter.get(c, 0) for c in CARD_ORDER)

    def can_remove(self, hand, action):
        hand_counter = Counter(hand)
        action_counter = Counter(action)
        for c, n in action_counter.items():
            if hand_counter.get(c, 0) < n:
                return False
        return True

    def is_subhand(self, hand, root):
        hand_counter = Counter(hand)
        root_counter = Counter(root)
        for c, n in hand_counter.items():
            if n > root_counter.get(c, 0):
                return False
        return True

    def sort_hand(self, hand):
        return "".join(sorted(str(hand or ""), key=lambda c: INDEX[c]))

    def main_rank(self, action):
        if not action:
            return -1
        counts = Counter(action)
        max_count = max(counts.values())
        return max(INDEX[c] for c, n in counts.items() if n == max_count)

    def save_plan(self, out_dir, filename=None):
        """Save the current root hand values/policy to a JSON file."""
        if not self.root_hand:
            return None
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        filename = filename or "plan_{}.json".format(self.safe_filename(self.root_hand))
        path = os.path.join(out_dir, filename)
        obj = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "root_hand": self.root_hand,
            "params": {
                "gamma": self.gamma,
                "step_reward": self.step_reward,
                "terminal_reward": self.terminal_reward,
                "pass_reward": self.pass_reward,
                "theta": self.theta,
                "max_iterations": self.max_iterations,
                "max_states": self.max_states,
                "max_actions_per_state": self.max_actions_per_state,
                "max_wing_combinations": self.max_wing_combinations,
            },
            "stats": dict(self.stats),
            "values": dict(self.values),
            "policy": dict(self.policy),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        return path

    def load_plan(self, path):
        """Load a previously saved reduced-MDP plan."""
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.root_hand = obj.get("root_hand")
        self.global_table_loaded = obj.get("table_type") == "global"
        self.values = dict(obj.get("values", {"": 0.0}))
        self.policy = dict(obj.get("policy", {}))
        self.stats.update(obj.get("stats", {}))
        self.action_cache = {}
        return obj

    def save_table(self, out_dir):
        """Save a global value table into a directory."""
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        config = {
            "gamma": self.gamma,
            "step_reward": self.step_reward,
            "terminal_reward": self.terminal_reward,
            "pass_reward": self.pass_reward,
            "theta": self.theta,
            "max_iterations": self.max_iterations,
            "max_states": self.max_states,
            "max_actions_per_state": self.max_actions_per_state,
            "max_wing_combinations": self.max_wing_combinations,
        }
        metadata = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "table_type": "global",
            "stats": dict(self.stats),
            "num_values": len(self.values),
            "num_policy_states": len(self.policy),
        }

        self._write_json(os.path.join(out_dir, "config.json"), config)
        self._write_json(os.path.join(out_dir, "metadata.json"), metadata)
        self._write_json(os.path.join(out_dir, "values.json"), self.values)
        self._write_json(os.path.join(out_dir, "policy.json"), self.policy)
        return out_dir

    def load_table(self, table_dir):
        """Load a global value table saved by save_table()."""
        with open(os.path.join(table_dir, "values.json"), "r", encoding="utf-8") as f:
            self.values = dict(json.load(f))
        policy_path = os.path.join(table_dir, "policy.json")
        if os.path.exists(policy_path):
            with open(policy_path, "r", encoding="utf-8") as f:
                self.policy = dict(json.load(f))
        else:
            self.policy = {}
        metadata_path = os.path.join(table_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.stats.update(metadata.get("stats", {}))
        self.root_hand = None
        self.global_table_loaded = True
        self.allow_online_fallback = False
        self.action_cache = {}
        return self

    def _write_json(self, path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)

    def safe_filename(self, text):
        return "".join(c if c.isalnum() else "_" for c in str(text or "empty"))
