# nn_policy_agent.py
# ------------------------------------------------------------
# Dou Dizhu neural policy agent.
#
# Core idea:
#   1. The rule engine / environment provides legal_actions.
#   2. For each legal action a, build feature phi(infoset, a).
#   3. A small MLP outputs score(infoset, a).
#   4. Return the legal action with the highest score.
#
# This file is Python 3.6 compatible and has no torch dependency.
# It can be used for inference after training/exporting weights with
# train_selfplay_policy_gradient.py.
# ------------------------------------------------------------

import json
import math
import random
from collections import Counter

ALL_POSITIONS = ["landlord", "landlord_down", "landlord_up"]
POS_INDEX = {"landlord": 0, "landlord_down": 1, "landlord_up": 2}

# Local DouZero/RLCard rank codes.
ENV_RANKS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30]
ENV_TO_IDX = {c: i for i, c in enumerate(ENV_RANKS)}
IDX_TO_ENV = {i: c for i, c in enumerate(ENV_RANKS)}

ACTION_TYPES = [
    "PASS", "SINGLE", "PAIR", "STRAIGHT", "STRAIGHT2",
    "TRIPLET", "TRIPLET1", "TRIPLET2", "BOMB",
    "QUAD2", "QUAD4", "PLANE", "PLANE1", "PLANE2",
    "ROCKET", "OTHER"
]
ACTION_TYPE_INDEX = {t: i for i, t in enumerate(ACTION_TYPES)}


def concrete_to_env(card):
    """Botzone concrete card id 0..53 -> local rank code."""
    card = int(card)
    level = card // 4 + (1 if card == 53 else 0)
    if level <= 11:
        return level + 3
    if level == 12:
        return 17
    if level == 13:
        return 20
    if level == 14:
        return 30
    return 3


def card_to_idx(card, concrete=False):
    if concrete:
        card = concrete_to_env(card)
    return ENV_TO_IDX.get(int(card), None)


def count_vec(cards, concrete=False):
    vec = [0.0] * 15
    if cards is None:
        return vec
    for c in list(cards):
        idx = card_to_idx(c, concrete=concrete)
        if idx is not None:
            vec[idx] += 1.0
    # Normalize count. Normal cards max 4; jokers max 1, but /4 is still fine.
    return [x / 4.0 for x in vec]


def raw_count(cards, concrete=False):
    cnt = [0] * 15
    if cards is None:
        return cnt
    for c in list(cards):
        idx = card_to_idx(c, concrete=concrete)
        if idx is not None:
            cnt[idx] += 1
    return cnt


def _is_consecutive(indices):
    if not indices:
        return False
    return indices == list(range(indices[0], indices[0] + len(indices)))


def detect_action_type(action, concrete=False):
    """Detect a coarse Dou Dizhu action type from rank/concrete cards."""
    if not action:
        return "PASS", -1
    cnt = raw_count(action, concrete=concrete)
    n = sum(cnt)
    nonzero = [i for i, c in enumerate(cnt) if c > 0]
    vals = [cnt[i] for i in nonzero]
    main_rank = max(nonzero) if nonzero else -1

    if n == 2 and cnt[13] == 1 and cnt[14] == 1:
        return "ROCKET", 14
    if n == 1:
        return "SINGLE", main_rank
    if n == 2 and len(nonzero) == 1 and vals[0] == 2:
        return "PAIR", main_rank
    if n == 3 and len(nonzero) == 1 and vals[0] == 3:
        return "TRIPLET", main_rank
    if n == 4 and len(nonzero) == 1 and vals[0] == 4:
        return "BOMB", main_rank

    # Straights cannot contain 2 or jokers, so indices must be <= 11.
    if n >= 5 and all(cnt[i] == 1 for i in nonzero) and max(nonzero) <= 11 and _is_consecutive(nonzero):
        return "STRAIGHT", max(nonzero)
    if n >= 6 and n % 2 == 0 and all(cnt[i] == 2 for i in nonzero) and max(nonzero) <= 11 and _is_consecutive(nonzero):
        return "STRAIGHT2", max(nonzero)

    triple_ranks = [i for i, c in enumerate(cnt) if c == 3]
    pair_ranks = [i for i, c in enumerate(cnt) if c == 2]
    single_ranks = [i for i, c in enumerate(cnt) if c == 1]
    quad_ranks = [i for i, c in enumerate(cnt) if c == 4]

    if len(triple_ranks) == 1:
        if n == 4:
            return "TRIPLET1", triple_ranks[0]
        if n == 5 and len(pair_ranks) == 1:
            return "TRIPLET2", triple_ranks[0]

    if len(quad_ranks) == 1:
        if n == 6:
            return "QUAD2", quad_ranks[0]
        if n == 8 and len(pair_ranks) == 2:
            return "QUAD4", quad_ranks[0]

    if len(triple_ranks) >= 2 and max(triple_ranks) <= 11 and _is_consecutive(triple_ranks):
        k = len(triple_ranks)
        if n == 3 * k:
            return "PLANE", max(triple_ranks)
        if n == 4 * k and len(single_ranks) == k:
            return "PLANE1", max(triple_ranks)
        if n == 5 * k and len(pair_ranks) == k:
            return "PLANE2", max(triple_ranks)

    return "OTHER", main_rank


def same_team(a, b):
    if a == b:
        return True
    if a != "landlord" and b != "landlord":
        return True
    return False


def position_onehot(position):
    out = [0.0, 0.0, 0.0]
    if position in POS_INDEX:
        out[POS_INDEX[position]] = 1.0
    return out


def get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def get_num_cards_left(infoset, position):
    hand = get_attr(infoset, "player_hand_cards", []) or []
    default = {"landlord": 20, "landlord_down": 17, "landlord_up": 17}
    default[position] = len(hand)

    for attr in ["num_cards_left", "num_cards_left_dict", "player_num_cards_left"]:
        raw = get_attr(infoset, attr, None)
        if raw is None:
            continue
        if isinstance(raw, dict):
            out = default.copy()
            for k, v in raw.items():
                if k in out:
                    out[k] = int(v)
                else:
                    try:
                        idx = int(k)
                        if 0 <= idx < 3:
                            out[ALL_POSITIONS[idx]] = int(v)
                    except Exception:
                        pass
            return out
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            return {ALL_POSITIONS[i]: int(raw[i]) for i in range(3)}
    return default


def infer_last_pid(infoset, position):
    raw = get_attr(infoset, "last_pid", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw if raw in POS_INDEX else None
    try:
        i = int(raw)
        if 0 <= i < 3:
            return ALL_POSITIONS[i]
    except Exception:
        pass
    return None


def feature_vector(infoset, position, action, concrete=False):
    """Build a fixed-size feature vector phi(s, a)."""
    my_hand = get_attr(infoset, "player_hand_cards", []) or []
    last_move = get_attr(infoset, "last_move", []) or []
    last_two = get_attr(infoset, "last_two_moves", [[], []]) or [[], []]
    if len(last_two) < 2:
        last_two = ([[]] * (2 - len(last_two))) + list(last_two)
    last_two = list(last_two)[-2:]

    num_left = get_num_cards_left(infoset, position)
    last_pid = infer_last_pid(infoset, position)
    leading = 1.0 if (not last_move and (not last_two or all(len(x) == 0 for x in last_two))) else 0.0

    a_type, main_rank = detect_action_type(action, concrete=concrete)
    a_type_oh = [0.0] * len(ACTION_TYPES)
    a_type_oh[ACTION_TYPE_INDEX.get(a_type, ACTION_TYPE_INDEX["OTHER"])] = 1.0

    action_len = len(action or [])
    remaining_after = max(0, len(my_hand) - action_len)
    is_bomb = 1.0 if a_type == "BOMB" else 0.0
    is_rocket = 1.0 if a_type == "ROCKET" else 0.0
    can_finish = 1.0 if action_len > 0 and action_len == len(my_hand) else 0.0

    last_pid_oh = [0.0, 0.0, 0.0, 0.0]  # landlord/down/up/none
    if last_pid in POS_INDEX:
        last_pid_oh[POS_INDEX[last_pid]] = 1.0
    else:
        last_pid_oh[3] = 1.0

    teammate_danger = 0.0
    enemy_danger = 0.0
    for p in ALL_POSITIONS:
        if p == position:
            continue
        n = num_left.get(p, 17)
        if n <= 2:
            if same_team(p, position):
                teammate_danger = max(teammate_danger, (3 - n) / 2.0)
            else:
                enemy_danger = max(enemy_danger, (3 - n) / 2.0)

    # A rough estimate of how many control cards this action spends.
    cnt_action = raw_count(action, concrete=concrete)
    control_cost = (
        0.25 * cnt_action[11] +  # A
        0.65 * cnt_action[12] +  # 2
        1.00 * cnt_action[13] +  # small joker
        1.15 * cnt_action[14]    # big joker
    ) / 2.0

    feat = []
    feat.extend(position_onehot(position))
    feat.extend(count_vec(my_hand, concrete=concrete))
    feat.extend(count_vec(last_move, concrete=concrete))
    feat.extend(count_vec(last_two[0], concrete=concrete))
    feat.extend(count_vec(last_two[1], concrete=concrete))
    feat.extend(count_vec(action, concrete=concrete))
    feat.extend([num_left.get(p, 0) / 20.0 for p in ALL_POSITIONS])
    feat.extend([leading])
    feat.extend(last_pid_oh)
    feat.extend([teammate_danger, enemy_danger])
    feat.extend([action_len / 20.0, remaining_after / 20.0])
    feat.extend(a_type_oh)
    feat.extend([main_rank / 14.0 if main_rank >= 0 else 0.0])
    feat.extend([is_bomb, is_rocket, can_finish, control_cost])
    return feat


def feature_dim():
    dummy = {
        "player_hand_cards": [],
        "last_move": [],
        "last_two_moves": [[], []],
        "num_cards_left": {"landlord": 20, "landlord_down": 17, "landlord_up": 17},
        "last_pid": None,
    }
    return len(feature_vector(dummy, "landlord", []))


class PureMLP(object):
    """Tiny pure-Python MLP for Botzone/Python 3.6 inference."""

    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # New format: one independent network per role.
        # {
        #   "role_models": {
        #       "landlord": {"weights": ..., "biases": ...},
        #       "landlord_down": {"weights": ..., "biases": ...},
        #       "landlord_up": {"weights": ..., "biases": ...}
        #   }
        # }
        if isinstance(obj, dict) and "role_models" in obj:
            models = {}
            for role, sub in obj["role_models"].items():
                models[role] = cls(sub["weights"], sub["biases"])
            return models

        # Old shared-policy format, kept for compatibility.
        return cls(obj["weights"], obj["biases"])

    def _linear(self, x, w, b):
        # w shape: out_dim x in_dim
        out = []
        for i in range(len(w)):
            s = b[i]
            row = w[i]
            # Manual dot product for Python 3.6 / no numpy.
            for j in range(len(x)):
                s += row[j] * x[j]
            out.append(s)
        return out

    def score(self, x):
        h = list(x)
        for layer in range(len(self.weights)):
            h = self._linear(h, self.weights[layer], self.biases[layer])
            if layer != len(self.weights) - 1:
                h = [v if v > 0.0 else 0.0 for v in h]
        return float(h[0])


class NeuralPolicyAgent(object):
    """DouZero/RLCard-style agent: action = agent.act(infoset)."""

    def __init__(self, position, weights_path=None, fallback_agent=None, concrete=False, seed=None):
        self.name = "NeuralPolicyAgent"
        self.position = position
        self.concrete = concrete
        self.rng = random.Random(seed)
        self.fallback_agent = fallback_agent

        # self.model is used for old shared-policy weights.
        # self.models is used for new role-specific weights.
        self.model = None
        self.models = None

        self.last_error = None
        if weights_path:
            loaded = PureMLP.from_json(weights_path)
            if isinstance(loaded, dict):
                self.models = loaded
            else:
                self.model = loaded

    def get_model(self):
        """Return the network corresponding to this agent's role.

        This keeps the public interface unchanged:
            NeuralPolicyAgent(position, weights_path)
        If the weights file is role-specific, the correct sub-network is selected
        automatically by self.position. If the weights file is old/shared, the
        shared model is used.
        """
        if self.models is not None:
            if self.position in self.models:
                return self.models[self.position]
            # Safety fallbacks.
            if self.position != "landlord" and "farmer" in self.models:
                return self.models["farmer"]
            if "shared" in self.models:
                return self.models["shared"]
            if self.models:
                return list(self.models.values())[0]
            return None
        return self.model

    def action_scores(self, infoset, legal_actions):
        """Return [(score, action), ...] for all legal actions.

        This is still only a policy scorer. It does not search or rollout.
        """
        model = self.get_model()
        if model is None:
            return [(0.0, a) for a in legal_actions]
        out = []
        hand = get_attr(infoset, "player_hand_cards", []) or []
        for action in legal_actions:
            x = feature_vector(infoset, self.position, action, concrete=self.concrete)
            s = model.score(x)
            # A small deterministic safety bonus; this is not search.
            if action and len(action) == len(hand):
                s += 1000.0
            out.append((s, action))
        return out

    def act(self, infoset):
        legal_actions = get_attr(infoset, "legal_actions", []) or []
        if not legal_actions:
            return []
        if len(legal_actions) == 1:
            return legal_actions[0]

        if self.get_model() is None:
            return self.heuristic_action(infoset, legal_actions)

        try:
            scored = self.action_scores(infoset, legal_actions)
            best_score = max(s for s, _ in scored)
            best_actions = [a for s, a in scored if s == best_score]
            return self.tie_break(best_actions)
        except Exception as e:
            self.last_error = repr(e)
            if self.fallback_agent is not None:
                return self.fallback_agent.act(infoset)
            return self.heuristic_action(infoset, legal_actions)

    def tie_break(self, actions):
        # Prefer shorter/lower-control among equal scores.
        def key(a):
            t, main = detect_action_type(a, concrete=self.concrete)
            bomb_cost = 10 if t in ("BOMB", "ROCKET") and len(a) > 0 else 0
            return (bomb_cost, len(a), main)
        actions = list(actions)
        actions.sort(key=key)
        return actions[0]

    def heuristic_action(self, infoset, legal_actions):
        # Safe fallback: finish if possible; pass to teammate; otherwise cheapest non-bomb response.
        hand = get_attr(infoset, "player_hand_cards", []) or []
        for a in legal_actions:
            if a and len(a) == len(hand):
                return a
        if [] in legal_actions:
            last_pid = infer_last_pid(infoset, self.position)
            if last_pid is not None and same_team(last_pid, self.position):
                return []
        non_pass = [a for a in legal_actions if a]
        if not non_pass:
            return [] if [] in legal_actions else legal_actions[0]
        non_bomb = []
        for a in non_pass:
            t, _ = detect_action_type(a, concrete=self.concrete)
            if t not in ("BOMB", "ROCKET"):
                non_bomb.append(a)
        candidates = non_bomb if non_bomb else non_pass
        return self.tie_break(candidates)


# Backward-compatible alias.
NeuralActionAgent = NeuralPolicyAgent
