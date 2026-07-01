# valueiteration_agent.py
# ------------------------------------------------------------
# Value-table agent.
#
# Decision rule stays pure:
#     Q(s, a) = R(s, a, s') + gamma * V(s')
#     action = argmax_a Q(s, a)
#
# The state key can be either:
#   1. hand mode:    s = current hand string
#   2. feature mode: s = observable feature key built from infoset
#
# Feature mode uses a coarse observable state. It keeps public context that
# matters for decisions, but buckets it to avoid the exact-state explosion.
# ------------------------------------------------------------

import random
from collections import Counter

try:
    from douzero.evaluation.value_iteration_planner import ValueIterationPlanner, INDEX
except Exception:
    from value_iteration_planner import ValueIterationPlanner, INDEX


ENV_CARD_TO_REAL = {
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
    17: "2",
    20: "B",
    30: "R",
}

ALL_POSITIONS = ["landlord", "landlord_down", "landlord_up"]


class ValueIterationAgent(object):
    """Choose legal actions by maximizing Bellman score from a loaded V table."""

    def __init__(self, position, planner=None, table_path=None, debug=False,
                 state_mode="auto", missing_value=None):
        self.name = "ValueTableAgent"
        self.position = position
        self.debug = debug
        self.state_mode = state_mode
        self.missing_value = None if missing_value is None else float(missing_value)
        self.planner = planner or ValueIterationPlanner()
        if table_path:
            self.planner.load_table(table_path)
        self.last_error = None
        self.stats = {
            "acts": 0,
            "fallbacks": 0,
            "table_hits": 0,
            "table_misses": 0,
            "feature_hits": 0,
            "hand_hits": 0,
        }

    def act(self, infoset):
        self.stats["acts"] += 1
        try:
            legal_actions = getattr(infoset, "legal_actions", []) or []
            if not legal_actions:
                return []
            if len(legal_actions) == 1:
                return legal_actions[0]

            hand = self.cards_to_str(getattr(infoset, "player_hand_cards", []))
            mode = self.active_state_mode()

            best_q = -float("inf")
            best_actions = []
            for action in legal_actions:
                q = self.action_q(infoset, hand, action, mode)
                if q > best_q:
                    best_q = q
                    best_actions = [action]
                elif q == best_q:
                    best_actions.append(action)

            return self.tie_break(best_actions) if best_actions else random.choice(legal_actions)
        except Exception as exc:
            self.stats["fallbacks"] += 1
            self.last_error = repr(exc)
            legal_actions = getattr(infoset, "legal_actions", []) or []
            return random.choice(legal_actions) if legal_actions else []

    def active_state_mode(self):
        if self.state_mode in ("hand", "feature"):
            return self.state_mode
        for key in self.planner.values:
            if isinstance(key, str) and key.startswith("feat_v"):
                return "feature"
        return "hand"

    def action_q(self, infoset, hand, action, mode):
        if action == []:
            value = self.lookup_value(infoset, hand, mode)
            return self.planner.pass_reward + self.planner.gamma * value

        action_str = self.cards_to_str(action)
        if not action_str or not self.planner.can_remove(hand, action_str):
            return -float("inf")
        next_hand = self.planner.next_hand(hand, action_str)
        value = self.lookup_value(infoset, next_hand, mode)
        return self.planner.reward(hand, action_str, next_hand) + self.planner.gamma * value

    def lookup_value(self, infoset, hand, mode, last_move_override=None):
        if mode == "feature":
            for key in self.feature_state_keys(infoset, hand, last_move_override=last_move_override):
                if key in self.planner.values:
                    self.stats["table_hits"] += 1
                    self.stats["feature_hits"] += 1
                    return self.planner.values[key]
            self.stats["table_misses"] += 1
            return self.fallback_value(hand)

        hand = self.planner.sort_hand(hand)
        if hand in self.planner.values:
            self.stats["table_hits"] += 1
            self.stats["hand_hits"] += 1
            return self.planner.values[hand]
        self.stats["table_misses"] += 1
        return self.fallback_value(hand)

    def fallback_value(self, hand):
        if self.missing_value is not None:
            return self.missing_value
        return -float(len(self.planner.sort_hand(hand)))

    def feature_state_key(self, infoset, hand, last_move_override=None):
        return self.feature_state_keys(infoset, hand, last_move_override=last_move_override)[0]

    def feature_state_keys(self, infoset, hand, last_move_override=None):
        hand = self.planner.sort_hand(hand)
        last_move = last_move_override
        if last_move is None:
            last_move = self.cards_to_str(getattr(infoset, "last_move", []))
        last_two = self.last_two_moves(infoset)
        num_left = self.num_cards_left(infoset, hand)
        played_counts = self.played_counts(infoset)
        last_pid = self.normalize_position(getattr(infoset, "last_pid", None))

        exact = "|".join([
            "feat_v3",
            "pos={}".format(self.position),
            "hand={}".format(self.hand_feature_key(hand)),
            "lead={}".format(1 if self.is_leading_round(last_move, last_two) else 0),
            "last={}".format(self.move_feature_key(last_move)),
            "actor={}".format(self.actor_relation(last_pid)),
            "enemy={}".format(self.enemy_left_bucket(num_left)),
            "team={}".format(self.teammate_left_bucket(num_left)),
            "played={}".format(self.played_summary_key(played_counts)),
        ])
        base = self.base_feature_state_key(hand)
        return [exact, base]

    def base_feature_state_key(self, hand):
        hand = self.planner.sort_hand(hand)
        return "|".join([
            "feat_v3_base",
            "pos={}".format(self.position),
            "hand={}".format(self.hand_feature_key(hand)),
        ])

    def tie_break(self, actions):
        if not actions:
            return []
        return sorted(actions, key=lambda a: (self.cards_to_str(a), tuple(a)))[0]

    def cards_to_str(self, cards):
        out = []
        if cards is None:
            return ""
        if isinstance(cards, str):
            out = [c for c in cards if c in INDEX]
        else:
            for card in cards:
                if card in ENV_CARD_TO_REAL:
                    out.append(ENV_CARD_TO_REAL[card])
                elif isinstance(card, str) and card in INDEX:
                    out.append(card)
        return "".join(sorted(out, key=lambda c: INDEX[c]))

    def count_key(self, hand):
        counts = Counter(hand)
        return ",".join(str(counts.get(c, 0)) for c in INDEX)

    def hand_feature_key(self, hand):
        """Coarse hand abstraction used as feature-state key.

        This keeps the value table small enough to hit during evaluation. It
        intentionally describes hand structure instead of the exact cards.
        """
        counts = Counter(hand)
        singles = 0
        pairs = 0
        trios = 0
        bombs = 0
        for card, count in counts.items():
            if card in ("B", "R"):
                continue
            if count == 1:
                singles += 1
            elif count == 2:
                pairs += 1
            elif count == 3:
                trios += 1
            elif count >= 4:
                bombs += 1
        rocket = 1 if counts.get("B", 0) and counts.get("R", 0) else 0
        control = (
            counts.get("A", 0)
            + 2 * counts.get("2", 0)
            + 3 * counts.get("B", 0)
            + 3 * counts.get("R", 0)
        )
        control_bucket = self.bucket(control, [1, 3, 5, 8])
        chain_bucket = self.bucket(self.longest_chain(counts), [4, 5, 7, 9])
        length_bucket = self.bucket(len(hand), [1, 2, 5, 10, 15])
        return ",".join(str(x) for x in (
            length_bucket,
            singles,
            pairs,
            trios,
            bombs,
            rocket,
            control_bucket,
            chain_bucket,
        ))

    def move_feature_key(self, move):
        move = self.planner.sort_hand(move)
        if not move:
            return "none"
        counts = Counter(move)
        nonzero = [c for c in INDEX if counts.get(c, 0)]
        length_bucket = self.bucket(len(move), [1, 2, 3, 5, 8])
        main_rank = max((INDEX[c] for c in nonzero), default=0)
        rank_bucket = self.bucket(main_rank, [6, 10, 12, 14])

        if len(move) == 2 and counts.get("B", 0) and counts.get("R", 0):
            move_type = "rocket"
        elif len(move) == 4 and any(n == 4 for n in counts.values()):
            move_type = "bomb"
        elif len(move) == 1:
            move_type = "single"
        elif len(move) == 2 and len(nonzero) == 1:
            move_type = "pair"
        elif len(move) == 3 and len(nonzero) == 1:
            move_type = "trio"
        elif self.is_chain(counts, need=1, min_len=5):
            move_type = "chain"
        elif self.is_chain(counts, need=2, min_len=3):
            move_type = "pair_chain"
        elif self.is_chain(counts, need=3, min_len=2):
            move_type = "plane"
        else:
            move_type = "other"
        return "{},{},{}".format(move_type, length_bucket, rank_bucket)

    def is_chain(self, counts, need, min_len):
        ranks = [
            INDEX[c] for c in ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
            if counts.get(c, 0) == need
        ]
        if len(ranks) < min_len:
            return False
        return ranks == list(range(ranks[0], ranks[0] + len(ranks)))

    def actor_relation(self, last_pid):
        if last_pid is None:
            return "none"
        if last_pid == self.position:
            return "self"
        if self.same_team(last_pid, self.position):
            return "team"
        return "enemy"

    def enemy_left_bucket(self, num_left):
        values = [num_left.get(p, 17) for p in self.enemy_positions()]
        return self.bucket(min(values) if values else 17, [1, 2, 5, 10])

    def teammate_left_bucket(self, num_left):
        teammate = self.teammate_position()
        if teammate is None:
            return "none"
        return str(self.bucket(num_left.get(teammate, 17), [1, 2, 5, 10]))

    def played_summary_key(self, played_counts):
        control = 0
        for card in ("A", "2", "B", "R"):
            control += played_counts[INDEX[card]]
        bombs_seen = 0
        for card in INDEX:
            if card not in ("B", "R") and played_counts[INDEX[card]] >= 4:
                bombs_seen += 1
        total = sum(played_counts)
        return "{},{},{}".format(
            self.bucket(control, [1, 3, 5, 8]),
            self.bucket(bombs_seen, [0, 1, 2]),
            self.bucket(total, [5, 15, 30, 45]),
        )

    def bucket(self, value, cuts):
        for i, cut in enumerate(cuts):
            if value <= cut:
                return i
        return len(cuts)

    def longest_chain(self, counts):
        order = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        best = 0
        run = 0
        for card in order:
            if counts.get(card, 0) >= 1:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best

    def last_two_moves(self, infoset):
        raw = getattr(infoset, "last_two_moves", [[], []]) or [[], []]
        out = [self.cards_to_str(move) for move in list(raw)[-2:]]
        while len(out) < 2:
            out.insert(0, "")
        return out

    def num_cards_left(self, infoset, hand):
        default = {"landlord": 20, "landlord_down": 17, "landlord_up": 17}
        default[self.position] = len(hand)
        for attr in ["num_cards_left", "num_cards_left_dict", "player_num_cards_left"]:
            raw = getattr(infoset, attr, None)
            if raw is None:
                continue
            if isinstance(raw, dict):
                out = default.copy()
                for key, value in raw.items():
                    pos = self.normalize_position(key)
                    if pos in out:
                        out[pos] = int(value)
                out[self.position] = len(hand)
                return out
            if isinstance(raw, (list, tuple)):
                out = default.copy()
                for i, value in enumerate(raw[:3]):
                    out[ALL_POSITIONS[i]] = int(value)
                out[self.position] = len(hand)
                return out
        return default

    def played_counts(self, infoset):
        counts = Counter()
        raw = getattr(infoset, "played_cards", None)
        seq = getattr(infoset, "card_play_action_seq", None)
        if isinstance(raw, dict):
            for cards in raw.values():
                counts.update(self.cards_to_str(cards))
        elif isinstance(raw, list):
            for item in raw:
                cards = item if isinstance(item, list) else [item]
                counts.update(self.cards_to_str(cards))
        elif seq is not None:
            for action in seq:
                counts.update(self.cards_to_str(action))
        else:
            for move in self.last_two_moves(infoset):
                counts.update(move)
        return [counts.get(c, 0) for c in INDEX]

    def normalize_position(self, value):
        if value in ALL_POSITIONS:
            return value
        try:
            idx = int(value)
            if 0 <= idx < len(ALL_POSITIONS):
                return ALL_POSITIONS[idx]
        except Exception:
            pass
        return None

    def same_team(self, a, b):
        if a == b:
            return True
        return a != "landlord" and b != "landlord"

    def enemy_positions(self):
        if self.position == "landlord":
            return ["landlord_down", "landlord_up"]
        return ["landlord"]

    def teammate_position(self):
        if self.position == "landlord_down":
            return "landlord_up"
        if self.position == "landlord_up":
            return "landlord_down"
        return None

    def is_leading_round(self, last_move, last_two):
        if last_move:
            return False
        return all(move == "" for move in last_two)


ValueDPAgent = ValueIterationAgent
MemoizedValueDPAgent = ValueIterationAgent
