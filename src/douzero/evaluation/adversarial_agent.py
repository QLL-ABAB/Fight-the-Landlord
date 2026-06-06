import random
import math
from collections import Counter

from rlcard.games.doudizhu.utils import CARD_TYPE


EnvCard2RealCard = {
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

RealCard2EnvCard = {
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
    "2": 17,
    "B": 20,
    "R": 30,
}

INDEX = {
    "3": 0,
    "4": 1,
    "5": 2,
    "6": 3,
    "7": 4,
    "8": 5,
    "9": 6,
    "T": 7,
    "J": 8,
    "Q": 9,
    "K": 10,
    "A": 11,
    "2": 12,
    "B": 13,
    "R": 14,
}

CARD_ORDER = [
    "3", "4", "5", "6", "7", "8", "9", "T",
    "J", "Q", "K", "A", "2", "B", "R"
]

NORMAL_CHAIN_ORDER = [
    "3", "4", "5", "6", "7", "8", "9", "T",
    "J", "Q", "K", "A"
]

ALL_POSITIONS = ["landlord", "landlord_down", "landlord_up"]


class AdversarialSearchAgent(object):
    """
    Sampled Team Minimax Agent for Dou Dizhu.

    核心思想：
    1. 斗地主是不完全信息博弈，所以先对未知手牌采样；
    2. 每个采样得到一个完整假想状态；
    3. 在完整假想状态上做有限深度 team minimax；
    4. 同阵营玩家取 max，敌对阵营玩家取 min；
    5. 多个采样结果取平均，选择平均价值最高的动作。

    接口：
        agent = AdversarialSearchAgent(position)
        action = agent.act(infoset)
    """

    def __init__(self, position, debug=False):
        self.name = "AdversarialSearchAgent"
        self.position = position
        self.debug = debug
        self.fallback_count = 0
        self.last_error = None

        self.cfg = {
            # 搜索参数
            "num_samples": 6,
            "search_depth": 3,
            "max_root_actions": 18,
            "max_actions_per_node": 10,

            # 终局分数
            "terminal_score": 100000.0,

            # 评价函数参数：地主视角
            "landlord_count_weight": 16.0,
            "farmer_count_weight": 8.0,
            "landlord_badness_weight": 2.0,
            "farmer_badness_weight": 1.2,
            "near_win_bonus": 80.0,
            "near_win_threshold": 2,
            "initiative_bonus": 14.0,
            "initiative_penalty": 10.0,
            "control_weight": 3.0,
            "bomb_weight": 8.0,

            # pruning / move ordering 参数
            "local_hand_improve_weight": 8.0,
            "local_action_len_weight": 0.8,
            "local_min_card_bonus": 4.0,
            "local_single_rank_penalty": 0.25,
            "local_control_cost_weight": 1.2,
            "local_bomb_penalty": 30.0,
            "local_pass_score": -3.0,

            # hand_badness 参数
            "badness_turn_weight": 7.0,
            "badness_single_weight": 3.0,
            "badness_len_weight": 1.2,
            "badness_chain_discount_weight": 5.0,
            "badness_control_discount": 2.2,
            "badness_bomb_discount": 3.0,

            # chain discount
            "solo_chain_coef": 1.0,
            "pair_chain_coef": 1.5,
            "trio_chain_coef": 2.0,

            # 动作成本
            "cost_len_weight": 0.5,
            "cost_rank_weight": 0.08,
            "cost_bomb_weight": 50.0,

            # 控制牌成本
            "A_cost": 1.5,
            "2_cost": 4.5,
            "B_cost": 7.0,
            "R_cost": 8.0,
        }

        self._all_action_cache = None
        self._legal_cache = {}

    # ============================================================
    # 0. 主入口
    # ============================================================

    def act(self, infoset):
        try:
            legal_actions = getattr(infoset, "legal_actions", [])

            if not legal_actions:
                return []

            root_state = self.extract_root_state(infoset)

            # 有直接出完的动作，直接出完
            finish_actions = [
                a for a in legal_actions
                if a != [] and len(a) == root_state["my_count"]
            ]
            if finish_actions:
                return self.choose_lowest_cost_env_action(finish_actions)

            root_actions = self.prune_root_actions(legal_actions, root_state)

            if not root_actions:
                return random.choice(legal_actions)

            action_values = {tuple(a): 0.0 for a in root_actions}
            action_counts = {tuple(a): 0 for a in root_actions}

            for _ in range(self.cfg["num_samples"]):
                sampled_state = self.sample_full_state(infoset, root_state)

                for action in root_actions:
                    action_key = tuple(action)
                    action_str = self.env_cards_to_real_str(action)

                    next_state = self.apply_action(
                        sampled_state,
                        self.position,
                        action_str
                    )

                    next_player = self.next_position(self.position)

                    value = self.minimax(
                        state=next_state,
                        depth=self.cfg["search_depth"] - 1,
                        current_position=next_player,
                        root_position=self.position,
                        alpha=-float("inf"),
                        beta=float("inf"),
                    )

                    action_values[action_key] += value
                    action_counts[action_key] += 1

            best_value = -float("inf")
            best_actions = []

            for action in root_actions:
                key = tuple(action)
                if action_counts[key] == 0:
                    avg_value = -float("inf")
                else:
                    avg_value = action_values[key] / action_counts[key]

                if avg_value > best_value:
                    best_value = avg_value
                    best_actions = [action]
                elif avg_value == best_value:
                    best_actions.append(action)

            chosen = self.choose_lowest_cost_env_action(best_actions)

            if chosen not in legal_actions:
                chosen = random.choice(legal_actions)

            return chosen

        except Exception as e:
            self.fallback_count += 1
            self.last_error = repr(e)

            legal_actions = getattr(infoset, "legal_actions", [])
            if legal_actions:
                return random.choice(legal_actions)
            return []

    # ============================================================
    # 1. root 状态提取
    # ============================================================

    def extract_root_state(self, infoset):
        my_hand = self.env_cards_to_real_str(
            getattr(infoset, "player_hand_cards", [])
        )

        last_move = self.env_cards_to_real_str(
            getattr(infoset, "last_move", [])
        )

        last_two_moves = self.get_last_two_moves(infoset)
        last_pid = self.normalize_position(getattr(infoset, "last_pid", None))

        leading_round = self.is_leading_round(last_move, last_two_moves)
        num_cards_left = self.get_num_cards_left(infoset)

        if self.position not in num_cards_left:
            num_cards_left[self.position] = len(my_hand)

        pass_count = self.infer_pass_count(last_two_moves, leading_round)

        if leading_round:
            last_move_for_search = ""
            last_pid_for_search = None
            pass_count = 0
        else:
            last_move_for_search = last_move
            last_pid_for_search = last_pid

        return {
            "my_hand": my_hand,
            "my_count": len(my_hand),
            "last_move": last_move_for_search,
            "last_two_moves": last_two_moves,
            "last_pid": last_pid_for_search,
            "leading_round": leading_round,
            "pass_count": pass_count,
            "num_cards_left": num_cards_left,
        }

    def get_last_two_moves(self, infoset):
        raw = getattr(infoset, "last_two_moves", [[], []])
        result = []

        for move in raw:
            result.append(self.env_cards_to_real_str(move))

        while len(result) < 2:
            result.append("")

        return result[:2]

    def is_leading_round(self, last_move, last_two_moves):
        return last_two_moves[0] == "" and last_two_moves[1] == ""

    def infer_pass_count(self, last_two_moves, leading_round):
        if leading_round:
            return 0

        count = 0
        for move in reversed(last_two_moves):
            if move == "":
                count += 1
            else:
                break

        if count >= 2:
            return 0

        return count

    def get_num_cards_left(self, infoset):
        for attr in ["num_cards_left", "num_cards_left_dict", "player_num_cards_left"]:
            raw = getattr(infoset, attr, None)

            if raw is None:
                continue

            if isinstance(raw, dict):
                result = {}
                for k, v in raw.items():
                    pos = self.normalize_position(k)
                    if pos is not None:
                        result[pos] = int(v)
                return result

            if isinstance(raw, (list, tuple)):
                result = {}
                for i, p in enumerate(ALL_POSITIONS):
                    if i < len(raw):
                        result[p] = int(raw[i])
                return result

        return {}

    # ============================================================
    # 2. 采样完整状态
    # ============================================================

    def sample_full_state(self, infoset, root_state):
        """
        对未知手牌做一次 determinization。

        返回 state:
        {
            "hands": {
                "landlord": "...",
                "landlord_down": "...",
                "landlord_up": "..."
            },
            "last_move": "...",
            "last_pid": "...",
            "pass_count": int
        }
        """
        root_pos = self.position
        my_hand = root_state["my_hand"]

        unknown_counter = self.get_unknown_cards(infoset, my_hand)
        unknown_cards = []

        for c in CARD_ORDER:
            unknown_cards.extend([c] * unknown_counter[c])

        random.shuffle(unknown_cards)

        num_cards_left = dict(root_state["num_cards_left"])
        num_cards_left[root_pos] = len(my_hand)

        other_positions = [p for p in ALL_POSITIONS if p != root_pos]

        needed = {}
        known_sum = 0
        missing = []

        for p in other_positions:
            if p in num_cards_left:
                needed[p] = int(num_cards_left[p])
                known_sum += needed[p]
            else:
                missing.append(p)

        remaining_unknown = len(unknown_cards) - known_sum

        if missing:
            remaining_unknown = max(0, remaining_unknown)
            base = remaining_unknown // len(missing)
            extra = remaining_unknown % len(missing)

            for i, p in enumerate(missing):
                needed[p] = base + (1 if i < extra else 0)

        self.adjust_needed_counts(needed, len(unknown_cards))

        hands = {
            root_pos: my_hand
        }

        index = 0
        for p in other_positions:
            cnt = needed.get(p, 0)
            cards = unknown_cards[index:index + cnt]
            index += cnt
            hands[p] = self.sort_card_str("".join(cards))

        for p in ALL_POSITIONS:
            if p not in hands:
                hands[p] = ""

        return {
            "hands": hands,
            "last_move": root_state["last_move"],
            "last_pid": root_state["last_pid"],
            "pass_count": root_state["pass_count"],
        }

    def adjust_needed_counts(self, needed, total_cards):
        current = sum(needed.values())

        if not needed:
            return

        positions = list(needed.keys())

        if current > total_cards:
            diff = current - total_cards

            while diff > 0:
                p = max(positions, key=lambda x: needed[x])
                if needed[p] > 0:
                    needed[p] -= 1
                    diff -= 1
                else:
                    break

        elif current < total_cards:
            diff = total_cards - current
            i = 0

            while diff > 0:
                p = positions[i % len(positions)]
                needed[p] += 1
                diff -= 1
                i += 1

    def get_unknown_cards(self, infoset, my_hand):
        deck = Counter()

        for card in CARD_ORDER:
            deck[card] = 1 if card in ["B", "R"] else 4

        for c in my_hand:
            deck[c] -= 1

        played = self.extract_played_cards(infoset)
        for c in played:
            deck[c] -= 1

        for c in list(deck.keys()):
            if deck[c] < 0:
                deck[c] = 0

        return deck

    def extract_played_cards(self, infoset):
        played = []

        seq = getattr(infoset, "card_play_action_seq", None)
        if seq is not None:
            for action in seq:
                played.extend(self.env_cards_to_real_list(action))
            return played

        raw = getattr(infoset, "played_cards", None)
        if raw is not None:
            if isinstance(raw, dict):
                for cards in raw.values():
                    played.extend(self.env_cards_to_real_list(cards))
            elif isinstance(raw, list):
                for item in raw:
                    if isinstance(item, list):
                        played.extend(self.env_cards_to_real_list(item))
                    else:
                        played.extend(self.env_cards_to_real_list([item]))
            return played

        last_two = getattr(infoset, "last_two_moves", [[], []])
        for move in last_two:
            played.extend(self.env_cards_to_real_list(move))

        return played

    # ============================================================
    # 3. Team Minimax
    # ============================================================

    def minimax(self, state, depth, current_position, root_position, alpha, beta):
        terminal_value = self.terminal_value(state, root_position)

        if terminal_value is not None:
            return terminal_value

        if depth <= 0:
            return self.evaluate_state(state, root_position, current_position)

        hand = state["hands"].get(current_position, "")

        actions = self.get_legal_actions_for_hand(
            hand_str=hand,
            last_move=state["last_move"]
        )

        if not actions:
            return self.evaluate_state(state, root_position, current_position)

        actions = self.limit_actions(
            actions=actions,
            state=state,
            current_position=current_position,
        )

        next_player = self.next_position(current_position)

        if self.same_camp(current_position, root_position):
            value = -float("inf")

            for action_str in actions:
                next_state = self.apply_action(
                    state,
                    current_position,
                    action_str
                )

                child_value = self.minimax(
                    state=next_state,
                    depth=depth - 1,
                    current_position=next_player,
                    root_position=root_position,
                    alpha=alpha,
                    beta=beta,
                )

                value = max(value, child_value)
                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return value

        else:
            value = float("inf")

            for action_str in actions:
                next_state = self.apply_action(
                    state,
                    current_position,
                    action_str
                )

                child_value = self.minimax(
                    state=next_state,
                    depth=depth - 1,
                    current_position=next_player,
                    root_position=root_position,
                    alpha=alpha,
                    beta=beta,
                )

                value = min(value, child_value)
                beta = min(beta, value)

                if alpha >= beta:
                    break

            return value

    def apply_action(self, state, position, action_str):
        new_hands = dict(state["hands"])

        if action_str == "":
            new_pass_count = state["pass_count"] + 1

            # 两家 pass 后，下一家重新获得主动权
            if new_pass_count >= 2:
                new_last_move = ""
                new_last_pid = None
                new_pass_count = 0
            else:
                new_last_move = state["last_move"]
                new_last_pid = state["last_pid"]

            return {
                "hands": new_hands,
                "last_move": new_last_move,
                "last_pid": new_last_pid,
                "pass_count": new_pass_count,
            }

        new_hands[position] = self.remove_action_from_hand(
            new_hands.get(position, ""),
            action_str
        )

        return {
            "hands": new_hands,
            "last_move": action_str,
            "last_pid": position,
            "pass_count": 0,
        }

    # ============================================================
    # 4. 状态评价函数
    # ============================================================

    def terminal_value(self, state, root_position):
        hands = state["hands"]

        landlord_empty = len(hands.get("landlord", "")) == 0
        farmer_down_empty = len(hands.get("landlord_down", "")) == 0
        farmer_up_empty = len(hands.get("landlord_up", "")) == 0

        if landlord_empty:
            landlord_score = self.cfg["terminal_score"]
        elif farmer_down_empty or farmer_up_empty:
            landlord_score = -self.cfg["terminal_score"]
        else:
            return None

        if root_position == "landlord":
            return landlord_score

        return -landlord_score

    def evaluate_state(self, state, root_position, current_position):
        """
        统一先计算地主视角分数：
        分数越高，地主越有利；
        分数越低，农民越有利。

        最后如果 root 是农民，就取负数。
        """
        hands = state["hands"]

        landlord_hand = hands.get("landlord", "")
        down_hand = hands.get("landlord_down", "")
        up_hand = hands.get("landlord_up", "")

        landlord_count = len(landlord_hand)
        down_count = len(down_hand)
        up_count = len(up_hand)

        landlord_score = 0.0

        # 剩牌数：地主剩越少越好，农民剩越多越好
        landlord_score -= self.cfg["landlord_count_weight"] * landlord_count
        landlord_score += self.cfg["farmer_count_weight"] * (down_count + up_count)

        # 手牌结构：地主 badness 越低越好，农民 badness 越高越好
        landlord_score -= self.cfg["landlord_badness_weight"] * self.hand_badness(landlord_hand)
        landlord_score += self.cfg["farmer_badness_weight"] * self.hand_badness(down_hand)
        landlord_score += self.cfg["farmer_badness_weight"] * self.hand_badness(up_hand)

        # 接近胜利
        threshold = self.cfg["near_win_threshold"]

        if landlord_count <= threshold:
            landlord_score += self.cfg["near_win_bonus"]

        if down_count <= threshold:
            landlord_score -= self.cfg["near_win_bonus"]

        if up_count <= threshold:
            landlord_score -= self.cfg["near_win_bonus"]

        # 牌权 / 主动权估计
        initiative_holder = None

        if state["last_move"] == "":
            # 当前玩家即将主动出牌
            initiative_holder = current_position
        else:
            initiative_holder = state["last_pid"]

        if initiative_holder == "landlord":
            landlord_score += self.cfg["initiative_bonus"]
        elif initiative_holder in ["landlord_down", "landlord_up"]:
            landlord_score -= self.cfg["initiative_penalty"]

        # 控制牌与炸弹
        landlord_score += self.cfg["control_weight"] * self.control_count(landlord_hand)
        landlord_score -= self.cfg["control_weight"] * 0.7 * self.control_count(down_hand)
        landlord_score -= self.cfg["control_weight"] * 0.7 * self.control_count(up_hand)

        landlord_score += self.cfg["bomb_weight"] * self.bomb_count(landlord_hand)
        landlord_score -= self.cfg["bomb_weight"] * 0.8 * self.bomb_count(down_hand)
        landlord_score -= self.cfg["bomb_weight"] * 0.8 * self.bomb_count(up_hand)

        if root_position == "landlord":
            return landlord_score

        return -landlord_score

    # ============================================================
    # 5. 生成未来节点合法动作
    # ============================================================

    def get_legal_actions_for_hand(self, hand_str, last_move):
        """
        在采样出来的完整状态里，为某个玩家生成合法动作。

        注意：
        - 这里是搜索内部用的近似 legal action generator；
        - root 节点仍然使用环境给出的 infoset.legal_actions；
        - 未来节点使用 CARD_TYPE 动作表 + 手牌包含关系生成。
        """
        hand_str = self.sort_card_str(hand_str)
        last_move = self.sort_card_str(last_move)

        key = (hand_str, last_move)

        if key in self._legal_cache:
            return list(self._legal_cache[key])

        if not hand_str:
            self._legal_cache[key] = []
            return []

        hand_counter = Counter(hand_str)

        leading = last_move == ""
        actions = []

        for action_str in self.all_action_strings():
            if self.counter_leq(Counter(action_str), hand_counter):
                if leading:
                    actions.append(action_str)
                else:
                    if self.can_beat(action_str, last_move):
                        actions.append(action_str)

        if not leading:
            actions.append("")

        self._legal_cache[key] = list(actions)
        return actions

    def all_action_strings(self):
        if self._all_action_cache is not None:
            return self._all_action_cache

        result = set()

        try:
            keys = CARD_TYPE[0].keys()
        except Exception:
            keys = []

        for s in keys:
            if not isinstance(s, str):
                continue

            if not s:
                continue

            valid = True
            for c in s:
                if c not in INDEX:
                    valid = False
                    break

            if not valid:
                continue

            result.add(self.sort_card_str(s))

        self._all_action_cache = sorted(
            result,
            key=lambda x: (len(x), [INDEX[c] for c in x])
        )

        return self._all_action_cache

    def can_beat(self, action_str, last_move):
        if action_str == "":
            return False

        if last_move == "":
            return True

        action_type, action_rank = self.get_card_type(action_str)
        last_type, last_rank = self.get_card_type(last_move)

        action_type_str = str(action_type).lower()
        last_type_str = str(last_type).lower()

        # 王炸最大
        if self.is_rocket(action_str, action_type_str):
            return not self.is_rocket(last_move, last_type_str)

        if self.is_rocket(last_move, last_type_str):
            return False

        # 炸弹逻辑
        action_is_bomb = self.is_bomb_only(action_type_str)
        last_is_bomb = self.is_bomb_only(last_type_str)

        if action_is_bomb:
            if last_is_bomb:
                return action_rank > last_rank
            return True

        if last_is_bomb:
            return False

        # 普通同牌型比较
        if action_type != last_type:
            return False

        if len(action_str) != len(last_move):
            return False

        return action_rank > last_rank

    def limit_actions(self, actions, state, current_position):
        if len(actions) <= self.cfg["max_actions_per_node"]:
            return actions

        hand = state["hands"].get(current_position, "")
        leading = state["last_move"] == ""

        pass_actions = [a for a in actions if a == ""]
        non_pass = [a for a in actions if a != ""]

        scored = []

        for a in non_pass:
            score = self.local_action_score(
                action_str=a,
                hand_str=hand,
                leading=leading,
            )
            scored.append((score, a))

        scored.sort(reverse=True, key=lambda x: x[0])

        limited = []

        # pass 保留
        limited.extend(pass_actions)

        # 炸弹/王炸尽量保留
        bombs = [
            a for _, a in scored
            if self.is_bomb_or_rocket(self.get_card_type(a)[0])
        ]

        for b in bombs:
            if b not in limited:
                limited.append(b)

        for _, a in scored:
            if a not in limited:
                limited.append(a)

            if len(limited) >= self.cfg["max_actions_per_node"]:
                break

        return limited[:self.cfg["max_actions_per_node"]]

    def prune_root_actions(self, legal_actions, root_state):
        if len(legal_actions) <= self.cfg["max_root_actions"]:
            return list(legal_actions)

        my_hand = root_state["my_hand"]
        leading = root_state["last_move"] == ""

        scored = []

        for action in legal_actions:
            action_str = self.env_cards_to_real_str(action)

            score = self.local_action_score(
                action_str=action_str,
                hand_str=my_hand,
                leading=leading,
            )

            scored.append((score, action))

        scored.sort(reverse=True, key=lambda x: x[0])

        result = []

        # pass 如果存在，保留
        for score, action in scored:
            if action == []:
                result.append(action)
                break

        # 炸弹/王炸尽量保留
        for score, action in scored:
            action_str = self.env_cards_to_real_str(action)
            action_type, _ = self.get_card_type(action_str)

            if self.is_bomb_or_rocket(action_type):
                if action not in result:
                    result.append(action)

        for score, action in scored:
            if action not in result:
                result.append(action)

            if len(result) >= self.cfg["max_root_actions"]:
                break

        return result[:self.cfg["max_root_actions"]]

    # ============================================================
    # 6. 本地动作启发式，用于剪枝和排序
    # ============================================================

    def local_action_score(self, action_str, hand_str, leading):
        if action_str == "":
            return self.cfg["local_pass_score"]

        current_badness = self.hand_badness(hand_str)
        next_hand = self.remove_action_from_hand(hand_str, action_str)
        next_badness = self.hand_badness(next_hand)

        score = 0.0

        score += self.cfg["local_hand_improve_weight"] * (
            current_badness - next_badness
        )

        score += self.cfg["local_action_len_weight"] * len(action_str)

        if action_str and self.min_card(hand_str) in action_str:
            score += self.cfg["local_min_card_bonus"]

        if leading and len(action_str) == 1:
            score -= self.cfg["local_single_rank_penalty"] * self.main_rank_value(action_str)

        score -= self.cfg["local_control_cost_weight"] * self.control_cost(action_str)

        action_type, _ = self.get_card_type(action_str)

        if self.is_bomb_or_rocket(action_type):
            score -= self.cfg["local_bomb_penalty"]

        return score

    # ============================================================
    # 7. hand_badness 与辅助评价
    # ============================================================

    def hand_badness(self, hand_str):
        if not hand_str:
            return 0.0

        counts = Counter(hand_str)

        singles = 0
        pairs = 0
        trios = 0
        bombs = 0
        controls = 0

        for c, cnt in counts.items():
            if cnt == 1:
                singles += 1
            elif cnt == 2:
                pairs += 1
            elif cnt == 3:
                trios += 1
            elif cnt == 4:
                bombs += 1

            if c in ["A", "2", "B", "R"]:
                controls += cnt

        turns = singles + pairs + trios + bombs
        chain_discount = self.chain_discount(counts)

        badness = 0.0
        badness += self.cfg["badness_turn_weight"] * turns
        badness += self.cfg["badness_single_weight"] * singles
        badness += self.cfg["badness_len_weight"] * len(hand_str)
        badness -= self.cfg["badness_chain_discount_weight"] * chain_discount
        badness -= self.cfg["badness_control_discount"] * controls
        badness -= self.cfg["badness_bomb_discount"] * bombs

        return badness

    def chain_discount(self, counts):
        discount = 0.0

        # 单顺
        run = 0
        for c in NORMAL_CHAIN_ORDER:
            if counts.get(c, 0) >= 1:
                run += 1
            else:
                if run >= 5:
                    discount += self.cfg["solo_chain_coef"] * (run - 4)
                run = 0

        if run >= 5:
            discount += self.cfg["solo_chain_coef"] * (run - 4)

        # 连对
        run = 0
        for c in NORMAL_CHAIN_ORDER:
            if counts.get(c, 0) >= 2:
                run += 1
            else:
                if run >= 3:
                    discount += self.cfg["pair_chain_coef"] * (run - 2)
                run = 0

        if run >= 3:
            discount += self.cfg["pair_chain_coef"] * (run - 2)

        # 飞机
        run = 0
        for c in NORMAL_CHAIN_ORDER:
            if counts.get(c, 0) >= 3:
                run += 1
            else:
                if run >= 2:
                    discount += self.cfg["trio_chain_coef"] * (run - 1)
                run = 0

        if run >= 2:
            discount += self.cfg["trio_chain_coef"] * (run - 1)

        return discount

    def control_count(self, hand_str):
        return sum(1 for c in hand_str if c in ["A", "2", "B", "R"])

    def bomb_count(self, hand_str):
        counts = Counter(hand_str)
        bombs = 0

        for c, cnt in counts.items():
            if c not in ["B", "R"] and cnt == 4:
                bombs += 1

        if "B" in counts and "R" in counts:
            bombs += 1

        return bombs

    # ============================================================
    # 8. 动作成本与牌型工具
    # ============================================================

    def choose_lowest_cost_env_action(self, actions):
        if not actions:
            return []

        best = None
        best_cost = float("inf")

        for action in actions:
            action_str = self.env_cards_to_real_str(action)
            cost = self.action_cost(action_str)

            if cost < best_cost:
                best_cost = cost
                best = action

        return best if best is not None else random.choice(actions)

    def action_cost(self, action_str):
        if action_str == "":
            return 0.0

        action_type, _ = self.get_card_type(action_str)

        cost = 0.0
        cost += self.cfg["cost_len_weight"] * len(action_str)
        cost += self.cfg["cost_rank_weight"] * self.main_rank_value(action_str)
        cost += self.control_cost(action_str)

        if self.is_bomb_or_rocket(action_type):
            cost += self.cfg["cost_bomb_weight"]

        return cost

    def control_cost(self, action_str):
        cost = 0.0

        for c in action_str:
            if c == "A":
                cost += self.cfg["A_cost"]
            elif c == "2":
                cost += self.cfg["2_cost"]
            elif c == "B":
                cost += self.cfg["B_cost"]
            elif c == "R":
                cost += self.cfg["R_cost"]

        return cost

    def get_card_type(self, action_str):
        if action_str == "":
            return "pass", -1

        candidates = [
            action_str,
            self.sort_card_str(action_str)
        ]

        for s in candidates:
            try:
                info = CARD_TYPE[0][s][0]
                return str(info[0]), int(info[1])
            except Exception:
                continue

        counts = Counter(action_str)
        n = len(action_str)
        max_count = max(counts.values())

        if n == 1:
            return "solo", self.main_rank_value(action_str)

        if n == 2:
            if set(action_str) == set(["B", "R"]):
                return "rocket", 100
            if max_count == 2:
                return "pair", self.main_rank_value(action_str)

        if n == 4 and max_count == 4:
            return "bomb", self.main_rank_value(action_str)

        if max_count == 3:
            return "trio", self.main_rank_value(action_str)

        return "unknown", self.main_rank_value(action_str)

    def main_rank_value(self, action_str):
        if not action_str:
            return -1

        counts = Counter(action_str)
        max_count = max(counts.values())
        main_cards = [c for c, cnt in counts.items() if cnt == max_count]

        return max(RealCard2EnvCard[c] for c in main_cards)

    def is_bomb_or_rocket(self, action_type):
        s = str(action_type).lower()
        return "bomb" in s or "rocket" in s

    def is_bomb_only(self, action_type):
        s = str(action_type).lower()
        return "bomb" in s and "rocket" not in s

    def is_rocket(self, action_str, action_type):
        s = str(action_type).lower()
        return "rocket" in s or (
            len(action_str) == 2 and set(action_str) == set(["B", "R"])
        )

    # ============================================================
    # 9. 阵营与行动顺序
    # ============================================================

    def next_position(self, position):
        """
        斗地主固定顺序：
        landlord -> landlord_down -> landlord_up -> landlord
        """
        if position == "landlord":
            return "landlord_down"

        if position == "landlord_down":
            return "landlord_up"

        if position == "landlord_up":
            return "landlord"

        return "landlord"

    def same_camp(self, p1, p2):
        """
        判断两个位置是否同阵营。
        地主单独一队，两个农民一队。
        """
        if p1 == "landlord" and p2 == "landlord":
            return True

        if p1 != "landlord" and p2 != "landlord":
            return True

        return False

    def normalize_position(self, pid):
        if pid is None:
            return None

        if isinstance(pid, str):
            if pid in ALL_POSITIONS:
                return pid
            return None

        pos_index = {
            0: "landlord",
            1: "landlord_down",
            2: "landlord_up",
        }

        if isinstance(pid, int):
            return pos_index.get(pid, None)

        return None

    # ============================================================
    # 10. 通用工具函数
    # ============================================================

    def counter_leq(self, small, big):
        for k, v in small.items():
            if big.get(k, 0) < v:
                return False
        return True

    def min_card(self, hand_str):
        if not hand_str:
            return ""
        return min(hand_str, key=lambda c: INDEX[c])

    def remove_action_from_hand(self, hand_str, action_str):
        counter = Counter(hand_str)

        for c in action_str:
            counter[c] -= 1
            if counter[c] <= 0:
                del counter[c]

        result = ""

        for c in CARD_ORDER:
            result += c * counter.get(c, 0)

        return result

    def env_cards_to_real_list(self, cards):
        result = []

        if cards is None:
            return result

        if isinstance(cards, str):
            for c in cards:
                if c in INDEX:
                    result.append(c)

            result.sort(key=lambda x: INDEX[x])
            return result

        for c in cards:
            if c in EnvCard2RealCard:
                result.append(EnvCard2RealCard[c])
            elif isinstance(c, str) and c in INDEX:
                result.append(c)

        result.sort(key=lambda x: INDEX[x])
        return result

    def env_cards_to_real_str(self, cards):
        return "".join(self.env_cards_to_real_list(cards))

    def sort_card_str(self, card_str):
        return "".join(sorted(card_str, key=lambda x: INDEX[x]))