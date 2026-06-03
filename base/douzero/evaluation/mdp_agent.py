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

CARD_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "B", "R"]

NORMAL_CHAIN_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

ALL_POSITIONS = ["landlord", "landlord_down", "landlord_up"]


class BayesianMDPAgent(object):
    """
    更稳的 Bayesian + MDP 斗地主 Agent。

    设计原则：
    1. 先保证基本斗地主策略不离谱；
    2. 再加入 Bayesian 风险估计；
    3. MDP 不是深度训练，而是用 Q(s,a)=即时收益+未来手牌价值 的近似评分。
    """

    def __init__(self, position, debug=False):
        self.name = "BayesianMDP_v2"
        self.position = position
        self.debug = debug
        self.fallback_count = 0
        self.last_error = None

    def act(self, infoset):
        try:
            legal_actions = getattr(infoset, "legal_actions", [])

            if not legal_actions:
                return []

            state = self.extract_state(infoset)
            belief = self.infer_belief(infoset, state)

            # 1. 有直接出完的动作，直接出完
            finish_actions = [
                a for a in legal_actions if len(a) == state["my_count"] and a != []
            ]
            if finish_actions:
                return self.choose_lowest_cost_action(finish_actions)

            # 2. 主动出牌
            if state["leading_round"]:
                action = self.choose_leading_action(legal_actions, state, belief)
            # 3. 跟牌
            else:
                action = self.choose_following_action(legal_actions, state, belief)

            if action not in legal_actions:
                action = random.choice(legal_actions)

            return action

        except Exception as e:
            self.fallback_count += 1
            self.last_error = repr(e)
            action = random.choice(infoset.legal_actions)
            return action

    # ============================================================
    # 1. 状态提取
    # ============================================================

    def extract_state(self, infoset):
        my_hand = self.env_cards_to_real_str(getattr(infoset, "player_hand_cards", []))
        last_move = self.env_cards_to_real_str(getattr(infoset, "last_move", []))
        last_two_moves = self.get_last_two_moves(infoset)
        last_pid = getattr(infoset, "last_pid", None)

        leading_round = self.is_leading_round(last_move, last_two_moves)
        num_cards_left = self.get_num_cards_left(infoset)

        enemy_positions = self.get_enemy_positions()
        teammate_position = self.get_teammate_position()

        enemy_counts = []
        for p in enemy_positions:
            if p in num_cards_left:
                enemy_counts.append(num_cards_left[p])

        enemy_min_cards = min(enemy_counts) if enemy_counts else 17

        teammate_cards = None
        if teammate_position is not None and teammate_position in num_cards_left:
            teammate_cards = num_cards_left[teammate_position]

        return {
            "my_hand": my_hand,
            "my_count": len(my_hand),
            "last_move": last_move,
            "last_two_moves": last_two_moves,
            "last_pid": last_pid,
            "leading_round": leading_round,
            "num_cards_left": num_cards_left,
            "enemy_positions": enemy_positions,
            "teammate_position": teammate_position,
            "enemy_min_cards": enemy_min_cards,
            "teammate_cards": teammate_cards,
            "dangerous": enemy_min_cards <= 2,
            "very_dangerous": enemy_min_cards <= 1,
            "is_landlord": self.position == "landlord",
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
        # 连续两家 pass 后，当前玩家主动出牌
        return last_two_moves[0] == "" and last_two_moves[1] == ""

    def get_num_cards_left(self, infoset):
        for attr in ["num_cards_left", "num_cards_left_dict", "player_num_cards_left"]:
            raw = getattr(infoset, attr, None)

            if raw is None:
                continue

            if isinstance(raw, dict):
                return dict(raw)

            if isinstance(raw, (list, tuple)):
                result = {}
                for i, p in enumerate(ALL_POSITIONS):
                    if i < len(raw):
                        result[p] = raw[i]
                return result
        return {}

    # ============================================================
    # 2. 贝叶斯信念
    # ============================================================

    def infer_belief(self, infoset, state):
        unknown_counter = self.get_unknown_cards(infoset, state["my_hand"])

        return {
            "unknown_counter": unknown_counter,
            "bomb_prob": self.estimate_bomb_prob(unknown_counter),
            "rocket_prob": self.estimate_rocket_prob(unknown_counter),
        }

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

    def estimate_rocket_prob(self, unknown_counter):
        if unknown_counter["B"] > 0 and unknown_counter["R"] > 0:
            return 0.25
        return 0.0

    def estimate_bomb_prob(self, unknown_counter):
        possible = 0

        for card in CARD_ORDER:
            if card in ["B", "R"]:
                continue
            if unknown_counter[card] >= 4:
                possible += 1

        return min(0.65, possible * 0.13)

    # ============================================================
    # 3. 主动出牌策略
    # ============================================================

    def choose_leading_action(self, legal_actions, state, belief):
        candidates = [a for a in legal_actions if a != []]

        if not candidates:
            return random.choice(legal_actions)

        best_score = -float("inf")
        best_actions = []

        current_badness = self.hand_badness(state["my_hand"])

        for action in candidates:
            action_str = self.env_cards_to_real_str(action)
            action_type, action_rank = self.get_card_type(action_str)

            next_hand = self.remove_action_from_hand(state["my_hand"], action_str)
            next_badness = self.hand_badness(next_hand)

            score = 0.0

            # 核心：出完之后手牌更好，才是好动作
            score += 10.0 * (current_badness - next_badness)

            # 出牌数量有奖励，但不能压过手牌结构判断
            score += 1.2 * len(action)

            # 主动出牌时，优先带走最小牌
            if action_str and self.min_card(state["my_hand"]) in action_str:
                score += 8.0

            # 牌型奖励
            score += self.leading_type_bonus(action_str, action_type)

            # 不要主动用高单张开路
            if len(action_str) == 1:
                score -= 0.35 * self.main_rank_value(action_str)

            # 控制牌不要乱出
            score -= 1.8 * self.control_cost(action_str, state)

            # 炸弹和王炸，主动出牌时除非能大幅改善手牌，否则强烈惩罚
            if self.is_bomb_or_rocket(action_type):
                score -= 60.0
                if len(action) >= state["my_count"] - 2:
                    score += 40.0

            # Bayesian 风险只作为轻微惩罚
            risk = self.estimate_beaten_risk(action_str, action_type, belief)
            score -= 4.0 * risk

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return self.choose_lowest_cost_action(best_actions)

    def leading_type_bonus(self, action_str, action_type):
        n = len(action_str)
        counts = Counter(action_str)

        if n == 0:
            return -9999.0

        bonus = 0.0
        type_str = str(action_type)

        # 顺子、连对、飞机优先
        if self.is_chain_type(type_str):
            bonus += 12.0 + 0.8 * n

        # 三带、三张通常比单牌好
        if "trio" in type_str or "three" in type_str or max(counts.values()) == 3:
            bonus += 8.0

        # 对子比单牌略好
        if n == 2 and max(counts.values()) == 2:
            bonus += 4.0

        # 四带二不一定好，容易拆炸弹，略扣
        if max(counts.values()) == 4 and n > 4:
            bonus -= 8.0

        return bonus

    # ============================================================
    # 4. 跟牌策略
    # ============================================================

    def choose_following_action(self, legal_actions, state, belief):
        pass_legal = [] in legal_actions
        non_pass = [a for a in legal_actions if a != []]

        if not non_pass:
            return [] if pass_legal else random.choice(legal_actions)

        last_move = state["last_move"]
        last_type, last_rank = self.get_card_type(last_move)

        # 如果上家 pass（last_move 为空），需要判断是否是主动出牌轮
        if last_move == "":
            # 上家 pass，检查是否轮到自己主动出牌
            if state["leading_round"]:
                return self.choose_leading_action(legal_actions, state, belief)
            else:
                # 不是主动出牌轮，应该 pass 让下家出牌
                return [] if pass_legal else random.choice(legal_actions)

        # 农民：如果队友刚刚出牌，且队友快出完了，优先放行
        if self.is_teammate_last_player(state) and pass_legal:
            if state["teammate_cards"] is not None and state["teammate_cards"] <= 2:
                return []

        # 找能压制上家的动作
        beating_actions = []  # 能压制上家的同牌型动作
        bomb_actions = []  # 炸弹/王炸动作

        for action in non_pass:
            action_str = self.env_cards_to_real_str(action)
            action_type, action_rank = self.get_card_type(action_str)

            # 同牌型且能压制
            if action_type == last_type and action_rank > last_rank:
                beating_actions.append(action)
            elif self.is_bomb_or_rocket(action_type):
                bomb_actions.append(action)

        # 优先用最小的能压制的同牌型动作
        if beating_actions:
            return self.choose_lowest_cost_action(beating_actions)

        # 没有能压制的同牌型动作
        # 如果是队友刚出牌，pass 让队友继续
        if self.is_teammate_last_player(state) and pass_legal:
            return []

        # 敌人刚出牌，且危险局面，考虑用炸弹
        if self.is_enemy_last_player(state) and state["dangerous"]:
            if bomb_actions:
                return self.choose_lowest_cost_action(bomb_actions)

        # 其他情况：pass
        if pass_legal:
            return []

        # 兜底：没有 pass 选项时随机选
        return random.choice(non_pass)

    # ============================================================
    # 5. 手牌局面评价
    # ============================================================

    def hand_badness(self, hand_str):
        """
        数值越大，手牌越差。
        这是近似 MDP 的 V(s) 部分。
        """
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

        # 估计剩余出牌轮数
        turns = singles + pairs + trios + bombs

        # 顺子/连对/飞机可以减少出牌轮数
        chain_discount = self.chain_discount(counts)

        badness = 0.0
        badness += 7.0 * turns
        badness += 3.0 * singles
        badness += 1.2 * len(hand_str)
        badness -= 5.0 * chain_discount

        # 控制牌是好东西，降低 badness
        badness -= 2.2 * controls

        # 炸弹是强控制，但不鼓励乱拆
        badness -= 3.0 * bombs

        return badness

    def chain_discount(self, counts):
        discount = 0.0

        # 单顺：至少 5 张
        run = 0
        for c in NORMAL_CHAIN_ORDER:
            if counts.get(c, 0) >= 1:
                run += 1
            else:
                if run >= 5:
                    discount += run - 4
                run = 0
        if run >= 5:
            discount += run - 4

        # 连对：至少 3 对
        run = 0
        for c in NORMAL_CHAIN_ORDER:
            if counts.get(c, 0) >= 2:
                run += 1
            else:
                if run >= 3:
                    discount += 1.5 * (run - 2)
                run = 0
        if run >= 3:
            discount += 1.5 * (run - 2)

        # 飞机：至少 2 个三张
        run = 0
        for c in NORMAL_CHAIN_ORDER:
            if counts.get(c, 0) >= 3:
                run += 1
            else:
                if run >= 2:
                    discount += 2.0 * (run - 1)
                run = 0
        if run >= 2:
            discount += 2.0 * (run - 1)

        return discount

    # ============================================================
    # 6. Bayesian 风险估计
    # ============================================================

    def estimate_beaten_risk(self, action_str, action_type, belief):
        if not action_str:
            return 0.0

        type_str = str(action_type)
        unknown = belief["unknown_counter"]

        if self.is_rocket(action_str, type_str):
            return 0.0

        if self.is_bomb_or_rocket(type_str):
            main = self.main_rank_value(action_str)
            higher_bombs = 0

            for c in CARD_ORDER:
                if c in ["B", "R"]:
                    continue
                if RealCard2EnvCard[c] > main and unknown[c] >= 4:
                    higher_bombs += 1

            return min(0.9, belief["rocket_prob"] + 0.12 * higher_bombs)

        same_type_risk = self.estimate_same_type_risk(action_str, type_str, unknown)

        return min(
            0.9,
            same_type_risk + 0.25 * belief["bomb_prob"] + 0.15 * belief["rocket_prob"],
        )

    def estimate_same_type_risk(self, action_str, action_type, unknown):
        counts = Counter(action_str)
        main = self.main_rank_value(action_str)

        if not counts:
            return 0.0

        max_count = max(counts.values())
        n = len(action_str)

        if n == 1:
            need = 1
        elif n == 2 and max_count == 2:
            need = 2
        elif max_count == 3:
            need = 3
        else:
            need = max_count

        pressure = 0.0

        for c in CARD_ORDER:
            if c in ["B", "R"]:
                continue

            if RealCard2EnvCard[c] <= main:
                continue

            if unknown[c] >= need:
                pressure += (unknown[c] / 4.0) ** need

        return min(0.8, 1.0 - math.exp(-0.35 * pressure))

    # ============================================================
    # 7. 动作成本与工具函数
    # ============================================================

    def choose_lowest_cost_action(self, actions):
        """
        从候选动作中选择成本最低的。
        跟牌时尤其重要：能用小牌压就不用大牌压。
        """
        best = None
        best_cost = float("inf")

        for action in actions:
            action_str = self.env_cards_to_real_str(action)
            action_type, action_rank = self.get_card_type(action_str)

            cost = 0.0
            cost += 0.5 * len(action)
            cost += 0.08 * self.main_rank_value(action_str)
            cost += self.control_cost(action_str, {"dangerous": False})

            if self.is_bomb_or_rocket(action_type):
                cost += 50.0

            if cost < best_cost:
                best_cost = cost
                best = action

        return best if best is not None else random.choice(actions)

    def control_cost(self, action_str, state):
        cost = 0.0

        for c in action_str:
            if c == "A":
                cost += 1.5
            elif c == "2":
                cost += 4.5
            elif c == "B":
                cost += 7.0
            elif c == "R":
                cost += 8.0

        if state.get("dangerous", False):
            cost *= 0.45

        return cost

    def get_card_type(self, action_str):
        if action_str == "":
            return "pass", -1

        candidates = [action_str, self.sort_card_str(action_str)]

        for s in candidates:
            try:
                info = CARD_TYPE[0][s][0]
                return str(info[0]), int(info[1])
            except Exception:
                continue

        # fallback
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

    def is_rocket(self, action_str, action_type):
        s = str(action_type).lower()
        return "rocket" in s or set(action_str) == set(["B", "R"])

    def is_chain_type(self, action_type):
        s = str(action_type).lower()
        return (
            "chain" in s
            or "sequence" in s
            or "plane" in s
            or "solo_chain" in s
            or "pair_chain" in s
            or "trio_chain" in s
        )

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

        # 兼容字符串动作，比如 "3344"
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

    # ============================================================
    # 8. 身份与队友判断
    # ============================================================

    def get_enemy_positions(self):
        if self.position == "landlord":
            return ["landlord_down", "landlord_up"]
        return ["landlord"]

    def get_teammate_position(self):
        if self.position == "landlord":
            return None

        if self.position == "landlord_down":
            return "landlord_up"

        if self.position == "landlord_up":
            return "landlord_down"

        return None

    def is_teammate_last_player(self, state):
        """
        判断上一个出牌的是不是队友。
        兼容 last_pid 是位置字符串或数字索引两种情况。
        """
        if self.position == "landlord":
            return False

        last_pid = state.get("last_pid")
        teammate_position = state.get("teammate_position")

        if last_pid is None or teammate_position is None:
            return False

        # 兼容数字索引
        pos_index = {"landlord": 0, "landlord_down": 1, "landlord_up": 2}

        if isinstance(last_pid, int):
            return last_pid == pos_index.get(teammate_position, -1)

        return last_pid == teammate_position

    def is_enemy_last_player(self, state):
        """
        判断上一个出牌的是不是敌人。
        兼容 last_pid 是位置字符串或数字索引两种情况。
        """
        last_pid = state.get("last_pid")

        if last_pid is None:
            return False

        pos_index = {"landlord": 0, "landlord_down": 1, "landlord_up": 2}

        if isinstance(last_pid, int):
            if self.position == "landlord":
                return last_pid in [1, 2]  # 地主的敌人是两个农民
            else:
                return last_pid == 0  # 农民的敌人是地主

        if self.position == "landlord":
            return last_pid in ["landlord_down", "landlord_up"]

        return last_pid == "landlord"
