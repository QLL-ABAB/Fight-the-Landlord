# mcts_agent.py
# Enhanced MCTS with RLCard rollout (default) and prior utility from HighRankMonteCarloAgent
# Improved determinization sampling using random subset selection
# Added enhanced bidding via fast sampling rollouts
# ------------------------------------------------------------

import random
import copy
import math
import time
from collections import Counter

from douzero.env.move_generator import MovesGener
from douzero.env import move_detector as md
from douzero.env import move_selector as ms

# 导入 HR-MC 相关类（用于 utility）
from .high_rank_montecarlo_agent import (
    HighRankMonteCarloAgent,
    CardCombinations,
    env_action_to_concrete,
)

# 导入 RLCardAgent 用于默认 rollout
from .rlcard_agent import RLCardAgent

# 导入 ApproxDouFeatureAgent 作为可选 RL rollout
from .approx_doufeature_agent import ApproxDouFeatureAgent

ALL_POSITIONS = ['landlord', 'landlord_down', 'landlord_up']

# 卡牌映射
EnvCard2RealCard = {
    3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A", 17: "2",
    20: "B", 30: "R",
}
RealCard2EnvCard = {v: k for k, v in EnvCard2RealCard.items()}
CARD_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "B", "R"]
INDEX = {c: i for i, c in enumerate(CARD_ORDER)}
NORMAL_CHAIN_ORDER = CARD_ORDER[:-2]   # 不含2和王


class MCTSAgent:
    def __init__(self, num_simulations=200, position=None, c=1.414,
                 max_rollout_steps=50, time_budget=0.3,
                 num_determinizations=8, objective="logadp",
                 rollout_policy='rlcard'):
        self.num_simulations = num_simulations
        self.position = position
        self.c = c
        self.max_rollout_steps = max_rollout_steps
        self.time_budget = time_budget
        self.num_determinizations = num_determinizations
        self.objective = objective
        self.rollout_policy = rollout_policy
        self.name = "MCTS_Enhanced"

        self._legal_cache = {}

        self.cfg = {
            # 评估函数权重（已修改，保留部分用于上下文奖励）
            "eval_enemy_danger_penalty": 180.0,
            "eval_team_danger_bonus": 120.0,
            "eval_initiative_bonus": 16.0,
            "eval_last_pid_bonus": 24.0,
            "eval_bomb_bonus": 16.0,
            "eval_control_bonus": 3.5,      # 保留但不使用（已用utility替代）
            "danger_cards": 2,
            "very_danger_cards": 1,
            "leading_initiative_bonus": 20.0,   # 新增：主动出牌额外奖励
            # 保留旧的hand_badness参数（作为回退）
            "badness_turn_weight": 7.0,
            "badness_single_weight": 3.0,
            "badness_len_weight": 1.2,
            "badness_chain_discount_weight": 5.0,
            "badness_control_discount": 2.2,
            "badness_bomb_discount": 3.0,
            "solo_chain_coef": 1.0,
            "pair_chain_coef": 1.5,
            "trio_chain_coef": 2.0,
            # MCTS相关
            "leading_c_multiplier": 0.8,
            "following_c_multiplier": 1.2,
            "leading_initiative_bonus": 25.0,
            "following_beat_bonus": 10.0,
            "teammate_release_bonus": 50.0,
            "teammate_block_penalty": 35.0,
        }

        # [PRIOR] 创建 HR-MC 代理实例，用于计算 utility
        try:
            self._hr_agent = HighRankMonteCarloAgent(
                position=self.position,
                time_limit_sec=0.01,
                n_root_actions=1,
                seed=42
            )
        except Exception:
            self._hr_agent = None
        self._hr_alpha = 19
        self._utility_cache = {}

        # 初始化 rollout 策略
        self._rollout_agent = None
        if rollout_policy == 'approx':
            try:
                self._rollout_agent = ApproxDouFeatureAgent(self.position)
            except Exception:
                print("Warning: ApproxDouFeatureAgent init failed, fallback to heuristic")
                self._rollout_agent = None
        elif rollout_policy == 'rlcard':
            try:
                self._rollout_agent = RLCardAgent(self.position)
            except Exception:
                print("Warning: RLCardAgent init failed, fallback to heuristic")
                self._rollout_agent = None
        elif isinstance(rollout_policy, str) and rollout_policy != 'heuristic':
            try:
                self._rollout_agent = ApproxDouFeatureAgent(self.position, model_path=rollout_policy)
            except Exception:
                print(f"Warning: Failed to load rollout model from {rollout_policy}, fallback to heuristic")
                self._rollout_agent = None
        elif hasattr(rollout_policy, 'act'):
            self._rollout_agent = rollout_policy
        else:
            self._rollout_agent = None

    # ================== 叫牌（增强版） ==================
    def bid(self, hand_cards, three_landlord_cards=None):
        """
        增强叫牌决策：通过采样隐藏牌和多局快速模拟，评估每个叫分（0~3）的期望胜率。
        返回最佳叫分（0-3）。
        """
        # 1. 准备工作：获取手牌和未知牌池
        my_hand_str = self._env_cards_to_real_str(hand_cards)
        full_deck = Counter()
        for c in CARD_ORDER:
            full_deck[c] = 1 if c in ['B','R'] else 4
        for c in my_hand_str:
            full_deck[c] -= 1
        # 若底牌已知（测试时），从牌池移除
        if three_landlord_cards:
            for c in three_landlord_cards:
                if c in full_deck:
                    full_deck[c] -= 1
        for c in list(full_deck.keys()):
            if full_deck[c] < 0:
                full_deck[c] = 0
        unknown_cards = []
        for c, cnt in full_deck.items():
            unknown_cards.extend([c]*cnt)

        other_positions = [p for p in ALL_POSITIONS if p != self.position]
        num_worlds = 30  # 采样世界数
        bid_scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        # 2. 对每个叫分进行模拟
        for bid_score in [0, 1, 2, 3]:
            wins = 0
            for _ in range(num_worlds):
                # 随机打乱未知牌
                shuffled = unknown_cards[:]
                random.shuffle(shuffled)
                # 分配两个农民各17张，底牌3张（总牌数可能不够，但正常应为37张）
                if len(shuffled) < 37:
                    continue
                farmer1 = shuffled[:17]
                farmer2 = shuffled[17:34]
                public = shuffled[34:37]  # 三张底牌

                if bid_score > 0:
                    # 当前玩家叫地主，获得底牌
                    landlord = self.position
                    landlord_hand = my_hand_str + ''.join(public)
                    farmer_pos = other_positions
                    assign = {
                        landlord: landlord_hand,
                        farmer_pos[0]: ''.join(farmer1),
                        farmer_pos[1]: ''.join(farmer2)
                    }
                    target_landlord = landlord
                else:
                    # 当前玩家不叫，随机选一个其他玩家成为地主（并获得底牌）
                    landlord_idx = random.randint(0,1)
                    landlord = other_positions[landlord_idx]
                    if landlord_idx == 0:
                        farmer_other = other_positions[1]
                        landlord_hand = ''.join(farmer1) + ''.join(public)
                        farmer_hand = ''.join(farmer2)
                    else:
                        farmer_other = other_positions[0]
                        landlord_hand = ''.join(farmer2) + ''.join(public)
                        farmer_hand = ''.join(farmer1)
                    assign = {
                        landlord: landlord_hand,
                        self.position: my_hand_str,
                        farmer_other: farmer_hand
                    }
                    target_landlord = landlord

                # 将字符串转为环境整数列表
                hand_env = {}
                for pos, hand_str in assign.items():
                    hand_env[pos] = [RealCard2EnvCard[c] for c in hand_str if c in RealCard2EnvCard]

                # 创建游戏状态，当前玩家为 self.position
                state = _GameState(hand_env, self.position, (), None, 0)
                # 快速模拟一局
                winner = self._fast_rollout(state)
                # 判断胜利条件
                if bid_score > 0:
                    if winner == self.position:
                        wins += 1
                else:
                    # 农民胜利：地主输
                    if winner != target_landlord:
                        wins += 1
            bid_scores[bid_score] = wins / max(1, num_worlds)

        # 3. 选择最优叫分
        best_score = max(bid_scores.values())
        best_bids = [b for b, v in bid_scores.items() if v == best_score]
        return random.choice(best_bids)

    def _fast_rollout(self, state):
        """
        快速模拟一局完整斗地主，返回获胜者位置（'landlord'/'landlord_up'/'landlord_down'）。
        使用启发式策略（复用现有的 _heuristic_rollout_policy）。
        """
        sim = copy.deepcopy(state)
        while not sim.is_done():
            legal = sim.get_legal_actions()
            if not legal:
                break
            is_leading = (sim.last_move is None or len(sim.last_move) == 0)
            action = self._heuristic_rollout_policy(sim, legal, is_leading)
            sim.step(action)
        # 找出手牌为空的玩家
        for pos in ALL_POSITIONS:
            if len(sim.hand_cards[pos]) == 0:
                return pos
        return None

    # ================== 核心决策 ==================
    def act(self, infoset):
        legal_actions = infoset.legal_actions
        if not legal_actions:
            return []
        if len(legal_actions) == 1:
            return legal_actions[0]

        my_hand = self._env_cards_to_real_str(infoset.player_hand_cards)
        finish_actions = [a for a in legal_actions
                          if a != [] and len(self._env_cards_to_real_str(a)) == len(my_hand)]
        if finish_actions:
            return self._choose_lowest_cost_action(finish_actions, my_hand)

        # 农民队友快出完时强制过牌
        if (self.position != "landlord" and
            self._is_teammate_last_player(infoset) and
            self._teammate_near_finish(infoset) and
            [] in legal_actions):
            return []

        worlds = self._sample_determinizations(infoset, self.num_determinizations)
        if not worlds:
            return self._greedy_action(legal_actions, infoset)

        is_leading = self._is_leading_round(infoset)

        root_visits = {self._action_key(a): 0 for a in legal_actions}
        deadline = time.time() + self.time_budget

        self._utility_cache.clear()

        for hand_assign in worlds:
            if time.time() > deadline:
                break
            root = self.Node(None, None)
            state = _GameState.from_infoset_with_assign(infoset, hand_assign, self.position)
            legal = state.get_legal_actions()
            if not legal:
                continue

            root.untried_actions = [self._action_key(a) for a in legal]
            for a in legal:
                a_key = self._action_key(a)
                child_state = copy.deepcopy(state)
                child_state.step(a)
                prior = self._compute_prior(child_state, self.position)
                child_node = self.Node(state, a, prior=prior)
                root.children[a_key] = child_node

            effective_c = self.c
            if is_leading:
                effective_c *= self.cfg["leading_c_multiplier"]
            else:
                effective_c *= self.cfg["following_c_multiplier"]

            for _ in range(self.num_simulations):
                if time.time() > deadline:
                    break
                node = self._select(root, state, effective_c)
                reward = self._simulate(node.state, is_leading)
                self._backup(node, reward, state.current_player, is_leading)

            for a_key, child in root.children.items():
                root_visits[a_key] += child.n

        if not root_visits:
            return self._greedy_action(legal_actions, infoset)
        best_key = max(root_visits.items(), key=lambda x: x[1])[0]
        best_action = self._key_to_action(best_key, legal_actions)
        return best_action if best_action is not None else random.choice(legal_actions)

    # ================== MCTS 内部类 ==================
    class Node:
        __slots__ = ['state', 'action', 'parent', 'children', 'n', 'w', 'untried_actions', 'prior']
        def __init__(self, state, action, parent=None, prior=0.0):
            self.state = state
            self.action = action
            self.parent = parent
            self.children = {}
            self.n = 0
            self.w = 0
            self.untried_actions = []
            self.prior = prior

    def _select(self, node, state, c):
        while not state.is_done():
            if node.untried_actions:
                return self._expand(node, state)
            best_child = self._best_child(node, c)
            if best_child is None:
                break
            node = best_child
            state = node.state if node.state is not None else state
        return node

    def _expand(self, node, state):
        if not node.untried_actions:
            return node
        action_key = random.choice(node.untried_actions)
        node.untried_actions.remove(action_key)
        action = self._key_to_action(action_key, state.get_legal_actions())
        if action is None:
            return node
        new_state = copy.deepcopy(state)
        new_state.step(action)
        prior = self._compute_prior(new_state, self.position)
        child = self.Node(new_state, action, parent=node, prior=prior)
        node.children[action_key] = child
        return child

    def _best_child(self, node, c):
        if not node.children:
            return None
        log_parent = math.log(node.n) if node.n > 0 else 0.0
        best_score = -float('inf')
        best = None
        for child in node.children.values():
            if child.n == 0:
                exploit = child.prior
                explore = c * math.sqrt(log_parent / 1.0)
            else:
                exploit = child.w / child.n
                explore = c * math.sqrt(log_parent / child.n)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best

    # ===== 模拟 =====
    def _simulate(self, state, is_leading):
        sim_state = copy.deepcopy(state)
        steps = 0
        while not sim_state.is_done() and steps < self.max_rollout_steps:
            legal = sim_state.get_legal_actions()
            if not legal:
                break
            if self._rollout_agent is not None:
                action = self._rl_rollout_policy(sim_state, legal)
            else:
                action = self._heuristic_rollout_policy(sim_state, legal, is_leading)
            sim_state.step(action)
            steps += 1
        if not sim_state.is_done():
            return self._evaluate_state(sim_state, is_leading)
        winner = sim_state.get_winner()
        bomb_num = sim_state.bomb_num
        return self._terminal_value(winner, bomb_num)

    def _rl_rollout_policy(self, state, legal_actions):
        infoset = self._build_infoset_from_state(state, legal_actions)
        try:
            action = self._rollout_agent.act(infoset)
            if action in legal_actions:
                return action
            else:
                return self._heuristic_rollout_policy(state, legal_actions, False)
        except Exception:
            return self._heuristic_rollout_policy(state, legal_actions, False)

    def _build_infoset_from_state(self, state, legal_actions):
        class SimInfoset:
            pass
        infoset = SimInfoset()
        infoset.legal_actions = legal_actions
        infoset.player_hand_cards = state.hand_cards[state.current_player].copy()
        infoset.last_move = list(state.last_move) if state.last_move else []
        if state.last_move:
            infoset.last_two_moves = [[], list(state.last_move)]
        else:
            infoset.last_two_moves = [[], []]
        infoset.last_pid = state.last_pid
        infoset.card_play_action_seq = []
        infoset.played_cards = {}
        infoset.all_handcards = {p: h.copy() for p, h in state.hand_cards.items()}
        infoset.player_position = state.current_player
        infoset.bomb_num = state.bomb_num
        num_cards_left = {p: len(h) for p, h in state.hand_cards.items()}
        infoset.num_cards_left_dict = num_cards_left
        infoset.three_landlord_cards = []
        infoset.last_move_dict = {}
        return infoset

    def _heuristic_rollout_policy(self, state, legal_actions, is_leading):
        if is_leading:
            structural = []
            for a in legal_actions:
                if not a:
                    continue
                if self._is_bomb_or_rocket(a):
                    continue
                a_str = self._env_cards_to_real_str(a)
                counts = Counter(a_str)
                if (len(set(counts.values())) > 1 or
                    self._is_solo_chain(a_str) or
                    self._is_pair_chain(a_str) or
                    self._is_trio_chain(a_str) or
                    max(counts.values()) >= 3):
                    structural.append(a)
            if structural:
                structural.sort(key=lambda a: (-len(a), self._main_rank_value(a)))
                return structural[0]
            singles = [a for a in legal_actions if a and len(a) == 1]
            if singles:
                return min(singles, key=lambda a: self._main_rank_value(a))
            non_bomb = [a for a in legal_actions if a and not self._is_bomb_or_rocket(a)]
            if non_bomb:
                return min(non_bomb, key=lambda a: self._main_rank_value(a))
            return random.choice(legal_actions)
        else:
            non_bomb = [a for a in legal_actions if a and not self._is_bomb_or_rocket(a)]
            candidates = non_bomb if non_bomb else legal_actions
            def score(a):
                if not a:
                    return 1000
                return len(a) * 10 - self._main_rank_value(a)
            return min(candidates, key=score)

    def _backup(self, node, value, current_player, is_leading):
        while node is not None:
            node.n += 1
            if node.action is not None and node.parent is not None:
                actor = node.parent.state.current_player if node.parent.state else current_player
                if (self.position != "landlord" and
                    node.parent.state and
                    self._same_team(actor, self.position) and
                    node.action and
                    self._is_teammate_last_player_from_state(node.parent.state) and
                    len(node.action) > 0):
                    value -= self.cfg["teammate_block_penalty"] / max(1, node.n)
                if self._same_team(actor, self.position):
                    node.w += value
                else:
                    node.w -= value
            else:
                node.w += value
            node = node.parent

    # ================== 先验计算 ==================
    def _compute_prior(self, state, root_position):
        if self._hr_agent is None:
            return 0.0
        player = state.current_player
        hand_env = state.hand_cards[player]
        key = (player, tuple(sorted(hand_env)))
        if key in self._utility_cache:
            util = self._utility_cache[key]
        else:
            try:
                concrete = env_action_to_concrete(hand_env, used=set())
            except Exception:
                util = 0.0
            else:
                hand_cc = CardCombinations(concrete)
                util = self._hr_agent.utility(hand_cc, alpha=self._hr_alpha)
            self._utility_cache[key] = util

        if self._same_team(player, root_position):
            sign = 1.0
        else:
            sign = -1.0
        normalized = sign * util / 100.0
        return max(-1.0, min(1.0, normalized))

    # ================== 评估函数（重写，基于HR-MC utility） ==================
    def _evaluate_state(self, state, is_leading):
        """
        基于 HR-MC 的 utility 评估局面价值，同时保留上下文奖励。
        值越大表示对根阵营越有利。
        """
        hands = state.hand_cards
        landlord = "landlord"
        farmers = ["landlord_down", "landlord_up"]

        # 1. 使用 HR-MC utility 计算手牌结构价值
        if self.position == landlord:
            root_team_value = self._get_hand_utility(landlord, hands[landlord])
            enemy_value = max(
                self._get_hand_utility(farmers[0], hands[farmers[0]]),
                self._get_hand_utility(farmers[1], hands[farmers[1]])
            )
            val = root_team_value - enemy_value
        else:
            teammate = self._teammate_position()
            root_team_value = max(
                self._get_hand_utility(self.position, hands[self.position]),
                self._get_hand_utility(teammate, hands[teammate])
            )
            enemy_value = self._get_hand_utility(landlord, hands[landlord])
            val = root_team_value - enemy_value

        # 2. 危险牌数奖励/惩罚（纯utility无法体现即时威胁）
        for p in ALL_POSITIONS:
            n = len(hands[p])
            if n <= self.cfg["very_danger_cards"]:
                bonus = self.cfg["eval_team_danger_bonus"] if self._same_team(p, self.position) \
                        else -self.cfg["eval_enemy_danger_penalty"]
                val += bonus
            elif n <= self.cfg["danger_cards"]:
                bonus = 0.45 * self.cfg["eval_team_danger_bonus"] if self._same_team(p, self.position) \
                        else -0.45 * self.cfg["eval_enemy_danger_penalty"]
                val += bonus

        # 3. 主动权和最后出牌权奖励
        if self._same_team(state.current_player, self.position) and not state.last_move:
            val += self.cfg["eval_initiative_bonus"]
            if is_leading:
                val += self.cfg["leading_initiative_bonus"]
        if state.last_pid is not None and self._same_team(state.last_pid, self.position):
            val += self.cfg["eval_last_pid_bonus"]

        # 4. 炸弹显式奖励（utility已含，这里额外加强）
        for p in ALL_POSITIONS:
            sign = 1.0 if self._same_team(p, self.position) else -1.0
            bomb_count = self._count_bombs(hands[p])
            val += sign * self.cfg["eval_bomb_bonus"] * bomb_count

        # 5. 归一化，HR-MC utility 通常在数百量级，除以200缩放
        return max(-1.0, min(1.0, val / 200.0))

    # ================== HR-MC utility 辅助方法 ==================
    def _env_hand_to_concrete(self, hand):
        """将环境手牌（int list）转换为 HR-MC 的具体牌（0~53）"""
        used = set()
        concrete = []
        for card in hand:
            # 环境牌映射到等级
            if card == 20:      # 小王
                level = 13
            elif card == 30:    # 大王
                level = 14
            elif card == 17:    # 2
                level = 12
            else:
                level = card - 3   # 3->0, 4->1, ..., A->11
            # 获取该等级的具体牌 ID
            ids = self._concrete_ids_for_level(level)
            chosen = None
            for cid in ids:
                if cid not in used:
                    chosen = cid
                    break
            if chosen is None:
                chosen = ids[0]  # fallback
            used.add(chosen)
            concrete.append(chosen)
        return sorted(concrete)

    def _concrete_ids_for_level(self, level):
        """返回等级对应的具体牌 ID 列表"""
        if level == 13:   # 小王
            return [52]
        if level == 14:   # 大王
            return [53]
        return [level * 4 + i for i in range(4)]

    def _get_hand_utility(self, player, hand_cards):
        """获取手牌的 HR-MC 效用值（带缓存）"""
        hand_tuple = tuple(sorted(hand_cards))
        key = (player, hand_tuple)
        if key in self._utility_cache:
            return self._utility_cache[key]

        if self._hr_agent is None:
            # 回退到旧的 hand_badness（取负值使其方向一致，越大越好）
            hand_str = self._env_cards_to_real_str(hand_cards)
            util = -self._hand_badness(hand_str) - self.cfg.get("eval_count_weight", 7.0) * len(hand_str)
        else:
            try:
                concrete = self._env_hand_to_concrete(hand_cards)
                cc = CardCombinations(concrete)
                util = self._hr_agent.utility(cc, alpha=self._hr_alpha)
            except Exception:
                # 异常回退
                hand_str = self._env_cards_to_real_str(hand_cards)
                util = -self._hand_badness(hand_str) - self.cfg.get("eval_count_weight", 7.0) * len(hand_str)

        self._utility_cache[key] = util
        return util

    # ================== 辅助函数（保留原功能） ==================
    def _hand_badness(self, hand_str):
        if not hand_str:
            return 0.0
        counts = Counter(hand_str)
        singles = sum(1 for c, cnt in counts.items() if cnt == 1)
        pairs = sum(1 for c, cnt in counts.items() if cnt == 2)
        trios = sum(1 for c, cnt in counts.items() if cnt == 3)
        bombs = sum(1 for c, cnt in counts.items() if cnt == 4)
        controls = sum(cnt for c, cnt in counts.items() if c in ["A", "2", "B", "R"])
        turns = singles + pairs + trios + bombs
        chain_discount = self._chain_discount(counts)
        bad = (self.cfg["badness_turn_weight"] * turns
               + self.cfg["badness_single_weight"] * singles
               + self.cfg["badness_len_weight"] * len(hand_str)
               - self.cfg["badness_chain_discount_weight"] * chain_discount
               - self.cfg["badness_control_discount"] * controls
               - self.cfg["badness_bomb_discount"] * bombs)
        return bad

    def _chain_discount(self, counts):
        discount = 0.0
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

    def _count_bombs(self, hand):
        counts = Counter(hand)
        n = 0
        for c in NORMAL_CHAIN_ORDER + ["2"]:
            if counts.get(c, 0) >= 4:
                n += 1
        if counts.get(20, 0) and counts.get(30, 0):
            n += 1
        return n

    def _control_count(self, hand):
        return sum(1 for c in hand if c in (14, 17, 20, 30))

    def _terminal_value(self, winner, bomb_num):
        if self.objective == "adp":
            scale = 2.0 ** bomb_num
        elif self.objective == "logadp":
            scale = float(bomb_num + 1)
        else:
            scale = 1.0
        norm = 64.0 if self.objective == "adp" else 8.0
        if self.position == "landlord":
            reward = scale if winner == "landlord" else -scale
        else:
            reward = scale if winner != "landlord" else -scale
        return max(-1.0, min(1.0, reward / norm))

    def _is_bomb_or_rocket(self, action):
        if len(action) == 2 and set(action) == {20, 30}:
            return True
        if len(action) == 4 and len(set(action)) == 1:
            return True
        return False

    def _main_rank_value(self, action):
        if not action:
            return -1
        counts = Counter(action)
        max_count = max(counts.values())
        main_cards = [c for c, cnt in counts.items() if cnt == max_count]
        return max(RealCard2EnvCard.get(c, 0) for c in main_cards)

    def _is_solo_chain(self, action_str):
        counts = Counter(action_str)
        return (len(action_str) >= 5 and all(v == 1 for v in counts.values()) and
                self._is_consecutive(list(counts.keys())))
    def _is_pair_chain(self, action_str):
        counts = Counter(action_str)
        return (len(counts) >= 3 and all(v == 2 for v in counts.values()) and
                self._is_consecutive(list(counts.keys())))
    def _is_trio_chain(self, action_str):
        counts = Counter(action_str)
        return (len(counts) >= 2 and all(v == 3 for v in counts.values()) and
                self._is_consecutive(list(counts.keys())))
    def _is_consecutive(self, ranks):
        if not ranks:
            return False
        try:
            idxs = [NORMAL_CHAIN_ORDER.index(c) for c in ranks]
        except ValueError:
            return False
        return max(idxs) - min(idxs) + 1 == len(idxs) and len(set(idxs)) == len(idxs)

    # ================== 改进的确定化采样方法 ==================
    def _sample_determinizations(self, infoset, n):
        """
        生成 n 个可能的隐藏手牌分配。
        改进：使用随机子集选择（不放回抽样）代替顺序截取，确保每个玩家的牌是均匀随机从剩余牌中抽取的。
        """
        my_hand = self._env_cards_to_real_str(infoset.player_hand_cards)
        unknown_counter = self._get_unknown_cards(infoset, my_hand)
        unknown_cards = []
        for c in CARD_ORDER:
            unknown_cards.extend([c] * unknown_counter[c])

        # 获取每个玩家的剩余牌数
        num_left = {}
        if hasattr(infoset, 'num_cards_left_dict') and infoset.num_cards_left_dict:
            num_left = infoset.num_cards_left_dict
        elif hasattr(infoset, 'num_cards_left') and infoset.num_cards_left:
            raw = infoset.num_cards_left
            if isinstance(raw, dict):
                num_left = raw
            elif isinstance(raw, (list, tuple)):
                for i, p in enumerate(ALL_POSITIONS):
                    if i < len(raw):
                        num_left[p] = raw[i]
        else:
            for pos in ALL_POSITIONS:
                num_left[pos] = len(infoset.all_handcards.get(pos, []))

        num_left[self.position] = len(my_hand)

        targets = {}
        for p in ALL_POSITIONS:
            if p != self.position:
                targets[p] = num_left.get(p, 0)

        # 确保总数匹配，以防信息不一致
        total_need = sum(targets.values())
        if total_need > len(unknown_cards):
            overflow = total_need - len(unknown_cards)
            for p in sorted(targets, key=lambda x: targets[x], reverse=True):
                take = min(overflow, targets[p])
                targets[p] -= take
                overflow -= take
                if overflow <= 0:
                    break
        elif total_need < len(unknown_cards):
            # 如果有剩余未分配牌，安全起见丢弃（理论上不应发生）
            pass

        worlds = []
        for _ in range(n):
            # 复制未知牌列表
            cards = unknown_cards[:]
            assign = {p: "" for p in ALL_POSITIONS}
            assign[self.position] = my_hand

            # 对每个非我方玩家，使用 random.sample 从当前剩余牌中抽取所需数量
            for p in ALL_POSITIONS:
                if p == self.position:
                    continue
                cnt = targets.get(p, 0)
                if cnt > 0:
                    # 从 cards 中随机选取 cnt 张（不放回）
                    chosen = random.sample(cards, cnt)
                    # 从 cards 中移除这些牌
                    for card in chosen:
                        cards.remove(card)
                    assign[p] = "".join(sorted(chosen, key=lambda x: INDEX[x]))
                else:
                    assign[p] = ""
            worlds.append(assign)
        return worlds

    # ================== 原有采样辅助（保留） ==================
    def _get_unknown_cards(self, infoset, my_hand):
        deck = Counter()
        for c in CARD_ORDER:
            deck[c] = 1 if c in ["B", "R"] else 4
        for c in my_hand:
            deck[c] -= 1
        played = []
        seq = getattr(infoset, "card_play_action_seq", [])
        for act in seq:
            played.extend(self._env_cards_to_real_list(act))
        for c in played:
            deck[c] -= 1
        for c in list(deck.keys()):
            deck[c] = max(0, deck[c])
        return deck

    def _env_cards_to_real_list(self, cards):
        if cards is None:
            return []
        if isinstance(cards, str):
            return [c for c in cards if c in INDEX]
        return [EnvCard2RealCard.get(c, str(c)) for c in cards if c in EnvCard2RealCard]

    def _env_cards_to_real_str(self, cards):
        return "".join(self._env_cards_to_real_list(cards))

    def _action_key(self, action):
        return tuple(sorted(action))

    def _key_to_action(self, key, legal_actions):
        for a in legal_actions:
            if self._action_key(a) == key:
                return a
        return None

    def _same_team(self, p1, p2):
        if p1 is None or p2 is None:
            return False
        if isinstance(p1, int):
            if 0 <= p1 < len(ALL_POSITIONS):
                p1 = ALL_POSITIONS[p1]
            else:
                return False
        if isinstance(p2, int):
            if 0 <= p2 < len(ALL_POSITIONS):
                p2 = ALL_POSITIONS[p2]
            else:
                return False
        if p1 == "landlord" or p2 == "landlord":
            return p1 == p2
        return True

    def _teammate_position(self):
        if self.position == "landlord":
            return None
        return "landlord_up" if self.position == "landlord_down" else "landlord_down"

    def _choose_lowest_cost_action(self, actions, hand):
        def cost(a):
            if not a:
                return 0
            s = self._env_cards_to_real_str(a)
            return len(s) * 0.5 + self._main_rank_value(s) * 0.1
        return min(actions, key=cost)

    def _greedy_action(self, legal_actions, infoset):
        my_hand = self._env_cards_to_real_str(infoset.player_hand_cards)
        finish = [a for a in legal_actions if a != [] and len(self._env_cards_to_real_str(a)) == len(my_hand)]
        if finish:
            return finish[0]
        non_pass = [a for a in legal_actions if a != []]
        if non_pass:
            return min(non_pass, key=lambda a: len(self._env_cards_to_real_str(a)))
        return [] if [] in legal_actions else legal_actions[0]

    def _is_leading_round(self, infoset):
        last_two = getattr(infoset, "last_two_moves", [[], []])
        return len(last_two[0]) == 0 and len(last_two[1]) == 0

    def _is_teammate_last_player(self, infoset):
        if self.position == "landlord":
            return False
        last_pid = infoset.last_pid
        if last_pid is None:
            return False
        if isinstance(last_pid, int):
            if 0 <= last_pid < len(ALL_POSITIONS):
                last_pid = ALL_POSITIONS[last_pid]
            else:
                return False
        return last_pid == self._teammate_position()

    def _teammate_near_finish(self, infoset):
        if self.position == "landlord":
            return False
        teammate = self._teammate_position()
        if not teammate:
            return False
        left = getattr(infoset, "num_cards_left_dict", {})
        return left.get(teammate, 17) <= self.cfg["danger_cards"]

    def _is_teammate_last_player_from_state(self, state):
        if self.position == "landlord":
            return False
        last = state.last_pid
        if last is None:
            return False
        return last == self._teammate_position()


# ========== 内部状态类（修复类型问题） ==========
class _GameState:
    _legal_cache = {}
    _players_order = ['landlord', 'landlord_down', 'landlord_up']

    def __init__(self, hand_cards, current_player, last_move, last_pid, bomb_num=0):
        self.hand_cards = copy.deepcopy(hand_cards)
        self.current_player = current_player
        self.last_move = last_move if last_move else ()
        self.last_pid = last_pid
        self.bomb_num = bomb_num

    @staticmethod
    def _normalize_cards(cards):
        """将任意形式的牌（字符串、字符串列表、整数列表）统一转为环境整数列表"""
        if cards is None:
            return []
        if isinstance(cards, str):
            return [RealCard2EnvCard[c] for c in cards if c in RealCard2EnvCard]
        result = []
        for c in cards:
            if isinstance(c, int):
                result.append(c)
            elif isinstance(c, str) and c in RealCard2EnvCard:
                result.append(RealCard2EnvCard[c])
        return sorted(result)

    @classmethod
    def from_infoset_with_assign(cls, infoset, hand_assign, root_position):
        hand_cards = {}
        for pos in cls._players_order:
            if pos == root_position:
                raw = infoset.player_hand_cards
            elif pos in hand_assign:
                raw = hand_assign[pos]
            else:
                raw = infoset.all_handcards.get(pos, [])
            cards = cls._normalize_cards(raw)
            hand_cards[pos] = cards

        last_move = infoset.last_move if infoset.last_move else []
        if last_move and isinstance(last_move[0], str):
            last_move = [RealCard2EnvCard[c] for c in last_move if c in RealCard2EnvCard]
        last_move = tuple(sorted(last_move))

        raw_last_pid = infoset.last_pid
        if isinstance(raw_last_pid, int) and 0 <= raw_last_pid < len(cls._players_order):
            last_pid = cls._players_order[raw_last_pid]
        else:
            last_pid = raw_last_pid

        bomb_num = getattr(infoset, 'bomb_num', 0)
        return cls(hand_cards, infoset.player_position, last_move, last_pid, bomb_num)

    def _next_player(self):
        idx = self._players_order.index(self.current_player)
        return self._players_order[(idx + 1) % 3]

    def get_legal_actions(self):
        player = self.current_player
        hand = self.hand_cards[player]
        hand = self._normalize_cards(hand)
        self.hand_cards[player] = hand
        if not hand:
            return []

        if self.last_move and isinstance(self.last_move[0], str):
            self.last_move = tuple(self._normalize_cards(self.last_move))
        last_move = self.last_move

        hand_key = tuple(sorted(hand))
        last_move_key = last_move
        key = (hand_key, last_move_key, player)
        if key in self._legal_cache:
            return list(self._legal_cache[key])

        mg = MovesGener(hand)
        all_moves = mg.gen_moves()
        all_moves.extend(mg.gen_type_4_bomb())
        all_moves.extend(mg.gen_type_5_king_bomb())

        if not self.last_move:
            moves = []
            seen = set()
            for m in all_moves:
                m_tuple = tuple(sorted(m))
                if m_tuple not in seen:
                    seen.add(m_tuple)
                    moves.append(m_tuple)
            result = moves
        else:
            rival_move = list(self.last_move)
            rival_type_info = md.get_move_type(rival_move)
            rival_type = rival_type_info['type']
            rival_len = rival_type_info.get('len', 1)

            same_type_moves = []
            for m in all_moves:
                m_type_info = md.get_move_type(m)
                if m_type_info['type'] == rival_type:
                    if rival_type in (md.TYPE_8_SERIAL_SINGLE, md.TYPE_9_SERIAL_PAIR,
                                      md.TYPE_10_SERIAL_TRIPLE, md.TYPE_11_SERIAL_3_1,
                                      md.TYPE_12_SERIAL_3_2):
                        if m_type_info.get('len', 1) == rival_len:
                            same_type_moves.append(m)
                    else:
                        same_type_moves.append(m)

            if rival_type == md.TYPE_1_SINGLE:
                moves = ms.filter_type_1_single(same_type_moves, rival_move)
            elif rival_type == md.TYPE_2_PAIR:
                moves = ms.filter_type_2_pair(same_type_moves, rival_move)
            elif rival_type == md.TYPE_3_TRIPLE:
                moves = ms.filter_type_3_triple(same_type_moves, rival_move)
            elif rival_type == md.TYPE_4_BOMB:
                moves = ms.filter_type_4_bomb(same_type_moves, rival_move)
            elif rival_type == md.TYPE_5_KING_BOMB:
                moves = []
            elif rival_type == md.TYPE_6_3_1:
                moves = ms.filter_type_6_3_1(same_type_moves, rival_move)
            elif rival_type == md.TYPE_7_3_2:
                moves = ms.filter_type_7_3_2(same_type_moves, rival_move)
            elif rival_type == md.TYPE_8_SERIAL_SINGLE:
                moves = ms.filter_type_8_serial_single(same_type_moves, rival_move)
            elif rival_type == md.TYPE_9_SERIAL_PAIR:
                moves = ms.filter_type_9_serial_pair(same_type_moves, rival_move)
            elif rival_type == md.TYPE_10_SERIAL_TRIPLE:
                moves = ms.filter_type_10_serial_triple(same_type_moves, rival_move)
            elif rival_type == md.TYPE_11_SERIAL_3_1:
                moves = ms.filter_type_11_serial_3_1(same_type_moves, rival_move)
            elif rival_type == md.TYPE_12_SERIAL_3_2:
                moves = ms.filter_type_12_serial_3_2(same_type_moves, rival_move)
            else:
                moves = []

            if not moves:
                result = [()]
            else:
                unique = []
                seen = set()
                for m in moves:
                    m_tuple = tuple(sorted(m))
                    if m_tuple not in seen:
                        seen.add(m_tuple)
                        unique.append(m_tuple)
                result = unique + [()]

        if not result:
            result = [()]
        self._legal_cache[key] = tuple(result)
        return list(result)

    def step(self, action):
        player = self.current_player
        if action:
            for card in action:
                if card in self.hand_cards[player]:
                    self.hand_cards[player].remove(card)
                else:
                    # 如果牌不在手牌中，打印警告并尝试继续（避免崩溃）
                    print(f"Warning: Card {card} not in hand {self.hand_cards[player]}, skipping removal.")
            self.last_move = action
            self.last_pid = player
            if len(action) == 4 and len(set(action)) == 1 or (len(action) == 2 and set(action) == {20, 30}):
                self.bomb_num += 1
        self.current_player = self._next_player()

    def is_done(self):
        return any(len(cards) == 0 for cards in self.hand_cards.values())

    def get_winner(self):
        return 'landlord' if len(self.hand_cards['landlord']) == 0 else 'farmer'
