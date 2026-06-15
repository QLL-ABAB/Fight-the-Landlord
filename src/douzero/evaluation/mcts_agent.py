# mcts_agent.py
# Enhanced MCTS for Dou Dizhu – with determinization, bomb-aware, leading/following differentiation,
# and explicit farmer cooperation.

import random
import copy
import math
import time
from collections import Counter

from douzero.env.move_generator import MovesGener
from douzero.env import move_detector as md
from douzero.env import move_selector as ms

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
                 max_rollout_steps=50, time_budget=0.2,
                 num_determinizations=8, objective="logadp"):
        self.num_simulations = num_simulations
        self.position = position
        self.c = c
        self.max_rollout_steps = max_rollout_steps
        self.time_budget = time_budget
        self.num_determinizations = num_determinizations
        self.objective = objective
        self.name = "MCTS_Enhanced"

        self._legal_cache = {}

        self.cfg = {
            "eval_badness_weight": 1.0,
            "eval_count_weight": 7.0,
            "eval_enemy_danger_penalty": 180.0,
            "eval_team_danger_bonus": 120.0,
            "eval_initiative_bonus": 16.0,
            "eval_last_pid_bonus": 24.0,
            "eval_bomb_bonus": 16.0,
            "eval_control_bonus": 3.5,
            "badness_turn_weight": 7.0,
            "badness_single_weight": 3.0,
            "badness_len_weight": 1.2,
            "badness_chain_discount_weight": 5.0,
            "badness_control_discount": 2.2,
            "badness_bomb_discount": 3.0,
            "solo_chain_coef": 1.0,
            "pair_chain_coef": 1.5,
            "trio_chain_coef": 2.0,
            "danger_cards": 2,
            "very_danger_cards": 1,
            "leading_c_multiplier": 0.8,
            "following_c_multiplier": 1.2,
            "leading_initiative_bonus": 25.0,
            "following_beat_bonus": 10.0,
            "teammate_release_bonus": 50.0,
            "teammate_block_penalty": 35.0,
        }

    # ================== 叫牌（可选） ==================
    def bid(self, hand_cards, three_landlord_cards=None):
        hand_str = self._env_cards_to_real_str(hand_cards)
        if three_landlord_cards:
            hand_str += self._env_cards_to_real_str(three_landlord_cards)
        bad = self._hand_badness(hand_str)
        counts = Counter(hand_str)
        bombs = sum(1 for c in NORMAL_CHAIN_ORDER + ["2"] if counts.get(c, 0) >= 4)
        rockets = 1 if counts.get("B", 0) and counts.get("R", 0) else 0
        control = sum(cnt for c, cnt in counts.items() if c in ["A", "2", "B", "R"])
        strength = -bad + 5 * bombs + 8 * rockets + 0.5 * control
        if strength > 35:
            return 3
        elif strength > 20:
            return 2
        elif strength > 8:
            return 1
        return 0

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
                child_node = self.Node(state, a)
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
        __slots__ = ['state', 'action', 'parent', 'children', 'n', 'w', 'untried_actions']
        def __init__(self, state, action, parent=None):
            self.state = state
            self.action = action
            self.parent = parent
            self.children = {}
            self.n = 0
            self.w = 0
            self.untried_actions = []

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
        child = self.Node(new_state, action, parent=node)
        node.children[action_key] = child
        return child

    def _best_child(self, node, c):
        if not node.children:
            return None
        log_parent = math.log(node.n) if node.n > 0 else 0.0
        best_score = -float('inf')
        best = None
        for child in node.children.values():
            exploit = child.w / child.n if child.n > 0 else 0.0
            explore = c * math.sqrt(log_parent / child.n) if child.n > 0 else float('inf')
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best

    def _simulate(self, state, is_leading):
        sim_state = copy.deepcopy(state)
        steps = 0
        while not sim_state.is_done() and steps < self.max_rollout_steps:
            legal = sim_state.get_legal_actions()
            if not legal:
                break
            action = self._heuristic_rollout_policy(sim_state, legal, is_leading)
            sim_state.step(action)
            steps += 1
        if not sim_state.is_done():
            return self._evaluate_state(sim_state, is_leading)
        winner = sim_state.get_winner()
        bomb_num = sim_state.bomb_num
        return self._terminal_value(winner, bomb_num)

    def _backup(self, node, value, current_player, is_leading):
        while node is not None:
            node.n += 1
            if node.action is not None and node.parent is not None:
                actor = node.parent.state.current_player if node.parent.state else current_player
                # 农民协作惩罚：队友压制队友
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

    # ================== 启发式评估 ==================
    def _evaluate_state(self, state, is_leading):
        hands = state.hand_cards
        landlord = "landlord"
        farmers = ["landlord_down", "landlord_up"]

        def good_value(pos):
            h = "".join(self._env_cards_to_real_list(hands[pos]))
            return -self.cfg["eval_badness_weight"] * self._hand_badness(h) \
                   - self.cfg["eval_count_weight"] * len(h)

        if self.position == landlord:
            root_team_good = good_value(landlord)
            enemy_good = max(good_value(farmers[0]), good_value(farmers[1]))
            val = root_team_good - enemy_good
        else:
            teammate = self._teammate_position()
            root_team_good = max(good_value(self.position), good_value(teammate))
            enemy_good = good_value(landlord)
            val = root_team_good - enemy_good

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

        if self._same_team(state.current_player, self.position) and not state.last_move:
            val += self.cfg["eval_initiative_bonus"]
            if is_leading:
                val += self.cfg["leading_initiative_bonus"]
        if state.last_pid is not None and self._same_team(state.last_pid, self.position):
            val += self.cfg["eval_last_pid_bonus"]

        if (not is_leading and state.last_move and state.last_pid is not None and
            not self._same_team(state.last_pid, self.position) and
            self._same_team(state.current_player, self.position)):
            val += self.cfg["following_beat_bonus"]

        if self.position != "landlord":
            teammate = self._teammate_position()
            if teammate and len(hands.get(teammate, [])) <= self.cfg["teammate_release_bonus"]:
                val += self.cfg["teammate_release_bonus"] / 10.0

        for p in ALL_POSITIONS:
            sign = 1.0 if self._same_team(p, self.position) else -1.0
            val += sign * self.cfg["eval_bomb_bonus"] * self._count_bombs(hands[p])
            val += sign * self.cfg["eval_control_bonus"] * self._control_count(hands[p])

        return max(-1.0, min(1.0, val / 200.0))

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

    # ================== 辅助函数（牌型、手牌、采样等） ==================
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

    def _sample_determinizations(self, infoset, n):
        my_hand = self._env_cards_to_real_str(infoset.player_hand_cards)
        unknown_counter = self._get_unknown_cards(infoset, my_hand)
        unknown_cards = []
        for c in CARD_ORDER:
            unknown_cards.extend([c] * unknown_counter[c])

        num_left = {}
        for pos in ALL_POSITIONS:
            num_left[pos] = len(infoset.all_handcards[pos]) if pos in infoset.all_handcards else 0

        targets = {}
        for p in ALL_POSITIONS:
            if p != self.position:
                targets[p] = num_left.get(p, 0)
        total_need = sum(targets.values())
        if total_need > len(unknown_cards):
            overflow = total_need - len(unknown_cards)
            for p in sorted(targets, key=lambda x: targets[x], reverse=True):
                take = min(overflow, targets[p])
                targets[p] -= take
                overflow -= take
                if overflow <= 0:
                    break

        worlds = []
        for _ in range(n):
            cards = unknown_cards[:]
            random.shuffle(cards)
            assign = {p: "" for p in ALL_POSITIONS}
            assign[self.position] = my_hand
            idx = 0
            for p in ALL_POSITIONS:
                if p == self.position:
                    continue
                cnt = targets.get(p, 0)
                assign[p] = "".join(cards[idx:idx+cnt])
                idx += cnt
            worlds.append(assign)
        return worlds

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
        return True  # 两个农民同队

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


# ========== 内部状态类（必须放在类外） ==========
class _GameState:
    _legal_cache = {}
    _players_order = ['landlord', 'landlord_down', 'landlord_up']

    def __init__(self, hand_cards, current_player, last_move, last_pid, bomb_num=0):
        self.hand_cards = copy.deepcopy(hand_cards)
        self.current_player = current_player
        self.last_move = last_move if last_move else ()
        self.last_pid = last_pid
        self.bomb_num = bomb_num

    @classmethod
    def from_infoset_with_assign(cls, infoset, hand_assign, root_position):
        hand_cards = {}
        for pos in cls._players_order:
            if pos in hand_assign:
                cards = [RealCard2EnvCard[c] for c in hand_assign[pos] if c in RealCard2EnvCard]
                cards.sort()
                hand_cards[pos] = cards
            else:
                hand_cards[pos] = infoset.all_handcards.get(pos, []).copy()
        current_player = infoset.player_position
        last_move = infoset.last_move if infoset.last_move else []
        raw_last_pid = infoset.last_pid
        if isinstance(raw_last_pid, int):
            if 0 <= raw_last_pid < len(cls._players_order):
                last_pid = cls._players_order[raw_last_pid]
            else:
                last_pid = None
        else:
            last_pid = raw_last_pid
        bomb_num = infoset.bomb_num
        return cls(hand_cards, current_player, tuple(last_move), last_pid, bomb_num)

    def _next_player(self):
        idx = self._players_order.index(self.current_player)
        return self._players_order[(idx + 1) % 3]

    def get_legal_actions(self):
        player = self.current_player
        hand = self.hand_cards[player]
        if not hand:
            return []

        hand_key = tuple(sorted(hand))
        last_move_key = self.last_move
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
                self.hand_cards[player].remove(card)
            self.last_move = action
            self.last_pid = player
            if len(action) == 4 and len(set(action)) == 1 or (len(action) == 2 and set(action) == {20, 30}):
                self.bomb_num += 1
        self.current_player = self._next_player()

    def is_done(self):
        return any(len(cards) == 0 for cards in self.hand_cards.values())

    def get_winner(self):
        return 'landlord' if len(self.hand_cards['landlord']) == 0 else 'farmer'
