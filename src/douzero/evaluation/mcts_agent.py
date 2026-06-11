import random
import copy
import math
from douzero.env.move_generator import MovesGener
from douzero.env import move_detector as md
from douzero.env import move_selector as ms


class MCTSAgent:
    """经典蒙特卡洛树搜索智能体，使用 UCB 选择、树节点和回溯"""

    def __init__(self, num_simulations=200, position=None, c=1.414, max_rollout_steps=100):
        self.num_simulations = num_simulations
        self.position = position
        self.c = c
        self.max_rollout_steps = max_rollout_steps
        self.name = 'MCTS'

    class Node:
        __slots__ = ['state', 'parent', 'children', 'n', 'w', 'untried_actions']

        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent
            self.children = {}
            self.n = 0
            self.w = 0
            self.untried_actions = state.get_legal_actions()

        def is_fully_expanded(self):
            return len(self.untried_actions) == 0

        def best_child(self, c):
            if not self.children:
                return None
            log_parent = math.log(self.n)
            best_score = -float('inf')
            best_child = None
            for child in self.children.values():
                exploit = child.w / child.n
                explore = c * math.sqrt(log_parent / child.n)
                score = exploit + explore
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child

    def act(self, infoset):
        # 预处理：如果只有一个合法动作，直接返回
        legal_actions = infoset.legal_actions
        if not legal_actions:
            return []
        if len(legal_actions) == 1:
            return legal_actions[0]

        root_state = _GameState.from_infoset(infoset)
        root = self.Node(root_state)

        # MCTS 迭代
        for _ in range(self.num_simulations):
            node = self._select(root)
            abs_reward = self._simulate(node.state)
            self._backup(node, abs_reward)

        # 如果没有扩展出任何子节点（例如所有合法动作都无法模拟），回退到环境的第一个动作
        if not root.children:
            return legal_actions[0]

        best_action_tuple = max(root.children.items(), key=lambda x: x[1].n)[0]
        # 将元组转换为列表（环境期望的格式）
        best_action = list(best_action_tuple) if best_action_tuple else []
        return best_action

    def _select(self, node):
        # 如果游戏已结束，直接返回 node
        if node.state.is_done():
            return node
        # 向下选择直到找到一个未完全扩展的节点或叶子节点
        while not node.state.is_done() and node.is_fully_expanded():
            child = node.best_child(self.c)
            if child is None:
                # 没有子节点，停止选择
                break
            node = child
        # 如果当前节点还有未扩展的动作，则扩展一个
        if not node.state.is_done() and node.untried_actions:
            return self._expand(node)
        return node

    def _expand(self, node):
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        new_state = copy.deepcopy(node.state)
        new_state.step(action)

        child_node = self.Node(new_state, parent=node)
        node.children[action] = child_node
        return child_node

    def _simulate(self, state):
        sim_state = copy.deepcopy(state)
        steps = 0
        while not sim_state.is_done() and steps < self.max_rollout_steps:
            legal_actions = sim_state.get_legal_actions()
            if not legal_actions:
                break
            action = self._select_random_action(legal_actions)
            sim_state.step(action)
            steps += 1

        if not sim_state.is_done():
            return self._heuristic_abs_reward(sim_state)

        winner = sim_state.get_winner()
        return 1 if winner == 'landlord' else -1

    def _select_random_action(self, actions):
        non_bomb = [a for a in actions if not self._is_bomb_or_rocket(a)]
        if non_bomb:
            return random.choice(non_bomb)
        return random.choice(actions)

    @staticmethod
    def _is_bomb_or_rocket(action):
        if not action:
            return False
        if len(action) == 2 and set(action) == {20, 30}:
            return True
        if len(action) == 4 and len(set(action)) == 1:
            return True
        return False

    def _heuristic_abs_reward(self, state):
        landlord_cards = len(state.hand_cards['landlord'])
        farmer_up_cards = len(state.hand_cards['landlord_up'])
        farmer_down_cards = len(state.hand_cards['landlord_down'])
        value = (20 - landlord_cards) - (20 - min(farmer_up_cards, farmer_down_cards))
        value = value / 20.0
        return max(-1, min(1, value))

    def _backup(self, node, abs_reward):
        while node is not None:
            node.n += 1
            if node.state.current_player == 'landlord':
                node.w += abs_reward
            else:
                node.w += -abs_reward
            node = node.parent


class _GameState:
    _legal_cache = {}
    _players_order = ['landlord', 'landlord_down', 'landlord_up']

    def __init__(self, hand_cards, current_player, last_move, last_player, bomb_num=0):
        self.hand_cards = copy.deepcopy(hand_cards)
        self.current_player = current_player
        self.last_move = last_move if last_move else ()
        self.last_player = last_player
        self.bomb_num = bomb_num

    @classmethod
    def from_infoset(cls, infoset):
        hand_cards = infoset.all_handcards
        current_player = infoset.player_position
        last_move = infoset.last_move if infoset.last_move else []
        last_move_tuple = tuple(last_move) if last_move else ()
        last_player = infoset.last_pid
        bomb_num = infoset.bomb_num
        return cls(hand_cards, current_player, last_move_tuple, last_player, bomb_num)

    def _next_player(self):
        idx = self._players_order.index(self.current_player)
        return self._players_order[(idx + 1) % 3]

    def get_legal_actions(self):
        player = self.current_player
        hand = self.hand_cards[player]
        if not hand:
            # 当前玩家手牌为空，游戏应该已经结束，但为了安全返回空列表
            return []

        hand_key = tuple(sorted(hand))
        last_move_key = self.last_move
        key = (hand_key, last_move_key, player)

        if key in self._legal_cache:
            return self._legal_cache[key]

        mg = MovesGener(hand)
        all_moves = mg.gen_moves()
        all_moves.extend(mg.gen_type_4_bomb())
        all_moves.extend(mg.gen_type_5_king_bomb())

        if not self.last_move:
            # 主动出牌
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
                if not m:
                    continue
                m_type_info = md.get_move_type(m)
                if m_type_info['type'] == rival_type:
                    if rival_type in (md.TYPE_8_SERIAL_SINGLE, md.TYPE_9_SERIAL_PAIR,
                                      md.TYPE_10_SERIAL_TRIPLE, md.TYPE_11_SERIAL_3_1,
                                      md.TYPE_12_SERIAL_3_2):
                        if m_type_info.get('len', 1) == rival_len:
                            same_type_moves.append(m)
                    else:
                        same_type_moves.append(m)

            # 压制过滤器
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
                result = [()]   # 只能 Pass
            else:
                unique = []
                seen = set()
                for m in moves:
                    m_tuple = tuple(sorted(m))
                    if m_tuple not in seen:
                        seen.add(m_tuple)
                        unique.append(m_tuple)
                result = unique + [()]  # 加 Pass

        # 如果 result 为空（理论上不应该），至少保证有一个 Pass
        if not result:
            result = [()]

        self._legal_cache[key] = result
        return result

    def step(self, action):
        player = self.current_player
        if action:  # 非 Pass
            for card in action:
                self.hand_cards[player].remove(card)
            self.last_move = action
            self.last_player = player
        self.current_player = self._next_player()

    def is_done(self):
        return any(len(cards) == 0 for cards in self.hand_cards.values())

    def get_winner(self):
        return 'landlord' if len(self.hand_cards['landlord']) == 0 else 'farmer'
