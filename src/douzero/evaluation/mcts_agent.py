import random
import copy
from douzero.env.game import EnvCard2RealCard, RealCard2EnvCard
from douzero.env.move_generator import MovesGener
from douzero.env import move_detector as md
from douzero.env import move_selector as ms
from douzero.env.utils import TYPE_0_PASS, TYPE_4_BOMB, TYPE_5_KING_BOMB


class MCTSAgent:
    """基于蒙特卡洛树搜索的智能体，使用随机模拟评估每个合法动作的价值"""

    def __init__(self, num_simulations=50, position=None, max_rollout_steps=100):
        """
        参数:
            num_simulations: 每个合法动作的模拟次数
            position: 当前玩家的角色 ('landlord', 'landlord_up', 'landlord_down')
            max_rollout_steps: 单次模拟的最大步数，防止过长模拟
        """
        self.num_simulations = num_simulations
        self.position = position
        self.max_rollout_steps = max_rollout_steps
        self.name = 'MCTS'

    def act(self, infoset):
        legal_actions = infoset.legal_actions
        if len(legal_actions) == 1:
            return legal_actions[0]

        action_scores = []
        for action in legal_actions:
            score = self._evaluate_action(infoset, action)
            action_scores.append((action, score))

        best_action = max(action_scores, key=lambda x: x[1])[0]
        return best_action

    def _evaluate_action(self, infoset, action):
        total_reward = 0.0
        for _ in range(self.num_simulations):
            reward = self._simulate(infoset, action)
            total_reward += reward
        return total_reward / self.num_simulations

    def _simulate(self, infoset, first_action):
        state = _GameState.from_infoset(infoset)
        state.step(first_action)
        steps = 0
        while not state.is_done() and steps < self.max_rollout_steps:
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                break
            action = self._select_random_action(legal_actions)
            state.step(action)
            steps += 1

        if not state.is_done():
            return self._heuristic_value(state)

        winner = state.get_winner()
        if infoset.player_position == 'landlord':
            return 1 if winner == 'landlord' else -1
        else:
            return 1 if winner == 'farmer' else -1

    def _select_random_action(self, actions):
        # 优先选择非炸弹动作，加速模拟
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

    def _heuristic_value(self, state):
        my_pos = self.position
        if my_pos == 'landlord':
            my_cards = len(state.hand_cards['landlord'])
            enemy_cards = min(len(state.hand_cards['landlord_up']), len(state.hand_cards['landlord_down']))
            value = (20 - my_cards) - (20 - enemy_cards)
            value = max(-1, min(1, value / 20.0))
            return value
        else:
            landlord_cards = len(state.hand_cards['landlord'])
            my_cards = len(state.hand_cards[my_pos])
            teammate = 'landlord_up' if my_pos == 'landlord_down' else 'landlord_down'
            teammate_cards = len(state.hand_cards[teammate])
            value = (landlord_cards - my_cards - teammate_cards) / 20.0
            return max(-1, min(1, value))


class _GameState:
    """
    轻量级游戏状态模拟器，用于 MCTS 的 rollout。
    维护当前各玩家的手牌、当前轮到谁、上一个有效动作等信息。
    优化：缓存动作生成结果，使用 lru_cache 加速。
    """
    _legal_cache = {}
    _players_order = ['landlord', 'landlord_down', 'landlord_up']

    def __init__(self, hand_cards, current_player, last_move, last_player, bomb_num=0):
        self.hand_cards = copy.deepcopy(hand_cards)
        self.current_player = current_player
        self.last_move = copy.deepcopy(last_move) if last_move else []
        self.last_player = last_player
        self.bomb_num = bomb_num

    @classmethod
    def from_infoset(cls, infoset):
        hand_cards = infoset.all_handcards
        current_player = infoset.player_position
        last_move = infoset.last_move if infoset.last_move else []
        last_player = infoset.last_pid
        bomb_num = infoset.bomb_num
        return cls(hand_cards, current_player, last_move, last_player, bomb_num)

    def _next_player(self):
        idx = self._players_order.index(self.current_player)
        next_idx = (idx + 1) % 3
        return self._players_order[next_idx]

    def get_legal_actions(self):
        player = self.current_player
        hand = self.hand_cards[player]
        if not hand:
            return []

        hand_key = tuple(sorted(hand))
        last_move_key = tuple(self.last_move) if self.last_move else ()
        key = (hand_key, last_move_key, player)

        if key in self._legal_cache:
            return self._legal_cache[key]

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
                    moves.append(m)
            result = moves
        else:
            rival_move = self.last_move
            rival_type_info = md.get_move_type(rival_move)
            rival_type = rival_type_info['type']
            rival_len = rival_type_info.get('len', 1)

            # 1. 先筛选出与上家同类型的动作
            same_type_moves = []
            for m in all_moves:
                if not m:
                    continue
                m_type_info = md.get_move_type(m)
                if m_type_info['type'] == rival_type:
                    if rival_type in (md.TYPE_8_SERIAL_SINGLE, md.TYPE_9_SERIAL_PAIR,
                                      md.TYPE_10_SERIAL_TRIPLE, md.TYPE_11_SERIAL_3_1,
                                      md.TYPE_12_SERIAL_3_2):
                        # 链式牌型还需要检查长度相同
                        if m_type_info.get('len', 1) == rival_len:
                            same_type_moves.append(m)
                    else:
                        same_type_moves.append(m)

            # 2. 根据 rival_type 调用对应的压制过滤器
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
                result = [[]]
            else:
                # 去重并加上 pass
                unique = []
                seen = set()
                for m in moves:
                    m_sorted = tuple(sorted(m))
                    if m_sorted not in seen:
                        seen.add(m_sorted)
                        unique.append(m)
                result = unique + [[]]

        self._legal_cache[key] = result
        return result

    def step(self, action):
        player = self.current_player
        if action:
            for card in action:
                self.hand_cards[player].remove(card)
            self.last_move = action
            self.last_player = player
        self.current_player = self._next_player()

    def is_done(self):
        return any(len(cards) == 0 for cards in self.hand_cards.values())

    def get_winner(self):
        return 'landlord' if len(self.hand_cards['landlord']) == 0 else 'farmer'