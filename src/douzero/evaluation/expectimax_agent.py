import random
from copy import deepcopy

from ..env.game import GameEnv


class ExpectimaxAgent:
    """A simple expectimax agent with limited depth.

    At max nodes (the acting player's turns) it picks the action with
    highest expected value. At opponent nodes it takes the expectation
    (assumes opponents choose uniformly among legal actions). Leaves
    are evaluated with a small number of random rollouts.
    """

    def __init__(self, depth=2, rollout_per_leaf=8):
        self.depth = int(depth)
        self.rollout_per_leaf = int(rollout_per_leaf)
        self.name = f'Expectimax(d={self.depth},r={self.rollout_per_leaf})'

    def act(self, infoset):
        from .random_agent import RandomAgent

        # Build an env snapshot from the infoset
        def build_env():
            players = {
                'landlord': RandomAgent(),
                'landlord_up': RandomAgent(),
                'landlord_down': RandomAgent()
            }
            env = GameEnv(players)
            env.card_play_action_seq = deepcopy(infoset.card_play_action_seq)
            env.three_landlord_cards = deepcopy(infoset.three_landlord_cards)
            env.played_cards = deepcopy(infoset.played_cards)
            env.last_move_dict = deepcopy(infoset.last_move_dict)
            env.bomb_num = deepcopy(infoset.bomb_num) if hasattr(infoset, 'bomb_num') else 0
            env.last_pid = deepcopy(infoset.last_pid) if hasattr(infoset, 'last_pid') else 'landlord'
            for pos in ['landlord', 'landlord_up', 'landlord_down']:
                env.info_sets[pos].player_hand_cards = deepcopy(infoset.all_handcards[pos])
            env.acting_player_position = infoset.player_position
            env.game_infoset = env.get_infoset()
            return env

        root_env = build_env()
        legal = infoset.legal_actions
        if len(legal) == 1:
            return legal[0]

        # identify team of root agent for scoring
        root_team = 'landlord' if infoset.player_position == 'landlord' else 'farmer'

        def rollout(env_snapshot):
            # random play until terminal
            while not env_snapshot.game_over:
                env_snapshot.get_acting_player_position()
                env_snapshot.game_infoset = env_snapshot.get_infoset()
                env_snapshot.step()
            winner = env_snapshot.get_winner()
            return 1 if (root_team == 'landlord' and winner == 'landlord') or (root_team == 'farmer' and winner == 'farmer') else 0

        def leaf_value(env_snapshot):
            wins = 0
            for _ in range(self.rollout_per_leaf):
                sim = deepcopy(env_snapshot)
                wins += rollout(sim)
            return wins / max(1, self.rollout_per_leaf)

        def expectimax_value(env_snapshot, depth):
            if env_snapshot.game_over:
                winner = env_snapshot.get_winner()
                return 1.0 if (root_team == 'landlord' and winner == 'landlord') or (root_team == 'farmer' and winner == 'farmer') else 0.0

            if depth <= 0:
                return leaf_value(env_snapshot)

            acting = env_snapshot.acting_player_position
            legal_actions = env_snapshot.get_legal_card_play_actions()
            if len(legal_actions) == 0:
                # no legal actions? unlikely, but fallback to rollout
                return leaf_value(env_snapshot)

            # If it's the root agent's team and position, treat as max node
            is_max_node = (acting == infoset.player_position)

            if is_max_node:
                best = -1.0
                for a in legal_actions:
                    child = deepcopy(env_snapshot)
                    # apply action
                    child.card_play_action_seq.append(deepcopy(a))
                    child.update_acting_player_hand_cards(a)
                    child.played_cards[child.acting_player_position] += deepcopy(a)
                    # update three landlord cards if needed
                    if child.acting_player_position == 'landlord' and len(a) > 0 and child.three_landlord_cards:
                        for card in a:
                            if card in child.three_landlord_cards:
                                child.three_landlord_cards.remove(card)
                    from ..env.game import bombs
                    if a in bombs:
                        child.bomb_num += 1
                    child.game_done()
                    if not child.game_over:
                        child.get_acting_player_position()
                        child.game_infoset = child.get_infoset()
                    val = expectimax_value(child, depth - 1)
                    if val > best:
                        best = val
                return best
            else:
                # expectation over uniformly-chosen opponent actions
                total = 0.0
                for a in legal_actions:
                    child = deepcopy(env_snapshot)
                    child.card_play_action_seq.append(deepcopy(a))
                    child.update_acting_player_hand_cards(a)
                    child.played_cards[child.acting_player_position] += deepcopy(a)
                    if child.acting_player_position == 'landlord' and len(a) > 0 and child.three_landlord_cards:
                        for card in a:
                            if card in child.three_landlord_cards:
                                child.three_landlord_cards.remove(card)
                    from ..env.game import bombs
                    if a in bombs:
                        child.bomb_num += 1
                    child.game_done()
                    if not child.game_over:
                        child.get_acting_player_position()
                        child.game_infoset = child.get_infoset()
                    total += expectimax_value(child, depth - 1)
                return total / len(legal_actions)

        # Evaluate each root action using expectimax
        best_score = -1.0
        best_actions = []
        for a in legal:
            child = deepcopy(root_env)
            child.card_play_action_seq.append(deepcopy(a))
            child.update_acting_player_hand_cards(a)
            child.played_cards[child.acting_player_position] += deepcopy(a)
            if child.acting_player_position == 'landlord' and len(a) > 0 and child.three_landlord_cards:
                for card in a:
                    if card in child.three_landlord_cards:
                        child.three_landlord_cards.remove(card)
            from ..env.game import bombs
            if a in bombs:
                child.bomb_num += 1
            child.game_done()
            if not child.game_over:
                child.get_acting_player_position()
                child.game_infoset = child.get_infoset()
            val = expectimax_value(child, self.depth - 1)
            if val > best_score:
                best_score = val
                best_actions = [a]
            elif val == best_score:
                best_actions.append(a)

        return random.choice(best_actions)
