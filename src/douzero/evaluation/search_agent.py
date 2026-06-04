import random
from copy import deepcopy

from ..env.game import GameEnv


class SearchAgent:
    """A simple Monte-Carlo search agent that selects actions by rollouts.

    For each legal action it performs N random rollouts (other players act
    uniformly at random) and picks the action with highest empirical
    win-rate for the acting player's team.
    """

    def __init__(self, num_rollouts=30):
        self.num_rollouts = int(num_rollouts)
        self.name = f'SearchAgent({self.num_rollouts})'

    def act(self, infoset):
        # If there's only one legal action, return it quickly
        legal = infoset.legal_actions
        if len(legal) == 1:
            return legal[0]

        # Prepare a lightweight GameEnv snapshot from the infoset
        # We'll create a GameEnv with Random agents for simulation.
        from .random_agent import RandomAgent

        players = {
            'landlord': RandomAgent(),
            'landlord_up': RandomAgent(),
            'landlord_down': RandomAgent()
        }

        # Helper to build a fresh env based on the current infoset
        def build_env_from_infoset():
            env = GameEnv(players)
            # copy global sequences and cards
            env.card_play_action_seq = deepcopy(infoset.card_play_action_seq)
            env.three_landlord_cards = deepcopy(infoset.three_landlord_cards)
            env.played_cards = deepcopy(infoset.played_cards)
            env.last_move_dict = deepcopy(infoset.last_move_dict)
            env.bomb_num = deepcopy(infoset.bomb_num) if hasattr(infoset, 'bomb_num') else 0
            env.last_pid = deepcopy(infoset.last_pid) if hasattr(infoset, 'last_pid') else 'landlord'

            # set hand cards for all players
            for pos in ['landlord', 'landlord_up', 'landlord_down']:
                env.info_sets[pos].player_hand_cards = deepcopy(infoset.all_handcards[pos])

            # set acting player
            env.acting_player_position = infoset.player_position

            # Ensure game_infoset is consistent when env.step() is called
            env.game_infoset = env.get_infoset()
            return env

        # For each candidate action, run rollouts and record wins
        scores = []
        for action in legal:
            wins = 0
            for _ in range(self.num_rollouts):
                env = build_env_from_infoset()

                # apply the candidate action for the current acting player
                env.card_play_action_seq.append(deepcopy(action))
                # update bomb count and last_move_dict/playeds
                if action in env.get_legal_card_play_actions() and action in env.get_legal_card_play_actions():
                    pass
                if len(action) > 0:
                    env.last_pid = env.acting_player_position

                if action in env.get_legal_card_play_actions():
                    pass

                if action in env.card_play_action_seq[-1:]:
                    pass

                # Use GameEnv methods to remove cards and update internal state
                env.update_acting_player_hand_cards(action)
                env.played_cards[env.acting_player_position] += deepcopy(action)
                if env.acting_player_position == 'landlord' and len(action) > 0 and env.three_landlord_cards:
                    for card in action:
                        if card in env.three_landlord_cards:
                            env.three_landlord_cards.remove(card)

                if action in env.get_legal_card_play_actions():
                    pass

                if action in env.card_play_action_seq[-1:]:
                    pass

                if action in env.card_play_action_seq[-1:]:
                    pass

                # update bomb count
                from ..env.game import bombs
                if action in bombs:
                    env.bomb_num += 1

                # check terminal condition after applying action
                env.game_done()

                # continue simulation with random agents
                while not env.game_over:
                    env.get_acting_player_position()
                    env.game_infoset = env.get_infoset()
                    env.step()

                winner = env.get_winner()
                # map positions to teams: landlord vs farmers
                acting_team = 'landlord' if infoset.player_position == 'landlord' else 'farmer'
                if acting_team == 'landlord':
                    if winner == 'landlord':
                        wins += 1
                else:
                    if winner == 'farmer':
                        wins += 1

            scores.append(wins / max(1, self.num_rollouts))

        # pick the action with highest score (break ties randomly)
        best_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if s == best_score]
        choice_idx = random.choice(best_indices)
        return legal[choice_idx]
