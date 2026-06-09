# value_dp_mdp_agent.py
# ------------------------------------------------------------
# A faster MDP-style Dou Dizhu agent with greedy-anchor safety.
#
# Compared with online Value Iteration:
#   - It does NOT run 80 Bellman iterations in every act().
#   - The reduced MDP is acyclic: hand -> hand - action.
#   - Therefore it uses memoized dynamic programming:
#
#       V(s) = max_a [ R_heur(s, a, s') + gamma * V(s') ]
#
#     where s is the player's remaining hand and s' is the hand
#     after removing action a.
#
# This is still NOT a full Dou Dizhu MDP/POMDP. It solves a reduced
# "self-hand decomposition" MDP and uses a greedy-anchor safety layer:
# default to a stable greedy action, and let DP override only when clearly better.
# ------------------------------------------------------------

import random
import math
from collections import Counter
from itertools import combinations

try:
    from rlcard.games.doudizhu.utils import CARD_TYPE
except Exception:
    CARD_TYPE = None


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


class ValueDPAgent(object):
    """
    Faster MDP-style agent with memoized DP.

    Public interface:
        agent = MemoizedValueDPAgent(position="landlord")
        action = agent.act(infoset)

    The reduced MDP is:
        state  s  = current player's remaining hand string, e.g. "334455A2"
        action a  = one playable combination generated from s
        next   s' = s - a
        reward    = handcrafted heuristic reward
        gamma     = discount factor

    Since every non-pass action strictly reduces the hand size, the reduced MDP
    is a DAG. Therefore recursive DP is faster and more appropriate than
    repeatedly running generic value iteration.
    """

    def __init__(self, position, debug=False):
        self.name = "ValueDP_MDPStyle_v3_greedy_anchor"
        self.position = position
        self.debug = debug

        self.cfg = {
            # ------------------------------
            # DP / reduced MDP parameters
            # ------------------------------
            "gamma": 0.94,
            "max_dp_states": 3500,
            "max_generated_actions_per_state": 90,
            "max_wing_combinations": 16,
            "cutoff_value_badness_weight": 0.85,
            "cutoff_value_len_weight": 0.35,

            # ------------------------------
            # heuristic reward R_heur(s,a,s')
            # ------------------------------
            "terminal_reward": 90.0,      # reward for entering empty hand
            "turn_penalty": -4.0,           # each move costs one turn
            "card_reward": 0.35,            # prefer playing more cards
            "structure_reward": 1.35,        # reward hand_badness improvement
            "min_card_bonus": 2.2,          # prefer taking away lowest cards
            "chain_bonus": 4.0,
            "chain_len_bonus": 0.45,
            "trio_bonus": 2.2,
            "pair_bonus": 1.0,
            "four_with_penalty": -4.0,
            "bomb_use_penalty": -24.0,
            "rocket_use_penalty": -28.0,
            "control_use_penalty_weight": 0.55,
            "finish_bonus_online": 70.0,

            # ------------------------------
            # online context adjustment
            # ------------------------------
            "pass_penalty": -7.0,
            "leading_pass_penalty": -9999.0,
            "danger_pass_penalty": -45.0,
            "teammate_release_cards": 2,
            "teammate_release_bonus": 80.0,
            "beat_teammate_penalty": 45.0,
            "enemy_danger_beat_bonus": 35.0,
            "unnecessary_bomb_follow_penalty": 48.0,
            "risk_penalty_weight": 5.0,
            "enemy_follow_beat_bonus": 13.0,
            "same_type_follow_bonus": 7.0,
            "ordinary_enemy_pass_penalty": -12.0,
            "leading_single_high_penalty": 0.18,
            "leading_bomb_extra_penalty": 18.0,
            # v3: greedy-anchor safety layer.
            # DP only overrides the greedy baseline when the estimated advantage
            # is large enough. This prevents the reduced MDP from overfitting to
            # self-hand decomposition and making tactically poor plays.
            "dp_override_margin_leading": 10.0,
            "dp_override_margin_following": 16.0,
            "dp_override_margin_enemy_danger": 3.0,
            "max_override_risk": 0.72,
            "allow_non_finish_bomb_override": False,
            "greedy_lead_bomb_penalty": 80.0,
            "greedy_lead_high_single_penalty": 0.40,
            "greedy_lead_hand_improve_weight": 7.0,
            "greedy_lead_len_weight": 1.0,
            "greedy_lead_chain_bonus": 10.0,
            "greedy_lead_trio_bonus": 5.0,
            "greedy_lead_pair_bonus": 2.0,

            # ------------------------------
            # state danger thresholds
            # ------------------------------
            "danger_cards": 2,
            "very_danger_cards": 1,

            # ------------------------------
            # Bayesian belief parameters
            # ------------------------------
            "rocket_prob": 0.25,
            "bomb_prob_per_possible": 0.13,
            "bomb_prob_cap": 0.65,
            "higher_bomb_risk_weight": 0.12,
            "risk_cap": 0.9,
            "bomb_risk_weight": 0.25,
            "rocket_risk_weight": 0.15,
            "same_type_risk_cap": 0.8,
            "same_type_exp_weight": 0.35,

            # ------------------------------
            # hand_badness parameters
            # ------------------------------
            "badness_turn_weight": 7.0,
            "badness_single_weight": 3.0,
            "badness_len_weight": 1.2,
            "badness_chain_discount_weight": 5.0,
            "badness_control_discount": 2.2,
            "badness_bomb_discount": 3.0,
            "solo_chain_coef": 1.0,
            "pair_chain_coef": 1.5,
            "trio_chain_coef": 2.0,

            # ------------------------------
            # action cost / tie breaking
            # ------------------------------
            "cost_len_weight": 0.5,
            "cost_rank_weight": 0.08,
            "cost_bomb_weight": 50.0,
            "A_cost": 1.5,
            "2_cost": 4.5,
            "B_cost": 7.0,
            "R_cost": 8.0,
            "danger_control_scale": 0.45,
        }

        # One-game/root cache. If the current hand is no longer a subset of
        # root_hand, we treat it as a new game and reset caches.
        self.root_hand = None
        self.value_cache = {"": 0.0}
        self.best_action_cache = {}
        self.action_cache = {}

        self.fallback_count = 0
        self.last_error = None
        self.stats = {
            "new_game_resets": 0,
            "value_cache_hits": 0,
            "value_cache_misses": 0,
            "cutoff_uses": 0,
        }

    # ============================================================
    # Public API
    # ============================================================

    def act(self, infoset):
        """
        Input:  infoset with legal_actions and game information.
        Output: one action from infoset.legal_actions.

        v3 change:
        - Build a robust greedy baseline first.
        - Use the DP score only as an override signal.
        - If the reduced MDP is not clearly better, fall back to greedy.

        This is intentionally less "pure MDP" than v1/v2, but it is usually
        stronger in actual Dou Dizhu games because the reduced MDP only models
        self-hand decomposition, not initiative, hidden hands, or opponent plans.
        """
        try:
            legal_actions = getattr(infoset, "legal_actions", [])
            if not legal_actions:
                return []
            if len(legal_actions) == 1:
                return legal_actions[0]

            state = self.extract_state(infoset)
            belief = self.infer_belief(infoset, state)
            hand = state["my_hand"]
            self._maybe_reset_for_new_game(hand)

            pass_legal = [] in legal_actions
            non_pass = [a for a in legal_actions if a != []]

            # 1) Direct finish is always the highest-priority rule.
            finish_actions = [
                a for a in non_pass
                if len(self.env_cards_to_real_str(a)) == state["my_count"]
            ]
            if finish_actions:
                return self.choose_lowest_cost_action(finish_actions)

            # 2) If teammate is about to go out, do not block.
            if (
                not state["leading_round"]
                and pass_legal
                and self.is_teammate_last_player(state)
                and state["teammate_cards"] is not None
                and state["teammate_cards"] <= self.cfg["teammate_release_cards"]
            ):
                return []

            # 3) Greedy anchor: this is the safe action. DP must beat it by a
            # margin before it is allowed to override.
            greedy_action = self.greedy_baseline_action(legal_actions, state, belief)
            if greedy_action not in legal_actions:
                greedy_action = random.choice(legal_actions)

            # 4) Candidate set. In enemy-danger situations, do not let pass be
            # selected if any legal response exists.
            if state["leading_round"]:
                candidates = non_pass if non_pass else legal_actions
            elif self.is_enemy_last_player(state) and state["dangerous"] and non_pass:
                candidates = non_pass
            else:
                candidates = legal_actions

            best_q = -float("inf")
            best_actions = []
            for action in candidates:
                q = self.online_q_value(action, state, belief)
                if q > best_q:
                    best_q = q
                    best_actions = [action]
                elif q == best_q:
                    best_actions.append(action)

            if not best_actions:
                return greedy_action

            dp_action = self.choose_lowest_cost_action(best_actions)
            if dp_action not in legal_actions:
                return greedy_action

            # 5) Decide whether the DP action is reliable enough to override.
            if self.should_override_greedy(dp_action, greedy_action, best_q, state, belief):
                return dp_action
            return greedy_action

        except Exception as e:
            self.fallback_count += 1
            self.last_error = repr(e)
            legal_actions = getattr(infoset, "legal_actions", [])
            if legal_actions:
                return random.choice(legal_actions)
            return []

    # ============================================================
    # v3 Greedy anchor layer
    # ============================================================

    def greedy_baseline_action(self, legal_actions, state, belief):
        """
        A stable greedy baseline. The DP layer may override this action only
        when it has a clear advantage.

        The purpose is not to be perfect, but to prevent the reduced MDP from
        making obviously weak tactical choices such as passing to an enemy too
        often, blocking a teammate, or wasting bombs in ordinary positions.
        """
        pass_legal = [] in legal_actions
        non_pass = [a for a in legal_actions if a != []]

        if not non_pass:
            return [] if pass_legal else random.choice(legal_actions)

        # Direct finish first.
        finish_actions = [
            a for a in non_pass
            if len(self.env_cards_to_real_str(a)) == state["my_count"]
        ]
        if finish_actions:
            return self.choose_lowest_cost_action(finish_actions)

        # Leading: choose a hand-structure-friendly play, but preserve bombs.
        if state["leading_round"]:
            return self.greedy_leading_action(non_pass, state, belief)

        # Following teammate: pass unless we can finish. This is a very strong
        # rule in landlord-vs-farmers settings.
        if self.is_teammate_last_player(state):
            return [] if pass_legal else self.choose_lowest_cost_action(non_pass)

        # Following enemy: legal non-pass actions are already beating actions in
        # common Dou Dizhu environments. Use the cheapest non-bomb response if
        # possible. If enemy is close to going out, bombs are allowed.
        if self.is_enemy_last_player(state):
            non_bombs = []
            bombs = []
            for a in non_pass:
                a_str = self.env_cards_to_real_str(a)
                a_type, _ = self.get_card_type(a_str)
                if self.is_bomb_or_rocket(a_type):
                    bombs.append(a)
                else:
                    non_bombs.append(a)

            if non_bombs:
                return self.choose_lowest_cost_action(non_bombs)
            if bombs and (state["dangerous"] or not pass_legal):
                return self.choose_lowest_cost_action(bombs)
            return [] if pass_legal else self.choose_lowest_cost_action(non_pass)

        # Unknown last player: be conservative.
        return [] if pass_legal else self.choose_lowest_cost_action(non_pass)

    def greedy_leading_action(self, actions, state, belief):
        """
        Greedy leading policy: prefer actions that improve hand structure,
        remove low cards, and play useful combinations; avoid opening with high
        singles and bombs unless near finish.
        """
        hand = state["my_hand"]
        current_badness = self.hand_badness(hand)
        best_score = -float("inf")
        best = []

        for action in actions:
            a_str = self.env_cards_to_real_str(action)
            if not a_str or not self.can_remove(hand, a_str):
                continue
            a_type, _ = self.get_card_type(a_str)
            next_hand = self.remove_action_from_hand(hand, a_str)
            counts = Counter(a_str)

            score = 0.0
            score += self.cfg["greedy_lead_hand_improve_weight"] * (
                current_badness - self.hand_badness(next_hand)
            )
            score += self.cfg["greedy_lead_len_weight"] * len(a_str)

            if self.min_card(hand) in a_str:
                score += self.cfg["min_card_bonus"]

            type_str = str(a_type).lower()
            if self.is_chain_type(type_str):
                score += self.cfg["greedy_lead_chain_bonus"] + 0.4 * len(a_str)
            if "trio" in type_str or "three" in type_str or (counts and max(counts.values()) == 3):
                score += self.cfg["greedy_lead_trio_bonus"]
            if len(a_str) == 2 and counts and max(counts.values()) == 2:
                score += self.cfg["greedy_lead_pair_bonus"]

            # Do not open with high single cards unless the hand is almost done.
            if len(a_str) == 1:
                score -= self.cfg["greedy_lead_high_single_penalty"] * self.main_rank_value(a_str)

            # Preserve bombs/rocket unless near finishing.
            if self.is_bomb_or_rocket(a_type) and next_hand != "":
                score -= self.cfg["greedy_lead_bomb_penalty"]

            # Light risk penalty, mainly to avoid silly low single openings.
            if next_hand != "":
                score -= 2.0 * self.estimate_beaten_risk(a_str, a_type, belief)

            if score > best_score:
                best_score = score
                best = [action]
            elif score == best_score:
                best.append(action)

        return self.choose_lowest_cost_action(best) if best else self.choose_lowest_cost_action(actions)

    def should_override_greedy(self, dp_action, greedy_action, dp_q, state, belief):
        """
        DP override gate. The reduced MDP is allowed to override greedy only
        when the online Q advantage is sufficiently large and the action is not
        tactically suspicious.
        """
        if dp_action == greedy_action:
            return True

        # Direct finish was already handled before this method. Keep this for
        # safety if act() is edited later.
        dp_str = self.env_cards_to_real_str(dp_action)
        greedy_str = self.env_cards_to_real_str(greedy_action)
        hand = state["my_hand"]
        if dp_str and len(dp_str) == state["my_count"]:
            return True

        dp_type, _ = self.get_card_type(dp_str)
        greedy_q = self.online_q_value(greedy_action, state, belief)
        advantage = dp_q - greedy_q

        if state["leading_round"]:
            margin = self.cfg["dp_override_margin_leading"]
        elif self.is_enemy_last_player(state) and state["dangerous"]:
            margin = self.cfg["dp_override_margin_enemy_danger"]
        else:
            margin = self.cfg["dp_override_margin_following"]

        if advantage < margin:
            return False

        # Do not let DP spend bombs/rocket over greedy in ordinary situations.
        if (
            self.is_bomb_or_rocket(dp_type)
            and self.remove_action_from_hand(hand, dp_str) != ""
            and not state["dangerous"]
            and not self.cfg["allow_non_finish_bomb_override"]
        ):
            return False

        # Avoid high-risk DP overrides. Greedy is usually safer tactically.
        if dp_str:
            risk = self.estimate_beaten_risk(dp_str, dp_type, belief)
            if risk > self.cfg["max_override_risk"] and self.remove_action_from_hand(hand, dp_str) != "":
                return False

        # When teammate played, only override by finishing; otherwise pass/greedy is safer.
        if self.is_teammate_last_player(state) and self.remove_action_from_hand(hand, dp_str) != "":
            return False

        return True

    # ============================================================
    # Reduced MDP solver: memoized DP, not online value iteration
    # ============================================================

    def _maybe_reset_for_new_game(self, current_hand):
        current_hand = self.sort_card_str(current_hand)
        if self.root_hand is None or not self.is_subhand(current_hand, self.root_hand):
            self.root_hand = current_hand
            self.value_cache = {"": 0.0}
            self.best_action_cache = {}
            self.action_cache = {}
            self.stats["new_game_resets"] += 1

    def V(self, hand):
        """
        Memoized Bellman value for the reduced acyclic MDP:
            V(s) = max_a [ R(s,a,s') + gamma * V(s') ]

        Terminal state:
            V("") = 0
        The terminal reward is paid when next_hand == "" in mdp_reward().
        """
        hand = self.sort_card_str(hand)

        if hand in self.value_cache:
            self.stats["value_cache_hits"] += 1
            return self.value_cache[hand]

        self.stats["value_cache_misses"] += 1

        # Safety cutoff: if the subproblem grows too large, approximate V(s)
        # with a fast heuristic rather than exploring more states.
        if len(self.value_cache) >= self.cfg["max_dp_states"]:
            self.stats["cutoff_uses"] += 1
            return self.cutoff_value(hand)

        actions = self.generate_actions_from_hand(hand)
        if not actions:
            val = self.cutoff_value(hand)
            self.value_cache[hand] = val
            return val

        best_val = -float("inf")
        best_action = None

        for action_str in actions:
            if not self.can_remove(hand, action_str):
                continue
            next_hand = self.remove_action_from_hand(hand, action_str)
            r = self.mdp_reward(hand, action_str, next_hand)
            q = r + self.cfg["gamma"] * self.V(next_hand)
            if q > best_val:
                best_val = q
                best_action = action_str

        if best_val == -float("inf"):
            best_val = self.cutoff_value(hand)

        self.value_cache[hand] = best_val
        self.best_action_cache[hand] = best_action
        return best_val

    def cutoff_value(self, hand):
        """
        Fast fallback value used when the DP state budget is exhausted.
        Higher is better. A lower hand_badness means a better hand.
        """
        if not hand:
            return 0.0
        return (
            -self.cfg["cutoff_value_badness_weight"] * self.hand_badness(hand)
            -self.cfg["cutoff_value_len_weight"] * len(hand)
        )

    def online_q_value(self, action, state, belief):
        """
        Action score used in the real game.

        The DP value V(s) is only a decomposition value. The online score adds
        tactical context:
          - pass is punished when the enemy has initiative;
          - beating an enemy is rewarded, especially if the enemy is close out;
          - beating a teammate is heavily punished unless it finishes our hand;
          - bombs/rocket are preserved unless they finish or stop danger.
        """
        hand = state["my_hand"]

        if action == []:
            if state["leading_round"]:
                return self.cfg["leading_pass_penalty"]

            base = self.cfg["pass_penalty"] + self.cfg["gamma"] * self.V(hand)

            if self.is_teammate_last_player(state):
                if (
                    state["teammate_cards"] is not None
                    and state["teammate_cards"] <= self.cfg["teammate_release_cards"]
                ):
                    base += self.cfg["teammate_release_bonus"]
                else:
                    # Passing teammate's non-critical play is often fine.
                    base += 8.0

            if self.is_enemy_last_player(state):
                base += self.cfg["ordinary_enemy_pass_penalty"]
                if state["dangerous"]:
                    base += self.cfg["danger_pass_penalty"]

            return base

        action_str = self.env_cards_to_real_str(action)
        if not action_str or not self.can_remove(hand, action_str):
            return -float("inf")

        action_type, action_rank = self.get_card_type(action_str)
        next_hand = self.remove_action_from_hand(hand, action_str)

        q = self.mdp_reward(hand, action_str, next_hand)
        q += self.cfg["gamma"] * self.V(next_hand)

        # Direct finish should dominate nearly everything.
        if next_hand == "":
            q += self.cfg["finish_bonus_online"]

        # Online risk: can this play be beaten by hidden cards?
        if next_hand != "":
            risk = self.estimate_beaten_risk(action_str, action_type, belief)
            q -= self.cfg["risk_penalty_weight"] * risk

        # Leading-specific corrections.
        if state["leading_round"]:
            # Do not open with high solo unless it is structurally necessary.
            if len(action_str) == 1:
                q -= self.cfg["leading_single_high_penalty"] * self.main_rank_value(action_str)
            if self.is_bomb_or_rocket(action_type) and next_hand != "":
                q -= self.cfg["leading_bomb_extra_penalty"]

        # Following-specific corrections.
        else:
            last_type, _ = self.get_card_type(state.get("last_move", ""))

            if self.is_teammate_last_player(state):
                if next_hand != "":
                    q -= self.cfg["beat_teammate_penalty"]

            if self.is_enemy_last_player(state):
                q += self.cfg["enemy_follow_beat_bonus"]
                if action_type == last_type:
                    q += self.cfg["same_type_follow_bonus"]
                if state["dangerous"]:
                    q += self.cfg["enemy_danger_beat_bonus"]

                # Avoid wasting bomb/rocket in ordinary following situations.
                if (
                    self.is_bomb_or_rocket(action_type)
                    and not state["dangerous"]
                    and next_hand != ""
                ):
                    q -= self.cfg["unnecessary_bomb_follow_penalty"]

        return q

    def mdp_reward(self, hand, action_str, next_hand):
        """
        Handcrafted reward for the reduced MDP.
        This is NOT the true Dou Dizhu terminal reward. It is a heuristic reward
        designed to make the self-hand decomposition MDP useful.
        """
        if not action_str:
            return 0.0

        action_type, _ = self.get_card_type(action_str)
        type_str = str(action_type).lower()
        counts = Counter(action_str)

        r = 0.0
        r += self.cfg["turn_penalty"]
        r += self.cfg["card_reward"] * len(action_str)

        # Reward improvement of future hand shape.
        r += self.cfg["structure_reward"] * (
            self.hand_badness(hand) - self.hand_badness(next_hand)
        )

        # Prefer removing the smallest card in leading/decomposition planning.
        if action_str and self.min_card(hand) in action_str:
            r += self.cfg["min_card_bonus"]

        # Type bonuses.
        if self.is_chain_type(type_str):
            r += self.cfg["chain_bonus"] + self.cfg["chain_len_bonus"] * len(action_str)

        if "trio" in type_str or "three" in type_str or (counts and max(counts.values()) == 3):
            r += self.cfg["trio_bonus"]

        if len(action_str) == 2 and counts and max(counts.values()) == 2:
            r += self.cfg["pair_bonus"]

        if counts and max(counts.values()) == 4 and len(action_str) > 4:
            r += self.cfg["four_with_penalty"]

        # Bomb/rocket are valuable, but should not be wasted in decomposition.
        if self.is_rocket(action_str, type_str):
            r += self.cfg["rocket_use_penalty"]
        elif self.is_bomb_or_rocket(type_str):
            r += self.cfg["bomb_use_penalty"]

        # Control cards are useful; spending them has a cost.
        r -= self.cfg["control_use_penalty_weight"] * self.control_cost(
            action_str, {"dangerous": False}
        )

        # Terminal reward is paid when the hand becomes empty.
        if next_hand == "":
            r += self.cfg["terminal_reward"]

        return r

    # ============================================================
    # State extraction from infoset
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
            "dangerous": enemy_min_cards <= self.cfg["danger_cards"],
            "very_dangerous": enemy_min_cards <= self.cfg["very_danger_cards"],
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
        # In these environments, two consecutive pass actions usually mean
        # current player has initiative. last_move is kept for compatibility.
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
    # Bayesian belief estimation
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
            return self.cfg["rocket_prob"]
        return 0.0

    def estimate_bomb_prob(self, unknown_counter):
        possible = 0
        for card in CARD_ORDER:
            if card in ["B", "R"]:
                continue
            if unknown_counter[card] >= 4:
                possible += 1
        return min(self.cfg["bomb_prob_cap"], possible * self.cfg["bomb_prob_per_possible"])

    def estimate_beaten_risk(self, action_str, action_type, belief):
        if not action_str:
            return 0.0

        type_str = str(action_type).lower()
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
            return min(
                self.cfg["risk_cap"],
                belief["rocket_prob"] + self.cfg["higher_bomb_risk_weight"] * higher_bombs,
            )

        same_type_risk = self.estimate_same_type_risk(action_str, type_str, unknown)
        return min(
            self.cfg["risk_cap"],
            same_type_risk
            + self.cfg["bomb_risk_weight"] * belief["bomb_prob"]
            + self.cfg["rocket_risk_weight"] * belief["rocket_prob"],
        )

    def estimate_same_type_risk(self, action_str, action_type, unknown):
        counts = Counter(action_str)
        if not counts:
            return 0.0

        main = self.main_rank_value(action_str)
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

        return min(
            self.cfg["same_type_risk_cap"],
            1.0 - math.exp(-self.cfg["same_type_exp_weight"] * pressure),
        )

    # ============================================================
    # Action generation for reduced MDP
    # ============================================================

    def generate_actions_from_hand(self, hand):
        """
        Generate a bounded set of possible decomposition actions from hand.
        These are not the environment's legal_actions; they are internal actions
        for the reduced self-hand MDP.
        """
        hand = self.sort_card_str(hand)
        if hand in self.action_cache:
            return self.action_cache[hand]

        counts = Counter(hand)
        actions = set()

        # Basic groups: solo, pair, trio, bomb.
        for c in CARD_ORDER:
            cnt = counts.get(c, 0)
            if cnt >= 1:
                actions.add(c)
            if cnt >= 2:
                actions.add(c * 2)
            if cnt >= 3:
                actions.add(c * 3)
            if cnt >= 4 and c not in ["B", "R"]:
                actions.add(c * 4)

        # Rocket.
        if counts.get("B", 0) >= 1 and counts.get("R", 0) >= 1:
            actions.add("BR")

        # Trio with solo / pair.
        trio_cards = [c for c in NORMAL_CHAIN_ORDER + ["2"] if counts.get(c, 0) >= 3]
        solo_cards = [c for c in CARD_ORDER if counts.get(c, 0) >= 1]
        pair_cards = [c for c in CARD_ORDER if counts.get(c, 0) >= 2 and c not in ["B", "R"]]

        for t in trio_cards:
            base = t * 3
            for s in solo_cards:
                if s != t:
                    actions.add(base + s)
            for p in pair_cards:
                if p != t:
                    actions.add(base + p * 2)

        # Four with two singles / two pairs, limited to avoid explosion.
        bomb_cards = [c for c in NORMAL_CHAIN_ORDER + ["2"] if counts.get(c, 0) >= 4]
        for b in bomb_cards:
            base = b * 4
            available_solos = [c for c in solo_cards if c != b]
            available_pairs = [c for c in pair_cards if c != b]
            for wings in list(combinations(available_solos, 2))[: self.cfg["max_wing_combinations"]]:
                actions.add(base + "".join(wings))
            for wings in list(combinations(available_pairs, 2))[: self.cfg["max_wing_combinations"]]:
                actions.add(base + "".join(w * 2 for w in wings))

        # Chains: solo chain, pair chain, trio chain without wings.
        self._add_chains(actions, counts, need=1, min_len=5)
        self._add_chains(actions, counts, need=2, min_len=3)
        self._add_chains(actions, counts, need=3, min_len=2)

        # Limited plane with solo/pair wings.
        self._add_limited_planes_with_wings(actions, counts)

        valid = []
        for a in actions:
            a = self.sort_card_str(a)
            if a and self.can_remove(hand, a):
                valid.append(a)

        # Prefer long useful decompositions first, then lower ranks.
        valid = sorted(set(valid), key=lambda x: (-len(x), self.main_rank_value(x), x))
        valid = valid[: self.cfg["max_generated_actions_per_state"]]
        self.action_cache[hand] = valid
        return valid

    def _add_chains(self, actions, counts, need, min_len):
        run = []
        for c in NORMAL_CHAIN_ORDER:
            if counts.get(c, 0) >= need:
                run.append(c)
            else:
                self._emit_subchains(actions, run, need, min_len)
                run = []
        self._emit_subchains(actions, run, need, min_len)

    def _emit_subchains(self, actions, run, need, min_len):
        if len(run) < min_len:
            return
        for L in range(min_len, len(run) + 1):
            for i in range(0, len(run) - L + 1):
                segment = run[i : i + L]
                actions.add("".join(c * need for c in segment))

    def _add_limited_planes_with_wings(self, actions, counts):
        # Find trio-chain bases.
        runs = []
        run = []
        for c in NORMAL_CHAIN_ORDER:
            if counts.get(c, 0) >= 3:
                run.append(c)
            else:
                if len(run) >= 2:
                    runs.append(run)
                run = []
        if len(run) >= 2:
            runs.append(run)

        for run in runs:
            for L in range(2, len(run) + 1):
                for i in range(0, len(run) - L + 1):
                    trio_seq = run[i : i + L]
                    base = "".join(c * 3 for c in trio_seq)
                    base_set = set(trio_seq)

                    solo_wings = [c for c in CARD_ORDER if c not in base_set and counts.get(c, 0) >= 1]
                    pair_wings = [c for c in NORMAL_CHAIN_ORDER + ["2"] if c not in base_set and counts.get(c, 0) >= 2]

                    for wings in list(combinations(solo_wings, L))[: self.cfg["max_wing_combinations"]]:
                        actions.add(base + "".join(wings))
                    for wings in list(combinations(pair_wings, L))[: self.cfg["max_wing_combinations"]]:
                        actions.add(base + "".join(w * 2 for w in wings))

    # ============================================================
    # Hand evaluation helpers
    # ============================================================

    def hand_badness(self, hand_str):
        """
        Lower is better. Used only as a heuristic inside R_heur and cutoff V.
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

    # ============================================================
    # Action cost and type helpers
    # ============================================================

    def choose_lowest_cost_action(self, actions):
        best = None
        best_cost = float("inf")

        for action in actions:
            action_str = self.env_cards_to_real_str(action)
            action_type, _ = self.get_card_type(action_str)

            cost = 0.0
            cost += self.cfg["cost_len_weight"] * len(action_str)
            cost += self.cfg["cost_rank_weight"] * self.main_rank_value(action_str)
            cost += self.control_cost(action_str, {"dangerous": False})

            if self.is_bomb_or_rocket(action_type):
                cost += self.cfg["cost_bomb_weight"]

            if cost < best_cost:
                best_cost = cost
                best = action

        return best if best is not None else random.choice(actions)

    def control_cost(self, action_str, state):
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

        if state.get("dangerous", False):
            cost *= self.cfg["danger_control_scale"]
        return cost

    def get_card_type(self, action_str):
        if action_str == "":
            return "pass", -1

        action_str = self.sort_card_str(action_str)

        if CARD_TYPE is not None:
            for s in [action_str, self.sort_card_str(action_str)]:
                try:
                    info = CARD_TYPE[0][s][0]
                    return str(info[0]), int(info[1])
                except Exception:
                    pass

        # Fallback parser.
        counts = Counter(action_str)
        n = len(action_str)
        max_count = max(counts.values()) if counts else 0

        if n == 1:
            return "solo", self.main_rank_value(action_str)

        if n == 2:
            if set(action_str) == set(["B", "R"]):
                return "rocket", 100
            if max_count == 2:
                return "pair", self.main_rank_value(action_str)

        if n == 3 and max_count == 3:
            return "trio", self.main_rank_value(action_str)

        if n == 4 and max_count == 4:
            return "bomb", self.main_rank_value(action_str)

        if max_count == 3:
            return "trio_with", self.main_rank_value(action_str)

        if max_count == 4 and n > 4:
            return "four_with", self.main_rank_value(action_str)

        if self._is_solo_chain(action_str):
            return "solo_chain", self.main_rank_value(action_str)
        if self._is_pair_chain(action_str):
            return "pair_chain", self.main_rank_value(action_str)
        if self._is_trio_chain(action_str):
            return "trio_chain", self.main_rank_value(action_str)

        return "unknown", self.main_rank_value(action_str)

    def _is_consecutive(self, ranks):
        if not ranks:
            return False
        try:
            idxs = [NORMAL_CHAIN_ORDER.index(c) for c in ranks]
        except ValueError:
            return False
        return max(idxs) - min(idxs) + 1 == len(idxs) and len(set(idxs)) == len(idxs)

    def _is_solo_chain(self, action_str):
        counts = Counter(action_str)
        return (
            len(action_str) >= 5
            and all(v == 1 for v in counts.values())
            and self._is_consecutive(list(counts.keys()))
        )

    def _is_pair_chain(self, action_str):
        counts = Counter(action_str)
        return (
            len(counts) >= 3
            and all(v == 2 for v in counts.values())
            and self._is_consecutive(list(counts.keys()))
        )

    def _is_trio_chain(self, action_str):
        counts = Counter(action_str)
        return (
            len(counts) >= 2
            and all(v == 3 for v in counts.values())
            and self._is_consecutive(list(counts.keys()))
        )

    def main_rank_value(self, action_str):
        if not action_str:
            return -1
        counts = Counter(action_str)
        max_count = max(counts.values())
        main_cards = [c for c, cnt in counts.items() if cnt == max_count]
        return max(RealCard2EnvCard[c] for c in main_cards if c in RealCard2EnvCard)

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

    # ============================================================
    # Card/string conversion and hand manipulation
    # ============================================================

    def min_card(self, hand_str):
        if not hand_str:
            return ""
        return min(hand_str, key=lambda c: INDEX[c])

    def can_remove(self, hand_str, action_str):
        hand_counter = Counter(hand_str)
        action_counter = Counter(action_str)
        for c, cnt in action_counter.items():
            if hand_counter.get(c, 0) < cnt:
                return False
        return True

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

    def is_subhand(self, hand_str, root_hand):
        h = Counter(hand_str)
        r = Counter(root_hand)
        for c, cnt in h.items():
            if cnt > r.get(c, 0):
                return False
        return True

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

    # ============================================================
    # Player role helpers
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
        if self.position == "landlord":
            return False

        last_pid = state.get("last_pid")
        teammate_position = state.get("teammate_position")
        if last_pid is None or teammate_position is None:
            return False

        pos_index = {
            "landlord": 0,
            "landlord_down": 1,
            "landlord_up": 2,
        }
        if isinstance(last_pid, int):
            return last_pid == pos_index.get(teammate_position, -1)
        return last_pid == teammate_position

    def is_enemy_last_player(self, state):
        last_pid = state.get("last_pid")
        if last_pid is None:
            return False

        if isinstance(last_pid, int):
            if self.position == "landlord":
                return last_pid in [1, 2]
            return last_pid == 0

        if self.position == "landlord":
            return last_pid in ["landlord_down", "landlord_up"]
        return last_pid == "landlord"
