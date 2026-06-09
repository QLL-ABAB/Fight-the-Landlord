# probabilistic_response_agent.py
# ------------------------------------------------------------
# A Dou Dizhu agent using probability-inference instead of many hidden-hand
# samples.
#
# It calls probability_inference.estimate_response_distribution() to estimate
# how the next player may respond to a root action:
#
#   Q(a) = immediate_score(a)
#          + response_weight * sum_r P(r | a, history) * response_value(a, r)
#
# This is not full POMDP search. It is a fast one-step probabilistic response
# model. The goal is to reduce the high variance of small-N determinization
# sampling while keeping the online decision fast.
# ------------------------------------------------------------

from __future__ import annotations

import random
import math
import copy
from collections import Counter
from typing import Dict, List, Tuple, Optional

try:
    from rlcard.games.doudizhu.utils import CARD_TYPE
except Exception:
    CARD_TYPE = None

from .probability_inference import (
    EnvCard2RealCard,
    RealCard2EnvCard,
    INDEX,
    CARD_ORDER,
    NORMAL_CHAIN_ORDER,
    ResponseOption,
    sort_card_str,
    main_rank_value,
    parse_action,
    can_beat,
    contains_probability,
    estimate_response_distribution,
)


try:
    from .rlcard_agent import RLCardAgent
except Exception:
    from rlcard_agent import RLCardAgent

ALL_POSITIONS = ["landlord", "landlord_down", "landlord_up"]


class ProbabilisticResponseAgent(object):
    """
    Probability-inference Dou Dizhu agent.

    Public interface:
        agent = ProbabilisticResponseAgent(position="landlord")
        action = agent.act(infoset)

    Main difference from Bayesian sampled search:
        - No repeated hidden-hand determinization is needed for the next player.
        - For every candidate root action, directly estimates the next player's
          response distribution using combinatorial probabilities.
    """

    def __init__(self, position, debug=False):
        self.name = "ProbabilisticResponseAgent_RLCardBalanced"
        self.position = position
        self.debug = debug
        self.fallback_count = 0
        self.last_error = None
        self.last_debug = None
        self.greedy_policy = RLCardAgent(position)
        self.stats = {
            "acts": 0,
            "overrides": 0,
            "override_leading": 0,
            "override_enemy_danger": 0,
            "override_finish": 0,
            "blocked_overrides": 0,
            "fallbacks": 0,
        }

        self.cfg = {
            # Candidate pruning.
            "root_topk_leading": 8,
            "root_topk_following": 6,
            "always_include_greedy": True,
            "use_greedy_anchor": True,

            # IMPORTANT:
            # If this is False, the agent becomes exactly an RLCard-greedy wrapper.
            # Use this first to verify that the integration itself is not worse than RLCard.
            "enable_reasoning_override": True,

            # Balanced mode: RLCard is still the default, but probability reasoning
            # is allowed to override in several concrete RLCard-weak cases:
            #   1) leading with a clearly better structured action;
            #   2) avoiding non-finish bomb / high-single openings;
            #   3) stopping a dangerous enemy when RLCard passes;
            #   4) finishing immediately.
            "conservative_override": False,

            # These are intentionally much lower than the previous safe wrapper,
            # otherwise the agent becomes almost identical to RLCard.
            "override_margin_leading": 22.0,
            "override_margin_following": 38.0,
            "override_margin_enemy_danger": 8.0,

            # Probability response model.
            "max_responses": 18,
            "response_temperature": 8.0,
            "response_weight": 1.00,
            "strategic_pass_enemy": 0.22,
            "strategic_pass_teammate": 0.80,
            "strategic_pass_unknown": 0.42,

            # Immediate root action scoring.
            "finish_bonus": 10000.0,
            "turn_penalty": -2.5,
            "action_len_weight": 0.7,
            "hand_improve_weight": 6.5,
            "min_card_bonus": 5.0,
            "chain_bonus": 11.0,
            "chain_len_bonus": 0.50,
            "trio_bonus": 6.0,
            "pair_bonus": 2.5,
            "lead_high_single_penalty": 0.34,
            "control_cost_weight": 1.20,
            "lead_bomb_penalty": 58.0,
            "nonfinish_bomb_penalty": 38.0,

            # Pass / teammate / enemy context.
            "leading_pass_penalty": -9999.0,
            "pass_enemy_penalty": -20.0,
            "pass_enemy_danger_penalty": -70.0,
            "pass_teammate_bonus": 16.0,
            "teammate_release_cards": 2,
            "teammate_release_bonus": 120.0,
            "beat_teammate_penalty": -65.0,
            "beat_enemy_bonus": 18.0,
            "enemy_danger_beat_bonus": 55.0,

            # Response value scoring.
            "next_pass_bonus_enemy": 18.0,
            "next_pass_bonus_teammate": 14.0,
            "enemy_response_penalty": -34.0,
            "enemy_response_len_penalty": -2.4,
            "enemy_response_finish_penalty": -9000.0,
            "enemy_bomb_response_extra_penalty": -18.0,
            "teammate_response_nonfinish_penalty": -18.0,
            "teammate_response_finish_bonus": 9000.0,
            "responder_uses_control_bonus": 0.65,
            "keep_initiative_bonus": 12.0,
            "lose_initiative_penalty": -18.0,

            # Danger thresholds.
            "danger_cards": 2,
            "very_danger_cards": 1,

            # Hand evaluation.
            "badness_turn_weight": 7.0,
            "badness_single_weight": 3.0,
            "badness_len_weight": 1.2,
            "badness_chain_discount_weight": 5.0,
            "badness_control_discount": 2.2,
            "badness_bomb_discount": 3.0,
            "solo_chain_coef": 1.0,
            "pair_chain_coef": 1.5,
            "trio_chain_coef": 2.0,

            # Cost / control.
            "cost_len_weight": 0.45,
            "cost_rank_weight": 0.08,
            "cost_bomb_weight": 55.0,
            "A_cost": 1.5,
            "2_cost": 4.5,
            "B_cost": 7.0,
            "R_cost": 8.0,
            "danger_control_scale": 0.45,
        }

    # ============================================================
    # Public API
    # ============================================================

    def act(self, infoset):
        self.stats["acts"] += 1
        try:
            legal_actions = getattr(infoset, "legal_actions", [])
            if not legal_actions:
                return []
            if len(legal_actions) == 1:
                return legal_actions[0]

            state = self.extract_state(infoset)
            belief = self.infer_belief(infoset, state)

            # Direct finish should dominate all probabilistic reasoning.
            finish_actions = [
                a for a in legal_actions
                if a != [] and len(self.env_cards_to_real_str(a)) == state["my_count"]
            ]
            if finish_actions:
                return self.choose_lowest_cost_action(finish_actions, state)

            try:
                # RLCardAgent mutates some infoset lists internally, so call it on a deepcopy.
                # This keeps the current agent's state extraction and later environment use safe.
                greedy_action = self.greedy_policy.act(copy.deepcopy(infoset))
            except Exception:
                greedy_action = self.greedy_fallback_action(legal_actions, state, belief)

            if greedy_action not in legal_actions:
                greedy_action = self.greedy_fallback_action(legal_actions, state, belief)

            # Debug / ablation mode: exact greedy wrapper.
            # If this mode does not match RLCard's performance, the benchmark wiring is wrong.
            if not self.cfg["enable_reasoning_override"]:
                return greedy_action
            candidates = self.prune_root_actions(legal_actions, state, belief)
            if self.cfg["always_include_greedy"] and greedy_action not in candidates:
                candidates.insert(0, greedy_action)

            scored = []
            for a in candidates:
                q = self.root_action_value(a, state, belief)
                scored.append((q, a))

            if not scored:
                return greedy_action if greedy_action in legal_actions else random.choice(legal_actions)

            scored.sort(key=lambda x: x[0], reverse=True)
            best_q, best_action = scored[0]

            if self.cfg["use_greedy_anchor"]:
                greedy_q = self.root_action_value(greedy_action, state, belief)
                ok, reason = self.should_override_greedy(best_action, greedy_action, best_q, greedy_q, state)
                if ok:
                    if best_action != greedy_action:
                        self.stats["overrides"] += 1
                        if reason == "leading":
                            self.stats["override_leading"] += 1
                        elif reason == "enemy_danger":
                            self.stats["override_enemy_danger"] += 1
                        elif reason == "finish":
                            self.stats["override_finish"] += 1
                else:
                    self.stats["blocked_overrides"] += 1
                    best_action = greedy_action

            if self.debug:
                self.last_debug = {
                    "state": state,
                    "candidates": [(round(q, 2), self.env_cards_to_real_str(a) if a != [] else "pass") for q, a in scored],
                    "greedy": self.env_cards_to_real_str(greedy_action) if greedy_action != [] else "pass",
                    "chosen": self.env_cards_to_real_str(best_action) if best_action != [] else "pass",
                }

            if best_action not in legal_actions:
                return greedy_action if greedy_action in legal_actions else random.choice(legal_actions)
            return best_action

        except Exception as e:
            self.fallback_count += 1
            self.stats["fallbacks"] += 1
            self.last_error = repr(e)
            legal_actions = getattr(infoset, "legal_actions", [])
            return random.choice(legal_actions) if legal_actions else []

    # ============================================================
    # Root action evaluation
    # ============================================================

    def root_action_value(self, action, state, belief) -> float:
        if action == []:
            return self.pass_value(state)

        action_str = self.env_cards_to_real_str(action)
        if not action_str or not self.can_remove(state["my_hand"], action_str):
            return -float("inf")

        immediate = self.immediate_action_score(action_str, state, belief)
        next_relation = self.relation_to_root(self.next_position(self.position))
        responder = self.next_position(self.position)
        responder_count = state["num_cards_left"].get(responder, None)
        if responder_count is None:
            # Fallback: assume unknown average remaining size.
            responder_count = max(1, int(sum(state["num_cards_left"].values()) / max(1, len(state["num_cards_left"]))))

        dist = estimate_response_distribution(
            target_action=action_str,
            unknown_counter=belief["unknown_counter"],
            responder_card_count=responder_count,
            relation=next_relation,
            dangerous=state["dangerous"],
            responder_near_finish=(responder_count <= self.cfg["danger_cards"]),
            max_responses=self.cfg["max_responses"],
            temperature=self.cfg["response_temperature"],
            strategic_pass_enemy=self.cfg["strategic_pass_enemy"],
            strategic_pass_teammate=self.cfg["strategic_pass_teammate"],
            strategic_pass_unknown=self.cfg["strategic_pass_unknown"],
        )

        expected_response = 0.0
        next_hand = self.remove_action_from_hand(state["my_hand"], action_str)
        for opt in dist:
            expected_response += opt.prob * self.response_value(opt, action_str, next_hand, state, next_relation, responder_count)

        return immediate + self.cfg["response_weight"] * expected_response

    def immediate_action_score(self, action_str: str, state: Dict, belief: Dict) -> float:
        hand = state["my_hand"]
        next_hand = self.remove_action_from_hand(hand, action_str)
        info = self.get_card_type(action_str)
        type_str = str(info[0]).lower()
        counts = Counter(action_str)

        if next_hand == "":
            return self.cfg["finish_bonus"]

        score = 0.0
        score += self.cfg["turn_penalty"]
        score += self.cfg["action_len_weight"] * len(action_str)
        score += self.cfg["hand_improve_weight"] * (self.hand_badness(hand) - self.hand_badness(next_hand))

        if self.min_card(hand) in action_str:
            score += self.cfg["min_card_bonus"]

        if self.is_chain_type(type_str):
            score += self.cfg["chain_bonus"] + self.cfg["chain_len_bonus"] * len(action_str)
        if "trio" in type_str or "three" in type_str or (counts and max(counts.values()) == 3):
            score += self.cfg["trio_bonus"]
        if len(action_str) == 2 and counts and max(counts.values()) == 2:
            score += self.cfg["pair_bonus"]

        if state["leading_round"] and len(action_str) == 1:
            score -= self.cfg["lead_high_single_penalty"] * self.main_rank_value(action_str)

        score -= self.cfg["control_cost_weight"] * self.control_cost(action_str, state)

        if self.is_bomb_or_rocket(type_str):
            if state["leading_round"]:
                score -= self.cfg["lead_bomb_penalty"]
            else:
                score -= self.cfg["nonfinish_bomb_penalty"]

        if not state["leading_round"]:
            if self.is_teammate_last_player(state):
                score += self.cfg["beat_teammate_penalty"]
            elif self.is_enemy_last_player(state):
                score += self.cfg["beat_enemy_bonus"]
                if state["dangerous"]:
                    score += self.cfg["enemy_danger_beat_bonus"]

        return score

    def response_value(
        self,
        opt: ResponseOption,
        root_action: str,
        root_next_hand: str,
        state: Dict,
        relation: str,
        responder_count: int,
    ) -> float:
        # If next player passes, our action survives for now; this is usually good.
        if opt.action == "":
            if relation == "enemy":
                return self.cfg["next_pass_bonus_enemy"] + self.cfg["keep_initiative_bonus"]
            if relation == "teammate":
                return self.cfg["next_pass_bonus_teammate"]
            return 8.0

        info = parse_action(opt.action)
        val = 0.0

        if relation == "enemy":
            val += self.cfg["enemy_response_penalty"]
            val += self.cfg["enemy_response_len_penalty"] * len(opt.action)
            val += self.cfg["lose_initiative_penalty"]
            if len(opt.action) >= responder_count:
                val += self.cfg["enemy_response_finish_penalty"]
            if info.action_type in ("bomb", "rocket"):
                val += self.cfg["enemy_bomb_response_extra_penalty"]
                # Enemy spending a bomb also consumes a strong resource; give a tiny compensation.
                val += self.cfg["responder_uses_control_bonus"] * self.control_cost(opt.action, {"dangerous": False})

        elif relation == "teammate":
            if len(opt.action) >= responder_count:
                val += self.cfg["teammate_response_finish_bonus"]
            else:
                val += self.cfg["teammate_response_nonfinish_penalty"]

        else:
            val -= 8.0

        # After our own action, having a cleaner/lighter remaining hand is still good.
        val += -0.25 * self.hand_badness(root_next_hand) - 0.15 * len(root_next_hand)
        return val

    def pass_value(self, state: Dict) -> float:
        if state["leading_round"]:
            return self.cfg["leading_pass_penalty"]

        score = 0.0
        if self.is_teammate_last_player(state):
            score += self.cfg["pass_teammate_bonus"]
            if state["teammate_cards"] is not None and state["teammate_cards"] <= self.cfg["teammate_release_cards"]:
                score += self.cfg["teammate_release_bonus"]
        elif self.is_enemy_last_player(state):
            score += self.cfg["pass_enemy_penalty"]
            if state["dangerous"]:
                score += self.cfg["pass_enemy_danger_penalty"]
        else:
            score -= 4.0
        return score

    def should_override_greedy(self, best_action, greedy_action, best_q: float, greedy_q: float, state: Dict):
        """
        Balanced RLCard+ override gate.

        The safe wrapper was too conservative, so this version allows overrides,
        but only in cases where RLCard has known weaknesses. It returns
        (allowed: bool, reason: str).
        """
        if best_action == greedy_action:
            return True, "same"

        best_str = self.env_cards_to_real_str(best_action) if best_action != [] else ""
        greedy_str = self.env_cards_to_real_str(greedy_action) if greedy_action != [] else ""
        hand = state["my_hand"]

        # Direct finish: always allow.
        if best_str and len(best_str) == state["my_count"]:
            return True, "finish"

        # Never replace a concrete RLCard response with pass. This was one of
        # the easiest ways for heuristic agents to become worse than RLCard.
        if best_action == [] and greedy_action != []:
            return False, "no_pass_over_nonpass"

        # Do not block teammate unless finishing; farmers need cooperation.
        if self.is_teammate_last_player(state):
            if best_str and len(best_str) < state["my_count"]:
                return False, "no_block_teammate"

        margin = self.override_margin(state)
        advantage = best_q - greedy_q
        if advantage < margin:
            return False, "low_advantage"

        best_type, _ = self.get_card_type(best_str)
        greedy_type, _ = self.get_card_type(greedy_str)
        best_type_s = str(best_type).lower()
        greedy_type_s = str(greedy_type).lower()

        best_next = self.remove_action_from_hand(hand, best_str) if best_str else hand
        greedy_next = self.remove_action_from_hand(hand, greedy_str) if greedy_str else hand
        best_is_bomb = self.is_bomb_or_rocket(best_type_s)
        greedy_is_bomb = self.is_bomb_or_rocket(greedy_type_s)

        # ------------------------------------------------------------
        # Case 1: leading round.
        # RLCard is strong, but it can still make two kinds of bad plays:
        #   - opening a non-finish bomb/rocket if the smallest card is inside it;
        #   - opening with a high single when a structure action is available.
        # It also does not explicitly compare several structured decompositions.
        # ------------------------------------------------------------
        if state["leading_round"]:
            # Never use a new non-finish bomb as an override in leading.
            if best_is_bomb and best_next != "":
                return False, "no_lead_bomb"

            # Avoid replacing RLCard by a high single in leading.
            if self.is_high_single(best_str) and best_next != "":
                return False, "no_high_single"

            best_structure = self.is_structured_leading_action(best_str)
            greedy_suspicious = (
                (greedy_is_bomb and greedy_next != "")
                or (self.is_high_single(greedy_str) and greedy_next != "")
            )

            # If RLCard's leading action is suspicious, allow a reasonable
            # structured / low-cost alternative.
            if greedy_suspicious and (best_structure or len(best_str) >= 2):
                return True, "leading"

            # Otherwise allow only a clearly structured action that improves
            # hand shape more than RLCard. This is where probability/hand-value
            # reasoning can actually differ from RLCard without becoming random.
            best_improve = self.hand_badness(hand) - self.hand_badness(best_next)
            greedy_improve = self.hand_badness(hand) - self.hand_badness(greedy_next)
            if best_structure and best_improve >= greedy_improve + 1.5:
                return True, "leading"

            return False, "lead_not_strong_enough"

        # ------------------------------------------------------------
        # Case 2: following teammate.
        # Already blocked above unless finish. Keep RLCard.
        # ------------------------------------------------------------
        if self.is_teammate_last_player(state):
            return False, "keep_teammate_pass"

        # ------------------------------------------------------------
        # Case 3: following enemy.
        # RLCard's ordinary lowest-cost same-type response is strong. The main
        # place to improve is danger handling: when enemy is close to going out,
        # do not pass too easily; allow bomb only in very dangerous situations.
        # ------------------------------------------------------------
        if self.is_enemy_last_player(state):
            if state["dangerous"]:
                # If RLCard passes but we can respond, this is the most useful
                # override case.
                if greedy_action == [] and best_action != []:
                    if best_is_bomb and not state.get("very_dangerous", False):
                        return False, "bomb_only_very_danger"
                    return True, "enemy_danger"

                # If both respond, allow a non-bomb response only when it is
                # materially better by the heuristic. Do not upgrade to bomb
                # unless enemy is down to 1 card.
                if best_action != [] and greedy_action != []:
                    if best_is_bomb and not state.get("very_dangerous", False):
                        return False, "no_bomb_upgrade"
                    if not best_is_bomb:
                        return True, "enemy_danger"

                return False, "danger_not_useful"

            # In ordinary enemy-following situations, RLCard is usually best.
            # Allow only non-bomb non-pass replacement if it has much better
            # hand-shape improvement and does not spend controls excessively.
            if best_action != [] and greedy_action != [] and not best_is_bomb:
                best_improve = self.hand_badness(hand) - self.hand_badness(best_next)
                greedy_improve = self.hand_badness(hand) - self.hand_badness(greedy_next)
                if best_improve >= greedy_improve + 3.0 and self.control_cost(best_str, state) <= self.control_cost(greedy_str, state) + 1.0:
                    return True, "ordinary_enemy_shape"

            return False, "ordinary_keep_rlcard"

        return False, "unknown_keep_rlcard"

    def is_high_single(self, action_str: str) -> bool:
        if not action_str or len(action_str) != 1:
            return False
        return action_str in ("A", "2", "B", "R")

    def is_structured_leading_action(self, action_str: str) -> bool:
        if not action_str:
            return False
        action_type, _ = self.get_card_type(action_str)
        t = str(action_type).lower()
        counts = Counter(action_str)
        if self.is_chain_type(t):
            return True
        if "trio" in t or "three" in t:
            return True
        if len(action_str) >= 4 and counts and max(counts.values()) == 3:
            return True
        if len(action_str) == 2 and counts and max(counts.values()) == 2:
            return True
        # Longer non-single actions are usually more structural than opening a single.
        if len(action_str) >= 5 and not self.is_bomb_or_rocket(t):
            return True
        return False

    def override_margin(self, state: Dict) -> float:
        if state["leading_round"]:
            return self.cfg["override_margin_leading"]
        if self.is_enemy_last_player(state) and state["dangerous"]:
            return self.cfg["override_margin_enemy_danger"]
        return self.cfg["override_margin_following"]

    # ============================================================
    # Candidate pruning and greedy fallback
    # ============================================================

    def prune_root_actions(self, legal_actions, state, belief):
        if state["leading_round"]:
            topk = self.cfg["root_topk_leading"]
            candidates = [a for a in legal_actions if a != []]
        else:
            topk = self.cfg["root_topk_following"]
            if self.is_enemy_last_player(state) and state["dangerous"]:
                candidates = [a for a in legal_actions if a != []] or legal_actions[:]
            else:
                candidates = legal_actions[:]

        scored = []
        for a in candidates:
            if a == []:
                s = self.pass_value(state)
            else:
                a_str = self.env_cards_to_real_str(a)
                s = self.fast_action_rank_score(a_str, state, belief)
            scored.append((s, a))
        scored.sort(key=lambda x: x[0], reverse=True)

        chosen = [a for _, a in scored[:topk]]
        # Direct finish safety.
        for a in legal_actions:
            if a != [] and len(self.env_cards_to_real_str(a)) == state["my_count"] and a not in chosen:
                chosen.insert(0, a)
        return chosen

    def fast_action_rank_score(self, action_str: str, state, belief) -> float:
        if not action_str:
            return self.pass_value(state)
        if not self.can_remove(state["my_hand"], action_str):
            return -float("inf")
        hand = state["my_hand"]
        next_hand = self.remove_action_from_hand(hand, action_str)
        score = 0.0
        score += 5.0 * (self.hand_badness(hand) - self.hand_badness(next_hand))
        score += 0.8 * len(action_str)
        if next_hand == "":
            score += self.cfg["finish_bonus"]
        if self.min_card(hand) in action_str:
            score += self.cfg["min_card_bonus"]
        info = self.get_card_type(action_str)
        t = str(info[0]).lower()
        if self.is_chain_type(t):
            score += self.cfg["chain_bonus"]
        if len(action_str) == 1:
            score -= 0.25 * self.main_rank_value(action_str)
        if self.is_bomb_or_rocket(t) and next_hand != "":
            score -= 45.0
        return score

    def greedy_fallback_action(self, legal_actions, state, belief):
        pass_legal = [] in legal_actions
        non_pass = [a for a in legal_actions if a != []]
        if not non_pass:
            return [] if pass_legal else random.choice(legal_actions)

        # Finish first.
        finish_actions = [a for a in non_pass if len(self.env_cards_to_real_str(a)) == state["my_count"]]
        if finish_actions:
            return self.choose_lowest_cost_action(finish_actions, state)

        if state["leading_round"]:
            scored = [(self.fast_action_rank_score(self.env_cards_to_real_str(a), state, belief), a) for a in non_pass]
            scored.sort(key=lambda x: x[0], reverse=True)
            top_score = scored[0][0]
            best = [a for s, a in scored if s == top_score]
            return self.choose_lowest_cost_action(best, state)

        if self.is_teammate_last_player(state):
            return [] if pass_legal else self.choose_lowest_cost_action(non_pass, state)

        if self.is_enemy_last_player(state):
            non_bombs, bombs = [], []
            for a in non_pass:
                a_str = self.env_cards_to_real_str(a)
                t = str(self.get_card_type(a_str)[0]).lower()
                if self.is_bomb_or_rocket(t):
                    bombs.append(a)
                else:
                    non_bombs.append(a)
            if non_bombs:
                return self.choose_lowest_cost_action(non_bombs, state)
            if bombs and (state["dangerous"] or not pass_legal):
                return self.choose_lowest_cost_action(bombs, state)
            return [] if pass_legal else self.choose_lowest_cost_action(non_pass, state)

        return [] if pass_legal else self.choose_lowest_cost_action(non_pass, state)

    # ============================================================
    # State and belief
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
        enemy_counts = [num_cards_left[p] for p in enemy_positions if p in num_cards_left]
        enemy_min_cards = min(enemy_counts) if enemy_counts else 17
        teammate_cards = num_cards_left.get(teammate_position, None) if teammate_position else None

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

    def infer_belief(self, infoset, state):
        unknown_counter = self.get_unknown_cards(infoset, state["my_hand"])
        return {"unknown_counter": unknown_counter}

    def get_unknown_cards(self, infoset, my_hand):
        deck = Counter()
        for c in CARD_ORDER:
            deck[c] = 1 if c in ("B", "R") else 4
        for c in my_hand:
            deck[c] -= 1
        for c in self.extract_played_cards(infoset):
            deck[c] -= 1
        for c in list(deck.keys()):
            deck[c] = max(0, deck[c])
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
    # Evaluation helpers
    # ============================================================

    def hand_badness(self, hand_str):
        if not hand_str:
            return 0.0
        counts = Counter(hand_str)
        singles = pairs = trios = bombs = controls = 0
        for c, n in counts.items():
            if n == 1:
                singles += 1
            elif n == 2:
                pairs += 1
            elif n == 3:
                trios += 1
            elif n == 4:
                bombs += 1
            if c in ("A", "2", "B", "R"):
                controls += n
        turns = singles + pairs + trios + bombs
        chain_discount = self.chain_discount(counts)
        bad = 0.0
        bad += self.cfg["badness_turn_weight"] * turns
        bad += self.cfg["badness_single_weight"] * singles
        bad += self.cfg["badness_len_weight"] * len(hand_str)
        bad -= self.cfg["badness_chain_discount_weight"] * chain_discount
        bad -= self.cfg["badness_control_discount"] * controls
        bad -= self.cfg["badness_bomb_discount"] * bombs
        return bad

    def chain_discount(self, counts):
        discount = 0.0
        for need, min_len, coef in [(1, 5, self.cfg["solo_chain_coef"]), (2, 3, self.cfg["pair_chain_coef"]), (3, 2, self.cfg["trio_chain_coef"])]:
            run = 0
            for c in NORMAL_CHAIN_ORDER:
                if counts.get(c, 0) >= need:
                    run += 1
                else:
                    if run >= min_len:
                        discount += coef * (run - min_len + 1)
                    run = 0
            if run >= min_len:
                discount += coef * (run - min_len + 1)
        return discount

    # ============================================================
    # Card / action / role utilities
    # ============================================================

    def get_last_two_moves(self, infoset):
        raw = getattr(infoset, "last_two_moves", [[], []])
        result = [self.env_cards_to_real_str(x) for x in raw]
        while len(result) < 2:
            result.append("")
        return result[:2]

    def is_leading_round(self, last_move, last_two_moves):
        return last_two_moves[0] == "" and last_two_moves[1] == ""

    def get_num_cards_left(self, infoset):
        for attr in ["num_cards_left", "num_cards_left_dict", "player_num_cards_left"]:
            raw = getattr(infoset, attr, None)
            if raw is None:
                continue
            if isinstance(raw, dict):
                return dict(raw)
            if isinstance(raw, (list, tuple)):
                return {p: raw[i] for i, p in enumerate(ALL_POSITIONS) if i < len(raw)}
        return {}

    def get_enemy_positions(self):
        return ["landlord_down", "landlord_up"] if self.position == "landlord" else ["landlord"]

    def get_teammate_position(self):
        if self.position == "landlord":
            return None
        return "landlord_up" if self.position == "landlord_down" else "landlord_down"

    def next_position(self, pos):
        i = ALL_POSITIONS.index(pos)
        return ALL_POSITIONS[(i + 1) % 3]

    def relation_to_root(self, pos):
        if pos == self.position:
            return "self"
        if self.same_team(pos, self.position):
            return "teammate"
        return "enemy"

    def same_team(self, p1, p2):
        if p1 == "landlord" or p2 == "landlord":
            return p1 == p2
        return True

    def is_teammate_last_player(self, state):
        if self.position == "landlord":
            return False
        last_pid = state.get("last_pid")
        teammate = state.get("teammate_position")
        if last_pid is None or teammate is None:
            return False
        pos_index = {"landlord": 0, "landlord_down": 1, "landlord_up": 2}
        return last_pid == pos_index.get(teammate, -1) if isinstance(last_pid, int) else last_pid == teammate

    def is_enemy_last_player(self, state):
        last_pid = state.get("last_pid")
        if last_pid is None:
            return False
        if isinstance(last_pid, int):
            return last_pid in [1, 2] if self.position == "landlord" else last_pid == 0
        return last_pid in ["landlord_down", "landlord_up"] if self.position == "landlord" else last_pid == "landlord"

    def env_cards_to_real_list(self, cards):
        result = []
        if cards is None:
            return result
        if isinstance(cards, str):
            result = [c for c in cards if c in INDEX]
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

    def sort_card_str(self, s):
        return sort_card_str(s)

    def can_remove(self, hand_str, action_str):
        h = Counter(hand_str)
        a = Counter(action_str)
        return all(h.get(c, 0) >= n for c, n in a.items())

    def remove_action_from_hand(self, hand_str, action_str):
        counter = Counter(hand_str)
        for c in action_str:
            counter[c] -= 1
            if counter[c] <= 0:
                del counter[c]
        return "".join(c * counter.get(c, 0) for c in CARD_ORDER)

    def get_card_type(self, action_str):
        action_str = sort_card_str(action_str)
        if not action_str:
            return "pass", -1
        if CARD_TYPE is not None:
            for s in [action_str, sort_card_str(action_str)]:
                try:
                    info = CARD_TYPE[0][s][0]
                    return str(info[0]), int(info[1])
                except Exception:
                    pass
        info = parse_action(action_str)
        return info.action_type, info.main_rank

    def main_rank_value(self, action_str):
        return main_rank_value(action_str)

    def is_bomb_or_rocket(self, action_type):
        s = str(action_type).lower()
        return "bomb" in s or "rocket" in s

    def is_chain_type(self, action_type):
        s = str(action_type).lower()
        return "chain" in s or "sequence" in s or "plane" in s

    def min_card(self, hand_str):
        return min(hand_str, key=lambda c: INDEX[c]) if hand_str else ""

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

    def choose_lowest_cost_action(self, actions, state):
        if not actions:
            return []
        best, best_cost = None, float("inf")
        for a in actions:
            a_str = self.env_cards_to_real_str(a)
            t = str(self.get_card_type(a_str)[0]).lower()
            cost = self.cfg["cost_len_weight"] * len(a_str)
            cost += self.cfg["cost_rank_weight"] * self.main_rank_value(a_str)
            cost += self.control_cost(a_str, state)
            if self.is_bomb_or_rocket(t):
                cost += self.cfg["cost_bomb_weight"]
            if cost < best_cost:
                best, best_cost = a, cost
        return best if best is not None else random.choice(actions)
