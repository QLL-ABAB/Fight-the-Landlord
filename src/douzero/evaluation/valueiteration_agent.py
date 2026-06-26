# valueiteration_agent.py
# ------------------------------------------------------------
# Agent wrapper for pure reduced-MDP value iteration.
#
# This file intentionally contains only online decision glue:
#   1. read infoset.legal_actions;
#   2. convert the current hand and legal actions to planner strings;
#   3. choose the legal action with the largest Bellman score.
#
# The value-iteration implementation itself lives in:
#   value_iteration_planner.py
# ------------------------------------------------------------

import random

try:
    from douzero.evaluation.value_iteration_planner import ValueIterationPlanner, INDEX
except Exception:
    from value_iteration_planner import ValueIterationPlanner, INDEX


ENV_CARD_TO_REAL = {
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


class ValueIterationAgent(object):
    """Choose actions using a pure value-iteration planner over own hand."""

    def __init__(self, position, planner=None, debug=False):
        self.name = "PureReducedMDPValueIterationAgent"
        self.position = position
        self.debug = debug
        self.planner = planner or ValueIterationPlanner()
        self.last_error = None
        self.stats = {
            "acts": 0,
            "fallbacks": 0,
        }

    def act(self, infoset):
        self.stats["acts"] += 1
        try:
            legal_actions = getattr(infoset, "legal_actions", []) or []
            if not legal_actions:
                return []
            if len(legal_actions) == 1:
                return legal_actions[0]

            hand = self.cards_to_str(getattr(infoset, "player_hand_cards", []))
            self.planner.plan(hand)

            best_q = -float("inf")
            best_actions = []
            for action in legal_actions:
                q = self.action_q(hand, action)
                if q > best_q:
                    best_q = q
                    best_actions = [action]
                elif q == best_q:
                    best_actions.append(action)

            return self.tie_break(best_actions) if best_actions else random.choice(legal_actions)
        except Exception as exc:
            self.stats["fallbacks"] += 1
            self.last_error = repr(exc)
            legal_actions = getattr(infoset, "legal_actions", []) or []
            return random.choice(legal_actions) if legal_actions else []

    def action_q(self, hand, action):
        if action == []:
            return self.planner.pass_q_value(hand)

        action_str = self.cards_to_str(action)
        if not action_str or not self.planner.can_remove(hand, action_str):
            return -float("inf")
        return self.planner.q_value(hand, action_str)

    def tie_break(self, actions):
        """Deterministic tie break; no strategic heuristic is applied."""
        if not actions:
            return []
        return sorted(actions, key=lambda a: (self.cards_to_str(a), tuple(a)))[0]

    def cards_to_str(self, cards):
        out = []
        if cards is None:
            return ""
        if isinstance(cards, str):
            out = [c for c in cards if c in INDEX]
        else:
            for card in cards:
                if card in ENV_CARD_TO_REAL:
                    out.append(ENV_CARD_TO_REAL[card])
                elif isinstance(card, str) and card in INDEX:
                    out.append(card)
        return "".join(sorted(out, key=lambda c: INDEX[c]))


# Backward-compatible names used by config.py and older notes.
ValueDPAgent = ValueIterationAgent
MemoizedValueDPAgent = ValueIterationAgent
