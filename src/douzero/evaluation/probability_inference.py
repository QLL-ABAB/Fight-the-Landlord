# probability_inference.py
# ------------------------------------------------------------
# Probability tools for imperfect-information Dou Dizhu agents.
#
# This file is intentionally independent from any agent class.  It estimates
# how likely a hidden player is to be able to respond to a played action, using
# only:
#   - the multiset of unknown cards,
#   - that player's remaining card count,
#   - the current action that must be beaten.
#
# Core idea:
#   P(player has action r) is computed by a multivariate-hypergeometric DP:
#
#       numerator   = # ways to draw m hidden cards containing req(r)
#       denominator = # ways to draw any m hidden cards
#       p_has(r)    = numerator / denominator
#
# Then a lightweight behavioral model converts p_has(r) into a response
# distribution over {pass, same-type responses, bombs, rocket}.
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter
from typing import Dict, Iterable, List, Tuple, Optional

EnvCard2RealCard = {
    3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A", 17: "2", 20: "B", 30: "R",
}

RealCard2EnvCard = {
    "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "2": 17, "B": 20, "R": 30,
}

INDEX = {
    "3": 0, "4": 1, "5": 2, "6": 3, "7": 4, "8": 5, "9": 6,
    "T": 7, "J": 8, "Q": 9, "K": 10, "A": 11, "2": 12, "B": 13, "R": 14,
}

CARD_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "B", "R"]
NORMAL_CHAIN_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]


@dataclass(frozen=True)
class ActionInfo:
    action_type: str
    main_rank: int
    length: int
    group_len: int = 0
    repeat: int = 0


@dataclass(frozen=True)
class ResponseOption:
    action: str              # "" means pass
    prob: float              # normalized response probability
    p_has: float             # probability that the player has the cards for this action
    policy_weight: float     # behavioral model weight before normalization
    tag: str = ""            # pass / same_type / bomb / rocket / etc.


def sort_card_str(card_str: str) -> str:
    return "".join(sorted(card_str or "", key=lambda c: INDEX[c]))


def main_rank_value(action_str: str) -> int:
    if not action_str:
        return -1
    counts = Counter(action_str)
    max_count = max(counts.values())
    mains = [c for c, n in counts.items() if n == max_count]
    return max(RealCard2EnvCard[c] for c in mains if c in RealCard2EnvCard)


def _is_consecutive(cards: Iterable[str]) -> bool:
    cards = list(cards)
    if not cards:
        return False
    try:
        idxs = [NORMAL_CHAIN_ORDER.index(c) for c in cards]
    except ValueError:
        return False
    return len(set(idxs)) == len(idxs) and max(idxs) - min(idxs) + 1 == len(idxs)


def parse_action(action_str: str) -> ActionInfo:
    """Fallback action parser. It covers the common patterns needed by the
    probability model. It does not need to be a perfect Dou Dizhu parser; root
    legal actions still come from the environment.
    """
    action_str = sort_card_str(action_str)
    if not action_str:
        return ActionInfo("pass", -1, 0)

    counts = Counter(action_str)
    n = len(action_str)
    vals = sorted(counts.values(), reverse=True)
    max_count = vals[0]

    if n == 1:
        return ActionInfo("solo", main_rank_value(action_str), n)
    if n == 2:
        if set(action_str) == {"B", "R"}:
            return ActionInfo("rocket", 100, n)
        if max_count == 2:
            return ActionInfo("pair", main_rank_value(action_str), n)
    if n == 3 and max_count == 3:
        return ActionInfo("trio", main_rank_value(action_str), n)
    if n == 4:
        if max_count == 4:
            return ActionInfo("bomb", main_rank_value(action_str), n)
        if max_count == 3:
            return ActionInfo("trio_solo", main_rank_value(action_str), n)
    if n == 5 and sorted(vals) == [2, 3]:
        return ActionInfo("trio_pair", main_rank_value(action_str), n)

    if max_count == 4 and n in (6, 8):
        return ActionInfo("four_with", main_rank_value(action_str), n)

    # Solo chain: 5+ consecutive singles, no 2 / jokers.
    if n >= 5 and all(v == 1 for v in counts.values()) and _is_consecutive(counts.keys()):
        return ActionInfo("solo_chain", main_rank_value(action_str), n, group_len=len(counts), repeat=1)

    # Pair chain: 3+ consecutive pairs.
    if len(counts) >= 3 and all(v == 2 for v in counts.values()) and _is_consecutive(counts.keys()):
        return ActionInfo("pair_chain", main_rank_value(action_str), n, group_len=len(counts), repeat=2)

    # Trio chain without wings: 2+ consecutive trios.
    if len(counts) >= 2 and all(v == 3 for v in counts.values()) and _is_consecutive(counts.keys()):
        return ActionInfo("trio_chain", main_rank_value(action_str), n, group_len=len(counts), repeat=3)

    # Plane with wings: approximate detection by consecutive trio bases.
    trio_cards = [c for c, v in counts.items() if v >= 3 and c in NORMAL_CHAIN_ORDER]
    if len(trio_cards) >= 2 and _is_consecutive(trio_cards):
        L = len(trio_cards)
        if n == L * 4:
            return ActionInfo("plane_solo", max(RealCard2EnvCard[c] for c in trio_cards), n, group_len=L, repeat=3)
        if n == L * 5:
            return ActionInfo("plane_pair", max(RealCard2EnvCard[c] for c in trio_cards), n, group_len=L, repeat=3)

    if max_count == 3:
        return ActionInfo("trio_with", main_rank_value(action_str), n)

    return ActionInfo("unknown", main_rank_value(action_str), n)


def is_bomb_or_rocket(action_str: str) -> bool:
    t = parse_action(action_str).action_type
    return t in ("bomb", "rocket")


def can_beat(action: str, target: str) -> bool:
    """Rule-level comparison for common Dou Dizhu actions."""
    action = sort_card_str(action)
    target = sort_card_str(target)
    if not action:
        return False
    if not target:
        return True

    a = parse_action(action)
    b = parse_action(target)

    if a.action_type == "rocket":
        return b.action_type != "rocket"
    if b.action_type == "rocket":
        return False

    if a.action_type == "bomb" and b.action_type != "bomb":
        return True
    if a.action_type == "bomb" and b.action_type == "bomb":
        return a.main_rank > b.main_rank
    if b.action_type == "bomb":
        return False

    return (
        a.action_type == b.action_type
        and a.length == b.length
        and a.main_rank > b.main_rank
    )


def _counter_to_tuple(counter: Counter) -> Tuple[int, ...]:
    return tuple(int(counter.get(c, 0)) for c in CARD_ORDER)


def _req_to_tuple(req: Counter) -> Tuple[int, ...]:
    return tuple(int(req.get(c, 0)) for c in CARD_ORDER)


def total_cards(counter: Counter) -> int:
    return int(sum(max(0, int(counter.get(c, 0))) for c in CARD_ORDER))


@lru_cache(maxsize=200_000)
def _contains_prob_cached(unknown_tuple: Tuple[int, ...], hand_size: int, req_tuple: Tuple[int, ...]) -> float:
    N = sum(unknown_tuple)
    m = int(hand_size)
    if m < 0 or m > N:
        return 0.0
    if sum(req_tuple) > m:
        return 0.0
    for u, r in zip(unknown_tuple, req_tuple):
        if r > u:
            return 0.0

    denom = math.comb(N, m) if 0 <= m <= N else 0
    if denom <= 0:
        return 0.0

    # DP over ranks: dp[j] = #ways to choose j cards satisfying lower bounds so far.
    dp = [0] * (m + 1)
    dp[0] = 1
    for u, r in zip(unknown_tuple, req_tuple):
        ndp = [0] * (m + 1)
        for used in range(m + 1):
            base = dp[used]
            if base == 0:
                continue
            max_take = min(u, m - used)
            for take in range(r, max_take + 1):
                ndp[used + take] += base * math.comb(u, take)
        dp = ndp
    return float(dp[m] / denom)


def contains_probability(unknown_counter: Counter, hand_size: int, required: Counter | str) -> float:
    """Probability that a random hidden hand of size hand_size contains all
    cards in `required`. `required` can be a Counter or an action string.
    """
    if isinstance(required, str):
        required = Counter(required)
    return _contains_prob_cached(_counter_to_tuple(unknown_counter), int(hand_size), _req_to_tuple(required))


def possible_response_actions(
    target_action: str,
    unknown_counter: Counter,
    max_same_type: int = 24,
    include_bombs: bool = True,
    include_rocket: bool = True,
) -> List[str]:
    """Generate symbolic actions that could beat target_action using cards from
    unknown_counter. These are candidate responses, not guaranteed holdings.
    """
    target_action = sort_card_str(target_action)
    target = parse_action(target_action)
    actions: List[str] = []

    if not target_action:
        return actions

    def add(a: str):
        a = sort_card_str(a)
        if a and a not in actions and can_beat(a, target_action):
            # Make sure the unknown pool has at least the required rank counts.
            req = Counter(a)
            if all(unknown_counter.get(c, 0) >= n for c, n in req.items()):
                actions.append(a)

    # Same-type responses.
    if target.action_type == "solo":
        for c in CARD_ORDER:
            if c in ("B", "R") or RealCard2EnvCard[c] > target.main_rank:
                add(c)

    elif target.action_type == "pair":
        for c in CARD_ORDER:
            if c in ("B", "R"):
                continue
            if RealCard2EnvCard[c] > target.main_rank:
                add(c * 2)

    elif target.action_type == "trio":
        for c in CARD_ORDER:
            if c in ("B", "R"):
                continue
            if RealCard2EnvCard[c] > target.main_rank:
                add(c * 3)

    elif target.action_type in ("trio_solo", "trio_with") and target.length == 4:
        for t in CARD_ORDER:
            if t in ("B", "R") or RealCard2EnvCard[t] <= target.main_rank:
                continue
            if unknown_counter.get(t, 0) < 3:
                continue
            for w in CARD_ORDER:
                if w != t and unknown_counter.get(w, 0) >= 1:
                    add(t * 3 + w)
                    break

    elif target.action_type in ("trio_pair",) or (target.action_type == "trio_with" and target.length == 5):
        for t in CARD_ORDER:
            if t in ("B", "R") or RealCard2EnvCard[t] <= target.main_rank:
                continue
            if unknown_counter.get(t, 0) < 3:
                continue
            for w in CARD_ORDER:
                if w != t and w not in ("B", "R") and unknown_counter.get(w, 0) >= 2:
                    add(t * 3 + w * 2)
                    break

    elif target.action_type in ("solo_chain", "pair_chain", "trio_chain"):
        L = target.group_len
        repeat = target.repeat
        # Generate same length chains with higher highest rank.
        for start in range(0, len(NORMAL_CHAIN_ORDER) - L + 1):
            seg = NORMAL_CHAIN_ORDER[start : start + L]
            high = RealCard2EnvCard[seg[-1]]
            if high <= target.main_rank:
                continue
            if all(unknown_counter.get(c, 0) >= repeat for c in seg):
                add("".join(c * repeat for c in seg))

    elif target.action_type in ("plane_solo", "plane_pair"):
        L = target.group_len
        wing_need = 1 if target.action_type == "plane_solo" else 2
        for start in range(0, len(NORMAL_CHAIN_ORDER) - L + 1):
            seg = NORMAL_CHAIN_ORDER[start : start + L]
            high = RealCard2EnvCard[seg[-1]]
            if high <= target.main_rank:
                continue
            if not all(unknown_counter.get(c, 0) >= 3 for c in seg):
                continue
            base = "".join(c * 3 for c in seg)
            wings = []
            for w in CARD_ORDER:
                if w in seg:
                    continue
                if wing_need == 2 and w in ("B", "R"):
                    continue
                if unknown_counter.get(w, 0) >= wing_need:
                    wings.append(w * wing_need)
                if len(wings) >= L:
                    break
            if len(wings) >= L:
                add(base + "".join(wings[:L]))

    # Bombs and rocket can beat ordinary actions.
    if include_bombs:
        for c in CARD_ORDER:
            if c in ("B", "R"):
                continue
            if unknown_counter.get(c, 0) >= 4:
                # If target is bomb, only higher bombs beat it; otherwise all bombs beat.
                if target.action_type != "bomb" or RealCard2EnvCard[c] > target.main_rank:
                    add(c * 4)

    if include_rocket and unknown_counter.get("B", 0) >= 1 and unknown_counter.get("R", 0) >= 1:
        add("BR")

    actions.sort(key=lambda a: (is_bomb_or_rocket(a), len(a), main_rank_value(a), a))
    return actions[:max_same_type]


def action_cost(action: str) -> float:
    if not action:
        return 0.0
    info = parse_action(action)
    cost = 0.12 * main_rank_value(action) + 0.25 * len(action)
    for c in action:
        if c == "A":
            cost += 1.3
        elif c == "2":
            cost += 3.5
        elif c == "B":
            cost += 6.0
        elif c == "R":
            cost += 7.0
    if info.action_type == "bomb":
        cost += 35.0
    elif info.action_type == "rocket":
        cost += 45.0
    return cost


def _softplus_score_to_weight(score: float, temp: float) -> float:
    temp = max(1e-6, float(temp))
    # Clamp exponent to avoid overflow.
    z = max(-40.0, min(40.0, score / temp))
    return math.exp(z)


def estimate_response_distribution(
    target_action: str,
    unknown_counter: Counter,
    responder_card_count: int,
    relation: str = "enemy",       # enemy / teammate / unknown
    dangerous: bool = False,        # whether responder/root context is dangerous
    responder_near_finish: bool = False,
    max_responses: int = 18,
    temperature: float = 8.0,
    strategic_pass_enemy: float = 0.22,
    strategic_pass_teammate: float = 0.78,
    strategic_pass_unknown: float = 0.40,
) -> List[ResponseOption]:
    """Estimate P(responder plays r | target_action, known info).

    This is an approximate inference model, not exact game theory.

    It combines:
      1. card-holding probability p_has(r),
      2. a behavioral weight saying how attractive r is,
      3. a pass model that includes both "cannot beat" and strategic pass.
    """
    target_action = sort_card_str(target_action)
    m = int(responder_card_count)
    if m <= 0:
        return [ResponseOption("", 1.0, 1.0, 1.0, tag="pass")]

    candidates = possible_response_actions(target_action, unknown_counter, max_same_type=max_responses)

    raw_items: List[Tuple[str, float, float, float, str]] = []
    p_no_beat_product = 1.0

    for a in candidates:
        p_has = contains_probability(unknown_counter, m, a)
        if p_has <= 1e-12:
            continue
        p_no_beat_product *= max(0.0, 1.0 - p_has)

        info = parse_action(a)
        score = 0.0
        # Prefer cheap/minimal beating responses.
        score -= action_cost(a)
        # If responder can finish with this action, it is very attractive.
        if len(a) == m:
            score += 80.0
        # Enemy wants to beat more than teammate does.
        if relation == "enemy":
            score += 16.0
            if dangerous or responder_near_finish:
                score += 24.0
        elif relation == "teammate":
            # A teammate following our play usually should not beat us unless finishing.
            score -= 20.0
            if len(a) == m:
                score += 90.0
        else:
            score += 4.0

        if info.action_type in ("bomb", "rocket") and len(a) != m:
            score -= 25.0
            if dangerous:
                score += 12.0

        weight = _softplus_score_to_weight(score, temperature)
        tag = "rocket" if info.action_type == "rocket" else "bomb" if info.action_type == "bomb" else "same_type"
        raw_items.append((a, p_has, weight, p_has * weight, tag))

    # Approximate probability that responder has at least one beating action.
    p_can_beat = max(0.0, min(1.0, 1.0 - p_no_beat_product))

    if relation == "enemy":
        strategic_pass = strategic_pass_enemy
        if dangerous or responder_near_finish:
            strategic_pass *= 0.45
    elif relation == "teammate":
        strategic_pass = strategic_pass_teammate
    else:
        strategic_pass = strategic_pass_unknown

    raw_pass = (1.0 - p_can_beat) + p_can_beat * strategic_pass
    raw_pass = max(1e-9, raw_pass)

    denom = raw_pass + sum(x[3] for x in raw_items)
    if denom <= 0:
        return [ResponseOption("", 1.0, 1.0, raw_pass, tag="pass")]

    dist = [ResponseOption("", raw_pass / denom, 1.0 - p_can_beat, raw_pass, tag="pass")]
    for a, p_has, weight, raw, tag in raw_items:
        dist.append(ResponseOption(a, raw / denom, p_has, weight, tag=tag))

    # Keep most likely responses plus pass.
    pass_item = dist[0]
    others = sorted(dist[1:], key=lambda x: x.prob, reverse=True)[:max_responses]
    total = pass_item.prob + sum(x.prob for x in others)
    if total <= 0:
        return [ResponseOption("", 1.0, 1.0, 1.0, tag="pass")]
    return [ResponseOption(pass_item.action, pass_item.prob / total, pass_item.p_has, pass_item.policy_weight, pass_item.tag)] + [
        ResponseOption(x.action, x.prob / total, x.p_has, x.policy_weight, x.tag) for x in others
    ]


def summarize_distribution(dist: List[ResponseOption], topn: int = 8) -> str:
    parts = []
    for x in sorted(dist, key=lambda z: z.prob, reverse=True)[:topn]:
        name = "pass" if x.action == "" else x.action
        parts.append(f"{name}:{x.prob:.3f}")
    return ", ".join(parts)
