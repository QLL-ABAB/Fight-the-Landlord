# shared_greedy_baseline.py
# ------------------------------------------------------------
# Shared Greedy Baseline for Dou Dizhu agents.
#
# Purpose:
#   Put the strongest/simple RLCard-style greedy policy in one place so that
#   value-iteration, adversarial-search, and probability-inference agents can
#   all call the same baseline instead of copying greedy code repeatedly.
#
# Public API:
#   policy = GreedyBaselinePolicy(position)
#   action = policy.act(infoset)
#
# It returns one action from infoset.legal_actions.
# ------------------------------------------------------------

import random

try:
    from rlcard.games.doudizhu.utils import CARD_TYPE
except Exception:
    CARD_TYPE = None

EnvCard2RealCard = {
    3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A", 17: "2",
    20: "B", 30: "R",
}
RealCard2EnvCard = {
    "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "2": 17,
    "B": 20, "R": 30,
}
INDEX = {
    "3": 0, "4": 1, "5": 2, "6": 3, "7": 4, "8": 5, "9": 6,
    "T": 7, "J": 8, "Q": 9, "K": 10, "A": 11, "2": 12,
    "B": 13, "R": 14,
}
CARD_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "B", "R"]


class GreedyBaselinePolicy(object):
    """
    A safe wrapper around the RLCard-style greedy policy.

    Difference from the original rlcard_agent.py:
      - Does not mutate infoset.player_hand_cards or infoset.last_two_moves.
      - Can be imported and reused by other agents as a low-level baseline.
      - Keeps the original core logic:
          leading: combine hand, play a combination containing the minimum card;
          following: use the lowest-rank same-type legal response;
          farmer teammate: pass when teammate played and no same-type response is chosen.
    """

    def __init__(self, position, seed=None):
        self.name = "SharedGreedyBaseline"
        self.position = position
        self.rng = random.Random(seed)
        self.last_error = None

    def act(self, infoset):
        legal_actions = getattr(infoset, "legal_actions", [])
        if not legal_actions:
            return []
        if len(legal_actions) == 1:
            return legal_actions[0]

        try:
            hand_cards = self.env_cards_to_real_str(getattr(infoset, "player_hand_cards", []))
            last_move = self.env_cards_to_real_str(getattr(infoset, "last_move", []))
            last_two_cards = self.get_last_two_moves(infoset)
            last_pid = getattr(infoset, "last_pid", None)

            action = None

            # Leading round: two consecutive passes, so we have initiative.
            if last_two_cards[0] == "" and last_two_cards[1] == "":
                comb = combine_cards(hand_cards)
                min_card = hand_cards[0] if hand_cards else ""
                chosen_action = None

                # Preserve original behavior/order: rocket -> bomb -> trio -> ...
                # The final selected action is the last combination containing min_card
                # encountered in this ordered traversal, matching the original code.
                for _, acs in comb.items():
                    for ac in acs:
                        if min_card and min_card in ac:
                            chosen_action = ac

                if chosen_action:
                    action = real_str_to_env_action(chosen_action)

            # Following round: prefer the smallest same-type legal response.
            else:
                chosen_action = ""
                rank = 1000

                if last_move != "" and CARD_TYPE is not None:
                    the_type = CARD_TYPE[0][last_move][0][0]
                    for ac in legal_actions:
                        ac_str = self.env_cards_to_real_str(ac)
                        if ac_str == "":
                            continue
                        try:
                            ac_type, ac_rank = CARD_TYPE[0][ac_str][0]
                        except Exception:
                            continue
                        if the_type == ac_type and int(ac_rank) < rank:
                            rank = int(ac_rank)
                            chosen_action = ac_str

                if chosen_action != "":
                    action = real_str_to_env_action(chosen_action)
                elif last_pid != "landlord" and self.position != "landlord":
                    # Farmer does not block teammate if no cheap same-type response is selected.
                    action = []

            if action is None or action not in legal_actions:
                action = self.rng.choice(legal_actions)

            return action

        except Exception as e:
            self.last_error = repr(e)
            return self.rng.choice(legal_actions)

    def env_cards_to_real_list(self, cards):
        if cards is None:
            return []
        result = []
        if isinstance(cards, str):
            for c in cards:
                if c in INDEX:
                    result.append(c)
            return sorted(result, key=lambda x: INDEX[x])
        for c in list(cards):
            if c in EnvCard2RealCard:
                result.append(EnvCard2RealCard[c])
            elif isinstance(c, str) and c in INDEX:
                result.append(c)
        return sorted(result, key=lambda x: INDEX[x])

    def env_cards_to_real_str(self, cards):
        return "".join(self.env_cards_to_real_list(cards))

    def get_last_two_moves(self, infoset):
        raw = getattr(infoset, "last_two_moves", [[], []])
        result = []
        for move in list(raw)[:2]:
            result.append(self.env_cards_to_real_str(move))
        while len(result) < 2:
            result.append("")
        return result[:2]


def real_str_to_env_action(card_str):
    return [RealCard2EnvCard[c] for c in card_str if c in RealCard2EnvCard]


def sort_card_str(card_str):
    return "".join(sorted(card_str, key=lambda c: INDEX[c]))


def card_str2list(hand):
    hand_list = [0 for _ in range(15)]
    for card in hand:
        hand_list[INDEX[card]] += 1
    return hand_list


def list2card_str(hand_list):
    card_str = ""
    cards = [card for card in INDEX]
    for index, count in enumerate(hand_list):
        card_str += cards[index] * count
    return card_str


def pick_chain(hand_list, count):
    chains = []
    str_card = [card for card in INDEX]
    hand_list = [str(card) for card in hand_list]
    hand = "".join(hand_list[:12])
    chain_list = hand.split("0")
    add = 0
    for index, chain in enumerate(chain_list):
        if len(chain) > 0:
            if len(chain) >= 5:
                start = index + add
                min_count = int(min(chain)) // count
                if min_count != 0:
                    str_chain = ""
                    for num in range(len(chain)):
                        str_chain += str_card[start + num]
                        hand_list[start + num] = int(hand_list[start + num]) - int(min(chain))
                    for _ in range(min_count):
                        chains.append(str_chain)
            add += len(chain)
    hand_list = [int(card) for card in hand_list]
    return chains, hand_list


def combine_cards(hand):
    """Get greedy combinations of cards in hand, matching the original RLCardAgent."""
    hand = sort_card_str(hand)
    comb = {
        "rocket": [],
        "bomb": [],
        "trio": [],
        "trio_chain": [],
        "solo_chain": [],
        "pair_chain": [],
        "pair": [],
        "solo": [],
    }

    # 1. pick rocket
    if hand[-2:] == "BR":
        comb["rocket"].append("BR")
        hand = hand[:-2]

    # 2. pick bomb
    hand_cp = hand
    for index in range(len(hand_cp) - 3):
        if hand_cp[index] == hand_cp[index + 3]:
            bomb = hand_cp[index : index + 4]
            comb["bomb"].append(bomb)
            hand = hand.replace(bomb, "", 1)

    # 3. pick trio and trio_chain
    hand_cp = hand
    for index in range(len(hand_cp) - 2):
        if hand_cp[index] == hand_cp[index + 2]:
            trio = hand_cp[index : index + 3]
            if (
                len(comb["trio"]) > 0
                and INDEX[trio[-1]] < 12
                and (INDEX[trio[-1]] - 1) == INDEX[comb["trio"][-1][-1]]
            ):
                comb["trio"][-1] += trio
            else:
                comb["trio"].append(trio)
            hand = hand.replace(trio, "", 1)

    only_trio = []
    only_trio_chain = []
    for trio in comb["trio"]:
        if len(trio) == 3:
            only_trio.append(trio)
        else:
            only_trio_chain.append(trio)
    comb["trio"] = only_trio
    comb["trio_chain"] = only_trio_chain

    # 4. pick solo chain
    hand_list = card_str2list(hand)
    chains, hand_list = pick_chain(hand_list, 1)
    comb["solo_chain"] = chains

    # 5. pick pair chain
    chains, hand_list = pick_chain(hand_list, 2)
    comb["pair_chain"] = chains
    hand = list2card_str(hand_list)

    # 6. pick pair and solo
    index = 0
    while index < len(hand) - 1:
        if hand[index] == hand[index + 1]:
            comb["pair"].append(hand[index] + hand[index + 1])
            index += 2
        else:
            comb["solo"].append(hand[index])
            index += 1
    if index == (len(hand) - 1):
        comb["solo"].append(hand[index])

    return comb


# Backward-compatible alias.
SharedGreedyBaseline = GreedyBaselinePolicy
