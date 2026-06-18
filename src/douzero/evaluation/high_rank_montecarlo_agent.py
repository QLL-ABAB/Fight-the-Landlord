# high_rank_montecarlo_agent.py
# ------------------------------------------------------------
# Python translation of the high-ranked Botzone C++ Dou Dizhu bot.
#
# Goal:
#   Keep the original implementation logic as close as possible, while changing
#   only language/interface so it can be used locally with the same interface as
#   rlcard_agent.py:
#       agent = HighRankMonteCarloAgent(position="landlord")
#       action = agent.act(infoset)
#
# The original C++ bot uses Botzone concrete cards 0..53. RLCard/DouZero infoset
# usually uses rank codes 3,4,...,14,17,20,30. This file internally reconstructs
# concrete 0..53 cards, runs the translated logic, then maps the chosen action
# back to the exact legal action object from infoset.legal_actions whenever
# possible.
# ------------------------------------------------------------

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import IntEnum
from collections import Counter, defaultdict
from typing import Dict, List, Iterable, Tuple, Optional, Any


PLAYER_COUNT = 3


class Stage(IntEnum):
    BIDDING = 0
    PLAYING = 1


class CardComboType(IntEnum):
    PASS_ = 0
    SINGLE = 1
    PAIR = 2
    STRAIGHT = 3
    STRAIGHT2 = 4
    TRIPLET = 5
    TRIPLET1 = 6
    TRIPLET2 = 7
    BOMB = 8
    QUADRUPLE2 = 9
    QUADRUPLE4 = 10
    PLANE = 11
    PLANE1 = 12
    PLANE2 = 13
    SSHUTTLE = 14
    SSHUTTLE2 = 15
    SSHUTTLE4 = 16
    ROCKET = 17
    INVALID = 18


cardComboScores = [
    0,   # PASS
    1,   # SINGLE
    2,   # PAIR
    6,   # STRAIGHT
    6,   # STRAIGHT2
    4,   # TRIPLET
    4,   # TRIPLET1
    4,   # TRIPLET2
    10,  # BOMB
    8,   # QUADRUPLE2
    8,   # QUADRUPLE4
    8,   # PLANE
    8,   # PLANE1
    8,   # PLANE2
    10,  # SSHUTTLE
    10,  # SSHUTTLE2
    10,  # SSHUTTLE4
    16,  # ROCKET
    0,   # INVALID
]

card_joker = 52
card_JOKER = 53
MAX_LEVEL = 15
MAX_STRAIGHT_LEVEL = 11
level_joker = 13
level_JOKER = 14

ALL_POSITIONS = ["landlord", "landlord_down", "landlord_up"]
POS_INDEX = {"landlord": 0, "landlord_down": 1, "landlord_up": 2}
INDEX_POS = {0: "landlord", 1: "landlord_down", 2: "landlord_up"}

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


def card2level(card: int) -> int:
    return card // 4 + (1 if card == 53 else 0)


def level_to_env(level: int) -> int:
    if level <= 11:
        return level + 3
    if level == 12:
        return 17
    if level == 13:
        return 20
    if level == 14:
        return 30
    raise ValueError(f"bad level: {level}")


def env_to_level(card: int) -> int:
    if card == 17:
        return 12
    if card == 20:
        return 13
    if card == 30:
        return 14
    return int(card) - 3


def concrete_ids_for_level(level: int) -> List[int]:
    if level == level_joker:
        return [52]
    if level == level_JOKER:
        return [53]
    return [level * 4 + i for i in range(4)]


def env_action_to_levels(action: Iterable[int]) -> List[int]:
    return [env_to_level(c) for c in action]


def env_action_to_concrete(action: Iterable[int], used: Optional[set] = None) -> List[int]:
    """Allocate concrete 0..53 cards for a rank-only action."""
    if used is None:
        used = set()
    out = []
    for level in env_action_to_levels(action):
        chosen = None
        for cid in concrete_ids_for_level(level):
            if cid not in used:
                chosen = cid
                break
        if chosen is None:
            # In inconsistent local infosets, reuse the first id rather than crashing.
            chosen = concrete_ids_for_level(level)[0]
        used.add(chosen)
        out.append(chosen)
    out.sort()
    return out


def concrete_action_to_env(action: Iterable[int]) -> List[int]:
    return [level_to_env(card2level(c)) for c in sorted(action, key=lambda x: (card2level(x), x))]


def canon_env(action: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted([int(x) for x in action], key=lambda x: (env_to_level(x), x)))


@dataclass(order=False)
class CardPack:
    level: int
    count: int

    def sort_key(self):
        # C++ CardPack operator< puts larger count first, and larger level first.
        return (-self.count, -self.level)


class CardCombo:
    def __init__(self, cards: Optional[Iterable[int]] = None, combo_type: Optional[CardComboType] = None):
        self.cards: List[int] = sorted(list(cards or []))
        self.packs: List[CardPack] = []
        self.comboType: CardComboType = CardComboType.PASS_
        self.comboLevel: int = 0

        if combo_type is not None:
            self.comboType = combo_type
            if combo_type == CardComboType.PASS_:
                return
            self._build_packs_only()
            return

        self._detect_type()

    def clone(self) -> "CardCombo":
        return CardCombo(self.cards, self.comboType)

    def key(self) -> Tuple[int, ...]:
        return tuple(sorted(self.cards))

    def sort_key(self):
        # Python version of C++ CardCombo::operator< key.
        return tuple((p.count, p.level) for p in self.packs) + ((-len(self.packs),),)

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CardCombo) and self.key() == other.key()

    def __lt__(self, other: "CardCombo") -> bool:
        n = min(len(self.packs), len(other.packs))
        for i in range(n):
            a = self.packs[i]
            b = other.packs[i]
            if a.count != b.count:
                return a.count > b.count
            if a.level != b.level:
                return a.level > b.level
        return len(self.packs) < len(other.packs)

    def _build_packs_only(self):
        counts = [0] * (MAX_LEVEL + 1)
        for c in self.cards:
            counts[card2level(c)] += 1
        self.packs = [CardPack(level=l, count=counts[l]) for l in range(MAX_LEVEL + 1) if counts[l]]
        self.packs.sort(key=lambda p: p.sort_key())
        if self.packs:
            self.comboLevel = self.packs[0].level

    def findMaxSeq(self) -> int:
        if not self.packs:
            return 0
        for c in range(1, len(self.packs)):
            if self.packs[c].count != self.packs[0].count or self.packs[c].level != self.packs[c - 1].level - 1:
                return c
        return len(self.packs)

    def getWeight(self) -> int:
        if self.comboType in (CardComboType.SSHUTTLE, CardComboType.SSHUTTLE2, CardComboType.SSHUTTLE4):
            return cardComboScores[int(self.comboType)] + (10 if self.findMaxSeq() > 2 else 0)
        return cardComboScores[int(self.comboType)]

    def _detect_type(self):
        if not self.cards:
            self.comboType = CardComboType.PASS_
            return

        counts = [0] * (MAX_LEVEL + 1)
        countOfCount = [0] * 5
        for c in self.cards:
            counts[card2level(c)] += 1
        for l in range(MAX_LEVEL + 1):
            if counts[l]:
                self.packs.append(CardPack(l, counts[l]))
                if counts[l] <= 4:
                    countOfCount[counts[l]] += 1
        self.packs.sort(key=lambda p: p.sort_key())
        self.comboLevel = self.packs[0].level

        kind = [i for i in range(5) if countOfCount[i]]
        kind.sort()

        if len(kind) == 1:
            curr = countOfCount[kind[0]]
            k = kind[0]
            if k == 1:
                if curr == 1:
                    self.comboType = CardComboType.SINGLE
                    return
                if curr == 2 and len(self.packs) > 1 and self.packs[1].level == level_joker:
                    self.comboType = CardComboType.ROCKET
                    return
                if curr >= 5 and self.findMaxSeq() == curr and self.packs[0].level <= MAX_STRAIGHT_LEVEL:
                    self.comboType = CardComboType.STRAIGHT
                    return
            elif k == 2:
                if curr == 1:
                    self.comboType = CardComboType.PAIR
                    return
                if curr >= 3 and self.findMaxSeq() == curr and self.packs[0].level <= MAX_STRAIGHT_LEVEL:
                    self.comboType = CardComboType.STRAIGHT2
                    return
            elif k == 3:
                if curr == 1:
                    self.comboType = CardComboType.TRIPLET
                    return
                if self.findMaxSeq() == curr and self.packs[0].level <= MAX_STRAIGHT_LEVEL:
                    self.comboType = CardComboType.PLANE
                    return
            elif k == 4:
                if curr == 1:
                    self.comboType = CardComboType.BOMB
                    return
                if self.findMaxSeq() == curr and self.packs[0].level <= MAX_STRAIGHT_LEVEL:
                    self.comboType = CardComboType.SSHUTTLE
                    return

        elif len(kind) == 2:
            curr = countOfCount[kind[1]]
            lesser = countOfCount[kind[0]]
            if kind[1] == 3:
                if kind[0] == 1:
                    if curr == 1 and lesser == 1:
                        self.comboType = CardComboType.TRIPLET1
                        return
                    if self.findMaxSeq() == curr and lesser == curr and self.packs[0].level <= MAX_STRAIGHT_LEVEL:
                        self.comboType = CardComboType.PLANE1
                        return
                if kind[0] == 2:
                    if curr == 1 and lesser == 1:
                        self.comboType = CardComboType.TRIPLET2
                        return
                    if self.findMaxSeq() == curr and lesser == curr and self.packs[0].level <= MAX_STRAIGHT_LEVEL:
                        self.comboType = CardComboType.PLANE2
                        return
            if kind[1] == 4:
                if kind[0] == 1:
                    if curr == 1 and lesser == 2:
                        self.comboType = CardComboType.QUADRUPLE2
                        return
                    if self.findMaxSeq() == curr and lesser == curr * 2 and self.packs[0].level <= MAX_STRAIGHT_LEVEL:
                        self.comboType = CardComboType.SSHUTTLE2
                        return
                if kind[0] == 2:
                    if curr == 1 and lesser == 2:
                        self.comboType = CardComboType.QUADRUPLE4
                        return
                    if self.findMaxSeq() == curr and lesser == curr * 2 and self.packs[0].level <= MAX_STRAIGHT_LEVEL:
                        self.comboType = CardComboType.SSHUTTLE4
                        return

        self.comboType = CardComboType.INVALID

    def canBeBeatenBy(self, b: "CardCombo") -> bool:
        if self.comboType == CardComboType.INVALID or b.comboType == CardComboType.INVALID:
            return False
        if b.comboType == CardComboType.ROCKET:
            return True
        if b.comboType == CardComboType.BOMB:
            if self.comboType == CardComboType.ROCKET:
                return False
            if self.comboType == CardComboType.BOMB:
                return b.comboLevel > self.comboLevel
            return True
        return b.comboType == self.comboType and len(b.cards) == len(self.cards) and b.comboLevel > self.comboLevel

    def findFirstValid(self, cards: Iterable[int]) -> "CardCombo":
        deck = sorted(list(cards))
        if self.comboType == CardComboType.PASS_:
            return CardCombo(deck[:1]) if deck else CardCombo()
        if self.comboType == CardComboType.ROCKET:
            return CardCombo()

        counts = [0] * (MAX_LEVEL + 1)
        for c in deck:
            counts[card2level(c)] += 1
        kindCount = sum(1 for c in counts if c)
        if len(deck) >= len(self.cards):
            mainPackCount = self.findMaxSeq()
            isSequential = self.comboType in (
                CardComboType.STRAIGHT, CardComboType.STRAIGHT2,
                CardComboType.PLANE, CardComboType.PLANE1, CardComboType.PLANE2,
                CardComboType.SSHUTTLE, CardComboType.SSHUTTLE2, CardComboType.SSHUTTLE4,
            )
            i = 1
            while True:
                failed_boundary = False
                can_try = True
                for j in range(mainPackCount):
                    level = self.packs[j].level + i
                    if ((self.comboType == CardComboType.SINGLE and level > MAX_LEVEL)
                            or (isSequential and level > MAX_STRAIGHT_LEVEL)
                            or (self.comboType != CardComboType.SINGLE and not isSequential and level >= level_joker)):
                        failed_boundary = True
                        break
                    if counts[level] < self.packs[j].count:
                        can_try = False
                        break
                if failed_boundary:
                    break
                if can_try:
                    if kindCount >= len(self.packs):
                        required = [0] * (MAX_LEVEL + 1)
                        for j in range(mainPackCount):
                            required[self.packs[j].level + i] = self.packs[j].count
                        ok = True
                        for j in range(mainPackCount, len(self.packs)):
                            found = False
                            for k in range(MAX_LEVEL + 1):
                                if required[k] or counts[k] < self.packs[j].count:
                                    continue
                                required[k] = self.packs[j].count
                                found = True
                                break
                            if not found:
                                ok = False
                                break
                        if ok:
                            solve = []
                            req = required[:]
                            for c in deck:
                                lv = card2level(c)
                                if req[lv]:
                                    solve.append(c)
                                    req[lv] -= 1
                            return CardCombo(solve)
                i += 1

        # failure: bombs, then rocket
        for i in range(level_joker):
            if counts[i] == 4 and (self.comboType != CardComboType.BOMB or i > self.packs[0].level):
                return CardCombo([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
        if counts[level_joker] + counts[level_JOKER] == 2:
            return CardCombo([card_joker, card_JOKER])
        return CardCombo()


class CardCombinations:
    def __init__(self, cards: Optional[Iterable[int]] = None):
        self.packs = []
        for i in range(MAX_LEVEL):
            self.packs.append({"level": i, "count": 0, "card": []})
        if cards is not None:
            for c in cards:
                l = card2level(int(c))
                self.packs[l]["card"].append(int(c))
                self.packs[l]["card"].sort()
                self.packs[l]["count"] += 1

        self.single: List[CardCombo] = []
        self.pair: List[CardCombo] = []
        self.straight: List[List[CardCombo]] = [[] for _ in range(13)]
        self.straight2: List[List[CardCombo]] = [[] for _ in range(11)]
        self.triplet: List[CardCombo] = []
        self.triplet1: List[CardCombo] = []
        self.triplet2: List[CardCombo] = []
        self.bomb: List[CardCombo] = []
        self.quadruple2: List[CardCombo] = []
        self.quadruple4: List[CardCombo] = []
        self.plane: List[List[CardCombo]] = [[] for _ in range(7)]
        self.plane1: List[List[CardCombo]] = [[] for _ in range(7)]
        self.plane2: List[List[CardCombo]] = [[] for _ in range(7)]
        self.sshuttle: List[CardCombo] = []
        self.sshuttle2: List[CardCombo] = []
        self.sshuttle4: List[CardCombo] = []
        self.rocket: List[CardCombo] = []

    def clone(self) -> "CardCombinations":
        return CardCombinations(self.all_cards())

    def all_cards(self) -> List[int]:
        out = []
        for p in self.packs:
            out.extend(p["card"][:p["count"]])
        return sorted(out)

    def getLength(self) -> int:
        return sum(p["count"] for p in self.packs)

    def erase(self, cards: Iterable[int]):
        for c in cards:
            l = card2level(int(c))
            pack = self.packs[l]
            if int(c) in pack["card"]:
                pack["card"].remove(int(c))
            elif pack["card"]:
                # Should not happen if state is consistent. Match C++'s assumption by
                # removing one card of that level anyway.
                pack["card"].pop(0)
            pack["count"] -= 1
            if pack["count"] < 0:
                pack["count"] = 0

    def insert(self, cards: Iterable[int]):
        for c in cards:
            l = card2level(int(c))
            pack = self.packs[l]
            if int(c) not in pack["card"]:
                pack["card"].append(int(c))
                pack["card"].sort()
                pack["count"] += 1

    def getAllSingle(self):
        self.single = []
        for i in range(15):
            if self.packs[i]["count"]:
                self.single.append(CardCombo(self.packs[i]["card"][:1], CardComboType.SINGLE))

    def getAllPair(self):
        self.pair = []
        for i in range(13):
            if self.packs[i]["count"] >= 2:
                self.pair.append(CardCombo(self.packs[i]["card"][:2], CardComboType.PAIR))

    def getAllStraight(self):
        for length in range(5, 13):
            self.straight[length] = []
            for i in range(0, 12 - length + 1):
                if all(self.packs[j]["count"] >= 1 for j in range(i, i + length)):
                    cards = [self.packs[j]["card"][0] for j in range(i, i + length)]
                    self.straight[length].append(CardCombo(cards, CardComboType.STRAIGHT))

    def getAllStraight2(self):
        for length in range(3, 11):
            self.straight2[length] = []
            for i in range(0, 12 - length + 1):
                if all(self.packs[j]["count"] >= 2 for j in range(i, i + length)):
                    cards = []
                    for j in range(i, i + length):
                        cards.extend(self.packs[j]["card"][:2])
                    self.straight2[length].append(CardCombo(cards, CardComboType.STRAIGHT2))

    def getAllTriplet(self):
        self.triplet = []
        for i in range(13):
            if self.packs[i]["count"] >= 3:
                self.triplet.append(CardCombo(self.packs[i]["card"][:3], CardComboType.TRIPLET))

    def getAllTriplet1(self):
        self.triplet1 = []
        for i in range(13):
            if self.packs[i]["count"] < 3:
                continue
            for j in range(15):
                if j == i:
                    continue
                if self.packs[j]["count"]:
                    cards = self.packs[i]["card"][:3] + self.packs[j]["card"][:1]
                    self.triplet1.append(CardCombo(cards, CardComboType.TRIPLET1))

    def getAllTriplet2(self):
        self.triplet2 = []
        for i in range(13):
            if self.packs[i]["count"] < 3:
                continue
            for j in range(13):
                if j == i:
                    continue
                if self.packs[j]["count"] >= 2:
                    cards = self.packs[i]["card"][:3] + self.packs[j]["card"][:2]
                    self.triplet2.append(CardCombo(cards, CardComboType.TRIPLET2))

    def getAllBomb(self):
        self.bomb = []
        for i in range(13):
            if self.packs[i]["count"] == 4:
                self.bomb.append(CardCombo(self.packs[i]["card"][:4], CardComboType.BOMB))

    def getAllQuadruple2(self):
        self.quadruple2 = []
        for i in range(13):
            if self.packs[i]["count"] != 4:
                continue
            for j in range(15):
                if j == i or self.packs[j]["count"] == 0:
                    continue
                for k in range(j + 1, 15):
                    if k == i or k == j or self.packs[k]["count"] == 0:
                        continue
                    cards = self.packs[i]["card"][:4] + self.packs[j]["card"][:1] + self.packs[k]["card"][:1]
                    self.quadruple2.append(CardCombo(cards, CardComboType.QUADRUPLE2))

    def getAllQuadruple4(self):
        self.quadruple4 = []
        for i in range(13):
            if self.packs[i]["count"] < 4:
                continue
            for j in range(13):
                if j == i or self.packs[j]["count"] < 2:
                    continue
                for k in range(j + 1, 13):
                    if k == i or k == j or self.packs[k]["count"] < 2:
                        continue
                    cards = self.packs[i]["card"][:4] + self.packs[j]["card"][:2] + self.packs[k]["card"][:2]
                    self.quadruple4.append(CardCombo(cards, CardComboType.QUADRUPLE4))

    def getAllPlane(self):
        for length in range(2, 7):
            self.plane[length] = []
            for i in range(0, 12 - length + 1):
                if all(self.packs[j]["count"] >= 3 for j in range(i, i + length)):
                    cards = []
                    for j in range(i, i + length):
                        cards.extend(self.packs[j]["card"][:3])
                    self.plane[length].append(CardCombo(cards, CardComboType.PLANE))

    def getAllPlane1(self):
        for length in range(2, 6):
            self.plane1[length] = []
            for i in range(0, 12 - length + 1):
                if not all(self.packs[j]["count"] >= 3 for j in range(i, i + length)):
                    continue
                seq = []
                for j in range(i, i + length):
                    seq.extend(self.packs[j]["card"][:3])
                solution = seq[:]

                def dfs(w: int, low: int):
                    if w == 0:
                        self.plane1[length].append(CardCombo(solution, CardComboType.PLANE1))
                        return
                    for l in range(low, 15):
                        if i <= l <= i + length - 1:
                            continue
                        if self.packs[l]["count"] == 0:
                            continue
                        solution.append(self.packs[l]["card"][0])
                        dfs(w - 1, l + 1)
                        solution.pop()

                dfs(length, 0)

    def getAllPlane2(self):
        for length in range(2, 5):
            self.plane2[length] = []
            for i in range(0, 12 - length + 1):
                if not all(self.packs[j]["count"] >= 3 for j in range(i, i + length)):
                    continue
                seq = []
                for j in range(i, i + length):
                    seq.extend(self.packs[j]["card"][:3])
                solution = seq[:]

                def dfs(w: int, low: int):
                    if w == 0:
                        self.plane2[length].append(CardCombo(solution, CardComboType.PLANE2))
                        return
                    for l in range(low, 13):
                        if i <= l <= i + length - 1:
                            continue
                        if self.packs[l]["count"] < 2:
                            continue
                        solution.extend(self.packs[l]["card"][:2])
                        dfs(w - 1, l + 1)
                        solution.pop(); solution.pop()

                dfs(length, 0)

    def getAllSshuttle(self):
        self.sshuttle = []
        for i in range(0, 11):
            if self.packs[i]["count"] != 4:
                continue
            j = i + 1
            solution = self.packs[i]["card"][:4]
            while j <= 11 and self.packs[j]["count"] == 4:
                solution.extend(self.packs[j]["card"][:4])
                self.sshuttle.append(CardCombo(solution, CardComboType.SSHUTTLE))
                j += 1

    def getAllSshuttle2(self):
        self.sshuttle2 = []
        for i in range(0, 11):
            if self.packs[i]["count"] != 4:
                continue
            sequence = self.packs[i]["card"][:4]
            k = i + 1
            while k <= 11 and self.packs[k]["count"] == 4:
                sequence.extend(self.packs[k]["card"][:4])
                w_need = 2 * (k - i + 1)
                if w_need > 13 - k + i:
                    break
                solution = sequence[:]

                def dfs(w: int, low: int):
                    if w == 0:
                        self.sshuttle2.append(CardCombo(solution, CardComboType.SSHUTTLE2))
                        return
                    for l in range(low, 15):
                        if i <= l <= k:
                            continue
                        if self.packs[l]["count"] == 0:
                            continue
                        solution.append(self.packs[l]["card"][0])
                        dfs(w - 1, l + 1)
                        solution.pop()

                dfs(w_need, 0)
                k += 1

    def getAllSshuttle4(self):
        self.sshuttle4 = []
        for i in range(0, 11):
            if self.packs[i]["count"] != 4:
                continue
            sequence = self.packs[i]["card"][:4]
            k = i + 1
            while k <= 11 and self.packs[k]["count"] == 4:
                sequence.extend(self.packs[k]["card"][:4])
                w_need = 2 * (k - i + 1)
                if w_need > 11 - k + i:
                    break
                solution = sequence[:]

                def dfs(w: int, low: int):
                    if w == 0:
                        self.sshuttle4.append(CardCombo(solution, CardComboType.SSHUTTLE4))
                        return
                    for l in range(low, 13):
                        if i <= l <= k:
                            continue
                        if self.packs[l]["count"] < 2:
                            continue
                        solution.extend(self.packs[l]["card"][:2])
                        dfs(w - 1, l + 1)
                        solution.pop(); solution.pop()

                dfs(w_need, 0)
                k += 1

    def getAllRocket(self):
        self.rocket = []
        if self.packs[13]["count"] and self.packs[14]["count"]:
            self.rocket.append(CardCombo([52, 53], CardComboType.ROCKET))

    def getAllCombos(self):
        self.getAllSingle()
        self.getAllPair()
        self.getAllStraight()
        self.getAllStraight2()
        self.getAllTriplet()
        self.getAllTriplet1()
        self.getAllTriplet2()
        self.getAllBomb()
        self.getAllQuadruple2()
        self.getAllQuadruple4()
        self.getAllPlane()
        self.getAllPlane1()
        self.getAllPlane2()
        self.getAllSshuttle()
        self.getAllSshuttle2()
        self.getAllSshuttle4()
        self.getAllRocket()


@dataclass
class TreeNode:
    turn: int
    hands: List[CardCombinations]

    def clone(self) -> "TreeNode":
        return TreeNode(self.turn, [h.clone() for h in self.hands])


class HighRankMonteCarloAgent:
    """
    Local Python version of the high-ranked Botzone C++ bot.

    Public interface is compatible with rlcard_agent.py:
        agent = HighRankMonteCarloAgent(position="landlord")
        action = agent.act(infoset)

    Parameters:
        time_limit_sec: simulation budget per action. The original C++ code uses
                        about 0.95s. Python is slower, so you may reduce it for
                        local batch evaluation.
        n_root_actions: original code uses top 3 root actions.
    """

    def __init__(self, position: str, debug: bool = False, time_limit_sec: float = 0.95, n_root_actions: int = 3, seed: Optional[int] = None):
        self.name = "HighRankMonteCarloBot_Python"
        self.position = position
        self.debug = debug
        self.time_limit_sec = time_limit_sec
        self.n_root_actions = n_root_actions
        self.rng = random.Random(seed)
        self.last_error = None
        self.last_debug = None
        self.stats = {"acts": 0, "fallbacks": 0, "rollouts": 0}
        self.reset_runtime_state()

    # ---------------- interface ----------------
    def act(self, infoset):
        self.stats["acts"] += 1
        try:
            legal_actions = getattr(infoset, "legal_actions", [])
            if not legal_actions:
                return []
            if len(legal_actions) == 1:
                return legal_actions[0]

            self.reset_runtime_state()
            self.read_infoset(infoset)
            self.prepareData()

            # Direct finish should always be legal and dominant.
            for a in legal_actions:
                if a and len(a) == len(getattr(infoset, "player_hand_cards", [])):
                    return a

            if not self.whatTheyPlayed[self.myPosition]:
                myAction = self.findBestAction(self.myHand, self.lastValidCombo, self.myPosition, self.landlordPosition)
            else:
                myAction = self.returnAction()
            if not myAction.cards and self.rootActions:
                myAction = self.rootActions[0]

            action_env = concrete_action_to_env(myAction.cards)
            matched = self.match_legal_action(action_env, legal_actions)
            if matched is not None:
                return matched

            # Safety: if translated action is not in the RLCard legal set, use
            # the closest legal action according to the same heuristic measure.
            fallback = self.choose_best_from_legal_actions(infoset, legal_actions)
            return fallback if fallback is not None else random.choice(legal_actions)

        except Exception as e:
            self.stats["fallbacks"] += 1
            self.last_error = repr(e)
            legal_actions = getattr(infoset, "legal_actions", [])
            return random.choice(legal_actions) if legal_actions else []

    # ---------------- C++ global-state reset/read ----------------
    def reset_runtime_state(self):
        self.myCards: set[int] = set()
        self.landlordPublicCards: set[int] = set()
        self.whatTheyPlayed: List[List[List[int]]] = [[] for _ in range(PLAYER_COUNT)]
        self.lastValidCombo = CardCombo()
        self.lastValidPlayer = -1
        self.cardRemaining = [17, 17, 17]
        self.myPosition = POS_INDEX.get(self.position, 0)
        self.landlordPosition = 0
        self.landlordBid = -1
        self.stage = Stage.PLAYING
        self.bidInput: List[int] = []
        self.startTime = time.perf_counter()
        self.gamesSimulated = 0
        self.differenceBetweenFirstAndSecond = 0

        self.remainingCards: set[int] = set()
        self.myHand = CardCombinations()
        self.nextHand = CardCombinations()
        self.previousHand = CardCombinations()
        self.rootActions: List[CardCombo] = []
        self.winningTimes: Dict[Tuple[int, ...], int] = {}
        self.rootActionByKey: Dict[Tuple[int, ...], CardCombo] = {}
        self.totalScore = 1
        self.landlordHasNotPlayed = True
        self.landlordHasNotPlayed_ = True
        self.root = TreeNode(self.myPosition, [CardCombinations(), CardCombinations(), CardCombinations()])
        self.sons: List[Tuple[CardCombo, TreeNode]] = []

    def read_infoset(self, infoset):
        self.myPosition = POS_INDEX.get(self.position, 0)
        self.landlordPosition = 0
        self.landlordBid = 3
        self.stage = Stage.PLAYING

        # Own hand: rank-only RLCard cards -> concrete 0..53 cards.
        used = set()
        self.myCards = set(env_action_to_concrete(getattr(infoset, "player_hand_cards", []), used))

        # Remaining card counts.
        self.cardRemaining = self.get_num_cards_left(infoset)
        self.cardRemaining[self.myPosition] = len(self.myCards)

        # Last two moves. In local DouZero/RLCard infoset, these are rank-only actions.
        history_env = getattr(infoset, "last_two_moves", [[], []])
        history_env = [list(x) for x in history_env]
        while len(history_env) < 2:
            history_env.insert(0, [])
        history_env = history_env[-2:]

        played_used = set(self.myCards)
        history = [env_action_to_concrete(h, played_used) for h in history_env]

        howManyPass = 0
        whoInHistory = [(self.myPosition - 2 + PLAYER_COUNT) % PLAYER_COUNT, (self.myPosition - 1 + PLAYER_COUNT) % PLAYER_COUNT]
        for p in range(2):
            player = whoInHistory[p]
            playedCards = history[p]
            self.whatTheyPlayed[player].append(playedCards)
            self.cardRemaining[player] = max(0, self.cardRemaining[player] - len(playedCards))
            if len(playedCards) == 0:
                howManyPass += 1
            else:
                self.lastValidCombo = CardCombo(playedCards)

        if howManyPass == 2:
            self.lastValidCombo = CardCombo()
        self.lastValidPlayer = (self.myPosition + 2 - howManyPass) % 3

        # If full action history is available, use it only to improve remainingCards
        # and landlord spring information. Exact player assignment is best-effort.
        seq = getattr(infoset, "card_play_action_seq", None)
        if seq:
            self.whatTheyPlayed = [[] for _ in range(PLAYER_COUNT)]
            seq_used = set(self.myCards)
            # In Dou Dizhu, play starts from landlord and rotates.
            for i, action in enumerate(seq):
                player = (self.landlordPosition + i) % 3
                concrete = env_action_to_concrete(action, seq_used)
                self.whatTheyPlayed[player].append(concrete)
            # Recompute last non-pass from seq if available.
            trailing = 0
            self.lastValidCombo = CardCombo()
            self.lastValidPlayer = -1
            for idx in range(len(seq) - 1, -1, -1):
                player = (self.landlordPosition + idx) % 3
                concrete = env_action_to_concrete(seq[idx], set())
                if concrete:
                    self.lastValidCombo = CardCombo(concrete)
                    self.lastValidPlayer = player
                    break
                trailing += 1
            if trailing >= 2:
                self.lastValidCombo = CardCombo()
                self.lastValidPlayer = self.myPosition

    def get_num_cards_left(self, infoset) -> List[int]:
        default = [20, 17, 17]
        hand_len = len(getattr(infoset, "player_hand_cards", []))
        default[self.myPosition] = hand_len
        for attr in ["num_cards_left", "num_cards_left_dict", "player_num_cards_left"]:
            raw = getattr(infoset, attr, None)
            if raw is None:
                continue
            if isinstance(raw, dict):
                out = default[:]
                for k, v in raw.items():
                    if isinstance(k, str) and k in POS_INDEX:
                        out[POS_INDEX[k]] = int(v)
                    else:
                        try:
                            out[int(k)] = int(v)
                        except Exception:
                            pass
                return out
            if isinstance(raw, (list, tuple)) and len(raw) >= 3:
                return [int(raw[0]), int(raw[1]), int(raw[2])]
        return default

    # ---------------- original action generation ----------------
    def getActions(self, hand: CardCombinations, lastValidCombo: CardCombo) -> List[CardCombo]:
        actions: List[CardCombo] = []
        if lastValidCombo.comboType != CardComboType.PASS_:
            actions.append(CardCombo())

        t = lastValidCombo.comboType
        if t == CardComboType.INVALID:
            return actions
        if t == CardComboType.ROCKET:
            return actions
        if t == CardComboType.BOMB:
            hand.getAllRocket(); actions.extend(hand.rocket)
            hand.getAllBomb()
            actions.extend([bomb for bomb in hand.bomb if bomb.comboLevel > lastValidCombo.comboLevel])
            return actions
        if t == CardComboType.PASS_:
            hand.getAllCombos()
            actions.extend(hand.single)
            actions.extend(hand.pair)
            for length in range(5, 13): actions.extend(hand.straight[length])
            for length in range(3, 11): actions.extend(hand.straight2[length])
            actions.extend(hand.triplet)
            actions.extend(hand.triplet1)
            actions.extend(hand.triplet2)
            actions.extend(hand.bomb)
            actions.extend(hand.quadruple2)
            actions.extend(hand.quadruple4)
            for length in range(2, 7): actions.extend(hand.plane[length])
            for length in range(2, 6): actions.extend(hand.plane1[length])
            for length in range(2, 5): actions.extend(hand.plane2[length])
            actions.extend(hand.sshuttle)
            actions.extend(hand.sshuttle2)
            actions.extend(hand.sshuttle4)
            actions.extend(hand.rocket)
            return actions

        hand.getAllRocket(); actions.extend(hand.rocket)
        hand.getAllBomb(); actions.extend(hand.bomb)

        if t == CardComboType.SINGLE:
            hand.getAllSingle()
            actions.extend([x for x in hand.single if x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.PAIR:
            hand.getAllPair()
            actions.extend([x for x in hand.pair if x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.STRAIGHT:
            length = len(lastValidCombo.cards)
            hand.getAllStraight()
            for L in range(5, 13):
                actions.extend([x for x in hand.straight[L] if len(x.cards) == length and x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.STRAIGHT2:
            length = len(lastValidCombo.cards)
            hand.getAllStraight2()
            for L in range(3, 11):
                actions.extend([x for x in hand.straight2[L] if len(x.cards) == length and x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.TRIPLET:
            hand.getAllTriplet()
            actions.extend([x for x in hand.triplet if x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.TRIPLET1:
            hand.getAllTriplet1()
            actions.extend([x for x in hand.triplet1 if x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.TRIPLET2:
            hand.getAllTriplet2()
            actions.extend([x for x in hand.triplet2 if x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.QUADRUPLE2:
            hand.getAllQuadruple2()
            actions.extend([x for x in hand.quadruple2 if x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.QUADRUPLE4:
            hand.getAllQuadruple4()
            actions.extend([x for x in hand.quadruple4 if x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.PLANE:
            length = len(lastValidCombo.cards)
            hand.getAllPlane()
            for L in range(2, 7):
                actions.extend([x for x in hand.plane[L] if len(x.cards) == length and x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.PLANE1:
            length = len(lastValidCombo.cards)
            hand.getAllPlane1()
            for L in range(2, 6):
                actions.extend([x for x in hand.plane1[L] if len(x.cards) == length and x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.PLANE2:
            length = len(lastValidCombo.cards)
            hand.getAllPlane2()
            for L in range(2, 5):
                actions.extend([x for x in hand.plane2[L] if len(x.cards) == length and x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.SSHUTTLE:
            length = len(lastValidCombo.cards)
            hand.getAllSshuttle()
            actions.extend([x for x in hand.sshuttle if len(x.cards) == length and x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.SSHUTTLE2:
            length = len(lastValidCombo.cards)
            hand.getAllSshuttle2()
            actions.extend([x for x in hand.sshuttle2 if len(x.cards) == length and x.comboLevel > lastValidCombo.comboLevel])
        elif t == CardComboType.SSHUTTLE4:
            length = len(lastValidCombo.cards)
            hand.getAllSshuttle4()
            actions.extend([x for x in hand.sshuttle4 if len(x.cards) == length and x.comboLevel > lastValidCombo.comboLevel])

        # Remove duplicate concrete actions while preserving order.
        seen = set(); unique = []
        for a in actions:
            if a.key() not in seen:
                seen.add(a.key()); unique.append(a)
        return unique

    # ---------------- original utility/action policy ----------------
    def utility(self, hand: CardCombinations, alpha: int = 10) -> int:
        return self.getUtility(hand, 0, alpha)

    def getUtility(self, hand: CardCombinations, type_: int, alpha: int) -> int:
        if type_ == 0:
            hand.getAllRocket()
            if not hand.rocket:
                return self.getUtility(hand, 1, alpha)
            myRocket = hand.rocket[0]
            hand.erase(myRocket.cards)
            utility1 = self.getUtility(hand, 1, alpha) + 15
            hand.insert(myRocket.cards)
            utility2 = self.getUtility(hand, 1, alpha)
            return max(utility1, utility2)

        if type_ == 1:
            hand.getAllBomb()
            if not hand.bomb:
                return self.getUtility(hand, 2, alpha)
            biggestBomb = hand.bomb[0]
            hand.erase(biggestBomb.cards)
            utility1 = self.getUtility(hand, 1, alpha) + biggestBomb.comboLevel
            hand.insert(biggestBomb.cards)
            utility2 = self.getUtility(hand, 2, alpha)
            return max(utility1, utility2)

        if type_ == 2:
            hand.getAllPlane(); hand.getAllPlane1(); hand.getAllPlane2()
            if not hand.plane[2]:
                return self.getUtility(hand, 3, alpha)
            maxutility = -233333
            for L in range(2, 7):
                for planei in list(hand.plane[L]):
                    hand.erase(planei.cards)
                    utility1 = self.getUtility(hand, 2, alpha) + planei.comboLevel - L + 1 - alpha
                    if len(hand.plane[L]) > 1: utility1 += 2
                    maxutility = max(maxutility, utility1)
                    hand.insert(planei.cards)
                if L <= 5:
                    for plane1i in list(hand.plane1[L]):
                        hand.erase(plane1i.cards)
                        utility1 = self.getUtility(hand, 2, alpha) + plane1i.comboLevel - L + 1 - alpha
                        if len(hand.plane1[L]) > 1: utility1 += 2
                        maxutility = max(maxutility, utility1)
                        hand.insert(plane1i.cards)
                if L <= 4:
                    for plane2i in list(hand.plane2[L]):
                        hand.erase(plane2i.cards)
                        utility1 = self.getUtility(hand, 2, alpha) + plane2i.comboLevel - L + 1 - alpha
                        if len(hand.plane2[L]) > 1: utility1 += 2
                        maxutility = max(maxutility, utility1)
                        hand.insert(plane2i.cards)
            utility2 = self.getUtility(hand, 3, alpha)
            return max(maxutility, utility2)

        if type_ == 3:
            hand.getAllStraight2()
            if not hand.straight2[3]:
                return self.getUtility(hand, 4, alpha)
            maxutility = -233333
            for L in range(3, 11):
                for straight2i in list(hand.straight2[L]):
                    hand.erase(straight2i.cards)
                    utility1 = self.getUtility(hand, 3, alpha) + straight2i.comboLevel - L + 1 - alpha
                    if len(hand.straight2[L]) > 1: utility1 += 2
                    if straight2i.comboLevel == 11: utility1 += 1
                    maxutility = max(maxutility, utility1)
                    hand.insert(straight2i.cards)
            utility2 = self.getUtility(hand, 4, alpha)
            return max(maxutility, utility2)

        if type_ == 4:
            hand.getAllStraight()
            if not hand.straight[5]:
                return self.getUtility(hand, 5, alpha)
            maxutility = -233333
            for L in range(5, 13):
                for straighti in list(hand.straight[L]):
                    hand.erase(straighti.cards)
                    utility1 = self.getUtility(hand, 4, alpha) + straighti.comboLevel - L + 1 - alpha
                    if len(hand.straight[L]) > 1: utility1 += 2
                    if straighti.comboLevel == 11: utility1 += 1
                    maxutility = max(maxutility, utility1)
                    hand.insert(straighti.cards)
            utility2 = self.getUtility(hand, 5, alpha)
            return max(maxutility, utility2)

        if type_ == 5:
            hand.getAllSingle()
            utility1 = 0
            for singlei in hand.single:
                lv = singlei.comboLevel
                utility1 += lv + hand.packs[lv]["count"] - alpha
                if lv >= 11:
                    utility1 += hand.packs[lv]["count"] * hand.packs[lv]["count"] * (lv - 10)
            return utility1

        return 233333333

    def _findBestAction(self, hand: CardCombinations, lastValidCombo: CardCombo, alpha: int = 19, beta: int = 6) -> CardCombo:
        actions = self.getActions(hand, lastValidCombo)
        if not actions:
            return CardCombo()
        bestAction = CardCombo()
        bestMeasure = 0 if len(lastValidCombo.cards) else -233333
        if not len(lastValidCombo.cards):
            alpha += 3
        originalUtility = self.utility(hand, alpha)
        for action in actions:
            hand.erase(action.cards)
            if not hand.getLength():
                hand.insert(action.cards)
                return action
            tempMeasure = 10 * (self.utility(hand, alpha) - originalUtility) + beta * action.comboLevel
            if action.comboType in (CardComboType.BOMB, CardComboType.ROCKET):
                tempMeasure += beta * 6
            elif action.comboLevel >= 11:
                tempMeasure += beta * (action.comboLevel - 10 + 3 * (1 if action.comboLevel >= 13 else 0))
            hand.insert(action.cards)
            if tempMeasure > bestMeasure:
                bestMeasure = tempMeasure
                bestAction = action

        if (self.myPosition == (self.landlordPosition + 1) % 3
                and self.cardRemaining[(self.myPosition + 1) % 3] == 1
                and not len(lastValidCombo.cards)):
            hand.getAllSingle()
            if hand.single:
                return CardCombo(hand.single[0].cards, CardComboType.SINGLE)
        return bestAction

    def findBestAction(self, hand: CardCombinations, lastValidCombo: CardCombo, position: int, landlordPosition: int) -> CardCombo:
        alpha, beta = 19, 6
        if position == (landlordPosition + 2) % 3 and self.lastValidPlayer == landlordPosition:
            beta += 5
        if position == (landlordPosition + 2) % 3 and self.lastValidPlayer == (landlordPosition + 1) % 3 and lastValidCombo.comboLevel >= 9:
            beta -= lastValidCombo.comboLevel - 8
        if position == (landlordPosition + 1) % 3 and self.lastValidPlayer == (landlordPosition + 2) % 3:
            beta -= 5
        return self._findBestAction(hand, lastValidCombo, alpha, beta)

    def _findBestNActions(self, n_: int, hand: CardCombinations, lastValidCombo: CardCombo, alpha: int = 19, beta: int = 6) -> List[CardCombo]:
        actions = self.getActions(hand, lastValidCombo)
        n = min(n_, len(actions))
        if n == 0:
            return []
        if not len(lastValidCombo.cards):
            alpha += 3
        originalUtility = self.utility(hand, alpha)
        scored = []
        for action in actions:
            hand.erase(action.cards)
            tempMeasure = 10 * (self.utility(hand, alpha) - originalUtility) + beta * action.comboLevel
            if action.comboType in (CardComboType.BOMB, CardComboType.ROCKET):
                tempMeasure += beta * 6
            elif action.comboLevel >= 11:
                tempMeasure += beta * (action.comboLevel - 9 + (1 if action.comboLevel >= 13 else 0))
            hand.insert(action.cards)
            scored.append((tempMeasure, action))
        scored.sort(key=lambda x: (x[0], x[1].sort_key()), reverse=True)
        return [a for _, a in scored[:n]]

    def findBestNActions(self, n_: int, hand: CardCombinations, lastValidCombo: CardCombo, position: int, landlordPosition: int) -> List[CardCombo]:
        alpha, beta = 19, 6
        if position == landlordPosition:
            enemyLeft = min(self.cardRemaining[(position + 1) % 3], self.cardRemaining[(position + 2) % 3])
        else:
            enemyLeft = self.cardRemaining[landlordPosition]
        if enemyLeft <= 5:
            beta += 6 - enemyLeft
        if position == (landlordPosition + 2) % 3 and self.lastValidPlayer == landlordPosition:
            beta += 5
        if position == (landlordPosition + 2) % 3 and self.lastValidPlayer == (landlordPosition + 1) % 3 and lastValidCombo.comboLevel >= 9:
            beta -= lastValidCombo.comboLevel - 8
        if position == (landlordPosition + 1) % 3 and self.lastValidPlayer == (landlordPosition + 2) % 3:
            beta -= 5
        return self._findBestNActions(n_, hand, lastValidCombo, alpha, beta)

    # ---------------- original Monte Carlo rollout ----------------
    def prepareData(self):
        self.myHand = CardCombinations(sorted(self.myCards))
        self.remainingCards = set(range(54))
        for c in self.myCards:
            self.remainingCards.discard(c)
        for i in range(3):
            for vc in self.whatTheyPlayed[i]:
                for v in vc:
                    self.remainingCards.discard(v)
        self.root = TreeNode(self.myPosition, [CardCombinations(), CardCombinations(), CardCombinations()])
        self.root.hands[self.myPosition] = self.myHand.clone()
        self.root.turn = self.myPosition
        self.rootActions = self.findBestNActions(self.n_root_actions, self.myHand, self.lastValidCombo, self.myPosition, self.landlordPosition)
        if not self.rootActions:
            self.rootActions = [CardCombo()]
        self.winningTimes = {a.key(): 0 for a in self.rootActions}
        self.rootActionByKey = {a.key(): a for a in self.rootActions}
        if len(self.whatTheyPlayed[self.landlordPosition]):
            for vc in self.whatTheyPlayed[self.landlordPosition][1:]:
                if len(vc):
                    self.landlordHasNotPlayed = False

    def determinization(self, nextPlayerRemainingCards: int):
        a = list(self.remainingCards)
        self.rng.shuffle(a)
        next_cnt = max(0, min(int(nextPlayerRemainingCards), len(a)))
        self.nextHand = CardCombinations(a[:next_cnt])
        self.previousHand = CardCombinations(a[next_cnt:])
        hands = [h.clone() for h in self.root.hands]
        if self.myPosition == 0:
            hands[1] = self.nextHand.clone(); hands[2] = self.previousHand.clone()
        elif self.myPosition == 1:
            hands[0] = self.previousHand.clone(); hands[2] = self.nextHand.clone()
        elif self.myPosition == 2:
            hands[0] = self.nextHand.clone(); hands[1] = self.previousHand.clone()
        hands[self.myPosition] = self.myHand.clone()
        self.root.hands = hands

        self.sons = []
        for action in self.rootActions:
            root_hands = [h.clone() for h in self.root.hands]
            root_hands[self.myPosition].erase(action.cards)
            self.sons.append((action, TreeNode((self.myPosition + 1) % 3, root_hands)))

    def MCTS(self, node: TreeNode, lastValidCombo_: CardCombo, lastValidPlayer_: int) -> int:
        turn = node.turn
        if not node.hands[(turn + 2) % 3].getLength():
            winner = (turn + 2) % 3
            if (winner == self.landlordPosition
                    and node.hands[(self.landlordPosition + 1) % 3].getLength() == 17
                    and node.hands[(self.landlordPosition + 2) % 3].getLength() == 17):
                self.totalScore *= 2
            elif winner != self.landlordPosition and self.landlordHasNotPlayed_:
                self.totalScore *= 2
            if winner == self.myPosition:
                return self.totalScore
            elif self.myPosition != self.landlordPosition and winner != self.landlordPosition:
                return self.totalScore
            return -self.totalScore

        action = self.findBestAction(node.hands[turn], lastValidCombo_, turn, self.landlordPosition)
        nextLastCombo = action
        nextLastPlayer = turn
        if turn == self.landlordPosition and len(action.cards):
            self.landlordHasNotPlayed_ = False
        if action.comboType == CardComboType.PASS_:
            nextLastPlayer = lastValidPlayer_
            if lastValidPlayer_ == (turn + 2) % 3:
                nextLastCombo = lastValidCombo_
        elif action.comboType in (CardComboType.BOMB, CardComboType.ROCKET):
            self.totalScore *= 2

        new_hands = [h.clone() for h in node.hands]
        new_hands[turn].erase(action.cards)
        newNode = TreeNode((turn + 1) % 3, new_hands)
        return self.MCTS(newNode, nextLastCombo, nextLastPlayer)

    def calculateScores(self):
        deadline = self.startTime + self.time_limit_sec
        while time.perf_counter() < deadline:
            self.determinization(self.cardRemaining[(self.myPosition + 1) % 3])
            for action, son in self.sons:
                lastValidCombo_ = action
                lastValidPlayer_ = self.myPosition
                if action.comboType == CardComboType.PASS_:
                    lastValidPlayer_ = self.lastValidPlayer
                    if self.lastValidPlayer == (self.myPosition + 2) % 3:
                        lastValidCombo_ = self.lastValidCombo
                self.totalScore = 1
                if lastValidCombo_.comboType in (CardComboType.BOMB, CardComboType.ROCKET):
                    self.totalScore *= 2
                self.landlordHasNotPlayed_ = self.landlordHasNotPlayed
                if self.myPosition == self.landlordPosition and len(self.lastValidCombo.cards):
                    self.landlordHasNotPlayed_ = False
                self.winningTimes[action.key()] = self.winningTimes.get(action.key(), 0) + self.MCTS(son.clone(), lastValidCombo_, lastValidPlayer_)
                self.stats["rollouts"] += 1
                if time.perf_counter() >= deadline:
                    break

    def returnAction(self) -> CardCombo:
        self.calculateScores()
        if not self.rootActions:
            return CardCombo()
        bestKey = self.rootActions[0].key()
        bestWinningTimes = -10**18
        for key, score in self.winningTimes.items():
            if score >= bestWinningTimes:
                bestWinningTimes = score
                bestKey = key
        bestAction = self.rootActionByKey.get(bestKey, self.rootActions[0])

        remaining = dict(self.winningTimes)
        remaining.pop(bestKey, None)
        if not remaining:
            return bestAction
        secondBestWinningTimes = max(remaining.values()) if remaining else -10**18
        self.differenceBetweenFirstAndSecond = bestWinningTimes - secondBestWinningTimes
        first_key = self.rootActions[0].key()
        first_score = self.winningTimes.get(first_key, 0)
        if bestWinningTimes - secondBestWinningTimes >= 40:
            return bestAction
        elif bestWinningTimes - first_score >= 50:
            return bestAction
        return self.rootActions[0]

    # ---------------- local-interface helpers ----------------
    def match_legal_action(self, action_env: List[int], legal_actions: List[List[int]]) -> Optional[List[int]]:
        target = Counter(action_env)
        for a in legal_actions:
            if Counter(a) == target:
                return a
        return None

    def choose_best_from_legal_actions(self, infoset, legal_actions) -> Optional[List[int]]:
        # Re-score only RLCard legal actions with the same _findBestAction measure,
        # but restricted to legal_actions. This is a safety wrapper for rank-only
        # interfaces where concrete suit reconstruction may differ.
        legal_nonempty = [a for a in legal_actions if a]
        if not legal_nonempty and [] in legal_actions:
            return []
        hand = self.myHand.clone()
        last = self.lastValidCombo
        best = None
        best_score = -10**18 if not len(last.cards) else 0
        alpha, beta = 19, 6
        if not len(last.cards): alpha += 3
        original = self.utility(hand, alpha)
        used_base = set(self.myCards)
        for env_a in legal_actions:
            concrete = env_action_to_concrete(env_a, set())
            action = CardCombo(concrete)
            if action.comboType == CardComboType.INVALID:
                continue
            # Need concrete cards actually from my hand for erase. Convert by levels from myCards.
            try:
                concrete_from_hand = self.concrete_action_from_my_hand(env_a)
            except Exception:
                continue
            action = CardCombo(concrete_from_hand)
            hand.erase(action.cards)
            if not hand.getLength():
                hand.insert(action.cards)
                return env_a
            score = 10 * (self.utility(hand, alpha) - original) + beta * action.comboLevel
            if action.comboType in (CardComboType.BOMB, CardComboType.ROCKET):
                score += beta * 6
            elif action.comboLevel >= 11:
                score += beta * (action.comboLevel - 10 + 3 * (1 if action.comboLevel >= 13 else 0))
            hand.insert(action.cards)
            if score > best_score:
                best_score = score; best = env_a
        return best

    def concrete_action_from_my_hand(self, env_action: Iterable[int]) -> List[int]:
        by_level = defaultdict(list)
        for c in sorted(self.myCards):
            by_level[card2level(c)].append(c)
        out = []
        for lv in env_action_to_levels(env_action):
            if not by_level[lv]:
                raise ValueError("missing card")
            out.append(by_level[lv].pop(0))
        return sorted(out)


# Optional alias, if your evaluation script expects the same class name style.
# Prefer importing HighRankMonteCarloAgent explicitly to avoid confusing it with
# the original RLCard baseline.
MonteCarloUtilityAgent = HighRankMonteCarloAgent
