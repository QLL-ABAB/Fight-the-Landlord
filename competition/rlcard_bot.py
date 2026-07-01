import collections
import itertools
import json
import random
import sys


# Botzone/Botzeno card id:
# 0-3 are 3s, 4-7 are 4s, ..., 48-51 are 2s, 52 is small joker, 53 is big joker.
# Internal DouZero-style rank values:
# 3..14 are 3..A, 17 is 2, 20 is small joker, 30 is big joker.
BOTZONE_TO_ENV = {i: (i // 4) + 3 for i in range(48)}
for i in range(48, 52):
    BOTZONE_TO_ENV[i] = 17
BOTZONE_TO_ENV[52] = 20
BOTZONE_TO_ENV[53] = 30

ENV_TO_REAL = {
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
REAL_TO_ENV = {v: k for k, v in ENV_TO_REAL.items()}
REAL_ORDER = {
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

TYPE_PASS = 0
TYPE_SINGLE = 1
TYPE_PAIR = 2
TYPE_TRIPLE = 3
TYPE_BOMB = 4
TYPE_ROCKET = 5
TYPE_3_1 = 6
TYPE_3_2 = 7
TYPE_SERIAL_SINGLE = 8
TYPE_SERIAL_PAIR = 9
TYPE_SERIAL_TRIPLE = 10
TYPE_SERIAL_3_1 = 11
TYPE_SERIAL_3_2 = 12
TYPE_WRONG = 15

MIN_SINGLE_CARDS = 5
MIN_PAIRS = 3
MIN_TRIPLES = 2
POSITIONS = ["landlord", "landlord_down", "landlord_up"]


def botzone_to_env_cards(cards):
    return sorted(BOTZONE_TO_ENV[c] for c in cards)


def env_action_to_botzone(action, hand_ids):
    remaining = sorted(hand_ids)
    picked = []
    for rank in sorted(action):
        for card_id in remaining:
            if BOTZONE_TO_ENV[card_id] == rank:
                picked.append(card_id)
                remaining.remove(card_id)
                break
        else:
            return []
    return sorted(picked)


def is_continuous_seq(move):
    return all(move[i + 1] - move[i] == 1 for i in range(len(move) - 1))


def move_type(move):
    move = sorted(move)
    size = len(move)
    counts = collections.Counter(move)

    if size == 0:
        return {"type": TYPE_PASS}
    if size == 1:
        return {"type": TYPE_SINGLE, "rank": move[0]}
    if size == 2:
        if move == [20, 30]:
            return {"type": TYPE_ROCKET, "rank": 30}
        if move[0] == move[1]:
            return {"type": TYPE_PAIR, "rank": move[0]}
        return {"type": TYPE_WRONG}
    if size == 3:
        if len(counts) == 1:
            return {"type": TYPE_TRIPLE, "rank": move[0]}
        return {"type": TYPE_WRONG}
    if size == 4:
        if len(counts) == 1:
            return {"type": TYPE_BOMB, "rank": move[0]}
        if sorted(counts.values()) == [1, 3]:
            return {"type": TYPE_3_1, "rank": rank_with_count(counts, 3)}
        return {"type": TYPE_WRONG}
    if is_plain_sequence(move):
        return {"type": TYPE_SERIAL_SINGLE, "rank": move[0], "len": len(move)}
    if size == 5:
        if sorted(counts.values()) == [2, 3]:
            return {"type": TYPE_3_2, "rank": rank_with_count(counts, 3)}
        return {"type": TYPE_WRONG}

    value_counts = collections.Counter(counts.values())
    ranks = sorted(counts.keys())
    if len(counts) == value_counts.get(2, 0) and is_valid_chain(ranks, MIN_PAIRS):
        return {"type": TYPE_SERIAL_PAIR, "rank": ranks[0], "len": len(ranks)}
    if len(counts) == value_counts.get(3, 0) and is_valid_chain(ranks, MIN_TRIPLES):
        return {"type": TYPE_SERIAL_TRIPLE, "rank": ranks[0], "len": len(ranks)}

    triples = sorted(k for k, v in counts.items() if v == 3)
    singles = [k for k, v in counts.items() if v == 1]
    pairs = [k for k, v in counts.items() if v == 2]
    if len(triples) >= MIN_TRIPLES and is_valid_chain(triples, MIN_TRIPLES):
        if len(triples) == len(singles) and not pairs:
            return {"type": TYPE_SERIAL_3_1, "rank": triples[0], "len": len(triples)}
        if len(triples) == len(pairs) and not singles:
            return {"type": TYPE_SERIAL_3_2, "rank": triples[0], "len": len(triples)}

    return {"type": TYPE_WRONG}


def rank_with_count(counts, n):
    return max(k for k, v in counts.items() if v == n)


def is_valid_chain(ranks, min_len):
    return len(ranks) >= min_len and ranks[-1] < 17 and is_continuous_seq(ranks)


def is_plain_sequence(move):
    return (
        len(move) >= MIN_SINGLE_CARDS
        and len(set(move)) == len(move)
        and move[-1] < 17
        and is_continuous_seq(move)
    )


def select(cards, n):
    return [list(x) for x in itertools.combinations(cards, n)]


class MovesGener:
    def __init__(self, cards):
        self.cards = sorted(cards)
        self.counts = collections.Counter(cards)
        self.singles = [[c] for c in sorted(set(cards))]
        self.pairs = [[c, c] for c, n in sorted(self.counts.items()) if n >= 2]
        self.triples = [[c, c, c] for c, n in sorted(self.counts.items()) if n >= 3]
        self.bombs = [[c, c, c, c] for c, n in sorted(self.counts.items()) if n == 4]
        self.rocket = [[20, 30]] if self.counts[20] and self.counts[30] else []

    def serial_moves(self, source, min_len, repeat=1, exact_len=0):
        ranks = sorted(set(c for c in source if c < 17))
        runs = []
        start = 0
        for i in range(len(ranks)):
            if i + 1 == len(ranks) or ranks[i + 1] != ranks[i] + 1:
                runs.append(ranks[start : i + 1])
                start = i + 1

        out = []
        for run in runs:
            if len(run) < min_len:
                continue
            lengths = [exact_len] if exact_len else range(min_len, len(run) + 1)
            for length in lengths:
                if length < min_len or length > len(run):
                    continue
                for idx in range(0, len(run) - length + 1):
                    out.append(sorted(run[idx : idx + length] * repeat))
        return out

    def serial_single(self, exact_len=0):
        return self.serial_moves(self.cards, MIN_SINGLE_CARDS, 1, exact_len)

    def serial_pair(self, exact_len=0):
        pair_ranks = [c for c, n in self.counts.items() if n >= 2]
        return self.serial_moves(pair_ranks, MIN_PAIRS, 2, exact_len)

    def serial_triple(self, exact_len=0):
        triple_ranks = [c for c, n in self.counts.items() if n >= 3]
        return self.serial_moves(triple_ranks, MIN_TRIPLES, 3, exact_len)

    def type_3_1(self):
        out = []
        for triple in self.triples:
            for single in self.singles:
                if single[0] != triple[0]:
                    out.append(sorted(triple + single))
        return out

    def type_3_2(self):
        out = []
        for triple in self.triples:
            for pair in self.pairs:
                if pair[0] != triple[0]:
                    out.append(sorted(triple + pair))
        return out

    def serial_3_1(self, exact_len=0):
        out = []
        for base in self.serial_triple(exact_len):
            triple_ranks = set(base)
            wings = [c for c in self.cards if c not in triple_ranks]
            for wing in select(wings, len(triple_ranks)):
                out.append(sorted(base + wing))
        return unique_moves(out)

    def serial_3_2(self, exact_len=0):
        out = []
        for base in self.serial_triple(exact_len):
            triple_ranks = set(base)
            pair_ranks = [c for c, n in self.counts.items() if n >= 2 and c not in triple_ranks]
            for wing in select(pair_ranks, len(triple_ranks)):
                out.append(sorted(base + [c for c in wing for _ in range(2)]))
        return unique_moves(out)

    def all_moves(self):
        moves = []
        moves.extend(self.singles)
        moves.extend(self.pairs)
        moves.extend(self.triples)
        moves.extend(self.bombs)
        moves.extend(self.rocket)
        moves.extend(self.type_3_1())
        moves.extend(self.type_3_2())
        moves.extend(self.serial_single())
        moves.extend(self.serial_pair())
        moves.extend(self.serial_triple())
        moves.extend(self.serial_3_1())
        moves.extend(self.serial_3_2())
        return unique_moves(moves)


def unique_moves(moves):
    seen = set()
    out = []
    for move in moves:
        key = tuple(sorted(move))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out


def beats(move, rival):
    mt = move_type(move)
    rt = move_type(rival)
    if mt["type"] == TYPE_WRONG:
        return False
    if rt["type"] in (TYPE_PASS, TYPE_WRONG):
        return len(move) > 0
    if mt["type"] == TYPE_ROCKET:
        return rt["type"] != TYPE_ROCKET
    if mt["type"] == TYPE_BOMB and rt["type"] != TYPE_BOMB:
        return rt["type"] != TYPE_ROCKET
    if mt["type"] != rt["type"]:
        return False
    if mt.get("len", 1) != rt.get("len", 1):
        return False
    return mt.get("rank", -1) > rt.get("rank", -1)


def legal_actions(hand, last_non_pass):
    generator = MovesGener(hand)
    if not last_non_pass:
        return generator.all_moves()
    moves = [m for m in generator.all_moves() if beats(m, last_non_pass)]
    moves.append([])
    return unique_moves(moves)


def card_str(hand):
    return "".join(ENV_TO_REAL[c] for c in sorted(hand))


def combine_cards(hand_str):
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
    hand = "".join(sorted(hand_str, key=lambda c: REAL_ORDER[c]))
    counts = collections.Counter(hand)
    if counts["B"] and counts["R"]:
        comb["rocket"].append("BR")
        hand = hand.replace("B", "", 1).replace("R", "", 1)

    for card in list(REAL_ORDER):
        bomb = card * 4
        if hand.count(card) == 4:
            comb["bomb"].append(bomb)
            hand = hand.replace(bomb, "")

    for card in list(REAL_ORDER)[:13]:
        trio = card * 3
        if hand.count(card) >= 3:
            comb["trio"].append(trio)
            hand = hand.replace(trio, "", 1)

    hand_list = [hand.count(card) for card in REAL_ORDER]
    chains, hand_list = pick_chain(hand_list, 1)
    comb["solo_chain"] = chains
    chains, hand_list = pick_chain(hand_list, 2)
    comb["pair_chain"] = chains

    rest = ""
    cards = list(REAL_ORDER)
    for idx, count in enumerate(hand_list):
        rest += cards[idx] * count
    for card in cards:
        while rest.count(card) >= 2:
            comb["pair"].append(card * 2)
            rest = rest.replace(card * 2, "", 1)
    for card in rest:
        comb["solo"].append(card)
    return comb


def pick_chain(hand_list, repeat):
    cards = list(REAL_ORDER)[:12]
    chains = []
    i = 0
    while i < len(cards):
        if hand_list[i] < repeat:
            i += 1
            continue
        j = i
        while j < len(cards) and hand_list[j] >= repeat:
            j += 1
        if j - i >= MIN_SINGLE_CARDS:
            chain = "".join(cards[i:j])
            chains.append(chain)
            for k in range(i, j):
                hand_list[k] -= repeat
        i = j
    return chains, hand_list


def choose_rlcard_like_action(position, hand, actions, last_move, last_two, last_pid):
    if not actions:
        return []
    if last_two == [[], []]:
        hand_str = card_str(hand)
        min_card = hand_str[0] if hand_str else ""
        for _, candidates in combine_cards(hand_str).items():
            for candidate in candidates:
                action = sorted(REAL_TO_ENV[c] for c in candidate)
                if min_card in candidate and action in actions:
                    return action
    else:
        rival_type = move_type(last_move)["type"]
        candidates = [
            action
            for action in actions
            if action and move_type(action)["type"] == rival_type
        ]
        if candidates:
            return min(candidates, key=action_sort_key)
        if last_pid != "landlord" and position != "landlord" and [] in actions:
            return []
    return min(actions, key=action_sort_key)


def action_sort_key(action):
    mt = move_type(action)
    if not action:
        return (99, 0, 0)
    return (len(action), mt.get("rank", max(action)), max(action))


def choose_bid(request):
    bid_history = request.get("bid", [])
    current = max([0] + [int(x) for x in bid_history])
    hand = botzone_to_env_cards(request.get("own", []))
    counts = collections.Counter(hand)
    strength = 0
    strength += 2 * sum(1 for _, n in counts.items() if n == 4)
    strength += 3 if counts[20] and counts[30] else 0
    strength += counts[17]
    strength += counts[14]
    desired = 0
    if strength >= 6:
        desired = 3
    elif strength >= 4:
        desired = 2
    elif strength >= 2:
        desired = 1
    return desired if desired > current else 0


def active_position(requests):
    latest = requests[-1]
    if "landlord" not in latest:
        return None
    landlord = latest["landlord"]
    pos = latest.get("pos")
    if pos is None:
        for req in reversed(requests):
            if "pos" in req:
                pos = req["pos"]
                break
    if pos is None:
        pos = landlord
    return position_from_player_id(pos, landlord)


def position_from_player_id(player_id, landlord):
    if player_id == landlord:
        return "landlord"
    if player_id == (landlord + 1) % 3:
        return "landlord_down"
    return "landlord_up"


def collect_action_history(requests, responses):
    actions = []
    owners = []
    for idx, response in enumerate(responses):
        req = requests[idx] if idx < len(requests) else {}
        if "landlord" in req and isinstance(response, list):
            actions.append(response)
            owners.append(req.get("pos"))
    last_req = requests[-1]
    for item in last_req.get("history", []):
        if isinstance(item, list):
            actions.append(item)
    return actions


def latest_non_pass(actions):
    passes = 0
    for action in reversed(actions):
        if action:
            return botzone_to_env_cards(action), passes
        passes += 1
    return [], passes


def reconstruct_hand(requests, responses):
    own = []
    public = []
    landlord = None
    for req in requests:
        if "own" in req:
            own = list(req["own"])
        if "publiccard" in req:
            public = list(req["publiccard"])
        if "landlord" in req:
            landlord = req["landlord"]

    if landlord is not None:
        pos = None
        for req in requests:
            if "pos" in req:
                pos = req["pos"]
                break
        if pos == landlord:
            for card in public:
                if card not in own:
                    own.append(card)

    for response in responses:
        if isinstance(response, list):
            for card in response:
                if card in own:
                    own.remove(card)
    return sorted(own)


def main():
    raw_input = sys.stdin.readline().lstrip("\ufeff")
    full_input = json.loads(raw_input)
    requests = full_input.get("requests", [])
    responses = full_input.get("responses", [])
    if not requests:
        print(json.dumps(0))
        return

    current_request = requests[-1]
    if "landlord" not in current_request:
        print(json.dumps(choose_bid(current_request)))
        return

    hand_ids = reconstruct_hand(requests, responses)
    hand = botzone_to_env_cards(hand_ids)
    history = collect_action_history(requests, responses)
    last_move, pass_count = latest_non_pass(history)
    if pass_count >= 2:
        last_move = []

    actions = legal_actions(hand, last_move)
    position = active_position(requests) or "landlord"
    last_two = [botzone_to_env_cards(x) for x in history[-2:]]
    while len(last_two) < 2:
        last_two.insert(0, [])
    last_pid = "landlord"
    for idx in range(len(history) - 1, -1, -1):
        if history[idx]:
            landlord = current_request["landlord"]
            player_id = (landlord + idx) % 3
            last_pid = position_from_player_id(player_id, landlord)
            break

    action = choose_rlcard_like_action(position, hand, actions, last_move, last_two, last_pid)
    print(json.dumps(env_action_to_botzone(action, hand_ids), separators=(",", ":")))


if __name__ == "__main__":
    main()
