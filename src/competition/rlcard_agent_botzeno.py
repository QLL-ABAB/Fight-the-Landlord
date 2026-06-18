from __future__ import annotations

import json
import random
import sys
from collections import Counter

from douzero.env.move_detector import get_move_type
from douzero.env.move_generator import MovesGener
from douzero.env import move_selector as ms
from douzero.evaluation.rlcard_agent import RLCardAgent


CARD_ID_TO_ENV_RANK = {}
for card_id in range(0, 52):
    raw_rank = card_id // 4 + 3
    CARD_ID_TO_ENV_RANK[card_id] = 17 if raw_rank == 15 else raw_rank
CARD_ID_TO_ENV_RANK[52] = 20
CARD_ID_TO_ENV_RANK[53] = 30


class SimpleInfoset:
    def __init__(self, position, player_hand_cards, last_move, last_two_moves, legal_actions, last_pid):
        self.player_position = position
        self.player_hand_cards = player_hand_cards
        self.last_move = last_move
        self.last_two_moves = last_two_moves
        self.legal_actions = legal_actions
        self.last_pid = last_pid


def card_id_to_env_rank(card_id):
    return CARD_ID_TO_ENV_RANK[int(card_id)]


def card_ids_to_env_ranks(card_ids):
    return [card_id_to_env_rank(card_id) for card_id in card_ids]


def env_rank_to_card_ids(env_cards, available_cards):
    buckets = {}
    for card_id in available_cards:
        rank = card_id_to_env_rank(card_id)
        buckets.setdefault(rank, []).append(card_id)

    for ids in buckets.values():
        ids.sort()

    result = []
    for rank in env_cards:
        ids = buckets.get(rank)
        if not ids:
            raise ValueError("cannot map env card %r back to raw card ids" % rank)
        result.append(ids.pop(0))
    return result


def move_key(move):
    return tuple(sorted(move))


def card_counts(cards):
    counts = {}
    for card in cards:
        counts[card] = counts.get(card, 0) + 1
    return counts


def build_move_infos(hand_cards):
    moves = MovesGener(hand_cards).gen_moves()
    move_infos = []
    seen = set()
    for move in moves:
        move = move_key(move)
        if move in seen:
            continue
        seen.add(move)
        move_type = get_move_type(list(move))
        if move_type["type"] == 15:
            continue
        move_infos.append(
            {
                "move": move,
                "type": move_type["type"],
                "len": move_type.get("len", 1),
                "counts": tuple(sorted(card_counts(move).items())),
            }
        )
    return move_infos


def current_move_infos(hand_cards):
    return build_move_infos(hand_cards)


def moves_from_cache(hand_cards, move_type=None, move_len=None):
    moves = []
    for info in current_move_infos(hand_cards):
        if move_type is not None and info["type"] != move_type:
            continue
        if move_len is not None and info["len"] != move_len:
            continue
        moves.append(list(info["move"]))
    return moves


def bomb_moves_from_cache(hand_cards):
    moves = []
    for info in current_move_infos(hand_cards):
        if info["type"] in [4, 5]:
            moves.append(list(info["move"]))
    return moves


def get_legal_card_play_actions(hand_cards, history):
    rival_move = []
    if history:
        if history[-1] == [] and len(history) >= 2:
            rival_move = history[-2]
        else:
            rival_move = history[-1]

    rival_type = get_move_type(rival_move)
    rival_move_type = rival_type["type"]
    rival_move_len = rival_type.get("len", 1)
    moves = []

    if rival_move_type == 0:
        moves = moves_from_cache(hand_cards)
    elif rival_move_type == 1:
        all_moves = moves_from_cache(hand_cards, 1)
        moves = ms.filter_type_1_single(all_moves, list(rival_move))
    elif rival_move_type == 2:
        all_moves = moves_from_cache(hand_cards, 2)
        moves = ms.filter_type_2_pair(all_moves, list(rival_move))
    elif rival_move_type == 3:
        all_moves = moves_from_cache(hand_cards, 3)
        moves = ms.filter_type_3_triple(all_moves, list(rival_move))
    elif rival_move_type == 4:
        all_moves = bomb_moves_from_cache(hand_cards)
        moves = ms.filter_type_4_bomb(all_moves, list(rival_move))
    elif rival_move_type == 5:
        moves = []
    elif rival_move_type == 6:
        all_moves = moves_from_cache(hand_cards, 6)
        moves = ms.filter_type_6_3_1(all_moves, list(rival_move))
    elif rival_move_type == 7:
        all_moves = moves_from_cache(hand_cards, 7)
        moves = ms.filter_type_7_3_2(all_moves, list(rival_move))
    elif rival_move_type == 8:
        all_moves = moves_from_cache(hand_cards, 8, rival_move_len)
        moves = ms.filter_type_8_serial_single(all_moves, list(rival_move))
    elif rival_move_type == 9:
        all_moves = moves_from_cache(hand_cards, 9, rival_move_len)
        moves = ms.filter_type_9_serial_pair(all_moves, list(rival_move))
    elif rival_move_type == 10:
        all_moves = moves_from_cache(hand_cards, 10, rival_move_len)
        moves = ms.filter_type_10_serial_triple(all_moves, list(rival_move))
    elif rival_move_type == 11:
        all_moves = moves_from_cache(hand_cards, 11, rival_move_len)
        moves = ms.filter_type_11_serial_3_1(all_moves, list(rival_move))
    elif rival_move_type == 12:
        all_moves = moves_from_cache(hand_cards, 12, rival_move_len)
        moves = ms.filter_type_12_serial_3_2(all_moves, list(rival_move))

    if rival_move_type not in [0, 4, 5]:
        moves = moves + bomb_moves_from_cache(hand_cards)

    if len(rival_move) != 0:
        moves = moves + [[]]

    for move in moves:
        move.sort()
    return moves


def choose_bid(hand_cards):
    counts = Counter(hand_cards)
    score = 0
    if 20 in counts and 30 in counts:
        score += 4
    score += sum(1 for v in counts.values() if v >= 4) * 2
    score += sum(1 for card in hand_cards if card >= 14)
    score += sum(1 for v in counts.values() if v >= 3)
    if score >= 8:
        return 3
    if score >= 5:
        return 2
    if score >= 2:
        return 1
    return 0


def parse_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")

    content = payload.get("content", {})
    if not isinstance(content, dict) or not content:
        raise ValueError("missing content")

    player_key = next(iter(content.keys()))
    data = content[player_key]
    if not isinstance(data, dict):
        raise ValueError("content payload must be an object")
    return player_key, data


def act_from_request(payload):
    _, data = parse_request(payload)

    if "own" not in data:
        raise ValueError("missing own cards")

    own_cards = list(data.get("own", []))
    history = data.get("history", [])
    if not isinstance(history, list):
        history = []

    is_bidding = "publiccard" not in data and "landlord" not in data and "finalbid" not in data and not history
    if is_bidding:
        return choose_bid(own_cards)

    hand_cards = [card_id_to_env_rank(card_id) for card_id in own_cards]
    if "publiccard" in data and data.get("pos") == data.get("landlord"):
        hand_cards.extend(card_id_to_env_rank(card_id) for card_id in data.get("publiccard", []))
        hand_cards.sort()

    last_move = []
    if history:
        if history[-1] == [] and len(history) >= 2:
            last_move = card_ids_to_env_ranks(history[-2])
        else:
            last_move = card_ids_to_env_ranks(history[-1])

    last_two_moves = [[], []]
    if len(history) >= 2:
        last_two_moves = [card_ids_to_env_ranks(history[-2]), card_ids_to_env_ranks(history[-1])]
    elif len(history) == 1:
        last_two_moves = [[], card_ids_to_env_ranks(history[-1])]

    env_history = [card_ids_to_env_ranks(move) for move in history]
    legal_actions = get_legal_card_play_actions(hand_cards, env_history)
    infoset = SimpleInfoset(
        position=data.get("position") or data.get("pos") or data.get("landlord") or "landlord",
        player_hand_cards=hand_cards,
        last_move=last_move,
        last_two_moves=last_two_moves,
        legal_actions=legal_actions,
        last_pid=data.get("last_pid") or data.get("landlord") or "landlord",
    )

    action_env = RLCardAgent(infoset.player_position).act(infoset)
    return env_rank_to_card_ids(action_env, own_cards)


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        return
    payload = json.loads(raw)
    response = act_from_request(payload)
    sys.stdout.write(json.dumps({"verdict": "OK", "response": response}, ensure_ascii=False))


if __name__ == "__main__":
    main()