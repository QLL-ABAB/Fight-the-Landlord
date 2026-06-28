import functools
import math
import os
import pickle
import random
import time
from collections import deque

try:
    import torch
except ImportError:
    torch = None

from douzero.env import move_detector as md
from douzero.env.game import GameEnv
from douzero.rl.qlearning import (
    BOMB_ACTIONS,
    CARD_RANKS,
    CARD_TO_INDEX,
    POSITIONS,
    action_key,
    cards_to_counts,
    format_duration,
    generate_card_play_data,
    hand_badness,
    relation_to_last_player,
    reward_for_position,
    shaped_reward_for_action,
    teammate_position,
)


DEFAULT_APPROX_Q_PATH = "approx_qlearning_checkpoints/approx_qlearning/model.pkl"
FULL_DECK_COUNTS = tuple(
    1 if card in (20, 30) else 4
    for card in CARD_RANKS
)
ACTION_TYPE_VALUES = (
    md.TYPE_0_PASS,
    md.TYPE_1_SINGLE,
    md.TYPE_2_PAIR,
    md.TYPE_3_TRIPLE,
    md.TYPE_4_BOMB,
    md.TYPE_5_KING_BOMB,
    md.TYPE_6_3_1,
    md.TYPE_7_3_2,
    md.TYPE_8_SERIAL_SINGLE,
    md.TYPE_9_SERIAL_PAIR,
    md.TYPE_10_SERIAL_TRIPLE,
    md.TYPE_11_SERIAL_3_1,
    md.TYPE_12_SERIAL_3_2,
)


#note: 为 15 种牌面生成稳定的特征名，便于 checkpoint 和报告解释每一维含义。
def rank_feature_names(prefix):
    return tuple("{}_{}".format(prefix, card) for card in CARD_RANKS)


COMPACT_FEATURE_NAMES = (
    "bias",
    "position_landlord",
    "position_landlord_up",
    "position_landlord_down",
    "is_farmer",
    "my_cards_left",
    "landlord_cards_left",
    "landlord_up_cards_left",
    "landlord_down_cards_left",
    "teammate_cards_left",
    "min_enemy_cards_left",
    "max_enemy_cards_left",
    "leading_round",
    "last_player_self",
    "last_player_teammate",
    "last_player_enemy",
    "bomb_num",
    "hand_singles",
    "hand_pairs",
    "hand_triples",
    "hand_bombs",
    "hand_control",
    "hand_badness",
    "action_cards",
    "action_is_pass",
    "action_is_nonpass",
    "action_is_bomb",
    "action_is_king_bomb",
    "action_finish",
    "next_cards_left",
    "next_le_two",
    "action_min_rank",
    "action_max_rank",
    "action_avg_rank",
) + tuple("action_type_{}".format(t) for t in ACTION_TYPE_VALUES) + (
    "badness_delta",
    "singles_delta",
    "pairs_delta",
    "triples_delta",
    "bombs_delta",
    "control_delta",
    "teammate_danger",
    "teammate_danger_pass",
    "teammate_danger_block",
    "enemy_danger",
    "enemy_danger_press",
    "enemy_danger_pass",
    "bomb_not_near_finish",
    "bomb_near_finish",
)
HISTORY_FEATURE_NAMES = (
    COMPACT_FEATURE_NAMES
    + rank_feature_names("my_hand")
    + rank_feature_names("played_landlord")
    + rank_feature_names("played_landlord_up")
    + rank_feature_names("played_landlord_down")
    + rank_feature_names("total_played")
    + rank_feature_names("unseen")
    + rank_feature_names("last_move")
    + rank_feature_names("last_two_move_0")
    + rank_feature_names("last_two_move_1")
    + rank_feature_names("landlord_bottom")
    + rank_feature_names("action")
    + (
        "unseen_possible_bombs",
        "unseen_control_cards",
        "played_control_cards",
        "played_bomb_like_ranks",
    )
)
FEATURE_NAMES_BY_MODE = {
    "compact": COMPACT_FEATURE_NAMES,
    "history": HISTORY_FEATURE_NAMES,
}
FEATURE_NAMES = HISTORY_FEATURE_NAMES


#note: 按 feature_mode 返回特征名；所有训练和评测都通过这里保证维度一致。
def feature_names_for_mode(feature_mode):
    if feature_mode not in FEATURE_NAMES_BY_MODE:
        raise ValueError("Unknown feature_mode: {}".format(feature_mode))
    return FEATURE_NAMES_BY_MODE[feature_mode]


#note: 老 checkpoint 可能没有 feature_mode，用保存的 feature_names 反推是哪套特征。
def infer_feature_mode(saved_feature_names):
    names = tuple(saved_feature_names or [])
    if not names:
        return "history"
    for mode, expected_names in FEATURE_NAMES_BY_MODE.items():
        if names == expected_names:
            return mode
    raise ValueError("Unknown feature layout with {} features".format(len(names)))


#note: 启动时检查特征名没有重复，避免线性权重和解释文档错位。
def validate_feature_definitions():
    for mode, names in FEATURE_NAMES_BY_MODE.items():
        if len(names) != len(set(names)):
            raise ValueError("Duplicate feature names in mode {}".format(mode))


validate_feature_definitions()


#note: 根据参数选择训练设备；auto 会在可用时使用 CUDA，否则回退到 CPU；没有 torch 时使用纯 Python 后端。
def resolve_device(device):
    if torch is None:
        return "python"
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


#note: 判断当前后端是否是 torch，用于在 GPU/CPU tensor 和纯 Python list 之间切换。
def use_torch_backend(device):
    return torch is not None and isinstance(device, torch.device)


#note: 将牌面归一化到 0 到 1，避免线性模型因为大牌数值过大而不稳定。
def normalized_rank(card):
    if card is None:
        return 0.0
    index = CARD_TO_INDEX.get(card)
    if index is None:
        return 0.0
    return index / float(len(CARD_RANKS) - 1)


#note: 将 15 维牌数按每种牌最大数量归一化，普通牌除以 4，大小王除以 1。
def normalized_counts(counts):
    return [
        counts[index] / float(max_count)
        for index, max_count in enumerate(FULL_DECK_COUNTS)
    ]


#note: 三家已出牌按位置展开成固定 45 维，同时保留每个位置的身份信息。
def played_counts_by_position(infoset):
    played = infoset.played_cards or {}
    return {
        position: cards_to_counts(played.get(position, []))
        for position in POSITIONS
    }


#note: 汇总三家已出牌，得到全局可见历史牌面计数。
def total_played_counts(played_counts):
    return tuple(
        sum(played_counts[position][index] for position in POSITIONS)
        for index in range(len(CARD_RANKS))
    )


#note: 计算当前玩家视角下仍未见过的牌，只表示隐藏牌集合，不泄露具体归属。
def unseen_counts_from_view(hand_counts, played_counts):
    total_played = total_played_counts(played_counts)
    return tuple(
        max(0, FULL_DECK_COUNTS[index] - hand_counts[index] - total_played[index])
        for index in range(len(CARD_RANKS))
    )


#note: 取最近两手牌并补齐为空动作，保证 last_two_moves 始终是 2 x 15 维。
def last_two_move_counts(infoset):
    moves = list(infoset.last_two_moves or [])
    while len(moves) < 2:
        moves.append([])
    return [cards_to_counts(move) for move in moves[:2]]


#note: 给 history-aware 特征补充完整牌史、未见牌、当前动作和底牌信息。
def extend_history_features(values, infoset, action, hand_counts):
    played_counts = played_counts_by_position(infoset)
    total_played = total_played_counts(played_counts)
    unseen_counts = unseen_counts_from_view(hand_counts, played_counts)
    last_moves = last_two_move_counts(infoset)
    action_counts = cards_to_counts(action)
    landlord_bottom_counts = cards_to_counts(infoset.three_landlord_cards)

    values.extend(normalized_counts(hand_counts))
    for position in POSITIONS:
        values.extend(normalized_counts(played_counts[position]))
    values.extend(normalized_counts(total_played))
    values.extend(normalized_counts(unseen_counts))
    values.extend(normalized_counts(cards_to_counts(infoset.last_move)))
    values.extend(normalized_counts(last_moves[0]))
    values.extend(normalized_counts(last_moves[1]))
    values.extend(normalized_counts(landlord_bottom_counts))
    values.extend(normalized_counts(action_counts))

    values.extend([
        sum(1 for count in unseen_counts if count == 4) / 13.0,
        (
            unseen_counts[CARD_TO_INDEX[14]]
            + unseen_counts[CARD_TO_INDEX[17]]
            + unseen_counts[CARD_TO_INDEX[20]]
            + unseen_counts[CARD_TO_INDEX[30]]
        ) / 6.0,
        (
            total_played[CARD_TO_INDEX[14]]
            + total_played[CARD_TO_INDEX[17]]
            + total_played[CARD_TO_INDEX[20]]
            + total_played[CARD_TO_INDEX[30]]
        ) / 6.0,
        sum(1 for count in total_played if count >= 4) / 13.0,
    ])


#note: 可选地检查单个特征向量；默认关闭，避免大规模训练时产生额外开销。
def validate_feature_vector(values, feature_names, feature_mode):
    if len(values) != len(feature_names):
        raise ValueError(
            "Feature length mismatch for {}: {} != {}".format(
                feature_mode, len(values), len(feature_names)
            )
        )
    if os.environ.get("APPROXQ_VALIDATE_FEATURES", "0") != "1":
        return
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise ValueError(
                "Non-finite feature {}={} in mode {}".format(
                    feature_names[index], value, feature_mode
                )
            )


#note: 复制一手牌并移除动作中的牌，用于估计出牌后的手牌结构。
def remove_action_from_hand(hand_cards, action):
    next_hand = list(hand_cards or [])
    for card in action:
        if card in next_hand:
            next_hand.remove(card)
    return next_hand


#note: 统计固定 15 维牌数中的单牌、对子、三张、炸弹和控制牌数量。
@functools.lru_cache(maxsize=200000)
def hand_stats_from_counts(counts):
    singles = sum(1 for count in counts if count == 1)
    pairs = sum(1 for count in counts if count == 2)
    triples = sum(1 for count in counts if count == 3)
    bombs = sum(1 for count in counts if count == 4)
    control_cards = (
        counts[CARD_TO_INDEX[14]]
        + counts[CARD_TO_INDEX[17]]
        + counts[CARD_TO_INDEX[20]]
        + counts[CARD_TO_INDEX[30]]
    )
    return {
        "singles": singles,
        "pairs": pairs,
        "triples": triples,
        "bombs": bombs,
        "control": control_cards,
        "badness": hand_badness_from_counts(counts),
    }


#note: 用和 tabular Q-learning 一致的手牌坏度公式，但基于缓存后的 counts 直接计算。
@functools.lru_cache(maxsize=200000)
def hand_badness_from_counts(counts):
    singles = sum(1 for count in counts if count == 1)
    pairs = sum(1 for count in counts if count == 2)
    triples = sum(1 for count in counts if count == 3)
    bombs = sum(1 for count in counts if count == 4)
    groups = singles + pairs + triples + bombs
    control_cards = (
        counts[CARD_TO_INDEX[14]]
        + counts[CARD_TO_INDEX[17]]
        + counts[CARD_TO_INDEX[20]]
        + counts[CARD_TO_INDEX[30]]
    )
    return (
        0.03 * groups
        + 0.02 * singles
        + 0.003 * sum(counts)
        - 0.015 * control_cards
        - 0.02 * bombs
    )


#note: 缓存动作牌型、长度和牌面统计，避免每次评分都重复调用 move_detector。
@functools.lru_cache(maxsize=200000)
def cached_action_info(action):
    action = tuple(action)
    move_type = md.get_move_type(list(action))
    cards = list(action)
    ranks = [normalized_rank(card) for card in cards]
    action_type = move_type["type"]
    return {
        "type": action_type,
        "type_len": move_type.get("len", 1),
        "len": len(action),
        "is_pass": len(action) == 0,
        "is_bomb": action in BOMB_ACTIONS,
        "is_king_bomb": action == (20, 30),
        "min_rank": min(ranks) if ranks else 0.0,
        "max_rank": max(ranks) if ranks else 0.0,
        "avg_rank": sum(ranks) / len(ranks) if ranks else 0.0,
    }


#note: 返回某个位置的敌人列表，地主的敌人是两个农民，农民的敌人是地主。
def enemy_positions(position):
    if position == "landlord":
        return ("landlord_up", "landlord_down")
    return ("landlord",)


#note: 读取三家剩余牌数，并给缺失字段提供合理默认值。
def num_cards_left(infoset):
    num_left = infoset.num_cards_left_dict or {}
    return {position: num_left.get(position, 0) for position in POSITIONS}


#note: 判断当前是否处于主动出牌轮；last_move 为空时可以主动出任意合法牌。
def is_leading_round(infoset):
    return len(action_key(infoset.last_move)) == 0


#note: 生成固定维度人工特征，用线性模型近似 Q(s,a)，history 模式会额外编码完整公共牌史。
def make_feature_vector(position, infoset, action, feature_mode="history"):
    feature_names = feature_names_for_mode(feature_mode)
    action = action_key(action)
    info = cached_action_info(action)
    hand_cards = list(infoset.player_hand_cards or [])
    hand_count = len(hand_cards)
    next_hand = remove_action_from_hand(hand_cards, action)
    next_count = len(next_hand)

    hand_counts = cards_to_counts(hand_cards)
    next_counts = cards_to_counts(next_hand)
    hand_stats = hand_stats_from_counts(hand_counts)
    next_stats = hand_stats_from_counts(next_counts)

    left = num_cards_left(infoset)
    teammate = teammate_position(position)
    enemies = enemy_positions(position)
    enemy_left = [left.get(enemy, 0) for enemy in enemies]
    teammate_left = left.get(teammate, 0) if teammate else 0
    last_relation = relation_to_last_player(position, infoset.last_pid)
    leading = is_leading_round(infoset)
    teammate_danger = bool(teammate and teammate_left <= 2)
    enemy_danger = bool(enemy_left and min(enemy_left) <= 2)
    has_follow_action = any(legal_action for legal_action in infoset.legal_actions)

    values = [
        1.0,
        1.0 if position == "landlord" else 0.0,
        1.0 if position == "landlord_up" else 0.0,
        1.0 if position == "landlord_down" else 0.0,
        0.0 if position == "landlord" else 1.0,
        hand_count / 20.0,
        left.get("landlord", 0) / 20.0,
        left.get("landlord_up", 0) / 17.0,
        left.get("landlord_down", 0) / 17.0,
        teammate_left / 17.0,
        (min(enemy_left) if enemy_left else 0) / 20.0,
        (max(enemy_left) if enemy_left else 0) / 20.0,
        1.0 if leading else 0.0,
        1.0 if last_relation == "self" else 0.0,
        1.0 if last_relation == "teammate" else 0.0,
        1.0 if last_relation == "enemy" else 0.0,
        min(float(infoset.bomb_num or 0), 10.0) / 10.0,
        hand_stats["singles"] / 15.0,
        hand_stats["pairs"] / 15.0,
        hand_stats["triples"] / 15.0,
        hand_stats["bombs"] / 15.0,
        hand_stats["control"] / 10.0,
        hand_stats["badness"],
        info["len"] / 20.0,
        1.0 if info["is_pass"] else 0.0,
        0.0 if info["is_pass"] else 1.0,
        1.0 if info["is_bomb"] else 0.0,
        1.0 if info["is_king_bomb"] else 0.0,
        1.0 if action and next_count == 0 else 0.0,
        next_count / 20.0,
        1.0 if next_count <= 2 else 0.0,
        info["min_rank"],
        info["max_rank"],
        info["avg_rank"],
    ]
    values.extend(1.0 if info["type"] == t else 0.0 for t in ACTION_TYPE_VALUES)
    values.extend([
        max(-1.0, min(1.0, hand_stats["badness"] - next_stats["badness"])),
        (hand_stats["singles"] - next_stats["singles"]) / 15.0,
        (hand_stats["pairs"] - next_stats["pairs"]) / 15.0,
        (hand_stats["triples"] - next_stats["triples"]) / 15.0,
        (hand_stats["bombs"] - next_stats["bombs"]) / 15.0,
        (hand_stats["control"] - next_stats["control"]) / 10.0,
        1.0 if teammate_danger else 0.0,
        1.0 if teammate_danger and last_relation == "teammate" and info["is_pass"] else 0.0,
        1.0 if teammate_danger and last_relation == "teammate" and action else 0.0,
        1.0 if enemy_danger else 0.0,
        1.0 if enemy_danger and last_relation == "enemy" and action else 0.0,
        1.0 if enemy_danger and last_relation == "enemy" and has_follow_action and info["is_pass"] else 0.0,
        1.0 if info["is_bomb"] and next_count > 2 else 0.0,
        1.0 if info["is_bomb"] and next_count <= 2 else 0.0,
    ])

    if feature_mode == "history":
        extend_history_features(values, infoset, action, hand_counts)

    validate_feature_vector(values, feature_names, feature_mode)
    return values


#note: 用轻量启发式给动作排序，只在合法动作过多时剪枝以降低 max Q 计算压力。
def heuristic_action_score(position, infoset, action):
    action = action_key(action)
    info = cached_action_info(action)
    hand_cards = list(infoset.player_hand_cards or [])
    hand_count = len(hand_cards)
    next_hand = remove_action_from_hand(hand_cards, action)
    next_count = len(next_hand)

    if info["is_pass"]:
        score = -2.0 if is_leading_round(infoset) else 0.0
    else:
        score = 2.0 * info["len"] - info["max_rank"]
        score += 20.0 * max(-1.0, min(1.0, hand_badness(hand_cards) - hand_badness(next_hand)))
        if next_count == 0:
            score += 100.0
        elif next_count <= 2:
            score += 20.0
        if info["is_bomb"] and next_count > 2:
            score -= 20.0

    left = num_cards_left(infoset)
    teammate = teammate_position(position)
    if teammate and left.get(teammate, 17) <= 2:
        score += 12.0 if info["is_pass"] else -12.0

    enemies = enemy_positions(position)
    enemy_left = [left.get(enemy, 17) for enemy in enemies]
    if enemy_left and min(enemy_left) <= 2:
        score += 16.0 if action else -16.0

    if action and info["max_rank"] <= normalized_rank(10):
        score += 1.0
    if action and info["type"] in (md.TYPE_8_SERIAL_SINGLE, md.TYPE_9_SERIAL_PAIR):
        score += 4.0
    return score


#note: 保留出完、pass 和炸弹等关键动作，再从剩余动作里选启发式较好的候选。
def prune_legal_actions(position, infoset, max_candidate_actions):
    legal_actions = list(infoset.legal_actions or [])
    if max_candidate_actions <= 0 or len(legal_actions) <= max_candidate_actions:
        return legal_actions

    hand_count = len(infoset.player_hand_cards or [])
    protected = []
    protected_keys = set()
    scored = []
    for action in legal_actions:
        key = action_key(action)
        info = cached_action_info(key)
        is_protected = (
            info["is_pass"]
            or info["is_bomb"]
            or (key and len(key) == hand_count)
        )
        if is_protected and key not in protected_keys:
            protected.append(list(key))
            protected_keys.add(key)
        else:
            scored.append((heuristic_action_score(position, infoset, key), list(key)))

    scored.sort(key=lambda item: item[0], reverse=True)
    actions = protected[:max_candidate_actions]
    remaining = max_candidate_actions - len(actions)
    if remaining > 0:
        actions.extend(action for _, action in scored[:remaining])
    return actions or legal_actions[:max_candidate_actions]


#note: 将候选动作批量转成特征矩阵；有 torch 时返回 tensor，否则返回 Python list。
def features_for_actions(position, infoset, actions, device, feature_mode="history"):
    rows = [
        make_feature_vector(position, infoset, action, feature_mode)
        for action in actions
    ]
    feature_names = feature_names_for_mode(feature_mode)
    if use_torch_backend(device):
        if not rows:
            return torch.empty((0, len(feature_names)), dtype=torch.float32, device=device)
        return torch.tensor(rows, dtype=torch.float32, device=device)
    return rows


#note: 统一判断特征矩阵是否为空，兼容 torch tensor 和 Python list。
def features_are_empty(features):
    if use_torch_backend(getattr(features, "device", None)):
        return features.numel() == 0
    return not features


#note: 纯 Python 点积，用作无 torch 环境的轻量后端。
def list_dot(left, right):
    return sum(a * b for a, b in zip(left, right))


#note: 拼出当前任务的 checkpoint 目录。
def checkpoint_dir(flags):
    return os.path.join(flags.savedir, flags.name)


#note: 根据训练局数生成 checkpoint 路径；显式 output 会覆盖默认命名。
def checkpoint_path(flags, episodes):
    if flags.output:
        return flags.output
    return os.path.join(checkpoint_dir(flags), "{}.pkl".format(episodes))


#note: 从 checkpoint 文件名中恢复局数，便于 resume 继续训练。
def episode_from_checkpoint_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem) if stem.isdigit() else 0


#note: 在任务目录下寻找最新的完整 .pkl checkpoint。
def latest_checkpoint_path(flags):
    directory = checkpoint_dir(flags)
    if not os.path.isdir(directory):
        return None
    candidates = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if not filename.endswith(".pkl") or not os.path.isfile(path):
            continue
        episode = episode_from_checkpoint_path(path)
        if episode > 0:
            candidates.append((episode, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


#note: 保存训练配置和进度，评测 agent 会读取其中的剪枝参数和 reward 设置。
def training_metadata(flags, global_episode, epsilon, load_path, total_steps,
                      last_episode_steps, device, model, elapsed_sec=0.0,
                      completed_episodes=0):
    trained_episodes = max(0, global_episode - completed_episodes)
    return {
        "algorithm": "feature_based_approx_qlearning",
        "name": flags.name,
        "episodes": global_episode,
        "total_steps": total_steps,
        "last_episode_steps": last_episode_steps,
        "elapsed_sec": elapsed_sec,
        "episodes_per_sec": trained_episodes / max(1e-6, elapsed_sec),
        "seconds_per_episode": elapsed_sec / max(1, trained_episodes),
        "objective": flags.objective,
        "alpha": flags.alpha,
        "gamma": flags.gamma,
        "epsilon": epsilon,
        "reward_scale": flags.reward_scale,
        "reward_shaping": flags.reward_shaping,
        "max_candidate_actions": flags.max_candidate_actions,
        "feature_mode": model.feature_mode,
        "l2": flags.l2,
        "clip_td": flags.clip_td,
        "device": str(device),
        "savedir": flags.savedir,
        "resumed_from": load_path,
        "feature_dim": len(model.feature_names),
        "feature_names": list(model.feature_names),
        "updates": model.num_updates,
    }


#note: 打印 approximate Q-learning 的单行进度条。
def print_progress_bar(episode, total_episodes, completed_episodes, model,
                       epsilon, recent_steps, recent_landlord_wins,
                       recent_td, start_time, total_steps):
    if total_episodes <= 0:
        return
    progress = episode / float(total_episodes)
    width = 30
    filled = int(width * progress)
    bar = "#" * filled + "." * (width - filled)
    elapsed = time.time() - start_time
    speed = episode / elapsed if elapsed > 0 else 0.0
    remaining = (total_episodes - episode) / speed if speed > 0 else 0.0
    avg_steps = sum(recent_steps) / float(len(recent_steps)) if recent_steps else 0.0
    landlord_wp = (
        sum(recent_landlord_wins) / float(len(recent_landlord_wins))
        if recent_landlord_wins else 0.0
    )
    avg_td = sum(recent_td) / float(len(recent_td)) if recent_td else 0.0
    global_episode = completed_episodes + episode
    line = (
        "\r[{}] {:6.2f}% episode={}/{} global={} updates={} "
        "epsilon={:.4f} landlord_wp={:.3f} avg_steps={:.1f} "
        "avg_abs_td={:.4f} speed={:.2f}eps/s steps={} eta={}"
    ).format(
        bar,
        progress * 100.0,
        episode,
        total_episodes,
        global_episode,
        model.num_updates,
        epsilon,
        landlord_wp,
        avg_steps,
        avg_td,
        speed,
        total_steps,
        format_duration(remaining),
    )
    print(line + " " * 8, end="", flush=True)


class ApproxQModel(object):
    #note: 线性 Q 模型，三种位置各有一组权重，内存大小固定为 3 x feature_dim。
    def __init__(self, device="auto", weights=None, metadata=None,
                 feature_mode="history"):
        self.device = resolve_device(device)
        self.feature_mode = feature_mode
        self.feature_names = feature_names_for_mode(feature_mode)
        self.torch_backend = use_torch_backend(self.device)
        self.weights = {}
        for position in POSITIONS:
            values = list(weights[position]) if weights and position in weights else None
            if self.torch_backend:
                if values is not None:
                    tensor = torch.as_tensor(values, dtype=torch.float32)
                else:
                    tensor = torch.zeros(len(self.feature_names), dtype=torch.float32)
                self.weights[position] = tensor.to(self.device)
            else:
                self.weights[position] = (
                    [float(value) for value in values]
                    if values is not None else
                    [0.0 for _ in self.feature_names]
                )
            if len(self.weights[position]) != len(self.feature_names):
                raise ValueError(
                    "Weight length mismatch for {}: {} != {}".format(
                        position,
                        len(self.weights[position]),
                        len(self.feature_names),
                    )
                )
        self.metadata = metadata or {}
        self.num_updates = int(self.metadata.get("updates", 0) or 0)

    #note: 批量计算某个位置下所有候选动作的 Q 值。
    def q_values(self, position, features):
        if self.torch_backend:
            if features.numel() == 0:
                return torch.empty((0,), dtype=torch.float32, device=self.device)
            return features.matmul(self.weights[position])
        if not features:
            return []
        weights = self.weights[position]
        return [list_dot(weights, feature) for feature in features]

    #note: 返回候选动作中的最大 Q 值；没有候选动作时返回 0。
    def best_value(self, position, features):
        if features_are_empty(features):
            return 0.0
        q_values = self.q_values(position, features)
        if self.torch_backend:
            return float(torch.max(q_values).item())
        return float(max(q_values))

    #note: epsilon-greedy 选择动作，并返回被选动作对应的特征向量。
    def select_action(self, position, actions, features, epsilon=0.0, rng=None):
        if not actions:
            if self.torch_backend:
                empty_feature = torch.zeros(
                    len(self.feature_names), dtype=torch.float32, device=self.device
                )
            else:
                empty_feature = [0.0 for _ in self.feature_names]
            return [], empty_feature
        rng = rng or random
        if epsilon > 0.0 and rng.random() < epsilon:
            index = rng.randrange(len(actions))
            feature = features[index].clone() if self.torch_backend else list(features[index])
            return list(actions[index]), feature

        q_values = self.q_values(position, features)
        if self.torch_backend:
            q_values = q_values.detach().cpu().tolist()
        best = max(q_values)
        best_indices = [idx for idx, value in enumerate(q_values) if value == best]
        index = rng.choice(best_indices)
        feature = features[index].clone() if self.torch_backend else list(features[index])
        return list(actions[index]), feature

    #note: 按课程 Q-learning 公式更新线性权重 w。
    def update(self, position, feature, reward, next_features, alpha, gamma,
               l2=0.0, clip_td=0.0):
        if self.torch_backend:
            with torch.no_grad():
                feature = feature.to(self.device)
                old_value = torch.dot(self.weights[position], feature)
                next_best = 0.0
                if next_features is not None and next_features.numel() > 0:
                    next_best = torch.max(self.q_values(position, next_features))
                target = (
                    torch.as_tensor(reward, dtype=torch.float32, device=self.device)
                    + gamma * next_best
                )
                delta = target - old_value
                if clip_td and clip_td > 0:
                    delta = torch.clamp(delta, -clip_td, clip_td)
                update = delta * feature
                if l2 and l2 > 0:
                    update = update - l2 * self.weights[position]
                self.weights[position].add_(alpha * update)
                self.num_updates += 1
                return float(abs(delta).item())

        feature = list(feature)
        weights = self.weights[position]
        old_value = list_dot(weights, feature)
        next_best = 0.0
        if next_features is not None and next_features:
            next_best = max(self.q_values(position, next_features))
        delta = reward + gamma * next_best - old_value
        if clip_td and clip_td > 0:
            delta = max(-clip_td, min(clip_td, delta))
        for index, value in enumerate(feature):
            update = delta * value
            if l2 and l2 > 0:
                update -= l2 * weights[index]
            weights[index] += alpha * update
        self.num_updates += 1
        return abs(float(delta))

    #note: 保存模型权重；先写临时文件再原子替换，避免进程中断留下坏 checkpoint。
    def save(self, path, metadata=None):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.metadata = metadata or self.metadata
        payload = {
            "version": 1,
            "algorithm": "feature_based_approx_qlearning",
            "feature_mode": self.feature_mode,
            "feature_names": list(self.feature_names),
            "weights": {
                position: (
                    weight.detach().cpu().tolist()
                    if self.torch_backend else
                    list(weight)
                )
                for position, weight in self.weights.items()
            },
            "metadata": self.metadata,
        }
        tmp_path = "{}.tmp".format(path)
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

    #note: 从 checkpoint 加载线性模型，并可选择加载到 CPU 或 GPU。
    @classmethod
    def load(cls, path, device="auto"):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "weights" not in payload:
            raise ValueError("Invalid approximate Q-learning checkpoint: {}".format(path))
        feature_names = tuple(payload.get("feature_names", []))
        feature_mode = payload.get("feature_mode")
        if feature_mode is None:
            feature_mode = payload.get("metadata", {}).get("feature_mode")
        if feature_mode is None:
            feature_mode = infer_feature_mode(feature_names)
        expected_feature_names = feature_names_for_mode(feature_mode)
        if feature_names and feature_names != expected_feature_names:
            raise ValueError(
                "Feature mismatch: checkpoint has {} features, code expects {}".format(
                    len(feature_names), len(expected_feature_names)
                )
            )
        return cls(
            device=device,
            weights=payload["weights"],
            metadata=payload.get("metadata", {}),
            feature_mode=feature_mode,
        )


class SelfPlayApproxQLearningAgent(object):
    #note: 自博弈训练用 agent，负责暂存上一次动作并在下一次轮到自己时做 TD 更新。
    def __init__(self, position, model, alpha, gamma, epsilon,
                 max_candidate_actions=64, reward_scale=1.0,
                 reward_shaping=False, l2=0.0, clip_td=0.0, rng=None,
                 td_log=None):
        self.name = "SelfPlayApproxQLearning"
        self.position = position
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_candidate_actions = max_candidate_actions
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.l2 = l2
        self.clip_td = clip_td
        self.rng = rng or random.Random()
        self.pending = None
        self.td_log = td_log

    #note: 每局开始时清空 pending，避免跨局错误更新。
    def begin_episode(self):
        self.pending = None

    #note: 取当前合法动作候选，并批量构建特征。
    def action_features(self, infoset):
        actions = prune_legal_actions(
            self.position, infoset, self.max_candidate_actions
        )
        features = features_for_actions(
            self.position, infoset, actions, self.model.device,
            self.model.feature_mode
        )
        return actions, features

    #note: 被环境调用时先更新上一状态，再对当前状态选择动作。
    def act(self, infoset):
        actions, features = self.action_features(infoset)
        if self.pending is not None:
            prev_feature, prev_reward = self.pending
            td_error = self.model.update(
                self.position,
                prev_feature,
                reward=prev_reward,
                next_features=features,
                alpha=self.alpha,
                gamma=self.gamma,
                l2=self.l2,
                clip_td=self.clip_td,
            )
            if self.td_log is not None:
                self.td_log.append(td_error)

        action, feature = self.model.select_action(
            self.position, actions, features, epsilon=self.epsilon, rng=self.rng
        )
        shaped_reward = 0.0
        if self.reward_shaping:
            shaped_reward = shaped_reward_for_action(
                self.position, infoset, action, self.reward_scale
            )
        self.pending = (feature, shaped_reward)
        return action

    #note: 游戏结束时用终局 reward 更新该角色最后一次动作。
    def finish_episode(self, reward):
        if self.pending is None:
            return
        feature, shaped_reward = self.pending
        td_error = self.model.update(
            self.position,
            feature,
            reward=reward + shaped_reward,
            next_features=None,
            alpha=self.alpha,
            gamma=self.gamma,
            l2=self.l2,
            clip_td=self.clip_td,
        )
        if self.td_log is not None:
            self.td_log.append(td_error)
        self.pending = None


#note: 训练入口，使用自博弈生成经验并在线更新固定维度线性 Q 模型。
def train(flags):
    rng = random.Random(flags.seed)
    device = resolve_device(flags.device)
    load_path = None
    if flags.load:
        load_path = flags.load
    elif flags.resume and flags.output:
        load_path = flags.output
    elif flags.resume:
        load_path = latest_checkpoint_path(flags)

    if flags.resume and not load_path:
        raise FileNotFoundError(
            "Cannot resume: no checkpoint found in {}".format(checkpoint_dir(flags))
        )

    if load_path:
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                "Cannot resume: approximate Q model not found at {}".format(load_path)
            )
        model = ApproxQModel.load(load_path, device=device)
        completed_episodes = int(
            model.metadata.get("episodes", 0) or episode_from_checkpoint_path(load_path)
        )
        total_steps = int(model.metadata.get("total_steps", 0))
        print("loaded approximate Q model from {} (episodes={}, updates={})".format(
            load_path, completed_episodes, model.num_updates
        ))
    else:
        model = ApproxQModel(device=device, feature_mode=flags.feature_mode)
        completed_episodes = 0
        total_steps = 0

    recent_landlord_wins = deque(maxlen=max(1, flags.log_interval))
    recent_steps = deque(maxlen=max(1, flags.log_interval))
    recent_td = deque(maxlen=max(1, flags.log_interval * 4))
    agents = {
        position: SelfPlayApproxQLearningAgent(
            position=position,
            model=model,
            alpha=flags.alpha,
            gamma=flags.gamma,
            epsilon=flags.epsilon,
            max_candidate_actions=flags.max_candidate_actions,
            reward_scale=flags.reward_scale,
            reward_shaping=flags.reward_shaping,
            l2=flags.l2,
            clip_td=flags.clip_td,
            rng=random.Random(flags.seed + index + 1),
            td_log=recent_td,
        )
        for index, position in enumerate(POSITIONS)
    }
    env = GameEnv(agents)

    epsilon = flags.epsilon
    if load_path and "epsilon" in model.metadata:
        epsilon = float(model.metadata["epsilon"])
    progress_enabled = bool(flags.progress_interval and flags.progress_interval > 0)
    start_time = time.time()
    last_steps = 0

    print("Approx Q-learning device: {}".format(model.device))
    print("Feature mode: {}".format(model.feature_mode))
    print("Feature dim: {}".format(len(model.feature_names)))
    print("Reward objective: {} scale={} shaping={}".format(
        flags.objective, flags.reward_scale, flags.reward_shaping
    ))

    for episode in range(1, flags.episodes + 1):
        global_episode = completed_episodes + episode
        for agent in agents.values():
            agent.begin_episode()
            agent.epsilon = epsilon

        env.reset()
        env.card_play_init(generate_card_play_data(rng))

        steps = 0
        while not env.game_over and steps < flags.max_steps:
            env.step()
            steps += 1

        if env.game_over:
            winner = env.get_winner()
            bomb_num = env.get_bomb_num()
            for position, agent in agents.items():
                reward = reward_for_position(
                    position, winner, bomb_num, flags.objective,
                    flags.reward_scale
                )
                agent.finish_episode(reward)
            recent_landlord_wins.append(1 if winner == "landlord" else 0)
        else:
            for agent in agents.values():
                agent.finish_episode(0.0)
            recent_landlord_wins.append(0)

        recent_steps.append(steps)
        last_steps = steps
        total_steps += steps
        epsilon = max(flags.min_epsilon, epsilon * flags.epsilon_decay)

        should_log = flags.log_interval and episode % flags.log_interval == 0
        should_save = flags.save_interval and episode % flags.save_interval == 0
        should_progress = (
            progress_enabled
            and (
                episode == 1
                or episode == flags.episodes
                or episode % flags.progress_interval == 0
                or should_log
                or should_save
            )
        )

        if should_progress:
            print_progress_bar(
                episode, flags.episodes, completed_episodes, model, epsilon,
                recent_steps, recent_landlord_wins, recent_td, start_time,
                total_steps
            )

        if should_log:
            if progress_enabled:
                print()
            win_rate = sum(recent_landlord_wins) / float(len(recent_landlord_wins))
            avg_steps = sum(recent_steps) / float(len(recent_steps))
            avg_td = sum(recent_td) / float(len(recent_td)) if recent_td else 0.0
            elapsed = max(1e-6, time.time() - start_time)
            print(
                "episode={} updates={} epsilon={:.4f} landlord_wp={:.3f} "
                "avg_steps={:.1f} avg_abs_td={:.4f} elapsed_sec={:.1f} "
                "speed={:.2f}eps/s sec_per_ep={:.4f}".format(
                    global_episode, model.num_updates, epsilon, win_rate,
                    avg_steps, avg_td, elapsed,
                    episode / elapsed,
                    elapsed / max(1, episode)
                )
            )

        if should_save:
            if progress_enabled and not should_log:
                print()
            path = checkpoint_path(flags, global_episode)
            model.save(path, metadata=training_metadata(
                flags, global_episode, epsilon, load_path, total_steps,
                steps, device, model, max(1e-6, time.time() - start_time),
                completed_episodes
            ))
            print("saved approximate Q model to {}".format(path))

    if progress_enabled:
        print()

    final_episode = completed_episodes + flags.episodes
    final_path = checkpoint_path(flags, final_episode)
    model.save(final_path, metadata=training_metadata(
        flags, final_episode, epsilon, load_path, total_steps,
        last_steps, device, model, max(1e-6, time.time() - start_time),
        completed_episodes
    ))
    print("saved approximate Q model to {}".format(final_path))
    return model
