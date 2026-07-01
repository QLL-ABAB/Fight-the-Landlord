from __future__ import annotations

import os

from douzero.env import move_detector as md
from douzero.rl import approx_qlearning as base
from douzero.rl import better_approx_qlearning as better
from douzero.rl.qlearning import (
    CARD_TO_INDEX,
    POSITIONS,
    action_key,
    cards_to_counts,
    relation_to_last_player,
    teammate_position,
)


DEFAULT_PRECISE_APPROX_Q_PATH = (
    "approx_qlearning_checkpoints/approxq_precise/model.pkl"
)
DEFAULT_PRECISE_APPROX_Q_DIR = "approx_qlearning_checkpoints/approxq_precise"

NEXT_POSITION = {
    "landlord": "landlord_down",
    "landlord_down": "landlord_up",
    "landlord_up": "landlord",
}

ROUGH_FEATURE_NAMES = {
    "action_type_0",
    "action_is_pass",
    "action_preserves_control",
}


def precise_base_feature_names():
    return tuple(
        name
        for name in better.BETTER_HISTORY_FULL_FEATURE_NAMES
        if name not in ROUGH_FEATURE_NAMES
    )


PRECISE_PASS_FEATURE_NAMES = (
    "pass_after_self",
    "pass_after_teammate",
    "pass_after_landlord",
    "pass_after_farmer",
    "pass_after_enemy",
    "pass_vs_single",
    "pass_vs_pair",
    "pass_vs_triple",
    "pass_vs_straight",
    "pass_vs_serial_pair",
    "pass_vs_bomb",
    "pass_when_teammate_next",
    "pass_when_landlord_next",
    "pass_when_enemy_next",
    "pass_when_landlord_low",
    "pass_when_teammate_low",
    "pass_when_self_low",
    "pass_when_have_noncontrol_response",
    "pass_when_only_control_response",
    "pass_when_have_bomb_response",
    "pass_after_teammate_to_landlord_next",
    "pass_after_landlord_to_teammate_next",
    "farmer_pass_after_teammate_low",
    "farmer_pass_after_landlord_low",
    "landlord_pass_when_farmer_low",
)

PRECISE_CONTROL_FEATURE_NAMES = (
    "preserve_control_as_landlord",
    "preserve_control_as_farmer",
    "preserve_control_when_leading",
    "preserve_control_when_following",
    "preserve_control_after_teammate",
    "preserve_control_after_landlord",
    "preserve_control_after_enemy",
    "preserve_control_when_landlord_low",
    "preserve_control_when_teammate_low",
    "preserve_control_when_enemy_low",
    "preserve_control_when_self_low",
    "preserve_control_with_noncontrol_response",
    "preserve_control_with_multiple_control_left",
    "use_control_to_finish",
    "use_control_to_block_landlord",
    "use_control_to_block_farmer",
    "use_control_over_teammate",
    "use_control_when_only_response",
    "use_control_when_bomb_available",
    "control_saved_for_endgame",
)

PRECISE_HAND_SUPERVISION_NAMES = (
    "residual_single_low_after_best_straight",
    "residual_single_mid_after_best_straight",
    "residual_single_high_after_best_straight",
    "residual_single_low_after_action_straight",
    "residual_single_mid_after_action_straight",
    "residual_single_high_after_action_straight",
    "stranded_low_single_count",
    "stranded_low_single_ratio",
    "stranded_low_single_after_action_count",
    "stranded_low_single_after_action_ratio",
    "stranded_low_single_delta",
    "residual_pair_low_after_best_serial_pair",
    "residual_pair_mid_after_best_serial_pair",
    "residual_pair_high_after_best_serial_pair",
    "residual_pair_low_after_action_serial_pair",
    "residual_pair_mid_after_action_serial_pair",
    "residual_pair_high_after_action_serial_pair",
    "stranded_low_pair_count",
    "stranded_low_pair_ratio",
    "stranded_low_pair_after_action_count",
    "stranded_low_pair_after_action_ratio",
    "stranded_low_pair_delta",
    "best_straight_len",
    "best_serial_pair_len",
)

PRECISE_EXTRA_FEATURE_NAMES = (
    PRECISE_PASS_FEATURE_NAMES
    + PRECISE_CONTROL_FEATURE_NAMES
    + PRECISE_HAND_SUPERVISION_NAMES
)

PRECISE_HISTORY_FULL_FEATURE_NAMES = (
    precise_base_feature_names()
    + PRECISE_EXTRA_FEATURE_NAMES
)

FEATURE_NAMES_BY_MODE = dict(better.FEATURE_NAMES_BY_MODE)
FEATURE_NAMES_BY_MODE["precise_history_full"] = PRECISE_HISTORY_FULL_FEATURE_NAMES


def feature_names_for_mode(feature_mode):
    if feature_mode not in FEATURE_NAMES_BY_MODE:
        raise ValueError("Unknown precise feature_mode: {}".format(feature_mode))
    return FEATURE_NAMES_BY_MODE[feature_mode]


def validate_feature_definitions():
    for mode, names in FEATURE_NAMES_BY_MODE.items():
        if len(names) != len(set(names)):
            raise ValueError("Duplicate feature names in precise mode {}".format(mode))


validate_feature_definitions()


def count_at(counts, card):
    return counts[CARD_TO_INDEX[card]]


def low_rank_count(counts):
    return sum(count_at(counts, card) for card in (3, 4, 5, 6, 7, 8, 9, 10))


def next_position(position):
    return NEXT_POSITION[position]


def relation_to_position(position, other):
    if other == position:
        return "self"
    teammate = teammate_position(position)
    if teammate and other == teammate:
        return "teammate"
    return "enemy"


def move_type_group(move_type):
    if move_type == md.TYPE_1_SINGLE:
        return "single"
    if move_type == md.TYPE_2_PAIR:
        return "pair"
    if move_type in (md.TYPE_3_TRIPLE, md.TYPE_6_3_1, md.TYPE_7_3_2):
        return "triple"
    if move_type in (md.TYPE_8_SERIAL_SINGLE, md.TYPE_10_SERIAL_TRIPLE,
                     md.TYPE_11_SERIAL_3_1, md.TYPE_12_SERIAL_3_2):
        return "straight"
    if move_type == md.TYPE_9_SERIAL_PAIR:
        return "serial_pair"
    if move_type in (md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB):
        return "bomb"
    return "other"


def action_is_control(action_counts):
    return better.control_count(action_counts) > 0


def response_context(position, infoset):
    legal_actions = [action_key(action) for action in (infoset.legal_actions or [])]
    nonpass = [action for action in legal_actions if action]
    noncontrol = []
    control = []
    bombs = []
    for action in nonpass:
        counts = cards_to_counts(action)
        info = base.cached_action_info(action)
        if action_is_control(counts):
            control.append(action)
        else:
            noncontrol.append(action)
        if info["is_bomb"]:
            bombs.append(action)

    last_move = action_key(infoset.last_move)
    last_info = base.cached_action_info(last_move)
    last_pid = infoset.last_pid
    last_relation = relation_to_last_player(position, last_pid)
    next_pos = next_position(position)
    next_relation = relation_to_position(position, next_pos)

    return {
        "last_move": last_move,
        "last_info": last_info,
        "last_relation": last_relation,
        "last_pid": last_pid,
        "last_group": move_type_group(last_info["type"]),
        "next_position": next_pos,
        "next_relation": next_relation,
        "have_noncontrol_response": bool(noncontrol),
        "have_control_response": bool(control),
        "only_control_response": bool(control) and not noncontrol,
        "have_bomb_response": bool(bombs),
    }


def rank_slots():
    return (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)


def best_sequence_length(counts, repeat, min_len):
    best = 0
    current = 0
    for card in rank_slots():
        if count_at(counts, card) >= repeat:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best if best >= min_len else 0


def remove_best_sequence_counts(counts, repeat, min_len):
    best_start = None
    best_len = 0
    current_start = None
    current_len = 0
    for card in rank_slots():
        if count_at(counts, card) >= repeat:
            if current_len == 0:
                current_start = card
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
        else:
            current_len = 0
            current_start = None
    remaining = list(counts)
    if best_start is None or best_len < min_len:
        return tuple(remaining), 0
    cards = rank_slots()
    start_index = cards.index(best_start)
    for card in cards[start_index:start_index + best_len]:
        remaining[CARD_TO_INDEX[card]] = max(0, remaining[CARD_TO_INDEX[card]] - repeat)
    return tuple(remaining), best_len


def residual_low_single_metrics(counts):
    remaining, best_len = remove_best_sequence_counts(tuple(counts), 1, 5)
    low_singles = sum(
        1
        for card in (3, 4, 5, 6, 7, 8, 9, 10)
        if remaining[CARD_TO_INDEX[card]] == 1
    )
    mid_singles = sum(
        1 for card in (11, 12, 13) if remaining[CARD_TO_INDEX[card]] == 1
    )
    high_singles = sum(
        1 for card in (14, 17, 20, 30) if remaining[CARD_TO_INDEX[card]] == 1
    )
    return {
        "low": low_singles / 8.0,
        "mid": mid_singles / 3.0,
        "high": high_singles / 4.0,
        "low_count": low_singles,
        "low_ratio": low_singles / 8.0,
        "best_len": best_len / 12.0,
    }


def residual_low_pair_metrics(counts):
    remaining, best_len = remove_best_sequence_counts(tuple(counts), 2, 3)
    low_pairs = sum(
        1
        for card in (3, 4, 5, 6, 7, 8, 9, 10)
        if remaining[CARD_TO_INDEX[card]] == 2
    )
    mid_pairs = sum(
        1 for card in (11, 12, 13) if remaining[CARD_TO_INDEX[card]] == 2
    )
    high_pairs = sum(
        1 for card in (14, 17) if remaining[CARD_TO_INDEX[card]] == 2
    )
    return {
        "low": low_pairs / 8.0,
        "mid": mid_pairs / 3.0,
        "high": high_pairs / 2.0,
        "low_count": low_pairs,
        "low_ratio": low_pairs / 8.0,
        "best_len": best_len / 10.0,
    }


def is_action_straight(action_info):
    return action_info["type"] in (
        md.TYPE_8_SERIAL_SINGLE,
        md.TYPE_10_SERIAL_TRIPLE,
        md.TYPE_11_SERIAL_3_1,
        md.TYPE_12_SERIAL_3_2,
    )


def is_action_serial_pair(action_info):
    return action_info["type"] == md.TYPE_9_SERIAL_PAIR


def precise_extra_features(position, infoset, action, cached=None):
    action = action_key(action)
    hand_cards = list(infoset.player_hand_cards or [])
    next_hand = base.remove_action_from_hand(hand_cards, action)
    hand_counts = cached["hand_counts"] if cached else cards_to_counts(hand_cards)
    next_counts = cards_to_counts(next_hand)
    action_counts = cards_to_counts(action)
    unseen_counts = (
        cached["unseen_counts"]
        if cached else
        base.unseen_counts_from_view(hand_counts, base.played_counts_by_position(infoset))
    )

    ctx = cached["response_context"] if cached else response_context(position, infoset)
    action_info = base.cached_action_info(action)
    action_is_pass = len(action) == 0
    action_uses_control = action_is_control(action_counts)

    left = base.num_cards_left(infoset)
    teammate = teammate_position(position)
    teammate_left = left.get(teammate, 0) if teammate else 0
    landlord_left = left.get("landlord", 0)
    enemies = base.enemy_positions(position)
    enemy_left = [left.get(enemy, 0) for enemy in enemies]
    enemy_min_left = min(enemy_left) if enemy_left else 0

    is_landlord = position == "landlord"
    is_farmer = not is_landlord
    self_low = len(next_hand) <= 2
    teammate_low = bool(teammate and teammate_left <= 2)
    landlord_low = landlord_left <= 2
    enemy_low = enemy_min_left <= 2
    next_control = better.control_count(next_counts)
    hand_control = better.control_count(hand_counts)
    preserves_control = hand_control > 0 and next_control == hand_control
    multiple_control_left = next_control >= 2
    leading = base.is_leading_round(infoset)

    last_pid = ctx["last_pid"]
    last_after_landlord = last_pid == "landlord"
    last_after_farmer = last_pid in ("landlord_up", "landlord_down")

    pass_features = [
        action_is_pass and ctx["last_relation"] == "self",
        action_is_pass and ctx["last_relation"] == "teammate",
        action_is_pass and last_after_landlord,
        action_is_pass and last_after_farmer,
        action_is_pass and ctx["last_relation"] == "enemy",
        action_is_pass and ctx["last_group"] == "single",
        action_is_pass and ctx["last_group"] == "pair",
        action_is_pass and ctx["last_group"] == "triple",
        action_is_pass and ctx["last_group"] == "straight",
        action_is_pass and ctx["last_group"] == "serial_pair",
        action_is_pass and ctx["last_group"] == "bomb",
        action_is_pass and ctx["next_relation"] == "teammate",
        action_is_pass and ctx["next_position"] == "landlord",
        action_is_pass and ctx["next_relation"] == "enemy",
        action_is_pass and landlord_low,
        action_is_pass and teammate_low,
        action_is_pass and self_low,
        action_is_pass and ctx["have_noncontrol_response"],
        action_is_pass and ctx["only_control_response"],
        action_is_pass and ctx["have_bomb_response"],
        action_is_pass and ctx["last_relation"] == "teammate" and ctx["next_position"] == "landlord",
        action_is_pass and last_after_landlord and ctx["next_relation"] == "teammate",
        action_is_pass and is_farmer and ctx["last_relation"] == "teammate" and teammate_low,
        action_is_pass and is_farmer and last_after_landlord and landlord_low,
        action_is_pass and is_landlord and enemy_low,
    ]

    control_features = [
        preserves_control and is_landlord,
        preserves_control and is_farmer,
        preserves_control and leading,
        preserves_control and not leading,
        preserves_control and ctx["last_relation"] == "teammate",
        preserves_control and last_after_landlord,
        preserves_control and ctx["last_relation"] == "enemy",
        preserves_control and landlord_low,
        preserves_control and teammate_low,
        preserves_control and enemy_low,
        preserves_control and self_low,
        preserves_control and ctx["have_noncontrol_response"],
        preserves_control and multiple_control_left,
        action_uses_control and len(next_hand) == 0,
        action_uses_control and is_farmer and landlord_low,
        action_uses_control and is_landlord and enemy_low,
        action_uses_control and is_farmer and ctx["last_relation"] == "teammate",
        action_uses_control and ctx["only_control_response"],
        action_uses_control and ctx["have_bomb_response"],
        preserves_control and len(next_hand) <= 5,
    ]

    hand_straight = (
        cached["hand_straight"]
        if cached else
        residual_low_single_metrics(hand_counts)
    )
    next_straight = residual_low_single_metrics(next_counts)
    hand_pair = cached["hand_pair"] if cached else residual_low_pair_metrics(hand_counts)
    next_pair = residual_low_pair_metrics(next_counts)

    action_straight = is_action_straight(action_info)
    action_serial_pair = is_action_serial_pair(action_info)

    hand_features = [
        hand_straight["low"],
        hand_straight["mid"],
        hand_straight["high"],
        next_straight["low"] if action_straight else 0.0,
        next_straight["mid"] if action_straight else 0.0,
        next_straight["high"] if action_straight else 0.0,
        hand_straight["low_count"] / 8.0,
        hand_straight["low_ratio"],
        next_straight["low_count"] / 8.0,
        next_straight["low_ratio"],
        (next_straight["low_count"] - hand_straight["low_count"]) / 8.0,
        hand_pair["low"],
        hand_pair["mid"],
        hand_pair["high"],
        next_pair["low"] if action_serial_pair else 0.0,
        next_pair["mid"] if action_serial_pair else 0.0,
        next_pair["high"] if action_serial_pair else 0.0,
        hand_pair["low_count"] / 8.0,
        hand_pair["low_ratio"],
        next_pair["low_count"] / 8.0,
        next_pair["low_ratio"],
        (next_pair["low_count"] - hand_pair["low_count"]) / 8.0,
        hand_straight["best_len"],
        hand_pair["best_len"],
    ]

    bool_values = pass_features + control_features
    values = [1.0 if value else 0.0 for value in bool_values]
    values.extend(hand_features)
    return values


def make_feature_vector(position, infoset, action, feature_mode="precise_history_full",
                        cached=None):
    if feature_mode != "precise_history_full":
        return better.make_feature_vector(position, infoset, action, feature_mode)
    full_values = better.make_feature_vector(
        position, infoset, action, "better_history_full"
    )
    values = [
        value
        for name, value in zip(better.BETTER_HISTORY_FULL_FEATURE_NAMES, full_values)
        if name not in ROUGH_FEATURE_NAMES
    ]
    values.extend(precise_extra_features(position, infoset, action, cached))
    base.validate_feature_vector(values, feature_names_for_mode(feature_mode), feature_mode)
    return values


def features_for_actions(position, infoset, actions, device, feature_mode="precise_history_full"):
    cached = None
    if feature_mode == "precise_history_full":
        hand_counts = cards_to_counts(infoset.player_hand_cards or [])
        played_counts = base.played_counts_by_position(infoset)
        cached = {
            "hand_counts": hand_counts,
            "unseen_counts": base.unseen_counts_from_view(hand_counts, played_counts),
            "response_context": response_context(position, infoset),
            "hand_straight": residual_low_single_metrics(hand_counts),
            "hand_pair": residual_low_pair_metrics(hand_counts),
        }
    rows = [
        make_feature_vector(position, infoset, action, feature_mode, cached)
        for action in actions
    ]
    names = feature_names_for_mode(feature_mode)
    if base.use_torch_backend(device):
        if not rows:
            return base.torch.empty((0, len(names)), dtype=base.torch.float32, device=device)
        return base.torch.tensor(rows, dtype=base.torch.float32, device=device)
    return rows


def migrate_weights_by_name(source_payload, target_feature_names):
    source_names = tuple(source_payload.get("feature_names", []))
    if not source_names:
        source_mode = (
            source_payload.get("feature_mode")
            or source_payload.get("metadata", {}).get("feature_mode")
            or "better_history_full"
        )
        source_names = better.feature_names_for_mode(source_mode)
    source_index = {name: index for index, name in enumerate(source_names)}
    migrated = {}
    for position in POSITIONS:
        source_weights = list(source_payload["weights"][position])
        values = []
        for name in target_feature_names:
            index = source_index.get(name)
            values.append(float(source_weights[index]) if index is not None else 0.0)
        migrated[position] = values
    return migrated


class PreciseApproxQModel(better.BetterApproxQModel):
    def __init__(self, device="auto", weights=None, metadata=None,
                 feature_mode="precise_history_full"):
        self.device = base.resolve_device(device)
        self.feature_mode = feature_mode
        self.feature_names = feature_names_for_mode(feature_mode)
        self.torch_backend = base.use_torch_backend(self.device)
        self.weights = {}
        for position in POSITIONS:
            values = list(weights[position]) if weights and position in weights else None
            if self.torch_backend:
                if values is not None:
                    tensor = base.torch.as_tensor(values, dtype=base.torch.float32)
                else:
                    tensor = base.torch.zeros(len(self.feature_names), dtype=base.torch.float32)
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
                        position, len(self.weights[position]), len(self.feature_names)
                    )
                )
        self.metadata = metadata or {}
        self.num_updates = int(self.metadata.get("updates", 0) or 0)

    def save(self, path, metadata=None):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.metadata = metadata or self.metadata
        payload = {
            "version": 1,
            "algorithm": "precise_feature_based_approx_qlearning",
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
            base.pickle.dump(payload, f, base.pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path, device="auto", feature_mode=None, allow_migrate=True):
        with open(path, "rb") as f:
            payload = base.pickle.load(f)
        if not isinstance(payload, dict) or "weights" not in payload:
            raise ValueError("Invalid precise approximate Q checkpoint: {}".format(path))
        target_mode = feature_mode or payload.get("feature_mode") or "precise_history_full"
        expected_names = feature_names_for_mode(target_mode)
        feature_names = tuple(payload.get("feature_names", []))
        if feature_names == expected_names:
            return cls(
                device=device,
                weights=payload["weights"],
                metadata=payload.get("metadata", {}),
                feature_mode=target_mode,
            )
        if not allow_migrate:
            raise ValueError(
                "Feature mismatch: checkpoint has {} features, code expects {}".format(
                    len(feature_names), len(expected_names)
                )
            )
        migrated = migrate_weights_by_name(payload, expected_names)
        metadata = dict(payload.get("metadata", {}))
        metadata["warm_start_from"] = path
        metadata["warm_start_source_features"] = len(feature_names)
        metadata["warm_start_target_features"] = len(expected_names)
        return cls(
            device=device,
            weights=migrated,
            metadata=metadata,
            feature_mode=target_mode,
        )


class PreciseSelfPlayApproxQLearningAgent(better.BetterSelfPlayApproxQLearningAgent):
    def action_features(self, infoset):
        actions = base.prune_legal_actions(
            self.position, infoset, self.max_candidate_actions
        )
        features = features_for_actions(
            self.position, infoset, actions, self.model.device,
            self.model.feature_mode
        )
        return actions, features


def checkpoint_dir(flags):
    return os.path.join(flags.savedir, flags.name)


def checkpoint_path(flags, episodes):
    if flags.output:
        return flags.output
    return os.path.join(checkpoint_dir(flags), "{}.pkl".format(episodes))


def latest_checkpoint_path(flags):
    directory = checkpoint_dir(flags)
    if not os.path.isdir(directory):
        return None
    candidates = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if not filename.endswith(".pkl") or not os.path.isfile(path):
            continue
        episode = base.episode_from_checkpoint_path(path)
        if episode > 0:
            candidates.append((episode, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def training_metadata(flags, global_episode, epsilon, load_path, total_steps,
                      last_episode_steps, device, model, elapsed_sec=0.0,
                      completed_episodes=0, feature_diag_path=""):
    metadata = base.training_metadata(
        flags, global_episode, epsilon, load_path, total_steps,
        last_episode_steps, device, model, elapsed_sec,
        completed_episodes, feature_diag_path
    )
    metadata["algorithm"] = "precise_feature_based_approx_qlearning"
    metadata["feature_schema"] = "history_full_without_rough_pass_control_plus_precise_context"
    metadata["removed_rough_features"] = sorted(ROUGH_FEATURE_NAMES)
    metadata["precise_feature_groups"] = [
        "pass_context_by_last_player_next_player_hand_pressure",
        "control_preserve_by_role_turn_pressure",
        "straight_residual_low_singles",
        "serial_pair_residual_low_pairs",
    ]
    return metadata


def train(flags):
    flags.feature_mode = getattr(flags, "feature_mode", "precise_history_full") or "precise_history_full"
    flags.savedir = getattr(flags, "savedir", DEFAULT_PRECISE_APPROX_Q_DIR) or DEFAULT_PRECISE_APPROX_Q_DIR

    rng = base.random.Random(flags.seed)
    device = base.resolve_device(flags.device)
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
                "Cannot load precise approximate Q model from {}".format(load_path)
            )
        model = PreciseApproxQModel.load(
            load_path, device=device, feature_mode=flags.feature_mode, allow_migrate=True
        )
        completed_episodes = int(
            model.metadata.get("episodes", 0) or base.episode_from_checkpoint_path(load_path)
        )
        total_steps = int(model.metadata.get("total_steps", 0))
        if model.metadata.get("warm_start_from"):
            completed_episodes = 0
            total_steps = 0
            print(
                "warm-started precise approximate Q model from {} "
                "(source_features={}, target_features={})".format(
                    load_path,
                    model.metadata.get("warm_start_source_features"),
                    model.metadata.get("warm_start_target_features"),
                )
            )
        else:
            print("loaded precise approximate Q model from {} (episodes={}, updates={})".format(
                load_path, completed_episodes, model.num_updates
            ))
    else:
        model = PreciseApproxQModel(device=device, feature_mode=flags.feature_mode)
        completed_episodes = 0
        total_steps = 0

    recent_landlord_wins = base.deque(maxlen=max(1, flags.log_interval))
    recent_steps = base.deque(maxlen=max(1, flags.log_interval))
    recent_td = base.deque(maxlen=max(1, flags.log_interval * 4))
    feature_diagnostics = None
    feature_diag_path = ""
    feature_diag_interval = 0
    if getattr(flags, "feature_diag", False):
        feature_diag_path = flags.feature_diag_path or os.path.join(
            checkpoint_dir(flags), "feature_diagnostics.csv"
        )
        feature_diag_interval = int(
            getattr(flags, "feature_diag_interval", 0) or flags.log_interval or 1
        )
        flags._feature_diag_effective_interval = feature_diag_interval
        feature_diagnostics = base.FeatureDiagnostics(
            feature_diag_path,
            model.feature_names,
            getattr(flags, "feature_diag_topk", 0),
        )
        feature_diagnostics.window_start_episode = completed_episodes + 1

    agents = {
        position: PreciseSelfPlayApproxQLearningAgent(
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
            rng=base.random.Random(flags.seed + index + 1),
            td_log=recent_td,
            feature_diagnostics=feature_diagnostics,
        )
        for index, position in enumerate(POSITIONS)
    }
    env = base.GameEnv(agents)

    epsilon = flags.epsilon
    if load_path and "epsilon" in model.metadata and not model.metadata.get("warm_start_from"):
        epsilon = float(model.metadata["epsilon"])
    progress_enabled = bool(flags.progress_interval and flags.progress_interval > 0)
    start_time = base.time.time()
    last_steps = 0

    print("Precise Approx Q-learning device: {}".format(model.device))
    print("Feature mode: {}".format(model.feature_mode))
    print("Feature dim: {}".format(len(model.feature_names)))
    if feature_diagnostics is not None:
        print("Feature diagnostics: {} interval={} topk={}".format(
            feature_diag_path,
            feature_diag_interval,
            getattr(flags, "feature_diag_topk", 0),
        ))
    print("Reward objective: {} scale={} shaping={}".format(
        flags.objective, flags.reward_scale, flags.reward_shaping
    ))

    for episode in range(1, flags.episodes + 1):
        global_episode = completed_episodes + episode
        for agent in agents.values():
            agent.begin_episode()
            agent.epsilon = epsilon

        env.reset()
        env.card_play_init(base.generate_card_play_data(rng))

        steps = 0
        while not env.game_over and steps < flags.max_steps:
            env.step()
            steps += 1

        if env.game_over:
            winner = env.get_winner()
            bomb_num = env.get_bomb_num()
            for position, agent in agents.items():
                reward = base.reward_for_position(
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
        should_feature_diag = (
            feature_diagnostics is not None
            and feature_diag_interval > 0
            and (episode % feature_diag_interval == 0 or episode == flags.episodes)
        )
        should_progress = (
            progress_enabled
            and (
                episode == 1
                or episode == flags.episodes
                or episode % flags.progress_interval == 0
                or should_log
                or should_save
                or should_feature_diag
            )
        )

        if should_progress:
            base.print_progress_bar(
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
            elapsed = max(1e-6, base.time.time() - start_time)
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

        if should_feature_diag:
            win_rate = sum(recent_landlord_wins) / float(len(recent_landlord_wins))
            avg_steps = sum(recent_steps) / float(len(recent_steps))
            avg_td = sum(recent_td) / float(len(recent_td)) if recent_td else 0.0
            feature_diagnostics.flush(
                global_episode,
                model,
                epsilon,
                total_steps,
                max(1e-6, base.time.time() - start_time),
                win_rate,
                avg_steps,
                avg_td,
            )

        if should_save:
            if progress_enabled and not should_log:
                print()
            path = checkpoint_path(flags, global_episode)
            model.save(path, metadata=training_metadata(
                flags, global_episode, epsilon, load_path, total_steps,
                steps, device, model, max(1e-6, base.time.time() - start_time),
                completed_episodes, feature_diag_path
            ))
            print("saved precise approximate Q model to {}".format(path))

    if progress_enabled:
        print()

    final_episode = completed_episodes + flags.episodes
    final_path = checkpoint_path(flags, final_episode)
    if feature_diagnostics is not None and feature_diagnostics.has_updates():
        win_rate = sum(recent_landlord_wins) / float(len(recent_landlord_wins))
        avg_steps = sum(recent_steps) / float(len(recent_steps))
        avg_td = sum(recent_td) / float(len(recent_td)) if recent_td else 0.0
        feature_diagnostics.flush(
            final_episode,
            model,
            epsilon,
            total_steps,
            max(1e-6, base.time.time() - start_time),
            win_rate,
            avg_steps,
            avg_td,
        )
    model.save(final_path, metadata=training_metadata(
        flags, final_episode, epsilon, load_path, total_steps,
        last_steps, device, model, max(1e-6, base.time.time() - start_time),
        completed_episodes, feature_diag_path
    ))
    print("saved precise approximate Q model to {}".format(final_path))
    return model
