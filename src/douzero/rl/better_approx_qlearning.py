from __future__ import annotations

import os

from douzero.env import move_detector as md
from douzero.rl import approx_qlearning as base
from douzero.rl.qlearning import (
    CARD_TO_INDEX,
    POSITIONS,
    action_key,
    cards_to_counts,
    relation_to_last_player,
    teammate_position,
)


DEFAULT_BETTER_APPROX_Q_PATH = (
    "approx_qlearning_checkpoints/better_approxq/model.pkl"
)
DEFAULT_BETTER_APPROX_Q_DIR = "approx_qlearning_checkpoints/better_approxq"


def grouped_rank_names(prefix):
    return (
        "{}_low_3_to_7".format(prefix),
        "{}_mid_8_to_k".format(prefix),
        "{}_ace_2".format(prefix),
        "{}_joker".format(prefix),
        "{}_control_a_2_joker".format(prefix),
    )


ROLE_CONTEXT_FEATURE_NAMES = (
    "lead_after_teammate",
    "lead_after_enemy",
    "last_move_cards",
    "last_move_is_pass",
    "last_move_is_bomb",
    "last_move_min_rank",
    "last_move_max_rank",
    "last_move_avg_rank",
    "last_move_action_same_type",
    "last_move_action_raises_rank",
    "action_over_last_rank_gap",
    "action_uses_control_card",
    "action_uses_2",
    "action_uses_joker",
    "action_uses_low_card",
    "action_preserves_control",
    "control_cards_after_action",
    "control_delta_ratio",
    "my_control_advantage_over_unseen",
    "my_control_advantage_over_played",
    "unseen_low_cards_ratio",
    "unseen_mid_cards_ratio",
    "unseen_high_cards_ratio",
    "unseen_jokers_ratio",
    "played_low_cards_ratio",
    "played_mid_cards_ratio",
    "played_high_cards_ratio",
    "played_jokers_ratio",
    "my_low_cards_ratio",
    "my_mid_cards_ratio",
    "my_high_cards_ratio",
    "my_jokers_ratio",
    "landlord_enemy_min_left_le_1",
    "landlord_enemy_min_left_le_2",
    "landlord_enemy_min_left_le_3",
    "landlord_enemy_both_low",
    "landlord_should_press_enemy",
    "landlord_action_blocks_enemy_finish",
    "landlord_bomb_to_finish",
    "landlord_bomb_waste",
    "landlord_bottom_control_count",
    "landlord_control_advantage",
    "farmer_landlord_left_le_1",
    "farmer_landlord_left_le_2",
    "farmer_landlord_left_le_3",
    "farmer_teammate_left_le_1",
    "farmer_teammate_left_le_2",
    "farmer_teammate_left_le_3",
    "farmer_teammate_about_to_win",
    "farmer_landlord_about_to_win",
    "farmer_pass_to_teammate",
    "farmer_pass_to_landlord",
    "farmer_action_blocks_teammate",
    "farmer_action_blocks_landlord",
    "farmer_save_control_for_landlord",
    "farmer_control_advantage",
    "up_before_landlord_action_is_pass",
    "up_before_landlord_action_is_control",
    "up_should_block_landlord",
    "down_after_landlord_action_is_pass",
    "down_follow_landlord_pressure",
    "down_should_take_lead_from_landlord",
)


REMOVED_HISTORY_PREFIXES = (
    "my_hand_",
    "played_landlord_",
    "played_landlord_up_",
    "played_landlord_down_",
    "total_played_",
    "unseen_",
    "last_move_",
    "last_two_move_",
    "landlord_bottom_",
    "action_3",
    "action_4",
    "action_5",
    "action_6",
    "action_7",
    "action_8",
    "action_9",
    "action_10",
    "action_11",
    "action_12",
    "action_13",
    "action_14",
    "action_17",
    "action_20",
    "action_30",
)
REMOVED_HISTORY_NAMES = {
    "unseen_possible_bombs",
    "unseen_control_cards",
    "played_control_cards",
    "played_bomb_like_ranks",
}


def keep_base_feature(name):
    if name in REMOVED_HISTORY_NAMES:
        return False
    return not any(name.startswith(prefix) for prefix in REMOVED_HISTORY_PREFIXES)


BETTER_BASE_FEATURE_NAMES = tuple(
    name for name in base.HISTORY_FEATURE_NAMES if keep_base_feature(name)
)


BETTER_HISTORY_FEATURE_NAMES = (
    BETTER_BASE_FEATURE_NAMES
    + grouped_rank_names("unseen_group")
    + grouped_rank_names("played_group")
    + grouped_rank_names("my_group")
    + grouped_rank_names("action_group")
    + ROLE_CONTEXT_FEATURE_NAMES
)
BETTER_HISTORY_FULL_FEATURE_NAMES = (
    base.HISTORY_FEATURE_NAMES
    + grouped_rank_names("unseen_group")
    + grouped_rank_names("played_group")
    + grouped_rank_names("my_group")
    + grouped_rank_names("action_group")
    + ROLE_CONTEXT_FEATURE_NAMES
)

FEATURE_NAMES_BY_MODE = dict(base.FEATURE_NAMES_BY_MODE)
FEATURE_NAMES_BY_MODE["better_history"] = BETTER_HISTORY_FEATURE_NAMES
FEATURE_NAMES_BY_MODE["better_history_full"] = BETTER_HISTORY_FULL_FEATURE_NAMES


def feature_names_for_mode(feature_mode):
    if feature_mode not in FEATURE_NAMES_BY_MODE:
        raise ValueError("Unknown better feature_mode: {}".format(feature_mode))
    return FEATURE_NAMES_BY_MODE[feature_mode]


def validate_feature_definitions():
    for mode, names in FEATURE_NAMES_BY_MODE.items():
        if len(names) != len(set(names)):
            raise ValueError("Duplicate feature names in better mode {}".format(mode))


validate_feature_definitions()


def count_at(counts, card):
    return counts[CARD_TO_INDEX[card]]


def control_count(counts):
    return (
        count_at(counts, 14)
        + count_at(counts, 17)
        + count_at(counts, 20)
        + count_at(counts, 30)
    )


def high_count(counts):
    return count_at(counts, 14) + count_at(counts, 17)


def grouped_counts(counts):
    low = sum(counts[CARD_TO_INDEX[card]] for card in (3, 4, 5, 6, 7))
    mid = sum(counts[CARD_TO_INDEX[card]] for card in (8, 9, 10, 11, 12, 13))
    ace_two = high_count(counts)
    jokers = count_at(counts, 20) + count_at(counts, 30)
    control = control_count(counts)
    return [low / 20.0, mid / 24.0, ace_two / 8.0, jokers / 2.0, control / 10.0]


def move_summary(action):
    info = base.cached_action_info(action_key(action))
    return {
        "cards": info["len"] / 20.0,
        "is_pass": 1.0 if info["is_pass"] else 0.0,
        "is_bomb": 1.0 if info["is_bomb"] else 0.0,
        "min_rank": info["min_rank"],
        "max_rank": info["max_rank"],
        "avg_rank": info["avg_rank"],
        "type": info["type"],
    }


def better_extra_features(position, infoset, action):
    action = action_key(action)
    hand_cards = list(infoset.player_hand_cards or [])
    next_hand = base.remove_action_from_hand(hand_cards, action)
    hand_counts = cards_to_counts(hand_cards)
    next_counts = cards_to_counts(next_hand)
    action_counts = cards_to_counts(action)

    played_counts = base.played_counts_by_position(infoset)
    total_played = base.total_played_counts(played_counts)
    unseen_counts = base.unseen_counts_from_view(hand_counts, played_counts)
    bottom_counts = cards_to_counts(infoset.three_landlord_cards)

    left = base.num_cards_left(infoset)
    teammate = teammate_position(position)
    teammate_left = left.get(teammate, 0) if teammate else 0
    enemy_positions = base.enemy_positions(position)
    enemy_left = [left.get(enemy, 0) for enemy in enemy_positions]
    enemy_min_left = min(enemy_left) if enemy_left else 0
    last_relation = relation_to_last_player(position, infoset.last_pid)
    last_move = action_key(infoset.last_move)
    last_info = move_summary(last_move)
    action_info = base.cached_action_info(action)
    same_type = (
        last_move
        and action
        and last_info["type"] == action_info["type"]
    )
    raises_rank = (
        same_type
        and action_info["max_rank"] > last_info["max_rank"]
    )

    action_control = control_count(action_counts)
    hand_control = control_count(hand_counts)
    next_control = control_count(next_counts)
    unseen_control = control_count(unseen_counts)
    played_control = control_count(total_played)
    action_low = grouped_counts(action_counts)[0]

    is_landlord = position == "landlord"
    is_farmer = not is_landlord
    landlord_left = left.get("landlord", 0)
    farmer_enemy_low = is_farmer and landlord_left <= 2
    teammate_low = is_farmer and teammate_left <= 2
    action_is_pass = len(action) == 0
    action_uses_control = action_control > 0

    values = []
    values.extend(grouped_counts(unseen_counts))
    values.extend(grouped_counts(total_played))
    values.extend(grouped_counts(hand_counts))
    values.extend(grouped_counts(action_counts))
    values.extend([
        1.0 if last_relation == "teammate" and base.is_leading_round(infoset) else 0.0,
        1.0 if last_relation == "enemy" and base.is_leading_round(infoset) else 0.0,
        last_info["cards"],
        last_info["is_pass"],
        last_info["is_bomb"],
        last_info["min_rank"],
        last_info["max_rank"],
        last_info["avg_rank"],
        1.0 if same_type else 0.0,
        1.0 if raises_rank else 0.0,
        max(0.0, action_info["max_rank"] - last_info["max_rank"]) if action else 0.0,
        1.0 if action_uses_control else 0.0,
        1.0 if count_at(action_counts, 17) > 0 else 0.0,
        1.0 if count_at(action_counts, 20) + count_at(action_counts, 30) > 0 else 0.0,
        1.0 if action_low > 0 else 0.0,
        1.0 if next_control >= hand_control else 0.0,
        next_control / 10.0,
        (hand_control - next_control) / 10.0,
        (hand_control - unseen_control) / 10.0,
        (hand_control - played_control) / 10.0,
        grouped_counts(unseen_counts)[0],
        grouped_counts(unseen_counts)[1],
        grouped_counts(unseen_counts)[2],
        grouped_counts(unseen_counts)[3],
        grouped_counts(total_played)[0],
        grouped_counts(total_played)[1],
        grouped_counts(total_played)[2],
        grouped_counts(total_played)[3],
        grouped_counts(hand_counts)[0],
        grouped_counts(hand_counts)[1],
        grouped_counts(hand_counts)[2],
        grouped_counts(hand_counts)[3],
        1.0 if is_landlord and enemy_min_left <= 1 else 0.0,
        1.0 if is_landlord and enemy_min_left <= 2 else 0.0,
        1.0 if is_landlord and enemy_min_left <= 3 else 0.0,
        1.0 if is_landlord and len(enemy_left) == 2 and max(enemy_left) <= 3 else 0.0,
        1.0 if is_landlord and enemy_min_left <= 2 and action else 0.0,
        1.0 if is_landlord and enemy_min_left <= 2 and action and not action_is_pass else 0.0,
        1.0 if is_landlord and action_info["is_bomb"] and len(next_hand) <= 2 else 0.0,
        1.0 if is_landlord and action_info["is_bomb"] and len(next_hand) > 2 else 0.0,
        control_count(bottom_counts) / 6.0,
        (hand_control - unseen_control) / 10.0 if is_landlord else 0.0,
        1.0 if is_farmer and landlord_left <= 1 else 0.0,
        1.0 if is_farmer and landlord_left <= 2 else 0.0,
        1.0 if is_farmer and landlord_left <= 3 else 0.0,
        1.0 if is_farmer and teammate_left <= 1 else 0.0,
        1.0 if is_farmer and teammate_left <= 2 else 0.0,
        1.0 if is_farmer and teammate_left <= 3 else 0.0,
        1.0 if teammate_low else 0.0,
        1.0 if farmer_enemy_low else 0.0,
        1.0 if is_farmer and action_is_pass and last_relation == "teammate" else 0.0,
        1.0 if is_farmer and action_is_pass and last_relation == "enemy" else 0.0,
        1.0 if is_farmer and last_relation == "teammate" and action else 0.0,
        1.0 if is_farmer and last_relation == "enemy" and action else 0.0,
        1.0 if is_farmer and farmer_enemy_low and next_control >= hand_control else 0.0,
        (hand_control - unseen_control) / 10.0 if is_farmer else 0.0,
        1.0 if position == "landlord_up" and last_relation == "enemy" and action_is_pass else 0.0,
        1.0 if position == "landlord_up" and last_relation == "enemy" and action_uses_control else 0.0,
        1.0 if position == "landlord_up" and landlord_left <= 2 and action else 0.0,
        1.0 if position == "landlord_down" and last_relation == "enemy" and action_is_pass else 0.0,
        1.0 if position == "landlord_down" and landlord_left <= 2 else 0.0,
        1.0 if position == "landlord_down" and base.is_leading_round(infoset) and landlord_left <= 3 else 0.0,
    ])
    return values


def make_feature_vector(position, infoset, action, feature_mode="better_history"):
    if feature_mode not in ("better_history", "better_history_full"):
        return base.make_feature_vector(position, infoset, action, feature_mode)
    base_values = base.make_feature_vector(position, infoset, action, "history")
    if feature_mode == "better_history_full":
        values = list(base_values)
    else:
        values = [
            value
            for name, value in zip(base.HISTORY_FEATURE_NAMES, base_values)
            if keep_base_feature(name)
        ]
    values.extend(better_extra_features(position, infoset, action))
    base.validate_feature_vector(values, feature_names_for_mode(feature_mode), feature_mode)
    return values


def features_for_actions(position, infoset, actions, device, feature_mode="better_history"):
    rows = [
        make_feature_vector(position, infoset, action, feature_mode)
        for action in actions
    ]
    names = feature_names_for_mode(feature_mode)
    if base.use_torch_backend(device):
        if not rows:
            return base.torch.empty((0, len(names)), dtype=base.torch.float32, device=device)
        return base.torch.tensor(rows, dtype=base.torch.float32, device=device)
    return rows


class BetterApproxQModel(base.ApproxQModel):
    def __init__(self, device="auto", weights=None, metadata=None,
                 feature_mode="better_history"):
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
            "algorithm": "better_feature_based_approx_qlearning",
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
    def load(cls, path, device="auto"):
        with open(path, "rb") as f:
            payload = base.pickle.load(f)
        if not isinstance(payload, dict) or "weights" not in payload:
            raise ValueError("Invalid better approximate Q checkpoint: {}".format(path))
        feature_mode = payload.get("feature_mode") or payload.get("metadata", {}).get("feature_mode")
        if feature_mode is None:
            feature_mode = "better_history"
        expected_names = feature_names_for_mode(feature_mode)
        feature_names = tuple(payload.get("feature_names", []))
        if feature_names and feature_names != expected_names:
            raise ValueError(
                "Feature mismatch: checkpoint has {} features, code expects {}".format(
                    len(feature_names), len(expected_names)
                )
            )
        return cls(
            device=device,
            weights=payload["weights"],
            metadata=payload.get("metadata", {}),
            feature_mode=feature_mode,
        )


class BetterSelfPlayApproxQLearningAgent(base.SelfPlayApproxQLearningAgent):
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
    metadata["algorithm"] = "better_feature_based_approx_qlearning"
    metadata["role_specific_features"] = True
    metadata["better_feature_groups"] = [
        "grouped_rank_counts",
        "landlord_pressure",
        "farmer_cooperation",
        "pass_context",
        "control_card_advantage",
    ]
    return metadata


def train(flags):
    flags.feature_mode = getattr(flags, "feature_mode", "better_history") or "better_history"
    flags.savedir = getattr(flags, "savedir", DEFAULT_BETTER_APPROX_Q_DIR) or DEFAULT_BETTER_APPROX_Q_DIR

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
                "Cannot resume: better approximate Q model not found at {}".format(load_path)
            )
        model = BetterApproxQModel.load(load_path, device=device)
        completed_episodes = int(
            model.metadata.get("episodes", 0) or base.episode_from_checkpoint_path(load_path)
        )
        total_steps = int(model.metadata.get("total_steps", 0))
        print("loaded better approximate Q model from {} (episodes={}, updates={})".format(
            load_path, completed_episodes, model.num_updates
        ))
    else:
        model = BetterApproxQModel(device=device, feature_mode=flags.feature_mode)
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
        position: BetterSelfPlayApproxQLearningAgent(
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
    if load_path and "epsilon" in model.metadata:
        epsilon = float(model.metadata["epsilon"])
    progress_enabled = bool(flags.progress_interval and flags.progress_interval > 0)
    start_time = base.time.time()
    last_steps = 0

    print("Better Approx Q-learning device: {}".format(model.device))
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
            print("saved better approximate Q model to {}".format(path))

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
    print("saved better approximate Q model to {}".format(final_path))
    return model
