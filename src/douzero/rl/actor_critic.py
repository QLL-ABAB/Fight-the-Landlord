import math
import os
import pickle
import random
import time
from collections import deque

from douzero.env import move_detector as md
from douzero.env.game import GameEnv
try:
    from douzero.rl.approx_qlearning import (
        ACTION_TYPE_VALUES,
        FULL_DECK_COUNTS,
        cached_action_info,
        feature_names_for_mode,
        features_for_actions,
        list_dot,
        normalized_counts,
        played_counts_by_position,
        prune_legal_actions,
        total_played_counts,
        unseen_counts_from_view,
    )
except ModuleNotFoundError as exc:
    if exc.name != "douzero.rl.approx_qlearning":
        raise
    from douzero.rl.approx_qlearning_fasle import (
        ACTION_TYPE_VALUES,
        FULL_DECK_COUNTS,
        cached_action_info,
        feature_names_for_mode,
        features_for_actions,
        list_dot,
        normalized_counts,
        played_counts_by_position,
        prune_legal_actions,
        total_played_counts,
        unseen_counts_from_view,
    )
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


DEFAULT_ACTOR_CRITIC_PATH = "actor_critic_checkpoints/actor_critic/model.pkl"


def checkpoint_dir(flags):
    return os.path.join(flags.savedir, flags.name)


def checkpoint_path(flags, episodes):
    if flags.output:
        return flags.output
    return os.path.join(checkpoint_dir(flags), "{}.pkl".format(episodes))


def episode_from_checkpoint_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem) if stem.isdigit() else 0


def latest_checkpoint_path(flags):
    directory = checkpoint_dir(flags)
    if not os.path.isdir(directory):
        return None
    candidates = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if filename.endswith(".pkl") and os.path.isfile(path):
            episode = episode_from_checkpoint_path(path)
            if episode > 0:
                candidates.append((episode, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def softmax(logits):
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(max(-60.0, min(60.0, value - max_logit)))
            for value in logits]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(logits) for _ in logits]
    return [value / total for value in exps]


def weighted_average(features, weights):
    if not features:
        return []
    result = [0.0 for _ in features[0]]
    for weight, feature in zip(weights, features):
        for index, value in enumerate(feature):
            result[index] += weight * value
    return result


def rank_feature_names(prefix):
    return tuple("{}_{}".format(prefix, card) for card in CARD_RANKS)


STATE_FEATURE_NAMES = (
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
    "legal_actions",
    "legal_nonpass_actions",
    "legal_pass_available",
    "legal_bomb_actions",
    "legal_finish_actions",
    "last_move_cards",
    "last_move_min_rank",
    "last_move_max_rank",
    "last_move_avg_rank",
) + tuple("last_move_type_{}".format(t) for t in ACTION_TYPE_VALUES) + (
    "unseen_possible_bombs",
    "unseen_control_cards",
    "played_control_cards",
    "played_bomb_like_ranks",
) + rank_feature_names("my_hand") + rank_feature_names("played_landlord") + (
    rank_feature_names("played_landlord_up")
    + rank_feature_names("played_landlord_down")
    + rank_feature_names("total_played")
    + rank_feature_names("unseen")
    + rank_feature_names("last_move")
    + rank_feature_names("landlord_bottom")
)


def enemy_positions(position):
    if position == "landlord":
        return ("landlord_up", "landlord_down")
    return ("landlord",)


def num_cards_left(infoset):
    num_left = infoset.num_cards_left_dict or {}
    return {position: num_left.get(position, 0) for position in POSITIONS}


def hand_stats_from_counts(counts):
    singles = sum(1 for count in counts if count == 1)
    pairs = sum(1 for count in counts if count == 2)
    triples = sum(1 for count in counts if count == 3)
    bombs = sum(1 for count in counts if count == 4)
    control = (
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
        "control": control,
    }


def state_features(position, infoset):
    hand_cards = list(infoset.player_hand_cards or [])
    hand_counts = cards_to_counts(hand_cards)
    hand_stats = hand_stats_from_counts(hand_counts)
    left = num_cards_left(infoset)
    teammate = teammate_position(position)
    enemies = enemy_positions(position)
    enemy_left = [left.get(enemy, 0) for enemy in enemies]
    teammate_left = left.get(teammate, 0) if teammate else 0
    last_relation = relation_to_last_player(position, infoset.last_pid)
    last_move = action_key(infoset.last_move)
    last_info = cached_action_info(last_move)
    legal_actions = list(infoset.legal_actions or [])
    legal_nonpass = [action for action in legal_actions if action]
    legal_bombs = [action for action in legal_actions if action_key(action) in BOMB_ACTIONS]
    legal_finish = [
        action for action in legal_actions
        if action and len(action) == len(hand_cards)
    ]

    played_counts = played_counts_by_position(infoset)
    total_played = total_played_counts(played_counts)
    unseen_counts = unseen_counts_from_view(hand_counts, played_counts)
    landlord_bottom_counts = cards_to_counts(infoset.three_landlord_cards)

    values = [
        1.0,
        1.0 if position == "landlord" else 0.0,
        1.0 if position == "landlord_up" else 0.0,
        1.0 if position == "landlord_down" else 0.0,
        0.0 if position == "landlord" else 1.0,
        len(hand_cards) / 20.0,
        left.get("landlord", 0) / 20.0,
        left.get("landlord_up", 0) / 17.0,
        left.get("landlord_down", 0) / 17.0,
        teammate_left / 17.0,
        (min(enemy_left) if enemy_left else 0) / 20.0,
        (max(enemy_left) if enemy_left else 0) / 20.0,
        1.0 if not last_move else 0.0,
        1.0 if last_relation == "self" else 0.0,
        1.0 if last_relation == "teammate" else 0.0,
        1.0 if last_relation == "enemy" else 0.0,
        min(float(infoset.bomb_num or 0), 10.0) / 10.0,
        hand_stats["singles"] / 15.0,
        hand_stats["pairs"] / 15.0,
        hand_stats["triples"] / 15.0,
        hand_stats["bombs"] / 15.0,
        hand_stats["control"] / 10.0,
        hand_badness(hand_cards),
        min(len(legal_actions), 200) / 200.0,
        min(len(legal_nonpass), 200) / 200.0,
        1.0 if [] in legal_actions else 0.0,
        min(len(legal_bombs), 16) / 16.0,
        min(len(legal_finish), 16) / 16.0,
        last_info["len"] / 20.0,
        last_info["min_rank"],
        last_info["max_rank"],
        last_info["avg_rank"],
    ]
    values.extend(1.0 if last_info["type"] == t else 0.0
                  for t in ACTION_TYPE_VALUES)
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
    values.extend(normalized_counts(hand_counts))
    for played_position in POSITIONS:
        values.extend(normalized_counts(played_counts[played_position]))
    values.extend(normalized_counts(total_played))
    values.extend(normalized_counts(unseen_counts))
    values.extend(normalized_counts(cards_to_counts(last_move)))
    values.extend(normalized_counts(landlord_bottom_counts))

    if len(values) != len(STATE_FEATURE_NAMES):
        raise ValueError(
            "State feature length mismatch: {} != {}".format(
                len(values), len(STATE_FEATURE_NAMES)
            )
        )
    return values


def clipped(values, max_norm):
    if not max_norm or max_norm <= 0:
        return values
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= max_norm:
        return values
    scale = max_norm / max(1e-12, norm)
    return [value * scale for value in values]


class LinearActorCriticModel(object):
    def __init__(self, actor_weights=None, critic_weights=None, metadata=None,
                 feature_mode="history"):
        self.feature_mode = feature_mode
        self.actor_feature_names = feature_names_for_mode(feature_mode)
        self.critic_feature_names = STATE_FEATURE_NAMES
        self.actor_weights = {}
        self.critic_weights = {}
        for position in POSITIONS:
            actor_values = (
                list(actor_weights[position])
                if actor_weights and position in actor_weights else
                None
            )
            critic_values = (
                list(critic_weights[position])
                if critic_weights and position in critic_weights else
                None
            )
            self.actor_weights[position] = (
                [float(value) for value in actor_values]
                if actor_values is not None else
                [0.0 for _ in self.actor_feature_names]
            )
            self.critic_weights[position] = (
                [float(value) for value in critic_values]
                if critic_values is not None else
                [0.0 for _ in self.critic_feature_names]
            )
            if len(self.actor_weights[position]) != len(self.actor_feature_names):
                raise ValueError("Actor weight length mismatch for {}".format(position))
            if len(self.critic_weights[position]) != len(self.critic_feature_names):
                raise ValueError("Critic weight length mismatch for {}".format(position))
        self.metadata = metadata or {}
        self.num_actor_updates = int(self.metadata.get("actor_updates", 0) or 0)
        self.num_critic_updates = int(self.metadata.get("critic_updates", 0) or 0)

    def logits(self, position, action_features, temperature=1.0):
        if not action_features:
            return []
        scale = 1.0 / max(1e-6, float(temperature))
        weights = self.actor_weights[position]
        return [scale * list_dot(weights, feature) for feature in action_features]

    def action_probs(self, position, action_features, temperature=1.0):
        return softmax(self.logits(position, action_features, temperature))

    def select_action(self, position, actions, action_features, temperature=1.0,
                      rng=None):
        if not actions:
            return [], [], []
        rng = rng or random
        probs = self.action_probs(position, action_features, temperature)
        threshold = rng.random()
        cumulative = 0.0
        index = len(actions) - 1
        for idx, prob in enumerate(probs):
            cumulative += prob
            if threshold <= cumulative:
                index = idx
                break
        return list(actions[index]), list(action_features[index]), probs

    def greedy_action(self, position, actions, action_features):
        if not actions:
            return []
        logits = self.logits(position, action_features, temperature=1.0)
        best = max(logits)
        for index, value in enumerate(logits):
            if value == best:
                return list(actions[index])
        return list(actions[0])

    def value(self, position, feature):
        return list_dot(self.critic_weights[position], feature)

    def update_actor(self, position, feature, expected_feature, advantage,
                     learning_rate, l2=0.0, grad_clip=0.0):
        weights = self.actor_weights[position]
        gradient = [
            float(advantage) * (float(value) - float(expected))
            for value, expected in zip(feature, expected_feature)
        ]
        if l2 and l2 > 0:
            for index, weight in enumerate(weights):
                gradient[index] -= l2 * weight
        gradient = clipped(gradient, grad_clip)
        for index, value in enumerate(gradient):
            weights[index] += learning_rate * value
        self.num_actor_updates += 1

    def update_critic(self, position, feature, target, learning_rate,
                      l2=0.0, grad_clip=0.0):
        weights = self.critic_weights[position]
        prediction = self.value(position, feature)
        error = float(target) - prediction
        gradient = [error * float(value) for value in feature]
        if l2 and l2 > 0:
            for index, weight in enumerate(weights):
                gradient[index] -= l2 * weight
        gradient = clipped(gradient, grad_clip)
        for index, value in enumerate(gradient):
            weights[index] += learning_rate * value
        self.num_critic_updates += 1
        return error

    def save(self, path, metadata=None):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.metadata = metadata or self.metadata
        payload = {
            "version": 1,
            "algorithm": "linear_actor_critic",
            "feature_mode": self.feature_mode,
            "actor_feature_names": list(self.actor_feature_names),
            "critic_feature_names": list(self.critic_feature_names),
            "actor_weights": {
                position: list(weight)
                for position, weight in self.actor_weights.items()
            },
            "critic_weights": {
                position: list(weight)
                for position, weight in self.critic_weights.items()
            },
            "metadata": self.metadata,
        }
        tmp_path = "{}.tmp".format(path)
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "actor_weights" not in payload:
            raise ValueError("Invalid actor-critic checkpoint: {}".format(path))
        feature_mode = payload.get("feature_mode")
        if feature_mode is None:
            feature_mode = payload.get("metadata", {}).get("feature_mode", "history")
        actor_names = tuple(payload.get("actor_feature_names", []))
        expected_actor_names = feature_names_for_mode(feature_mode)
        if actor_names and actor_names != expected_actor_names:
            raise ValueError("Actor feature mismatch in {}".format(path))
        critic_names = tuple(payload.get("critic_feature_names", []))
        if critic_names and critic_names != STATE_FEATURE_NAMES:
            raise ValueError("Critic feature mismatch in {}".format(path))
        return cls(
            actor_weights=payload["actor_weights"],
            critic_weights=payload["critic_weights"],
            metadata=payload.get("metadata", {}),
            feature_mode=feature_mode,
        )


class SelfPlayActorCriticAgent(object):
    def __init__(self, position, model, actor_learning_rate,
                 critic_learning_rate, gamma, temperature,
                 max_candidate_actions=64, reward_scale=1.0,
                 reward_shaping=False, actor_l2=0.0, critic_l2=0.0,
                 actor_grad_clip=5.0, critic_grad_clip=5.0,
                 advantage_clip=10.0, rng=None, td_log=None):
        self.name = "SelfPlayActorCritic"
        self.position = position
        self.model = model
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.temperature = temperature
        self.max_candidate_actions = max_candidate_actions
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.actor_l2 = actor_l2
        self.critic_l2 = critic_l2
        self.actor_grad_clip = actor_grad_clip
        self.critic_grad_clip = critic_grad_clip
        self.advantage_clip = advantage_clip
        self.rng = rng or random.Random()
        self.td_log = td_log
        self.trajectory = []
        self.last_episode_return = 0.0

    def begin_episode(self):
        self.trajectory = []
        self.last_episode_return = 0.0

    def action_features(self, infoset):
        actions = prune_legal_actions(
            self.position,
            infoset,
            self.max_candidate_actions,
        )
        features = features_for_actions(
            self.position,
            infoset,
            actions,
            "python",
            self.model.feature_mode,
        )
        return actions, features

    def act(self, infoset):
        actions, action_features = self.action_features(infoset)
        action, action_feature, probs = self.model.select_action(
            self.position,
            actions,
            action_features,
            self.temperature,
            self.rng,
        )
        expected_action_feature = weighted_average(action_features, probs)
        state_feature = state_features(self.position, infoset)
        shaped_reward = 0.0
        if self.reward_shaping:
            shaped_reward = shaped_reward_for_action(
                self.position,
                infoset,
                action,
                self.reward_scale,
            )
        policy_scale = 1.0 / max(1e-6, float(self.temperature))
        self.trajectory.append(
            (
                state_feature,
                action_feature,
                expected_action_feature,
                shaped_reward,
                policy_scale,
            )
        )
        return action

    def finish_episode(self, terminal_reward):
        returns = []
        running_return = float(terminal_reward)
        for _, _, _, shaped_reward, _ in reversed(self.trajectory):
            running_return = float(shaped_reward) + running_return
            returns.append(running_return)
            running_return *= self.gamma
        returns.reverse()

        if returns:
            self.last_episode_return = returns[0]

        for step, reward_to_go in zip(self.trajectory, returns):
            state_feature, action_feature, expected_action_feature, _, policy_scale = step
            value_before = self.model.value(self.position, state_feature)
            advantage = reward_to_go - value_before
            if self.advantage_clip and self.advantage_clip > 0:
                advantage = max(-self.advantage_clip,
                                min(self.advantage_clip, advantage))
            td_error = self.model.update_critic(
                self.position,
                state_feature,
                reward_to_go,
                self.critic_learning_rate,
                l2=self.critic_l2,
                grad_clip=self.critic_grad_clip,
            )
            if self.td_log is not None:
                self.td_log.append(abs(float(td_error)))
            self.model.update_actor(
                self.position,
                action_feature,
                expected_action_feature,
                advantage * policy_scale,
                self.actor_learning_rate,
                l2=self.actor_l2,
                grad_clip=self.actor_grad_clip,
            )
        self.trajectory = []


def training_metadata(flags, global_episode, temperature, load_path,
                      total_steps, last_episode_steps, model):
    return {
        "algorithm": "linear_actor_critic",
        "name": flags.name,
        "episodes": global_episode,
        "total_steps": total_steps,
        "last_episode_steps": last_episode_steps,
        "objective": flags.objective,
        "actor_learning_rate": flags.actor_learning_rate,
        "critic_learning_rate": flags.critic_learning_rate,
        "gamma": flags.gamma,
        "temperature": temperature,
        "reward_scale": flags.reward_scale,
        "reward_shaping": flags.reward_shaping,
        "max_candidate_actions": flags.max_candidate_actions,
        "feature_mode": model.feature_mode,
        "actor_l2": flags.actor_l2,
        "critic_l2": flags.critic_l2,
        "actor_grad_clip": flags.actor_grad_clip,
        "critic_grad_clip": flags.critic_grad_clip,
        "advantage_clip": flags.advantage_clip,
        "savedir": flags.savedir,
        "resumed_from": load_path,
        "actor_feature_dim": len(model.actor_feature_names),
        "critic_feature_dim": len(model.critic_feature_names),
        "actor_feature_names": list(model.actor_feature_names),
        "critic_feature_names": list(model.critic_feature_names),
        "actor_updates": model.num_actor_updates,
        "critic_updates": model.num_critic_updates,
    }


def print_progress_bar(episode, total_episodes, completed_episodes, model,
                       temperature, recent_steps, recent_landlord_wins,
                       recent_returns, recent_td, start_time, total_steps):
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
    avg_return = (
        sum(recent_returns) / float(len(recent_returns))
        if recent_returns else 0.0
    )
    avg_td = sum(recent_td) / float(len(recent_td)) if recent_td else 0.0
    global_episode = completed_episodes + episode
    line = (
        "\r[{}] {:6.2f}% episode={}/{} global={} actor_updates={} "
        "critic_updates={} temperature={:.4f} landlord_wp={:.3f} "
        "avg_return={:.3f} avg_abs_td={:.4f} avg_steps={:.1f} "
        "speed={:.2f}eps/s steps={} eta={}"
    ).format(
        bar,
        progress * 100.0,
        episode,
        total_episodes,
        global_episode,
        model.num_actor_updates,
        model.num_critic_updates,
        temperature,
        landlord_wp,
        avg_return,
        avg_td,
        avg_steps,
        speed,
        total_steps,
        format_duration(remaining),
    )
    print(line + " " * 8, end="", flush=True)


def train(flags):
    rng = random.Random(flags.seed)
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
                "Cannot resume: actor-critic model not found at {}".format(load_path)
            )
        model = LinearActorCriticModel.load(load_path)
        completed_episodes = int(
            model.metadata.get("episodes", 0)
            or episode_from_checkpoint_path(load_path)
        )
        total_steps = int(model.metadata.get("total_steps", 0))
        print(
            "loaded actor-critic model from {} (episodes={}, actor_updates={}, "
            "critic_updates={})".format(
                load_path,
                completed_episodes,
                model.num_actor_updates,
                model.num_critic_updates,
            )
        )
    else:
        model = LinearActorCriticModel(feature_mode=flags.feature_mode)
        completed_episodes = 0
        total_steps = 0

    temperature = flags.temperature
    if load_path and "temperature" in model.metadata:
        temperature = float(model.metadata["temperature"])

    recent_landlord_wins = deque(maxlen=max(1, flags.log_interval))
    recent_steps = deque(maxlen=max(1, flags.log_interval))
    recent_returns = deque(maxlen=max(1, flags.log_interval * 3))
    recent_td = deque(maxlen=max(1, flags.log_interval * 4))
    agents = {
        position: SelfPlayActorCriticAgent(
            position=position,
            model=model,
            actor_learning_rate=flags.actor_learning_rate,
            critic_learning_rate=flags.critic_learning_rate,
            gamma=flags.gamma,
            temperature=temperature,
            max_candidate_actions=flags.max_candidate_actions,
            reward_scale=flags.reward_scale,
            reward_shaping=flags.reward_shaping,
            actor_l2=flags.actor_l2,
            critic_l2=flags.critic_l2,
            actor_grad_clip=flags.actor_grad_clip,
            critic_grad_clip=flags.critic_grad_clip,
            advantage_clip=flags.advantage_clip,
            rng=random.Random(flags.seed + index + 1),
            td_log=recent_td,
        )
        for index, position in enumerate(POSITIONS)
    }
    env = GameEnv(agents)
    progress_enabled = bool(flags.progress_interval and flags.progress_interval > 0)
    start_time = time.time()
    last_steps = 0

    print("Actor-Critic feature mode: {}".format(model.feature_mode))
    print("Actor feature dim: {}".format(len(model.actor_feature_names)))
    print("Critic state feature dim: {}".format(len(model.critic_feature_names)))
    print("Reward objective: {} scale={} shaping={}".format(
        flags.objective, flags.reward_scale, flags.reward_shaping
    ))

    for episode in range(1, flags.episodes + 1):
        global_episode = completed_episodes + episode
        for agent in agents.values():
            agent.begin_episode()
            agent.temperature = temperature

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
                    position,
                    winner,
                    bomb_num,
                    flags.objective,
                    flags.reward_scale,
                )
                agent.finish_episode(reward)
                recent_returns.append(agent.last_episode_return)
            recent_landlord_wins.append(1 if winner == "landlord" else 0)
        else:
            for agent in agents.values():
                agent.finish_episode(0.0)
                recent_returns.append(agent.last_episode_return)
            recent_landlord_wins.append(0)

        recent_steps.append(steps)
        last_steps = steps
        total_steps += steps
        temperature = max(flags.min_temperature, temperature * flags.temperature_decay)

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
                episode,
                flags.episodes,
                completed_episodes,
                model,
                temperature,
                recent_steps,
                recent_landlord_wins,
                recent_returns,
                recent_td,
                start_time,
                total_steps,
            )

        if should_log:
            if progress_enabled:
                print()
            win_rate = sum(recent_landlord_wins) / float(len(recent_landlord_wins))
            avg_steps = sum(recent_steps) / float(len(recent_steps))
            avg_return = (
                sum(recent_returns) / float(len(recent_returns))
                if recent_returns else 0.0
            )
            avg_td = sum(recent_td) / float(len(recent_td)) if recent_td else 0.0
            print(
                "episode={} actor_updates={} critic_updates={} "
                "temperature={:.4f} landlord_wp={:.3f} avg_steps={:.1f} "
                "avg_return={:.3f} avg_abs_td={:.4f}".format(
                    global_episode,
                    model.num_actor_updates,
                    model.num_critic_updates,
                    temperature,
                    win_rate,
                    avg_steps,
                    avg_return,
                    avg_td,
                )
            )

        if should_save:
            if progress_enabled and not should_log:
                print()
            path = checkpoint_path(flags, global_episode)
            model.save(
                path,
                metadata=training_metadata(
                    flags,
                    global_episode,
                    temperature,
                    load_path,
                    total_steps,
                    steps,
                    model,
                ),
            )
            print("saved actor-critic model to {}".format(path))

    if progress_enabled:
        print()

    final_episode = completed_episodes + flags.episodes
    final_path = checkpoint_path(flags, final_episode)
    model.save(
        final_path,
        metadata=training_metadata(
            flags,
            final_episode,
            temperature,
            load_path,
            total_steps,
            last_steps,
            model,
        ),
    )
    print("saved actor-critic model to {}".format(final_path))
    return model
