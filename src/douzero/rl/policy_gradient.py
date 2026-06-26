import math
import os
import pickle
import random
import time
from collections import deque

from douzero.env.game import GameEnv
try:
    from douzero.rl.approx_qlearning import (
        feature_names_for_mode,
        features_for_actions,
        list_dot,
        prune_legal_actions,
    )
except ModuleNotFoundError as exc:
    if exc.name != "douzero.rl.approx_qlearning":
        raise
    from douzero.rl.approx_qlearning_fasle import (
        feature_names_for_mode,
        features_for_actions,
        list_dot,
        prune_legal_actions,
    )
from douzero.rl.qlearning import (
    POSITIONS,
    format_duration,
    generate_card_play_data,
    reward_for_position,
    shaped_reward_for_action,
)


DEFAULT_POLICY_PATH = "policy_gradient_checkpoints/policy_gradient/model.pkl"


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
        if not filename.endswith(".pkl") or not os.path.isfile(path):
            continue
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
    exps = [
        math.exp(max(-60.0, min(60.0, value - max_logit)))
        for value in logits
    ]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(logits) for _ in logits]
    return [value / total for value in exps]


def weighted_average(features, weights):
    if not features:
        return []
    dim = len(features[0])
    result = [0.0 for _ in range(dim)]
    for weight, feature in zip(weights, features):
        for index, value in enumerate(feature):
            result[index] += weight * value
    return result


class LinearPolicyGradientModel(object):
    """Linear softmax policy over encoded legal state-action features."""

    def __init__(self, weights=None, metadata=None, feature_mode="history"):
        self.feature_mode = feature_mode
        self.feature_names = feature_names_for_mode(feature_mode)
        self.weights = {}
        for position in POSITIONS:
            values = list(weights[position]) if weights and position in weights else None
            self.weights[position] = (
                [float(value) for value in values]
                if values is not None
                else [0.0 for _ in self.feature_names]
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
        baselines = self.metadata.get("baselines", {})
        self.baselines = {
            position: float(baselines.get(position, 0.0))
            for position in POSITIONS
        }

    def logits(self, position, features, temperature=1.0):
        if not features:
            return []
        scale = 1.0 / max(1e-6, float(temperature))
        weights = self.weights[position]
        return [scale * list_dot(weights, feature) for feature in features]

    def action_probs(self, position, features, temperature=1.0):
        return softmax(self.logits(position, features, temperature))

    def select_action(self, position, actions, features, temperature=1.0, rng=None):
        if not actions:
            return [], [], []
        rng = rng or random
        probs = self.action_probs(position, features, temperature)
        threshold = rng.random()
        cumulative = 0.0
        index = len(actions) - 1
        for idx, prob in enumerate(probs):
            cumulative += prob
            if threshold <= cumulative:
                index = idx
                break
        return list(actions[index]), list(features[index]), probs

    def greedy_action(self, position, actions, features):
        if not actions:
            return []
        logits = self.logits(position, features, temperature=1.0)
        best = max(logits)
        for index, value in enumerate(logits):
            if value == best:
                return list(actions[index])
        return list(actions[0])

    def update(self, position, feature, expected_feature, advantage,
               learning_rate, l2=0.0, grad_clip=0.0):
        weights = self.weights[position]
        gradient = [
            float(advantage) * (float(value) - float(expected))
            for value, expected in zip(feature, expected_feature)
        ]

        if l2 and l2 > 0:
            for index, weight in enumerate(weights):
                gradient[index] -= l2 * weight

        if grad_clip and grad_clip > 0:
            norm = math.sqrt(sum(value * value for value in gradient))
            if norm > grad_clip:
                scale = grad_clip / max(1e-12, norm)
                gradient = [value * scale for value in gradient]

        for index, value in enumerate(gradient):
            weights[index] += learning_rate * value
        self.num_updates += 1

    def save(self, path, metadata=None):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.metadata = metadata or self.metadata
        payload = {
            "version": 1,
            "algorithm": "linear_policy_gradient",
            "feature_mode": self.feature_mode,
            "feature_names": list(self.feature_names),
            "weights": {
                position: list(weight)
                for position, weight in self.weights.items()
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
        if not isinstance(payload, dict) or "weights" not in payload:
            raise ValueError("Invalid policy-gradient checkpoint: {}".format(path))
        feature_mode = payload.get("feature_mode")
        if feature_mode is None:
            feature_mode = payload.get("metadata", {}).get("feature_mode", "history")
        saved_names = tuple(payload.get("feature_names", []))
        expected_names = feature_names_for_mode(feature_mode)
        if saved_names and saved_names != expected_names:
            raise ValueError(
                "Feature mismatch: checkpoint has {} features, code expects {}".format(
                    len(saved_names),
                    len(expected_names),
                )
            )
        return cls(
            weights=payload["weights"],
            metadata=payload.get("metadata", {}),
            feature_mode=feature_mode,
        )


class SelfPlayPolicyGradientAgent(object):
    def __init__(self, position, model, learning_rate, gamma, temperature,
                 max_candidate_actions=64, reward_scale=1.0,
                 reward_shaping=False, baseline_decay=0.99,
                 l2=0.0, grad_clip=5.0, rng=None):
        self.name = "SelfPlayPolicyGradient"
        self.position = position
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.temperature = temperature
        self.max_candidate_actions = max_candidate_actions
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.baseline_decay = baseline_decay
        self.l2 = l2
        self.grad_clip = grad_clip
        self.rng = rng or random.Random()
        self.trajectory = []
        self.last_episode_return = 0.0

    def begin_episode(self):
        self.trajectory = []
        self.last_episode_return = 0.0

    def masked_action_features(self, infoset):
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
        actions, features = self.masked_action_features(infoset)
        action, feature, probs = self.model.select_action(
            self.position,
            actions,
            features,
            self.temperature,
            self.rng,
        )
        expected_feature = weighted_average(features, probs)
        shaped_reward = 0.0
        if self.reward_shaping:
            shaped_reward = shaped_reward_for_action(
                self.position,
                infoset,
                action,
                self.reward_scale,
            )
        policy_scale = 1.0 / max(1e-6, float(self.temperature))
        self.trajectory.append((feature, expected_feature, shaped_reward, policy_scale))
        return action

    def finish_episode(self, terminal_reward):
        returns = []
        running_return = float(terminal_reward)
        for _, _, shaped_reward, _ in reversed(self.trajectory):
            running_return = float(shaped_reward) + running_return
            returns.append(running_return)
            running_return *= self.gamma
        returns.reverse()

        if returns:
            self.last_episode_return = returns[0]

        for (feature, expected_feature, _, policy_scale), reward_to_go in zip(
            self.trajectory,
            returns,
        ):
            baseline = self.model.baselines[self.position]
            advantage = (reward_to_go - baseline) * policy_scale
            self.model.baselines[self.position] = (
                self.baseline_decay * baseline
                + (1.0 - self.baseline_decay) * reward_to_go
            )
            self.model.update(
                self.position,
                feature,
                expected_feature,
                advantage,
                self.learning_rate,
                l2=self.l2,
                grad_clip=self.grad_clip,
            )
        self.trajectory = []


def training_metadata(flags, global_episode, temperature, load_path,
                      total_steps, last_episode_steps, model):
    return {
        "algorithm": "linear_policy_gradient",
        "name": flags.name,
        "episodes": global_episode,
        "total_steps": total_steps,
        "last_episode_steps": last_episode_steps,
        "objective": flags.objective,
        "learning_rate": flags.learning_rate,
        "gamma": flags.gamma,
        "temperature": temperature,
        "reward_scale": flags.reward_scale,
        "reward_shaping": flags.reward_shaping,
        "baseline_decay": flags.baseline_decay,
        "max_candidate_actions": flags.max_candidate_actions,
        "feature_mode": model.feature_mode,
        "l2": flags.l2,
        "grad_clip": flags.grad_clip,
        "savedir": flags.savedir,
        "resumed_from": load_path,
        "feature_dim": len(model.feature_names),
        "feature_names": list(model.feature_names),
        "updates": model.num_updates,
        "baselines": dict(model.baselines),
    }


def print_progress_bar(episode, total_episodes, completed_episodes, model,
                       temperature, recent_steps, recent_landlord_wins,
                       recent_returns, start_time, total_steps):
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
        if recent_landlord_wins
        else 0.0
    )
    avg_return = (
        sum(recent_returns) / float(len(recent_returns))
        if recent_returns
        else 0.0
    )
    global_episode = completed_episodes + episode
    line = (
        "\r[{}] {:6.2f}% episode={}/{} global={} updates={} "
        "temperature={:.4f} landlord_wp={:.3f} avg_return={:.3f} "
        "avg_steps={:.1f} speed={:.2f}eps/s steps={} eta={}"
    ).format(
        bar,
        progress * 100.0,
        episode,
        total_episodes,
        global_episode,
        model.num_updates,
        temperature,
        landlord_wp,
        avg_return,
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
                "Cannot resume: policy-gradient model not found at {}".format(
                    load_path
                )
            )
        model = LinearPolicyGradientModel.load(load_path)
        completed_episodes = int(
            model.metadata.get("episodes", 0)
            or episode_from_checkpoint_path(load_path)
        )
        total_steps = int(model.metadata.get("total_steps", 0))
        print(
            "loaded policy-gradient model from {} (episodes={}, updates={})".format(
                load_path,
                completed_episodes,
                model.num_updates,
            )
        )
    else:
        model = LinearPolicyGradientModel(feature_mode=flags.feature_mode)
        completed_episodes = 0
        total_steps = 0

    temperature = flags.temperature
    if load_path and "temperature" in model.metadata:
        temperature = float(model.metadata["temperature"])

    agents = {
        position: SelfPlayPolicyGradientAgent(
            position=position,
            model=model,
            learning_rate=flags.learning_rate,
            gamma=flags.gamma,
            temperature=temperature,
            max_candidate_actions=flags.max_candidate_actions,
            reward_scale=flags.reward_scale,
            reward_shaping=flags.reward_shaping,
            baseline_decay=flags.baseline_decay,
            l2=flags.l2,
            grad_clip=flags.grad_clip,
            rng=random.Random(flags.seed + index + 1),
        )
        for index, position in enumerate(POSITIONS)
    }
    env = GameEnv(agents)

    recent_landlord_wins = deque(maxlen=max(1, flags.log_interval))
    recent_steps = deque(maxlen=max(1, flags.log_interval))
    recent_returns = deque(maxlen=max(1, flags.log_interval * 3))
    progress_enabled = bool(flags.progress_interval and flags.progress_interval > 0)
    start_time = time.time()
    last_steps = 0

    print("Policy-gradient feature mode: {}".format(model.feature_mode))
    print("Feature dim: {}".format(len(model.feature_names)))
    print(
        "Reward objective: {} scale={} shaping={}".format(
            flags.objective,
            flags.reward_scale,
            flags.reward_shaping,
        )
    )
    print(
        "Action mask: legal actions pruned to max_candidate_actions={}".format(
            flags.max_candidate_actions
        )
    )

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
                if recent_returns
                else 0.0
            )
            print(
                "episode={} updates={} temperature={:.4f} landlord_wp={:.3f} "
                "avg_steps={:.1f} avg_return={:.3f}".format(
                    global_episode,
                    model.num_updates,
                    temperature,
                    win_rate,
                    avg_steps,
                    avg_return,
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
            print("saved policy-gradient model to {}".format(path))

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
    print("saved policy-gradient model to {}".format(final_path))
    return model
