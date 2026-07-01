import os
import random
import time
from collections import Counter, deque

from douzero.env.game import GameEnv
from douzero.rl.actor_critic import (
    LinearActorCriticModel,
    SelfPlayActorCriticAgent,
    checkpoint_dir,
    checkpoint_path,
    episode_from_checkpoint_path,
    latest_checkpoint_path,
    training_metadata,
    print_progress_bar,
)
from douzero.rl.qlearning import (
    POSITIONS,
    generate_card_play_data,
    reward_for_position,
)


class FrozenActorCriticAgent(object):
    def __init__(self, position, model_path, max_candidate_actions=64):
        from douzero.evaluation.actor_critic_agent import ActorCriticAgent

        self.agent = ActorCriticAgent(position, model_path)
        self.agent.max_candidate_actions = max_candidate_actions
        self.name = "FrozenActorCritic"

    def act(self, infoset):
        return self.agent.act(infoset)


class FrozenPolicyGradientAgent(object):
    def __init__(self, position, model_path, max_candidate_actions=64):
        from douzero.evaluation.policy_gradient_agent import PolicyGradientAgent

        self.agent = PolicyGradientAgent(position, model_path)
        self.agent.max_candidate_actions = max_candidate_actions
        self.name = "FrozenPolicyGradient"

    def act(self, infoset):
        return self.agent.act(infoset)


def split_csv(value):
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def parse_train_positions(value):
    if not value or value == "all":
        return set(POSITIONS)
    positions = set(split_csv(value))
    unknown = positions.difference(POSITIONS)
    if unknown:
        raise ValueError("Unknown train_positions: {}".format(sorted(unknown)))
    return positions


def parse_pool(flags):
    methods = split_csv(flags.opponent_pool)
    if not methods:
        raise ValueError("--opponent_pool must not be empty")

    if flags.opponent_weights:
        weights = [float(value) for value in split_csv(flags.opponent_weights)]
        if len(weights) != len(methods):
            raise ValueError("--opponent_weights must match --opponent_pool length")
    else:
        weights = [
            float(flags.current_opponent_weight) if method == "current" else 1.0
            for method in methods
        ]

    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("opponent weights must be non-negative with positive sum")
    return methods, weights


def make_fixed_agent(method, position, max_candidate_actions):
    if method == "current":
        raise ValueError("'current' must be handled by train episode assembly")
    if method.startswith("ac:") or method.startswith("actor_critic:"):
        return FrozenActorCriticAgent(
            position,
            method.split(":", 1)[1],
            max_candidate_actions=max_candidate_actions,
        )
    if method in ("actor_critic", "ac"):
        return FrozenActorCriticAgent(
            position,
            None,
            max_candidate_actions=max_candidate_actions,
        )
    if method.startswith("pg:") or method.startswith("policy_gradient:"):
        return FrozenPolicyGradientAgent(
            position,
            method.split(":", 1)[1],
            max_candidate_actions=max_candidate_actions,
        )
    if method in ("policy_gradient", "pg"):
        return FrozenPolicyGradientAgent(
            position,
            None,
            max_candidate_actions=max_candidate_actions,
        )
    if method == "rlcard":
        from douzero.evaluation.rlcard_agent import RLCardAgent

        return RLCardAgent(position)
    if method == "heuristic":
        from douzero.evaluation.heuristic_agent import HeuristicAgent

        return HeuristicAgent(position)
    if method == "random":
        from douzero.evaluation.random_agent import RandomAgent

        return RandomAgent()
    if method == "probability":
        from douzero.evaluation.probabilistic_response_agent import (
            ProbabilisticResponseAgent,
        )

        return ProbabilisticResponseAgent(position)
    raise ValueError("Unsupported opponent method: {}".format(method))


def choose_weighted(rng, items, weights):
    total = float(sum(weights))
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if threshold <= cumulative:
            return item
    return items[-1]


def current_positions_for_episode(flags, rng, train_positions):
    if flags.fixed_current_position:
        return {flags.fixed_current_position}
    return {rng.choice(sorted(train_positions))}


def training_metadata_with_pool(flags, global_episode, temperature, load_path,
                                total_steps, last_episode_steps, model,
                                pool_methods, pool_weights):
    metadata = training_metadata(
        flags,
        global_episode,
        temperature,
        load_path,
        total_steps,
        last_episode_steps,
        model,
    )
    metadata["algorithm"] = "linear_actor_critic_opponent_pool"
    metadata["train_positions"] = flags.train_positions
    metadata["opponent_pool"] = list(pool_methods)
    metadata["opponent_weights"] = list(pool_weights)
    metadata["fixed_current_position"] = flags.fixed_current_position
    return metadata


def train(flags):
    rng = random.Random(flags.seed)
    train_positions = parse_train_positions(flags.train_positions)
    pool_methods, pool_weights = parse_pool(flags)

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
            "loaded pool actor-critic model from {} "
            "(episodes={}, actor_updates={}, critic_updates={})".format(
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
    recent_current_wins = deque(maxlen=max(1, flags.log_interval))
    recent_current_positions = Counter()
    recent_opponents = Counter()
    current_agents = {
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
    progress_enabled = bool(flags.progress_interval and flags.progress_interval > 0)
    start_time = time.time()
    last_steps = 0

    print("Actor-Critic pool feature mode: {}".format(model.feature_mode))
    print("Actor feature dim: {}".format(len(model.actor_feature_names)))
    print("Critic state feature dim: {}".format(len(model.critic_feature_names)))
    print("Train positions: {}".format(",".join(sorted(train_positions))))
    print("Opponent pool: {}".format(
        ", ".join("{}:{:.3g}".format(m, w) for m, w in zip(pool_methods, pool_weights))
    ))
    print("Reward objective: {} scale={} shaping={}".format(
        flags.objective,
        flags.reward_scale,
        flags.reward_shaping,
    ))

    for episode in range(1, flags.episodes + 1):
        global_episode = completed_episodes + episode
        active_current_positions = current_positions_for_episode(
            flags,
            rng,
            train_positions,
        )
        agents = {}
        current_episode_agents = {}

        for position in POSITIONS:
            if position in active_current_positions:
                agent = current_agents[position]
                agent.begin_episode()
                agent.temperature = temperature
                agents[position] = agent
                current_episode_agents[position] = agent
                recent_current_positions[position] += 1
                continue

            method = choose_weighted(rng, pool_methods, pool_weights)
            if method == "current":
                agent = current_agents[position]
                agent.begin_episode()
                agent.temperature = temperature
                agents[position] = agent
                current_episode_agents[position] = agent
                recent_current_positions[position] += 1
            else:
                agents[position] = make_fixed_agent(
                    method,
                    position,
                    flags.max_candidate_actions,
                )
                recent_opponents[method] += 1

        env = GameEnv(agents)
        env.reset()
        env.card_play_init(generate_card_play_data(rng))

        steps = 0
        while not env.game_over and steps < flags.max_steps:
            env.step()
            steps += 1

        if env.game_over:
            winner = env.get_winner()
            bomb_num = env.get_bomb_num()
            for position, agent in current_episode_agents.items():
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
            current_win = any(
                (position == "landlord" and winner == "landlord")
                or (position != "landlord" and winner != "landlord")
                for position in current_episode_agents
            )
            recent_current_wins.append(1 if current_win else 0)
        else:
            for agent in current_episode_agents.values():
                agent.finish_episode(0.0)
                recent_returns.append(agent.last_episode_return)
            recent_landlord_wins.append(0)
            recent_current_wins.append(0)

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
            current_wp = (
                sum(recent_current_wins) / float(len(recent_current_wins))
                if recent_current_wins else
                0.0
            )
            avg_steps = sum(recent_steps) / float(len(recent_steps))
            avg_return = (
                sum(recent_returns) / float(len(recent_returns))
                if recent_returns else
                0.0
            )
            avg_td = sum(recent_td) / float(len(recent_td)) if recent_td else 0.0
            print(
                "episode={} actor_updates={} critic_updates={} "
                "temperature={:.4f} landlord_wp={:.3f} current_wp={:.3f} "
                "avg_steps={:.1f} avg_return={:.3f} avg_abs_td={:.4f} "
                "current_positions={} opponents={}".format(
                    global_episode,
                    model.num_actor_updates,
                    model.num_critic_updates,
                    temperature,
                    win_rate,
                    current_wp,
                    avg_steps,
                    avg_return,
                    avg_td,
                    dict(recent_current_positions),
                    dict(recent_opponents),
                )
            )
            recent_current_positions.clear()
            recent_opponents.clear()

        if should_save:
            if progress_enabled and not should_log:
                print()
            path = checkpoint_path(flags, global_episode)
            model.save(
                path,
                metadata=training_metadata_with_pool(
                    flags,
                    global_episode,
                    temperature,
                    load_path,
                    total_steps,
                    steps,
                    model,
                    pool_methods,
                    pool_weights,
                ),
            )
            print("saved actor-critic pool model to {}".format(path))

    if progress_enabled:
        print()

    final_episode = completed_episodes + flags.episodes
    final_path = checkpoint_path(flags, final_episode)
    model.save(
        final_path,
        metadata=training_metadata_with_pool(
            flags,
            final_episode,
            temperature,
            load_path,
            total_steps,
            last_steps,
            model,
            pool_methods,
            pool_weights,
        ),
    )
    print("saved actor-critic pool model to {}".format(final_path))
    return model
