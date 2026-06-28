import os
import pickle
import random
import time
from collections import deque

from douzero.env.game import GameEnv


POSITIONS = ("landlord", "landlord_up", "landlord_down")
CARD_RANKS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30)
CARD_TO_INDEX = {card: index for index, card in enumerate(CARD_RANKS)}
DEFAULT_QTABLE_PATH = "qlearning_checkpoints/qlearning/q_table.pkl"
BOMB_ACTIONS = {
    (3, 3, 3, 3), (4, 4, 4, 4), (5, 5, 5, 5),
    (6, 6, 6, 6), (7, 7, 7, 7), (8, 8, 8, 8),
    (9, 9, 9, 9), (10, 10, 10, 10), (11, 11, 11, 11),
    (12, 12, 12, 12), (13, 13, 13, 13), (14, 14, 14, 14),
    (17, 17, 17, 17), (20, 30),
}

DECK = []
for _card in range(3, 15):
    DECK.extend([_card for _ in range(4)])
DECK.extend([17 for _ in range(4)])
DECK.extend([20, 30])


def action_key(action):
    return tuple(sorted(action))


#note: 用固定 15 维数组统计手牌，替代 Counter 以降低高频状态编码开销。
def cards_to_counts(cards):
    counts = [0] * len(CARD_RANKS)
    for card in cards or []:
        index = CARD_TO_INDEX.get(card)
        if index is not None:
            counts[index] += 1
    return tuple(counts)


def action_seq_key(actions):
    return tuple(action_key(action) for action in (actions or []))


def make_state_key(infoset, state_mode="public"):
    num_left = infoset.num_cards_left_dict or {}
    num_left_key = tuple(num_left.get(position, 0) for position in POSITIONS)
    base = (
        infoset.player_position,
        cards_to_counts(infoset.player_hand_cards),
        action_key(infoset.last_move),
        num_left_key,
    )

    if state_mode == "hand_only":
        return base

    played = infoset.played_cards or {}
    played_key = tuple(cards_to_counts(played.get(position, []))
                       for position in POSITIONS)
    return base + (
        action_seq_key(infoset.last_two_moves),
        played_key,
        cards_to_counts(infoset.three_landlord_cards),
        infoset.last_pid,
        int(infoset.bomb_num or 0),
    )


def generate_card_play_data(rng):
    deck = DECK.copy()
    rng.shuffle(deck)
    card_play_data = {
        "landlord": deck[:20],
        "landlord_up": deck[20:37],
        "landlord_down": deck[37:54],
        "three_landlord_cards": deck[17:20],
    }
    for cards in card_play_data.values():
        cards.sort()
    return card_play_data


#note: 终局奖励仍以胜负为主，reward_scale 用于整体放大奖励数值。
def reward_for_position(position, winner, bomb_num, objective, reward_scale=1.0):
    if objective == "adp":
        scale = 2.0 ** bomb_num
    elif objective == "logadp":
        scale = float(bomb_num + 1)
    else:
        scale = 1.0
    scale *= reward_scale

    landlord_reward = scale if winner == "landlord" else -scale
    if position == "landlord":
        return landlord_reward
    return -landlord_reward


#note: 返回农民队友位置；地主没有队友。
def teammate_position(position):
    if position == "landlord_up":
        return "landlord_down"
    if position == "landlord_down":
        return "landlord_up"
    return None


#note: 判断某个位置最近出牌的人是队友还是敌人，用于中间奖励塑形。
def relation_to_last_player(position, last_pid):
    teammate = teammate_position(position)
    if teammate and last_pid == teammate:
        return "teammate"
    if last_pid and last_pid != position:
        return "enemy"
    return "self"


#note: 用简化手牌坏度估计出牌后结构是否变好，数值越大表示手牌越难打。
def hand_badness(cards):
    counts = cards_to_counts(cards)
    singles = sum(1 for count in counts if count == 1)
    pairs = sum(1 for count in counts if count == 2)
    triples = sum(1 for count in counts if count == 3)
    bombs = sum(1 for count in counts if count == 4)
    groups = singles + pairs + triples + bombs
    control_cards = (
        counts[CARD_TO_INDEX[14]] + counts[CARD_TO_INDEX[17]] +
        counts[CARD_TO_INDEX[20]] + counts[CARD_TO_INDEX[30]]
    )
    return (
        0.03 * groups +
        0.02 * singles +
        0.003 * sum(counts) -
        0.015 * control_cards -
        0.02 * bombs
    )


#note: 计算中间奖励，让 Q-learning 不必只依赖终局输赢往前回传信号。
def shaped_reward_for_action(position, infoset, action, reward_scale=1.0):
    action = action_key(action)
    reward = 0.0
    hand_cards = list(infoset.player_hand_cards or [])
    hand_count = len(hand_cards)
    next_hand = hand_cards.copy()
    for card in action:
        if card in next_hand:
            next_hand.remove(card)

    if action:
        reward += 0.005 * len(action)
        reward += max(-0.08, min(0.08, hand_badness(hand_cards) - hand_badness(next_hand)))

        next_count = hand_count - len(action)
        if next_count == 0:
            reward += 0.20
        elif next_count <= 2:
            reward += 0.05

        if action in BOMB_ACTIONS and next_count > 2:
            reward -= 0.04

    num_left = infoset.num_cards_left_dict or {}
    last_relation = relation_to_last_player(position, infoset.last_pid)
    if last_relation == "teammate":
        teammate = teammate_position(position)
        teammate_cards = num_left.get(teammate, 17)
        if teammate_cards <= 2:
            reward += 0.05 if not action else -0.05
    elif last_relation == "enemy":
        enemy_cards = num_left.get(infoset.last_pid, 17)
        can_follow = any(legal_action for legal_action in infoset.legal_actions)
        if enemy_cards <= 2:
            if action:
                reward += 0.06
            elif can_follow:
                reward -= 0.06

    return reward * reward_scale


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


def training_metadata(flags, global_episode, epsilon, load_path, total_steps,
                      last_episode_steps, elapsed_sec=0.0, completed_episodes=0):
    trained_episodes = max(0, global_episode - completed_episodes)
    return {
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
        "state_mode": flags.state_mode,
        "savedir": flags.savedir,
        "resumed_from": load_path,
    }


#note: 将秒数格式化成短时间字符串，供训练进度条显示 ETA。
def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "{}h{:02d}m".format(hours, minutes)
    if minutes:
        return "{}m{:02d}s".format(minutes, sec)
    return "{}s".format(sec)


#note: 打印单行训练进度条，展示进度、Q 表大小、速度和预计剩余时间。
def print_progress_bar(episode, total_episodes, completed_episodes, qtable,
                       epsilon, recent_steps, recent_landlord_wins,
                       start_time, total_steps):
    if total_episodes <= 0:
        return

    progress = episode / float(total_episodes)
    width = 30
    filled = int(width * progress)
    bar = "#" * filled + "." * (width - filled)

    elapsed = time.time() - start_time
    speed = episode / elapsed if elapsed > 0 else 0.0
    remaining = (total_episodes - episode) / speed if speed > 0 else 0.0
    avg_steps = (
        sum(recent_steps) / float(len(recent_steps)) if recent_steps else 0.0
    )
    landlord_wp = (
        sum(recent_landlord_wins) / float(len(recent_landlord_wins))
        if recent_landlord_wins else 0.0
    )
    global_episode = completed_episodes + episode

    line = (
        "\r[{}] {:6.2f}% episode={}/{} global={} q_size={} "
        "epsilon={:.4f} landlord_wp={:.3f} avg_steps={:.1f} "
        "speed={:.2f}eps/s steps={} eta={}"
    ).format(
        bar,
        progress * 100.0,
        episode,
        total_episodes,
        global_episode,
        len(qtable.values),
        epsilon,
        landlord_wp,
        avg_steps,
        speed,
        total_steps,
        format_duration(remaining),
    )
    print(line + " " * 8, end="", flush=True)


class QTable(object):
    def __init__(self, values=None, metadata=None):
        self.values = values or {}
        self.metadata = metadata or {}

    def _key(self, position, state, action):
        return (position, state, action)

    def get(self, position, state, action):
        return self.values.get(self._key(position, state, action), 0.0)

    def set(self, position, state, action, value):
        self.values[self._key(position, state, action)] = float(value)

    def best_value(self, position, state, legal_actions):
        if not legal_actions:
            return 0.0
        return max(self.get(position, state, action_key(action))
                   for action in legal_actions)

    def select_action(self, position, state, legal_actions, epsilon=0.0, rng=None):
        if not legal_actions:
            return []

        rng = rng or random
        if epsilon > 0.0 and rng.random() < epsilon:
            return list(action_key(rng.choice(legal_actions)))

        scored = []
        best = None
        for action in legal_actions:
            key = action_key(action)
            value = self.get(position, state, key)
            if best is None or value > best:
                best = value
                scored = [key]
            elif value == best:
                scored.append(key)
        return list(rng.choice(scored))

    def update(self, position, state, action, reward, next_state, next_actions,
               alpha, gamma):
        old_value = self.get(position, state, action)
        next_best = 0.0
        if next_state is not None:
            next_best = self.best_value(position, next_state, next_actions)
        target = reward + gamma * next_best
        new_value = old_value + alpha * (target - old_value)
        self.set(position, state, action, new_value)
        return new_value

    def save(self, path, metadata=None):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.metadata = metadata or self.metadata
        payload = {
            "version": 1,
            "values": self.values,
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
        if isinstance(payload, dict) and "values" in payload:
            return cls(payload["values"], payload.get("metadata", {}))
        return cls(payload, {})


class SelfPlayQLearningAgent(object):
    def __init__(self, position, qtable, alpha, gamma, epsilon,
                 state_mode="public", reward_scale=1.0,
                 reward_shaping=True, rng=None):
        self.name = "SelfPlayQLearning"
        self.position = position
        self.qtable = qtable
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_mode = state_mode
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.rng = rng or random.Random()
        self.pending = None
        self.num_updates = 0

    def begin_episode(self):
        self.pending = None

    def act(self, infoset):
        state = make_state_key(infoset, self.state_mode)
        if self.pending is not None:
            prev_state, prev_action, prev_reward = self.pending
            self.qtable.update(
                self.position,
                prev_state,
                prev_action,
                reward=prev_reward,
                next_state=state,
                next_actions=infoset.legal_actions,
                alpha=self.alpha,
                gamma=self.gamma,
            )
            self.num_updates += 1

        action = self.qtable.select_action(
            self.position,
            state,
            infoset.legal_actions,
            epsilon=self.epsilon,
            rng=self.rng,
        )
        shaped_reward = 0.0
        if self.reward_shaping:
            shaped_reward = shaped_reward_for_action(
                self.position, infoset, action, self.reward_scale
            )
        self.pending = (state, action_key(action), shaped_reward)
        return action

    def finish_episode(self, reward):
        if self.pending is None:
            return
        state, action, shaped_reward = self.pending
        self.qtable.update(
            self.position,
            state,
            action,
            reward=reward + shaped_reward,
            next_state=None,
            next_actions=None,
            alpha=self.alpha,
            gamma=self.gamma,
        )
        self.num_updates += 1
        self.pending = None


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
            raise FileNotFoundError("Cannot resume: Q table not found at {}".format(load_path))
        qtable = QTable.load(load_path)
        completed_episodes = int(
            qtable.metadata.get("episodes", 0) or episode_from_checkpoint_path(load_path)
        )
        total_steps = int(qtable.metadata.get("total_steps", 0))
        print("loaded Q table from {} (q_size={}, episodes={})".format(
            load_path, len(qtable.values), completed_episodes
        ))
    else:
        qtable = QTable()
        completed_episodes = 0
        total_steps = 0

    agents = {
        position: SelfPlayQLearningAgent(
            position=position,
            qtable=qtable,
            alpha=flags.alpha,
            gamma=flags.gamma,
            epsilon=flags.epsilon,
            state_mode=flags.state_mode,
            reward_scale=flags.reward_scale,
            reward_shaping=flags.reward_shaping,
            rng=random.Random(flags.seed + index + 1),
        )
        for index, position in enumerate(POSITIONS)
    }
    env = GameEnv(agents)

    recent_landlord_wins = deque(maxlen=max(1, flags.log_interval))
    recent_steps = deque(maxlen=max(1, flags.log_interval))
    epsilon = flags.epsilon
    if load_path and "epsilon" in qtable.metadata:
        epsilon = float(qtable.metadata["epsilon"])
    progress_enabled = bool(flags.progress_interval and flags.progress_interval > 0)
    start_time = time.time()
    last_steps = 0

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
            progress_enabled and
            (
                episode == 1 or
                episode == flags.episodes or
                episode % flags.progress_interval == 0 or
                should_log or
                should_save
            )
        )

        if should_progress:
            print_progress_bar(
                episode, flags.episodes, completed_episodes, qtable, epsilon,
                recent_steps, recent_landlord_wins, start_time, total_steps
            )

        if should_log:
            if progress_enabled:
                print()
            win_rate = sum(recent_landlord_wins) / float(len(recent_landlord_wins))
            avg_steps = sum(recent_steps) / float(len(recent_steps))
            elapsed = max(1e-6, time.time() - start_time)
            print(
                "episode={} q_size={} epsilon={:.4f} landlord_wp={:.3f} "
                "avg_steps={:.1f} elapsed_sec={:.1f} speed={:.2f}eps/s "
                "sec_per_ep={:.4f}".format(
                    global_episode, len(qtable.values), epsilon, win_rate, avg_steps,
                    elapsed,
                    episode / elapsed,
                    elapsed / max(1, episode)
                )
            )

        if should_save:
            if progress_enabled and not should_log:
                print()
            path = checkpoint_path(flags, global_episode)
            qtable.save(path, metadata=training_metadata(
                flags, global_episode, epsilon, load_path, total_steps, steps,
                max(1e-6, time.time() - start_time), completed_episodes
            ))
            print("saved Q table to {}".format(path))

    if progress_enabled:
        print()

    final_episode = completed_episodes + flags.episodes
    final_path = checkpoint_path(flags, final_episode)
    qtable.save(final_path, metadata=training_metadata(
        flags, final_episode, epsilon, load_path, total_steps, last_steps,
        max(1e-6, time.time() - start_time), completed_episodes
    ))
    print("saved Q table to {}".format(final_path))
    return qtable
