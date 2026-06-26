from __future__ import annotations

import csv
import os
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from douzero.env.env import get_obs
from douzero.env.game import GameEnv
from douzero.rl.qlearning import (
    POSITIONS,
    generate_card_play_data,
    reward_for_position,
    shaped_reward_for_action,
)


DEFAULT_APPROX_DOUFEATURE_PATH = "approx_qlearning_checkpoints/approx_doufeature/model.pkl"
DEFAULT_APPROX_DOUFEATURE_DIR = "approx_qlearning_checkpoints/approx_doufeature"
CARD_RANKS_13 = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17)
UPDATE_MODES = ("td", "mc", "mc_adv")
X_DIMS = {"landlord": 373, "landlord_up": 484, "landlord_down": 484}
Z_DIM = 5 * 162
FEATURE_DIMS = {position: X_DIMS[position] + Z_DIM for position in POSITIONS}
_TORCH_INTEROP_THREADS_SET = False


@dataclass
class Transition:
    position: str
    feature: np.ndarray
    reward: float
    next_features: np.ndarray | None = None
    next_value: float | None = None
    final_reward: float | None = None


#TODO: 生成 DouZero 54 维牌面编码中每一维的人类可读名称。
def card54_names(prefix):
    names = []
    for rank in CARD_RANKS_13:
        names.extend(f"{prefix}_{rank}_slot{slot}" for slot in range(1, 5))
    names.extend((f"{prefix}_black_joker", f"{prefix}_red_joker"))
    return tuple(names)


#TODO: 生成 DouZero 地主 x_batch 的特征名，顺序严格对应 env.py 中 _get_obs_landlord。
def landlord_x_names():
    names = []
    for block in (
        "x_my_hand",
        "x_other_hand",
        "x_last_action",
        "x_landlord_up_played",
        "x_landlord_down_played",
    ):
        names.extend(card54_names(block))
    names.extend(f"x_landlord_up_left_{i}" for i in range(1, 18))
    names.extend(f"x_landlord_down_left_{i}" for i in range(1, 18))
    names.extend(f"x_bomb_num_{i}" for i in range(15))
    names.extend(card54_names("x_action"))
    return tuple(names)


#TODO: 生成 DouZero 农民 x_batch 的特征名，顺序严格对应 env.py 中两个 farmer 分支。
def farmer_x_names():
    names = []
    for block in (
        "x_my_hand",
        "x_other_hand",
        "x_landlord_played",
        "x_teammate_played",
        "x_last_action",
        "x_last_landlord_action",
        "x_last_teammate_action",
    ):
        names.extend(card54_names(block))
    names.extend(f"x_landlord_left_{i}" for i in range(1, 21))
    names.extend(f"x_teammate_left_{i}" for i in range(1, 18))
    names.extend(f"x_bomb_num_{i}" for i in range(15))
    names.extend(card54_names("x_action"))
    return tuple(names)


#TODO: 生成 DouZero z_batch 最近 15 手牌历史的展平特征名。
def z_names():
    names = []
    for group in range(5):
        for move in range(3):
            names.extend(card54_names(f"z_group{group}_move{move}"))
    return tuple(names)


FEATURE_NAMES = {
    "landlord": landlord_x_names() + z_names(),
    "landlord_up": farmer_x_names() + z_names(),
    "landlord_down": farmer_x_names() + z_names(),
}


#TODO: 根据 --device 选择 batch 更新设备；没有 CUDA 时自动回退 CPU。
def resolve_batch_device(device):
    if torch is None:
        return None
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


#TODO: 返回某个位置的特征名，并检查维度和 DouZero 编码一致。
def feature_names_for_position(position):
    names = FEATURE_NAMES[position]
    if len(names) != FEATURE_DIMS[position]:
        raise ValueError(f"feature dim mismatch for {position}: {len(names)}")
    return names


#TODO: 把 infoset 转成 DouZero 原始 x_batch + z_batch 展平特征矩阵。
def douzero_features_for_infoset(position, infoset):
    obs = get_obs(infoset)
    actions = list(obs["legal_actions"])
    if not actions:
        return [], np.zeros((0, FEATURE_DIMS[position]), dtype=np.float32)
    x = np.asarray(obs["x_batch"], dtype=np.float32)
    z = np.asarray(obs["z_batch"], dtype=np.float32).reshape(len(actions), Z_DIM)
    features = np.concatenate((x, z), axis=1)
    if features.shape[1] != FEATURE_DIMS[position]:
        raise ValueError(f"{position} feature dim {features.shape[1]} != {FEATURE_DIMS[position]}")
    return actions, features


class ApproxDouFeatureModel:
    #TODO: 初始化三套线性权重，特征维度严格跟 DouZero 的 position-specific 输入对齐。
    def __init__(self, weights=None, metadata=None):
        self.weights = {}
        for position in POSITIONS:
            if weights and position in weights:
                value = np.asarray(weights[position], dtype=np.float32)
            else:
                value = np.zeros(FEATURE_DIMS[position], dtype=np.float32)
            if value.shape[0] != FEATURE_DIMS[position]:
                raise ValueError(f"{position} weight dim {value.shape[0]} != {FEATURE_DIMS[position]}")
            self.weights[position] = value
        self.metadata = metadata or {}
        self.num_updates = int(self.metadata.get("updates", 0) or 0)

    #TODO: 生成可发给并行 worker 的轻量权重快照。
    def snapshot(self):
        return {position: weight.copy() for position, weight in self.weights.items()}

    #TODO: 计算某个位置下所有候选动作的线性 Q 值。
    def q_values(self, position, features):
        if features is None or len(features) == 0:
            return np.zeros(0, dtype=np.float32)
        return features @ self.weights[position]

    #TODO: 返回下一状态候选动作的最大 Q 值，供 TD target 使用。
    def best_value(self, position, features):
        values = self.q_values(position, features)
        return float(np.max(values)) if values.size else 0.0

    #TODO: 用 epsilon-greedy 从 DouZero 候选动作 batch 中选动作。
    def select_action(self, position, actions, features, epsilon=0.0, rng=None):
        rng = rng or random
        if not actions:
            return [], np.zeros(FEATURE_DIMS[position], dtype=np.float32)
        if epsilon > 0.0 and rng.random() < epsilon:
            index = rng.randrange(len(actions))
        else:
            values = self.q_values(position, features)
            best = np.flatnonzero(values == values.max())
            index = int(rng.choice(best.tolist()))
        return list(actions[index]), features[index].copy()

    #TODO: 对单条样本做线性半梯度更新，并返回 error 与权重变化用于诊断。
    def update_one(self, position, feature, target, alpha, l2=0.0, clip_td=0.0):
        weight = self.weights[position]
        old_value = float(np.dot(weight, feature))
        raw_error = float(target - old_value)
        error = raw_error
        if clip_td and clip_td > 0:
            error = max(-clip_td, min(clip_td, error))
        delta_w = alpha * (error * feature - l2 * weight)
        weight += delta_w.astype(np.float32)
        self.num_updates += 1
        return raw_error, error, delta_w

    #TODO: 保存 checkpoint，包含 position-specific 特征名，便于解释权重。
    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or self.metadata
        payload = {
            "version": 1,
            "algorithm": "approx_doufeature",
            "feature_schema": "douzero_x_batch_plus_flat_z_batch",
            "feature_dims": FEATURE_DIMS,
            "feature_names": {p: list(feature_names_for_position(p)) for p in POSITIONS},
            "weights": {p: w.astype(float).tolist() for p, w in self.weights.items()},
            "metadata": self.metadata,
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as handle:
            pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

    #TODO: 加载 approx_doufeature checkpoint。
    @classmethod
    def load(cls, path):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("algorithm") != "approx_doufeature":
            raise ValueError(f"not an approx_doufeature checkpoint: {path}")
        return cls(payload["weights"], payload.get("metadata", {}))


class FeatureDiagnostics:
    #TODO: 初始化特征诊断累积器，用 error 和权重变化定位抖动来源。
    def __init__(self, path, topk=20):
        self.path = Path(path)
        self.topk = int(topk)
        self.reset()

    #TODO: 清空一个日志窗口内的诊断统计。
    def reset(self):
        self.abs_error_feature = {p: np.zeros(FEATURE_DIMS[p], dtype=np.float64) for p in POSITIONS}
        self.abs_delta_w = {p: np.zeros(FEATURE_DIMS[p], dtype=np.float64) for p in POSITIONS}
        self.signed_delta_w = {p: np.zeros(FEATURE_DIMS[p], dtype=np.float64) for p in POSITIONS}
        self.abs_error_sum = {p: 0.0 for p in POSITIONS}
        self.count = {p: 0 for p in POSITIONS}

    #TODO: 记录一次更新里哪些特征乘上较大 error 或产生较大权重变化。
    def observe(self, position, feature, raw_error, delta_w):
        self.abs_error_feature[position] += abs(raw_error) * np.abs(feature)
        self.abs_delta_w[position] += np.abs(delta_w)
        self.signed_delta_w[position] += delta_w
        self.abs_error_sum[position] += abs(raw_error)
        self.count[position] += 1

    #TODO: 记录一次 batch 更新的聚合诊断统计，避免逐条 Python 循环拖慢 GPU 训练。
    def observe_batch_stats(self, position, abs_error_feature, abs_delta_w,
                            signed_delta_w, abs_error_sum, count):
        self.abs_error_feature[position] += abs_error_feature
        self.abs_delta_w[position] += abs_delta_w
        self.signed_delta_w[position] += signed_delta_w
        self.abs_error_sum[position] += float(abs_error_sum)
        self.count[position] += int(count)

    #TODO: 把每个位置 top-k 抖动特征写入 CSV，然后重置窗口。
    def flush(self, episode, model):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists()
        with open(self.path, "a", newline="") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow([
                    "episode",
                    "position",
                    "rank",
                    "feature",
                    "weight",
                    "abs_delta_w",
                    "signed_delta_w",
                    "abs_error_x_feature",
                    "avg_abs_error",
                    "updates",
                ])
            for position in POSITIONS:
                names = feature_names_for_position(position)
                score = self.abs_delta_w[position] + self.abs_error_feature[position]
                top = np.argsort(score)[-self.topk:][::-1]
                avg_error = self.abs_error_sum[position] / max(1, self.count[position])
                for rank, index in enumerate(top, start=1):
                    writer.writerow([
                        episode,
                        position,
                        rank,
                        names[int(index)],
                        float(model.weights[position][int(index)]),
                        float(self.abs_delta_w[position][int(index)]),
                        float(self.signed_delta_w[position][int(index)]),
                        float(self.abs_error_feature[position][int(index)]),
                        float(avg_error),
                        self.count[position],
                    ])
        self.reset()


class ReturnBaseline:
    #TODO: 维护每个位置的滑动终局回报均值，把 MC target 转成 advantage 降低角色偏置。
    def __init__(self, values=None, beta=0.01):
        self.values = {position: 0.0 for position in POSITIONS}
        if values:
            for position in POSITIONS:
                self.values[position] = float(values.get(position, 0.0))
        self.beta = float(beta)

    #TODO: 用本批终局回报更新 position baseline。
    def update(self, returns_by_position):
        if self.beta <= 0:
            return
        for position, returns in returns_by_position.items():
            if not returns:
                continue
            mean_return = float(sum(returns) / len(returns))
            self.values[position] = (
                (1.0 - self.beta) * self.values[position]
                + self.beta * mean_return
            )

    #TODO: 返回某个位置的当前 baseline。
    def value(self, position):
        return self.values.get(position, 0.0)


class ReplayBuffer:
    #TODO: 按位置保存最近 transitions，并支持随机 mini-batch 采样。
    def __init__(self, capacity=0, seed=0):
        self.capacity = int(capacity or 0)
        self.rng = random.Random(seed)
        self.data = {
            position: deque(maxlen=self.capacity if self.capacity > 0 else None)
            for position in POSITIONS
        }

    #TODO: 将采样得到的 transitions 加入 buffer。
    def extend(self, transitions):
        if self.capacity <= 0:
            return
        for transition in transitions:
            self.data[transition.position].append(transition)

    #TODO: 从所有 position 的 buffer 中尽量均衡地抽取一个 mini-batch。
    def sample(self, batch_size):
        available = [p for p in POSITIONS if self.data[p]]
        if not available or batch_size <= 0:
            return []
        per_position = max(1, batch_size // len(available))
        batch = []
        for position in available:
            items = self.data[position]
            count = min(len(items), per_position)
            indices = self.rng.sample(range(len(items)), count)
            batch.extend(items[index] for index in indices)
        remaining = batch_size - len(batch)
        if remaining > 0:
            flat = [item for position in available for item in self.data[position]]
            count = min(remaining, len(flat))
            batch.extend(self.rng.sample(flat, count))
        self.rng.shuffle(batch)
        return batch

    #TODO: 返回当前 buffer 中样本总数，便于日志记录。
    def __len__(self):
        return sum(len(items) for items in self.data.values())


class _CollectAgent:
    #TODO: 构造只负责采样 trajectory 的 agent；worker 内不更新权重。
    def __init__(self, position, model, update_mode, epsilon, reward_scale,
                 reward_shaping, rng, transitions):
        self.position = position
        self.model = model
        self.update_mode = update_mode
        self.epsilon = epsilon
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.rng = rng
        self.transitions = transitions
        self.pending = None
        self.episode_features = []

    #TODO: 每局开始时清空 pending 和 Monte Carlo 轨迹缓存。
    def begin_episode(self):
        self.pending = None
        self.episode_features = []

    #TODO: 当前玩家行动；TD 模式闭合上一条 transition，MC/MC-adv 模式记录整局动作特征。
    def act(self, infoset):
        actions, features = douzero_features_for_infoset(self.position, infoset)
        if self.update_mode == "td" and self.pending is not None:
            feature, reward = self.pending
            next_value = self.model.best_value(self.position, features)
            self.transitions.append(Transition(
                self.position, feature, reward, None, next_value
            ))

        action, feature = self.model.select_action(
            self.position, actions, features, epsilon=self.epsilon, rng=self.rng
        )
        shaped = 0.0
        if self.reward_shaping:
            shaped = shaped_reward_for_action(self.position, infoset, action, self.reward_scale)
        if self.update_mode in ("mc", "mc_adv"):
            self.episode_features.append((feature, shaped))
        else:
            self.pending = (feature, shaped)
        return action

    #TODO: 终局时把最终回报接到 TD 最后一条 transition，或回填给 MC/MC-adv 全局轨迹。
    def finish_episode(self, final_reward):
        if self.update_mode in ("mc", "mc_adv"):
            for feature, shaped in self.episode_features:
                self.transitions.append(Transition(
                    self.position,
                    feature,
                    final_reward + shaped,
                    None,
                    None,
                    final_reward,
                ))
            self.episode_features = []
            return
        if self.pending is not None:
            feature, shaped = self.pending
            self.transitions.append(Transition(
                self.position,
                feature,
                final_reward + shaped,
                None,
                None,
                final_reward,
            ))
            self.pending = None


#TODO: worker 子进程入口，使用固定权重快照并行采样若干完整 episode。
def run_worker(task):
    configure_cpu_threads(task.get("cpu_threads", 1))
    rng = random.Random(task["seed"])
    model = ApproxDouFeatureModel(task["weights"])
    transitions = []
    agents = {
        position: _CollectAgent(
            position,
            model,
            task["update_mode"],
            task["epsilon"],
            task["reward_scale"],
            task["reward_shaping"],
            random.Random(task["seed"] + index + 1),
            transitions,
        )
        for index, position in enumerate(POSITIONS)
    }
    env = GameEnv(agents)
    wins = 0
    total_steps = 0
    completed = 0
    for _ in range(task["episodes"]):
        for agent in agents.values():
            agent.begin_episode()
        env.reset()
        env.card_play_init(generate_card_play_data(rng))
        steps = 0
        while not env.game_over and steps < task["max_steps"]:
            env.step()
            steps += 1
        total_steps += steps
        completed += 1
        if env.game_over:
            winner = env.get_winner()
            bomb_num = env.get_bomb_num()
            wins += 1 if winner == "landlord" else 0
            for position, agent in agents.items():
                reward = reward_for_position(
                    position, winner, bomb_num, task["objective"], task["reward_scale"]
                )
                agent.finish_episode(reward)
        else:
            for agent in agents.values():
                agent.finish_episode(0.0)
    return {
        "episodes": completed,
        "steps": total_steps,
        "landlord_wins": wins,
        "transitions": transitions,
    }


#TODO: 拼出当前任务 checkpoint 目录。
def checkpoint_dir(flags):
    return os.path.join(flags.savedir, flags.name)


#TODO: 按 episode 生成 checkpoint 路径，兼容显式 --output。
def checkpoint_path(flags, episode):
    if flags.output:
        return flags.output
    return os.path.join(checkpoint_dir(flags), f"{episode}.pkl")


#TODO: 从 checkpoint 文件名中提取 episode 数。
def episode_from_checkpoint_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem) if stem.isdigit() else 0


#TODO: 寻找任务目录下最新的数字命名 checkpoint。
def latest_checkpoint_path(flags):
    directory = checkpoint_dir(flags)
    if not os.path.isdir(directory):
        return None
    candidates = []
    for filename in os.listdir(directory):
        if filename.endswith(".pkl") and os.path.splitext(filename)[0].isdigit():
            path = os.path.join(directory, filename)
            candidates.append((episode_from_checkpoint_path(path), path))
    return sorted(candidates)[-1][1] if candidates else None


#TODO: 组装 checkpoint metadata，记录并行、更新模式和诊断文件位置。
def training_metadata(flags, episode, epsilon, load_path, model, total_steps, diag_path):
    return {
        "algorithm": "approx_doufeature",
        "name": flags.name,
        "episodes": episode,
        "updates": model.num_updates,
        "update_mode": flags.update_mode,
        "objective": flags.objective,
        "reward_scale": flags.reward_scale,
        "reward_shaping": flags.reward_shaping,
        "alpha": flags.alpha,
        "gamma": flags.gamma,
        "epsilon": epsilon,
        "l2": flags.l2,
        "clip_td": flags.clip_td,
        "num_workers": flags.num_workers,
        "worker_episodes": flags.worker_episodes,
        "buffer_size": flags.buffer_size,
        "learn_batch_size": flags.learn_batch_size,
        "learn_steps": flags.learn_steps,
        "baseline_beta": flags.baseline_beta,
        "return_baseline": model.metadata.get("return_baseline", {}),
        "feature_schema": "douzero_x_batch_plus_flat_z_batch",
        "feature_dims": FEATURE_DIMS,
        "total_steps": total_steps,
        "resumed_from": load_path,
        "diagnostics_csv": str(diag_path),
    }


#TODO: 根据更新模式计算监督 target；MC advantage 会减去 position baseline。
def transition_target(transition, flags, baseline):
    target = transition.reward
    if flags.update_mode == "td" and transition.next_value is not None:
        target += flags.gamma * transition.next_value
    elif flags.update_mode == "mc_adv":
        target -= baseline.value(transition.position)
    return target


#TODO: 使用 CPU 逐条更新，作为无 torch 或显式 CPU fallback。
def apply_transitions_cpu(model, transitions, flags, diagnostics, baseline):
    abs_error = 0.0
    for transition in transitions:
        target = transition_target(transition, flags, baseline)
        raw_error, _, delta_w = model.update_one(
            transition.position,
            transition.feature,
            target,
            flags.alpha,
            flags.l2,
            flags.clip_td,
        )
        diagnostics.observe(transition.position, transition.feature, raw_error, delta_w)
        abs_error += abs(raw_error)
    return abs_error / max(1, len(transitions))


#TODO: 使用 PyTorch 按位置批量更新线性权重，可在 CUDA 上加速大 batch 矩阵运算。
def apply_transitions_torch(model, transitions, flags, diagnostics, device, baseline):
    if not transitions:
        return 0.0
    total_abs_error = 0.0
    total_count = 0
    for position in POSITIONS:
        group = [item for item in transitions if item.position == position]
        if not group:
            continue
        features_np = np.stack([item.feature for item in group]).astype(np.float32)
        targets_np = np.asarray([
            transition_target(item, flags, baseline)
            for item in group
        ], dtype=np.float32)

        features = torch.from_numpy(features_np).to(device)
        targets = torch.from_numpy(targets_np).to(device)
        weights = torch.from_numpy(model.weights[position]).to(device)
        old_values = features.matmul(weights)
        raw_errors = targets - old_values
        errors = raw_errors
        if flags.clip_td and flags.clip_td > 0:
            errors = torch.clamp(errors, -flags.clip_td, flags.clip_td)

        grad = features.t().matmul(errors) / max(1, features.shape[0])
        if flags.l2 and flags.l2 > 0:
            grad = grad - flags.l2 * weights
        delta_w = flags.alpha * grad
        weights = weights + delta_w
        model.weights[position] = weights.detach().cpu().numpy().astype(np.float32)
        model.num_updates += int(features.shape[0])

        abs_raw = torch.abs(raw_errors)
        abs_error_feature = torch.sum(abs_raw[:, None] * torch.abs(features), dim=0)
        abs_delta_w = torch.abs(delta_w) * features.shape[0]
        signed_delta_w = delta_w * features.shape[0]
        diagnostics.observe_batch_stats(
            position,
            abs_error_feature.detach().cpu().numpy(),
            abs_delta_w.detach().cpu().numpy(),
            signed_delta_w.detach().cpu().numpy(),
            float(torch.sum(abs_raw).detach().cpu().item()),
            int(features.shape[0]),
        )
        total_abs_error += float(torch.sum(abs_raw).detach().cpu().item())
        total_count += int(features.shape[0])
    return total_abs_error / max(1, total_count)


#TODO: 根据 device 自动选择 CPU 逐条更新或 PyTorch batch 更新。
def apply_transitions(model, transitions, flags, diagnostics, device, baseline):
    if device is None:
        return apply_transitions_cpu(model, transitions, flags, diagnostics, baseline)
    return apply_transitions_torch(model, transitions, flags, diagnostics, device, baseline)


#TODO: 从 transitions 中提取终局回报，供 ReturnBaseline 更新。
def returns_by_position(transitions):
    grouped = {position: [] for position in POSITIONS}
    for transition in transitions:
        if transition.final_reward is not None:
            grouped[transition.position].append(float(transition.final_reward))
    return grouped


#TODO: 把剩余 episode 切成若干 worker task。
def build_tasks(flags, model, remaining, epsilon, start_episode):
    tasks = []
    snapshot = model.snapshot()
    workers = max(1, int(flags.num_workers))
    per_worker = max(1, int(flags.worker_episodes))
    for worker_id in range(workers):
        episodes = min(per_worker, remaining)
        if episodes <= 0:
            break
        remaining -= episodes
        tasks.append({
            "episodes": episodes,
            "seed": flags.seed + start_episode * 1009 + worker_id * 9176,
            "weights": snapshot,
            "update_mode": flags.update_mode,
            "epsilon": epsilon,
            "objective": flags.objective,
            "reward_scale": flags.reward_scale,
            "reward_shaping": flags.reward_shaping,
            "max_steps": flags.max_steps,
            "cpu_threads": flags.cpu_threads,
        })
    return tasks


#TODO: 限制 PyTorch/BLAS 在线程池里额外开线程，降低多 worker 训练时的 CPU 压力。
def configure_cpu_threads(cpu_threads):
    global _TORCH_INTEROP_THREADS_SET
    threads = max(1, int(cpu_threads or 1))
    if torch is not None:
        torch.set_num_threads(threads)
        if not _TORCH_INTEROP_THREADS_SET:
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            _TORCH_INTEROP_THREADS_SET = True


#TODO: 执行一次并行采样；num_workers=1 时直接本进程采样方便调试。
def collect_results(pool, tasks):
    if pool is None:
        return [run_worker(task) for task in tasks]
    return pool.map(run_worker, tasks)


#TODO: 把秒数格式化成短 ETA 字符串，方便训练进度条阅读。
def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


#TODO: 打印 approx_doufeature 的单行进度条，显示百分比、速度、ETA 和近期训练指标。
def print_progress_bar(current_episode, start_episode, target_episode, model,
                       epsilon, recent_steps, recent_wins, recent_errors,
                       start_time, total_steps):
    total = max(1, target_episode - start_episode)
    done = max(0, current_episode - start_episode)
    progress = min(1.0, done / float(total))
    width = 30
    filled = int(width * progress)
    bar = "#" * filled + "." * (width - filled)
    elapsed = max(1e-6, time.time() - start_time)
    speed = done / elapsed
    remaining = (total - done) / speed if speed > 0 else 0.0
    avg_steps = sum(recent_steps) / float(len(recent_steps)) if recent_steps else 0.0
    landlord_wp = sum(recent_wins) / float(len(recent_wins)) if recent_wins else 0.0
    avg_error = sum(recent_errors) / float(len(recent_errors)) if recent_errors else 0.0
    line = (
        "\r[{}] {:6.2f}% episode={}/{} global={} updates={} "
        "epsilon={:.4f} landlord_wp={:.3f} avg_steps={:.1f} "
        "avg_abs_error={:.4f} speed={:.2f}eps/s steps={} eta={}"
    ).format(
        bar,
        progress * 100.0,
        done,
        total,
        current_episode,
        model.num_updates,
        epsilon,
        landlord_wp,
        avg_steps,
        avg_error,
        speed,
        total_steps,
        format_duration(remaining),
    )
    print(line + " " * 8, end="", flush=True)


#TODO: 主训练入口，支持 TD 与 Monte Carlo 两种更新模式。
def train(flags):
    if flags.update_mode not in UPDATE_MODES:
        raise ValueError(f"update_mode must be one of {UPDATE_MODES}")
    configure_cpu_threads(flags.cpu_threads)
    device = resolve_batch_device(flags.device)

    load_path = flags.load or (latest_checkpoint_path(flags) if flags.resume else None)
    if load_path:
        model = ApproxDouFeatureModel.load(load_path)
        completed = int(model.metadata.get("episodes", 0) or episode_from_checkpoint_path(load_path))
        total_steps = int(model.metadata.get("total_steps", 0) or 0)
        epsilon = float(model.metadata.get("epsilon", flags.epsilon))
        print(f"loaded approx_doufeature from {load_path} episodes={completed} updates={model.num_updates}")
    else:
        model = ApproxDouFeatureModel()
        completed = 0
        total_steps = 0
        epsilon = flags.epsilon
    baseline = ReturnBaseline(
        model.metadata.get("return_baseline", {}),
        flags.baseline_beta,
    )
    replay_buffer = ReplayBuffer(flags.buffer_size, flags.seed)

    os.makedirs(checkpoint_dir(flags), exist_ok=True)
    diag_path = Path(checkpoint_dir(flags)) / "feature_diagnostics.csv"
    diagnostics = FeatureDiagnostics(diag_path, flags.diag_topk)
    recent_wins = deque(maxlen=max(1, flags.log_interval))
    recent_steps = deque(maxlen=max(1, flags.log_interval))
    recent_errors = deque(maxlen=max(1, flags.log_interval))
    next_log = completed + max(1, flags.log_interval)
    next_save = completed + flags.save_interval if flags.save_interval else None
    start_completed = completed
    target_total = start_completed + flags.episodes
    progress_enabled = bool(flags.progress_interval and flags.progress_interval > 0)
    next_progress = completed + max(1, flags.progress_interval)
    start = time.time()

    print(
        "ApproxDouFeature mode={} episodes={} workers={} worker_episodes={} "
        "device={} buffer_size={} learn_batch={} learn_steps={} baseline={} "
        "feature_dims={} diag={}".format(
            flags.update_mode, flags.episodes, flags.num_workers,
            flags.worker_episodes, device if device is not None else "numpy-cpu",
            flags.buffer_size, flags.learn_batch_size, flags.learn_steps,
            baseline.values if flags.update_mode == "mc_adv" else "off",
            FEATURE_DIMS, diag_path
        )
    )

    pool = Pool(flags.num_workers) if flags.num_workers > 1 else None
    try:
        while completed < target_total:
            tasks = build_tasks(flags, model, target_total - completed, epsilon, completed)
            results = collect_results(pool, tasks)
            transitions = []
            chunk_episodes = 0
            chunk_steps = 0
            chunk_wins = 0
            for result in results:
                transitions.extend(result["transitions"])
                chunk_episodes += result["episodes"]
                chunk_steps += result["steps"]
                chunk_wins += result["landlord_wins"]
            chunk_returns = (
                returns_by_position(transitions)
                if flags.update_mode == "mc_adv" else None
            )
            if flags.buffer_size > 0:
                replay_buffer.extend(transitions)
                learn_errors = []
                for _ in range(max(1, flags.learn_steps)):
                    batch = replay_buffer.sample(flags.learn_batch_size)
                    if not batch:
                        break
                    learn_errors.append(
                        apply_transitions(model, batch, flags, diagnostics, device, baseline)
                    )
                avg_error = (
                    sum(learn_errors) / len(learn_errors)
                    if learn_errors else 0.0
                )
            else:
                avg_error = apply_transitions(
                    model, transitions, flags, diagnostics, device, baseline
                )
            if chunk_returns is not None:
                baseline.update(chunk_returns)
            completed += chunk_episodes
            total_steps += chunk_steps
            epsilon = max(flags.min_epsilon, epsilon * (flags.epsilon_decay ** chunk_episodes))
            model.metadata["return_baseline"] = dict(baseline.values)
            recent_wins.append(chunk_wins / max(1, chunk_episodes))
            recent_steps.append(chunk_steps / max(1, chunk_episodes))
            recent_errors.append(avg_error)

            should_log = bool(flags.log_interval and completed >= next_log)
            should_save = bool(next_save and completed >= next_save)
            should_progress = (
                progress_enabled
                and (
                    completed >= next_progress
                    or completed >= target_total
                    or should_log
                    or should_save
                )
            )

            if should_progress:
                print_progress_bar(
                    completed, start_completed, target_total, model, epsilon,
                    recent_steps, recent_wins, recent_errors, start, total_steps
                )
                while next_progress <= completed:
                    next_progress += max(1, flags.progress_interval)

            if should_log:
                if progress_enabled:
                    print()
                elapsed = max(1e-6, time.time() - start)
                wp = sum(recent_wins) / len(recent_wins)
                avg_steps = sum(recent_steps) / len(recent_steps)
                avg_abs_error = sum(recent_errors) / len(recent_errors)
                speed = (completed - start_completed) / elapsed
                print(
                    "episode={} updates={} epsilon={:.4f} landlord_wp={:.3f} "
                    "avg_steps={:.1f} avg_abs_error={:.4f} buffer={} "
                    "baseline={} speed={:.2f}eps/s".format(
                        completed, model.num_updates, epsilon, wp, avg_steps,
                        avg_abs_error, len(replay_buffer),
                        baseline.values if flags.update_mode == "mc_adv" else "off",
                        speed
                    )
                )
                diagnostics.flush(completed, model)
                next_log += max(1, flags.log_interval)

            if should_save:
                if progress_enabled and not should_log:
                    print()
                model.save(checkpoint_path(flags, completed), training_metadata(
                    flags, completed, epsilon, load_path, model, total_steps, diag_path
                ))
                print(f"saved approx_doufeature to {checkpoint_path(flags, completed)}")
                next_save += flags.save_interval
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if progress_enabled:
        print()

    final_path = checkpoint_path(flags, completed)
    diagnostics.flush(completed, model)
    model.metadata["return_baseline"] = dict(baseline.values)
    model.save(final_path, training_metadata(
        flags, completed, epsilon, load_path, model, total_steps, diag_path
    ))
    print(f"saved approx_doufeature to {final_path}")
    return model


#TODO: 构建与旧 ApproxQ 相近的训练 CLI，并新增 update_mode/并行/诊断参数。
def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="DouZero-feature linear approximate Q-learning")
    parser.add_argument("--episodes", default=10000, type=int)
    parser.add_argument("--name", default="approx_doufeature_logadp", type=str)
    parser.add_argument("--objective", default="logadp", choices=["wp", "adp", "logadp"])
    parser.add_argument("--reward_scale", default=1.0, type=float)
    parser.add_argument("--reward_shaping", action="store_true")
    parser.add_argument("--savedir", default=DEFAULT_APPROX_DOUFEATURE_DIR)
    parser.add_argument("--output", default="")
    parser.add_argument("--load", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--alpha", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.98, type=float)
    parser.add_argument("--epsilon", default=0.1, type=float)
    parser.add_argument("--min_epsilon", default=0.02, type=float)
    parser.add_argument("--epsilon_decay", default=0.99998, type=float)
    parser.add_argument("--l2", default=0.00001, type=float)
    parser.add_argument("--clip_td", default=10.0, type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature_mode", default="douzero", choices=["douzero"])
    parser.add_argument("--max_candidate_actions", default=0, type=int)
    parser.add_argument("--max_steps", default=1000, type=int)
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--progress_interval", default=500, type=int)
    parser.add_argument("--save_interval", default=50000, type=int)
    parser.add_argument("--update_mode", default="td", choices=UPDATE_MODES)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--worker_episodes", default=8, type=int)
    parser.add_argument("--cpu_threads", default=1, type=int)
    parser.add_argument("--buffer_size", default=0, type=int)
    parser.add_argument("--learn_batch_size", default=4096, type=int)
    parser.add_argument("--learn_steps", default=1, type=int)
    parser.add_argument("--baseline_beta", default=0.01, type=float)
    parser.add_argument("--diag_topk", default=20, type=int)
    return parser
