from __future__ import annotations

import os
import pickle
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import multiprocessing as mp
    from torch import nn
except ImportError:
    torch = None
    mp = None
    nn = None

from douzero.env.env import Env, get_obs, _cards2array
from douzero.rl.approx_doufeature import (
    ReturnBaseline,
    UPDATE_MODES,
    configure_cpu_threads,
    format_duration,
)
from douzero.rl.qlearning import POSITIONS


DEFAULT_ATTENTION_DOU_DIR = "attention_dou_checkpoints/attention_dou"
DEFAULT_ATTENTION_DOU_PATH = "attention_dou_checkpoints/attention_dou/model.pkl"
TOKEN_DIM = 54
LANDLORD_X_BLOCKS = (54, 54, 54, 54, 54, 17, 17, 15, 54)
FARMER_X_BLOCKS = (54, 54, 54, 54, 54, 54, 54, 20, 17, 15, 54)
X_NO_ACTION_DIMS = {"landlord": 319, "landlord_up": 430, "landlord_down": 430}
X_DIMS = {"landlord": 373, "landlord_up": 484, "landlord_down": 484}
TOKEN_COUNTS = {
    "landlord": len(LANDLORD_X_BLOCKS) + 15,
    "landlord_up": len(FARMER_X_BLOCKS) + 15,
    "landlord_down": len(FARMER_X_BLOCKS) + 15,
}
MEAN_EPISODE_RETURN_BUF = {position: deque(maxlen=100) for position in POSITIONS}


def _device_key(device):
    #TODO: 把 auto/cuda:0/0/cpu 统一成 DouZero actor 使用的 device key。
    if device == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if isinstance(device, int):
        return device
    value = str(device)
    if value.startswith("cuda:"):
        return int(value.split(":", 1)[1])
    if value.isdigit():
        return int(value)
    return "cpu"


def _torch_device(device):
    #TODO: 把 DouZero device key 转成 torch.device。
    key = _device_key(device)
    if key == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{key}")


def resolve_device(device):
    #TODO: 兼容旧接口，返回真实 torch.device。
    if torch is None:
        raise ImportError("attention_dou requires PyTorch")
    return _torch_device(device)


def _format_observation(obs, device):
    #TODO: 严格复刻 DouZero env_utils._format_observation，只替换本文件内 device 解析。
    torch_device = _torch_device(device)
    position = obs["position"]
    x_batch = torch.from_numpy(obs["x_batch"]).to(torch_device)
    z_batch = torch.from_numpy(obs["z_batch"]).to(torch_device)
    x_no_action = torch.from_numpy(obs["x_no_action"])
    z = torch.from_numpy(obs["z"])
    return position, {
        "x_batch": x_batch,
        "z_batch": z_batch,
        "legal_actions": obs["legal_actions"],
    }, x_no_action, z


class Environment:
    #TODO: 严格复刻 DouZero Environment：终局自动 reset，并累计 landlord 视角 return。
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.episode_return = None

    #TODO: 初始化环境并返回 position、obs、env_output。
    def initial(self):
        position, obs, x_no_action, z = _format_observation(
            self.env.reset(), self.device
        )
        self.episode_return = torch.zeros(1, 1)
        return position, obs, {
            "done": torch.ones(1, 1, dtype=torch.bool),
            "episode_return": self.episode_return,
            "obs_x_no_action": x_no_action,
            "obs_z": z,
        }

    #TODO: 执行动作；若终局则自动 reset 到下一局首状态。
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.episode_return += reward
        episode_return = self.episode_return
        if done:
            obs = self.env.reset()
            self.episode_return = torch.zeros(1, 1)
        position, obs, x_no_action, z = _format_observation(obs, self.device)
        return position, obs, {
            "done": torch.tensor(done).view(1, 1),
            "episode_return": episode_return,
            "obs_x_no_action": x_no_action,
            "obs_z": z,
        }


def _cards2tensor(list_cards):
    #TODO: 复用 DouZero 54 维牌编码，用于 obs_action buffer。
    return torch.from_numpy(_cards2array(list_cards))


def _x_tokens_torch(x, position):
    #TODO: 将 DouZero x_batch 切成若干 54 维 token。
    blocks = LANDLORD_X_BLOCKS if position == "landlord" else FARMER_X_BLOCKS
    tokens = x.new_zeros((x.shape[0], len(blocks), TOKEN_DIM))
    offset = 0
    for index, size in enumerate(blocks):
        tokens[:, index, :size] = x[:, offset:offset + size]
        offset += size
    return tokens


def _obs_tokens_torch(z, x, position):
    #TODO: 将 obs_z 和 obs_x 转成 attention 输入 token 序列。
    z_tokens = z.reshape(z.shape[0], 15, TOKEN_DIM)
    return torch.cat((_x_tokens_torch(x, position), z_tokens), dim=1)


def _x_batch_to_tokens(x, position):
    #TODO: 评测时将 numpy x_batch 转为 token batch。
    blocks = LANDLORD_X_BLOCKS if position == "landlord" else FARMER_X_BLOCKS
    token_batch = np.zeros((x.shape[0], len(blocks), TOKEN_DIM), dtype=np.float32)
    offset = 0
    for index, size in enumerate(blocks):
        token_batch[:, index, :size] = x[:, offset:offset + size]
        offset += size
    return token_batch


def _obs_to_tokens(position, obs):
    #TODO: 评测入口：将 infoset obs 的所有合法动作转成 token batch。
    actions = list(obs["legal_actions"])
    if not actions:
        return [], np.zeros((0, TOKEN_COUNTS[position], TOKEN_DIM), dtype=np.float32)
    x = np.asarray(obs["x_batch"], dtype=np.float32)
    z = np.asarray(obs["z_batch"], dtype=np.float32).reshape(len(actions), 15, TOKEN_DIM)
    tokens = np.concatenate((_x_batch_to_tokens(x, position), z), axis=1)
    return actions, tokens


def tokens_for_infoset(position, infoset):
    #TODO: 评测 agent 共用的 infoset -> token batch 编码。
    return _obs_to_tokens(position, get_obs(infoset))


class AttentionPositionModel(nn.Module):
    #TODO: 单个 position 的 multi-head attention Q 网络。
    def __init__(self, position, hidden_dim=128, num_heads=4, num_layers=2,
                 dropout=0.0):
        super().__init__()
        self.position = position
        self.token_count = TOKEN_COUNTS[position]
        self.input_proj = nn.Linear(TOKEN_DIM, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.token_count, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    #TODO: 直接从 token 序列估计 Q-value。
    def forward_tokens(self, tokens):
        x = self.input_proj(tokens.float()) + self.pos_embed[:, :tokens.shape[1], :]
        x = self.encoder(x)
        return self.head(x.mean(dim=1)).squeeze(-1)

    #TODO: DouZero 风格 forward：训练时 return values，采样时 return action index。
    def forward(self, z, x, return_value=False, flags=None):
        tokens = _obs_tokens_torch(z.float(), x.float(), self.position)
        values = self.forward_tokens(tokens)
        if return_value:
            return {"values": values.unsqueeze(-1)}
        exp_epsilon = float(getattr(flags, "exp_epsilon", 0.0) or 0.0)
        if exp_epsilon > 0 and np.random.rand() < exp_epsilon:
            action = torch.randint(values.shape[0], (1,), device=values.device)[0]
        else:
            action = torch.argmax(values, dim=0)
        return {"action": action}


class AttentionDouModel:
    #TODO: 严格复刻 DouZero Model wrapper，内部换成 attention 网络。
    def __init__(self, device=0, hidden_dim=128, num_heads=4, num_layers=2,
                 dropout=0.0, learning_rate=0.0003, weight_decay=0.0,
                 state_dicts=None, metadata=None):
        self.device_key = _device_key(device)
        self.device = _torch_device(device)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.models = {
            position: AttentionPositionModel(
                position,
                self.hidden_dim,
                self.num_heads,
                self.num_layers,
                self.dropout,
            ).to(self.device)
            for position in POSITIONS
        }
        if state_dicts:
            for position in POSITIONS:
                self.models[position].load_state_dict(state_dicts[position])
        self.metadata = metadata or {}
        self.num_updates = int(self.metadata.get("updates", 0) or 0)

    #TODO: DouZero Model.forward 接口。
    def forward(self, position, z, x, training=False, flags=None):
        return self.models[position].forward(z, x, training, flags)

    #TODO: 共享 actor model 参数。
    def share_memory(self):
        for model in self.models.values():
            model.share_memory()

    #TODO: actor 采样使用 eval 模式。
    def eval(self):
        for model in self.models.values():
            model.eval()

    #TODO: learner 使用 train 模式。
    def train(self):
        for model in self.models.values():
            model.train()

    #TODO: 返回某个 position 网络参数。
    def parameters(self, position):
        return self.models[position].parameters()

    #TODO: 返回某个 position 网络。
    def get_model(self, position):
        return self.models[position]

    #TODO: 返回三套 position 网络。
    def get_models(self):
        return self.models

    #TODO: 评测时直接用 token batch 计算 Q-values。
    def q_values(self, position, tokens):
        if tokens is None or len(tokens) == 0:
            return np.zeros(0, dtype=np.float32)
        self.models[position].eval()
        with torch.no_grad():
            tensor = torch.from_numpy(tokens.astype(np.float32)).to(self.device)
            values = self.models[position].forward_tokens(tensor)
        return values.detach().cpu().numpy().astype(np.float32)

    #TODO: epsilon-greedy 选择动作，评测和旧接口共用。
    def select_action(self, position, actions, tokens, epsilon=0.0, rng=None):
        rng = rng or np.random
        if not actions:
            return [], np.zeros((TOKEN_COUNTS[position], TOKEN_DIM), dtype=np.float32)
        if epsilon > 0.0 and rng.random() < epsilon:
            index = int(rng.randrange(len(actions)) if hasattr(rng, "randrange")
                        else rng.randint(len(actions)))
        else:
            values = self.q_values(position, tokens)
            best = np.flatnonzero(values == values.max())
            index = int(np.random.choice(best))
        return list(actions[index]), tokens[index].copy()

    #TODO: 保存 attention checkpoint。
    def save(self, path, optimizer_states=None, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or self.metadata
        payload = {
            "version": 3,
            "algorithm": "attention_dou",
            "target_style": "douzero_dmc",
            "token_dim": TOKEN_DIM,
            "token_counts": TOKEN_COUNTS,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "state_dicts": {
                position: {
                    key: value.detach().cpu().clone()
                    for key, value in self.models[position].state_dict().items()
                }
                for position in POSITIONS
            },
            "optimizer_states": optimizer_states or {},
            "metadata": self.metadata,
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as handle:
            pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

    #TODO: 加载 attention checkpoint，评测默认不需要 optimizer。
    @classmethod
    def load(cls, path, device="cpu", load_optimizer=False):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("algorithm") != "attention_dou":
            raise ValueError(f"not an attention_dou checkpoint: {path}")
        model = cls(
            device=device,
            hidden_dim=payload.get("hidden_dim", 128),
            num_heads=payload.get("num_heads", 4),
            num_layers=payload.get("num_layers", 2),
            dropout=payload.get("dropout", 0.0),
            learning_rate=payload.get("learning_rate", 0.0003),
            weight_decay=payload.get("weight_decay", 0.0),
            state_dicts=payload["state_dicts"],
            metadata=payload.get("metadata", {}),
        )
        if load_optimizer:
            model.metadata["_optimizer_states"] = payload.get("optimizer_states", {})
        return model


def create_optimizers(flags, learner_model):
    #TODO: 复刻 DouZero 三位置 optimizer，只把参数换成 attention 网络参数。
    optimizers = {}
    for position in POSITIONS:
        optimizers[position] = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.optimizer_eps,
            alpha=flags.rmsprop_alpha,
            weight_decay=flags.weight_decay,
        )
    return optimizers


def create_buffers(flags, device_iterator):
    #TODO: 严格复刻 DouZero buffer 字段：obs_x_no_action/obs_action/obs_z/target/done。
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in POSITIONS:
            specs = {
                "done": {"size": (flags.unroll_length,), "dtype": torch.bool},
                "episode_return": {"size": (flags.unroll_length,), "dtype": torch.float32},
                "target": {"size": (flags.unroll_length,), "dtype": torch.float32},
                "obs_x_no_action": {
                    "size": (flags.unroll_length, X_NO_ACTION_DIMS[position]),
                    "dtype": torch.int8,
                },
                "obs_action": {
                    "size": (flags.unroll_length, TOKEN_DIM),
                    "dtype": torch.int8,
                },
                "obs_z": {
                    "size": (flags.unroll_length, 5, 162),
                    "dtype": torch.int8,
                },
            }
            position_buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key, spec in specs.items():
                    tensor = torch.empty(**spec).to(_torch_device(device)).share_memory_()
                    position_buffers[key].append(tensor)
            buffers[device][position] = position_buffers
    return buffers


def get_batch(free_queue, full_queue, buffers, flags, lock):
    #TODO: 复刻 DouZero get_batch：从 full queue 取 B 个 index，stack 后释放 index。
    with lock:
        indices = []
        for _ in range(flags.learn_batch_size):
            index = full_queue.get()
            if index is None:
                raise StopIteration
            indices.append(index)
    batch = {
        key: torch.stack([buffers[key][index] for index in indices], dim=1)
        for key in buffers
    }
    for index in indices:
        free_queue.put(index)
    return batch


def act(i, device, free_queue, full_queue, model, buffers, flags):
    #TODO: 严格复刻 DouZero act：actor 常驻采样，并按 position 写共享 unroll buffer。
    positions = POSITIONS
    try:
        configure_cpu_threads(flags.cpu_threads)
        np.random.seed((flags.seed + i * 1009) % (2**32 - 1))
        env = Environment(Env(flags.objective), device)
        done_buf = {position: [] for position in positions}
        episode_return_buf = {position: [] for position in positions}
        target_buf = {position: [] for position in positions}
        obs_x_no_action_buf = {position: [] for position in positions}
        obs_action_buf = {position: [] for position in positions}
        obs_z_buf = {position: [] for position in positions}
        size = {position: 0 for position in positions}
        position, obs, env_output = env.initial()

        while True:
            while True:
                obs_x_no_action_buf[position].append(env_output["obs_x_no_action"])
                obs_z_buf[position].append(env_output["obs_z"])
                with torch.no_grad():
                    agent_output = model.forward(
                        position, obs["z_batch"], obs["x_batch"], flags=flags
                    )
                action_index = int(agent_output["action"].detach().cpu().numpy())
                action = obs["legal_actions"][action_index]
                obs_action_buf[position].append(_cards2tensor(action))
                size[position] += 1
                position, obs, env_output = env.step(action)
                if env_output["done"]:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff <= 0:
                            continue
                        done_buf[p].extend([False for _ in range(diff - 1)])
                        done_buf[p].append(True)
                        episode_return = (
                            env_output["episode_return"] * flags.reward_scale
                            if p == "landlord"
                            else -env_output["episode_return"] * flags.reward_scale
                        )
                        episode_return_buf[p].extend([0.0 for _ in range(diff - 1)])
                        episode_return_buf[p].append(episode_return)
                        target_buf[p].extend([episode_return for _ in range(diff)])
                    break

            for p in positions:
                while size[p] > flags.unroll_length:
                    index = free_queue[p].get()
                    if index is None:
                        return
                    for t in range(flags.unroll_length):
                        buffers[p]["done"][index][t, ...] = done_buf[p][t]
                        buffers[p]["episode_return"][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]["target"][index][t, ...] = target_buf[p][t]
                        buffers[p]["obs_x_no_action"][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]["obs_action"][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]["obs_z"][index][t, ...] = obs_z_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][flags.unroll_length:]
                    episode_return_buf[p] = episode_return_buf[p][flags.unroll_length:]
                    target_buf[p] = target_buf[p][flags.unroll_length:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][flags.unroll_length:]
                    obs_action_buf[p] = obs_action_buf[p][flags.unroll_length:]
                    obs_z_buf[p] = obs_z_buf[p][flags.unroll_length:]
                    size[p] -= flags.unroll_length
    except KeyboardInterrupt:
        return
    except Exception:
        print(f"Exception in attention_dou actor {i}", flush=True)
        traceback.print_exc()
        raise


def compute_loss(values, targets):
    #TODO: DouZero DMC 的 value regression loss。
    return ((values.squeeze(-1) - targets) ** 2).mean()


def learn(position, actor_models, model, batch, optimizer, flags,
          position_lock, baseline, baseline_lock):
    #TODO: 复刻 DouZero learn：拼回 obs_x，训练 learner，再同步对应 actor model。
    device = _torch_device(flags.device)
    obs_x_no_action = batch["obs_x_no_action"].to(device)
    obs_action = batch["obs_action"].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch["obs_z"].to(device), 0, 1).float()
    target = torch.flatten(batch["target"].to(device), 0, 1)
    episode_returns = batch["episode_return"][batch["done"]]

    with baseline_lock:
        if flags.update_mode == "mc_adv":
            target = target - float(baseline.value(position))

    with position_lock:
        output = model(obs_z, obs_x, return_value=True)
        values = output["values"]
        raw_error = target - values.detach().squeeze(-1)
        loss = compute_loss(values, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())

    done_returns = [float(value) for value in episode_returns.cpu().tolist()]
    with baseline_lock:
        if flags.update_mode == "mc_adv" and done_returns:
            baseline.update({position: done_returns})
    if done_returns:
        MEAN_EPISODE_RETURN_BUF[position].append(
            torch.mean(torch.tensor(done_returns, device=device))
        )
    mean_return = (
        torch.mean(torch.stack(list(MEAN_EPISODE_RETURN_BUF[position]))).item()
        if MEAN_EPISODE_RETURN_BUF[position] else 0.0
    )
    return {
        f"mean_episode_return_{position}": mean_return,
        f"loss_{position}": float(loss.detach().cpu().item()),
        f"abs_error_{position}": float(torch.mean(torch.abs(raw_error)).cpu().item()),
        "done_count": len(done_returns),
        "landlord_wins": (
            sum(1 for value in done_returns if value > 0)
            if position == "landlord" else 0
        ),
    }


def checkpoint_dir(flags):
    return os.path.join(flags.savedir, flags.name)


def checkpoint_path(flags, episode):
    if flags.output:
        return flags.output
    return os.path.join(checkpoint_dir(flags), f"{episode}.pkl")


def episode_from_checkpoint_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem) if stem.isdigit() else 0


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


def _optimizer_states(optimizers):
    #TODO: 保存三位置 optimizer state。
    def to_cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(to_cpu(item) for item in value)
        return value

    return {
        position: to_cpu(optimizer.state_dict())
        for position, optimizer in optimizers.items()
    }


def training_metadata(flags, episode, epsilon, load_path, model, total_steps,
                      elapsed_sec, start_completed, frames, position_frames):
    #TODO: 保存训练配置、frames 和耗时。
    trained_episodes = max(0, episode - start_completed)
    return {
        "algorithm": "attention_dou",
        "target_style": "douzero_dmc",
        "name": flags.name,
        "episodes": episode,
        "frames": frames,
        "position_frames": dict(position_frames),
        "updates": model.num_updates,
        "update_mode": flags.update_mode,
        "objective": flags.objective,
        "reward_scale": flags.reward_scale,
        "reward_shaping": flags.reward_shaping,
        "learning_rate": flags.learning_rate,
        "gamma": flags.gamma,
        "epsilon": epsilon,
        "hidden_dim": flags.hidden_dim,
        "num_heads": flags.num_heads,
        "num_layers": flags.num_layers,
        "dropout": flags.dropout,
        "weight_decay": flags.weight_decay,
        "max_grad_norm": flags.max_grad_norm,
        "num_workers": flags.num_workers,
        "num_threads": flags.num_threads,
        "unroll_length": flags.unroll_length,
        "num_buffers": flags.num_buffers,
        "learn_batch_size": flags.learn_batch_size,
        "baseline_beta": flags.baseline_beta,
        "return_baseline": model.metadata.get("return_baseline", {}),
        "token_dim": TOKEN_DIM,
        "token_counts": TOKEN_COUNTS,
        "actor_device": flags.actor_device,
        "total_steps": total_steps,
        "elapsed_sec": elapsed_sec,
        "episodes_per_sec": trained_episodes / max(1e-6, elapsed_sec),
        "seconds_per_episode": elapsed_sec / max(1, trained_episodes),
        "resumed_from": load_path,
    }


def print_progress_bar(current_episode, start_episode, target_episode, frames,
                       model, recent_steps, recent_wins, recent_errors,
                       start_time):
    #TODO: 打印 attention_dou 训练进度。
    total = max(1, target_episode - start_episode)
    done = max(0, current_episode - start_episode)
    progress = min(1.0, done / float(total))
    width = 30
    filled = int(width * progress)
    elapsed = max(1e-6, time.time() - start_time)
    speed = done / elapsed
    remaining = (total - done) / speed if speed > 0 else 0.0
    avg_steps = sum(recent_steps) / float(len(recent_steps)) if recent_steps else 0.0
    landlord_wp = sum(recent_wins) / float(len(recent_wins)) if recent_wins else 0.0
    avg_error = sum(recent_errors) / float(len(recent_errors)) if recent_errors else 0.0
    line = (
        "\r[{}] {:6.2f}% episode={}/{} global={} updates={} frames={} "
        "landlord_wp={:.3f} avg_steps={:.1f} avg_abs_error={:.4f} "
        "speed={:.2f}eps/s eta={}"
    ).format(
        "#" * filled + "." * (width - filled),
        progress * 100.0,
        done,
        total,
        current_episode,
        model.num_updates,
        frames,
        landlord_wp,
        avg_steps,
        avg_error,
        speed,
        format_duration(remaining),
    )
    print(line + " " * 8, end="", flush=True)


def train(flags):
    #TODO: attention_dou 主训练入口；结构严格对齐 DouZero dmc.py。
    if torch is None:
        raise ImportError("attention_dou requires PyTorch")
    if flags.update_mode not in UPDATE_MODES:
        raise ValueError(f"update_mode must be one of {UPDATE_MODES}")
    if flags.reward_shaping:
        raise ValueError("attention_dou DouZero-style path does not support reward_shaping")
    configure_cpu_threads(flags.cpu_threads)
    os.makedirs(checkpoint_dir(flags), exist_ok=True)
    load_path = flags.load or (latest_checkpoint_path(flags) if flags.resume else None)
    training_device = _device_key(flags.device)
    actor_device = _device_key(flags.actor_device)
    if (training_device != "cpu" or actor_device != "cpu") and not torch.cuda.is_available():
        raise AssertionError("CUDA not available; use DEVICE=cpu ACTOR_DEVICE=cpu")

    device_iterator = ["cpu"] if actor_device == "cpu" else [actor_device]
    ctx = mp.get_context("spawn")
    actor_models = {}
    for device in device_iterator:
        actor_model = AttentionDouModel(
            device=device,
            hidden_dim=flags.hidden_dim,
            num_heads=flags.num_heads,
            num_layers=flags.num_layers,
            dropout=flags.dropout,
            learning_rate=flags.learning_rate,
            weight_decay=flags.weight_decay,
        )
        actor_model.share_memory()
        actor_model.eval()
        actor_models[device] = actor_model

    if load_path:
        learner_model = AttentionDouModel.load(load_path, device=training_device, load_optimizer=True)
        completed = int(learner_model.metadata.get("episodes", 0) or episode_from_checkpoint_path(load_path))
        frames = int(learner_model.metadata.get("frames", 0) or 0)
        position_frames = {
            position: int(learner_model.metadata.get("position_frames", {}).get(position, 0))
            for position in POSITIONS
        }
        epsilon = float(learner_model.metadata.get("epsilon", flags.epsilon))
        print(f"loaded attention_dou from {load_path} episodes={completed} frames={frames}")
    else:
        learner_model = AttentionDouModel(
            device=training_device,
            hidden_dim=flags.hidden_dim,
            num_heads=flags.num_heads,
            num_layers=flags.num_layers,
            dropout=flags.dropout,
            learning_rate=flags.learning_rate,
            weight_decay=flags.weight_decay,
        )
        completed = 0
        frames = 0
        position_frames = {position: 0 for position in POSITIONS}
        epsilon = flags.epsilon

    flags.exp_epsilon = epsilon
    optimizers = create_optimizers(flags, learner_model)
    optimizer_states = learner_model.metadata.pop("_optimizer_states", {})
    for position in POSITIONS:
        if position in optimizer_states:
            optimizers[position].load_state_dict(optimizer_states[position])
        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(
                learner_model.get_model(position).state_dict()
            )

    baseline = ReturnBaseline(
        learner_model.metadata.get("return_baseline", {}),
        flags.baseline_beta,
    )
    buffers = create_buffers(flags, device_iterator)
    free_queue = {}
    full_queue = {}
    for device in device_iterator:
        free_queue[device] = {
            position: ctx.SimpleQueue() for position in POSITIONS
        }
        full_queue[device] = {
            position: ctx.SimpleQueue() for position in POSITIONS
        }
        for index in range(flags.num_buffers):
            for position in POSITIONS:
                free_queue[device][position].put(index)

    actor_processes = []
    for device in device_iterator:
        for actor_id in range(flags.num_workers):
            process = ctx.Process(
                target=act,
                args=(
                    actor_id,
                    device,
                    free_queue[device],
                    full_queue[device],
                    actor_models[device],
                    buffers[device],
                    flags,
                ),
            )
            process.start()
            actor_processes.append(process)

    stats = {
        f"mean_episode_return_{position}": 0.0 for position in POSITIONS
    }
    stats.update({f"loss_{position}": 0.0 for position in POSITIONS})
    stats.update({f"abs_error_{position}": 0.0 for position in POSITIONS})
    target_total = completed + flags.episodes
    start_completed = completed
    provided_options = getattr(flags, "_provided_options", set())
    if "--save_interval" not in provided_options:
        flags.save_interval = max(1, int(flags.episodes) // 20)
    start_time = time.time()
    stop_event = threading.Event()
    global_lock = threading.Lock()
    baseline_lock = threading.Lock()
    queue_locks = {
        device: {position: threading.Lock() for position in POSITIONS}
        for device in device_iterator
    }
    position_locks = {position: threading.Lock() for position in POSITIONS}
    recent_wins = deque(maxlen=max(1, flags.log_interval))
    recent_steps = deque(maxlen=max(1, flags.log_interval))
    recent_errors = deque(maxlen=max(1, flags.log_interval))
    next_log = completed + max(1, flags.log_interval)
    next_progress = completed + max(1, flags.progress_interval)
    next_save = completed + flags.save_interval if flags.save_interval else None
    saved_episodes = set()

    print(
        "AttentionDou(DouZero-style) mode={} episodes={} workers={} threads={} "
        "unroll={} batch={} buffers={} device={} actor_device={} hidden={} "
        "heads={} layers={} baseline={}".format(
            flags.update_mode,
            flags.episodes,
            flags.num_workers,
            flags.num_threads,
            flags.unroll_length,
            flags.learn_batch_size,
            flags.num_buffers,
            _torch_device(training_device),
            _torch_device(actor_device),
            flags.hidden_dim,
            flags.num_heads,
            flags.num_layers,
            baseline.values if flags.update_mode == "mc_adv" else "off",
        ),
        flush=True,
    )

    def save_checkpoint(episode):
        if episode in saved_episodes:
            return
        elapsed = max(1e-6, time.time() - start_time)
        learner_model.metadata["return_baseline"] = dict(baseline.values)
        learner_model.save(
            checkpoint_path(flags, episode),
            optimizer_states=_optimizer_states(optimizers),
            metadata=training_metadata(
                flags,
                episode,
                epsilon,
                load_path,
                learner_model,
                frames,
                elapsed,
                start_completed,
                frames,
                position_frames,
            ),
        )
        saved_episodes.add(episode)
        print(f"saved attention_dou to {checkpoint_path(flags, episode)}", flush=True)

    def batch_and_learn(device, position, local_lock):
        nonlocal completed, frames, position_frames, next_log, next_progress, next_save
        while not stop_event.is_set():
            with global_lock:
                if completed >= target_total:
                    return
            try:
                batch = get_batch(
                    free_queue[device][position],
                    full_queue[device][position],
                    buffers[device][position],
                    flags,
                    local_lock,
                )
            except StopIteration:
                return
            output_stats = learn(
                position,
                actor_models,
                learner_model.get_model(position),
                batch,
                optimizers[position],
                flags,
                position_locks[position],
                baseline,
                baseline_lock,
            )
            with global_lock:
                for key, value in output_stats.items():
                    if key in stats:
                        stats[key] = value
                frames += flags.unroll_length * flags.learn_batch_size
                position_frames[position] += flags.unroll_length * flags.learn_batch_size
                learner_model.num_updates += flags.unroll_length * flags.learn_batch_size
                recent_errors.append(output_stats.get(f"abs_error_{position}", 0.0))
                if position == "landlord" and output_stats["done_count"] > 0:
                    done_count = output_stats["done_count"]
                    completed += done_count
                    recent_wins.append(output_stats["landlord_wins"] / max(1, done_count))
                    recent_steps.append(frames / max(1, completed))

    threads = []
    for device in device_iterator:
        for _ in range(flags.num_threads):
            for position in POSITIONS:
                thread = threading.Thread(
                    target=batch_and_learn,
                    args=(device, position, queue_locks[device][position]),
                    daemon=True,
                )
                thread.start()
                threads.append(thread)

    try:
        while completed < target_total:
            time.sleep(5)
            with global_lock:
                should_progress = (
                    flags.progress_interval
                    and (completed >= next_progress or completed >= target_total)
                )
                should_log = bool(flags.log_interval and completed >= next_log)
                save_targets = []
                while next_save and completed >= next_save:
                    save_targets.append(next_save)
                    next_save += flags.save_interval
                current_episode = completed
                current_frames = frames
                current_position_frames = dict(position_frames)
            if should_progress:
                print_progress_bar(
                    current_episode,
                    start_completed,
                    target_total,
                    current_frames,
                    learner_model,
                    recent_steps,
                    recent_wins,
                    recent_errors,
                    start_time,
                )
                with global_lock:
                    while next_progress <= completed:
                        next_progress += max(1, flags.progress_interval)
            if should_log:
                if flags.progress_interval:
                    print()
                elapsed = max(1e-6, time.time() - start_time)
                speed = (current_episode - start_completed) / elapsed
                print(
                    "episode={} frames={} position_frames={} updates={} "
                    "landlord_wp={:.3f} avg_steps={:.1f} avg_abs_error={:.4f} "
                    "losses=({:.4f},{:.4f},{:.4f}) baseline={} "
                    "elapsed_sec={:.1f} speed={:.2f}eps/s".format(
                        current_episode,
                        current_frames,
                        current_position_frames,
                        learner_model.num_updates,
                        sum(recent_wins) / max(1, len(recent_wins)),
                        sum(recent_steps) / max(1, len(recent_steps)),
                        sum(recent_errors) / max(1, len(recent_errors)),
                        stats["loss_landlord"],
                        stats["loss_landlord_up"],
                        stats["loss_landlord_down"],
                        baseline.values if flags.update_mode == "mc_adv" else "off",
                        elapsed,
                        speed,
                    ),
                    flush=True,
                )
                with global_lock:
                    next_log += max(1, flags.log_interval)
            if save_targets:
                if flags.progress_interval and not should_log:
                    print()
                for save_episode in save_targets:
                    save_checkpoint(save_episode)
    finally:
        stop_event.set()
        for device in device_iterator:
            for position in POSITIONS:
                for _ in range(flags.num_threads * flags.learn_batch_size + 4):
                    try:
                        full_queue[device][position].put(None)
                    except Exception:
                        pass
                for _ in actor_processes:
                    try:
                        free_queue[device][position].put(None)
                    except Exception:
                        pass
        for process in actor_processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)

    if flags.progress_interval:
        print()
    save_checkpoint(target_total)
    return learner_model


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Attention Q-learning on DouZero DMC buffers")
    parser.add_argument("--episodes", default=10000, type=int)
    parser.add_argument("--name", default="attention_dou_logadp", type=str)
    parser.add_argument("--objective", default="logadp", choices=["wp", "adp", "logadp"])
    parser.add_argument("--reward_scale", default=1.0, type=float)
    parser.add_argument("--reward_shaping", action="store_true")
    parser.add_argument("--savedir", default=DEFAULT_ATTENTION_DOU_DIR)
    parser.add_argument("--output", default="")
    parser.add_argument("--load", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--epsilon", default=0.1, type=float)
    parser.add_argument("--min_epsilon", default=0.02, type=float)
    parser.add_argument("--epsilon_decay", default=0.99998, type=float)
    parser.add_argument("--weight_decay", default=0.00001, type=float)
    parser.add_argument("--max_grad_norm", default=10.0, type=float)
    parser.add_argument("--rmsprop_alpha", default=0.99, type=float)
    parser.add_argument("--momentum", default=0.0, type=float)
    parser.add_argument("--optimizer_eps", default=0.00001, type=float)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--actor_device", default="auto")
    parser.add_argument("--max_steps", default=1000, type=int)
    parser.add_argument("--log_interval", default=10000, type=int)
    parser.add_argument("--progress_interval", default=5000, type=int)
    parser.add_argument("--save_interval", default=50000, type=int)
    parser.add_argument("--update_mode", default="mc_adv", choices=UPDATE_MODES)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--num_threads", default=1, type=int)
    parser.add_argument("--cpu_threads", default=1, type=int)
    parser.add_argument("--unroll_length", default=20, type=int)
    parser.add_argument("--num_buffers", default=64, type=int)
    parser.add_argument("--buffer_size", default=0, type=int)
    parser.add_argument("--learn_batch_size", default=4, type=int)
    parser.add_argument("--learn_steps", default=1, type=int)
    parser.add_argument("--baseline_beta", default=0.01, type=float)
    return parser
