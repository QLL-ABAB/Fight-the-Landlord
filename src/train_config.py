from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APPROXQ_SAVEDIR = "approx_qlearning_checkpoints/approx_qlearning"
BETTER_APPROXQ_SAVEDIR = "approx_qlearning_checkpoints/better_approxq"
PRECISE_APPROXQ_SAVEDIR = "approx_qlearning_checkpoints/approxq_precise"
APPROX_DOUFEATURE_SAVEDIR = "approx_qlearning_checkpoints/approx_doufeature"
ATTENTION_DOU_SAVEDIR = "attention_dou_checkpoints/attention_dou"
APPROXQ_1M_HISTORY_DIR = Path(APPROXQ_SAVEDIR) / "approxq_logadp_cmp_1m_history"
APPROXQ_1M_HISTORY_BEST_LANDLORD = APPROXQ_1M_HISTORY_DIR / "750000.pkl"


@dataclass(frozen=True)
class ApproxQTrainConfig:
    name: str
    algorithm: str = "approxq"
    episodes: int = 1000000
    savedir: str = APPROXQ_SAVEDIR
    output: str = ""
    load: str = ""
    resume: bool = False
    seed: int = 0

    alpha: float = 0.05
    gamma: float = 0.98
    epsilon: float = 0.1
    min_epsilon: float = 0.02
    epsilon_decay: float = 0.99998
    l2: float = 0.00001
    clip_td: float = 10.0

    objective: str = "logadp"
    reward_scale: float = 1.0
    reward_shaping: bool = False

    device: str = "cpu"
    feature_mode: str = "history"
    max_candidate_actions: int = 64
    max_steps: int = 1000
    log_interval: int = 10000
    progress_interval: int = 500
    save_interval: int = 50000
    feature_diag: bool = False
    feature_diag_path: str = ""
    feature_diag_interval: int = 0
    feature_diag_topk: int = 0


@dataclass(frozen=True)
class ApproxDouFeatureTrainConfig(ApproxQTrainConfig):
    algorithm: str = "approx_doufeature"
    episodes: int = 100000
    savedir: str = APPROX_DOUFEATURE_SAVEDIR
    device: str = "auto"
    feature_mode: str = "douzero"
    max_candidate_actions: int = 0
    log_interval: int = 1000
    progress_interval: int = 500

    update_mode: str = "td"
    num_workers: int = 4
    worker_episodes: int = 8
    buffer_size: int = 0
    learn_batch_size: int = 4096
    learn_steps: int = 1
    baseline_beta: float = 0.01
    diag_topk: int = 20


@dataclass(frozen=True)
class BetterApproxQTrainConfig(ApproxQTrainConfig):
    algorithm: str = "better_approxq"
    savedir: str = BETTER_APPROXQ_SAVEDIR
    feature_mode: str = "better_history"
    alpha: float = 0.006
    gamma: float = 1.0
    l2: float = 0.00003
    clip_td: float = 5.0


@dataclass(frozen=True)
class PreciseApproxQTrainConfig(BetterApproxQTrainConfig):
    algorithm: str = "approxq_precise"
    savedir: str = PRECISE_APPROXQ_SAVEDIR
    feature_mode: str = "precise_history_full"


@dataclass(frozen=True)
class AttentionDouTrainConfig(ApproxQTrainConfig):
    algorithm: str = "attention_dou"
    episodes: int = 100000
    savedir: str = ATTENTION_DOU_SAVEDIR
    device: str = "auto"
    actor_device: str = "auto"
    feature_mode: str = "attention_dou"
    max_candidate_actions: int = 0
    log_interval: int = 1000
    progress_interval: int = 500

    update_mode: str = "td"
    learning_rate: float = 0.0003
    weight_decay: float = 0.00001
    max_grad_norm: float = 10.0
    rmsprop_alpha: float = 0.99
    momentum: float = 0.0
    optimizer_eps: float = 0.00001
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    num_workers: int = 4
    num_threads: int = 1
    unroll_length: int = 20
    num_buffers: int = 64
    buffer_size: int = 0
    learn_batch_size: int = 4
    learn_steps: int = 1
    baseline_beta: float = 0.01


APPROXQ_TRAIN_CONFIGS = {
    "approxq_logadp_cmp_1m_history": ApproxQTrainConfig(
        name="approxq_logadp_cmp_1m_history",
        alpha=0.05,
        gamma=0.98,
        reward_shaping=False,
        progress_interval=500,
    ),
    "approxq_logadp_cmp_10k_history": ApproxQTrainConfig(
        name="approxq_logadp_cmp_10k_history",
        alpha=0.02,
        gamma=0.98,
        reward_shaping=False,
        progress_interval=5000,
    ),
    "approxq_logadp_stable_history": ApproxQTrainConfig(
        name="approxq_logadp_stable_history",
        alpha=0.01,
        gamma=0.95,
        reward_shaping=True,
        progress_interval=500,
    ),
    "approxq_logadp_best_landlord_finetune_50k": ApproxQTrainConfig(
        name="approxq_logadp_best_landlord_finetune_50k",
        episodes=50000,
        load=str(APPROXQ_1M_HISTORY_BEST_LANDLORD),
        alpha=0.01,
        gamma=0.95,
        reward_shaping=True,
        progress_interval=500,
        log_interval=2500,
        save_interval=2500,
    ),
    "approxq_logadp_best_landlord_finetune_50k_time_equal": ApproxQTrainConfig(
        name="approxq_logadp_best_landlord_finetune_50k_time_equal",
        episodes=50000,
        load=str(APPROXQ_1M_HISTORY_BEST_LANDLORD),
        alpha=0.01,
        gamma=1,
        reward_shaping=True,
        progress_interval=500,
        log_interval=2500,
        save_interval=2500,
    ),
}


BETTER_APPROXQ_TRAIN_CONFIGS = {
    "better_approxq_logadp_10k": BetterApproxQTrainConfig(
        name="better_approxq_logadp_10k",
        episodes=10000,
        reward_shaping=True,
        log_interval=1000,
        progress_interval=500,
        save_interval=5000,
        feature_diag=True,
        feature_diag_interval=1000,
    ),
    "better_approxq_logadp_finetune_50k_time_equal": BetterApproxQTrainConfig(
        name="better_approxq_logadp_finetune_50k_time_equal",
        episodes=50000,
        reward_shaping=True,
        log_interval=2500,
        progress_interval=500,
        save_interval=2500,
        feature_diag=True,
        feature_diag_interval=2500,
    ),
    "better_approxq_logadp_finetune_50k_time_equal_fast": BetterApproxQTrainConfig(
        name="better_approxq_logadp_finetune_50k_time_equal_fast",
        episodes=50000,
        reward_shaping=True,
        max_candidate_actions=32,
        log_interval=5000,
        progress_interval=1000,
        save_interval=10000,
        feature_diag=False,
    ),
    "better_approxq_logadp_finetune_50k_time_equal_lr_1e-1": BetterApproxQTrainConfig(
        name="better_approxq_logadp_finetune_50k_time_equal_lr_1e-1",
        episodes=50000,
        alpha=0.0006,
        reward_shaping=True,
        log_interval=2500,
        progress_interval=500,
        save_interval=2500,
        feature_diag=True,
        feature_diag_interval=2500,
    ),
    "better_approxq_logadp_finetune_50k_time_equal_lr_1e-2": BetterApproxQTrainConfig(
        name="better_approxq_logadp_finetune_50k_time_equal_lr_1e-2",
        episodes=50000,
        alpha=0.00006,
        reward_shaping=True,
        log_interval=2500,
        progress_interval=500,
        save_interval=2500,
        feature_diag=True,
        feature_diag_interval=2500,
    ),
}


PRECISE_APPROXQ_TRAIN_CONFIGS = {
    "approxq_precise_full_50k_warm": PreciseApproxQTrainConfig(
        name="approxq_precise_full_50k_warm",
        episodes=50000,
        load="approx_qlearning_checkpoints/better_approxq/better_approxq_full_50k_lr_1e-1/50000.pkl",
        alpha=0.0006,
        reward_shaping=True,
        log_interval=2500,
        progress_interval=500,
        save_interval=2500,
        feature_diag=True,
        feature_diag_interval=2500,
    ),
    "approxq_precise_full_50k_warm_fast": PreciseApproxQTrainConfig(
        name="approxq_precise_full_50k_warm_fast",
        episodes=50000,
        load="approx_qlearning_checkpoints/better_approxq/better_approxq_full_50k_lr_1e-1/50000.pkl",
        alpha=0.0006,
        reward_shaping=True,
        max_candidate_actions=32,
        log_interval=5000,
        progress_interval=1000,
        save_interval=10000,
        feature_diag=False,
    ),
    "approxq_precise_full_50k_scratch": PreciseApproxQTrainConfig(
        name="approxq_precise_full_50k_scratch",
        episodes=50000,
        alpha=0.0006,
        reward_shaping=True,
        log_interval=2500,
        progress_interval=500,
        save_interval=2500,
        feature_diag=True,
        feature_diag_interval=2500,
    ),
}


APPROX_DOUFEATURE_TRAIN_CONFIGS = {
    "approx_doufeature_logadp_td_1m_gamma": ApproxDouFeatureTrainConfig(
        name="approx_doufeature_logadp_td_1m_gamma",
        update_mode="td",
        episodes=1000000,
        alpha=0.01,
        gamma=0.98,
        reward_shaping=False,
        num_workers=4,
        log_interval=10000,
        progress_interval=5000,
        save_interval=50000,
    ),
    "approx_doufeature_logadp_td_1m": ApproxDouFeatureTrainConfig(
        name="approx_doufeature_logadp_td_1m",
        update_mode="td",
        episodes=1000000,
        alpha=0.01,
        gamma=1,
        reward_shaping=False,
        num_workers=4,
        log_interval=10000,
        progress_interval=5000,
        save_interval=50000,
    ),
    "approx_doufeature_logadp_td_buffer_1m": ApproxDouFeatureTrainConfig(
        name="approx_doufeature_logadp_td_buffer_1m",
        update_mode="td",
        episodes=1000000,
        alpha=0.003,
        gamma=1,
        reward_shaping=False,
        num_workers=4,
        buffer_size=200000,
        learn_batch_size=4096,
        learn_steps=10,
        log_interval=10000,
        progress_interval=5000,
        save_interval=50000,
    ),
    "approx_doufeature_logadp_mc_1m": ApproxDouFeatureTrainConfig(
        name="approx_doufeature_logadp_mc_1m",
        update_mode="mc",
        episodes=1000000,
        alpha=0.01,
        gamma=1,
        reward_shaping=False,
        num_workers=4,
        log_interval=10000,
        progress_interval=5000,
        save_interval=50000,
    ),
    "approx_doufeature_logadp_mc_adv_buffer_1m": ApproxDouFeatureTrainConfig(
        name="approx_doufeature_logadp_mc_adv_buffer_1m",
        update_mode="mc_adv",
        episodes=1000000,
        alpha=0.001,
        gamma=1,
        reward_shaping=False,
        num_workers=4,
        buffer_size=200000,
        learn_batch_size=4096,
        learn_steps=10,
        baseline_beta=0.01,
        log_interval=10000,
        progress_interval=5000,
        save_interval=50000,
    ),
}


ATTENTION_DOU_TRAIN_CONFIGS = {
    "attention_dou_logadp_td_1m": AttentionDouTrainConfig(
        name="attention_dou_logadp_td_1m",
        update_mode="td",
        episodes=1000000,
        learning_rate=0.0003,
        gamma=1,
        reward_shaping=False,
        num_workers=4,
        num_threads=1,
        unroll_length=20,
        num_buffers=64,
        learn_batch_size=4,
        learn_steps=1,
        log_interval=10000,
        progress_interval=5000,
        save_interval=50000,
    ),
    "attention_dou_logadp_td_buffer_1m": AttentionDouTrainConfig(
        name="attention_dou_logadp_td_buffer_1m",
        update_mode="td",
        episodes=1000000,
        learning_rate=0.0001,
        gamma=1,
        reward_shaping=False,
        num_workers=4,
        num_threads=1,
        unroll_length=20,
        num_buffers=64,
        buffer_size=0,
        learn_batch_size=4,
        learn_steps=1,
        log_interval=10000,
        progress_interval=5000,
        save_interval=50000,
    ),
    "attention_dou_logadp_mc_adv_buffer_1m": AttentionDouTrainConfig(
        name="attention_dou_logadp_mc_adv_buffer_1m",
        update_mode="mc_adv",
        episodes=1000000,
        learning_rate=0.0001,
        gamma=1,
        reward_shaping=False,
        num_workers=4,
        num_threads=1,
        unroll_length=20,
        num_buffers=64,
        buffer_size=0,
        learn_batch_size=4,
        learn_steps=1,
        baseline_beta=0.01,
        log_interval=10000,
        progress_interval=5000,
        save_interval=50000,
    ),
}


TRAIN_CONFIGS = {
    **APPROXQ_TRAIN_CONFIGS,
    **BETTER_APPROXQ_TRAIN_CONFIGS,
    **PRECISE_APPROXQ_TRAIN_CONFIGS,
    **APPROX_DOUFEATURE_TRAIN_CONFIGS,
    **ATTENTION_DOU_TRAIN_CONFIGS,
}


def train_configs_for_algorithm(algorithm: str):
    return {
        name: config
        for name, config in TRAIN_CONFIGS.items()
        if config.algorithm == algorithm
    }


def get_train_config(name: str, algorithm: str | None = None) -> ApproxQTrainConfig:
    configs = TRAIN_CONFIGS if algorithm is None else train_configs_for_algorithm(algorithm)
    try:
        return configs[name]
    except KeyError as exc:
        available = ", ".join(sorted(configs))
        scope = f" for {algorithm}" if algorithm else ""
        raise KeyError(
            "Unknown train config '{}'{}. Available: {}".format(name, scope, available)
        ) from exc


def override_train_config(
    config: ApproxQTrainConfig,
    **overrides,
) -> ApproxQTrainConfig:
    clean_overrides = {
        key: value
        for key, value in overrides.items()
        if value is not None
    }
    return replace(config, **clean_overrides)


def apply_train_config_to_args(args, config: ApproxQTrainConfig):
    for field_name in type(config).__dataclass_fields__:
        setattr(args, field_name, getattr(config, field_name))
    return args


def config_summary(config: ApproxQTrainConfig) -> str:
    if isinstance(config, AttentionDouTrainConfig):
        common = (
            "{} [{}] -> episodes={}, gamma={}, objective={}, "
            "reward_scale={}, reward_shaping={}, device={}, actor_device={}, log_interval={}, "
            "progress_interval={}, save_interval={}".format(
                config.name,
                config.algorithm,
                config.episodes,
                config.gamma,
                config.objective,
                config.reward_scale,
                config.reward_shaping,
                config.device,
                config.actor_device,
                config.log_interval,
                config.progress_interval,
                config.save_interval,
            )
        )
    else:
        common = (
            "{} [{}] -> episodes={}, alpha={}, gamma={}, objective={}, "
            "reward_scale={}, reward_shaping={}, device={}, feature_mode={}, "
            "max_candidate_actions={}, log_interval={}, progress_interval={}, "
            "save_interval={}".format(
                config.name,
                config.algorithm,
                config.episodes,
                config.alpha,
                config.gamma,
                config.objective,
                config.reward_scale,
                config.reward_shaping,
                config.device,
                config.feature_mode,
                config.max_candidate_actions,
                config.log_interval,
                config.progress_interval,
                config.save_interval,
            )
        )
    if isinstance(config, AttentionDouTrainConfig):
        common += (
            ", update_mode={}, learning_rate={}, hidden_dim={}, "
            "num_heads={}, num_layers={}, dropout={}, num_workers={}, num_threads={}, "
            "unroll_length={}, num_buffers={}, "
            "buffer_size={}, learn_batch_size={}, "
            "learn_steps={}, baseline_beta={}".format(
                config.update_mode,
                config.learning_rate,
                config.hidden_dim,
                config.num_heads,
                config.num_layers,
                config.dropout,
                config.num_workers,
                config.num_threads,
                config.unroll_length,
                config.num_buffers,
                config.buffer_size,
                config.learn_batch_size,
                config.learn_steps,
                config.baseline_beta,
            )
        )
    elif isinstance(config, ApproxDouFeatureTrainConfig):
        common += (
            ", update_mode={}, num_workers={}, worker_episodes={}, "
            "buffer_size={}, learn_batch_size={}, learn_steps={}, "
            "baseline_beta={}, diag_topk={}".format(
                config.update_mode,
                config.num_workers,
                config.worker_episodes,
                config.buffer_size,
                config.learn_batch_size,
                config.learn_steps,
                config.baseline_beta,
                config.diag_topk,
            )
        )
    return common
