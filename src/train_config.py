from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APPROXQ_SAVEDIR = "approx_qlearning_checkpoints/approx_qlearning"
APPROXQ_1M_HISTORY_DIR = Path(APPROXQ_SAVEDIR) / "approxq_logadp_cmp_1m_history"
APPROXQ_1M_HISTORY_BEST_LANDLORD = APPROXQ_1M_HISTORY_DIR / "750000.pkl"


@dataclass(frozen=True)
class ApproxQTrainConfig:
    name: str
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


TRAIN_CONFIGS = {
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


def get_train_config(name: str) -> ApproxQTrainConfig:
    try:
        return TRAIN_CONFIGS[name]
    except KeyError as exc:
        available = ", ".join(sorted(TRAIN_CONFIGS))
        raise KeyError(
            "Unknown train config '{}'. Available: {}".format(name, available)
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
    for field_name in ApproxQTrainConfig.__dataclass_fields__:
        setattr(args, field_name, getattr(config, field_name))
    return args


def config_summary(config: ApproxQTrainConfig) -> str:
    return (
        "{} -> episodes={}, alpha={}, gamma={}, objective={}, "
        "reward_scale={}, reward_shaping={}, device={}, feature_mode={}, "
        "max_candidate_actions={}, log_interval={}, progress_interval={}, "
        "save_interval={}".format(
            config.name,
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
