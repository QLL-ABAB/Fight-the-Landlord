from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DATA = REPO_ROOT / "eval_data.pkl"
EVALUATE_RESULTS_DIR = REPO_ROOT / "evaluate_results"
RUN_LOGS_DIR = REPO_ROOT / "run_logs"

APPROXQ_1M_HISTORY = (
    REPO_ROOT
    / "approx_qlearning_checkpoints"
    / "approx_qlearning"
    / "approxq_logadp_cmp_1m_history"
    / "1000000.pkl"
)
DOUZERO_LOGADP_60M_LANDLORD = (
    REPO_ROOT
    / "base"
    / "douzero_checkpoints"
    / "douzero_logadp_cmp_60000000"
    / "landlord_weights_60000000.ckpt"
)


@dataclass(frozen=True)
class EvalConfig:
    name: str
    methods: tuple[str, str, str]
    eval_mode: str = "rotate"
    eval_data: str = str(EVAL_DATA)
    num_workers: int = 5
    assignment_workers: int = 1
    result_dir: str = str(EVALUATE_RESULTS_DIR)
    gpu_device: str = ""


@dataclass(frozen=True)
class AgentSpec:
    name: str
    aliases: tuple[str, ...]
    factory: Callable[[str, str | None], object]
    needs_position: bool = True
    description: str = ""

    def matches(self, method: str) -> bool:
        prefix = method.split(":", 1)[0]
        return prefix in (self.name, *self.aliases)


def method_kind(method: str) -> str:
    if method.endswith(".ckpt"):
        return "douzero"
    return method.split(":", 1)[0]


def method_arg(method: str) -> str | None:
    if method.endswith(".ckpt"):
        return method
    if ":" not in method:
        return None
    return method.split(":", 1)[1]


def with_prefix(kind: str, path: str | Path) -> str:
    return "{}:{}".format(kind, path)


def normalized_method(method: str) -> str:
    if method.endswith(".ckpt"):
        return with_prefix("douzero", method)
    return method


def make_agent(method: str, position: str):
    kind = method_kind(method)
    arg = method_arg(method)
    spec = AGENT_SPECS_BY_NAME.get(kind)
    if spec is None:
        spec = AGENT_SPECS_BY_NAME["douzero"]
        arg = method

    if spec.needs_position:
        return spec.factory(position, arg)
    return spec.factory(position, arg)


def _rlcard(position: str, _: str | None):
    from douzero.evaluation.rlcard_agent import RLCardAgent

    return RLCardAgent(position)


def _random(_: str, __: str | None):
    from douzero.evaluation.random_agent import RandomAgent

    return RandomAgent()


def _heuristic(position: str, _: str | None):
    from douzero.evaluation.heuristic_agent import HeuristicAgent

    return HeuristicAgent(position)


def _value(position: str, _: str | None):
    from douzero.evaluation.valueiteration_agent import ValueDPAgent

    return ValueDPAgent(position)


def _probability(position: str, _: str | None):
    from douzero.evaluation.probabilistic_response_agent import (
        ProbabilisticResponseAgent,
    )

    return ProbabilisticResponseAgent(position)


def _adversarial(position: str, arg: str | None):
    from douzero.evaluation.adversarial_agent import AdversarialSearchAgent

    num_samples = int(arg) if arg and arg.isdigit() else 800  # 默认50
    agent = AdversarialSearchAgent(position)
    agent.cfg["num_samples"] = num_samples  # 设置采样数
    return agent


def _adversarial_mc(position: str, arg: str | None):
    from douzero.evaluation.adversarial_mc_agent import AdversarialSearchAgent

    agent = AdversarialSearchAgent(position)
    if arg:
        parts = [x.strip() for x in arg.split(":") if x.strip()]
        if len(parts) >= 1 and parts[0].isdigit():
            agent.cfg["num_samples"] = int(parts[0])
        if len(parts) >= 2 and parts[1].isdigit():
            agent.cfg["search_depth"] = int(parts[1])
        if len(parts) >= 3:
            try:
                agent.cfg["time_budget_sec"] = float(parts[2])
            except ValueError:
                pass
    return agent


def _adversarial_q(position: str, arg: str | None):
    from douzero.evaluation.adversarial_q_agent import AdversarialQSearchAgent

    model_path = None
    num_samples = None
    search_depth = None
    time_budget = None
    q_leaf_mix = None
    q_leaf_scale = None

    if arg:
        parts = [x.strip() for x in arg.split(":") if x.strip()]
        if len(parts) >= 2 and len(parts[0]) == 1 and (
            parts[1].startswith("\\") or parts[1].startswith("/")
        ):
            parts = [parts[0] + ":" + parts[1]] + parts[2:]
        if parts:
            model_path = parts[0]
        if len(parts) >= 2 and parts[1].isdigit():
            num_samples = int(parts[1])
        if len(parts) >= 3 and parts[2].isdigit():
            search_depth = int(parts[2])
        if len(parts) >= 4:
            try:
                time_budget = float(parts[3])
            except ValueError:
                pass
        if len(parts) >= 5:
            try:
                q_leaf_mix = float(parts[4])
            except ValueError:
                pass
        if len(parts) >= 6:
            try:
                q_leaf_scale = float(parts[5])
            except ValueError:
                pass

    agent = AdversarialQSearchAgent(position, model_path=model_path)
    if num_samples is not None:
        agent.cfg["num_samples"] = num_samples
    if search_depth is not None:
        agent.cfg["search_depth"] = search_depth
    if time_budget is not None:
        agent.cfg["time_budget_sec"] = time_budget
    if q_leaf_mix is not None:
        agent.cfg["q_leaf_mix"] = q_leaf_mix
    if q_leaf_scale is not None:
        agent.cfg["q_leaf_scale"] = q_leaf_scale
    return agent


def _qlearning(position: str, model_path: str | None):
    from douzero.evaluation.qlearning_agent import QLearningAgent

    return QLearningAgent(position, model_path)


def _approxq(position: str, model_path: str | None):
    from douzero.evaluation.approx_qlearning_agent import ApproxQLearningAgent

    return ApproxQLearningAgent(position, model_path)


#TODO: 注册 DouZero 原始特征版线性 ApproxQ，支持 TD/MC 训练产物评测。
def _approx_doufeature(position: str, model_path: str | None):
    from douzero.evaluation.approx_doufeature_agent import ApproxDouFeatureAgent

    return ApproxDouFeatureAgent(position, model_path)


def _attention_dou(position: str, model_path: str | None):
    from douzero.evaluation.attention_dou_agent import AttentionDouAgent

    return AttentionDouAgent(position, model_path)


def _search(_: str, arg: str | None):
    from douzero.evaluation.search_agent import SearchAgent

    return SearchAgent(int(arg) if arg and arg.isdigit() else 30)


def _expectimax(_: str, arg: str | None):
    from douzero.evaluation.expectimax_agent import ExpectimaxAgent

    if not arg:
        return ExpectimaxAgent()
    parts = arg.split(":")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return ExpectimaxAgent(int(parts[0]), int(parts[1]))
    if len(parts) == 1 and parts[0].isdigit():
        return ExpectimaxAgent(int(parts[0]))
    return ExpectimaxAgent()

def _nnpolicy(position: str, arg: str | None):
    from douzero.evaluation.nn_policy_agent import NeuralPolicyAgent

    # 使用方式：
    # nnpolicy:selfplay_policy_weights/weights.json
    weights_path = arg if arg else None

    return NeuralPolicyAgent(
        position=position,
        weights_path=weights_path,
        concrete=False,
    )

def _montecarlo(position: str, arg: str | None):
    from douzero.evaluation.high_rank_montecarlo_agent import HighRankMonteCarloAgent

    return HighRankMonteCarloAgent(position)


def _douzero(position: str, model_path: str | None):
    from douzero.evaluation.deep_agent import DeepAgent
    from douzero.evaluation.simulation import resolve_douzero_model_path

    if not model_path:
        raise ValueError("DouZero method requires douzero:/path/to/checkpoint.ckpt")
    return DeepAgent(position, resolve_douzero_model_path(position, model_path))


AGENT_SPECS = (
    AgentSpec("rlcard", (), _rlcard, description="RLCard rule-based agent"),
    AgentSpec("random", (), _random, needs_position=False, description="Random agent"),
    AgentSpec("heuristic", (), _heuristic, description="Hand-crafted heuristic agent"),
    AgentSpec("value", ("mdp",), _value, description="Memoized value-DP agent"),
    AgentSpec(
        "probability",
        ("prob",),
        _probability,
        description="Probabilistic response agent",
    ),
    AgentSpec(
        "adversarial",
        ("adv",),
        _adversarial,
        description="Bayesian sampled adversarial-search agent",
    ),
    AgentSpec(
        "adversarial_mc",
        ("advmc", "adv_mc"),
        _adversarial_mc,
        description="Adversarial search with integrated high-rank Monte Carlo leaf evaluator",
    ),
    AgentSpec(
        "adversarial_q",
        ("advq", "adv_q"),
        _adversarial_q,
        description="Adversarial search with attention_dou Q-network leaf evaluator",
    ),
    AgentSpec("qlearning", (), _qlearning, description="Tabular Q-learning agent"),
    AgentSpec(
        "approxq",
        ("approx_qlearning",),
        _approxq,
        description="Feature-based approximate Q-learning agent",
    ),
    AgentSpec(
        "approx_doufeature",
        ("approxdou", "approxdf"),
        _approx_doufeature,
        description="Linear ApproxQ with original DouZero x/z features",
    ),
    AgentSpec(
        "attention_dou",
        ("attentiondou", "attndou", "attn_dou"),
        _attention_dou,
        description="Multi-head attention Q-network on DouZero 54-dim tokens",
    ),
    AgentSpec(
        "nnpolicy",
        ("nn", "neural", "policy"),
        _nnpolicy,
        description="Self-play neural policy-gradient action scorer",
    ),
    AgentSpec("douzero", (), _douzero, description="Original DouZero DMC agent"),
    AgentSpec("search", (), _search, needs_position=False, description="Rollout search"),
    AgentSpec(
        "expectimax",
        (),
        _expectimax,
        needs_position=False,
        description="Expectimax search agent",
    ),
    AgentSpec(
        "montecarlo",
        ("mc", "highrankmc"),
        _montecarlo,
        description="High-rank Monte Carlo agent (translated from Botzone C++)",
    ),
)

AGENT_SPECS_BY_NAME = {
    name: spec
    for spec in AGENT_SPECS
    for name in (spec.name, *spec.aliases)
}


EVAL_CONFIGS = {
    "rlcard_fixed": EvalConfig(
        name="rlcard_fixed",
        methods=("rlcard", "rlcard", "rlcard"),
        eval_mode="fixed",
        assignment_workers=1,
    ),
    "rlcard_rotate": EvalConfig(
        name="rlcard_rotate",
        methods=("rlcard", "rlcard", "rlcard"),
        eval_mode="rotate",
        assignment_workers=3,
    ),
    "approxq_1m_vs_rlcard_fixed": EvalConfig(
        name="approxq_1m_vs_rlcard_fixed",
        methods=(with_prefix("approxq", APPROXQ_1M_HISTORY), "rlcard", "rlcard"),
        eval_mode="fixed",
        assignment_workers=1,
    ),
    "approxq_1m_vs_rlcard_rotate": EvalConfig(
        name="approxq_1m_vs_rlcard_rotate",
        methods=(with_prefix("approxq", APPROXQ_1M_HISTORY), "rlcard", "rlcard"),
        eval_mode="rotate",
        assignment_workers=3,
    ),
    "douzero_60m_vs_rlcard_rotate": EvalConfig(
        name="douzero_60m_vs_rlcard_rotate",
        methods=(with_prefix("douzero", DOUZERO_LOGADP_60M_LANDLORD), "rlcard", "rlcard"),
        eval_mode="rotate",
        assignment_workers=3,
    ),
    "probability_vs_rlcard_rotate": EvalConfig(
        name="probability_vs_rlcard_rotate",
        methods=("probability", "rlcard", "rlcard"),
        eval_mode="rotate",
        assignment_workers=3,
    ),
    "value_vs_rlcard_rotate": EvalConfig(
        name="value_vs_rlcard_rotate",
        methods=("value", "rlcard", "rlcard"),
        eval_mode="rotate",
        assignment_workers=3,
    ),
}


def get_eval_config(name: str) -> EvalConfig:
    try:
        return EVAL_CONFIGS[name]
    except KeyError as exc:
        available = ", ".join(sorted(EVAL_CONFIGS))
        raise KeyError("Unknown eval config '{}'. Available: {}".format(name, available)) from exc


def override_eval_config(
    config: EvalConfig,
    *,
    methods: tuple[str, str, str] | None = None,
    eval_mode: str | None = None,
    evaluate_name: str | None = None,
    result_dir: str | None = None,
    eval_data: str | None = None,
    num_workers: int | None = None,
    assignment_workers: int | None = None,
    gpu_device: str | None = None,
) -> EvalConfig:
    return replace(
        config,
        name=evaluate_name or config.name,
        methods=methods or config.methods,
        eval_mode=eval_mode or config.eval_mode,
        eval_data=eval_data or config.eval_data,
        num_workers=num_workers if num_workers is not None else config.num_workers,
        assignment_workers=(
            assignment_workers
            if assignment_workers is not None
            else config.assignment_workers
        ),
        result_dir=result_dir or config.result_dir,
        gpu_device=gpu_device if gpu_device is not None else config.gpu_device,
    )


def run_eval_config(config: EvalConfig):
    from douzero.evaluation.simulation import evaluate

    return evaluate(
        config.methods[0],
        config.methods[1],
        config.methods[2],
        config.eval_data,
        config.num_workers,
        eval_mode=config.eval_mode,
        methods=config.methods,
        evaluate_name=config.name,
        result_dir=config.result_dir,
        assignment_workers=config.assignment_workers,
    )
