from douzero.rl.approx_arguments import parser
try:
    from douzero.rl.approx_qlearning import train
except ModuleNotFoundError as exc:
    if exc.name != "douzero.rl.approx_qlearning":
        raise
    from douzero.rl.approx_qlearning_fasle import train
from train_config import (
    apply_train_config_to_args,
    config_summary,
    get_train_config,
    override_train_config,
    train_configs_for_algorithm,
)

import sys


CONFIG_OVERRIDE_FIELDS = (
    "episodes",
    "name",
    "objective",
    "reward_scale",
    "savedir",
    "output",
    "load",
    "seed",
    "alpha",
    "gamma",
    "epsilon",
    "min_epsilon",
    "epsilon_decay",
    "l2",
    "clip_td",
    "device",
    "feature_mode",
    "max_candidate_actions",
    "max_steps",
    "log_interval",
    "progress_interval",
    "save_interval",
    "feature_diag",
    "feature_diag_path",
    "feature_diag_interval",
    "feature_diag_topk",
)


def add_config_args():
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Use a named training config from src/train_config.py",
    )
    parser.add_argument(
        "--list_configs",
        action="store_true",
        help="List named training configs and exit",
    )


def cli_overrides(args):
    provided = {
        action.dest
        for action in parser._actions
        if action.option_strings
        for option in action.option_strings
        if option in args._provided_options
    }
    overrides = {
        field_name: getattr(args, field_name)
        for field_name in CONFIG_OVERRIDE_FIELDS
        if field_name in provided
    }
    if "reward_shaping" in provided:
        overrides["reward_shaping"] = args.reward_shaping
    if "resume" in provided:
        overrides["resume"] = args.resume
    if "feature_diag" in provided:
        overrides["feature_diag"] = args.feature_diag
    return overrides


if __name__ == "__main__":
    add_config_args()
    flags = parser.parse_args()
    flags._provided_options = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }

    if flags.list_configs:
        for name, config in sorted(train_configs_for_algorithm("approxq").items()):
            print(config_summary(config))
        raise SystemExit(0)

    if flags.config:
        config = get_train_config(flags.config, algorithm="approxq")
        config = override_train_config(config, **cli_overrides(flags))
        flags = apply_train_config_to_args(flags, config)

    train(flags)
