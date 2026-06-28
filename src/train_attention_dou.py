from douzero.rl.attention_dou import build_parser, train
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
    "learning_rate",
    "gamma",
    "epsilon",
    "min_epsilon",
    "epsilon_decay",
    "weight_decay",
    "max_grad_norm",
    "hidden_dim",
    "num_heads",
    "num_layers",
    "dropout",
    "device",
    "actor_device",
    "max_steps",
    "log_interval",
    "progress_interval",
    "save_interval",
    "update_mode",
    "num_workers",
    "num_threads",
    "cpu_threads",
    "unroll_length",
    "num_buffers",
    "buffer_size",
    "learn_batch_size",
    "learn_steps",
    "baseline_beta",
    "rmsprop_alpha",
    "momentum",
    "optimizer_eps",
)


#TODO: 给 attention_dou 增加统一 config.py 入口。
def add_config_args(parser):
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Use a named attention_dou training config from src/train_config.py",
    )
    parser.add_argument(
        "--list_configs",
        action="store_true",
        help="List named attention_dou training configs and exit",
    )


#TODO: 只让命令行显式传入的参数覆盖 config。
def cli_overrides(parser, args):
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
    return overrides


def main():
    parser = build_parser()
    add_config_args(parser)
    flags = parser.parse_args()
    flags._provided_options = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }

    if flags.list_configs:
        for name, config in sorted(train_configs_for_algorithm("attention_dou").items()):
            print(config_summary(config))
        raise SystemExit(0)

    if flags.config:
        config = get_train_config(flags.config, algorithm="attention_dou")
        config = override_train_config(config, **cli_overrides(parser, flags))
        flags = apply_train_config_to_args(flags, config)

    train(flags)


if __name__ == "__main__":
    main()
