import os
import argparse

from config import EVAL_CONFIGS, get_eval_config, override_eval_config, run_eval_config

DEFAULT_CONFIG = "probability_vs_rlcard_rotate"

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dou Dizhu Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Use a named evaluation preset from src/config.py",
    )
    parser.add_argument(
        "--list_configs",
        action="store_true",
        help="List named evaluation presets and exit",
    )
    parser.add_argument("--landlord", type=str, default=None)
    parser.add_argument("--landlord_up", type=str, default=None)
    parser.add_argument("--landlord_down", type=str, default=None)
    parser.add_argument(
        "--methods",
        nargs=3,
        default=None,
        help="Three agents for fixed/rotate evaluation",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Shortcut for '--methods METHOD rlcard rlcard'",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["fixed", "rotate"],
        default=None,
        help="fixed keeps the first method as landlord; rotate lets each method be landlord once",
    )
    parser.add_argument(
        "--evaluate_name",
        type=str,
        default="evaluate",
        help="Name used for the saved evaluation JSON",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="evaluate_results",
        help="Directory where evaluation JSON files are saved",
    )

    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--assignment_workers",
        type=int,
        default=None,
        help="How many fixed/rotate role assignments to evaluate in parallel",
    )
    parser.add_argument("--gpu_device", type=str, default=None)
    args = parser.parse_args()

    if args.list_configs:
        for name in sorted(EVAL_CONFIGS):
            cfg = EVAL_CONFIGS[name]
            print(
                "{} -> mode={}, methods={}, workers={}, assignment_workers={}".format(
                    name,
                    cfg.eval_mode,
                    cfg.methods,
                    cfg.num_workers,
                    cfg.assignment_workers,
                )
            )
        raise SystemExit(0)

    config = None
    if args.config:
        config = get_eval_config(args.config)

    if args.method:
        methods = (args.method, "rlcard", "rlcard")
    elif args.methods:
        methods = tuple(args.methods)
    elif args.landlord or args.landlord_up or args.landlord_down:
        base = get_eval_config(DEFAULT_CONFIG).methods
        methods = (
            args.landlord or base[0],
            args.landlord_up or base[1],
            args.landlord_down or base[2],
        )
    else:
        methods = None

    if config is None:
        config = get_eval_config(DEFAULT_CONFIG)

    config = override_eval_config(
        config,
        methods=methods,
        eval_mode=args.eval_mode,
        evaluate_name=args.evaluate_name if args.evaluate_name != "evaluate" else None,
        result_dir=args.result_dir if args.result_dir != "evaluate_results" else None,
        eval_data=args.eval_data,
        num_workers=args.num_workers,
        assignment_workers=args.assignment_workers,
        gpu_device=args.gpu_device,
    )

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_device

    run_eval_config(config)
