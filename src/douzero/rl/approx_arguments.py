import argparse


parser = argparse.ArgumentParser(
    description="DouZero: feature-based approximate Q-learning"
)

parser.add_argument("--episodes", default=10000, type=int,
                    help="Number of self-play episodes")
parser.add_argument("--name", default="approxq_logadp_100k", type=str,
                    help="Training task name")
parser.add_argument("--objective", default="logadp", type=str,
                    choices=["wp", "adp", "logadp"],
                    help="Terminal reward objective aligned with DouZero")
parser.add_argument("--reward_scale", default=1.0, type=float,
                    help="Multiply terminal and optional shaped rewards")
parser.add_argument("--reward_shaping", action="store_true",
                    help="Enable optional intermediate shaped rewards")
parser.add_argument("--savedir",
                    default="approx_qlearning_checkpoints/approx_qlearning",
                    help="Root directory for approximate Q-learning checkpoints")
parser.add_argument("--output", default="",
                    help="Optional explicit output path; overrides --savedir/--name")
parser.add_argument("--load", default="", type=str,
                    help="Optional existing model to continue training")
parser.add_argument("--resume", action="store_true",
                    help="Resume from --load, --output, or latest checkpoint")
parser.add_argument("--seed", default=0, type=int,
                    help="Random seed")

parser.add_argument("--alpha", default=0.02, type=float,
                    help="Q-learning step size for linear weights")
parser.add_argument("--gamma", default=0.98, type=float,
                    help="Discount factor")
parser.add_argument("--epsilon", default=0.1, type=float,
                    help="Initial epsilon for epsilon-greedy exploration")
parser.add_argument("--min_epsilon", default=0.02, type=float,
                    help="Smallest epsilon after decay")
parser.add_argument("--epsilon_decay", default=0.99998, type=float,
                    help="Multiplicative epsilon decay after each episode")
parser.add_argument("--l2", default=0.00001, type=float,
                    help="Small L2 shrinkage for linear weights")
parser.add_argument("--clip_td", default=10.0, type=float,
                    help="Clip TD error magnitude; 0 disables clipping")

parser.add_argument("--device", default="auto", type=str,
                    help="auto, cpu, cuda, or cuda:0")
parser.add_argument("--feature_mode", default="history", type=str,
                    choices=["history", "compact"],
                    help="history keeps full fixed-size played-card features; compact uses summary features only")
parser.add_argument("--max_candidate_actions", default=64, type=int,
                    help="Keep at most N candidate actions per decision; 0 disables pruning")
parser.add_argument("--max_steps", default=1000, type=int,
                    help="Safety cap for one game")
parser.add_argument("--log_interval", default=1000, type=int,
                    help="Print training stats every N episodes")
parser.add_argument("--progress_interval", default=500, type=int,
                    help="Refresh progress bar every N episodes; 0 disables it")
parser.add_argument("--save_interval", default=50000, type=int,
                    help="Save checkpoint every N episodes; 0 disables interim saves")
