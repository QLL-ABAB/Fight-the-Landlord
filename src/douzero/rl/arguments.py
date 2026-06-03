import argparse


parser = argparse.ArgumentParser(description="DouZero: tabular Q-learning")

parser.add_argument("--episodes", default=10000, type=int,
                    help="Number of self-play episodes")
parser.add_argument("--name", default="default", type=str,
                    help="Training task name")
parser.add_argument("--objective", default="wp", type=str,
                    choices=["wp", "adp", "logadp"],
                    help="Terminal reward objective")
parser.add_argument("--savedir", default="qlearning_checkpoints/qlearning",
                    help="Root directory for Q-learning checkpoints")
parser.add_argument("--output", default="",
                    help="Optional explicit output path; overrides --savedir/--name")
parser.add_argument("--load", default="", type=str,
                    help="Optional existing Q table to continue training")
parser.add_argument("--resume", action="store_true",
                    help="Resume from --load, --output, or the latest checkpoint for --name")
parser.add_argument("--seed", default=0, type=int,
                    help="Random seed")

parser.add_argument("--alpha", default=0.1, type=float,
                    help="Q-learning step size")
parser.add_argument("--gamma", default=0.95, type=float,
                    help="Discount factor")
parser.add_argument("--epsilon", default=0.2, type=float,
                    help="Initial epsilon for epsilon-greedy exploration")
parser.add_argument("--min_epsilon", default=0.02, type=float,
                    help="Smallest epsilon after decay")
parser.add_argument("--epsilon_decay", default=0.9995, type=float,
                    help="Multiplicative epsilon decay after each episode")

parser.add_argument("--state_mode", default="public", type=str,
                    choices=["public", "hand_only"],
                    help="State abstraction used as the Q-table key")
parser.add_argument("--max_steps", default=1000, type=int,
                    help="Safety cap for one game")
parser.add_argument("--log_interval", default=100, type=int,
                    help="Print training stats every N episodes")
parser.add_argument("--save_interval", default=1000, type=int,
                    help="Save checkpoint every N episodes; 0 disables interim saves")
