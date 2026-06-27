import argparse


parser = argparse.ArgumentParser(
    description="DouZero: linear-feature actor-critic"
)

parser.add_argument("--episodes", default=10000, type=int,
                    help="Number of self-play episodes")
parser.add_argument("--name", default="ac_logadp_10k_history", type=str,
                    help="Training task name")
parser.add_argument("--objective", default="logadp", type=str,
                    choices=["wp", "adp", "logadp"],
                    help="Terminal reward objective aligned with DouZero")
parser.add_argument("--reward_scale", default=1.0, type=float,
                    help="Multiply terminal and optional shaped rewards")
parser.add_argument("--reward_shaping", action="store_true",
                    help="Enable optional intermediate shaped rewards")
parser.add_argument("--savedir",
                    default="actor_critic_checkpoints/actor_critic",
                    help="Root directory for actor-critic checkpoints")
parser.add_argument("--output", default="",
                    help="Optional explicit output path; overrides --savedir/--name")
parser.add_argument("--load", default="", type=str,
                    help="Optional existing model to continue training")
parser.add_argument("--resume", action="store_true",
                    help="Resume from --load, --output, or latest checkpoint")
parser.add_argument("--seed", default=0, type=int,
                    help="Random seed")

parser.add_argument("--actor_learning_rate", default=0.002, type=float,
                    help="Step size for linear actor weights")
parser.add_argument("--critic_learning_rate", default=0.01, type=float,
                    help="Step size for linear critic weights")
parser.add_argument("--gamma", default=0.98, type=float,
                    help="Discount factor for reward-to-go")
parser.add_argument("--actor_l2", default=0.00001, type=float,
                    help="Small L2 shrinkage for actor weights")
parser.add_argument("--critic_l2", default=0.00001, type=float,
                    help="Small L2 shrinkage for critic weights")
parser.add_argument("--actor_grad_clip", default=5.0, type=float,
                    help="Clip actor-gradient L2 norm; 0 disables clipping")
parser.add_argument("--critic_grad_clip", default=5.0, type=float,
                    help="Clip critic-gradient L2 norm; 0 disables clipping")
parser.add_argument("--advantage_clip", default=10.0, type=float,
                    help="Clip advantage/TD error; 0 disables clipping")

parser.add_argument("--temperature", default=1.0, type=float,
                    help="Initial softmax temperature for stochastic policy")
parser.add_argument("--min_temperature", default=0.2, type=float,
                    help="Smallest temperature after decay")
parser.add_argument("--temperature_decay", default=0.99998, type=float,
                    help="Multiplicative temperature decay after each episode")

parser.add_argument("--feature_mode", default="history", type=str,
                    choices=["history", "compact"],
                    help="Reuse approximate-Q encoded state-action features for actor")
parser.add_argument("--max_candidate_actions", default=64, type=int,
                    help="Action mask keeps at most N legal candidates; 0 disables pruning")
parser.add_argument("--max_steps", default=1000, type=int,
                    help="Safety cap for one game")
parser.add_argument("--log_interval", default=1000, type=int,
                    help="Print training stats every N episodes")
parser.add_argument("--progress_interval", default=500, type=int,
                    help="Refresh progress bar every N episodes; 0 disables it")
parser.add_argument("--save_interval", default=50000, type=int,
                    help="Save checkpoint every N episodes; 0 disables interim saves")
