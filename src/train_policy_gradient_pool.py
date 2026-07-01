from douzero.rl.policy_gradient_pool import train
from douzero.rl.policy_gradient_pool_arguments import parser


if __name__ == "__main__":
    flags = parser.parse_args()
    train(flags)
