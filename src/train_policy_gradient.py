from douzero.rl.policy_gradient import train
from douzero.rl.policy_gradient_arguments import parser


if __name__ == "__main__":
    flags = parser.parse_args()
    train(flags)
