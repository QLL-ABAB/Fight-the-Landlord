from douzero.rl.approx_arguments import parser
from douzero.rl.approx_qlearning import train


if __name__ == "__main__":
    flags = parser.parse_args()
    train(flags)
