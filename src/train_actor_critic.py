from douzero.rl.actor_critic import train
from douzero.rl.actor_critic_arguments import parser


if __name__ == "__main__":
    flags = parser.parse_args()
    train(flags)
