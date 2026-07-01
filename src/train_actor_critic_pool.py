from douzero.rl.actor_critic_pool import train
from douzero.rl.actor_critic_pool_arguments import parser


if __name__ == "__main__":
    flags = parser.parse_args()
    train(flags)
