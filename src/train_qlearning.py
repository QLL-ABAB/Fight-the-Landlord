from douzero.rl import parser, train


if __name__ == "__main__":
    flags = parser.parse_args()
    train(flags)

