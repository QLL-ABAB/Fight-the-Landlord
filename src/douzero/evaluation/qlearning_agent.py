import os

from douzero.rl.qlearning import DEFAULT_QTABLE_PATH, QTable, make_state_key


DEFAULT_QLEARNING_DIR = "qlearning_checkpoints/qlearning"


def _latest_pkl_in_dir(directory):
    if not os.path.isdir(directory):
        return None

    candidates = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".pkl"):
                continue
            path = os.path.join(root, filename)
            candidates.append((os.path.getmtime(path), path))

    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def resolve_qlearning_model_path(model_path=None):
    if model_path:
        if os.path.isfile(model_path):
            return model_path
        if os.path.isdir(model_path):
            latest = _latest_pkl_in_dir(model_path)
            if latest:
                return latest

        task_dir = os.path.join(DEFAULT_QLEARNING_DIR, model_path)
        latest = _latest_pkl_in_dir(task_dir)
        if latest:
            return latest
        return model_path

    if os.path.isfile(DEFAULT_QTABLE_PATH):
        return DEFAULT_QTABLE_PATH

    latest = _latest_pkl_in_dir(DEFAULT_QLEARNING_DIR)
    if latest:
        return latest

    raise FileNotFoundError(
        "No Q-learning checkpoint found. Use qlearning:/path/to/model.pkl "
        "or train one under {}".format(DEFAULT_QLEARNING_DIR)
    )


class QLearningAgent(object):
    def __init__(self, position, model_path=None):
        self.name = "QLearning"
        self.position = position
        self.model_path = resolve_qlearning_model_path(model_path)
        self.qtable = QTable.load(self.model_path)
        self.state_mode = self.qtable.metadata.get("state_mode", "public")

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        state = make_state_key(infoset, self.state_mode)
        return self.qtable.select_action(
            self.position,
            state,
            infoset.legal_actions,
            epsilon=0.0,
        )
