import os

from douzero.rl.approx_qlearning import (
    DEFAULT_APPROX_Q_PATH,
    ApproxQModel,
    features_for_actions,
    prune_legal_actions,
)


DEFAULT_APPROX_Q_DIR = "approx_qlearning_checkpoints/approx_qlearning"


#note: 在目录下寻找最新的数字命名 .pkl，避免选到临时文件或日志。
def _latest_pkl_in_dir(directory):
    if not os.path.isdir(directory):
        return None

    candidates = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".pkl"):
                continue
            path = os.path.join(root, filename)
            stem = os.path.splitext(filename)[0]
            episode = int(stem) if stem.isdigit() else 0
            candidates.append((episode, os.path.getmtime(path), path))

    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][2]


#note: 支持 approxq:path、approxq:task_name 和默认目录三种模型定位方式。
def resolve_approx_q_model_path(model_path=None):
    if model_path:
        if os.path.isfile(model_path):
            return model_path
        if os.path.isdir(model_path):
            latest = _latest_pkl_in_dir(model_path)
            if latest:
                return latest

        task_dir = os.path.join(DEFAULT_APPROX_Q_DIR, model_path)
        latest = _latest_pkl_in_dir(task_dir)
        if latest:
            return latest
        return model_path

    if os.path.isfile(DEFAULT_APPROX_Q_PATH):
        return DEFAULT_APPROX_Q_PATH

    latest = _latest_pkl_in_dir(DEFAULT_APPROX_Q_DIR)
    if latest:
        return latest

    raise FileNotFoundError(
        "No approximate Q-learning checkpoint found. Use approxq:/path/model.pkl "
        "or train one under {}".format(DEFAULT_APPROX_Q_DIR)
    )


class ApproxQLearningAgent(object):
    def __init__(self, position, model_path=None, device="cpu"):
        self.name = "ApproxQLearning"
        self.position = position
        self.model_path = resolve_approx_q_model_path(model_path)
        self.model = ApproxQModel.load(self.model_path, device=device)
        self.max_candidate_actions = int(
            self.model.metadata.get("max_candidate_actions", 64)
        )

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        actions = prune_legal_actions(
            self.position, infoset, self.max_candidate_actions
        )
        features = features_for_actions(
            self.position, infoset, actions, self.model.device,
            self.model.feature_mode
        )
        action, _ = self.model.select_action(
            self.position,
            actions,
            features,
            epsilon=0.0,
        )
        return action
