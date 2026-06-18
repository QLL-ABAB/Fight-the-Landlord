import os

from douzero.rl.approx_doufeature import (
    DEFAULT_APPROX_DOUFEATURE_DIR,
    DEFAULT_APPROX_DOUFEATURE_PATH,
    ApproxDouFeatureModel,
    douzero_features_for_infoset,
)


#TODO: 在目录树中寻找最新的数字命名 .pkl checkpoint。
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


#TODO: 支持 approx_doufeature:/file、目录、任务名和默认目录四种定位方式。
def resolve_approx_doufeature_model_path(model_path=None):
    if model_path:
        if os.path.isfile(model_path):
            return model_path
        if os.path.isdir(model_path):
            latest = _latest_pkl_in_dir(model_path)
            if latest:
                return latest
        task_dir = os.path.join(DEFAULT_APPROX_DOUFEATURE_DIR, model_path)
        latest = _latest_pkl_in_dir(task_dir)
        if latest:
            return latest
        return model_path
    if os.path.isfile(DEFAULT_APPROX_DOUFEATURE_PATH):
        return DEFAULT_APPROX_DOUFEATURE_PATH
    latest = _latest_pkl_in_dir(DEFAULT_APPROX_DOUFEATURE_DIR)
    if latest:
        return latest
    raise FileNotFoundError(
        "No approx_doufeature checkpoint found. Use approx_doufeature:/path/model.pkl"
    )


class ApproxDouFeatureAgent:
    #TODO: 加载 DouZero 特征线性模型，评测时始终贪心选择最大 Q 动作。
    def __init__(self, position, model_path=None):
        self.name = "ApproxDouFeature"
        self.position = position
        self.model_path = resolve_approx_doufeature_model_path(model_path)
        self.model = ApproxDouFeatureModel.load(self.model_path)

    #TODO: 用 DouZero 原始 observation 特征构造所有合法动作，并选 Q 最大者。
    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]
        actions, features = douzero_features_for_infoset(self.position, infoset)
        action, _ = self.model.select_action(self.position, actions, features, epsilon=0.0)
        return action
