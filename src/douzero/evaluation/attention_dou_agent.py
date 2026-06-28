import os

from douzero.rl.attention_dou import (
    DEFAULT_ATTENTION_DOU_DIR,
    DEFAULT_ATTENTION_DOU_PATH,
    AttentionDouModel,
    tokens_for_infoset,
)


#TODO: 在目录树中寻找最新的数字命名 attention_dou checkpoint。
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


#TODO: 支持 attention_dou:/file、目录、任务名和默认目录四种定位方式。
def resolve_attention_dou_model_path(model_path=None):
    if model_path:
        if os.path.isfile(model_path):
            return model_path
        if os.path.isdir(model_path):
            latest = _latest_pkl_in_dir(model_path)
            if latest:
                return latest
        task_dir = os.path.join(DEFAULT_ATTENTION_DOU_DIR, model_path)
        latest = _latest_pkl_in_dir(task_dir)
        if latest:
            return latest
        return model_path
    if os.path.isfile(DEFAULT_ATTENTION_DOU_PATH):
        return DEFAULT_ATTENTION_DOU_PATH
    latest = _latest_pkl_in_dir(DEFAULT_ATTENTION_DOU_DIR)
    if latest:
        return latest
    raise FileNotFoundError(
        "No attention_dou checkpoint found. Use attention_dou:/path/model.pkl"
    )


class AttentionDouAgent:
    #TODO: 加载 attention_dou 模型，评测时贪心选择最大 Q 动作。
    def __init__(self, position, model_path=None):
        self.name = "AttentionDou"
        self.position = position
        self.model_path = resolve_attention_dou_model_path(model_path)
        self.model = AttentionDouModel.load(
            self.model_path, device="cpu", load_optimizer=False
        )

    #TODO: 用 54 维 token 序列构造所有合法动作并选 Q 最大者。
    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]
        actions, tokens = tokens_for_infoset(self.position, infoset)
        action, _ = self.model.select_action(self.position, actions, tokens, epsilon=0.0)
        return action
