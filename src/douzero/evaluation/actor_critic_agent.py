import os

try:
    from douzero.rl.approx_qlearning import features_for_actions, prune_legal_actions
except ModuleNotFoundError as exc:
    if exc.name != "douzero.rl.approx_qlearning":
        raise
    from douzero.rl.approx_qlearning_fasle import (
        features_for_actions,
        prune_legal_actions,
    )
from douzero.rl.actor_critic import (
    DEFAULT_ACTOR_CRITIC_PATH,
    LinearActorCriticModel,
)


DEFAULT_ACTOR_CRITIC_DIR = "actor_critic_checkpoints/actor_critic"


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


def resolve_actor_critic_model_path(model_path=None):
    if model_path:
        if os.path.isfile(model_path):
            return model_path
        if os.path.isdir(model_path):
            latest = _latest_pkl_in_dir(model_path)
            if latest:
                return latest

        task_dir = os.path.join(DEFAULT_ACTOR_CRITIC_DIR, model_path)
        latest = _latest_pkl_in_dir(task_dir)
        if latest:
            return latest
        return model_path

    if os.path.isfile(DEFAULT_ACTOR_CRITIC_PATH):
        return DEFAULT_ACTOR_CRITIC_PATH

    latest = _latest_pkl_in_dir(DEFAULT_ACTOR_CRITIC_DIR)
    if latest:
        return latest

    raise FileNotFoundError(
        "No actor-critic checkpoint found. Use ac:/path/model.pkl "
        "or train one under {}".format(DEFAULT_ACTOR_CRITIC_DIR)
    )


class ActorCriticAgent(object):
    def __init__(self, position, model_path=None):
        self.name = "ActorCritic"
        self.position = position
        self.model_path = resolve_actor_critic_model_path(model_path)
        self.model = LinearActorCriticModel.load(self.model_path)
        self.max_candidate_actions = int(
            self.model.metadata.get("max_candidate_actions", 64)
        )

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        actions = prune_legal_actions(
            self.position,
            infoset,
            self.max_candidate_actions,
        )
        features = features_for_actions(
            self.position,
            infoset,
            actions,
            "python",
            self.model.feature_mode,
        )
        return self.model.greedy_action(self.position, actions, features)
