import torch
import numpy as np
import importlib.util
from pathlib import Path

from douzero.env.env import get_obs


def _load_base_model_dict():
    root = Path(__file__).resolve().parents[3]
    models_path = root / "base" / "douzero" / "dmc" / "models.py"
    if not models_path.exists():
        raise ModuleNotFoundError(
            "No module named 'douzero.dmc' and fallback model file was not found: "
            "{}".format(models_path)
        )

    spec = importlib.util.spec_from_file_location(
        "cadam_douzero_base_dmc_models", str(models_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model_dict


def _model_dict():
    try:
        from douzero.dmc.models import model_dict

        return model_dict
    except ModuleNotFoundError:
        return _load_base_model_dict()


def _load_model(position, model_path):
    model = _model_dict()[position]()
    model_state_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

class DeepAgent:

    def __init__(self, position, model_path):
        self.model = _load_model(position, model_path)

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs(infoset) 

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action
