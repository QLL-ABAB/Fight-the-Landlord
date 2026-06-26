# train_selfplay_policy_gradient.py
# ------------------------------------------------------------
# Role-specific policy-gradient training for Dou Dizhu neural action scorers.
#
# This version supports two modes:
#   1. Pure role-specific self-play, same public interface as before.
#   2. Mixed-opponent fine-tuning from an existing role-specific checkpoint:
#        current NN vs old NN checkpoint / low-budget Monte Carlo / self-play.
#
# No teacher labels / no distillation / no search enhancement inside the NN.
# Opponents only play games against the current NN; the current NN is still
# updated by REINFORCE from win/loss reward.
# ------------------------------------------------------------

import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from douzero.evaluation.high_rank_montecarlo_agent import (
        CardCombo, CardCombinations, HighRankMonteCarloAgent, card2level
    )
except Exception:
    from high_rank_montecarlo_agent import (
        CardCombo, CardCombinations, HighRankMonteCarloAgent, card2level
    )

try:
    from douzero.evaluation.nn_policy_agent import feature_vector, feature_dim
except Exception:
    from nn_policy_agent import feature_vector, feature_dim


ALL_POSITIONS = ["landlord", "landlord_down", "landlord_up"]
POS_INDEX = {"landlord": 0, "landlord_down": 1, "landlord_up": 2}


def repo_root():
    """Resolve project root when this file lives in src/douzero/evaluation."""
    here = os.path.abspath(__file__)
    cur = os.path.dirname(here)
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "src")) and os.path.isdir(os.path.join(cur, "src", "douzero")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # Fallback: current working directory.
    return os.getcwd()


def resolve_path(path, root=None):
    if path is None or path == "":
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    root = root or repo_root()
    return os.path.normpath(os.path.join(root, path))


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def card_sort_key(c):
    return (card2level(int(c)), int(c))


def level_to_env(level):
    if level <= 11:
        return level + 3
    if level == 12:
        return 17
    if level == 13:
        return 20
    if level == 14:
        return 30
    return 3


def concrete_to_env(card):
    return level_to_env(card2level(int(card)))


def concrete_action_to_env(action):
    return [concrete_to_env(c) for c in list(action or [])]


def env_action_key(action):
    return tuple(sorted([int(x) for x in list(action or [])]))


class SimpleInfoset(object):
    """Minimal infoset object compatible with feature_vector and local agents."""

    def __init__(self, position, player_hand_cards, legal_actions, last_move,
                 last_two_moves, last_pid, num_cards_left):
        self.position = position
        self.player_hand_cards = list(player_hand_cards)
        self.legal_actions = [list(a) for a in legal_actions]
        self.last_move = list(last_move)
        self.last_two_moves = [list(x) for x in last_two_moves]
        self.last_pid = last_pid
        self.num_cards_left = dict(num_cards_left)


class SimpleDoudizhuSelfPlayEnv(object):
    """Lightweight playing-stage Dou Dizhu environment for training.

    Landlord is fixed to player 0. Cards are Botzone concrete ids 0..53.
    Legal action enumeration is rule-based. There is no teacher signal here.
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.rule_engine = HighRankMonteCarloAgent(position="landlord", time_limit_sec=0.0)
        self.reset()

    def reset(self):
        deck = list(range(54))
        self.rng.shuffle(deck)
        public = deck[:3]
        p0 = deck[3:20] + public
        p1 = deck[20:37]
        p2 = deck[37:54]

        self.hands = [sorted(p0, key=card_sort_key),
                      sorted(p1, key=card_sort_key),
                      sorted(p2, key=card_sort_key)]
        self.current_player = 0
        self.last_valid_combo = CardCombo()
        self.last_valid_action = []
        self.last_valid_player = None
        self.pass_count = 0
        self.history = []
        self.player_history = []
        self.done = False
        self.winner = None
        return self.get_infoset()

    def position_name(self, player):
        return ALL_POSITIONS[int(player)]

    def get_num_cards_left(self):
        return {ALL_POSITIONS[i]: len(self.hands[i]) for i in range(3)}

    def last_two_moves(self):
        last = self.history[-2:]
        while len(last) < 2:
            last.insert(0, [])
        return [list(x) for x in last]

    def get_infoset(self):
        player = self.current_player
        legal_actions = self.legal_actions()
        last_pid = self.position_name(self.last_valid_player) if self.last_valid_player is not None else None
        return SimpleInfoset(
            position=self.position_name(player),
            player_hand_cards=list(self.hands[player]),
            legal_actions=legal_actions,
            last_move=list(self.last_valid_action),
            last_two_moves=self.last_two_moves(),
            last_pid=last_pid,
            num_cards_left=self.get_num_cards_left(),
        )

    def legal_actions(self):
        player = self.current_player
        hand = CardCombinations(self.hands[player])
        actions = self.rule_engine.getActions(hand, self.last_valid_combo)
        out = []
        seen = set()
        for combo in actions:
            a = sorted(list(combo.cards), key=card_sort_key)
            key = tuple(a)
            if key not in seen:
                seen.add(key)
                out.append(a)
        if not out and len(self.hands[player]) > 0 and len(self.last_valid_action) == 0:
            out = [[self.hands[player][0]]]
        return out

    def step(self, action):
        if self.done:
            return self.get_infoset(), 0.0, True, {"winner": self.winner}
        action = sorted(list(action or []), key=card_sort_key)

        legal = self.legal_actions()
        legal_keys = {tuple(a) for a in legal}
        if tuple(action) not in legal_keys:
            action = legal[0] if legal else []

        player = self.current_player
        self.history.append(list(action))
        self.player_history.append(player)

        if action:
            for c in action:
                self.hands[player].remove(c)
            self.last_valid_combo = CardCombo(action)
            self.last_valid_action = list(action)
            self.last_valid_player = player
            self.pass_count = 0

            if len(self.hands[player]) == 0:
                self.done = True
                self.winner = player
                return self.get_infoset(), 0.0, True, {"winner": self.winner}

            self.current_player = (player + 1) % 3
        else:
            self.pass_count += 1
            if self.pass_count >= 2 and self.last_valid_player is not None:
                self.current_player = self.last_valid_player
                self.last_valid_combo = CardCombo()
                self.last_valid_action = []
                self.last_valid_player = None
                self.pass_count = 0
            else:
                self.current_player = (player + 1) % 3

        return self.get_infoset(), 0.0, self.done, {"winner": self.winner}


class ActionScoringNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, layers=2):
        super(ActionScoringNet, self).__init__()
        mods = []
        d = input_dim
        for _ in range(layers):
            mods.append(nn.Linear(d, hidden_dim))
            mods.append(nn.ReLU())
            d = hidden_dim
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_role_models(device, hidden_dim=128, layers=2):
    return {p: ActionScoringNet(feature_dim(), hidden_dim=hidden_dim, layers=layers).to(device)
            for p in ALL_POSITIONS}


def make_role_optimizers(models, lr):
    return {p: optim.Adam(models[p].parameters(), lr=lr) for p in ALL_POSITIONS}


def team_reward(winner, player):
    if winner == 0:
        return 1.0 if player == 0 else -1.0
    return -1.0 if player == 0 else 1.0


def _linear_layers_to_lists(model):
    weights = []
    biases = []
    for m in model.net:
        if isinstance(m, nn.Linear):
            weights.append(m.weight.detach().cpu().tolist())
            biases.append(m.bias.detach().cpu().tolist())
    return weights, biases


def export_weights(models, out_path, note=""):
    role_models = {}
    for p in ALL_POSITIONS:
        weights, biases = _linear_layers_to_lists(models[p])
        role_models[p] = {"input_dim": feature_dim(), "weights": weights, "biases": biases}

    obj = {
        "input_dim": feature_dim(),
        "role_models": role_models,
        "roles": list(ALL_POSITIONS),
        "trained_by": "role_specific_mixed_opponent_policy_gradient",
        "card_encoding": "botzone_concrete_0_53_or_env_rank_if_concrete_false",
        "note": note or "No teacher labels. Opponents are used only as game opponents for RL fine-tuning.",
    }
    parent = os.path.dirname(os.path.abspath(out_path))
    ensure_dir(parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def load_role_weights(models, path, device, strict=True):
    """Load exported role_models JSON into torch role models."""
    if not path:
        return False
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "role_models" in obj:
        role_data = obj["role_models"]
    elif "weights" in obj and "biases" in obj:
        # Old shared-policy format: copy same network to all roles.
        role_data = {p: obj for p in ALL_POSITIONS}
    else:
        raise ValueError("Unsupported weights format: {}".format(path))

    for role in ALL_POSITIONS:
        if role not in role_data:
            if strict:
                raise ValueError("Missing role '{}' in weights {}".format(role, path))
            continue
        sub = role_data[role]
        weights = sub["weights"]
        biases = sub["biases"]
        linear_layers = [m for m in models[role].net if isinstance(m, nn.Linear)]
        if len(linear_layers) != len(weights):
            raise ValueError("Layer count mismatch for role '{}' while loading {}".format(role, path))
        for layer, w, b in zip(linear_layers, weights, biases):
            wt = torch.tensor(w, dtype=torch.float32, device=device)
            bt = torch.tensor(b, dtype=torch.float32, device=device)
            if layer.weight.data.shape != wt.shape or layer.bias.data.shape != bt.shape:
                raise ValueError(
                    "Shape mismatch for role '{}' while loading {}. Expected {} / {}, got {} / {}".format(
                        role, path, tuple(layer.weight.data.shape), tuple(layer.bias.data.shape), tuple(wt.shape), tuple(bt.shape)
                    )
                )
            layer.weight.data.copy_(wt)
            layer.bias.data.copy_(bt)
    return True


def clone_frozen_models_from_file(path, device, hidden_dim, layers):
    models = make_role_models(device, hidden_dim=hidden_dim, layers=layers)
    load_role_weights(models, path, device, strict=True)
    for p in ALL_POSITIONS:
        models[p].eval()
        for param in models[p].parameters():
            param.requires_grad = False
    return models


def select_nn_action(model, infoset, position, device, temperature=1.0,
                     greedy=False, concrete=True, need_logprob=True):
    legal = infoset.legal_actions
    if not legal:
        return [], None, None
    if len(legal) == 1:
        return legal[0], None, None

    feats = [feature_vector(infoset, position, a, concrete=concrete) for a in legal]
    x = torch.tensor(feats, dtype=torch.float32, device=device)
    scores = model(x)

    hand_len = len(infoset.player_hand_cards)
    finish_indices = [i for i, a in enumerate(legal) if a and len(a) == hand_len]
    if finish_indices:
        idx = finish_indices[0]
        if greedy or not need_logprob:
            return legal[idx], None, None
        logits = scores / max(1e-6, float(temperature))
        dist = torch.distributions.Categorical(logits=logits)
        action_index = torch.tensor(idx, dtype=torch.long, device=device)
        return legal[idx], dist.log_prob(action_index), dist.entropy()

    logits = scores / max(1e-6, float(temperature))
    if greedy:
        idx = int(torch.argmax(logits).item())
        return legal[idx], None, None
    dist = torch.distributions.Categorical(logits=logits)
    action_index = dist.sample()
    idx = int(action_index.item())
    if not need_logprob:
        return legal[idx], None, None
    return legal[idx], dist.log_prob(action_index), dist.entropy()


class LowBudgetMonteCarloOpponent(object):
    """Use the translated HighRankMonteCarloAgent as a concrete-card opponent.

    The training environment uses Botzone concrete cards. HighRankMonteCarloAgent
    expects rank-coded infosets, so this wrapper converts concrete legal actions
    to rank actions and maps the chosen rank action back to a concrete legal move.
    """

    def __init__(self, time_limit_sec=0.05, n_root_actions=2, seed=None):
        self.time_limit_sec = time_limit_sec
        self.n_root_actions = n_root_actions
        self.seed = seed
        self.agents = {}

    def _agent(self, position):
        if position not in self.agents:
            self.agents[position] = HighRankMonteCarloAgent(
                position=position,
                time_limit_sec=self.time_limit_sec,
                n_root_actions=self.n_root_actions,
                seed=self.seed,
            )
        return self.agents[position]

    def act(self, infoset, position):
        legal = infoset.legal_actions
        if not legal:
            return []
        if len(legal) == 1:
            return legal[0]

        rank_legal = [concrete_action_to_env(a) for a in legal]
        rank_infoset = SimpleInfoset(
            position=position,
            player_hand_cards=concrete_action_to_env(infoset.player_hand_cards),
            legal_actions=rank_legal,
            last_move=concrete_action_to_env(infoset.last_move),
            last_two_moves=[concrete_action_to_env(x) for x in infoset.last_two_moves],
            last_pid=infoset.last_pid,
            num_cards_left=dict(infoset.num_cards_left),
        )
        try:
            rank_action = self._agent(position).act(rank_infoset)
            key = env_action_key(rank_action)
            candidates = [a for a in legal if env_action_key(concrete_action_to_env(a)) == key]
            if candidates:
                # Prefer the exact legal concrete action with the same rank multiset.
                candidates.sort(key=lambda a: (len(a), tuple(a)))
                return candidates[0]
        except Exception:
            pass

        return safe_heuristic_action(infoset, position)


class OldNNOpponentPool(object):
    def __init__(self, checkpoint_paths, device, hidden_dim, layers, temperature=0.4):
        self.device = device
        self.temperature = temperature
        self.models_list = []
        for path in checkpoint_paths:
            if path and os.path.exists(path):
                self.models_list.append(clone_frozen_models_from_file(path, device, hidden_dim, layers))
        self.rng = random.Random(17)

    def has_any(self):
        return bool(self.models_list)

    def act(self, infoset, position):
        if not self.models_list:
            return safe_heuristic_action(infoset, position)
        models = self.rng.choice(self.models_list)
        action, _, _ = select_nn_action(
            models[position], infoset, position, self.device,
            temperature=self.temperature, greedy=True, concrete=True, need_logprob=False
        )
        return action


def safe_heuristic_action(infoset, position):
    legal = infoset.legal_actions
    if not legal:
        return []
    hand = infoset.player_hand_cards
    for a in legal:
        if a and len(a) == len(hand):
            return a
    # Pass if legal and last non-pass was teammate.
    if [] in legal and infoset.last_pid is not None:
        if position != "landlord" and infoset.last_pid != "landlord":
            return []
    non_pass = [a for a in legal if a]
    if not non_pass:
        return [] if [] in legal else legal[0]
    non_pass.sort(key=lambda a: (len(a), max([card2level(c) for c in a]) if a else -1))
    return non_pass[0]


def choose_mode(rng, self_ratio, mc_ratio, old_ratio, old_available):
    total = max(0.0, self_ratio) + max(0.0, mc_ratio) + (max(0.0, old_ratio) if old_available else 0.0)
    if total <= 0:
        return "self"
    x = rng.random() * total
    if x < self_ratio:
        return "self"
    x -= self_ratio
    if x < mc_ratio:
        return "mc"
    return "old"


def choose_focus_role(rng, landlord_prob, down_prob, up_prob):
    probs = [max(0.0, landlord_prob), max(0.0, down_prob), max(0.0, up_prob)]
    total = sum(probs)
    if total <= 0:
        probs = [1.0, 1.0, 1.0]
        total = 3.0
    x = rng.random() * total
    for role, p in zip(ALL_POSITIONS, probs):
        if x < p:
            return role
        x -= p
    return "landlord_up"


def run_mixed_episode(env, current_models, old_pool, mc_opponent, device, temperature,
                      mode="self", focus_role=None, max_steps=300):
    env.reset()
    transitions = []  # (log_prob, entropy, player_index, position_name)
    trainable_roles = set(ALL_POSITIONS) if mode == "self" else {focus_role}

    for _ in range(max_steps):
        if env.done:
            break
        infoset = env.get_infoset()
        player = env.current_player
        position = ALL_POSITIONS[player]

        if position in trainable_roles:
            action, log_prob, entropy = select_nn_action(
                current_models[position], infoset, position, device,
                temperature=temperature, greedy=False, concrete=True, need_logprob=True
            )
            if log_prob is not None:
                transitions.append((log_prob, entropy, player, position))
        else:
            if mode == "mc":
                action = mc_opponent.act(infoset, position)
            elif mode == "old" and old_pool is not None and old_pool.has_any():
                action = old_pool.act(infoset, position)
            else:
                action, _, _ = select_nn_action(
                    current_models[position], infoset, position, device,
                    temperature=temperature, greedy=True, concrete=True, need_logprob=False
                )

        env.step(action)
        if env.done:
            break

    return env.winner, transitions, mode, focus_role


def evaluate_selfplay(models, device, games=200, seed=1234, max_steps=300):
    env = SimpleDoudizhuSelfPlayEnv(seed=seed)
    landlord_wins = 0
    finished = 0
    role_winner_counts = {p: 0 for p in ALL_POSITIONS}
    with torch.no_grad():
        for _ in range(games):
            env.reset()
            for _step in range(max_steps):
                infoset = env.get_infoset()
                player = env.current_player
                position = ALL_POSITIONS[player]
                action, _, _ = select_nn_action(
                    models[position], infoset, position, device,
                    temperature=1.0, greedy=True, concrete=True, need_logprob=False
                )
                env.step(action)
                if env.done:
                    break
            if env.winner is not None:
                finished += 1
                if env.winner == 0:
                    landlord_wins += 1
                    role_winner_counts["landlord"] += 1
                else:
                    role_winner_counts["landlord_down"] += 1
                    role_winner_counts["landlord_up"] += 1
    return landlord_wins / max(1, finished), finished, role_winner_counts


def parse_checkpoint_list(text, root):
    if not text:
        return []
    out = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        out.append(resolve_path(item, root))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--init-weights", default=None,
                    help="Optional role-specific weights JSON to continue training from")
    ap.add_argument("--old-checkpoints", default="",
                    help="Semicolon-separated old NN checkpoint JSON files used as opponents")
    ap.add_argument("--out", default="src/role_checkpoints/mixed_policy_weights.json",
                    help="Final exported weights JSON file. Relative paths are resolved from repo root.")
    ap.add_argument("--out-dir", default="src/role_checkpoints",
                    help="Directory to save checkpoint weights. Relative paths are resolved from repo root.")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.008)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--temperature-min", type=float, default=0.25)
    ap.add_argument("--gamma-temp", type=float, default=0.9997)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)

    # Mixed-opponent controls.
    ap.add_argument("--mix-self", type=float, default=0.30,
                    help="Ratio for pure current-NN self-play episodes")
    ap.add_argument("--mix-mc", type=float, default=0.40,
                    help="Ratio for current NN vs low-budget Monte Carlo opponents")
    ap.add_argument("--mix-old", type=float, default=0.30,
                    help="Ratio for current NN vs old NN checkpoints")
    ap.add_argument("--focus-landlord", type=float, default=0.25)
    ap.add_argument("--focus-down", type=float, default=0.50,
                    help="Sampling weight for focusing training on landlord_down in non-self modes")
    ap.add_argument("--focus-up", type=float, default=0.25)
    ap.add_argument("--mc-time-limit", type=float, default=0.04)
    ap.add_argument("--mc-root-actions", type=int, default=2)
    args = ap.parse_args()

    root = repo_root()
    out_path = resolve_path(args.out, root)
    out_dir = resolve_path(args.out_dir, root)
    init_path = resolve_path(args.init_weights, root) if args.init_weights else None
    old_paths = parse_checkpoint_list(args.old_checkpoints, root)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + 12345)

    if os.path.exists(out_dir) and os.path.isfile(out_dir):
        fallback = resolve_path("src/role_checkpoints", root)
        print("Warning: '{}' is a file, using fallback directory '{}'".format(out_dir, fallback))
        out_dir = fallback
    ensure_dir(out_dir)
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))

    print("REPO_ROOT", root)
    print("checkpoint_dir", out_dir)
    print("final_weights", out_path)
    if init_path:
        print("init_weights", init_path)
    if old_paths:
        print("old_checkpoints", old_paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = make_role_models(device, hidden_dim=args.hidden, layers=args.layers)
    if init_path:
        load_role_weights(models, init_path, device, strict=True)
        print("loaded init weights")

    opts = make_role_optimizers(models, args.lr)
    env = SimpleDoudizhuSelfPlayEnv(seed=args.seed)

    old_pool = OldNNOpponentPool(old_paths, device, args.hidden, args.layers, temperature=0.35)
    mc_opponent = LowBudgetMonteCarloOpponent(time_limit_sec=args.mc_time_limit,
                                              n_root_actions=args.mc_root_actions,
                                              seed=args.seed)

    baseline = [0.0, 0.0, 0.0]
    baseline_beta = 0.98
    temperature = args.temperature
    t0 = time.time()
    recent_finished = 0
    recent_landlord_wins = 0
    mode_count = {"self": 0, "mc": 0, "old": 0}
    focus_count = {p: 0 for p in ALL_POSITIONS}

    for ep in range(1, args.episodes + 1):
        mode = choose_mode(rng, args.mix_self, args.mix_mc, args.mix_old, old_pool.has_any())
        focus_role = None
        if mode != "self":
            focus_role = choose_focus_role(rng, args.focus_landlord, args.focus_down, args.focus_up)
            focus_count[focus_role] += 1
        mode_count[mode] += 1

        winner, transitions, _, _ = run_mixed_episode(
            env, models, old_pool, mc_opponent, device, temperature,
            mode=mode, focus_role=focus_role, max_steps=args.max_steps
        )
        if winner is None or not transitions:
            continue

        recent_finished += 1
        recent_landlord_wins += int(winner == 0)

        losses_by_role = {p: [] for p in ALL_POSITIONS}
        for log_prob, entropy, player, position in transitions:
            r = team_reward(winner, player)
            adv = r - baseline[player]
            baseline[player] = baseline_beta * baseline[player] + (1.0 - baseline_beta) * r
            losses_by_role[position].append(-log_prob * adv - args.entropy_coef * entropy)

        total_loss_value = 0.0
        updated_roles = 0
        for position in ALL_POSITIONS:
            role_losses = losses_by_role[position]
            if not role_losses:
                continue
            loss = torch.stack(role_losses).sum() / max(1, len(role_losses))
            opts[position].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[position].parameters(), 5.0)
            opts[position].step()
            total_loss_value += float(loss.item())
            updated_roles += 1

        avg_loss_value = total_loss_value / max(1, updated_roles)
        temperature = max(args.temperature_min, temperature * args.gamma_temp)

        if ep % 50 == 0:
            recent_lw = recent_landlord_wins / max(1, recent_finished)
            print("episode", ep,
                  "loss", round(avg_loss_value, 4),
                  "recent_landlord_win", round(recent_lw, 3),
                  "temp", round(temperature, 3),
                  "baseline", [round(x, 3) for x in baseline],
                  "modes", dict(mode_count),
                  "focus", dict(focus_count),
                  "time", round(time.time() - t0, 1))
            recent_finished = 0
            recent_landlord_wins = 0
            mode_count = {"self": 0, "mc": 0, "old": 0}
            focus_count = {p: 0 for p in ALL_POSITIONS}

        if args.eval_every and ep % args.eval_every == 0:
            lw, fin, role_win = evaluate_selfplay(models, device, games=args.eval_games,
                                                  seed=args.seed + 100000 + ep,
                                                  max_steps=args.max_steps)
            print("EVAL episode", ep,
                  "landlord_win", round(lw, 3),
                  "finished", fin,
                  "role_wins", role_win)

        if args.save_every and ep % args.save_every == 0:
            tmp = os.path.join(out_dir, "weights.ep%d.json" % ep)
            export_weights(models, tmp, note="mixed-opponent fine-tuning checkpoint")
            print("saved", tmp)

    export_weights(models, out_path, note="mixed-opponent fine-tuned final weights")
    print("saved", out_path)


if __name__ == "__main__":
    main()
