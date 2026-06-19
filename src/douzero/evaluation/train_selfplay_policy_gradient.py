# train_selfplay_policy_gradient.py
# ------------------------------------------------------------
# Role-specific self-play policy-gradient training for Dou Dizhu neural action scorers.
#
# No teacher / no distillation / no search enhancement.
#
# Core idea:
#   1. A rule engine only enumerates legal actions.
#   2. The neural network scores each legal action.
#   3. During training, actions are sampled from softmax(scores).
#   4. After the game ends, REINFORCE updates the network by win/loss reward.
#
# This script is intended for local training. It uses PyTorch.
# The exported weights JSON can be used by nn_policy_agent.py, which has no
# torch dependency and is Python 3.6-compatible for inference.
# ------------------------------------------------------------

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

# Try both project-style and same-folder imports.
try:
    from douzero.evaluation.high_rank_montecarlo_agent import CardCombo, CardCombinations, HighRankMonteCarloAgent, card2level
except Exception:
    from high_rank_montecarlo_agent import CardCombo, CardCombinations, HighRankMonteCarloAgent, card2level

try:
    from douzero.evaluation.nn_policy_agent import feature_vector, feature_dim
except Exception:
    try:
        from nn_policy_agent import feature_vector, feature_dim
    except Exception:
        from nn_action_scorer import feature_vector, feature_dim


ALL_POSITIONS = ["landlord", "landlord_down", "landlord_up"]


def card_sort_key(c):
    return (card2level(int(c)), int(c))


class SimpleInfoset(object):
    """Minimal infoset object compatible with feature_vector and policy agents."""

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
    """A lightweight playing-stage Dou Dizhu environment for self-play RL.

    This environment intentionally does not implement bidding. Landlord is fixed
    to player 0, and the three public cards are added to player 0. That is enough
    to train the playing policy without relying on a teacher.

    Legal action enumeration is rule-based, using CardCombinations/CardCombo from
    high_rank_montecarlo_agent. There is no heuristic search or Monte Carlo here.
    """

    def __init__(self, seed=None, landlord_fixed=True):
        self.rng = random.Random(seed)
        self.rule_engine = HighRankMonteCarloAgent(position="landlord", time_limit_sec=0.0)
        self.landlord_fixed = landlord_fixed
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
        self.history = []       # list of concrete-card actions in chronological order
        self.player_history = [] # corresponding player index for each action
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
        # Safety: if something goes wrong, at least allow a single card when leading.
        if not out and len(self.hands[player]) > 0 and len(self.last_valid_action) == 0:
            out = [[self.hands[player][0]]]
        return out

    def step(self, action):
        if self.done:
            return self.get_infoset(), 0.0, True, {"winner": self.winner}
        action = sorted(list(action or []), key=card_sort_key)

        # Do not crash training on an occasional numerical/action mismatch.
        # If illegal, replace by a safe legal action.
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
            # Passing is legal only when following. After two passes, control returns
            # to the last valid player and a new leading round begins.
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


def team_reward(winner, player):
    """+1 if player's side wins, -1 otherwise."""
    if winner == 0:
        return 1.0 if player == 0 else -1.0
    return -1.0 if player == 0 else 1.0


def select_action(model, infoset, position, device, temperature=1.0,
                  greedy=False, concrete=True):
    legal = infoset.legal_actions
    if not legal:
        return [], None, None
    if len(legal) == 1:
        return legal[0], None, None

    feats = [feature_vector(infoset, position, a, concrete=concrete) for a in legal]
    x = torch.tensor(feats, dtype=torch.float32, device=device)
    scores = model(x)

    # Direct finishing move should not be missed. This is a legality/tactical prior,
    # not a search step.
    hand_len = len(infoset.player_hand_cards)
    finish_indices = [i for i, a in enumerate(legal) if a and len(a) == hand_len]
    if finish_indices:
        idx = finish_indices[0]
        if greedy:
            return legal[idx], None, None
        # Still create a nearly deterministic log_prob for training stability.
        logits = scores / max(1e-6, float(temperature))
        dist = torch.distributions.Categorical(logits=logits)
        action_index = torch.tensor(idx, dtype=torch.long, device=device)
        return legal[idx], dist.log_prob(action_index), dist.entropy()

    logits = scores / max(1e-6, float(temperature))
    dist = torch.distributions.Categorical(logits=logits)
    if greedy:
        idx = int(torch.argmax(logits).item())
        return legal[idx], None, None
    action_index = dist.sample()
    idx = int(action_index.item())
    return legal[idx], dist.log_prob(action_index), dist.entropy()



def make_role_models(device, hidden_dim=128, layers=2):
    """Create one independent policy network for each role.

    This is the main change from the previous version:
        landlord, landlord_down, landlord_up do NOT share parameters.
    """
    models = {}
    for p in ALL_POSITIONS:
        models[p] = ActionScoringNet(feature_dim(), hidden_dim=hidden_dim, layers=layers).to(device)
    return models


def make_role_optimizers(models, lr):
    return {p: optim.Adam(models[p].parameters(), lr=lr) for p in ALL_POSITIONS}


def run_episode(env, models, device, temperature, max_steps=300):
    env.reset()
    # (log_prob, entropy, player_index, position_name)
    transitions = []

    for _ in range(max_steps):
        if env.done:
            break
        infoset = env.get_infoset()
        player = env.current_player
        position = ALL_POSITIONS[player]
        model = models[position]
        action, log_prob, entropy = select_action(
            model, infoset, position, device,
            temperature=temperature, greedy=False, concrete=True
        )
        if log_prob is not None:
            transitions.append((log_prob, entropy, player, position))
        env.step(action)
        if env.done:
            break

    return env.winner, transitions


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
                action, _, _ = select_action(
                    models[position], infoset, position, device,
                    temperature=1.0, greedy=True, concrete=True
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


def _linear_layers_to_lists(model):
    weights = []
    biases = []
    for m in model.net:
        if isinstance(m, nn.Linear):
            weights.append(m.weight.detach().cpu().tolist())
            biases.append(m.bias.detach().cpu().tolist())
    return weights, biases


def export_weights(models, out_path):
    """Export role-specific networks to one JSON file.

    The inference agent keeps the same interface:
        NeuralPolicyAgent(position, weights_path=...)
    It automatically selects the sub-network by position.
    """
    role_models = {}
    for p in ALL_POSITIONS:
        weights, biases = _linear_layers_to_lists(models[p])
        role_models[p] = {
            "input_dim": feature_dim(),
            "weights": weights,
            "biases": biases,
        }

    obj = {
        "input_dim": feature_dim(),
        "role_models": role_models,
        "roles": list(ALL_POSITIONS),
        "trained_by": "role_specific_selfplay_policy_gradient",
        "card_encoding": "botzone_concrete_0_53_or_env_rank_if_concrete_false",
        "note": "No teacher, no distillation, no search. Three independent policies are trained for landlord, landlord_down, and landlord_up.",
    }

    parent = os.path.dirname(os.path.abspath(out_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--out", default="role_selfplay_policy_weights.json",
                    help="Final exported weights JSON file")
    ap.add_argument("--out-dir", default="role_selfplay_policy_checkpoints",
                    help="Directory to save checkpoint weights")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temperature-min", type=float, default=0.35)
    ap.add_argument("--gamma-temp", type=float, default=0.9995,
                    help="Temperature exponential decay per episode")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create checkpoint directory. If it conflicts with a file, use a fallback.
    if os.path.exists(args.out_dir) and os.path.isfile(args.out_dir):
        print("Warning: '{}' is a file, using fallback directory 'role_selfplay_policy_checkpoints'".format(args.out_dir))
        args.out_dir = "role_selfplay_policy_checkpoints"
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = make_role_models(device, hidden_dim=args.hidden, layers=args.layers)
    opts = make_role_optimizers(models, args.lr)
    env = SimpleDoudizhuSelfPlayEnv(seed=args.seed)

    # Per-player moving baseline, only for variance reduction. It is not a teacher.
    baseline = [0.0, 0.0, 0.0]
    baseline_beta = 0.98

    temperature = args.temperature
    t0 = time.time()
    landlord_win_count = 0
    finished_count = 0

    for ep in range(1, args.episodes + 1):
        winner, transitions = run_episode(env, models, device, temperature, max_steps=args.max_steps)
        if winner is None or not transitions:
            continue

        finished_count += 1
        landlord_win_count += int(winner == 0)

        # Group policy-gradient losses by role, because each role has its own model.
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
            recent_lw = landlord_win_count / max(1, finished_count)
            print("episode", ep,
                  "loss", round(avg_loss_value, 4),
                  "recent_landlord_win", round(recent_lw, 3),
                  "temp", round(temperature, 3),
                  "baseline", [round(x, 3) for x in baseline],
                  "roles", "separate",
                  "time", round(time.time() - t0, 1))
            landlord_win_count = 0
            finished_count = 0

        if args.eval_every and ep % args.eval_every == 0:
            lw, fin, role_win = evaluate_selfplay(
                models, device, games=args.eval_games,
                seed=args.seed + 100000 + ep,
                max_steps=args.max_steps
            )
            print("EVAL episode", ep,
                  "landlord_win", round(lw, 3),
                  "finished", fin,
                  "role_wins", role_win)

        if args.save_every and ep % args.save_every == 0:
            tmp = os.path.join(args.out_dir, "weights.ep%d.json" % ep)
            export_weights(models, tmp)
            print("saved", tmp)

    export_weights(models, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
