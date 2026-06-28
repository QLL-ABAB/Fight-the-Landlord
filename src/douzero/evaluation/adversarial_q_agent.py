# adversarial_q_agent.py
# ------------------------------------------------------------
# Adversarial search with a learned Q-network leaf evaluator.
#
# Idea:
#   The original adversarial_agent uses hand-written heuristics at search
#   leaves. If a checkpoint has learned Q(s, a), either from attention_dou or
#   feature-based approximate Q-learning, we can use it as a stronger value
#   estimate:
#
#       leaf_value(s) ~= max_a Q_theta(s, a)
#
#   The search logic is unchanged: hidden-card determinization, shallow
#   adversarial/team search, and root action scoring all stay in the parent
#   AdversarialSearchAgent. Only evaluate_sim_state() is replaced. If the Q
#   model cannot encode a simulated leaf state, we fall back to the original
#   heuristic evaluation.
# ------------------------------------------------------------

from collections import Counter

try:
    from douzero.evaluation.adversarial_agent import (
        ALL_POSITIONS,
        RealCard2EnvCard,
        AdversarialSearchAgent,
    )
    from douzero.evaluation.approx_qlearning_agent import resolve_approx_q_model_path
    from douzero.evaluation.attention_dou_agent import resolve_attention_dou_model_path
    try:
        from douzero.rl.approx_qlearning import (
            ApproxQModel,
            features_for_actions as approx_features_for_actions,
            prune_legal_actions as approx_prune_legal_actions,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "douzero.rl.approx_qlearning":
            raise
        from douzero.rl.approx_qlearning_fasle import (
            ApproxQModel,
            features_for_actions as approx_features_for_actions,
            prune_legal_actions as approx_prune_legal_actions,
        )
    from douzero.rl.attention_dou import AttentionDouModel, tokens_for_infoset
except Exception:
    from adversarial_agent import (
        ALL_POSITIONS,
        RealCard2EnvCard,
        AdversarialSearchAgent,
    )
    from approx_qlearning_agent import resolve_approx_q_model_path
    from attention_dou_agent import resolve_attention_dou_model_path
    from approx_qlearning_fasle import (
        ApproxQModel,
        features_for_actions as approx_features_for_actions,
        prune_legal_actions as approx_prune_legal_actions,
    )
    from attention_dou import AttentionDouModel, tokens_for_infoset


class _LeafInfoSet(object):
    """Minimal infoset accepted by douzero.env.env.get_obs()."""

    def __init__(
        self,
        player_position,
        player_hand_cards,
        legal_actions,
        last_move,
        last_move_dict,
        num_cards_left_dict,
        played_cards,
        card_play_action_seq,
        bomb_num=0,
    ):
        self.player_position = player_position
        self.position = player_position
        self.player_hand_cards = list(player_hand_cards)
        self.legal_actions = [list(a) for a in legal_actions]
        self.last_move = list(last_move)
        self.last_move_dict = {p: list(last_move_dict.get(p, [])) for p in ALL_POSITIONS}
        self.num_cards_left_dict = dict(num_cards_left_dict)
        self.num_cards_left = dict(num_cards_left_dict)
        self.player_num_cards_left = dict(num_cards_left_dict)
        self.played_cards = {p: list(played_cards.get(p, [])) for p in ALL_POSITIONS}
        self.card_play_action_seq = [list(a) for a in card_play_action_seq]
        self.bomb_num = int(bomb_num)

        # DouZero features use "other_hand_cards" as a compact public/hidden
        # aggregate. At simulated leaves we have a determinized full state, so
        # using the other two hands is consistent with that sampled world.
        self.other_hand_cards = []


class AdversarialQSearchAgent(AdversarialSearchAgent):
    """Adversarial sampled search whose leaf value comes from a learned Q model."""

    def __init__(self, position, model_path=None, debug=False, seed=None):
        super().__init__(position=position, debug=debug, seed=seed)
        self.name = "AdversarialQLeafSearch"
        self.q_backend = None
        self.q_model_path = None
        self.q_model = None
        self.max_candidate_actions = 64
        self.load_q_model(model_path)
        self.cfg.update({
            # Multiplier applied to max_a Q_theta(s,a) before returning it to
            # the adversarial search. Keep this separate because older
            # checkpoints may have different value scales.
            "q_leaf_scale": self.default_q_leaf_scale(),
            # Mix Q leaf with the old heuristic leaf:
            #   value = mix * q_leaf + (1 - mix) * heuristic_leaf
            "q_leaf_mix": 1.0,
        })
        self.stats.update({
            "q_leaf_calls": 0,
            "q_leaf_fallbacks": 0,
            "q_leaf_cache_hits": 0,
        })
        self._q_leaf_cache = {}

    def load_q_model(self, model_path):
        if model_path:
            try:
                approx_path = resolve_approx_q_model_path(model_path)
                self.q_model = ApproxQModel.load(approx_path, device="cpu")
                self.q_model_path = approx_path
                self.q_backend = "approxq"
                self.max_candidate_actions = int(
                    self.q_model.metadata.get("max_candidate_actions", 64)
                )
                self.name = "AdversarialApproxQLeafSearch"
                return
            except Exception:
                pass

        attention_path = resolve_attention_dou_model_path(model_path)
        self.q_model = AttentionDouModel.load(
            attention_path,
            device="cpu",
            load_optimizer=False,
        )
        self.q_model_path = attention_path
        self.q_backend = "attention_dou"

    def default_q_leaf_scale(self):
        if self.q_backend == "approxq":
            # ApproxQ checkpoints in this repo are usually trained on logADP-like
            # rewards, whose raw value is much smaller than adversarial_agent's
            # tactical bonuses. This keeps the learned leaf relevant without
            # making it swamp terminal wins/losses.
            return 30.0
        return 1.0

    def evaluate_sim_state(self, sim):
        mix = max(0.0, min(1.0, float(self.cfg.get("q_leaf_mix", 1.0))))
        heuristic_value = None
        try:
            q_value = self.q_leaf_value(sim)
            if mix >= 1.0:
                return q_value
            heuristic_value = super().evaluate_sim_state(sim)
            return mix * q_value + (1.0 - mix) * heuristic_value
        except Exception as exc:
            self.stats["q_leaf_fallbacks"] = self.stats.get("q_leaf_fallbacks", 0) + 1
            self.last_error = repr(exc)
            if heuristic_value is None:
                heuristic_value = super().evaluate_sim_state(sim)
            return heuristic_value

    def q_leaf_value(self, sim):
        self.stats["q_leaf_calls"] = self.stats.get("q_leaf_calls", 0) + 1
        cache_key = (
            sim.current_player,
            sim.hands_tuple,
            sim.last_move,
            sim.last_pid,
            sim.pass_count,
        )
        if cache_key in self._q_leaf_cache:
            self.stats["q_leaf_cache_hits"] = self.stats.get("q_leaf_cache_hits", 0) + 1
            return self._q_leaf_cache[cache_key]

        actor = sim.current_player
        actions_str = self.legal_actions_for_sim(sim)
        if not actions_str:
            return super().evaluate_sim_state(sim)

        infoset = self.sim_to_infoset(sim, actor, actions_str)
        if self.q_backend == "approxq":
            value = self.approxq_leaf_value(actor, infoset)
        else:
            value = self.attention_leaf_value(actor, infoset)

        value = value * float(self.cfg.get("q_leaf_scale", 1.0))
        value = value if self.same_team(actor, self.position) else -value
        self._q_leaf_cache[cache_key] = value
        return value

    def attention_leaf_value(self, actor, infoset):
        actions, tokens = tokens_for_infoset(actor, infoset)
        if not actions:
            raise ValueError("No attention_dou leaf actions")
        q_values = self.q_model.q_values(actor, tokens)
        if len(q_values) == 0:
            raise ValueError("Empty attention_dou q_values")
        return float(q_values.max())

    def approxq_leaf_value(self, actor, infoset):
        actions = approx_prune_legal_actions(
            actor,
            infoset,
            self.max_candidate_actions,
        )
        if not actions:
            raise ValueError("No approxq leaf actions")
        features = approx_features_for_actions(
            actor,
            infoset,
            actions,
            self.q_model.device,
            self.q_model.feature_mode,
        )
        return float(self.q_model.best_value(actor, features))

    def sim_to_infoset(self, sim, actor, actions_str):
        hands = self.tuple_to_hands(sim.hands_tuple)
        player_hand_cards = self.str_to_env_cards(hands[actor])
        legal_actions = [self.str_to_env_cards(a) for a in actions_str]
        last_move = self.str_to_env_cards(sim.last_move)
        num_cards_left_dict = {p: len(hands[p]) for p in ALL_POSITIONS}

        last_move_dict = {p: [] for p in ALL_POSITIONS}
        if sim.last_pid is not None and sim.last_move:
            last_move_dict[sim.last_pid] = last_move

        played_cards = self.estimate_played_cards(hands)
        card_play_action_seq = [last_move] if last_move else []
        bomb_num = self.estimate_bomb_num(played_cards, last_move)

        infoset = _LeafInfoSet(
            player_position=actor,
            player_hand_cards=player_hand_cards,
            legal_actions=legal_actions,
            last_move=last_move,
            last_move_dict=last_move_dict,
            num_cards_left_dict=num_cards_left_dict,
            played_cards=played_cards,
            card_play_action_seq=card_play_action_seq,
            bomb_num=bomb_num,
        )
        infoset.other_hand_cards = []
        for p in ALL_POSITIONS:
            if p != actor:
                infoset.other_hand_cards.extend(self.str_to_env_cards(hands[p]))
        return infoset

    def str_to_env_cards(self, cards):
        return [RealCard2EnvCard[c] for c in self.sort_card_str(cards)]

    def estimate_played_cards(self, hands):
        # The simulated state does not preserve exact per-player history. We can
        # still provide a useful aggregate by subtracting current hands from the
        # full deck and assigning that aggregate to all played piles conservatively
        # as empty except for the landlord pile. This keeps get_obs well-formed;
        # q-leaf fallback handles any mismatch.
        remaining = Counter()
        for hand in hands.values():
            remaining.update(hand)

        full = Counter()
        for c in ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2"]:
            full[c] = 4
        full["B"] = 1
        full["R"] = 1

        played = []
        for c, total in full.items():
            n = max(0, total - remaining.get(c, 0))
            played.extend([c] * n)

        return {
            "landlord": self.str_to_env_cards("".join(played)),
            "landlord_down": [],
            "landlord_up": [],
        }

    def estimate_bomb_num(self, played_cards, last_move):
        count = 0
        for cards in played_cards.values():
            ranks = Counter(cards)
            count += sum(1 for rank, n in ranks.items() if n >= 4 and rank not in (20, 30))
        if len(last_move) == 2 and sorted(last_move) == [20, 30]:
            count += 1
        return min(count, 14)


# Config-facing alias.
AdversarialSearchAgentQLeaf = AdversarialQSearchAgent
