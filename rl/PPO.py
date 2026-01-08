from __future__ import annotations

import argparse
import logging
import os
import random
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

if TYPE_CHECKING:
    from game import GameProtocol
from game import Game
from gnn.encode import SAMPLE_ENC, encode_game_to_graph
from gnn.game_generation import randomize_start
from gnn.model import GNNEval
from models import Move
from players.base import Player, RandomPlayer

logger = logging.getLogger("rl_population")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler("population_train.log", mode="a"), logging.StreamHandler()],
    )


def log(msg: str) -> None:
    logger.info(msg)


class PPOPlayer(Player):
    def __init__(self, player_number: int, model: nn.Module, device: str = "cpu"):
        super().__init__(player_number)
        self.model = model
        self.device = device

    def get_move(self, game: GameProtocol) -> Move:
        legal_moves = list(game.get_possible_moves(self.number))
        enc = encode_game_to_graph(game)
        data = enc.data.to(self.device)
        action_idx, _, _ = select_action(self.model, data, legal_moves)
        return legal_moves[action_idx]


class PPOGNNPolicy(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        global_feat_dim: int,
        # max_action_dim removed; policy scores per-legal-move instead
        hidden: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__() # type: ignore
        self.gnn = GNNEval(
            node_feat_dim=node_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden=hidden,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
        # move_head scores a candidate move given the node embedding, global features and move-type one-hot
        self.move_head = nn.Sequential(
            nn.Linear(hidden + global_feat_dim + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Linear(hidden + global_feat_dim, 1)
        self.hidden = hidden

    def _compute_node_embeddings(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr, batch, g = data.x, data.edge_index, data.edge_attr, data.batch, data.global_feats
        h = x
        for i, conv in enumerate(self.gnn.convs):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            # try to apply the same norm/relu/dropout behavior as GNNEval
            if i < len(self.gnn.norms):
                h = self.gnn.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.gnn.dropout_p, training=self.training)
            if h.shape == h_in.shape:
                h = h + h_in
        h = cast(torch.Tensor, h)
        pooled = global_mean_pool(h, batch)
        return h, torch.cat([pooled, g], dim=-1)

    def score_moves(self, data: Data, legal_moves: List[Move]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits for each legal_move and the scalar value for the state."""
        node_coords = getattr(data, "node_coords", None)
        h, features = self._compute_node_embeddings(data)
        # value
        value = self.value_head(features).squeeze(0)

        # prepare per-move features
        move_feats: list[torch.Tensor] = []
        g = data.global_feats
        for mv in legal_moves:
            if mv.t == "P":
                node_emb = torch.zeros(self.hidden, device=h.device)
                mv_type = torch.tensor([0.0, 0.0, 1.0], device=h.device)
            else:
                # find node index matching move coordinate
                if node_coords is None:
                    raise RuntimeError("Data missing node_coords")
                nc = node_coords
                if nc.dim() == 2 and nc.size(1) == 2:
                    coord_tensor = torch.tensor(list(mv.c), device=nc.device)
                    eq = (nc == coord_tensor).all(dim=1)
                else:
                    coord_hash = mv.c[0] * 10000 + mv.c[1]
                    eq = nc == coord_hash
                idx = eq.nonzero()
                if idx.numel() == 0:
                    node_emb = torch.zeros(self.hidden, device=h.device)
                else:
                    node_idx = int(idx[0, 0].item())
                    node_emb = h[node_idx]
                mv_type = torch.tensor([1.0 if mv.t == "R" else 0.0, 1.0 if mv.t == "S" else 0.0, 0.0], device=h.device)
            move_feat = torch.cat([node_emb, g.squeeze(0), mv_type], dim=-1)
            move_feats.append(move_feat)

        if move_feats:
            mv_tensor = torch.stack(move_feats, dim=0)
            logits = self.move_head(mv_tensor).squeeze(-1)
        else:
            logits = torch.tensor([], device=h.device)
        return logits, value


def select_action(model: nn.Module, obs: Data, legal_moves: List[Any]) -> Tuple[int, torch.Tensor, torch.Tensor]:
    # Prefer the newer `score_moves` API; fall back to older model(obs) callable.
    if isinstance(model, PPOGNNPolicy):
        logits, value = model.score_moves(obs, legal_moves)
    elif callable(model):
        logits, value = model(obs)
        logits = logits[: len(legal_moves)]
    else:
        raise RuntimeError("Model does not implement score_moves and is not callable")

    if logits.numel() == 0:
        raise RuntimeError("No legal moves to select from")
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action_idx = dist.sample()
    log_prob = cast(torch.Tensor, dist.log_prob(action_idx))
    value = cast(torch.Tensor, value)
    return int(action_idx.item()), log_prob, value


def run_episode(
    model: Optional[nn.Module],
    device: str = "cpu",
    gamma: float = 0.99,
    max_episode_length: Optional[int] = None,
) -> Tuple[List[Data], List[List[Move]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    # If model is None, play a game between RandomPlayers only and return empty tensors
    if model is None:
        players: list[Player] = [RandomPlayer(0), RandomPlayer(1)]
        game = Game(players)
        randomize_start(game)
        while game.winner is None:
            p = game.players[game.current_player]
            move = p.get_move(game)
            game.do_move(p.number, move)
        return [], [], torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float32), torch.tensor(
            [], dtype=torch.float32), torch.tensor([], dtype=torch.float32), 0

    players: list[Player] = [PPOPlayer(0, model, device), PPOPlayer(1, model, device)]
    game = Game(players)
    randomize_start(game)
    obs_list: List[Data] = []
    action_list: List[int] = []
    logprob_list: List[torch.Tensor] = []
    value_list: List[torch.Tensor] = []
    reward_list: List[float] = []
    legal_moves_list: List[List[Move]] = []
    max_legal = 0
    done = False
    while not done:
        player = game.players[game.current_player]
        legal_moves = list(game.get_possible_moves(player.number))
        max_legal = max(max_legal, len(legal_moves))
        enc = encode_game_to_graph(game)
        data = enc.data.to(device)
        with torch.no_grad():
            action_idx, log_prob, value = select_action(model, data, legal_moves)
        action = legal_moves[action_idx]
        game.do_move(player.number, action)
        # increment move counter and check max length
        if max_episode_length is not None:
            current_moves = len(action_list) + 1
        else:
            current_moves = None
        reward = 0.0
        if game.winner is not None:
            done = True
            reward = 1.0 if game.winner == player.number else -1.0
        elif max_episode_length is not None and current_moves is not None and current_moves >= max_episode_length:
            # reached maximum allowed moves -> treat as draw (reward 0)
            done = True
            reward = 0.0
        obs_list.append(data)
        legal_moves_list.append(legal_moves)
        action_list.append(int(action_idx))
        logprob_list.append(log_prob.detach().cpu())
        value_list.append(value.detach().cpu())
        reward_list.append(float(reward))

    # Compute returns
    returns: List[float] = []
    r_return: float = 0.0
    for r in reversed(reward_list):
        r_return = r + gamma * r_return
        returns.insert(0, r_return)

    if returns:
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    else:
        returns_tensor = torch.tensor([], dtype=torch.float32, device=device)
    values_tensor = torch.stack(value_list) if value_list else torch.tensor([], dtype=torch.float32, device=device)
    logprobs_tensor = torch.stack(logprob_list) if logprob_list else torch.tensor([], dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(action_list, dtype=torch.long, device=device) if action_list else torch.tensor(
        [], dtype=torch.long, device=device
    )
    return obs_list, legal_moves_list, actions_tensor, logprobs_tensor, values_tensor, returns_tensor, max_legal


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_list: List[Data],
    legal_moves_list: List[List[Move]],
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    max_legal: int,
    clip_epsilon: float = 0.1,
    vf_coef: float = 0.5,
    ent_coef: float = 0.5,
    epochs: int = 4,
    max_grad_norm: float = 0.5,
) -> Tuple[float, float, float, float]:
    # Compute advantage and normalize
    advantages = returns - values.detach()
    if advantages.numel() > 1:
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    best_loss = 0.0
    policy_loss_val = 0.0
    value_loss_val = 0.0
    entropy_val = 0.0

    for _ in range(epochs):
        new_logprobs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        # recompute logits for each observation
        for obs, legal_moves, action in zip(obs_list, legal_moves_list, actions):
            if isinstance(model, PPOGNNPolicy):
                logits, _ = model.score_moves(obs, legal_moves)
            elif callable(model):
                logits, _ = model(obs)
                logits = logits[:len(legal_moves)]
            else:
                raise RuntimeError("Model does not implement score_moves and is not callable")
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_logprobs.append(cast(torch.Tensor, dist.log_prob(action)))
            entropies.append(dist.entropy())

        ratio = (torch.stack(new_logprobs) - logprobs.detach()).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        # Value loss
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        # Entropy bonus
        entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0)
        if hasattr(model, "ent_coef"):
            ent_coef = getattr(model, "ent_coef")
        loss: torch.Tensor = policy_loss + vf_coef * value_loss - ent_coef * entropy
        optimizer.zero_grad()
        loss.backward()  # type: ignore
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        best_loss = loss.item()
        policy_loss_val = policy_loss.item()
        value_loss_val = value_loss.item()
        entropy_val = entropy.item()

    return best_loss, policy_loss_val, value_loss_val, entropy_val


def run_episode_and_update(
    agent: Optional[nn.Module],
    optimizer: Optional[torch.optim.Optimizer],
    label: str,
    gen: int,
    ep: int,
    device: str,
    log_fn: Callable[[str], None],
    max_episode_length: Optional[int] = None,
    ppo_epochs: int = 4,
) -> tuple[float, torch.Tensor]:
    obs_list, legal_moves_list, actions, logprobs, values, returns, max_legal = run_episode(
        agent, device=device, max_episode_length=max_episode_length
    )
    # removed verbose obs_list debugging logs to reduce clutter
    actions = actions.detach().cpu()
    logprobs = logprobs.detach().cpu()
    values = values.detach().cpu()
    returns = returns.detach().cpu()
    loss = 0.0
    if agent is not None and optimizer is not None and actions.numel() > 0:
        loss, _, _, _ = ppo_update(
            agent,
            optimizer,
            obs_list,
            legal_moves_list,
            actions,
            logprobs,
            values,
            returns,
            max_legal,
            epochs=ppo_epochs,
        )
    return loss, returns


def evaluate_agents(
    agents: List[Optional[PPOGNNPolicy]],
    device: str,
    eval_games: int,
    gen: int,
    log_fn: Callable[[str], None],
    max_episode_length: Optional[int] = None,
) -> None:
    log_fn(f"[Eval] Evaluating all agents vs RandomPlayer after generation {gen+1}")
    for i, agent in enumerate(agents):
        wins = 0
        for _ in range(eval_games):
            if agent is None:
                players: list[Player] = [RandomPlayer(0), RandomPlayer(1)]
            else:
                players = [PPOPlayer(0, agent, device), RandomPlayer(1)]
            game = Game(players)
            randomize_start(game)
            move_count = 0
            while game.winner is None and (max_episode_length is None or move_count < max_episode_length):
                p = game.players[game.current_player]
                move = p.get_move(game)
                game.do_move(p.number, move)
                move_count += 1
            if game.winner == 0:
                wins += 1
        win_rate = wins / eval_games
        log_fn(f"  Agent {i} win rate vs RandomPlayer: {win_rate:.2f}")


def replace_weakest_and_mutate(
    agents: List[Optional[PPOGNNPolicy]],
    optimizers: List[Optional[torch.optim.Optimizer]],
    scores: List[float],
    population_size: int,
    log_fn: Callable[[str], None],
    gen: int,
    mutation_noise: float = 0.02,
    keep_exact_prob: float = 0.2,
) -> None:
    log_fn(f"Generation {gen+1} scores: {scores}")
    weakest = scores.index(min(scores))
    model_indices = [i for i, a in enumerate(agents) if a is not None]
    if not model_indices:
        log_fn("No trainable agents available to mutate.")
        return
    strongest = max(model_indices, key=lambda i: scores[i])
    log_fn(f"Replacing agent {weakest} with mutated copy of agent {strongest}")
    source_agent = agents[strongest]
    if source_agent is None:
        log_fn("Strongest agent is not trainable; skipping mutation.")
        return

    # determine safe integer feature dims with fallbacks
    node_feat_dim = getattr(source_agent.gnn, "node_feat_dim", None)
    if node_feat_dim is None:
        node_feat_dim = getattr(getattr(source_agent.gnn, "convs", [None])[0], "in_channels", None)
    if node_feat_dim is None:
        node_feat_dim = int(cast(torch.Tensor, SAMPLE_ENC.data.x).size(1))
    global_feat_dim = getattr(source_agent.gnn, "global_feat_dim", None)
    if global_feat_dim is None:
        global_feat_dim = getattr(getattr(source_agent.gnn, "convs", [None])[0], "in_channels", None)
    if global_feat_dim is None:
        global_feat_dim = int(cast(torch.Tensor, SAMPLE_ENC.data.global_feats).size(1))

    if agents[weakest] is None:
        new_agent = PPOGNNPolicy(
            node_feat_dim=int(node_feat_dim),
            global_feat_dim=int(global_feat_dim),
        )
        new_agent.load_state_dict(source_agent.state_dict())
        agents[weakest] = new_agent
    # narrow to concrete type for static checkers and ensure state dict loaded
    m_weak = cast(PPOGNNPolicy, agents[weakest])
    m_weak.load_state_dict(source_agent.state_dict())
    # Mutate parameters with gaussian noise, but sometimes keep an exact copy
    if random.random() < keep_exact_prob:
        # exact copy (no mutation)
        pass
    else:
        for param in m_weak.parameters():
            noise = torch.normal(mean=0.0, std=mutation_noise, size=param.size(), device=param.device)
            param.data.add_(noise)
    old_lr = None
    opt_strong = optimizers[strongest]
    if opt_strong is not None:
        old_lr = opt_strong.param_groups[0]["lr"]
    # Slight LR perturbation (smaller variance)
    new_lr = max(1e-6, min(1e-2, (old_lr if old_lr is not None else 1e-4) * (1 + random.uniform(-0.1, 0.1))))
    optimizers[weakest] = torch.optim.Adam(m_weak.parameters(), lr=new_lr)
    old_ent_coef = getattr(source_agent, "ent_coef", 0.5)
    new_ent_coef = max(0.01, min(1.0, old_ent_coef * (1 + random.uniform(-0.2, 0.2))))
    setattr(m_weak, "ent_coef", new_ent_coef)
    log_fn(f"Mutated agent {weakest}: lr={new_lr:.2e}, ent_coef={new_ent_coef:.3f}")
    best_idx = max(model_indices, key=lambda i: scores[i])
    best_agent = cast(PPOGNNPolicy, agents[best_idx])
    # Save model and optimizer state (when available) so training can be resumed exactly.
    ckpt: dict[str, object] = {"model_state_dict": best_agent.state_dict()}
    opt = optimizers[best_idx] if 0 <= best_idx < len(optimizers) else None
    if opt is not None:
        try:
            ckpt["optimizer_state_dict"] = opt.state_dict()
        except Exception:
            pass
    torch.save(ckpt, "checkpoints/rl_population_best.pt")
    log_fn(f"Saved best population agent + optimizer (if available) to checkpoints/rl_population_best.pt (generation {gen+1})")

# best values for ent_coef (0.01), warmup_gens (3), and ppo_epochs (8) were found by a hyperparameter grid search,
# probably do not change
def population_train(
    population_size: int = 10,
    n_generations: int = 10,
    episodes_per_generation: int = 100,
    device: str = "cpu",
    eval_interval: int = 5,
    eval_games: int = 50,
    include_random: bool = False,
    load_model_path: Optional[str] = None,
    init_from_gnn: Optional[str] = None,
    freeze_gnn_generations: int = 0,
    head_lr: float = 1e-4,
    full_lr: float = 1e-5,
    continue_training: bool = False,
    max_episode_length: Optional[int] = None,
    ent_coef: float = 0.01,
    mutation_noise: float = 0.01,
    mutation_keep_exact_prob: float = 0.3,
    warmup_generations: int = 3,
    warmup_episodes: int = 100,
    warmup_lr: float = 3e-4,
    warmup_temp: float = 1.0,
    ppo_epochs: int = 8,
):
    # Initialize population of agents
    agents: List[Optional[PPOGNNPolicy]] = []
    optimizers: List[Optional[torch.optim.Optimizer]] = []
    if SAMPLE_ENC.data.x is None or SAMPLE_ENC.data.global_feats is None:
        raise ValueError("Node features or global features are missing in the encoded graph.")
    node_feat_dim = SAMPLE_ENC.data.x.size(1)
    global_feat_dim = SAMPLE_ENC.data.global_feats.size(1)
    num_models = population_size - (1 if include_random else 0)
    for _ in range(num_models):
        model = PPOGNNPolicy(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim)
        model.to(device)
        # set exploration entropy coefficient on the model for use in PPO loss
        try:
            setattr(model, "ent_coef", float(ent_coef))
        except Exception:
            pass
        agents.append(model)
        optimizers.append(None)
    if include_random:
        agents.append(None)
        optimizers.append(None)

    def make_optimizer_for_model(m: Optional[PPOGNNPolicy], lr: float) -> Optional[torch.optim.Optimizer]:
        if m is None:
            return None
        params = [p for p in m.parameters() if p.requires_grad]
        if not params:
            return None
        return torch.optim.Adam(params, lr=lr)

    # Load PPO checkpoint if continuing. If --continue-training is set we load into
    # all trainable agents so the population can continue from the same weights.
    if load_model_path is not None and (init_from_gnn is None or continue_training):
        if os.path.exists(load_model_path):
            try:
                ckpt = torch.load(load_model_path, map_location=device)
                # support both raw state_dict and wrapped checkpoints
                if isinstance(ckpt, dict) and ("model_state_dict" in ckpt or "state_dict" in ckpt):
                    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
                else:
                    state_dict = ckpt

                if continue_training:
                    loaded = 0
                    for i, a in enumerate(agents):
                        if a is None:
                            continue
                        try:
                            a.load_state_dict(state_dict) # type: ignore
                            loaded += 1
                        except Exception as e:
                            log(f"Failed to load PPO checkpoint into agent {i}: {e}")
                    log(f"Loaded PPO checkpoint from {load_model_path} into {loaded} agents")
                else:
                    # default: load into agent 0 if available
                    log(f"Loading PPO model weights from {load_model_path} into agent 0")
                    if len(agents) > 0 and agents[0] is not None:
                        try:
                            agents[0].load_state_dict(state_dict) # type: ignore
                        except Exception as e:
                            log(f"Failed to load PPO checkpoint into agent 0: {e}")
                    else:
                        log("No agent[0] available to load PPO checkpoint into; skipping.")
            except Exception as e:
                log(f"Failed to load PPO checkpoint: {e}")

    # Load supervised GNN into agents if requested
    if init_from_gnn is not None and os.path.exists(init_from_gnn):
        log(f"Initializing agents' GNN from supervised checkpoint {init_from_gnn}")
        sup_state = torch.load(init_from_gnn, map_location=device)
        for a in agents:
            if a is None:
                continue
            try:
                a.gnn.load_state_dict(sup_state)
            except Exception:
                try:
                    gnn_state = {k.replace("gnn.", ""): v for k, v in sup_state.items() if k.startswith("gnn.")}
                    if gnn_state:
                        a.gnn.load_state_dict(gnn_state)
                    else:
                        a.gnn.load_state_dict(sup_state)
                except Exception as e:
                    log(f"Failed to load supervised GNN into agent: {e}")

    # Freeze GNNs for initial phase if requested
    gnn_frozen = False
    if freeze_gnn_generations > 0:
        gnn_frozen = True
        log(f"Freezing agents' GNNs for first {freeze_gnn_generations} generations")
        for a in agents:
            if a is None:
                continue
            for p in a.gnn.parameters():
                p.requires_grad = False

    # Create optimizers
    for idx, a in enumerate(agents):
        if a is None:
            optimizers[idx] = None
        else:
            optimizers[idx] = make_optimizer_for_model(a, head_lr if gnn_frozen else full_lr)

    # If continuing from a checkpoint that contains optimizer state, try to
    # restore it into a matching optimizer (prefer agent 0). This makes
    # --continue-training preserve the optimizer state when possible.
    if load_model_path is not None and continue_training and os.path.exists(load_model_path):
        try:
            ckpt = torch.load(load_model_path, map_location=device)
            saved_opt_state = None
            if isinstance(ckpt, dict):
                saved_opt_state = ckpt.get("optimizer_state_dict")
            if saved_opt_state is not None:
                restored = False
                # Prefer optimizer 0
                candidates = list(range(len(optimizers)))
                for i in candidates:
                    opt = optimizers[i]
                    if opt is None:
                        continue
                    try:
                        if len(opt.state_dict().get("param_groups", [])) != len(saved_opt_state.get("param_groups", [])):
                            continue
                        opt.load_state_dict(saved_opt_state)
                        log(f"Restored optimizer state into optimizer {i} from {load_model_path}")
                        restored = True
                        break
                    except Exception as e:
                        log(f"Failed to restore optimizer state into optimizer {i}: {e}")
                if not restored:
                    log(f"Checkpoint {load_model_path} contained optimizer state but it could not be restored into any optimizer")
        except Exception as e:
            log(f"Failed to load optimizer state from checkpoint {load_model_path}: {e}")

    if warmup_generations > 0 and init_from_gnn is not None:
        log(f"Starting behavior-cloning warmup: {warmup_generations} generations, {warmup_episodes} episodes/gen, lr={warmup_lr}")

        from gnn.model import evaluate_game

        def _make_head_optimizer(agent: PPOGNNPolicy, lr: float):
            params = [p for n, p in agent.named_parameters() if ("move_head" in n or "value_head" in n) and p.requires_grad]
            if not params:
                return None
            return torch.optim.Adam(params, lr=lr)

        for wg in range(warmup_generations):
            for i, a in enumerate(agents):
                if a is None:
                    continue
                # ensure GNN is frozen during warmup
                for p in a.gnn.parameters():
                    p.requires_grad = False
                opt = _make_head_optimizer(a, warmup_lr)
                log(f"Warmup: agent {i} opt={'yes' if opt is not None else 'no'}")
                if opt is None:
                    log(f"Warmup: agent {i} has no trainable head params; skipping warmup")
                    continue
                a.to(device)
                a.train()
                warmup_loss_sum = 0.0
                warmup_steps = 0
                for ep in range(warmup_episodes):
                    if ep % max(1, warmup_episodes // 4) == 0:
                        log(f"Warmup: agent {i} generation {wg+1}/{warmup_generations} episode {ep+1}/{warmup_episodes}")
                    # generate a random short self-play to harvest states
                    players: list[Player] = [RandomPlayer(0), RandomPlayer(1)]
                    game = Game(players)
                    randomize_start(game)
                    steps = 0
                    while game.winner is None and steps < 20:
                        player = game.players[game.current_player]
                        legal_moves = list(game.get_possible_moves(player.number))
                        if not legal_moves:
                            break
                        enc = encode_game_to_graph(game)
                        data = enc.data.to(device)
                        # Teacher: score each legal move by rolling it and calling evaluate_game
                        scores = []
                        for mv in legal_moves:
                            try:
                                game.do_move(player.number, mv)
                                try:
                                    s = float(evaluate_game(game))
                                except Exception:
                                    s = 0.0
                            except Exception:
                                s = 0.0
                            finally:
                                try:
                                    game.undo_move()
                                except Exception:
                                    pass
                            scores.append(s)
                        scores_t = torch.tensor(scores, dtype=torch.float32, device=device)
                        target = torch.softmax(scores_t / max(1e-6, warmup_temp), dim=0)

                        logits, _ = a.score_moves(data, legal_moves)
                        if logits.numel() == 0:
                            break
                        log_probs = F.log_softmax(logits, dim=-1)
                        loss = -(target * log_probs).sum()
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_([p for p in a.parameters() if p.requires_grad], 0.5)
                        opt.step()
                        # increment local step counter to avoid infinite loops
                        steps += 1
                        try:
                            lval = float(loss.detach().cpu().item())
                        except Exception:
                            lval = 0.0
                        warmup_loss_sum += lval
                        warmup_steps += 1
                        if warmup_steps % 10 == 0:
                            avg = warmup_loss_sum / max(1, warmup_steps)
                            log(f"Warmup: agent {i} gen {wg+1} progress: steps={warmup_steps}, avg_loss={avg:.4f}")

                avg_loss = warmup_loss_sum / max(1, warmup_steps)
                log(f"Warmup gen {wg+1}/{warmup_generations} finished for agent {i} — avg_loss={avg_loss:.4f} steps={warmup_steps}")
        # After warmup, re-create optimizers to reflect possibly changed params
        for idx, a in enumerate(agents):
            if a is None:
                optimizers[idx] = None
            else:
                optimizers[idx] = make_optimizer_for_model(a, head_lr if gnn_frozen else full_lr)

    scores = [0.0 for _ in range(population_size)]

    for gen in range(n_generations):
        # Unfreeze when requested
        if gnn_frozen and gen == freeze_gnn_generations:
            log(f"Unfreezing agents' GNNs at generation {gen}")
            for a in agents:
                if a is None:
                    continue
                for p in a.gnn.parameters():
                    p.requires_grad = True
            for idx, a in enumerate(agents):
                if a is None:
                    optimizers[idx] = None
                else:
                    optimizers[idx] = torch.optim.Adam(a.parameters(), lr=full_lr)
            gnn_frozen = False

        log(f"=== Generation {gen+1}/{n_generations} ===")
        losses: list[float] = []
        for ep in range(episodes_per_generation):
            idx_a, idx_b = random.sample(range(population_size), 2)
            loss_a, returns_a = run_episode_and_update(
                agents[idx_a], optimizers[idx_a], "A", gen, ep, device, log, max_episode_length
                , ppo_epochs=ppo_epochs
            )
            losses.append(loss_a)
            scores[idx_a] += float(returns_a[0].item()) if returns_a.shape[0] > 0 else 0.0
            loss_b, returns_b = run_episode_and_update(
                agents[idx_b], optimizers[idx_b], "B", gen, ep, device, log, max_episode_length
                , ppo_epochs=ppo_epochs
            )
            losses.append(loss_b)
            scores[idx_b] += float(returns_b[0].item()) if returns_b.shape[0] > 0 else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        log(f"[Loss] Average loss after generation {gen+1}: {avg_loss:.4f}")

        if (gen + 1) % eval_interval == 0:
            evaluate_agents(agents, device, eval_games, gen, log, max_episode_length)

        replace_weakest_and_mutate(
            agents,
            optimizers,
            scores,
            population_size,
            log,
            gen,
            mutation_noise=mutation_noise,
            keep_exact_prob=mutation_keep_exact_prob,
        )
        scores = [0.0 for _ in range(population_size)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population-based training for Rocks and Sticks")
    parser.add_argument("--population-size", type=int, default=5)
    parser.add_argument("--n-generations", type=int, default=5000)
    parser.add_argument("--episodes-per-generation", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--include-random", dest="include_random", action="store_true", help="Include one RandomPlayer in the population"
    )
    parser.set_defaults(include_random=False)
    parser.add_argument("--load-model", dest="load_model", type=str, default=None, help="Path to a saved model checkpoint")
    parser.add_argument(
        "--init-from-gnn",
        dest="init_from_gnn",
        type=str,
        default=None,
        help="Path to supervised GNNEval state dict to initialize policy GNN",
    )
    parser.add_argument(
        "--freeze-gnn-generations",
        dest="freeze_gnn_generations",
        type=int,
        default=0,
        help="Number of generations to freeze GNN (train heads only)",
    )
    parser.add_argument("--head-lr", dest="head_lr", type=float, default=1e-4, help="Learning rate while GNN frozen")
    parser.add_argument("--full-lr", dest="full_lr", type=float, default=1e-5, help="Learning rate after unfreeze")
    parser.add_argument(
        "--continue-training",
        dest="continue_training",
        action="store_true",
        help="Load a PPO checkpoint and continue training (ignore --init-from-gnn)",
    )
    parser.add_argument(
        "--max-episode-length",
        dest="max_episode_length",
        type=int,
        default=100,
        help="Maximum number of moves per self-play game; if exceeded the game ends as a draw",
    )
    parser.add_argument(
        "--ent-coef",
        dest="ent_coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for PPO (lower => less random exploration)",
    )
    parser.add_argument(
        "--mutation-noise",
        dest="mutation_noise",
        type=float,
        default=0.01,
        help="Stddev of gaussian noise applied during mutation",
    )
    parser.add_argument(
        "--mutation-keep-prob",
        dest="mutation_keep_prob",
        type=float,
        default=0.3,
        help="Probability of copying the best agent exactly when replacing the weakest",
    )
    parser.add_argument(
        "--ppo-epochs",
        dest="ppo_epochs",
        type=int,
        default=8,
        help="Number of PPO epochs (gradient passes) per update",
    )
    args = parser.parse_args()
    population_train(
        population_size=args.population_size,
        n_generations=args.n_generations,
        episodes_per_generation=args.episodes_per_generation,
        device=args.device,
        eval_interval=5,
        eval_games=50,
        include_random=args.include_random,
        load_model_path=args.load_model,
        init_from_gnn=args.init_from_gnn,
        freeze_gnn_generations=args.freeze_gnn_generations,
        head_lr=args.head_lr,
        full_lr=args.full_lr,
        continue_training=args.continue_training,
        max_episode_length=args.max_episode_length,
        ent_coef=args.ent_coef,
        mutation_noise=args.mutation_noise,
        mutation_keep_exact_prob=args.mutation_keep_prob,
        ppo_epochs=args.ppo_epochs,
    )
