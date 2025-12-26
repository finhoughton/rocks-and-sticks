import argparse
import collections
import logging
import os
import platform
import random
import signal
import subprocess
import sys
import threading
import time
import tracemalloc
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import psutil
except ImportError:
    psutil = None
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from game import Game
from gnn.encode import SAMPLE_ENC, encode_game_to_graph
from gnn.game_generation import randomize_start
from gnn.model import GNNEval
from models import Move
from players import Player, RandomPlayer

logger = logging.getLogger("rl_population")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%H:%M:%S',
        handlers=[logging.FileHandler("population_train.log", mode="a"), logging.StreamHandler()]
    )
def log(msg: str) -> None:
    logger.info(msg)

class PPOPlayer(Player):
    def __init__(self, player_number: int, model: nn.Module, device: str = "cpu"):
        super().__init__(player_number)
        self.model = model
        self.device = device

    def get_move(self, game: Game) -> Move:
        legal_moves = list(game.get_possible_moves(self))
        enc = encode_game_to_graph(game)
        data = enc.data.to(self.device)
        action_idx, _, _ = select_action(self.model, data, legal_moves)
        return legal_moves[action_idx]

class PPOGNNPolicy(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        global_feat_dim: int,
        max_action_dim: int = 64,
        hidden: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__() # type: ignore
        self.gnn = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim, hidden=hidden, num_hidden_layers=num_hidden_layers, dropout=dropout)
        self.policy_head = nn.Linear(hidden + global_feat_dim, max_action_dim)
        self.value_head = nn.Linear(hidden + global_feat_dim, 1)
        self.max_action_dim = max_action_dim

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr, batch, g = data.x, data.edge_index, data.edge_attr, data.batch, data.global_feats
        h = self.gnn.convs[0](x, edge_index, edge_attr)
        for i in range(1, len(self.gnn.convs)):
            h = self.gnn.convs[i](h, edge_index, edge_attr)
        pooled = global_mean_pool(h, batch)
        features = torch.cat([pooled, g], dim=-1)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits.squeeze(0), value.squeeze(0)


def population_train(
    population_size: int = 10,
    n_generations: int = 10,
    episodes_per_generation: int = 100,
    device: str = "cpu",
    eval_interval: int = 5,  # Evaluate every 5 generations
    eval_games: int = 50    # Number of games per agent vs RandomPlayer
    , track_memory: bool = True,
    enable_malloc_profiler: bool = False,
    rss_threshold_mb: float | None = None,
):
    if track_memory:
        tracemalloc.start()

    def print_mem(tag: str) -> None:
        if not track_memory:
            return
        if psutil is not None:
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            log(f"[Memory] {tag}: {mem_mb:.2f} MB RSS")
        else:
            log(f"[Memory] {tag}: psutil not installed.")
    # Optional malloc/RSS sampler (samples native RSS over time)
    malloc_samples: List[Tuple[float, int]] = []
    malloc_profiler_running: bool = False
    malloc_profiler_thread: Optional[threading.Thread] = None

    def _malloc_profiler_loop(pid: int, interval: float = 0.5) -> None:
        nonlocal malloc_samples, malloc_profiler_running
        proc = psutil.Process(pid) if psutil is not None else None
        malloc_profiler_running = True
        try:
            while malloc_profiler_running:
                ts = time.time()
                rss = proc.memory_info().rss if proc is not None else 0
                malloc_samples.append((ts, int(rss)))
                time.sleep(interval)
        except Exception:
            pass

    def start_malloc_profiler(interval: float = 0.5) -> None:
        nonlocal malloc_profiler_thread
        if malloc_profiler_thread is not None:
            return
        pid = os.getpid()
        malloc_profiler_thread = threading.Thread(target=_malloc_profiler_loop, args=(pid, interval), daemon=True)
        malloc_profiler_thread.start()

    # Auto-start the malloc/RSS profiler if requested
    if enable_malloc_profiler:
        log("Starting malloc/RSS profiler")
        try:
            start_malloc_profiler()
        except Exception as e:
            log(f"Failed to start malloc profiler: {e}")

    # Signal handler to perform graceful cleanup on SIGTERM/SIGINT (cannot catch SIGKILL)
    def _signal_handler(signum: int, frame: Any) -> None:
        log(f"Received signal {signum}, performing cleanup before exit.")
        try:
            if enable_malloc_profiler:
                stop_and_write_malloc_profile()
        except Exception as e:
            log(f"Error while stopping malloc profiler in signal handler: {e}")
        try:
            if track_memory:
                curr, peak = tracemalloc.get_traced_memory()
                log(f"[tracemalloc] current={curr/1024/1024:.2f} MB peak={peak/1024/1024:.2f} MB")
        except Exception:
            pass
        try:
            import gc
            counts = collections.Counter(type(o).__name__ for o in gc.get_objects())
            top_types = counts.most_common(20)
            log("[GC] Top object types by count (signal handler):")
            for name, cnt in top_types:
                log(f"  {name}: {cnt}")
        except Exception:
            pass
        sys.exit(1)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Optional RSS watchdog: if RSS exceeds threshold, send SIGTERM to self
    rss_watchdog_thread: Optional[threading.Thread] = None
    rss_watchdog_running: bool = False

    def _rss_watchdog_loop(threshold_bytes: int, interval: float = 1.0) -> None:
        nonlocal rss_watchdog_running
        proc = psutil.Process(os.getpid()) if psutil is not None else None
        rss_watchdog_running = True
        try:
            while rss_watchdog_running:
                if proc is not None:
                    rss = proc.memory_info().rss
                    if rss > threshold_bytes:
                        log(f"RSS watchdog: RSS {rss/1024/1024:.2f} MB exceeded threshold {threshold_bytes/1024/1024:.2f} MB; sending SIGTERM")
                        os.kill(os.getpid(), signal.SIGTERM)
                        break
                time.sleep(interval)
        except Exception:
            pass

    def start_rss_watchdog(threshold_mb: float, interval: float = 1.0) -> None:
        nonlocal rss_watchdog_thread
        if rss_watchdog_thread is not None:
            return
        rss_watchdog_thread = threading.Thread(target=_rss_watchdog_loop, args=(int(threshold_mb * 1024 * 1024), interval), daemon=True)
        rss_watchdog_thread.start()

    # Auto-start RSS watchdog if threshold provided
    if rss_threshold_mb is not None:
        if psutil is None:
            log("RSS watchdog requested but psutil not installed; skipping watchdog.")
        else:
            try:
                start_rss_watchdog(rss_threshold_mb)
            except Exception as e:
                log(f"Failed to start RSS watchdog: {e}")

    def stop_and_write_malloc_profile(out_csv: str = "malloc_profile.csv") -> None:
        nonlocal malloc_profiler_running, malloc_profiler_thread, malloc_samples
        malloc_profiler_running = False
        if malloc_profiler_thread is not None:
            malloc_profiler_thread.join(timeout=1.0)
        try:
            with open(out_csv, "w") as f:
                f.write("timestamp,rss_bytes\n")
                for ts, rss in malloc_samples:
                    f.write(f"{ts},{rss}\n")
            log(f"Wrote malloc RSS samples to {out_csv}")
        except Exception as e:
            log(f"Failed to write malloc profile CSV: {e}")
        # On macOS, try to run vmmap for deeper native view
        if platform.system() == 'Darwin':
            try:
                vmmap_out = f"vmmap_{os.getpid()}.txt"
                subprocess.run(["vmmap", str(os.getpid())], stdout=open(vmmap_out, "w"), stderr=subprocess.DEVNULL)
                log(f"Wrote vmmap to {vmmap_out}")
            except Exception as e:
                log(f"vmmap failed: {e}")
    # Initialize population of agents
    agents: list[PPOGNNPolicy] = []
    optimizers: list[torch.optim.Optimizer] = []
    # Runtime check for SAMPLE_ENC
    if SAMPLE_ENC.data.x is None or SAMPLE_ENC.data.global_feats is None:
        raise ValueError("Node features or global features are missing in the encoded graph.")
    node_feat_dim = SAMPLE_ENC.data.x.size(1)
    global_feat_dim = SAMPLE_ENC.data.global_feats.size(1)
    for _ in range(population_size):
        model = PPOGNNPolicy(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim, max_action_dim=64)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        agents.append(model)
        optimizers.append(optimizer)

    scores = [0.0 for _ in range(population_size)]

    # Use top-level helpers instead of nested functions

    # main loop
    try:
        for gen in range(n_generations):
            log(f"=== Generation {gen+1}/{n_generations} ===")
            losses: list[float] = []
            for ep in range(episodes_per_generation):
                idx_a, idx_b = random.sample(range(population_size), 2)
                loss_a, returns_a = run_episode_and_update(agents[idx_a], optimizers[idx_a], "A", gen, ep, device, log, print_mem)
                losses.append(loss_a)
                scores[idx_a] += float(returns_a[0].item()) if returns_a.shape[0] > 0 else 0.0
                loss_b, returns_b = run_episode_and_update(agents[idx_b], optimizers[idx_b], "B", gen, ep, device, log, print_mem)
                losses.append(loss_b)
                scores[idx_b] += float(returns_b[0].item()) if returns_b.shape[0] > 0 else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            print_mem(f"After cleanup (gen {gen+1})")
            log(f"[Loss] Average loss after generation {gen+1}: {avg_loss:.4f}")
            if psutil is not None:
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
                log(f"[Memory] After generation {gen+1}: {mem_mb:.2f} MB RSS")
            else:
                log("[Memory] psutil not installed, can't report memory usage. Install with 'pip install psutil'.")
            if (gen + 1) % eval_interval == 0:
                evaluate_agents(agents, device, eval_games, gen, log, print_mem)
            replace_weakest_and_mutate(agents, optimizers, scores, population_size, log, gen)
            scores = [0.0 for _ in range(population_size)]
    except KeyboardInterrupt:
        # Dump memory usage and tracemalloc statistics for post-mortem before exiting
        log("KeyboardInterrupt received — dumping memory and tracer statistics before exit.")
        try:
            if psutil is not None:
                proc = psutil.Process(os.getpid())
                mem = proc.memory_info().rss / 1024 / 1024
                log(f"[Memory] RSS: {mem:.2f} MB")
        except Exception as e:
            log(f"Failed to read psutil memory info: {e}")
        try:
            curr, peak = tracemalloc.get_traced_memory()
            log(f"[tracemalloc] current={curr/1024/1024:.2f} MB peak={peak/1024/1024:.2f} MB")
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            log("[tracemalloc] Top 20 memory blocks:")
            for stat in top_stats[:20]:
                log(str(stat))
        except Exception as e:
            log(f"Failed to capture tracemalloc snapshot: {e}")
        try:
            import gc
            counts = collections.Counter(type(o).__name__ for o in gc.get_objects())
            top_types = counts.most_common(20)
            log("[GC] Top object types by count:")
            for name, cnt in top_types:
                log(f"  {name}: {cnt}")
        except Exception as e:
            log(f"Failed to gather GC stats: {e}")
        return
    finally:
        if enable_malloc_profiler:
            try:
                stop_and_write_malloc_profile()
            except Exception as e:
                log(f"Failed to stop/write malloc profiler: {e}")
        # stop RSS watchdog if running
        try:
            rss_watchdog_running = False
            if 'rss_watchdog_thread' in locals() and rss_watchdog_thread is not None:
                rss_watchdog_thread.join(timeout=1.0)
        except Exception:
            pass


def select_action(
    model: nn.Module,
    obs: Data,
    legal_moves: List[Any],
) -> Tuple[int, torch.Tensor, torch.Tensor]:

    logits, value = model(obs)
    mask = torch.zeros_like(logits)
    mask[:len(legal_moves)] = 1
    logits = logits.masked_fill(mask == 0, float('-inf'))
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action_idx = dist.sample()
    log_prob = dist.log_prob(action_idx) # type: ignore
    return int(action_idx.item()), log_prob, value # type: ignore

def run_episode(
    model: nn.Module,
    device: str = "cpu",
    gamma: float = 0.99,
    ) -> Tuple[List[Data], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:

    players: list[Player] = [PPOPlayer(0, model, device), PPOPlayer(1, model, device)]
    game = Game(players)
    randomize_start(game)
    obs_list: List[Data] = []
    action_list: List[int] = []
    logprob_list: List[torch.Tensor] = []
    value_list: List[torch.Tensor] = []
    reward_list: List[float] = []
    max_legal = 0
    done = False
    while not done:
        player = game.players[game.current_player]
        legal_moves = list(game.get_possible_moves(player))
        max_legal = max(max_legal, len(legal_moves))
        enc = encode_game_to_graph(game)
        data = enc.data.to(device)
        with torch.no_grad():
            action_idx, log_prob, value = select_action(model, data, legal_moves)
        action = legal_moves[action_idx]
        game.do_move(player, action)
        reward = 0.0
        if game.winner is not None:
            done = True
            reward = 1.0 if game.winner == player.number else -1.0
        obs_list.append(data)
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
    
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    values_tensor = torch.stack(value_list)
    logprobs_tensor = torch.stack(logprob_list)
    actions_tensor = torch.tensor(action_list, dtype=torch.long, device=device)
    return obs_list, actions_tensor, logprobs_tensor, values_tensor, returns_tensor, max_legal

def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_list: List[Data],
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    max_legal: int,
    clip_epsilon: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.5,
) -> Tuple[float, float, float, float]:

    # Compute advantage
    advantages = returns - values.detach()

    # Policy loss
    new_logprobs: list[Any] = []
    entropies: list[Any] = []
    for obs, action in zip(obs_list, actions):
        logits, _ = model(obs)
        logits = logits[:max_legal]
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_logprobs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

    ratio = (torch.stack(new_logprobs) - logprobs.detach()).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    # Value loss
    value_loss = F.mse_loss(values.squeeze(-1), returns)
    # Entropy bonus
    entropy = torch.stack(entropies).mean()
    # If model has ent_coef attribute, use it
    if hasattr(model, 'ent_coef'):
        ent_coef = getattr(model, 'ent_coef')
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    optimizer.zero_grad()
    loss.backward() # type: ignore
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()


def run_episode_and_update(
    agent: nn.Module,
    optimizer: torch.optim.Optimizer,
    label: str,
    gen: int,
    ep: int,
    device: str,
    log_fn: Callable[[str], None],
    print_mem_fn: Callable[[str], None],
    track_memory: bool = True,
) -> tuple[float, torch.Tensor]:

    obs_list, actions, logprobs, values, returns, max_legal = run_episode(agent, device=device)
    print_mem_fn(f"After run_episode {label} (gen {gen+1}, ep {ep+1})")
    log_fn(f"    obs_list len: {len(obs_list)}; actions shape: {getattr(actions, 'shape', None)}; values shape: {getattr(values, 'shape', None)}")
    if obs_list and hasattr(obs_list[0], 'x') and getattr(obs_list[0], 'x', None) is not None:
        shape = getattr(obs_list[0].x, 'shape', None)
        log_fn(f"    obs_list[0] x shape: {shape}")
    else:
        log_fn("    obs_list[0] x shape: None")
    if track_memory:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        log_fn(f"    [tracemalloc] Top 3 memory blocks after {label}:")
        for stat in top_stats[:3]:
            log_fn(str(stat).split("3.11/lib")[1] if "3.11/lib" in str(stat) else str(stat))
    actions = actions.detach().cpu()
    logprobs = logprobs.detach().cpu()
    values = values.detach().cpu()
    returns = returns.detach().cpu()
    loss, _, _, _ = ppo_update(agent, optimizer, obs_list, actions, logprobs, values, returns, max_legal)
    if track_memory:
        import gc
        gc.collect()
        log_fn(f"    [objgraph] Data objects: {sum(type(o).__name__ == 'Data' for o in gc.get_objects())}")
    return loss, returns

def evaluate_agents(agents: list[PPOGNNPolicy], device: str, eval_games: int, gen: int, log_fn: Callable[[str], None], print_mem_fn: Callable[[str], None], track_memory: bool = True) -> None:
    if track_memory:
        print_mem_fn(f"Before evaluation (gen {gen+1})")
    log_fn(f"[Eval] Evaluating all agents vs RandomPlayer after generation {gen+1}")
    for i, agent in enumerate(agents):
        wins = 0
        for _ in range(eval_games):
            players: list[Player] = [PPOPlayer(0, agent, device), RandomPlayer(1)]
            game = Game(players)
            randomize_start(game)
            while game.winner is None:
                p = game.players[game.current_player]
                move = p.get_move(game)
                game.do_move(p, move)
            if game.winner == 0:
                wins += 1
        win_rate = wins / eval_games
        if track_memory:
            print_mem_fn(f"After evaluation (gen {gen+1})")
        log_fn(f"  Agent {i} win rate vs RandomPlayer: {win_rate:.2f}")

def replace_weakest_and_mutate(agents: list[PPOGNNPolicy], optimizers: list[torch.optim.Optimizer], scores: list[float], population_size: int, log_fn: Callable[[str], None], gen: int) -> None:
    log_fn(f"Generation {gen+1} scores: {scores}")
    weakest = scores.index(min(scores))
    strongest = scores.index(max(scores))
    log_fn(f"Replacing agent {weakest} with mutated copy of agent {strongest}")
    agents[weakest].load_state_dict(agents[strongest].state_dict())
    for param in agents[weakest].parameters():
        noise = torch.normal(mean=0.0, std=0.02, size=param.size(), device=param.device)
        param.data.add_(noise)
    old_lr = optimizers[strongest].param_groups[0]['lr']
    new_lr = max(1e-6, min(1e-2, old_lr * (1 + random.uniform(-0.2, 0.2))))
    optimizers[weakest] = torch.optim.Adam(agents[weakest].parameters(), lr=new_lr)
    old_ent_coef = getattr(agents[strongest], 'ent_coef', 0.5)
    new_ent_coef = max(0.01, min(1.0, old_ent_coef * (1 + random.uniform(-0.2, 0.2))))
    setattr(agents[weakest], 'ent_coef', new_ent_coef)
    log_fn(f"Mutated agent {weakest}: lr={new_lr:.2e}, ent_coef={new_ent_coef:.3f}")
    best_idx = scores.index(max(scores))
    torch.save(agents[best_idx].state_dict(), "rl_population_best.pt")
    log_fn(f"Saved best population agent to rl_population_best.pt (generation {gen+1})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population-based training for Rocks and Sticks")
    parser.add_argument('--population-size', type=int, default=4)
    parser.add_argument('--n-generations', type=int, default=5000)
    parser.add_argument('--episodes-per-generation', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--track-memory', dest='track_memory', action='store_true', help='Enable memory tracking')
    group.add_argument('--no-track-memory', dest='track_memory', action='store_false', help='Disable memory tracking')
    parser.set_defaults(track_memory=True)
    parser.add_argument('--malloc-profiler', dest='malloc_profiler', action='store_true', help='Enable malloc/RSS profiler (writes malloc_profile.csv and vmmap if available)')
    parser.set_defaults(malloc_profiler=False)
    parser.add_argument('--rss-threshold', dest='rss_threshold', type=float, default=None, help='RSS threshold in MB; if exceeded the trainer will send SIGTERM to itself to allow cleanup')
    args = parser.parse_args()
    population_train(
        population_size=args.population_size,
        n_generations=args.n_generations,
        episodes_per_generation=args.episodes_per_generation,
        device=args.device,
        track_memory=args.track_memory,
        enable_malloc_profiler=args.malloc_profiler,
        rss_threshold_mb=args.rss_threshold,
    )
