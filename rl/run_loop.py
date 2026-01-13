import argparse
import gc
import json
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from game import Game
from gnn.encode import SAMPLE_ENC
from gnn.game_generation import randomize_start
from gnn.model import load_model
from players.base import RandomPlayer
from players.mcts import MCTSPlayer
from rl.convert import convert_games_to_dataset
from rl.play_games import play_self_play_games


def export_gnn_eval_from_policy(policy_ckpt: Path, out_path: Path):
    """Load policy+value checkpoint and copy conv/norm weights into GNNEval then save."""
    from gnn.encode import SAMPLE_ENC
    from gnn.model import GNNEval

    device = torch.device('cpu')
    ck = torch.load(policy_ckpt, map_location=device)
    state = ck if isinstance(ck, dict) and 'state_dict' not in ck else ck.get('state_dict', ck)

    # infer feature dims from sample encoding
    node_feat_dim = SAMPLE_ENC.data.x.shape[1] # type: ignore
    global_feat_dim = SAMPLE_ENC.data.global_feats.shape[1]
    # instantiate GNNEval with inferred dims
    model = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim)
    model_sd = model.state_dict()

    # Copy matching trunk keys (convs/norms/etc).
    new_sd: dict[str, torch.Tensor] = {}
    for k in model_sd.keys():
        if k in state:
            new_sd[k] = state[k]

    # If this checkpoint came from PolicyValueNet, map its value head into GNNEval.head.
    # PolicyValueNet has: value_mlp = [Linear, ReLU, Linear]
    # GNNEval has: head = [Linear, ReLU, Dropout, Linear]
    if isinstance(state, dict) and any(str(k).startswith('value_mlp.') for k in state.keys()):
        # First linear
        if 'value_mlp.0.weight' in state and 'head.0.weight' in model_sd:
            new_sd['head.0.weight'] = state['value_mlp.0.weight']
        if 'value_mlp.0.bias' in state and 'head.0.bias' in model_sd:
            new_sd['head.0.bias'] = state['value_mlp.0.bias']
        # Final linear
        if 'value_mlp.2.weight' in state and 'head.3.weight' in model_sd:
            new_sd['head.3.weight'] = state['value_mlp.2.weight']
        if 'value_mlp.2.bias' in state and 'head.3.bias' in model_sd:
            new_sd['head.3.bias'] = state['value_mlp.2.bias']

    model_sd.update(new_sd)
    model.load_state_dict(model_sd)
    torch.save(model.state_dict(), out_path)
    print(f"Saved GNNEval checkpoint to {out_path}")


def _cleanup_old_datasets(data_dir: Path, *, keep_last: int, current_iter: int) -> None:
    keep_last = max(0, int(keep_last))
    cutoff = current_iter - keep_last
    if cutoff <= 0:
        return

    for k in range(1, cutoff + 1):
        pt_path = data_dir / f"alpha_dataset_iter_{k}.pt"
        shards_dir = data_dir / f"alpha_dataset_iter_{k}.pt.shards"
        try:
            if shards_dir.exists() and shards_dir.is_dir():
                shutil.rmtree(shards_dir)
        except Exception as e:
            print(f"Warning: failed to delete {shards_dir}: {e}")
        try:
            if pt_path.exists() and pt_path.is_file():
                pt_path.unlink()
        except Exception as e:
            print(f"Warning: failed to delete {pt_path}: {e}")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _evaluate_vs_random(
    *,
    backend: str,
    device: str,
    eval_games: int,
    eval_rollouts: int,
    eval_max_moves: int,
    eval_seed: int,
    iteration: int,
    model_path: str,
    cpp_verbose: bool,
) -> dict:
    wins = 0
    losses = 0
    draws = 0
    max_moves_reached = 0
    move_counts: list[int] = []

    t0 = time.time()
    for gidx in range(int(eval_games)):
        print(f"Eval game {gidx + 1}/{eval_games}... vs RandomPlayer")
        random.seed(int(eval_seed) + int(gidx))

        agent_player = 0 if (gidx % 2 == 0) else 1
        random_player = 1 - agent_player

        if backend == 'cpp':
            import players_ext

            from players.game_total import GameTotal
            from players.mcts_cpp import MCTSPlayerCPP

            mcts = MCTSPlayerCPP(agent_player, n_rollouts=eval_rollouts, seed=gidx, verbose=bool(cpp_verbose))
            rnd = RandomPlayer(random_player, seed=gidx + 1)
            players = [mcts, rnd] if agent_player == 0 else [rnd, mcts]
            game = GameTotal(Game(players=players), players_ext.GameState())
            randomize_start(game)
        else:
            mcts = MCTSPlayer(agent_player, use_gnn=True, n_rollouts=eval_rollouts, seed=gidx)
            rnd = RandomPlayer(random_player, seed=gidx + 1)
            players = [mcts, rnd] if agent_player == 0 else [rnd, mcts]
            game = Game(players=players)
            randomize_start(game)

        move_count = 0
        while game.winner is None and move_count < int(eval_max_moves):
            p = game.players[game.current_player]
            mv = p.get_move(game)
            try:
                game.do_move(game.current_player, mv)
            except Exception:
                print(f"Error during eval game {gidx} move {move_count} by player {game.current_player}: {p}, {p.__class__.__name__}")
                raise
            move_count += 1

        move_counts.append(move_count)
        if game.winner == agent_player:
            wins += 1
        elif game.winner == random_player:
            losses += 1
        else:
            draws += 1
            if move_count >= int(eval_max_moves):
                max_moves_reached += 1

    dt = time.time() - t0
    games = max(1, int(eval_games))
    avg_moves = float(sum(move_counts)) / float(max(1, len(move_counts)))
    win_rate = float(wins) / float(games)

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "iter": int(iteration),
        "backend": str(backend),
        "device": str(device),
        "model_path": str(model_path),
        "eval_games": int(eval_games),
        "eval_rollouts": int(eval_rollouts),
        "eval_max_moves": int(eval_max_moves),
        "eval_seed": int(eval_seed),
        "eval_alternate_roles": True,
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "win_rate": win_rate,
        "avg_moves": avg_moves,
        "max_moves_reached": int(max_moves_reached),
        "seconds": float(dt),
    }


def _evaluate_cpp_model_vs_model(
    *,
    device: str,
    eval_games: int,
    eval_rollouts: int,
    eval_max_moves: int,
    eval_seed: int,
    iteration: int,
    model_a_path: str,
    model_b_path: str,
    model_a_label: str,
    model_b_label: str,
    cpp_verbose: bool,
) -> dict:

    import players_ext

    from players.game_total import GameTotal
    from players.mcts_cpp import MCTSPlayerCPP

    wins = 0
    losses = 0
    draws = 0
    max_moves_reached = 0
    move_counts: list[int] = []

    t0 = time.time()
    for gidx in range(int(eval_games)):
        print(f"Eval game {gidx + 1}/{eval_games}... vs {model_a_label} and {model_b_label}")
        random.seed(int(eval_seed) + int(gidx))

        a_as_p0 = (gidx % 2 == 0)

        p0 = MCTSPlayerCPP(0, n_rollouts=eval_rollouts, seed=gidx, verbose=bool(cpp_verbose))
        p1 = MCTSPlayerCPP(1, n_rollouts=eval_rollouts, seed=gidx + 1, verbose=bool(cpp_verbose))

        # disable exploration
        p0.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0, temperature=0.0, temperature_moves=0)
        p1.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0, temperature=0.0, temperature_moves=0)

        if a_as_p0:
            p0.set_model_checkpoint(model_a_path, device=device)
            p1.set_model_checkpoint(model_b_path, device=device)
        else:
            p0.set_model_checkpoint(model_b_path, device=device)
            p1.set_model_checkpoint(model_a_path, device=device)

        game = GameTotal(Game(players=[p0, p1]), players_ext.GameState())
        randomize_start(game)

        move_count = 0
        while game.winner is None and move_count < int(eval_max_moves):
            pl = game.players[game.current_player]
            mv = pl.get_move(game)
            game.do_move(game.current_player, mv)
            move_count += 1

        move_counts.append(move_count)

        if game.winner is None:
            draws += 1
            if move_count >= int(eval_max_moves):
                max_moves_reached += 1
        else:
            a_player_num = 0 if a_as_p0 else 1
            if int(game.winner) == a_player_num:
                wins += 1
            else:
                losses += 1

    dt = time.time() - t0
    games = max(1, int(eval_games))
    avg_moves = float(sum(move_counts)) / float(max(1, len(move_counts)))
    win_rate = float(wins) / float(games)

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "iter": int(iteration),
        "backend": "cpp",
        "device": str(device),
        "match": "checkpoint_vs_checkpoint",
        "model_a": str(model_a_path),
        "model_b": str(model_b_path),
        "model_a_label": str(model_a_label),
        "model_b_label": str(model_b_label),
        "eval_games": int(eval_games),
        "eval_rollouts": int(eval_rollouts),
        "eval_max_moves": int(eval_max_moves),
        "eval_seed": int(eval_seed),
        "eval_alternate_roles": True,
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "win_rate": win_rate,
        "avg_moves": avg_moves,
        "max_moves_reached": int(max_moves_reached),
        "seconds": float(dt),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=3)
    p.add_argument('--games', type=int, default=50)
    p.add_argument('--rollouts', type=int, default=100)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--train-num-workers', type=int, default=0, help='DataLoader workers for training subprocess')
    p.add_argument('--train-prefetch-factor', type=int, default=2, help='Prefetch factor for training subprocess (requires train-num-workers>0)')
    p.add_argument('--train-persistent-workers', action='store_true', help='Enable persistent DataLoader workers in training subprocess (requires train-num-workers>0)')
    p.add_argument('--train-steps-per-epoch', type=int, default=0, help='If >0, cap training batches per epoch in subprocess (stability on MPS)')
    p.add_argument('--out-dir', default='checkpoints')
    p.add_argument('--saved-games-dir', default='rl_self_play/iter_{iter}')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--dataset-shard-size', type=int, default=0, help='If >0, write training dataset in shards of this many samples (reduces peak memory). Suggested: 20000-50000 for ~300k+ samples.')
    p.add_argument('--keep-last-datasets', type=int, default=1, help='How many most-recent iteration datasets to keep on disk (pt + .shards). Older iterations are deleted after a successful iteration.')
    p.add_argument('--eval-games', type=int, default=100, help='Number of games to evaluate vs RandomPlayer after each iteration')
    p.add_argument('--eval-rollouts', type=int, default=500, help='MCTS rollouts to use during evaluation')
    p.add_argument('--eval-max-moves', type=int, default=256, help='Max moves per evaluation game before counting as draw')
    p.add_argument('--eval-seed', type=int, default=12345, help='Base RNG seed for deterministic evaluation starting positions')
    p.add_argument('--strength-log', default='logs/strength_curve.jsonl', help='Append JSONL evaluation records here each iteration')
    p.add_argument('--eval-vs-prev', type=int, default=0, help='If >0 and --backend=cpp: also evaluate vs the last N previous evaluator checkpoints')
    p.add_argument('--eval-prev-games', type=int, default=40, help='Games per previous-checkpoint opponent')
    p.add_argument('--eval-prev-rollouts', type=int, default=200, help='MCTS rollouts per move for previous-checkpoint evaluation')
    p.add_argument('--no-augment', action='store_true', help='Disable symmetric augmentation during conversion')
    p.add_argument('--backend', type=str, default='python', choices=['python', 'cpp'], help='MCTS backend for self-play/eval')
    p.add_argument(
        '--cpp-verbose',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If --backend=cpp: enable verbose C++ MCTS logging.',
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # detect existing checkpoints and choose starting iteration so numbering continues
    az_ckpts = list(out_dir.glob('gnn_az_iter_*.pt'))
    max_existing = 0
    for p in az_ckpts:
        m = re.search(r'gnn_az_iter_(\d+)\.pt$', p.name)
        if m:
            try:
                n = int(m.group(1))
            except ValueError:
                continue
            if n > max_existing:
                max_existing = n

    start_iter = max_existing + 1
    end_iter = start_iter + args.iters - 1
    if max_existing > 0:
        print(f"Resuming training: existing checkpoints up to iteration {max_existing}.")
    else:
        print("Starting training from iteration 1.")

    for i in range(start_iter, end_iter + 1):
        print(f"=== Iteration {i}/{end_iter} ===")

        saved_games_dir = args.saved_games_dir.format(iter=i)
        saved_games_path = Path(saved_games_dir)
        saved_games_path.mkdir(parents=True, exist_ok=True)

        # Resolve the evaluator checkpoint used for self-play.
        # When --out-dir is not the default 'checkpoints', we still want to be able to
        # fall back to an existing balanced evaluator shipped in ./checkpoints.
        candidates: list[Path] = []
        out_dir_path = Path(args.out_dir)
        default_ckpt_dir = Path('checkpoints')
        if i > 1:
            candidates.append(out_dir_path / f'gnn_eval_iter_{i-1}.pt')
            candidates.append(default_ckpt_dir / f'gnn_eval_iter_{i-1}.pt')
        candidates.append(out_dir_path / 'gnn_eval_balanced.pt')
        candidates.append(default_ckpt_dir / 'gnn_eval_balanced.pt')

        evaluator_path = next((p for p in candidates if p.exists()), None)
        model_path = str(evaluator_path) if evaluator_path is not None else None
        play_self_play_games(
            num_games=args.games,
            mcts_rollouts=args.rollouts,
            mcts_time_limit=None,
            save_games_dir=str(saved_games_path),
            model_path=model_path,
            device=args.device,
            backend=args.backend,
            cpp_verbose=bool(args.cpp_verbose),
            temp=1.1
        )

        dataset_path = data_dir / f'alpha_dataset_iter_{i}.pt'
        shard_size = int(args.dataset_shard_size) if int(args.dataset_shard_size) > 0 else None
        convert_games_to_dataset(str(saved_games_path), str(dataset_path), augment=not args.no_augment, shard_size=shard_size)

        policy_ckpt = out_dir / f'gnn_az_iter_{i}.pt'
        # Run training in a fresh process so macOS/MPS cached memory is released when training finishes.
        gc.collect()
        train_cmd = [
            sys.executable,
            '-m',
            'rl.train',
            '--dataset',
            str(dataset_path),
            '--epochs',
            str(args.epochs),
            '--batch-size',
            str(args.batch_size),
            '--lr',
            str(args.lr),
            '--device',
            str(args.device),
            '--out',
            str(policy_ckpt),
        ]
        # Speed: enable AMP automatically on mps/cuda.
        dev = str(args.device)
        if dev == 'mps' or dev.startswith('cuda'):
            train_cmd.append('--amp')
        # Speed: allow DataLoader workers for faster collation.
        if int(args.train_num_workers) > 0:
            train_cmd += ['--num-workers', str(int(args.train_num_workers))]
            train_cmd += ['--prefetch-factor', str(int(args.train_prefetch_factor))]
            if args.train_persistent_workers:
                train_cmd += ['--persistent-workers']
        if int(args.train_steps_per_epoch) > 0:
            train_cmd += ['--steps-per-epoch', str(int(args.train_steps_per_epoch))]
        print('Running training subprocess:', ' '.join(train_cmd))
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            # On macOS, DataLoader multiprocessing can be fragile and can get SIGKILLed under memory pressure,
            # leaving leaked semaphores. Fall back to single-process loading to keep the loop going.
            if e.returncode in (-9, 137):
                print('Training subprocess was SIGKILLed; retrying with --num-workers 0...')
                cleaned: list[str] = []
                skip_next = False
                for idx, c in enumerate(train_cmd):
                    if skip_next:
                        skip_next = False
                        continue
                    if c in ('--num-workers', '--prefetch-factor'):
                        skip_next = True
                        continue
                    if c == '--persistent-workers':
                        continue
                    cleaned.append(c)
                print('Retry command:', ' '.join(cleaned))
                try:
                    subprocess.run(cleaned, check=True)
                except subprocess.CalledProcessError as e2:
                    if e2.returncode in (-9, 137):
                        # Last-resort: disable AMP and cap steps-per-epoch + batch size.
                        print('Retry was SIGKILLed; retrying with smaller batch and capped steps (no AMP)...')
                        safe_bs = max(1, int(args.batch_size) // 2)
                        safe_steps = int(args.train_steps_per_epoch) if int(args.train_steps_per_epoch) > 0 else 2000
                        fallback_cmd = [
                            sys.executable,
                            '-m',
                            'rl.train',
                            '--dataset',
                            str(dataset_path),
                            '--epochs',
                            str(args.epochs),
                            '--batch-size',
                            str(safe_bs),
                            '--lr',
                            str(args.lr),
                            '--device',
                            str(args.device),
                            '--out',
                            str(policy_ckpt),
                            '--steps-per-epoch',
                            str(safe_steps),
                            '--clear-cache-interval',
                            '50',
                        ]
                        print('Fallback command:', ' '.join(fallback_cmd))
                        subprocess.run(fallback_cmd, check=True)
                    else:
                        raise
            else:
                raise

        print("Training completed for iteration", i, "cleaning up old datasets...")

        _cleanup_old_datasets(data_dir, keep_last=int(args.keep_last_datasets), current_iter=i)

        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

        gnn_eval_ckpt = out_dir / f'gnn_eval_iter_{i}.pt'
        export_gnn_eval_from_policy(policy_ckpt, gnn_eval_ckpt)

        if args.eval_games and args.eval_games > 0:
            try:
                node_dim = SAMPLE_ENC.data.x.size(1) # type: ignore
                global_dim = SAMPLE_ENC.data.global_feats.size(1)
                load_model(str(gnn_eval_ckpt), node_dim, global_dim, device=args.device)
                print(f"Loaded evaluator {gnn_eval_ckpt} for evaluation")
            except Exception as e:
                print(f"Failed to load evaluator for evaluation: {e}")
                continue

            record = _evaluate_vs_random(
                backend=str(args.backend),
                device=str(args.device),
                eval_games=int(args.eval_games),
                eval_rollouts=int(args.eval_rollouts),
                eval_max_moves=int(args.eval_max_moves),
                eval_seed=int(args.eval_seed),
                iteration=int(i),
                model_path=str(gnn_eval_ckpt),
                cpp_verbose=bool(args.cpp_verbose),
            )

            # Optional: ladder eval vs prior checkpoints (C++ backend only).
            if int(args.eval_vs_prev) > 0:
                if str(args.backend) != 'cpp':
                    print('Note: --eval-vs-prev requires --backend=cpp (needs per-engine checkpoint loading).')
                elif i <= 1:
                    pass
                else:
                    prev_n = int(args.eval_vs_prev)
                    prev_games = int(args.eval_prev_games)
                    prev_rollouts = int(args.eval_prev_rollouts)
                    start_opp = max(0, int(i) - prev_n)
                    vs_prev: dict[str, dict] = {}
                    for opp_iter in range(start_opp, int(i)):
                        opp_ckpt = out_dir / f'gnn_eval_iter_{opp_iter}.pt' if opp_iter > 0 else out_dir / 'gnn_eval_balanced.pt'
                        if not opp_ckpt.exists():
                            continue
                        print(f"Evaluating vs checkpoint iter {opp_iter}...")
                        res = _evaluate_cpp_model_vs_model(
                            device=str(args.device),
                            eval_games=prev_games,
                            eval_rollouts=prev_rollouts,
                            eval_max_moves=int(args.eval_max_moves),
                            eval_seed=int(args.eval_seed) + 100000 * int(opp_iter),
                            iteration=int(i),
                            model_a_path=str(gnn_eval_ckpt),
                            model_b_path=str(opp_ckpt),
                            model_a_label=f"iter_{i}",
                            model_b_label=f"iter_{opp_iter}",
                            cpp_verbose=bool(args.cpp_verbose),
                        )
                        vs_prev[str(opp_iter)] = {
                            "wins": int(res["wins"]),
                            "losses": int(res["losses"]),
                            "draws": int(res["draws"]),
                            "win_rate": float(res["win_rate"]),
                            "eval_games": int(res["eval_games"]),
                            "eval_rollouts": int(res["eval_rollouts"]),
                            "seconds": float(res["seconds"]),
                        }
                    record["vs_prev_checkpoints"] = vs_prev

            print(
                "Evaluation vs RandomPlayer: "
                f"{record['wins']}/{record['eval_games']} wins, "
                f"{record['losses']} losses, {record['draws']} draws "
                f"(win_rate={record['win_rate']:.2f}, avg_moves={record['avg_moves']:.1f})"
            )
            try:
                _append_jsonl(Path(str(args.strength_log)), record)
                print(f"Appended strength record to {args.strength_log}")
            except Exception as e:
                print(f"Warning: failed to write strength log {args.strength_log}: {e}")

    print('All iterations completed')


if __name__ == '__main__':
    main()
