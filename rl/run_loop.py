import argparse
import gc
import re
import shutil
import subprocess
import sys
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

    # copy matching keys
    new_sd = {}
    for k in model_sd.keys():
        if k in state:
            new_sd[k] = state[k]
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
    p.add_argument('--eval-games', type=int, default=10, help='Number of games to evaluate vs RandomPlayer after each iteration')
    p.add_argument('--eval-rollouts', type=int, default=100, help='MCTS rollouts to use during evaluation')
    p.add_argument('--no-augment', action='store_true', help='Disable symmetric augmentation during conversion')
    p.add_argument('--backend', type=str, default='python', choices=['python', 'cpp'], help='MCTS backend for self-play/eval')
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

        evaluator_path = (Path(args.out_dir) / f'gnn_eval_iter_{i-1}.pt') if i > 1 else (Path(args.out_dir) / 'gnn_eval_balanced.pt')
        model_path = str(evaluator_path) if evaluator_path.exists() else None
        play_self_play_games(
            num_games=args.games,
            mcts_rollouts=args.rollouts,
            mcts_time_limit=None,
            save_games_dir=str(saved_games_path),
            model_path=model_path,
            device=args.device,
            backend=args.backend,
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

            wins = 0
            for gidx in range(args.eval_games):
                if args.backend == 'cpp':
                    import mcts_ext

                    from players.game_total import GameTotal
                    from players.mcts_cpp import MCTSPlayerCPP

                    mcts = MCTSPlayerCPP(0, n_rollouts=args.eval_rollouts, seed=gidx)
                    rnd = RandomPlayer(1, seed=gidx + 1)
                    game = GameTotal(Game(players=[mcts, rnd]), mcts_ext.GameState())
                    randomize_start(game)
                else:
                    mcts = MCTSPlayer(0, use_gnn=True, n_rollouts=args.eval_rollouts, seed=gidx)
                    rnd = RandomPlayer(1, seed=gidx + 1)
                    game = Game(players=[mcts, rnd])
                    randomize_start(game)
                move_count = 0
                while game.winner is None:
                    p = game.players[game.current_player]
                    mv = p.get_move(game)
                    game.do_move(game.current_player, mv)
                    move_count += 1
                    if move_count > 100:
                        break
                if game.winner == 0:
                    wins += 1
            win_rate = wins / args.eval_games
            print(f"Evaluation vs RandomPlayer: {wins}/{args.eval_games} wins (win rate={win_rate:.2f})")

    print('All iterations completed')


if __name__ == '__main__':
    main()
