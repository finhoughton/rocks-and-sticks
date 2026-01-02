#!/usr/bin/env python3
"""Orchestrate self-play -> convert -> train loop.

Usage: python -m rl.run_loop --iters 3 --games 50 --rollouts 100 --epochs 3
"""
import argparse
from pathlib import Path

import torch

from rl.convert import convert_games_to_dataset
from rl.play_games import play_self_play_games
from rl.train import train

# calling subprocesses replaced by direct function calls below


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=3)
    p.add_argument('--games', type=int, default=50)
    p.add_argument('--rollouts', type=int, default=100)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out-dir', default='checkpoints')
    p.add_argument('--saved-games-dir', default='rl_self_play/iter_{iter}')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--no-augment', action='store_true', help='Disable symmetric augmentation during conversion')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, args.iters + 1):
        print(f"=== Iteration {i}/{args.iters} ===")

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
        )

        dataset_path = data_dir / f'alpha_dataset_iter_{i}.pt'
        convert_games_to_dataset(str(saved_games_path), str(dataset_path), augment=not args.no_augment)

        policy_ckpt = out_dir / f'gnn_az_iter_{i}.pt'
        train(str(dataset_path), args.epochs, args.batch_size, args.lr, args.device, str(policy_ckpt))

        gnn_eval_ckpt = out_dir / f'gnn_eval_iter_{i}.pt'
        export_gnn_eval_from_policy(policy_ckpt, gnn_eval_ckpt)

    print('All iterations completed')


if __name__ == '__main__':
    main()
