from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Return an ``ArgumentParser`` with all run-loop flags."""
    p = argparse.ArgumentParser()

    p.add_argument('--seed', type=int, default=0,
                   help='Base RNG seed for self-play/training determinism (0 is allowed).')
    p.add_argument('--iters', type=int, default=3)
    p.add_argument('--games', type=int, default=50)
    p.add_argument('--rollouts', type=int, default=100) # mcts rollouts for self-play and evaluation
    p.add_argument('--loop-mode', type=str, default='az',
                   choices=['az', 'ab-supervised'],
                   help='Loop style: az = MCTS self-play + PolicyValue training; '
                        'ab-supervised = C++ AlphaBeta game generation + supervised GNNEval retraining.')


    p.add_argument('--selfplay-temp', type=float, default=1.1,
                   help='Self-play temperature for sampling from visit counts (C++ backend only).')
    p.add_argument('--selfplay-temp-moves', type=int, default=20,
                   help='Number of opening moves to apply temperature sampling (C++ backend only).')
    p.add_argument('--selfplay-dirichlet-alpha', type=float, default=0.3,
                   help='Dirichlet alpha for root noise (C++ backend only).')
    p.add_argument('--selfplay-dirichlet-epsilon', type=float, default=0.25,
                   help='Dirichlet epsilon for root noise (C++ backend only).')
    p.add_argument('--selfplay-temp-decay', type=float, default=1.0,
                   help='Multiply self-play temperature by this each iteration after --selfplay-decay-start.')
    p.add_argument('--selfplay-epsilon-decay', type=float, default=1.0,
                   help='Multiply self-play Dirichlet epsilon by this each iteration after --selfplay-decay-start.')
    p.add_argument('--selfplay-decay-start', type=int, default=0,
                   help='Iteration index (1-based) after which to start decaying exploration. 0 = immediately.')
    p.add_argument('--selfplay-min-temp', type=float, default=0.0,
                   help='Floor for decayed self-play temperature.')
    p.add_argument('--selfplay-min-epsilon', type=float, default=0.0,
                   help='Floor for decayed self-play Dirichlet epsilon.')
    p.add_argument('--selfplay-vs-random-prob', type=float, default=0.0,
                   help='Probability a self-play game uses a random opponent.')

    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--value-weight', type=float, default=1.0,
                   help='Weight applied to value loss during training')
    p.add_argument('--value-lr-mult', type=float, default=1.0,
                   help='LR multiplier for the value head during training')
    p.add_argument('--device', default='cpu')
    p.add_argument('--train-num-workers', type=int, default=0,
                   help='DataLoader workers for training subprocess')
    p.add_argument('--train-prefetch-factor', type=int, default=2,
                   help='Prefetch factor for training subprocess (requires train-num-workers>0)')
    p.add_argument('--train-persistent-workers', action='store_true',
                   help='Enable persistent DataLoader workers (requires train-num-workers>0)')
    p.add_argument('--train-steps-per-epoch', type=int, default=0,
                   help='If >0, cap training batches per epoch in subprocess (stability on MPS)')

    p.add_argument('--out-dir', default='checkpoints')
    p.add_argument('--init-eval-model', type=str, default=None,
                   help='Evaluator checkpoint for iteration 1 self-play. '
                        'If not provided, a random-initialized GNNEval is created in --out-dir.')
    p.add_argument('--init-policy-from', type=str, default=None,
                   help='Checkpoint to warm-start PolicyValueNet from at iteration 1.')
    p.add_argument('--auto-init-policy', action='store_true', default=False,
                   help='Auto-create a random PolicyValueNet checkpoint for priors (C++ backend).')
    p.add_argument('--saved-games-dir', default='rl_self_play/iter_{iter}')
    p.add_argument('--data-dir', default='data')

    p.add_argument('--dataset-shard-size', type=int, default=0,
                   help='If >0, write training dataset in shards of this many samples.')
    p.add_argument('--dataset-grouped-policy',
                   action=argparse.BooleanOptionalAction, default=True,
                   help='Store one graph per decision with K moves (faster training).')
    p.add_argument('--dataset-policy-topk', type=int, default=0,
                   help='If >0, keep only top-K policy targets per decision.')
    p.add_argument('--dataset-policy-min-prob', type=float, default=0.0,
                   help='If >0, drop policy targets below this probability.')
    p.add_argument('--dataset-policy-topk-start', type=int, default=0,
                   help='Schedule top-K starting from this value.')
    p.add_argument('--dataset-policy-topk-end', type=int, default=0,
                   help='Scheduled top-K value at --dataset-policy-topk-end-iter.')
    p.add_argument('--dataset-policy-topk-start-iter', type=int, default=0,
                   help='Iteration (1-based) to start scheduling dataset-policy-topk.')
    p.add_argument('--dataset-policy-topk-end-iter', type=int, default=0,
                   help='Iteration (1-based) to finish scheduling dataset-policy-topk.')
    p.add_argument('--dataset-policy-topk-curve', type=str, default='cosine',
                   choices=['linear', 'cosine'],
                   help='Curve for scheduled dataset-policy-topk.')

    p.add_argument('--replay-window', type=int, default=1,
                   help='Recent iteration datasets to train on (1=current only, 0=ALL).')
    p.add_argument('--keep-last-datasets', type=int, default=1,
                   help='Recent iteration datasets to keep on disk.')

    p.add_argument('--eval-games', type=int, default=100,
                   help='Games to evaluate vs RandomPlayer after each iteration')
    p.add_argument('--eval-rollouts', type=int, default=500,
                   help='MCTS rollouts to use during evaluation')
    p.add_argument('--eval-jobs', type=int, default=1,
                   help='Parallel worker processes for evaluation. 1 disables parallelism.')
    p.add_argument('--eval-heavy-every', type=int, default=0,
                   help='If >0, run a heavier vs-Random eval every N iterations.')
    p.add_argument('--eval-heavy-games', type=int, default=0,
                   help='Games for the heavy vs-Random eval.')
    p.add_argument('--eval-heavy-rollouts', type=int, default=0,
                   help='Rollouts for the heavy vs-Random eval.')
    p.add_argument('--eval-max-moves', type=int, default=256,
                   help='Max moves per evaluation game before counting as draw')
    p.add_argument('--eval-seed', type=int, default=12345,
                   help='Base RNG seed for deterministic evaluation starting positions')
    p.add_argument('--eval-randomize-start',
                   action=argparse.BooleanOptionalAction, default=True,
                   help='Call randomize_start() before each eval game.')
    p.add_argument('--eval-games-low', type=int, default=0,
                   help='If >0, run an additional vs-Random eval at --eval-rollouts-low')
    p.add_argument('--eval-rollouts-low', type=int, default=0,
                   help='Rollouts for the additional low-rollouts eval.')
    p.add_argument('--strength-log', default='logs/strength_curve.jsonl',
                   help='Append JSONL evaluation records here each iteration')
    p.add_argument('--eval-vs-prev', type=int, default=0,
                   help='If >0 and --backend=cpp: evaluate vs the last N previous checkpoints')
    p.add_argument('--eval-vs-prev-every', type=int, default=1,
                   help='Run --eval-vs-prev only every N iterations.')
    p.add_argument('--eval-prev-games', type=int, default=40,
                   help='Games per previous-checkpoint opponent')
    p.add_argument('--eval-prev-rollouts', type=int, default=200,
                   help='MCTS rollouts per move for previous-checkpoint evaluation')

    p.add_argument('--no-augment', action='store_true',
                   help='Disable symmetric augmentation during conversion')
    p.add_argument('--backend', type=str, default='cpp',
                   choices=['python', 'cpp'], help='MCTS backend for self-play/eval')
    p.add_argument('--cpp-verbose', type=int, default=1,
                   help='C++ MCTS verbosity (0=silent, 1=summaries, 2=debug).')
    p.add_argument('--cpp-use-nn-value',
                   action=argparse.BooleanOptionalAction, default=True,
                   help='Use the neural net for leaf value evaluation.')
    p.add_argument('--prior-scale', type=float, default=1.0,
                   help='Scale applied to policy priors before mixing (C++ backend only).')
    p.add_argument('--prior-mix-uniform', type=float, default=0.04,
                   help='Weight of uniform mix applied to policy priors (C++ backend only).')
    p.add_argument('--filter-near-area',
                   action=argparse.BooleanOptionalAction, default=False,
                   help='Reject randomized starting positions where a short MCTS probe finds a quick win.')
    p.add_argument('--filter-probe-rollouts', type=int, default=200,
                   help='MCTS rollouts per move for the quick-win probe.')
    p.add_argument('--filter-probe-moves', type=int, default=6,
                   help='Max half-moves for the quick-win probe.')
    p.add_argument('--randomize-max-sticks', type=int, default=5,
                   help='Maximum sticks placed during randomize_start (default 5).')

    p.add_argument('--ab-time-limit', type=int, default=3000,
                   help='Per-move time limit in ms for ab-supervised generation. 0 = fixed depth.')
    p.add_argument('--ab-depth', type=int, default=4,
                   help='Fixed depth for ab-supervised when --ab-time-limit=0.')
    p.add_argument('--ab-max-depth', type=int, default=20,
                   help='Iterative deepening max depth cap for ab-supervised generation.')
    p.add_argument('--ab-move-cap', type=int, default=48,
                   help='Move cap passed to C++ AlphaBeta for ab-supervised generation.')
    p.add_argument('--ab-max-moves', type=int, default=200,
                   help='Max moves per generated AB game before draw.')
    p.add_argument('--ab-use-heuristic',
                   action=argparse.BooleanOptionalAction, default=True,
                   help='Enable handcrafted heuristic eval in C++ AlphaBeta.')
    p.add_argument('--ab-native-model', type=str,
                   default='checkpoints/gnn_eval_v3_native.bin',
                   help='Native C++ NN checkpoint for AB ordering.')
    p.add_argument('--ab-nn-ordering-depth', type=int, default=3,
                   help='Minimum depth to apply NN move ordering.')
    p.add_argument('--ab-randomize-start',
                   action=argparse.BooleanOptionalAction, default=True,
                   help='Randomize openings for ab-supervised generation.')

    return p
