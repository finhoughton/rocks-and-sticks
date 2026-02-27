# Rocks and Sticks

Implementation of the rocks and sicks abstract strategy game and AI opponents
- Python game/rules logic
- C++ search engine (`players_ext`) for fast AlphaBeta/MCTS
- GNN value model training and evaluation
- Self-play / retraining loops

## Repository Layout

- `rules.txt`: rules of the game
- `game.py`, `models.py`, `models.py`: core game logic
- `main.py`: play games between human players or against a bot
- `players_ext_src/`: C++ extension. Mainly for the oppoents but also has implementations of game logic etc for speed.
- `players/`: AI Opponents
- `gnn/`: GNN encoding, dataset loading, supervised training utilities
- `rl/`: run-loop orchestration, self-play conversion/training pipeline
- `scripts/`: scripts
- `tests/`: pytest test suite

## Starting

### 1) Install dependencies

```bash
python3 -m pip install -r requirements-dev.txt
```

### 2) Build C++ extension

```bash
python3 setup.py build_ext --inplace
```

### 3) Run tests

```bash
make test
```

## Training Workflows

## A) Supervised GNN training from saved games

```bash
python3 -m gnn.gnn_main \
  --epochs 25 \
  --batch-size 64 \
  --lr 5e-4 \
  --out checkpoints/gnn_eval_custom.pt \
  --soft-labels \
  --augment-sym \
  --only-extra-dirs \
  --extra-dirs saved_games_cpp_ab
```

## B) Train → Play Games → Retrain loop (AB-supervised)

This loop generates games with C++ AlphaBeta, then retrains `GNNEval` each iteration.

```bash
python3 -m rl.run_loop \
  --loop-mode ab-supervised \
  --iters 10 \
  --games 100 \
  --replay-window 3 \
  --epochs 15 \
  --batch-size 64 \
  --lr 5e-4 \
  --device cpu \
  --backend cpp \
  --saved-games-dir saved_games_cpp_ab_loop/iter_{iter} \
  --out-dir checkpoints \
  --ab-time-limit 3000 \
  --ab-nn-ordering-depth 3 \
  --ab-max-moves 120 \
  --ab-randomize-start \
  --randomize-max-sticks 5 \
  --eval-games 100 \
  --eval-rollouts 200 \
  --eval-jobs 4 \
  --seed 12345 \
  --strength-log logs/strength_curve.jsonl
```
