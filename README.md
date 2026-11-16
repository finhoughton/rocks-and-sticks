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
- `engine_src/`: the `rs_engine` C++ alpha-beta engine (the strongest opponent) and its self-play tool
- `tools/`: match runner and evaluation tuning for `rs_engine`
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

## Playing against the engine

```bash
python3 main.py --ai rs                    # you move first
python3 main.py --ai rs --human-player 1   # the engine moves first
python3 main.py --ai rs --rs-time 10       # seconds per engine move (default 5)
python3 main.py --ai rs --no-ponder        # do not let it think while you do
```

With `--ai rs` the engine keeps thinking while you consider your move ("pondering"): it
searches the position you face, so the search after your reply starts with much of its work
done. The longer you think, the stronger its reply.

To watch the engine play itself: `python3 tools/watch.py` (see `--help` for time per move,
number of games and pacing). To replay a saved game with the engine's evaluation of every
position: `python3 scripts/rewatch.py saved_games_human/game_00036.json --analyse 2`.

Evaluations read like chess ones. Positive favours Player 1, negative Player 2, scaled so
a number means the same win chance as that many pawns on a chess analysis board (+1.0 is
about 59%, +3.0 about 75%); the estimated win chance is shown beside it. `+M3` means Player 1
has a forced win in 3 of its own moves (`-M3` for Player 2). The win chances are calibrated
on self-play results (`tools/calibrate_rs_eval.py`), so they describe engine-vs-engine play.

Moves are typed as `X Y D` (stick from X,Y in direction N, NE, E, SE, S, SW, W or NW),
`X Y R` (rock) or `P` (pass). Beside the input box the window shows the rocks each player may
still place before their next stick. To give up, type `resign` or click **Resign** twice: the
game is saved and rated as a loss. The engine prints its evaluation, search depth and expected
line after each of its moves; add `--show-eval` to see the evaluation bar while you play.

## The rs engine

`engine_src/` is a self-contained C++ engine, built as the `rs_engine` Python module by
`setup.py` (or `./build_engine.sh`, which also builds the `engine_src/rs_selfplay` tool).

- `rs_state.hpp`: the rules on a fixed grid, with make/undo. It must match `game.py`
  exactly; `tests/test_rs_engine_parity.py` fuzzes the two against each other.
- `rs_search.hpp`: iterative-deepening alpha-beta. Passing is legal, so the null move is a
  real move. Sticks that complete a polygon are searched like captures, and facing a
  polygon that only the opponent can complete is treated like being in check.
- `rs_eval.hpp`: the evaluation. It counts features (small polygons by how complete they are
  and who has rocked their corners; pairs of adjacent rocks belonging to one player, since a
  stick between two of your own rocks can only ever be placed by you, which is how games are
  won; rocks in hand; side to move) and combines them either as a weighted sum ("linear") or
  as the average of that and a small neural network (69 inputs, hidden layers of 64 and 32;
  the "blend"). The blend is much stronger in short searches (+120 Elo at 100k positions per
  move, +135 at 300k) but weaker in long ones (-24 at 600k, -63 at 1M), so the engine uses
  it below about 400k positions per move (or about 0.7 s of thinking) and the linear
  evaluation above (`players/rs_player.py` `default_weights`).
- `rs_parallel.hpp`: runs several search threads over one shared transposition table.
- `weights.txt`: the linear evaluation. `eval.txt`: the blend (`blend 0.5`, then the linear
  weights, then the network). `eval_network.txt`: the network alone. Any of the three
  formats loads with `--weights`.
- `book.json`: the opening book (see below); `players/rs_book.py` reads it.
- `versions/v<n>/`: frozen copies of earlier engine versions, so they can still play rated
  games against the current one (`rs:100k:v3`); `./build_engine.sh` builds them.

Compare engines or settings:

```bash
python3 tools/arena.py --a rs:1s --b ab:3 --games 10       # against the older bots
engine_src/rs_selfplay match --games 400 --nodes 100000 --threads 4 --open-min 0 --open-max 0 \
    --open-noise 40 --open-noise-plies 4 --weights engine_src/eval.txt --weights2 other.txt
```

`rs_selfplay match` plays each opening from both sides; with node budgets the result does not
depend on how busy the computer is. To compare evaluations of different speeds at equal
time, give the faster one proportionally more nodes (`--nodes2`).

## Ratings

Every rated game is a line in `ratings.jsonl`, and `python3 tools/rate.py` fits Elo-scale
ratings to all of them at once, with a 95% uncertainty for each and the advantage of moving
first. The old depth-3 alpha-beta bot (`ab:3`) is fixed at 1000; 200 points apart means
the stronger side scores about 76%.

```bash
python3 tools/rate.py                                 # the table
python3 tools/rate.py match rs:100k ab:3 --games 20   # a rated bot-vs-bot match
python3 tools/rate.py record you rs:5s:4t --winner 2  # a game played elsewhere
python3 main.py --bot rs:100k                         # play a particular bot
```

Bots are named by spec (`players/specs.py`): `random`, `ab:<depth>`, `mcts:<rollouts>`,
`rs:<budget>[:<threads>t][:nobook]` with a budget such as `5s`, `500ms`, `100k` or `1M`.
Games you finish in `main.py` against `--bot` or `--ai rs` are recorded automatically under
the name `you` (`--name` to change, `--no-rate` to skip). For bot-vs-bot matches prefer node
budgets: timed bots get weaker when something else is using the CPU. After changing how the
engine plays, freeze the current version first (copy the engine sources, `eval.txt` as
`weights.txt` and `book.json` to `engine_src/versions/v<n>/` and run `./build_engine.sh`), then
bump `ENGINE_VERSION` in `players/rs_player.py`; benchmark with e.g.
`python3 tools/rate.py match rs:100k rs:100k:v<n> --games 300 --jobs 4`. A match
ends by saying how many of its games were different (repeats add no information) and how
many moves per game each bot took from the opening book.

Search one position for a long time (prints each depth as it completes):

```bash
python3 tools/analyse.py --minutes 60                      # the starting position
python3 tools/analyse.py --moves "0,1NE 0,0E" --minutes 10
```

Opening book. `engine_src/book.json` maps early positions to moves found by searches of
minutes rather than seconds; the engine plays from it when it can and searches as usual
otherwise. Build or extend it (it resumes where it stopped, and can be interrupted):

```bash
caffeinate -i python3 tools/build_book.py --hours 8
```

Retrain the evaluation network from fresh self-play (about an hour of data on 4 cores):

```bash
engine_src/rs_selfplay gen --games 8000 --nodes 30000 --threads 4 --open-min 0 --open-max 0 \
    --open-noise 60 --open-noise-plies 6 --leaves 3 --leaf-nodes 30000 \
    --weights engine_src/eval.txt --out data/rs/gen3.bin
python3 tools/train_rs_mlp.py data/rs/gen*.bin --test data/rs/calib_sound.bin --out data/rs/candidates/new.txt
# blend it with the linear weights, then play a match against engine_src/eval.txt:
{ echo "blend 0.5"; cat engine_src/weights.txt; cat data/rs/candidates/new.txt; } > data/rs/candidates/new_blend.txt
```

`--leaves 3` adds, for every game position, three positions reached by random moves from it,
labelled with a short search. They matter: a network trained only on positions from games
predicted results well but played about 50 Elo worse, because the search mostly asks the
evaluation about odd positions deep in its tree. (`tools/tune_rs_eval.py` fits the older
linear weights the same way.)

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
