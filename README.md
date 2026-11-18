# Rocks and Sticks

The rocks and sticks abstract strategy game (rules in `rules.txt`), with bots to play it.

## Where things are

Everything you run to play goes through one command, `main.py` (see "Playing" below).

- `main.py`: play, watch, run matches, replay.
- `game.py`, `models.py`, `constants.py`: the rules and the game window.
- `players/`: every player. `specs.py` names the bots (`rs:5s:4t`, `ab:3`, ...); `rs_player.py`,
  `rs_book.py` and `rs_display.py` wrap the rs engine, its opening book and its evaluation
  display; the rest are the older bots.
- `engine_src/`: the rs engine, the strongest bot (C++), with its opening book and
  evaluation files. Built by `./build_engine.sh`.
- `tools/`: working on the rs engine and the ratings: `rate.py` (ratings table),
  `analyse.py` (search one position for a long time), `build_book.py` and `book_lines.py`
  (opening book), `train_rs_mlp.py`, `tune_rs_eval.py` and `calibrate_rs_eval.py` (evaluation).
- `ratings.py`, `ratings.jsonl`: the rating system and every rated game.
- `tests/`: pytest suite.

The older approach (a GNN evaluation trained by self-play, with C++ MCTS and alpha-beta),
kept working but no longer developed:

- `players_ext_src/`: the `players_ext` C++ extension (older alpha-beta, MCTS, game state).
- `gnn/`, `rl/`, `scripts/`: GNN training, the self-play training loop and its evaluation scripts.

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

## Playing

One command starts every kind of game. Its two arguments are Player 1 and Player 2: each a
bot, or a person (`human`, or a name).

```bash
python3 main.py                              # you against the engine, you move first
python3 main.py rs:5s:4t:ponder human        # the engine moves first
python3 main.py rs:100k                      # you against a particular bot
python3 main.py human human                  # two people at one screen
python3 main.py rs:2s:4t rs:2s:4t --games 3  # watch two bots, with an evaluation bar
python3 main.py rs:100k ab:3 --games 20 --no-display --jobs 4   # a quick match, no window
python3 main.py --replay saved_games_human/game_00040.json --analyse 2   # replay a saved game
```

Bots are named by spec (`players/specs.py`): `rs:<budget>[:<threads>t][:ponder][:nobook][:v<n>]`
is the rs engine, with a budget of time per move (`5s`, `500ms`) or positions searched per
move (`100k`, `1M`); `ab:<depth>`, `mcts:<rollouts>` and `random` are the older bots (their
model-backed variants are listed in `specs.py`). With `:ponder` the engine keeps thinking
while you consider your move, so the longer you think, the stronger its reply; the default
opponent is `rs:5s:4t:ponder`.

Games played in a window are saved (yours to `saved_games_human/`, bot games to
`saved_games_bots/`) with the engine's evaluations, and `--replay` shows them move by move;
`--analyse SECONDS` has the engine evaluate every position afresh. Every finished game
between two differently named players is also added to the ratings (see below) unless you
pass `--no-rate`. `python3 main.py --help` lists the remaining options.

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
python3 main.py rs:1s ab:3 --games 10 --no-display          # against the older bots
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
python3 tools/rate.py record you rs:5s:4t --winner 2  # a game played elsewhere
python3 main.py rs:100k ab:3 --games 20 --no-display  # a rated bot-vs-bot match
```

Your games are recorded under the name `you` (`--name` to change, or give a name instead of
`human`). For bot-vs-bot matches prefer node
budgets: timed bots get weaker when something else is using the CPU. After changing how the
engine plays, freeze the current version first (copy the engine sources, `eval.txt` as
`weights.txt` and `book.json` to `engine_src/versions/v<n>/` and run `./build_engine.sh`), then
bump `ENGINE_VERSION` in `players/rs_player.py`; benchmark with e.g.
`python3 main.py rs:100k rs:100k:v<n> --games 300 --no-display --jobs 4`. A match
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
