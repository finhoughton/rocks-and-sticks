from operator import itemgetter

HALF_AREA_COUNTS = False
STARTING_STICK = True

N_ROCKS = 2
ALPHABETA_DEPTH = 4

# Set to an int (e.g. 0) to make MCTS deterministic for testing.
MCTS_SEED: int | None = 0

second = itemgetter(1)
