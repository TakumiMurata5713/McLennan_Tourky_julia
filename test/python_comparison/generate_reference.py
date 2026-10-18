"""
Generate reference results of QuantEcon.py's `mclennan_tourky` on random
N-player games, to be compared with the Julia port by `compare.jl`.

Games are not stored on disk (a 20-player game has 20 payoff arrays of 2^20
entries each). Instead, every payoff value is produced by a counter-based
splitmix64 generator that is implemented identically in `random_games.jl`,
so that Julia can regenerate exactly the same games bit-for-bit.

Usage:
    python3 generate_reference.py [num_workers] [output_path]

Output: a TSV file with one line per game:
    game_id  N  nums_actions  init  converged  num_iter  time_sec  br_path  x_star
where `nums_actions`, `init` (0-based pure actions) and `x_star` (flattened
mixed action profile, full precision) are comma-separated, and `br_path` is
the sequence of pure best response profiles computed in the iterations
(one token per iteration, one digit per player = 0-based action), separated
by ';'.
"""
import os
# One BLAS thread per process; parallelism comes from running games in
# separate processes
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import importlib
import sys
import time
from multiprocessing import Pool

import numpy as np
from quantecon.game_theory import Player, NormalFormGame, mclennan_tourky

# The module object (the package re-exports a function with the same name)
_mt_module = importlib.import_module("quantecon.game_theory.mclennan_tourky")

# Settings shared with compare.jl
SEED = 20261018                # Seed for the game sizes and initial actions
# Number of games for each number of players N (1000 games in total).
# Fewer games for large N because the cost grows like N * 2^N.
NUM_GAMES_BY_N = {**{N: 64 for N in range(3, 17)},
                  17: 40, 18: 30, 19: 20, 20: 14}
MAX_ACTIONS = 10               # Maximum number of actions of a player
MAX_PROFILES = 2**20           # Upper bound on the number of action profiles
EPSILON = 1e-3                 # Default of mclennan_tourky
MAX_ITER = 200                 # Default of mclennan_tourky

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
MIX1 = np.uint64(0xBF58476D1CE4E5B9)
MIX2 = np.uint64(0x94D049BB133111EB)
PAYOFF_BASE = np.uint64(0x5EED5EED5EED5EED)


def _mix64(z):
    """splitmix64 finalizer, applied elementwise to a uint64 array."""
    z = (z ^ (z >> np.uint64(30))) * MIX1
    z = (z ^ (z >> np.uint64(27))) * MIX2
    return z ^ (z >> np.uint64(31))


def uniform_stream(game_id, player, length):
    """`length` floats in [0, 1) for the given (game_id, player) pair."""
    key = _mix64(np.array([PAYOFF_BASE + np.uint64(game_id * 64 + player)],
                          dtype=np.uint64))[0]
    counters = np.arange(1, length + 1, dtype=np.uint64)
    z = _mix64(key + counters * GOLDEN)
    return (z >> np.uint64(11)).astype(np.float64) * 2.0**-53


def random_game(game_id, nums_actions):
    """
    Random game whose player i (0-based) has a payoff array of shape
    (n_i, n_{i+1}, ..., n_{i-1}), filled in C (row-major) order from
    `uniform_stream(game_id, i, ...)`.
    """
    N = len(nums_actions)
    players = []
    for i in range(N):
        shape = tuple(nums_actions[(i + k) % N] for k in range(N))
        flat = uniform_stream(game_id, i, int(np.prod(shape)))
        players.append(Player(flat.reshape(shape)))
    return NormalFormGame(players)


def game_specs():
    """Numbers of players/actions and initial actions of each game."""
    rng = np.random.default_rng(SEED)
    specs = []
    game_id = 0
    for N, num_games in NUM_GAMES_BY_N.items():
        max_a = int(np.floor(MAX_PROFILES ** (1 / N) + 1e-9))
        max_a = max(2, min(MAX_ACTIONS, max_a))
        for _ in range(num_games):
            nums_actions = tuple(int(n) for n in
                                 rng.integers(2, max_a + 1, size=N))
            init = tuple(int(rng.integers(0, n)) for n in nums_actions)
            specs.append((game_id, N, nums_actions, init))
            game_id += 1
    return specs


def _csv(xs):
    return ",".join(repr(float(x)) if isinstance(x, float) else str(x)
                    for x in xs)


def solve(spec):
    """Solve one game; return its TSV line (without newline)."""
    game_id, N, nums_actions, init = spec
    g = random_game(game_id, nums_actions)

    # Record the best response profiles computed in each iteration by
    # wrapping the module-level selection function used by mclennan_tourky
    br_path = []
    orig = _mt_module._best_response_selection

    def recording_selection(x, g, indptr=None):
        y = orig(x, g, indptr)
        br_path.append("".join(str(int(np.argmax(y[indptr[i]:indptr[i+1]])))
                               for i in range(N)))
        return y

    _mt_module._best_response_selection = recording_selection
    try:
        t0 = time.perf_counter()
        NE, res = mclennan_tourky(g, init=init, epsilon=EPSILON,
                                  max_iter=MAX_ITER, full_output=True)
        elapsed = time.perf_counter() - t0
    finally:
        _mt_module._best_response_selection = orig

    x_star = np.concatenate(NE)
    return (f"{game_id}\t{N}\t{_csv(nums_actions)}\t{_csv(init)}\t"
            f"{str(bool(res.converged)).lower()}\t{res.num_iter}\t"
            f"{elapsed:.6f}\t{';'.join(br_path)}\t{_csv(x_star.tolist())}")


def main():
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else \
        max(1, (os.cpu_count() or 2) - 1)
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(here, "results", "python_reference.tsv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    specs = game_specs()
    # Largest games first for better load balancing
    order = sorted(specs, key=lambda s: -s[1] * np.prod(s[2]))

    lines = {}
    t_start = time.perf_counter()
    with Pool(num_workers) as pool:
        for k, line in enumerate(pool.imap_unordered(solve, order), 1):
            f = line.split("\t")
            lines[int(f[0])] = line
            print(f"[{k:4d}/{len(specs)}] game {f[0]:>4}: N={f[1]:>2} "
                  f"converged={f[4]} num_iter={f[5]} ({float(f[6]):.2f}s)",
                  flush=True)

    with open(out_path, "w") as f:
        f.write("game_id\tN\tnums_actions\tinit\tconverged\tnum_iter\t"
                "time_sec\tbr_path\tx_star\n")
        for game_id in sorted(lines):
            f.write(lines[game_id] + "\n")
    print(f"Done in {time.perf_counter() - t_start:.1f}s; wrote {out_path}")


if __name__ == "__main__":
    main()
