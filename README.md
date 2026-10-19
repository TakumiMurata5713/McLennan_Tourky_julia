# McLennanTourky.jl

A Julia port of the parts of [QuantEcon.py](https://github.com/QuantEcon/QuantEcon.py)
needed to compute a mixed-strategy Nash equilibrium of an N-player normal form
game with the **imitation game algorithm** of McLennan and Tourky (2006).

The algorithm finds an approximate fixed point of a (selection of the) best
response correspondence. Internally it solves a sequence of two-player
"imitation games" with the **Lemke-Howson algorithm**.

## Layout

The source files mirror the module structure of QuantEcon.py:

| Julia file                  | Python counterpart                          | Contents |
|-----------------------------|---------------------------------------------|----------|
| `src/normal_form_game.jl`   | `quantecon/game_theory/normal_form_game.py` | `Player`, `NormalFormGame`, `payoff_vector`, `best_response`, `is_best_response`, `is_nash`, `pure2mixed` |
| `src/pivoting.jl`           | `quantecon/optimize/pivoting.py`            | Pivoting step and lexico-minimum ratio test |
| `src/lemke_howson.jl`       | `quantecon/game_theory/lemke_howson.py`     | `lemke_howson` (with capping), `_lemke_howson_tbl!`, `_get_mixed_actions` |
| `src/compute_fp.jl`         | `quantecon/_compute_fp.py`                  | `compute_fixed_point` (`:iteration` / `:imitation_game`), `_compute_fixed_point_ig`, `_initialize_tableaux_ig!` |
| `src/mclennan_tourky.jl`    | `quantecon/game_theory/mclennan_tourky.py`  | `mclennan_tourky` |
| `src/McLennanTourky.jl`     | –                                           | Module definition and exports |

```
McLennanTourky/
├── Project.toml
├── README.md
├── src/                         # Package source (see table above)
├── test/
│   ├── runtests.jl              # Unit tests (run by Pkg.test)
│   └── python_comparison/
│       ├── generate_reference.py  # Runs QuantEcon.py on 1000 random games
│       ├── random_games.jl        # Regenerates the same games in Julia
│       ├── compare.jl             # Runs Julia on them and compares
│       └── results/
│           ├── python_reference.tsv  # Python results
│           ├── comparison.csv        # Per-game comparison
│           └── summary.md            # Summary of the comparison
└── examples/
    ├── compare_template.jl      # Solve one game by hand in Julia ...
    └── compare_template.py      # ... and the same game in Python
```

## Differences from the Python version

- **Indices are 1-based.** Actions, `init`, and `init_pivot` are 1-based
  (Python value + 1).
- `full_output=true` returns a `NamedTuple` instead of `NashResult`.
- The imitation game routine stores the sequences of points in growable
  `Vector`s instead of the doubling fixed-size buffers used in Python for
  Numba; the algorithm is the same.
- `normal_form_game.jl` contains only what is needed here (e.g. no
  `delete_action`, `is_dominated`, or `tie_breaking='random'`).
- `compute_fixed_point` does not take `*args, **kwargs`; pass a closure
  instead.

## Usage

```julia
julia> using Pkg; Pkg.activate("path/to/McLennanTourky")

julia> using McLennanTourky

julia> # 3-player anti-coordination game: action 1 always pays 1,
       # action 2 pays 2 only if both opponents play action 1
       payoff_array = zeros(2, 2, 2);

julia> payoff_array[1, :, :] .= 1.0; payoff_array[2, 1, 1] = 2.0;

julia> g = NormalFormGame(ntuple(_ -> Player(payoff_array), 3));

julia> NE, res = mclennan_tourky(g; epsilon=1e-5, full_output=true);

julia> NE[1]   # ≈ [1/√2, 1 - 1/√2]
2-element Vector{Float64}:
 0.7071075439463927
 0.29289245605360725

julia> is_nash(g, NE; tol=1e-5)
true
```

Two-player games can also be solved directly with Lemke-Howson:

```julia
julia> A = [3.0 3.0; 2.0 5.0; 0.0 6.0]; B = [3.0 2.0 3.0; 2.0 6.0 1.0];

julia> lemke_howson(NormalFormGame((Player(A), Player(B))); init_pivot=2)
([0.0, 0.33333333333333326, 0.6666666666666667], [0.3333333333333334, 0.6666666666666666])
```

## Tests

### Unit tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

`test/runtests.jl` checks the game primitives, the pivoting / Lemke-Howson
internals against the examples in the QuantEcon.py docstrings,
`lemke_howson` for every initial pivot, `compute_fixed_point` (both methods),
and `mclennan_tourky` on known games, against values computed with
QuantEcon.py 0.11.2. It also checks that a duck-typed game object works,
and, if `test/python_comparison/results/python_reference.tsv` exists,
re-runs the Python comparison on the games with N ≤ 5.

### Comparison with Python on 1000 random games

```bash
# 1. Python reference (needs quantecon; runs games in parallel processes)
python3 test/python_comparison/generate_reference.py
# 2. Julia comparison (multi-threaded)
julia --project=. -t auto test/python_comparison/compare.jl
```

- 1000 random games with **N = 3, …, 20 players** (64 games for each
  N ≤ 16; 40, 30, 20, 14 games for N = 17, 18, 19, 20, since the cost grows
  like N·2^N). Each player has 2–10 actions, with at most 2^20 action
  profiles per game. Payoffs are uniform on [0, 1), initial actions random.
- Settings: `epsilon = 1e-3`, `max_iter = 200` (the defaults).
- The payoffs are produced by a counter-based splitmix64 generator
  implemented identically in Python and Julia, so both sides solve
  bit-for-bit identical games without storing them on disk.
- Compared: the pure best-response profile in **every** iteration (the
  discrete path of the algorithm), `converged`, `num_iter`, and the
  equilibrium `x_star`.

#### Results

Full details: [`test/python_comparison/results/summary.md`](test/python_comparison/results/summary.md)
(per-game data in `comparison.csv`).

- `converged` and `num_iter` agree in **all 1000 games**.
- All **115 games that converged** match completely: identical best
  response path and `x_star` within 1.2e-5 (most within 1e-10).
- In 918 games the best response path is identical in every iteration.
- The 82 games whose paths differ all ran to `max_iter = 200` without
  converging. Their paths agree exactly up to iteration 139 or later (median
  188), and only then diverge.
- Overall 892 / 1000 games satisfy every criterion, including
  `max |x_star_jl - x_star_py| <= 1e-3`. All 108 that do not are
  non-converged runs stopped at `max_iter`.

**Why the late divergences happen.** Given bit-identical inputs, the
Lemke-Howson step already differs by about 1e-16 between Numba (LLVM) and
Julia, because the two compilers round the floating point operations of the
pivoting slightly differently. Each imitation game is built from the previous
iterates, so these differences accumulate over the iterations (typically
1e-12 by iteration 80 and 1e-6 by 160). Eventually they flip a nearly tied
best response, and the two runs then follow different, equally valid paths.
A logic difference would show up in the early iterations, and none does.

The same effect occurs within Julia itself. Under `Pkg.test`, Julia runs with
`--check-bounds=yes`, which changes how the loops are compiled. In that
setting, one converged game (game 83, N = 4) leaves the Python path at
iteration 178 and converges at iteration 181 instead of 182. The unit test
therefore requires the paths to agree for the first 100 iterations, and
requires full agreement only for games whose paths agree throughout. The
results above come from `compare.jl`, which runs with default settings.

Julia is also faster on large games: for N = 20, about 25 s versus 204 s
per game. Both sides ran several games in parallel, so this is indicative.

## References

- A. McLennan and R. Tourky, "From Imitation Games to Kakutani," 2006.
- C. E. Lemke and J. T. Howson, "Equilibrium Points of Bimatrix Games,"
  Journal of the Society for Industrial and Applied Mathematics (1964).
- B. von Stengel, "Equilibrium Computation for Two-Player Games in Strategic
  and Extensive Form," Chapter 3, Algorithmic Game Theory, 2007.
- B. Codenotti, S. De Rossi, and M. Pagan, "An Experimental Analysis of
  Lemke-Howson Algorithm," arXiv:0811.3247, 2008.


Co-authored-by: Claude <noreply@anthropic.com>
