"""
    McLennanTourky

Julia port of the parts of QuantEcon.py needed to compute a mixed-strategy
Nash equilibrium of an N-player normal form game with the McLennan-Tourky
(2006) imitation game algorithm.

The file layout mirrors the Python package:

| Julia file              | Python counterpart                          |
|-------------------------|---------------------------------------------|
| `normal_form_game.jl`   | `quantecon/game_theory/normal_form_game.py` |
| `pivoting.jl`           | `quantecon/optimize/pivoting.py`            |
| `lemke_howson.jl`       | `quantecon/game_theory/lemke_howson.py`     |
| `compute_fp.jl`         | `quantecon/_compute_fp.py`                  |
| `mclennan_tourky.jl`    | `quantecon/game_theory/mclennan_tourky.py`  |

All action indices are 1-based (Python uses 0-based indices).
"""
module McLennanTourky

include("normal_form_game.jl")
include("pivoting.jl")
include("lemke_howson.jl")
include("compute_fp.jl")
include("mclennan_tourky.jl")

export Player, NormalFormGame, payoff_vector, best_response,
       is_best_response, is_nash, pure2mixed
export lemke_howson
export compute_fixed_point
export mclennan_tourky

end # module
