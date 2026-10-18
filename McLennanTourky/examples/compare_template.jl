#=
Template for comparing the Julia `mclennan_tourky` with the Python version
on a single game, by hand.

Put the same payoff values in this file and in `compare_template.py`, so that
both are applied to the same game, and check that they return the same Nash
equilibrium.

Usage:
1. Edit only the "Define the game" section below
   (and copy *the same values* to compare_template.py).
2. Run `julia --project=.. compare_template.jl` from this directory.
3. Compare the output with that of `python3 compare_template.py`.

Note: the payoff values are the same as in Python, but action indices (e.g.
in `init`) are 1-based in Julia, i.e., the Python value + 1.
=#

using McLennanTourky

# ============================================================
# Define the game (use exactly the same values as in the .py file)
# ============================================================

# Example: symmetric N-player game (3-player anti-coordination game, v=2)
# payoff_array[own_action, opp1_action, ..., opp_{N-1}_action]
# Action indices are 1-based
N = 3
v = 2.0

payoff_array = zeros(2, 2, 2)
payoff_array[1, :, :] .= 1.0    # Action 1 (safe): payoff 1 regardless
payoff_array[2, :, :] .= 0.0    # Action 2 (risky): payoff 0 in general
payoff_array[2, 1, 1] = v       # ... and v only if all the others play action 1

players = ntuple(_ -> Player(payoff_array), N)
g = NormalFormGame(players)

# For an asymmetric game (different payoff_array for each player), replace
# `players`/`g` above with, e.g.:
#
# A  = [3.0 3.0; 2.0 5.0; 0.0 6.0]        # player 1: own=3, opp=2
# B1 = [3.0 2.0 3.0; 2.0 6.0 1.0]         # player 2: own=2, opp=3
# g = NormalFormGame((Player(A), Player(B1)))

# ============================================================
# Computation and output (usually no need to edit)
# ============================================================
epsilon = 1e-5
init = nothing  # To specify pure initial actions, a 1-based Int tuple, e.g. (1, 1, 1)

NE, res = mclennan_tourky(g; init=init, epsilon=epsilon, full_output=true)

println("=== Julia ===")
for (i, action) in enumerate(NE)
    println("player $(i-1): ", round.(action; digits=8))
end
println("converged = ", res.converged, ", num_iter = ", res.num_iter)
println("is_nash(NE, tol=epsilon) = ", is_nash(g, NE; tol=epsilon))
