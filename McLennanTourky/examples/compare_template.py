"""
Template for comparing the Python (QuantEcon.py) `mclennan_tourky` with the
Julia version on a single game, by hand.

Put the same payoff values in this file and in `compare_template.jl`, so that
both are applied to the same game, and check that they return the same Nash
equilibrium.

Usage:
1. Edit only the "Define the game" section below
   (and copy *the same values* to compare_template.jl).
2. Run `python3 compare_template.py`.
3. Compare the output with that of `julia --project=.. compare_template.jl`.
"""
import numpy as np
from quantecon.game_theory import Player, NormalFormGame, mclennan_tourky

np.set_printoptions(precision=8, suppress=True)

# ============================================================
# Define the game (use exactly the same values as in the .jl file)
# ============================================================

# Example: symmetric N-player game (3-player anti-coordination game, v=2)
# payoff_array[own_action, opp1_action, ..., opp_{N-1}_action]
# Action indices are 0-based (as Python array indices)
N = 3
v = 2.0

payoff_array = np.zeros((2, 2, 2))
payoff_array[0, :, :] = 1.0     # Action 0 (safe): payoff 1 regardless
payoff_array[1, :, :] = 0.0     # Action 1 (risky): payoff 0 in general
payoff_array[1, 0, 0] = v       # ... and v only if all the others play action 0

players = [Player(payoff_array) for _ in range(N)]
g = NormalFormGame(players)

# For an asymmetric game (different payoff_array for each player), replace
# `players`/`g` above with, e.g.:
#
# A  = np.array([[3., 3.], [2., 5.], [0., 6.]])   # player 0: own=3, opp=2
# B1 = np.array([[3., 2., 3.], [2., 6., 1.]])      # player 1: own=2, opp=3
# g = NormalFormGame([Player(A), Player(B1)])

# ============================================================
# Computation and output (usually no need to edit)
# ============================================================
epsilon = 1e-5
init = None  # To specify pure initial actions, a 0-based int tuple, e.g. (0, 0, 0)

NE, res = mclennan_tourky(g, init=init, epsilon=epsilon, full_output=True)

print("=== Python (quantecon) ===")
for i, action in enumerate(NE):
    print(f"player {i}: {action}")
print(f"converged = {res.converged}, num_iter = {res.num_iter}")
print(f"is_nash(NE, tol=epsilon) = {g.is_nash(NE, tol=epsilon)}")
