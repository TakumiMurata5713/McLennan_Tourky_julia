#=
Random games shared with `generate_reference.py`.

Every payoff value is produced by a counter-based splitmix64 generator that
is implemented identically in Python, so that both languages build exactly
the same games (bit-for-bit) without storing them on disk.
=#

const GOLDEN = 0x9E3779B97F4A7C15
const MIX1 = 0xBF58476D1CE4E5B9
const MIX2 = 0x94D049BB133111EB
const PAYOFF_BASE = 0x5EED5EED5EED5EED

# splitmix64 finalizer (UInt64 arithmetic wraps around, as in NumPy)
function _mix64(z::UInt64)
    z = (z ⊻ (z >> 30)) * MIX1
    z = (z ⊻ (z >> 27)) * MIX2
    return z ⊻ (z >> 31)
end

"""
    uniform_stream(game_id, player, len)

`len` floats in [0, 1) for the given (0-based) `game_id` and `player`.
"""
function uniform_stream(game_id::Integer, player::Integer, len::Integer)
    key = _mix64(PAYOFF_BASE + UInt64(game_id * 64 + player))
    return [Float64(_mix64(key + UInt64(c) * GOLDEN) >> 11) * 2.0^-53 for c in 1:len]
end

"""
    random_game(game_id, nums_actions)

Random game whose player `i` has a payoff array of size
`(n_i, n_{i+1}, ..., n_{i-1})`. The values are those of
`uniform_stream(game_id, i-1, ...)` laid out in row-major (C) order, as in
the Python version.
"""
function random_game(game_id::Integer, nums_actions)
    N = length(nums_actions)
    players = map(1:N) do i
        shape = ntuple(k -> nums_actions[mod1(i + k - 1, N)], N)
        flat = uniform_stream(game_id, i - 1, prod(shape))
        # Row-major -> column-major: fill the reversed shape, then reverse the axes
        Player(permutedims(reshape(flat, reverse(shape)), N:-1:1))
    end
    return NormalFormGame(players)
end
