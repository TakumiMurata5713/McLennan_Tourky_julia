#=
Compute mixed Nash equilibria of an N-player normal form game by applying the
imitation game algorithm by McLennan and Tourky (2006) to the best response
correspondence.

Port of `quantecon/game_theory/mclennan_tourky.py`.

Requirements on the game `g` (satisfied by `NormalFormGame`; any duck-typed
object providing the same interface also works):
- `g.N`            : number of players
- `g.nums_actions` : tuple/vector of the numbers of pure actions
- `g.players[i]`   : the i-th player, for which
  `best_response(player, opponents_actions)` returns a 1-based pure best
  response (ties broken in favor of the smallest index), where
  `opponents_actions` is a single (pure or mixed) action if `N == 2` and a
  tuple of N-1 actions if `N > 2`
- `is_nash(g, action_profile; tol=epsilon)` returns a `Bool`
=#

"""
    mclennan_tourky(g; init=nothing, epsilon=1e-3, max_iter=200,
                    full_output=false)

Find one mixed-action epsilon-Nash equilibrium of an N-player normal form
game `g` by the fixed point computation algorithm by McLennan and Tourky
(2006).

# Arguments

- `g::NormalFormGame`: NormalFormGame instance.
- `init=nothing`: Initial action profile, a collection of length N whose
  elements are pure actions (1-based integers) or mixed actions (vectors).
  If `nothing`, the profile in which every player plays action 1 is used.
- `epsilon::Float64=1e-3`: Value of epsilon-optimality.
- `max_iter::Int=200`: Maximum number of iterations.
- `full_output::Bool=false`: If `true`, also return a NamedTuple with
  information about the computation.

# Returns

- `NE::Tuple`: Tuple of mixed actions (one `Vector{Float64}` per player).
- `res::NamedTuple` (only if `full_output=true`): fields `NE`, `converged`,
  `num_iter`, `max_iter`, `init`, `epsilon`. (The Python version returns a
  `NashResult` dict subclass; a NamedTuple serves the same purpose here.)

# Examples

Consider the 3-player anti-coordination game where each player has a "safe"
action 1 with payoff 1 and a "risky" action 2 with payoff `v` if all the
others play action 1 and 0 otherwise. Its symmetric mixed equilibrium plays
action 1 with probability `p* = 1 / v^(1/(N-1))`.

```julia
julia> N, v = 3, 2.0;

julia> payoff_array = zeros(2, 2, 2);

julia> payoff_array[1, :, :] .= 1.0; payoff_array[2, 1, 1] = v;

julia> g = NormalFormGame(ntuple(_ -> Player(payoff_array), N));

julia> NE = mclennan_tourky(g; epsilon=1e-5);

julia> is_nash(g, NE; tol=1e-5)
true
```

# References

- A. McLennan and R. Tourky, "From Imitation Games to Kakutani," 2006.
"""
function mclennan_tourky(g; init=nothing, epsilon::Float64=1e-3,
                         max_iter::Int=200, full_output::Bool=false)
    N = g.N
    N < 2 && throw(ArgumentError("Not implemented for 1-player games"))

    if init === nothing
        # Action 1 for every player
        init = ntuple(_ -> 1, N)
    end
    length(init) != N && throw(ArgumentError("init must be of length $N"))

    nums_actions = g.nums_actions
    indptr = Vector{Int}(undef, N + 1)
    indptr[1] = 1
    for i in 1:N
        indptr[i+1] = indptr[i] + nums_actions[i]
    end

    x_init = _flatten_action_profile(init, indptr)

    is_approx_fp = x -> _is_epsilon_nash(x, g, epsilon, indptr)
    br_selection = x -> _best_response_selection(x, g, indptr)

    x_star, converged, num_iter =
        _compute_fixed_point_ig(br_selection, x_init, max_iter, is_approx_fp)

    NE = _get_action_profile(x_star, indptr)

    if !full_output
        return NE
    end

    res = (NE=NE, converged=converged, num_iter=num_iter,
           max_iter=max_iter, init=init, epsilon=epsilon)

    return NE, res
end

"""
    _best_response_selection(x, g, indptr)

Selection of the best response correspondence of `g` that selects the best
response action with the smallest index when there are ties, where the input
and output are flattened action profiles.
"""
function _best_response_selection(x::AbstractVector, g, indptr::AbstractVector{Int})
    N = g.N
    out = zeros(Float64, indptr[end] - 1)

    if N == 2
        for i in 1:2
            j = 3 - i  # Opponent
            opponent_action = x[indptr[j]:indptr[j+1]-1]
            pure_br = best_response(g.players[i], opponent_action)
            out[indptr[i] + pure_br - 1] = 1.0
        end
    else
        for i in 1:N
            opponent_actions = ntuple(k -> begin
                j = mod1(i + k, N)
                x[indptr[j]:indptr[j+1]-1]
            end, N - 1)
            pure_br = best_response(g.players[i], opponent_actions)
            out[indptr[i] + pure_br - 1] = 1.0
        end
    end

    return out
end

"""
    _is_epsilon_nash(x, g, epsilon, indptr)

Determine whether the flattened action profile `x` is an epsilon-Nash
equilibrium of `g`.
"""
function _is_epsilon_nash(x::AbstractVector, g, epsilon::Float64, indptr::AbstractVector{Int})
    action_profile = _get_action_profile(x, indptr)
    return is_nash(g, action_profile; tol=epsilon)
end

"""
    _get_action_profile(x, indptr)

Obtain a tuple of mixed actions from a flattened action profile `x`.
"""
function _get_action_profile(x::AbstractVector, indptr::AbstractVector{Int})
    N = length(indptr) - 1
    return ntuple(i -> x[indptr[i]:indptr[i+1]-1], N)
end

"""
    _flatten_action_profile(action_profile, indptr)

Flatten the given action profile, whose elements are pure actions (`Int`) or
mixed actions (`Vector{Float64}`), into a single vector.
"""
function _flatten_action_profile(action_profile, indptr::AbstractVector{Int})
    N = length(indptr) - 1
    out = Vector{Float64}(undef, indptr[end] - 1)

    for i in 1:N
        ai = action_profile[i]
        if ai isa Integer  # Pure action
            num_actions = indptr[i+1] - indptr[i]
            mixed_action = pure2mixed(num_actions, ai)
        else  # Mixed action
            mixed_action = ai
        end
        out[indptr[i]:indptr[i+1]-1] = mixed_action
    end

    return out
end
