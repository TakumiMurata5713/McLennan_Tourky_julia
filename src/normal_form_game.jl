#=
Generic representation of an N-player normal form game.

Port of the subset of `quantecon/game_theory/normal_form_game.py` required by
`mclennan_tourky` and `lemke_howson` (`Player`, `NormalFormGame`,
`payoff_vector`, `best_response`, `is_best_response`, `is_nash`,
`pure2mixed`). Features not used here, such as `delete_action` and
`is_dominated`, are omitted.

Action indices are 1-based.
=#

# --- Player --------------------------------------------------------------

"""
    Player(payoff_array; tol=1e-8)

Create a player from an array such that `payoff_array[a_1, a_2, ..., a_N]` is
the player's payoff when they play action `a_1` and their N-1 opponents play
actions `a_2, ..., a_N`, respectively.
"""
struct Player{T<:Real, A<:AbstractArray{T}}
    payoff_array::A
    num_actions::Int
    num_opponents::Int
    tol::Float64
end

function Player(payoff_array::AbstractArray{T}; tol::Float64=1e-8) where {T<:Real}
    ndims(payoff_array) == 0 && throw(ArgumentError("payoff_array must be an array"))
    num_opponents = ndims(payoff_array) - 1
    num_actions = size(payoff_array, 1)
    return Player(payoff_array, num_actions, num_opponents, tol)
end

"""
    _reduce_last_player(payoff_array, action)

Contract the last axis of `payoff_array` with the action of the corresponding
opponent (an integer for a pure action, a vector for a mixed action).
"""
function _reduce_last_player(payoff_array::AbstractArray, action)
    nd = ndims(payoff_array)
    if action isa Integer
        idx = ntuple(k -> k == nd ? action : Colon(), nd)
        return payoff_array[idx...]
    else
        sz = size(payoff_array)
        A2 = reshape(payoff_array, :, sz[end])
        v = A2 * action
        return nd == 1 ? v : reshape(v, sz[1:end-1])
    end
end

"""
    payoff_vector(player, opponents_actions)

Return the vector of the player's payoffs for each of their own actions,
given the opponents' action profile `opponents_actions`.

- `num_opponents == 0`: `opponents_actions` is ignored.
- `num_opponents == 1`: `opponents_actions` is an integer (pure action) or a
  vector (mixed action).
- `num_opponents >= 2`: `opponents_actions` is a collection of length
  `num_opponents`, ordered as `(i+1, ..., N, 1, ..., i-1)` for player `i`.
"""
function payoff_vector(player::Player, opponents_actions)
    no = player.num_opponents
    if no == 0
        return player.payoff_array
    elseif no == 1
        return _reduce_last_player(player.payoff_array, opponents_actions)
    else
        pv = player.payoff_array
        for i in no:-1:1
            pv = _reduce_last_player(pv, opponents_actions[i])
        end
        return pv
    end
end

"""
    is_best_response(player, own_action, opponents_actions; tol=nothing)

Return `true` if `own_action` is a best response to `opponents_actions`.
If `tol` is `nothing`, `player.tol` is used.
"""
function is_best_response(player::Player, own_action, opponents_actions;
                          tol::Union{Nothing,Float64}=nothing)
    t = tol === nothing ? player.tol : tol
    pv = payoff_vector(player, opponents_actions)
    payoff_max = maximum(pv)

    if own_action isa Integer
        return pv[own_action] >= payoff_max - t
    else
        return (own_action' * pv) >= payoff_max - t
    end
end

"""
    best_response(player, opponents_actions; tol=nothing)

Return a pure best response to `opponents_actions`. Ties (up to `tol`) are
broken in favor of the smallest index, as with `tie_breaking='smallest'` in
the Python version.
"""
function best_response(player::Player, opponents_actions;
                       tol::Union{Nothing,Float64}=nothing)
    t = tol === nothing ? player.tol : tol
    pv = payoff_vector(player, opponents_actions)
    payoff_max = maximum(pv)
    for a in 1:player.num_actions
        if pv[a] >= payoff_max - t
            return a
        end
    end
    return player.num_actions  # unreachable
end

# --- NormalFormGame --------------------------------------------------------

"""
    NormalFormGame(players)

Create an N-player normal form game from a Tuple/Vector of `Player`s.
"""
struct NormalFormGame{P<:Player}
    players::Tuple{Vararg{P}}
    N::Int
    nums_actions::Tuple{Vararg{Int}}
end

function NormalFormGame(players)
    players_tup = Tuple(players)
    N = length(players_tup)
    nums_actions = ntuple(i -> players_tup[i].num_actions, N)
    return NormalFormGame(players_tup, N, nums_actions)
end

NormalFormGame(players::Player...) = NormalFormGame(players)

"""
    is_nash(g, action_profile; tol=nothing)

Return `true` if `action_profile` (a Tuple/Vector of pure or mixed actions,
one for each player) is a Nash equilibrium of `g`.
"""
function is_nash(g::NormalFormGame, action_profile; tol::Union{Nothing,Float64}=nothing)
    N = g.N

    if N == 2
        for i in 1:2
            own_action = action_profile[i]
            opponent_action = action_profile[3-i]
            if !is_best_response(g.players[i], own_action, opponent_action; tol=tol)
                return false
            end
        end
    else
        for i in 1:N
            own_action = action_profile[i]
            opponents_actions = ntuple(k -> action_profile[mod1(i + k, N)], N - 1)
            if !is_best_response(g.players[i], own_action, opponents_actions; tol=tol)
                return false
            end
        end
    end

    return true
end

# --- Utilities ---------------------------------------------------------------

"""
    pure2mixed(num_actions, action)

Convert the pure action `action` (1-based) to the corresponding mixed action.
"""
function pure2mixed(num_actions::Integer, action::Integer)
    mixed_action = zeros(Float64, num_actions)
    mixed_action[action] = 1.0
    return mixed_action
end
