#=
Lemke-Howson algorithm for two-player normal form games.

Port of `quantecon/game_theory/lemke_howson.py`. `_lemke_howson_tbl` and
`_get_mixed_actions` are also used by the imitation game algorithm in
`compute_fp.jl`.

Variable indexing (1-based): for a game where players 1 and 2 have `m` and
`n` actions, variables `1, ..., m` correspond to player 1's actions and
`m+1, ..., m+n` to player 2's actions.
=#

"""
    lemke_howson(g; init_pivot=1, max_iter=10^6, capping=nothing,
                 full_output=false)

Find one mixed-action Nash equilibrium of a 2-player normal form game `g` by
the Lemke-Howson algorithm, implemented with "complementary pivoting" (see,
e.g., von Stengel (2007) for details).

# Arguments

- `g::NormalFormGame`: 2-player NormalFormGame instance.
- `init_pivot::Int=1`: Initial pivot, an integer `k` such that
  `1 <= k <= m+n`, where `m` and `n` are the numbers of actions of players 1
  and 2, respectively.
- `max_iter::Int=10^6`: Maximum number of pivoting steps.
- `capping=nothing`: If supplied, the routine is executed with the heuristics
  proposed by Codenotti et al. (2008): run the algorithm with each initial
  pivot for at most `capping` steps, then with the last initial pivot until
  convergence or `max_iter`. If `nothing`, set to `max_iter`, in which case
  the routine is equivalent to the standard Lemke-Howson algorithm.
- `full_output::Bool=false`: If `true`, also return a NamedTuple with
  information about the computation.

# Returns

- `NE::NTuple{2,Vector{Float64}}`: The mixed-action Nash equilibrium found.
- `res::NamedTuple` (only if `full_output=true`): fields `NE`, `converged`,
  `num_iter`, `max_iter`, `init`.
"""
function lemke_howson(g::NormalFormGame; init_pivot::Int=1, max_iter::Int=10^6,
                      capping::Union{Nothing,Int}=nothing, full_output::Bool=false)
    g.N == 2 || throw(ArgumentError("Implemented only for 2-player games"))

    payoff_matrices = (g.players[1].payoff_array, g.players[2].payoff_array)
    nums_actions = g.nums_actions
    total_num = sum(nums_actions)

    if !(1 <= init_pivot <= total_num)
        throw(ArgumentError(
            "`init_pivot` must be an integer k such that 1 <= k <= $total_num"
        ))
    end

    if capping === nothing
        capping = max_iter
    end

    tableaux = ntuple(i -> Matrix{Float64}(undef, nums_actions[3-i], total_num + 1), 2)
    bases = ntuple(i -> Vector{Int}(undef, nums_actions[3-i]), 2)

    converged, num_iter, init_pivot_used =
        _lemke_howson_capping(payoff_matrices, tableaux, bases, init_pivot,
                              max_iter, capping)
    NE = _get_mixed_actions(tableaux, bases)

    if !full_output
        return NE
    end

    res = (NE=NE, converged=converged, num_iter=num_iter,
           max_iter=max_iter, init=init_pivot_used)

    return NE, res
end

"""
    _lemke_howson_capping(payoff_matrices, tableaux, bases, init_pivot,
                          max_iter, capping)

Execute the Lemke-Howson algorithm with the heuristics proposed by
Codenotti et al. (2008). Return `(converged, num_iter, init_pivot_used)`.
"""
function _lemke_howson_capping(payoff_matrices, tableaux, bases, init_pivot::Int,
                               max_iter::Int, capping::Int)
    m, n = size(tableaux[2], 1), size(tableaux[1], 1)
    init_pivot_curr = init_pivot
    max_iter_curr = max_iter
    total_num_iter = 0

    for k in 1:(m+n-1)
        capping_curr = min(max_iter_curr, capping)

        _initialize_tableaux!(payoff_matrices, tableaux, bases)
        converged, num_iter =
            _lemke_howson_tbl!(tableaux, bases, init_pivot_curr, capping_curr)

        total_num_iter += num_iter

        if converged || total_num_iter >= max_iter
            return converged, total_num_iter, init_pivot_curr
        end

        init_pivot_curr += 1
        if init_pivot_curr > m + n
            init_pivot_curr -= m + n
        end
        max_iter_curr -= num_iter
    end

    _initialize_tableaux!(payoff_matrices, tableaux, bases)
    converged, num_iter =
        _lemke_howson_tbl!(tableaux, bases, init_pivot_curr, max_iter_curr)
    total_num_iter += num_iter

    return converged, total_num_iter, init_pivot_curr
end

"""
    _initialize_tableaux!(payoff_matrices, tableaux, bases)

Given a tuple of payoff matrices `(A, B)` of sizes `(m, n)` and `(n, m)`,
initialize the tableau and basis arrays in place.

If `payoff_matrices[i]` has a non-positive minimum, the payoff values stored
in the tableau are shifted by `abs(minimum) + 1`, so that the tableau has no
negative entry or column identically zero.

- `tableaux[1]` has `n` rows and `m+n+1` columns; columns `1:m` are the
  non-slack variables and `m+1:m+n` the slack variables.
- `tableaux[2]` has `m` rows and `m+n+1` columns; columns `1:m` are the
  slack variables and `m+1:m+n` the non-slack variables.
- In each tableau, the last column holds the values of the basic variables
  (initially 1).
- `bases[1]` and `bases[2]` hold the basic variable indices, initially
  `m+1:m+n` and `1:m`, respectively.
"""
function _initialize_tableaux!(payoff_matrices, tableaux::NTuple{2,AbstractMatrix{Float64}},
                               bases::NTuple{2,AbstractVector{Int}})
    nums_actions = size(payoff_matrices[1])
    m = nums_actions[1]

    consts = zeros(2)  # To be added to payoffs if min <= 0
    for pl in 1:2
        min_ = minimum(payoff_matrices[pl])
        if min_ <= 0
            consts[pl] = min_ * (-1) + 1
        end
    end

    py_starts = (0, m)
    sl_starts = (m, 0)
    for pl in 1:2
        py_start, sl_start = py_starts[pl], sl_starts[pl]
        for i in 1:nums_actions[3-pl]
            for j in 1:nums_actions[pl]
                tableaux[pl][i, py_start+j] = payoff_matrices[3-pl][i, j] + consts[3-pl]
            end
            for j in 1:nums_actions[3-pl]
                tableaux[pl][i, sl_start+j] = (j == i) ? 1.0 : 0.0
            end
            tableaux[pl][i, end] = 1.0
        end

        for i in 1:nums_actions[3-pl]
            bases[pl][i] = sl_start + i
        end
    end

    return tableaux, bases
end

"""
    _lemke_howson_tbl!(tableaux, bases, init_pivot, max_iter)

Main body of the Lemke-Howson algorithm: perform complementary pivoting,
modifying `tableaux` and `bases` in place.

`init_pivot` is a (1-based) variable index in `1:(m+n)`. Return
`(converged, num_iter)`.
"""
function _lemke_howson_tbl!(tableaux::NTuple{2,AbstractMatrix{Float64}},
                            bases::NTuple{2,AbstractVector{Int}},
                            init_pivot::Int, max_iter::Int)
    init_player = 1
    for k in bases[1]
        if k == init_pivot
            init_player = 2
            break
        end
    end
    pls = init_player == 1 ? (1, 2) : (2, 1)

    pivot = init_pivot

    m = size(tableaux[2], 1)
    n = size(tableaux[1], 1)
    slack_starts = (m + 1, 1)  # Slacks start at column m+1 in tableaux[1], 1 in tableaux[2]

    # Array to store row indices in _lex_min_ratio_test
    argmins = Vector{Int}(undef, max(m, n))

    converged = false
    num_iter = 0

    while true
        should_break_outer = false
        for pl in pls
            # Determine the leaving variable
            _, row_min = _lex_min_ratio_test(tableaux[pl], pivot, slack_starts[pl], argmins)

            # Pivoting step: modify tableau in place
            _pivoting!(tableaux[pl], pivot, row_min)

            # Update the basic variables and the pivot
            bases[pl][row_min], pivot = pivot, bases[pl][row_min]

            num_iter += 1

            if pivot == init_pivot
                converged = true
                should_break_outer = true
                break
            end
            if num_iter >= max_iter
                should_break_outer = true
                break
            end
        end
        should_break_outer && break
    end

    return converged, num_iter
end

"""
    _get_mixed_actions(tableaux, bases)

From `tableaux` and `bases`, extract the non-slack basic variables and
normalize them to obtain the mixed actions of the two players.
"""
function _get_mixed_actions(tableaux::NTuple{2,AbstractMatrix{Float64}},
                            bases::NTuple{2,AbstractVector{Int}})
    m, n = size(tableaux[2], 1), size(tableaux[1], 1)
    num = m + n
    out = zeros(Float64, num)
    ranges = (1:m, (m+1):(m+n))

    for pl in 1:2
        rng = ranges[pl]
        sum_ = 0.0
        for i in 1:size(tableaux[pl], 1)
            k = bases[pl][i]
            if k in rng
                out[k] = tableaux[pl][i, end]
                sum_ += tableaux[pl][i, end]
            end
        end
        if sum_ != 0
            out[rng] ./= sum_
        end
    end

    return out[1:m], out[(m+1):(m+n)]
end
