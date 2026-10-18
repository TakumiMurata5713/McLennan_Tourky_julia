#=
Pivoting routines on tableaux shared by the simplex method and the
Lemke-Howson algorithm.

Port of `quantecon/optimize/pivoting.py`. The logic is identical except for
indices (0-based -> 1-based).
=#

const TOL_PIV = 1e-10
const TOL_RATIO_DIFF = 1e-15

"""
    _pivoting!(tableau, pivot_col, pivot_row)

Perform a pivoting step on `tableau` at (`pivot_row`, `pivot_col`), modifying
`tableau` in place.
"""
function _pivoting!(tableau::AbstractMatrix{Float64}, pivot_col::Int, pivot_row::Int)
    nrows, ncols = size(tableau)

    pivot_elt = tableau[pivot_row, pivot_col]
    @inbounds for j in 1:ncols
        tableau[pivot_row, j] /= pivot_elt
    end

    @inbounds for i in 1:nrows
        i == pivot_row && continue
        multiplier = tableau[i, pivot_col]
        multiplier == 0 && continue
        for j in 1:ncols
            tableau[i, j] -= tableau[pivot_row, j] * multiplier
        end
    end

    return tableau
end

"""
    _min_ratio_test_no_tie_breaking!(tableau, pivot, test_col, argmins,
                                     num_candidates, tol_piv, tol_ratio_diff)

Among the rows `argmins[1:num_candidates]`, find the rows that minimize the
ratio `tableau[i, test_col] / tableau[i, pivot]` (without tie breaking).
Return the number of minimizing rows, which are stored at the beginning of
`argmins`.
"""
function _min_ratio_test_no_tie_breaking!(tableau::AbstractMatrix{Float64}, pivot::Int,
                                          test_col::Int, argmins::AbstractVector{Int},
                                          num_candidates::Int,
                                          tol_piv::Float64, tol_ratio_diff::Float64)
    ratio_min = Inf
    num_argmins = 0

    for k in 1:num_candidates
        i = argmins[k]
        if tableau[i, pivot] <= tol_piv  # Treated as nonpositive
            continue
        end
        ratio = tableau[i, test_col] / tableau[i, pivot]
        if ratio > ratio_min + tol_ratio_diff  # Ratio large for i
            continue
        elseif ratio < ratio_min - tol_ratio_diff  # Ratio smaller for i
            ratio_min = ratio
            num_argmins = 1
        else  # Ratio equal
            num_argmins += 1
        end
        argmins[num_argmins] = i
    end

    return num_argmins
end

"""
    _lex_min_ratio_test(tableau, pivot, slack_start, argmins;
                        tol_piv=TOL_PIV, tol_ratio_diff=TOL_RATIO_DIFF)

Perform the lexico-minimum ratio test, which selects the leaving row
uniquely even at degenerate basic solutions.

`slack_start` is the (1-based) column index at which the slack variables
start. Return `(found, row_min)`, where `found` is `false` if there is no
positive entry in the pivot column.
"""
function _lex_min_ratio_test(tableau::AbstractMatrix{Float64}, pivot::Int, slack_start::Int,
                             argmins::AbstractVector{Int};
                             tol_piv::Float64=TOL_PIV, tol_ratio_diff::Float64=TOL_RATIO_DIFF)
    nrows = size(tableau, 1)
    ncols = size(tableau, 2)
    num_candidates = nrows

    found = false

    # Initialize `argmins`
    for i in 1:nrows
        argmins[i] = i
    end

    # First compare the ratios with the right hand side (last column)
    num_argmins = _min_ratio_test_no_tie_breaking!(
        tableau, pivot, ncols, argmins, num_candidates, tol_piv, tol_ratio_diff
    )
    if num_argmins == 1
        found = true
    elseif num_argmins >= 2
        # Break remaining ties with the slack variable columns, in order
        for j in slack_start:(slack_start + nrows - 1)
            j == pivot && continue
            num_argmins = _min_ratio_test_no_tie_breaking!(
                tableau, pivot, j, argmins, num_argmins, tol_piv, tol_ratio_diff
            )
            if num_argmins == 1
                found = true
                break
            end
        end
    end

    return found, argmins[1]
end
