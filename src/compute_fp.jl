#=
Compute an approximate fixed point of a given operator T, starting from
specified initial condition v.

Port of `quantecon/_compute_fp.py`: `compute_fixed_point` (methods
`:iteration` and `:imitation_game`), `_compute_fixed_point_ig`,
`_initialize_tableaux_ig` and `_square_sum`.

Difference from the Python version (intentional simplification): to make the
code Numba-friendly, the Python version stores the sequences X, Y and the
tableaux in fixed-size buffers that are doubled when they run out. This is an
optimization that avoids reallocating NumPy arrays and is not part of the
algorithm itself. In Julia, `push!` on a `Vector` is fast enough, so the
sequences are stored in growable vectors here.
=#

function _print_after_skip(skip::Int, it=nothing, dist=nothing, etime=nothing)
    if it === nothing
        # Print initial header
        msg = rpad("Iteration", 13) * rpad("Distance", 15) * rpad("Elapsed (seconds)", 17)
        println(msg)
        println("-"^length(msg))
        return
    end

    if it % skip == 0
        if etime === nothing
            println("After $it iterations dist is $dist")
        else
            println(rpad(string(it), 13),
                    rpad(_sci3(dist), 15),
                    rpad(_sci3(etime), 18))
        end
    end
    return
end

# Format like Python's "%.3e"
function _sci3(x::Real)
    x == 0 && return "0.000e+00"
    e = floor(Int, log10(abs(x)))
    m = x / 10.0^e
    if round(abs(m), digits=3) >= 10
        m /= 10
        e += 1
    end
    return string(round(m, digits=3)) * (e < 0 ? "e-" : "e+") * lpad(string(abs(e)), 2, '0')
end

const _non_convergence_msg = "max_iter attained before convergence in compute_fixed_point"

_is_approx_fp(T, v, error_tol) = maximum(abs, T(v) .- v) <= error_tol

"""
    compute_fixed_point(T, v; error_tol=1e-3, max_iter=50, verbose=2,
                        print_skip=5, method=:iteration)

Compute and return an approximate fixed point of the function `T`.

The default method `:iteration` simply iterates the function given an initial
condition `v` and returns ``T^k v`` when the condition
``\\lVert T^k v - T^{k-1} v \\rVert \\leq`` `error_tol` is satisfied or the
number of iterations ``k`` reaches `max_iter`. Provided that `T` is a
contraction mapping or similar, ``T^k v`` will be an approximation to the
fixed point.

The method `:imitation_game` uses the "imitation game algorithm" developed by
McLennan and Tourky (2006), which internally constructs a sequence of
two-player games called imitation games and utilizes their Nash equilibria,
computed by the Lemke-Howson algorithm routine. It finds an approximate fixed
point of `T`, a point ``v^*`` such that
``\\lVert T(v^*) - v^* \\rVert \\leq`` `error_tol`, provided `T` is a function
that satisfies the assumptions of Brouwer's fixed point theorem, i.e., a
continuous function that maps a compact and convex set to itself.

# Arguments

- `T`: A callable object (e.g., function) that acts on `v`.
- `v`: An object such that `T(v)` is defined.
- `error_tol::Real=1e-3`: Error tolerance.
- `max_iter::Int=50`: Maximum number of iterations.
- `verbose::Int=2`: Level of feedback (0 for no output, 1 for warnings only,
  2 for warning and residual error reports during iteration).
- `print_skip::Int=5`: How many iterations to apply between print messages
  (effective only when `verbose=2`).
- `method::Symbol=:iteration`: `:iteration` or `:imitation_game`.

# Returns

- The approximate fixed point.

# Examples

```julia
julia> T(x, μ) = 4.0 * μ * x * (1.0 - x);

julia> x_star = compute_fixed_point(x -> T(x, 0.3), 0.4, verbose=0);  # (4μ - 1)/(4μ)
```

# References

- A. McLennan and R. Tourky, "From Imitation Games to Kakutani," 2006.
"""
function compute_fixed_point(T, v; error_tol::Real=1e-3, max_iter::Int=50,
                             verbose::Int=2, print_skip::Int=5,
                             method::Symbol=:iteration)
    max_iter < 1 && throw(ArgumentError("max_iter must be a positive integer"))
    verbose in (0, 1, 2) || throw(ArgumentError("verbose should be 0, 1 or 2"))
    method in (:iteration, :imitation_game) || throw(ArgumentError("invalid method"))

    if method == :imitation_game
        is_approx_fp = x -> _is_approx_fp(T, x, error_tol)
        v_star, converged, iterate =
            _compute_fixed_point_ig(T, v, max_iter, is_approx_fp;
                                    verbose=verbose, print_skip=print_skip)
        return v_star
    end

    # method == :iteration
    iterate = 0
    error = Inf
    start_time = time()
    if verbose == 2
        _print_after_skip(print_skip, nothing)
    end

    while true
        new_v = T(v)
        iterate += 1
        error = maximum(abs, new_v .- v)
        v = new_v
        if error <= error_tol || iterate >= max_iter
            break
        end

        if verbose == 2
            _print_after_skip(print_skip, iterate, error, time() - start_time)
        end
    end

    if verbose == 2
        _print_after_skip(1, iterate, error, time() - start_time)
    end
    if verbose >= 1
        if error > error_tol
            @warn _non_convergence_msg
        elseif verbose == 2
            println("Converged in $iterate steps")
        end
    end

    return v
end

"""
    _compute_fixed_point_ig(T, v, max_iter, is_approx_fp;
                            verbose=0, print_skip=1, max_piv=10^6)

Implement the imitation game algorithm by McLennan and Tourky (2006) for
computing an approximate fixed point of `T`.

- `T`            : Function that maps `v` to an object of the same shape (in
                   `mclennan_tourky`, a best response selection).
- `v`            : Initial point (a scalar or an array).
- `max_iter`     : Maximum number of outer iterations (number of imitation
                   games solved).
- `is_approx_fp` : Function `is_approx_fp(v)` that determines whether `v` is
                   an approximate fixed point.
- `max_piv`      : Maximum number of pivoting steps in each call of the
                   Lemke-Howson routine.

Return `(x_star, converged, num_iter)`.
"""
function _compute_fixed_point_ig(T, v, max_iter::Int,
                                 is_approx_fp; verbose::Int=0, print_skip::Int=1,
                                 max_piv::Int=10^6)
    start_time = time()
    if verbose == 2
        _print_after_skip(print_skip, nothing)
    end

    x_new = copy(v)
    y_new = T(x_new)
    iterate = 1
    converged = is_approx_fp(x_new)

    if converged || iterate >= max_iter
        if verbose == 2
            error = maximum(abs, y_new .- x_new)
            _print_after_skip(1, iterate, error, time() - start_time)
        end
        if verbose >= 1
            if !converged
                @warn _non_convergence_msg
            elseif verbose == 2
                println("Converged in $iterate steps")
            end
        end
        return x_new, converged, iterate
    end

    if verbose == 2
        error = maximum(abs, y_new .- x_new)
        _print_after_skip(print_skip, iterate, error, time() - start_time)
    end

    # `v` may be a scalar or an array
    X = [copy(x_new)]
    Y = [copy(y_new)]
    x_new = copy(Y[1])

    while true
        y_new = T(x_new)
        iterate += 1
        converged = is_approx_fp(x_new)

        if converged || iterate >= max_iter
            break
        end

        if verbose == 2
            error = maximum(abs, y_new .- x_new)
            _print_after_skip(print_skip, iterate, error, time() - start_time)
        end

        push!(X, copy(x_new))
        push!(Y, copy(y_new))

        m = iterate
        tableaux = (zeros(Float64, m, 2m + 1), zeros(Float64, m, 2m + 1))
        bases = (Vector{Int}(undef, m), Vector{Int}(undef, m))

        _initialize_tableaux_ig!(X, Y, tableaux, bases)
        # Use the most recently added point (variable index m) as the initial pivot
        _lemke_howson_tbl!(tableaux, bases, m, max_piv)
        _, rho = _get_mixed_actions(tableaux, bases)

        x_new = sum(rho[k] .* Y[k] for k in 1:m)
    end

    if verbose == 2
        error = maximum(abs, y_new .- x_new)
        _print_after_skip(1, iterate, error, time() - start_time)
    end
    if verbose >= 1
        if !converged
            @warn _non_convergence_msg
        elseif verbose == 2
            println("Converged in $iterate steps")
        end
    end

    return x_new, converged, iterate
end

"""
    _initialize_tableaux_ig!(X, Y, tableaux, bases)

Given sequences `X = [X_1, ..., X_m]` and `Y = [Y_1, ..., Y_m]` of vectors of
the same length, initialize the tableau and basis arrays in place for the
"geometric" imitation game as defined in McLennan and Tourky (2006), to be
passed to `_lemke_howson_tbl!`.

- Mover (player 1, `tableaux[1]`): payoff 1 if their choice `i` equals the
  Imitator's choice `j`, 0 otherwise (i.e., the identity payoff matrix).
- Imitator (player 2, `tableaux[2]`): payoff `-‖X_i - Y_j‖²` when the Mover
  plays `i` and the Imitator plays `j`, shifted by a constant so that the
  tableau is positive.

`tableaux` (each `m × (2m+1)`) and `bases` (each of length `m`) must be
preallocated and are modified in place.
"""
function _initialize_tableaux_ig!(X::AbstractVector, Y::AbstractVector,
                                  tableaux::NTuple{2,AbstractMatrix{Float64}},
                                  bases::NTuple{2,AbstractVector{Int}})
    m = length(X)
    min_ = zeros(Float64, m)

    # Mover
    for i in 1:m
        for j in 1:(2m)
            tableaux[1][i, j] = (j == i || j == i + m) ? 1.0 : 0.0
        end
        # Right hand side
        tableaux[1][i, end] = 1.0
    end

    # Imitator
    for i in 1:m
        # Slack variables
        for j in 1:m
            tableaux[2][i, j] = (j == i) ? 1.0 : 0.0
        end
        # Payoff variables
        for j in 1:m
            d = X[i] .- Y[j]
            val = -_square_sum(d)
            tableaux[2][i, m+j] = val
            if val < min_[j]
                min_[j] = val
            end
        end
        # Right hand side
        tableaux[2][i, end] = 1.0
    end
    # Shift the payoff values
    for i in 1:m, j in 1:m
        tableaux[2][i, m+j] -= min_[j]
        tableaux[2][i, m+j] += 1.0
    end

    starts = (m, 0)
    for pl in 1:2, i in 1:m
        bases[pl][i] = starts[pl] + i
    end

    return tableaux, bases
end

_square_sum(a::Real) = a^2
_square_sum(a::AbstractArray{<:Real}) = sum(abs2, a)
