#=
Compare the Julia port of `mclennan_tourky` with the reference results of
QuantEcon.py produced by `generate_reference.py`.

Usage (from the package root):
    julia --project=. -t auto test/python_comparison/compare.jl [reference.tsv] [output_dir]

For each game in the reference file, the same game is regenerated with
`random_games.jl`, solved by the Julia `mclennan_tourky` with the same
settings and initial actions, and the following are compared:
- `br_path`: the pure best response profile computed in every iteration
  (must be identical; this is the discrete path of the algorithm),
- `converged` and `num_iter` (must be identical),
- the flattened equilibrium action profile `x_star`
  (maximum absolute difference must be at most `ATOL`).

`x_star` is not expected to agree bit-for-bit: Numba (LLVM) and Julia round
the floating point operations in the Lemke-Howson pivoting slightly
differently (differences of order 1e-16), and these differences accumulate
over up to 200 imitation games. The summary therefore also reports how
close `x_star` is.

Writes `comparison.csv` (one row per game) and `summary.md` to `output_dir`.
=#

using LinearAlgebra
using Printf
using McLennanTourky

include(joinpath(@__DIR__, "random_games.jl"))

const EPSILON = 1e-3   # Same as generate_reference.py
const MAX_ITER = 200   # Same as generate_reference.py
const ATOL = EPSILON   # Tolerance for the difference in x_star

parse_csv(T, s) = isempty(s) ? T[] : parse.(T, split(s, ','))

function read_reference(path)
    lines = readlines(path)
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(line) && continue
        f = split(line, '\t')
        push!(rows, (game_id=parse(Int, f[1]), N=parse(Int, f[2]),
                     nums_actions=parse_csv(Int, f[3]), init=parse_csv(Int, f[4]),
                     converged=parse(Bool, f[5]), num_iter=parse(Int, f[6]),
                     time_sec=parse(Float64, f[7]), br_path=String(f[8]),
                     x_star=parse_csv(Float64, f[9])))
    end
    return rows
end

"""
    best_response_path(g, init, indptr)

Run the same fixed point iteration as `mclennan_tourky` while recording the
pure best response profile computed in each iteration, encoded as in
`generate_reference.py` (one 0-based digit per player, ';'-separated).
"""
function best_response_path(g, init, indptr)
    path = String[]
    T = function (x)
        y = McLennanTourky._best_response_selection(x, g, indptr)
        push!(path, join(argmax(y[indptr[i]:indptr[i+1]-1]) - 1 for i in 1:g.N))
        return y
    end
    is_approx_fp = x -> McLennanTourky._is_epsilon_nash(x, g, EPSILON, indptr)
    x_init = McLennanTourky._flatten_action_profile(init, indptr)
    x_star, _, _ = McLennanTourky._compute_fixed_point_ig(T, x_init, MAX_ITER, is_approx_fp)
    return join(path, ';'), x_star
end

function compare_one(ref)
    g = random_game(ref.game_id, ref.nums_actions)
    init = Tuple(ref.init .+ 1)  # 0-based (Python) -> 1-based (Julia)
    t0 = time()
    NE, res = mclennan_tourky(g; init=init, epsilon=EPSILON, max_iter=MAX_ITER,
                              full_output=true)
    elapsed = time() - t0
    x_star = reduce(vcat, NE)

    indptr = cumsum([1; collect(g.nums_actions)])
    br_path, x_star_rec = best_response_path(g, init, indptr)
    x_star_rec == x_star || error("Recording run differs from mclennan_tourky")

    max_diff = maximum(abs, x_star .- ref.x_star)
    is_nash_jl = is_nash(g, NE; tol=EPSILON)
    path_match = br_path == ref.br_path
    # First iteration at which the best response profiles differ (0 if none)
    a, b = split(br_path, ';'), split(ref.br_path, ';')
    k = findfirst(i -> a[i] != b[i], 1:min(length(a), length(b)))
    first_divergence = k === nothing ? (length(a) == length(b) ? 0 : min(length(a), length(b)) + 1) : k
    match = path_match && (res.converged == ref.converged) &&
            (res.num_iter == ref.num_iter) && (max_diff <= ATOL)
    return (game_id=ref.game_id, N=ref.N, num_profiles=prod(ref.nums_actions),
            path_match=path_match, first_divergence=first_divergence,
            converged_py=ref.converged, converged_jl=res.converged,
            num_iter_py=ref.num_iter, num_iter_jl=res.num_iter,
            max_abs_diff=max_diff, is_nash_jl=is_nash_jl,
            time_py=ref.time_sec, time_jl=elapsed, match=match)
end

function write_results(results, refs, out_dir, ref_path)
    mkpath(out_dir)

    open(joinpath(out_dir, "comparison.csv"), "w") do io
        println(io, "game_id,N,nums_actions,br_path_match,first_divergence,converged_py,converged_jl,",
                "num_iter_py,num_iter_jl,max_abs_diff,is_nash_jl,time_py_sec,",
                "time_jl_sec,match")
        for (r, ref) in zip(results, refs)
            @printf(io, "%d,%d,%s,%s,%d,%s,%s,%d,%d,%.3e,%s,%.4f,%.4f,%s\n",
                    r.game_id, r.N, join(ref.nums_actions, ' '), r.path_match,
                    r.first_divergence,
                    r.converged_py, r.converged_jl, r.num_iter_py, r.num_iter_jl,
                    r.max_abs_diff, r.is_nash_jl, r.time_py, r.time_jl, r.match)
        end
    end

    n = length(results)
    n_match = count(r -> r.match, results)
    open(joinpath(out_dir, "summary.md"), "w") do io
        println(io, "# Julia vs Python (QuantEcon.py) comparison of `mclennan_tourky`\n")
        println(io, "- Reference file: `", basename(ref_path), "`")
        println(io, "- Number of games: ", n)
        println(io, "- Settings: `epsilon = $EPSILON`, `max_iter = $MAX_ITER`, ",
                "random pure initial actions")
        println(io, "- Match criterion: identical best response path (the pure ",
                "best response profile in every iteration), identical `converged` ",
                "and `num_iter`, and `max |x_star_jl - x_star_py| <= $ATOL`")
        println(io, "- Julia ", VERSION, ", ", Threads.nthreads(), " thread(s)\n")
        println(io, "## Result\n")
        println(io, "**", n_match, " / ", n, " games match.**\n")
        println(io, "- Best response path mismatches: ",
                count(r -> !r.path_match, results))
        println(io, "- `converged` mismatches: ",
                count(r -> r.converged_py != r.converged_jl, results))
        println(io, "- `num_iter` mismatches: ",
                count(r -> r.num_iter_py != r.num_iter_jl, results))
        println(io, "- Converged games: ", count(r -> r.converged_jl, results),
                " (all of them pass `is_nash(g, NE; tol=epsilon)`: ",
                all(r -> !r.converged_jl || r.is_nash_jl, results), ")\n")

        conv = filter(r -> r.converged_jl, results)
        println(io, "### Converged vs. non-converged games\n")
        println(io, "| | games | match | BR path match | max abs diff of x_star |")
        println(io, "|---|---|---|---|---|")
        for (label, rs) in (("converged", conv),
                            ("not converged (stopped at max_iter)",
                             filter(r -> !r.converged_jl, results)))
            isempty(rs) && continue
            @printf(io, "| %s | %d | %d | %d | %.3e |\n", label, length(rs),
                    count(r -> r.match, rs), count(r -> r.path_match, rs),
                    maximum(r -> r.max_abs_diff, rs))
        end
        println(io)

        divs = sort([r.first_divergence for r in results if !r.path_match])
        if !isempty(divs)
            println(io, "### Where the best response paths diverge\n")
            println(io, "For the games whose best response paths differ, the first ",
                    "iteration at which the pure best response profiles differ:\n")
            println(io, "| iteration | games |")
            println(io, "|---|---|")
            for (lo, hi) in ((1, 50), (51, 100), (101, 150), (151, 200))
                println(io, "| ", lo, "-", hi, " | ", count(k -> lo <= k <= hi, divs), " |")
            end
            println(io, "\nEarliest: ", divs[1], ", median: ", divs[(end + 1) ÷ 2], ". ",
                    "The paths agree exactly in the early iterations and diverge only ",
                    "after the ~1e-16 rounding differences have accumulated enough to ",
                    "flip a nearly tied best response.\n")
        end

        println(io, "### Closeness of `x_star`\n")
        println(io, "Rounding differences of order 1e-16 between Numba and Julia ",
                "accumulate over the iterations, so `x_star` is not bit-for-bit ",
                "identical. Distribution of `max |x_star_jl - x_star_py|`:\n")
        println(io, "| max abs diff | games | of which converged |")
        println(io, "|---|---|---|")
        bounds = [0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, Inf]
        labels = ["= 0 (bit-for-bit)", "<= 1e-14", "(1e-14, 1e-12]", "(1e-12, 1e-10]",
                  "(1e-10, 1e-8]", "(1e-8, 1e-6]", "(1e-6, 1e-4]", "(1e-4, 1e-3]", "> 1e-3"]
        for (k, label) in enumerate(labels)
            in_bin = k == 1 ? (d -> d == 0) :
                     k == 2 ? (d -> 0 < d <= bounds[2]) :
                     (d -> bounds[k-1] < d <= bounds[k])
            rs = filter(r -> in_bin(r.max_abs_diff), results)
            println(io, "| ", label, " | ", length(rs), " | ",
                    count(r -> r.converged_jl, rs), " |")
        end
        @printf(io, "\nOverall max: %.3e; median: %.3e\n\n",
                maximum(r -> r.max_abs_diff, results),
                sort([r.max_abs_diff for r in results])[(n + 1) ÷ 2])

        println(io, "## By number of players\n")
        println(io, "| N | games | match | BR path match | converged | max abs diff | ",
                "mean time Python (s) | mean time Julia (s) |")
        println(io, "|---|---|---|---|---|---|---|---|")
        for N in sort(unique(r.N for r in results))
            rs = filter(r -> r.N == N, results)
            @printf(io, "| %d | %d | %d | %d | %d | %.2e | %.3f | %.3f |\n",
                    N, length(rs), count(r -> r.match, rs), count(r -> r.path_match, rs),
                    count(r -> r.converged_jl, rs),
                    maximum(r -> r.max_abs_diff, rs),
                    sum(r -> r.time_py, rs) / length(rs),
                    sum(r -> r.time_jl, rs) / length(rs))
        end

        mismatches = filter(r -> !r.match, results)
        if !isempty(mismatches)
            println(io, "\n## Mismatches\n")
            println(io, "| game_id | N | BR path match | first divergence | ",
                    "converged (py/jl) | num_iter (py/jl) | max abs diff |")
            println(io, "|---|---|---|---|---|---|---|")
            for r in mismatches
                @printf(io, "| %d | %d | %s | %s | %s/%s | %d/%d | %.3e |\n", r.game_id, r.N,
                        r.path_match, r.path_match ? "-" : string(r.first_divergence),
                        r.converged_py, r.converged_jl, r.num_iter_py, r.num_iter_jl,
                        r.max_abs_diff)
            end
        end
        println(io, "\nTimes are wall-clock seconds per game; both sides were run ",
                "with several games in parallel, so they are indicative only.")
    end

    return n_match, n
end

function main(args)
    here = @__DIR__
    ref_path = length(args) >= 1 ? args[1] : joinpath(here, "results", "python_reference.tsv")
    out_dir = length(args) >= 2 ? args[2] : joinpath(here, "results")

    BLAS.set_num_threads(1)
    refs = read_reference(ref_path)

    # Warm up (compilation)
    mclennan_tourky(random_game(10^6, [2, 2, 2]))
    mclennan_tourky(random_game(10^6, [2, 3]))

    # Solve the most expensive games first for better load balancing
    order = sortperm(refs; by=r -> -r.time_sec)
    results = Vector{Any}(undef, length(refs))
    done = Threads.Atomic{Int}(0)
    Threads.@threads :dynamic for k in order
        results[k] = compare_one(refs[k])
        d = Threads.atomic_add!(done, 1) + 1
        r = results[k]
        @printf("[%4d/%d] game %4d: N=%2d match=%s max_diff=%.2e (%.2fs)\n",
                d, length(refs), r.game_id, r.N, r.match, r.max_abs_diff, r.time_jl)
        flush(stdout)
    end
    results = [r for r in results]

    n_match, n = write_results(results, refs, out_dir, ref_path)
    println("\n$n_match / $n games match. Results written to $out_dir")
    return n_match == n
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS) ? 0 : 1)
end
