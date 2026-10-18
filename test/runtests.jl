using Test
using McLennanTourky

# Helpers to regenerate the random games and compare with Python
include(joinpath(@__DIR__, "python_comparison", "compare.jl"))

# Expected values marked "Python" were computed with QuantEcon.py 0.11.2.

@testset "normal_form_game" begin
    # von Stengel's example (lemke_howson.py docstring)
    A = [3.0 3.0; 2.0 5.0; 0.0 6.0]
    B = [3.0 2.0 3.0; 2.0 6.0 1.0]
    g = NormalFormGame((Player(A), Player(B)))

    @test g.N == 2
    @test g.nums_actions == (3, 2)
    @test payoff_vector(g.players[1], 2) == [3.0, 5.0, 6.0]
    @test payoff_vector(g.players[1], [0.5, 0.5]) == [3.0, 3.5, 3.0]
    @test best_response(g.players[1], [1.0, 0.0]) == 1
    @test best_response(g.players[1], [0.0, 1.0]) == 3
    # Ties are broken in favor of the smallest index
    @test best_response(g.players[1], [1/3, 2/3]) == 2
    @test is_best_response(g.players[1], 2, [1/3, 2/3])
    @test is_nash(g, (1, 1))
    @test is_nash(g, ([0.0, 1/3, 2/3], [1/3, 2/3]))
    @test !is_nash(g, (2, 1))
    @test pure2mixed(3, 2) == [0.0, 1.0, 0.0]

    # 3-player game: opponents ordered (i+1, ..., N, 1, ..., i-1)
    payoff_array = zeros(2, 2, 2)
    payoff_array[1, :, :] .= 1.0
    payoff_array[2, 1, 1] = 2.0
    p = Player(payoff_array)
    @test p.num_opponents == 2
    @test payoff_vector(p, (1, 1)) == [1.0, 2.0]
    @test payoff_vector(p, ([0.5, 0.5], [0.5, 0.5])) == [1.0, 0.5]
end

@testset "pivoting and lemke_howson internals" begin
    # Example from the docstrings of `_initialize_tableaux` and
    # `_lemke_howson_tbl` in lemke_howson.py (indices shifted by +1)
    A = [3.0 3.0; 2.0 5.0; 0.0 6.0]
    B = [3.0 2.0 3.0; 2.0 6.0 1.0]
    m, n = size(A)
    tableaux = (Matrix{Float64}(undef, n, m + n + 1), Matrix{Float64}(undef, m, m + n + 1))
    bases = (Vector{Int}(undef, n), Vector{Int}(undef, m))
    McLennanTourky._initialize_tableaux!((A, B), tableaux, bases)
    @test tableaux[1] == [3.0 2.0 3.0 1.0 0.0 1.0;
                          2.0 6.0 1.0 0.0 1.0 1.0]
    @test tableaux[2] == [1.0 0.0 0.0 4.0 4.0 1.0;
                          0.0 1.0 0.0 3.0 6.0 1.0;
                          0.0 0.0 1.0 1.0 7.0 1.0]
    @test bases == ([4, 5], [1, 2, 3])

    converged, num_iter = McLennanTourky._lemke_howson_tbl!(tableaux, bases, 2, 10)
    @test (converged, num_iter) == (true, 4)
    @test tableaux[1] ≈ [0.875 0.0 1.0 0.375 -0.125 0.25;
                         0.1875 1.0 0.0 -0.0625 0.1875 0.125]
    @test isapprox(tableaux[2], [1.0 -1.6 0.8 0.0 0.0 0.2;
                                 0.0 7/15 -0.4 1.0 0.0 1/15;
                                 0.0 -1/15 0.2 0.0 1.0 2/15]; atol=1e-12)
    @test bases == ([3, 2], [1, 4, 5])

    x, y = McLennanTourky._get_mixed_actions(tableaux, bases)
    @test x ≈ [0.0, 1/3, 2/3]
    @test y ≈ [1/3, 2/3]

    # Lexico-minimum ratio test breaks the tie in the RHS column
    tableau = [1.0 0.0 1.0 0.0 1.0;
               1.0 0.0 0.0 1.0 1.0]
    argmins = Vector{Int}(undef, 2)
    @test McLennanTourky._lex_min_ratio_test(tableau, 1, 3, argmins) == (true, 2)
end

@testset "lemke_howson" begin
    A = [3.0 3.0; 2.0 5.0; 0.0 6.0]
    B = [3.0 2.0 3.0; 2.0 6.0 1.0]
    g = NormalFormGame((Player(A), Player(B)))

    # Python: lemke_howson(g, init_pivot=k-1, full_output=True)
    expected = Dict(
        1 => (([1.0, 0.0, 0.0], [1.0, 0.0]), 2),
        2 => (([0.0, 1/3, 2/3], [1/3, 2/3]), 4),
        3 => (([1.0, 0.0, 0.0], [1.0, 0.0]), 3),
        4 => (([1.0, 0.0, 0.0], [1.0, 0.0]), 2),
        5 => (([0.0, 1/3, 2/3], [1/3, 2/3]), 4),
    )
    for (k, (NE_py, num_iter_py)) in expected
        NE, res = lemke_howson(g; init_pivot=k, full_output=true)
        @test NE[1] ≈ NE_py[1] atol=1e-15
        @test NE[2] ≈ NE_py[2] atol=1e-15
        @test res.converged
        @test res.num_iter == num_iter_py
        @test res.init == k
        @test is_nash(g, NE)
    end

    @test lemke_howson(g; init_pivot=2, capping=1) isa NTuple{2,Vector{Float64}}
    @test_throws ArgumentError lemke_howson(g; init_pivot=6)
    g3 = NormalFormGame(ntuple(_ -> Player(zeros(2, 2, 2)), 3))
    @test_throws ArgumentError lemke_howson(g3)
end

@testset "compute_fixed_point" begin
    T(x) = 4.0 * 0.3 * x * (1.0 - x)  # Fixed point (4μ - 1)/(4μ) = 1/6

    # Python: compute_fixed_point(T, 0.4, verbose=0)
    @test compute_fixed_point(T, 0.4; verbose=0) ≈ 0.17009417355252082 atol=1e-14
    # Python: compute_fixed_point(T, 0.4, verbose=0, method='imitation_game')
    @test compute_fixed_point(T, 0.4; verbose=0, method=:imitation_game) ≈
        0.17097894387608062 atol=1e-12
    # Python: compute_fixed_point(T, np.array([0.4]), error_tol=1e-6, verbose=0,
    #                             method='imitation_game')
    @test compute_fixed_point(x -> T.(x), [0.4]; error_tol=1e-6, verbose=0,
                              method=:imitation_game) ≈ [0.16667080320828861] atol=1e-12

    @test_throws ArgumentError compute_fixed_point(T, 0.4; max_iter=0)
    @test_throws ArgumentError compute_fixed_point(T, 0.4; verbose=3)
    @test_throws ArgumentError compute_fixed_point(T, 0.4; method=:foo)
end

# N-player anti-coordination game: action 1 ("safe") always pays 1; action 2
# ("risky") pays v if all the others play action 1, and 0 otherwise. The
# symmetric mixed equilibrium plays action 1 with probability 1/v^(1/(N-1)).
function anti_coordination_game(N, v)
    payoff_array = zeros(ntuple(_ -> 2, N))
    payoff_array[1, ntuple(_ -> Colon(), N - 1)...] .= 1.0
    payoff_array[2, ntuple(_ -> 1, N - 1)...] = v
    return NormalFormGame(ntuple(_ -> Player(payoff_array), N))
end

@testset "mclennan_tourky" begin
    @testset "3-player anti-coordination game" begin
        g = anti_coordination_game(3, 2.0)
        epsilon = 1e-5
        NE, res = mclennan_tourky(g; epsilon=epsilon, full_output=true)
        # Python: mclennan_tourky(g, epsilon=1e-5, full_output=True)
        p_py = 0.7071075439463927
        for action in NE
            @test action ≈ [p_py, 1 - p_py] atol=1e-12
        end
        @test res.converged
        @test res.num_iter == 18
        @test res.epsilon == epsilon && res.max_iter == 200 && res.init == (1, 1, 1)
        @test is_nash(g, NE; tol=epsilon)
        @test NE[1][1] ≈ 1 / sqrt(2) atol=1e-4

        # Python: mclennan_tourky(g, init=(1, 0, 1), epsilon=1e-5, full_output=True)
        NE, res = mclennan_tourky(g; init=(2, 1, 2), epsilon=epsilon, full_output=true)
        @test NE[1] ≈ [0.7071075439458264, 0.2928924560541736] atol=1e-12
        @test res.num_iter == 18

        # Mixed initial actions are accepted
        NE = mclennan_tourky(g; init=([0.5, 0.5], 1, 2), epsilon=epsilon)
        @test is_nash(g, NE; tol=epsilon)
    end

    @testset "2-player anti-coordination game" begin
        g = anti_coordination_game(2, 2.0)
        NE, res = mclennan_tourky(g; epsilon=1e-5, full_output=true)
        @test res.converged
        @test is_nash(g, NE; tol=1e-5)
    end

    @testset "2-player 3x2 game (von Stengel)" begin
        A = [3.0 3.0; 2.0 5.0; 0.0 6.0]
        B = [3.0 2.0 3.0; 2.0 6.0 1.0]
        g = NormalFormGame((Player(A), Player(B)))
        # Python: mclennan_tourky(g, epsilon=1e-6, full_output=True)
        NE, res = mclennan_tourky(g; epsilon=1e-6, full_output=true)
        @test NE == ([1.0, 0.0, 0.0], [1.0, 0.0])
        @test res.converged && res.num_iter == 1
        NE = mclennan_tourky(g; init=(2, 2), epsilon=1e-6)
        @test is_nash(g, NE; tol=1e-6)
    end

    @testset "N-player anti-coordination games" begin
        for N in 3:6
            g = anti_coordination_game(N, 2.0)
            NE, res = mclennan_tourky(g; epsilon=1e-6, max_iter=500, full_output=true)
            @test res.converged
            @test is_nash(g, NE; tol=1e-6)
        end
    end

    @testset "errors" begin
        g1 = NormalFormGame((Player([1.0, 2.0]),))
        @test_throws ArgumentError mclennan_tourky(g1)
        g = anti_coordination_game(3, 2.0)
        @test_throws ArgumentError mclennan_tourky(g; init=(1, 1))
    end
end

# A game type that is not a NormalFormGame but provides the interface used
# by mclennan_tourky (`N`, `nums_actions`, `players`, `best_response`,
# `is_nash`), with payoffs computed in closed form.
struct AntiCoordGame
    N::Int
    v::Float64
end

Base.getproperty(g::AntiCoordGame, s::Symbol) =
    s === :nums_actions ? ntuple(_ -> 2, getfield(g, :N)) :
    s === :players      ? ntuple(_ -> g, getfield(g, :N)) :
    getfield(g, s)

function _anti_coord_payoffs(g::AntiCoordGame, opponents_actions)
    prob_all_safe = g.N == 2 ? opponents_actions[1] :
                    prod(oa[1] for oa in opponents_actions)
    return [1.0, g.v * prob_all_safe]
end

McLennanTourky.best_response(g::AntiCoordGame, opponents_actions) =
    argmax(_anti_coord_payoffs(g, opponents_actions))

function McLennanTourky.is_nash(g::AntiCoordGame, action_profile; tol::Float64=1e-3)
    N = g.N
    for i in 1:N
        opponents_actions = N == 2 ? action_profile[3-i] :
                            ntuple(k -> action_profile[mod1(i + k, N)], N - 1)
        pv = _anti_coord_payoffs(g, opponents_actions)
        action_profile[i]' * pv < maximum(pv) - tol && return false
    end
    return true
end

@testset "duck-typed game" begin
    for N in (2, 3)
        g_duck = AntiCoordGame(N, 2.0)
        g = anti_coordination_game(N, 2.0)
        NE_duck, res_duck = mclennan_tourky(g_duck; epsilon=1e-5, full_output=true)
        NE, res = mclennan_tourky(g; epsilon=1e-5, full_output=true)
        @test res_duck.converged
        @test res_duck.num_iter == res.num_iter
        @test all(NE_duck[i] ≈ NE[i] for i in 1:N)
    end
end

# Comparison with QuantEcon.py on a subset of the 1000 random games
# (the full comparison is run by test/python_comparison/compare.jl)
const REFERENCE = joinpath(@__DIR__, "python_comparison", "results", "python_reference.tsv")
if isfile(REFERENCE)
    @testset "comparison with Python (games with N <= 5)" begin
        refs = filter(r -> r.N <= 5, read_reference(REFERENCE))
        results = [compare_one(ref) for ref in refs]
        # Rounding differences (~1e-16 per step) between Numba and Julia -- and
        # even between Julia builds, e.g. with `--check-bounds=yes` as used by
        # Pkg.test -- accumulate and can flip a nearly tied best response late
        # in long runs. The paths must agree exactly for the first 100
        # iterations, and wherever they agree throughout, so must the results.
        @test all(r -> r.path_match || r.first_divergence > 100, results)
        @test all(r -> !r.path_match || (r.converged_jl == r.converged_py &&
                                        r.num_iter_jl == r.num_iter_py), results)
        @test all(r -> !r.path_match || !r.converged_jl || r.match, results)
        @test all(r -> !r.converged_jl || r.is_nash_jl, results)
    end
else
    @info "Skipping the comparison with Python: $REFERENCE not found"
end
