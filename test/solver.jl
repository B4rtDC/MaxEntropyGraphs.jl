using Test
using MaxEntropyGraphs
using Logging
using Random

# Regression tests for the optimiser solution interface and the ConvergenceError type.
# These guard against the two Optimization.jl-4 issues fixed during modernization:
#   * `OptimizationSolution` lost the `solve_time` field (now `sol.stats.time`), which the
#     `verbose=true` logging path interpolates;
#   * `ConvergenceError` must be constructible with a `nothing` retcode (the fixed-point path).
@testset "solver solution interface" begin
    G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
    for method in (:BFGS, :LBFGS, :Newton)
        m = UBCM(G)
        _, sol = solve_model!(m, method = method)
        @test sol.stats.time isa Real            # field the verbose @info relies on (Optimization 4)
        @test hasproperty(sol, :retcode)
        @test length(sol.u) == length(m.θᵣ)
        # The verbose path must not raise an error-level log record (it would if a solution field
        # were renamed again, since the failing @info interpolation is reported at :error level).
        @test_logs min_level = Logging.Error solve_model!(UBCM(G), method = method, verbose = true)
    end

    # ConvergenceError constructs and prints in both the typed and fixed-point (`nothing`) forms.
    e = MaxEntropyGraphs.ConvergenceError(:fixedpoint, nothing)
    @test e isa Exception
    @test occursin("did not converge", sprint(showerror, e))
end

# A seeded `rng` must make sampling fully reproducible. The batch `rand(m, n; rng)` pre-draws
# per-sample seeds, so it is reproducible AND independent of the thread schedule / thread count.
@testset "sampling reproducibility" begin
    # undirected (UBCM), directed (DBCM) and bipartite (BiCM) all support the `rng` keyword
    ubcm = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)); solve_model!(ubcm)
    dbcm = DBCM(MaxEntropyGraphs.maspalomas()); solve_model!(dbcm)
    bicm = BiCM(MaxEntropyGraphs.corporateclub()); solve_model!(bicm)
    for m in (ubcm, dbcm, bicm)
        # single sample: same seed -> identical graph
        @test rand(m; rng = Xoshiro(1)) == rand(m; rng = Xoshiro(1))
        # batch sample: same seed -> identical, different seed -> different
        @test rand(m, 8; rng = Xoshiro(1)) == rand(m, 8; rng = Xoshiro(1))
        @test rand(m, 8; rng = Xoshiro(1)) != rand(m, 8; rng = Xoshiro(2))
    end
end

# Low-precision (Float16/Float32) solving is kept but flagged: solve_model! warns that it is
# experimental and may not converge. The warning fires regardless of whether the solve ultimately
# converges (which is exactly why it exists), so the call is wrapped to swallow a ConvergenceError.
@testset "low-precision solve warning" begin
    G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
    @test_logs (:warn,) match_mode = :any try
        solve_model!(UBCM(G, precision = Float32))
    catch
    end
end

# `maxiters` is now forwarded to the gradient-based optimisers (it used to be silently ignored),
# and `g_tol` exposes Optim's gradient tolerance so the solve can stop before over-converging.
@testset "maxiters and g_tol kwargs" begin
    G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
    # A tiny iteration cap now actually takes effect -> BFGS cannot converge -> ConvergenceError.
    @test_throws MaxEntropyGraphs.ConvergenceError solve_model!(UBCM(G), method = :BFGS, maxiters = 1)
    # A looser gradient tolerance still yields a valid solution that respects the degree constraints.
    m = UBCM(G)
    solve_model!(m, method = :BFGS, analytical_gradient = true, g_tol = 1e-5)
    @test isapprox(vec(sum(MaxEntropyGraphs.Ĝ(m), dims = 2)), m.d, rtol = 1e-3)
end

# The weighted CReM/DCReM/CRWCM layers are solved in log-parameter space (`_logspace_fixedpoint`),
# which turns `ftol` into a *relative* strength tolerance. Solving in θ directly bounded
# |G_i - θ_i| = (θ_i/s_i)|⟨s_i⟩ - s_i| instead, whose conversion factor s_i/θ_i grows as the square of
# the weight scale: a solve reporting success at ftol=1e-8 could miss a strength by O(1) absolute units
# once the weights were rescaled (DCReM on rhesus: 5.3e-5 at c=1, but 0.45 at c=100 and 61.75 at c=1000).
@testset "log-space fixed point: ftol is a scale-invariant relative tolerance" begin
    A₀ = collect(MaxEntropyGraphs.SimpleWeightedGraphs.weights(MaxEntropyGraphs.rhesus_macaques()))
    build(M, c) = M === :CReM ?
                      CReM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph((A₀ .* c .+ (A₀ .* c)') ./ 2)) :
                  M === :DCReM ?
                      DCReM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(A₀ .* c)) :
                      CRWCM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(A₀ .* c))
    for M in (:CReM, :DCReM, :CRWCM)
        rel = map((1.0, 100.0, 1000.0)) do c
            m = build(M, c)
            solve_model!(m)                                     # default ftol
            MaxEntropyGraphs.constraint_residual(m, relative = true)
        end
        # the relative residual respects the default ftol whatever the weight scale ...
        @test all(<(MaxEntropyGraphs._DEFAULT_FTOL), rel)
        # ... and is scale-invariant (the log-space problem is invariant under s -> c*s, up to the
        # rounding introduced by forming the rescaled weights). Before the fix these drifted by ~c².
        @test rel[2] ≈ rel[1] rtol = 1e-4
        @test rel[3] ≈ rel[1] rtol = 1e-4
        # the absolute residual is in strength units, so it may only grow *linearly* with the scale
        m₁ = build(M, 1.0);    solve_model!(m₁)
        m₃ = build(M, 1000.0); solve_model!(m₃)
        @test MaxEntropyGraphs.constraint_residual(m₃) < 1e4 * MaxEntropyGraphs.constraint_residual(m₁)
    end
end

# `ftol` must still bite on the log-space path (a loose tolerance yields a worse relative residual).
@testset "log-space fixed point: ftol still controls the solve" begin
    G = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())
    m_loose = CReM(G); solve_model!(m_loose, ftol = 1e-2)
    m_tight = CReM(G); solve_model!(m_tight, ftol = 1e-12)
    @test MaxEntropyGraphs.constraint_residual(m_tight, relative = true) <
          MaxEntropyGraphs.constraint_residual(m_loose, relative = true)
    @test MaxEntropyGraphs.constraint_residual(m_tight, relative = true) < 1e-10
end

# Dead channels (zero strength). CRWCM admits them by construction (a node with no reciprocated edge)
# and masks them out of the log-space solve, pinning θ to its +Inf optimum. CReM/DCReM have no finite θ
# for a zero-strength node and keep failing loudly, exactly as they did before the log-space change.
@testset "log-space fixed point: dead channels" begin
    # CRWCM: rhesus has 9 dead channels out of 4n = 64; the live solve must be unaffected by them
    m = CRWCM(MaxEntropyGraphs.rhesus_macaques())
    @test count(iszero, vcat(m.s_out, m.s_in, m.s_rec_out, m.s_rec_in)) > 0
    solve_model!(m)
    dead = findall(iszero, vcat(m.s_out, m.s_in, m.s_rec_out, m.s_rec_in))
    @test all(isinf, m.θ[dead])                       # pinned at their analytical optimum
    @test all(isfinite, m.θ[setdiff(eachindex(m.θ), dead)])
    @test MaxEntropyGraphs.constraint_residual(m, relative = true) < MaxEntropyGraphs._DEFAULT_FTOL

    # CReM/DCReM: a zero-strength node is not solvable and must still throw rather than return garbage
    mc = @test_logs (:warn,) match_mode = :any CReM(d = [2, 2, 2, 2], s = [3.0, 5.0, 4.0, 0.0])
    @test_throws MaxEntropyGraphs.NLsolve.IsFiniteException solve_model!(mc)
end
