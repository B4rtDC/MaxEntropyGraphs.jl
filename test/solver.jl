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
