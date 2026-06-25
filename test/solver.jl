using Test
using MaxEntropyGraphs
using Logging

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
