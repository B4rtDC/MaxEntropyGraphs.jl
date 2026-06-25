using Aqua
using MaxEntropyGraphs

# Aqua.jl quality assurance: catches stale/missing deps (would have flagged the old Revise/Dates
# deps), method piracy, unbound type parameters (flagged the old ConvergenceError), undefined
# exports, and missing [compat] entries.
@testset "Aqua quality assurance" begin
    # Ambiguities are checked separately and disabled here: the bulk of method ambiguities in the
    # closure of this package come from the heavy upstream dependency tree (Optimization/SciMLBase),
    # not from MaxEntropyGraphs itself, and are outside our control.
    Aqua.test_all(MaxEntropyGraphs; ambiguities = false)
end
