##############################################################################################
#   ____  _____ ____ __  __     _                     _                          _
#  |  _ \| ____/ ___|  \/  |   | |__   ___ _ __   ___| |__  _ __ ___   __ _ _ __| | _____
#  | | | |  _|| |   | |\/| |   | '_ \ / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ / __|
#  | |_| | |__| |___| |  | |   | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <\__ \
#  |____/|_____\____|_|  |_|   |_.__/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\___/
#
#  Directed Enhanced Configuration Model (DECM): construction, solving and sampling of a weighted,
#  directed network with fixed out/in-degree AND (integer) out/in-strength sequences. NEMtropy solves
#  the same model through its `DirectedGraph` with model name `decm_exp`; see accuracy_comparison.jl
#  for the cross-package correctness check.
##############################################################################################

## Setup
cd(@__DIR__)
using Pkg
Pkg.activate(pwd())
using MaxEntropyGraphs
using Graphs
using BenchmarkTools
using JSON
using Dates
import LinearAlgebra

# SimpleWeightedGraphs is re-exported by MaxEntropyGraphs (not a direct dependency of this env)
const SWG = MaxEntropyGraphs.SimpleWeightedGraphs

# Cap BLAS to the core budget so the (single-threaded) creation/solve benchmarks are a fair
# same-core comparison against the thread-limited Python side (see benchmarks.sh).
LinearAlgebra.BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_CORES", string(Threads.nthreads()))))

@info """
$(now()) - Setting up DECM benchmarks on $(Threads.nthreads()) threads (BLAS: $(LinearAlgebra.BLAS.get_num_threads())).
"""

## Load up helper functions
include(joinpath(@__DIR__, "benchmark_helpers.jl"))

## Reference graphs setup
## ----------------------
## `rhesus_macaques` (used unsymmetrised) is a real integer-weighted directed network (the validated
## anchor). Random weighted graphs put the DECM near its feasibility boundary (yᵢ_out·yⱼ_in → 1) and are
## numerically ill-conditioned, so the larger problems are built by tiling the real network
## block-diagonally. This keeps every instance realizable/well-conditioned and highlights the
## reduced-model acceleration (the number of distinct constraint quadruples stays constant while N grows).
function tiled_rhesus_directed(k::Int)
    base = MaxEntropyGraphs.rhesus_macaques(); n = Graphs.nv(base)
    sources = Int[]; targets = Int[]; weights = Float64[]
    for c in 0:k-1, e in Graphs.edges(base)
        push!(sources, Graphs.src(e) + c*n); push!(targets, Graphs.dst(e) + c*n); push!(weights, e.weight)
    end
    return SWG.SimpleWeightedDiGraph(sources, targets, weights)
end

name_graphs = [("DECM_small",  MaxEntropyGraphs.rhesus_macaques(), Dict(:include_fixed_point => false, :include_BFGS => true, :include_newton => true)),
               ("DECM_medium", tiled_rhesus_directed(8),           Dict(:include_fixed_point => false, :include_BFGS => true, :include_newton => true)),
               ("DECM_large",  tiled_rhesus_directed(32),          Dict(:include_fixed_point => false, :include_BFGS => true, :include_newton => false))]

# Scale limiter: BENCH_MAX_SCALE=small|medium|large (default large) caps the problem size, and
# BENCH_MIN_SCALE (default small) skips the smaller problems, so a run can target only what is
# missing (e.g. BENCH_MIN_SCALE=large BENCH_MAX_SCALE=large). BENCH_QUICK=1 aliases small.
let scale = lowercase(get(ENV, "BENCH_QUICK", "0") == "1" ? "small" : get(ENV, "BENCH_MAX_SCALE", "large")),
    minscale = lowercase(get(ENV, "BENCH_MIN_SCALE", "small"))

    ncap = scale == "small" ? 1 : scale == "medium" ? 2 : length(name_graphs)
    ncap = min(ncap, length(name_graphs))
    nfloor = minscale == "medium" ? 2 : minscale == "large" ? 3 : 1
    nfloor = min(nfloor, ncap) # a floor above the cap degrades to the cap, never to an empty run
    (nfloor > 1 || ncap < length(name_graphs)) && @info "BENCH_MIN_SCALE=$(minscale), BENCH_MAX_SCALE=$(scale): restricting DECM benchmarks to problem(s) $(nfloor):$(ncap)."
    global name_graphs = name_graphs[nfloor:ncap]
end

# Write out the weighted edgelists (source, target, weight) for the reference graphs so Python uses the
# same networks. NEMtropy auto-detects the weighted (3-tuple) edge list.
for (name, G, _) in name_graphs
    open(joinpath(@__DIR__, "data", "$(name).csv"), "w") do f
        for e in edges(G)
            write(f, "$(e.src), $(e.dst), $(Int(e.weight))\n")
        end
    end
    @info """$(now()) - benchmark "$(name)" will test G($(nv(G)), $(ne(G))) [weighted, directed]"""
end

## Generate python scripts and the associated shell script
for (name, G, _) in name_graphs
    generate_DECM_python(name)
end
open(joinpath(@__DIR__, "DECM_script.sh"), "w") do f
    println(f, "#!/bin/bash")
    println(f, "source \"$(joinpath(@__DIR__, ".venv", "bin", "activate"))\"")
    # Every pytest job runs under the process-group watchdog; BENCH_JOB_TIMEOUT=0 (the
    # default) runs it untouched, so this changes nothing unless a budget is set.
    watchdog = "\"$(joinpath(@__DIR__, "run_with_timeout.sh"))\" \"\${BENCH_JOB_TIMEOUT:-0}\" "
    for (name, G, _) in name_graphs
        cmd = readlines("$(name).py")[2][3:end]
        if name == "DECM_large"
            # NEMtropy's decm_exp newton at N=512 (2048 parameters, dense per-iteration Hessian)
            # has never been measured and could dwarf everything else here, while the Julia side
            # deliberately drops Newton at this scale. Each test therefore runs as its own pytest
            # process (riskiest last), so the watchdog can kill a slow newton without losing
            # create/quasinewton, and the parts are merged into one result file afterwards (the
            # plotting scripts read only the newest file per scale).
            for nodeid in ("DECM_large.py::test_create_DECM",
                           "DECM_large.py::test_solve_DECM[decm_exp-quasinewton-strengths]",
                           "DECM_large.py::test_solve_DECM[decm_exp-newton-strengths]")
                println(f, watchdog * replace(cmd, "pytest DECM_large.py" => "pytest '$(nodeid)'"))
            end
            println(f, "python \"$(joinpath(@__DIR__, "merge_pytest_benchmarks.py"))\" \"$(joinpath(@__DIR__, "benchmarks"))\" DECM_large")
        else
            println(f, watchdog * cmd)
        end
    end
end

@info "$(now()) - Reference graphs and python scripts written."


## Start benchmarking Julia
## ------------------------
for (name, G, kwargs) in name_graphs
    @info "$(now()) - Started benchmarking $(name)."

    results = Dict(
        "system_info" => get_system_info(),
        "benchmarks" => [test_create_DECM(G);
                         test_solve_DECM(G; kwargs...);
                         test_sample_DECM(G, n);
                         ]
        )
    open(joinpath(outpath, "$(Dates.format(now(), "YYYY_mm_dd_HH_MM"))_$(name).json"),"w") do f
        write(f, JSON.json(results))
    end

    @info "$(now()) - Benchmarking for $(name) done."
end

@info "$(now()) - All Julia DECM benchmarks done."
