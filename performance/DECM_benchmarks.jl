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

# Scale limiter: BENCH_MAX_SCALE=small|medium|large (default large). BENCH_QUICK=1 aliases small.
let scale = lowercase(get(ENV, "BENCH_QUICK", "0") == "1" ? "small" : get(ENV, "BENCH_MAX_SCALE", "large"))
    ncap = scale == "small" ? 1 : scale == "medium" ? 2 : length(name_graphs)
    ncap = min(ncap, length(name_graphs))
    ncap < length(name_graphs) && @info "BENCH_MAX_SCALE=$(scale): restricting DECM benchmarks to the first $(ncap) problem(s)."
    global name_graphs = name_graphs[1:ncap]
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
    for (name, G, _) in name_graphs
        println(f, readlines("$(name).py")[2][3:end])
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
