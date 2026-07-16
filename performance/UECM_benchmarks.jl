##############################################################################################
#   _   _ _____ ____ __  __     _                     _                          _
#  | | | | ____/ ___|  \/  |   | |__   ___ _ __   ___| |__  _ __ ___   __ _ _ __| | _____
#  | | | |  _|| |   | |\/| |   | '_ \ / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ / __|
#  | |_| | |__| |___| |  | |   | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <\__ \
#   \___/|_____\____|_|  |_|   |_.__/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\___/
#
#  Undirected Enhanced Configuration Model (UECM): construction, solving and sampling of a weighted,
#  undirected network with a fixed degree AND (integer) strength sequence. NEMtropy solves the same
#  model through its `UndirectedGraph` with model name `ecm_exp`; see accuracy_comparison.jl for the
#  cross-package correctness check.
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
$(now()) - Setting up UECM benchmarks on $(Threads.nthreads()) threads (BLAS: $(LinearAlgebra.BLAS.get_num_threads())).
"""

## Load up helper functions
include(joinpath(@__DIR__, "benchmark_helpers.jl"))

## Reference graphs setup
## ----------------------
## `rhesus_macaques` (symmetrised) is a real integer-weighted network (the validated anchor). Random
## weighted Erdős–Rényi graphs put the UECM near its feasibility boundary (yᵢyⱼ → 1) and are numerically
## ill-conditioned, so the larger problems are built by tiling the real network block-diagonally. This
## keeps every instance realizable/well-conditioned and highlights the reduced-model acceleration (the
## number of distinct {degree, strength} pairs stays constant while N grows).
function tiled_rhesus(k::Int)
    base = MaxEntropyGraphs.rhesus_macaques(); n = Graphs.nv(base)
    sources = Int[]; targets = Int[]; weights = Float64[]
    for c in 0:k-1, e in Graphs.edges(base)
        push!(sources, Graphs.src(e) + c*n); push!(targets, Graphs.dst(e) + c*n); push!(weights, e.weight)
    end
    return SWG.SimpleWeightedGraph(SWG.SimpleWeightedDiGraph(sources, targets, weights)) # symmetrise
end

name_graphs = [("UECM_small",  SWG.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques()), Dict(:include_fixed_point => false, :include_BFGS => true, :include_newton => true)),
               ("UECM_medium", tiled_rhesus(8),                                         Dict(:include_fixed_point => false, :include_BFGS => true, :include_newton => true)),
               ("UECM_large",  tiled_rhesus(32),                                        Dict(:include_fixed_point => false, :include_BFGS => true, :include_newton => false))]

# Scale limiter: BENCH_MAX_SCALE=small|medium|large (default large) caps the problem size, and
# BENCH_MIN_SCALE (default small) skips the smaller problems, so a run can target only what is
# missing (e.g. BENCH_MIN_SCALE=large BENCH_MAX_SCALE=large). BENCH_QUICK=1 aliases small.
let scale = lowercase(get(ENV, "BENCH_QUICK", "0") == "1" ? "small" : get(ENV, "BENCH_MAX_SCALE", "large")),
    minscale = lowercase(get(ENV, "BENCH_MIN_SCALE", "small"))

    ncap = scale == "small" ? 1 : scale == "medium" ? 2 : length(name_graphs)
    ncap = min(ncap, length(name_graphs))
    nfloor = minscale == "medium" ? 2 : minscale == "large" ? 3 : 1
    nfloor = min(nfloor, ncap) # a floor above the cap degrades to the cap, never to an empty run
    (nfloor > 1 || ncap < length(name_graphs)) && @info "BENCH_MIN_SCALE=$(minscale), BENCH_MAX_SCALE=$(scale): restricting UECM benchmarks to problem(s) $(nfloor):$(ncap)."
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
    @info """$(now()) - benchmark "$(name)" will test G($(nv(G)), $(ne(G))) [weighted, undirected]"""
end

## Generate python scripts and the associated shell script
for (name, G, _) in name_graphs
    generate_UECM_python(name)
end
open(joinpath(@__DIR__, "UECM_script.sh"), "w") do f
    println(f, "#!/bin/bash")
    println(f, "source \"$(joinpath(@__DIR__, ".venv", "bin", "activate"))\"")
    for (name, G, _) in name_graphs
        # Wrap every pytest job in the process-group watchdog; BENCH_JOB_TIMEOUT=0 (the
        # default) runs it untouched, so this changes nothing unless a budget is set.
        println(f, "\"$(joinpath(@__DIR__, "run_with_timeout.sh"))\" \"\${BENCH_JOB_TIMEOUT:-0}\" " * readlines("$(name).py")[2][3:end])
    end
end

@info "$(now()) - Reference graphs and python scripts written."


## Start benchmarking Julia
## ------------------------
for (name, G, kwargs) in name_graphs
    @info "$(now()) - Started benchmarking $(name)."

    results = Dict(
        "system_info" => get_system_info(),
        "benchmarks" => [test_create_UECM(G);
                         test_solve_UECM(G; kwargs...);
                         test_sample_UECM(G, n);
                         ]
        )
    open(joinpath(outpath, "$(Dates.format(now(), "YYYY_mm_dd_HH_MM"))_$(name).json"),"w") do f
        write(f, JSON.json(results))
    end

    @info "$(now()) - Benchmarking for $(name) done."
end

@info "$(now()) - All Julia UECM benchmarks done."
