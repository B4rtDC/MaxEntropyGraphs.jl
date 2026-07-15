##############################################################################################
#    ____ ______        ______ __  __   _                     _                          _
#   / ___|  _ \ \      / / ___|  \/  | | |__   ___ _ __   ___| |__  _ __ ___   __ _ _ __| | _____
#  | |   | |_) \ \ /\ / / |   | |\/| | | '_ \ / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ / __|
#  | |___|  _ < \ V  V /| |___| |  | | | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <\__ \
#   \____|_| \_\ \_/\_/  \____|_|  |_| |_.__/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\___/
#
#  Conditionally Reciprocal Weighted Configuration Model (CRWCM): construction, solving, exact
#  expected triadic fluxes and sampling of a weighted, directed network with the four reciprocal
#  strength sequences fixed conditional on an RBCM topology. The reference Python implementation
#  is NuMeTriS (model 'RBCM+CRWCM', which solves both layers, matching the Julia two-step solve);
#  see accuracy_comparison.jl for the cross-package correctness check.
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

const SWG = MaxEntropyGraphs.SimpleWeightedGraphs

LinearAlgebra.BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_CORES", string(Threads.nthreads()))))

@info """
$(now()) - Setting up CRWCM benchmarks on $(Threads.nthreads()) threads (BLAS: $(LinearAlgebra.BLAS.get_num_threads())).
"""

## Load up helper functions
include(joinpath(@__DIR__, "benchmark_helpers.jl"))

## Reference graphs setup
## ----------------------
## rhesus_macaques is a real weighted directed network with high (weighted) reciprocity
## (r_t ≈ 0.76, r_w ≈ 0.90) — the regime the CRWCM is built for. The larger problems tile it
## block-diagonally (well-conditioned; the weighted layer has 4N unreduced parameters, so this is
## the heaviest of the conditional models).
function tiled_rhesus_digraph(k::Int)
    base = MaxEntropyGraphs.rhesus_macaques(); n = Graphs.nv(base)
    sources = Int[]; targets = Int[]; weights = Float64[]
    for c in 0:k-1, e in Graphs.edges(base)
        push!(sources, Graphs.src(e) + c*n); push!(targets, Graphs.dst(e) + c*n); push!(weights, e.weight)
    end
    return SWG.SimpleWeightedDiGraph(sources, targets, weights)
end

name_graphs = [("CRWCM_small",  tiled_rhesus_digraph(1),  Dict(:include_fixed_point => true, :include_BFGS => true, :include_newton => true)),
               ("CRWCM_medium", tiled_rhesus_digraph(8),  Dict(:include_fixed_point => true, :include_BFGS => true, :include_newton => true)),
               ("CRWCM_large",  tiled_rhesus_digraph(32), Dict(:include_fixed_point => true, :include_BFGS => true, :include_newton => false))]

# Scale limiter: BENCH_MAX_SCALE=small|medium|large (default large). BENCH_QUICK=1 aliases small.
let scale = lowercase(get(ENV, "BENCH_QUICK", "0") == "1" ? "small" : get(ENV, "BENCH_MAX_SCALE", "large"))
    ncap = scale == "small" ? 1 : scale == "medium" ? 2 : length(name_graphs)
    ncap = min(ncap, length(name_graphs))
    ncap < length(name_graphs) && @info "BENCH_MAX_SCALE=$(scale): restricting CRWCM benchmarks to the first $(ncap) problem(s)."
    global name_graphs = name_graphs[1:ncap]
end

# Write out the weighted edgelists (source, target, weight) for the reference graphs (Python side)
for (name, G, _) in name_graphs
    open(joinpath(@__DIR__, "data", "$(name).csv"), "w") do f
        for e in edges(G)
            write(f, "$(e.src), $(e.dst), $(Float64(e.weight))\n")
        end
    end
    @info """$(now()) - benchmark "$(name)" will test G($(nv(G)), $(ne(G))) [weighted, directed]"""
end

## Generate python scripts and the associated shell script
for (name, G, _) in name_graphs
    generate_NuMeTriS_python(name, "RBCM+CRWCM")
end
open(joinpath(@__DIR__, "CRWCM_script.sh"), "w") do f
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
        "benchmarks" => [test_create_CRWCM(G);
                         test_solve_CRWCM(G; kwargs...);
                         test_fluxes_CRWCM(G);
                         test_sample_CRWCM(G, n);
                         ]
        )
    open(joinpath(outpath, "$(Dates.format(now(), "YYYY_mm_dd_HH_MM"))_$(name).json"),"w") do f
        write(f, JSON.json(results))
    end

    @info "$(now()) - Benchmarking for $(name) done."
end

@info "$(now()) - All Julia CRWCM benchmarks done."
