##############################################################################################
#   ____  ____   ____ __  __   _                     _                          _
#  |  _ \| __ ) / ___|  \/  | | |__   ___ _ __   ___| |__  _ __ ___   __ _ _ __| | _____
#  | |_) |  _ \| |   | |\/| | | '_ \ / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ / __|
#  |  _ <| |_) | |___| |  | | | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <\__ \
#  |_| \_\____/ \____|_|  |_| |_.__/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\___/
#
#  Reciprocal Binary Configuration Model (RBCM): construction, solving, exact expected motif
#  spectrum and sampling for a directed network with fixed reciprocal degree sequences
#  (k→, k←, k↔). The reference Python implementation is NuMeTriS (model 'RBCM'); see
#  accuracy_comparison.jl for the cross-package correctness check.
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
$(now()) - Setting up RBCM benchmarks on $(Threads.nthreads()) threads (BLAS: $(LinearAlgebra.BLAS.get_num_threads())).
"""

## Load up helper functions
include(joinpath(@__DIR__, "benchmark_helpers.jl"))

## Reference graphs setup
## ----------------------
## The (binarised) rhesus_macaques network is a real directed network with high reciprocity
## (r_t ≈ 0.76), i.e. exactly the regime the RBCM is built for. The larger problems tile it
## block-diagonally, which keeps every instance realizable and keeps the number of distinct
## (k→, k←, k↔) triples constant while N grows (highlighting the reduced-model acceleration).
## The weights are kept in the dumped edge lists (NuMeTriS takes a weighted adjacency and
## binarises internally); the Julia model is built on the binarised digraph.
function tiled_rhesus_digraph(k::Int)
    base = MaxEntropyGraphs.rhesus_macaques(); n = Graphs.nv(base)
    sources = Int[]; targets = Int[]; weights = Float64[]
    for c in 0:k-1, e in Graphs.edges(base)
        push!(sources, Graphs.src(e) + c*n); push!(targets, Graphs.dst(e) + c*n); push!(weights, e.weight)
    end
    return SWG.SimpleWeightedDiGraph(sources, targets, weights)
end

name_wgraphs = [("RBCM_small",  tiled_rhesus_digraph(1),  Dict(:include_fixed_point => true, :include_BFGS => true, :include_newton => true)),
                ("RBCM_medium", tiled_rhesus_digraph(8),  Dict(:include_fixed_point => true, :include_BFGS => true, :include_newton => true)),
                ("RBCM_large",  tiled_rhesus_digraph(32), Dict(:include_fixed_point => true, :include_BFGS => true, :include_newton => false))]

# Scale limiter: BENCH_MAX_SCALE=small|medium|large (default large) caps the problem size, and
# BENCH_MIN_SCALE (default small) skips the smaller problems, so a run can target only what is
# missing (e.g. BENCH_MIN_SCALE=large BENCH_MAX_SCALE=large). BENCH_QUICK=1 aliases small.
let scale = lowercase(get(ENV, "BENCH_QUICK", "0") == "1" ? "small" : get(ENV, "BENCH_MAX_SCALE", "large")),
    minscale = lowercase(get(ENV, "BENCH_MIN_SCALE", "small"))

    ncap = scale == "small" ? 1 : scale == "medium" ? 2 : length(name_wgraphs)
    ncap = min(ncap, length(name_wgraphs))
    nfloor = minscale == "medium" ? 2 : minscale == "large" ? 3 : 1
    nfloor = min(nfloor, ncap) # a floor above the cap degrades to the cap, never to an empty run
    (nfloor > 1 || ncap < length(name_wgraphs)) && @info "BENCH_MIN_SCALE=$(minscale), BENCH_MAX_SCALE=$(scale): restricting RBCM benchmarks to problem(s) $(nfloor):$(ncap)."
    global name_wgraphs = name_wgraphs[nfloor:ncap]
end

# Write out the weighted edgelists (source, target, weight) for the reference graphs (Python side)
for (name, G, _) in name_wgraphs
    open(joinpath(@__DIR__, "data", "$(name).csv"), "w") do f
        for e in edges(G)
            write(f, "$(e.src), $(e.dst), $(Float64(e.weight))\n")
        end
    end
    @info """$(now()) - benchmark "$(name)" will test G($(nv(G)), $(ne(G))) [directed]"""
end

## Generate python scripts and the associated shell script
for (name, G, _) in name_wgraphs
    generate_NuMeTriS_python(name, "RBCM")
end
open(joinpath(@__DIR__, "RBCM_script.sh"), "w") do f
    println(f, "#!/bin/bash")
    println(f, "source \"$(joinpath(@__DIR__, ".venv", "bin", "activate"))\"")
    for (name, G, _) in name_wgraphs
        # Wrap every pytest job in the process-group watchdog; BENCH_JOB_TIMEOUT=0 (the
        # default) runs it untouched, so this changes nothing unless a budget is set.
        println(f, "\"$(joinpath(@__DIR__, "run_with_timeout.sh"))\" \"\${BENCH_JOB_TIMEOUT:-0}\" " * readlines("$(name).py")[2][3:end])
    end
end

@info "$(now()) - Reference graphs and python scripts written."


## Start benchmarking Julia
## ------------------------
for (name, Gw, kwargs) in name_wgraphs
    @info "$(now()) - Started benchmarking $(name)."
    G = Graphs.SimpleDiGraph(Gw) # the Julia RBCM works on the binarised digraph

    results = Dict(
        "system_info" => get_system_info(),
        "benchmarks" => [test_create_RBCM(G);
                         test_solve_RBCM(G; kwargs...);
                         test_motifs_RBCM(G);
                         test_sample_RBCM(G, n);
                         ]
        )
    open(joinpath(outpath, "$(Dates.format(now(), "YYYY_mm_dd_HH_MM"))_$(name).json"),"w") do f
        write(f, JSON.json(results))
    end

    @info "$(now()) - Benchmarking for $(name) done."
end

@info "$(now()) - All Julia RBCM benchmarks done."
